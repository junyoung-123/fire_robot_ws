#!/usr/bin/env python3
"""Feedback must prove motion; reaching a command limit cannot pass."""
import unittest
from unittest.mock import patch
import numpy as np
from types import SimpleNamespace as NS
from geometry_msgs.msg import PointStamped
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics

from fire_robot_manipulation.contact_feedback import (
    fresh_sample, observed_clearance, next_press_depth, stable_lever_motion,
    bounded_contact_target)
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode


class FeedbackGuardTests(unittest.TestCase):
    def test_ik_position_match_with_wrong_wrist_orientation_is_rejected(self):
        kinematics = PiperActualKinematics()
        point = np.array([.6, .15, .77])
        wrong = kinematics.approach_rotation @ np.diag([-1., -1., 1.])
        with patch.object(kinematics, 'forward', return_value=(point, wrong)):
            self.assertIsNone(kinematics.solve(point, max_iterations=2))

    def test_recorded_contact_goal_remains_reachable_with_orientation_guard(self):
        kinematics = PiperActualKinematics()
        solution = kinematics.solve([.847076, .220325, .663161], position_tolerance_m=.003)
        self.assertIsNotNone(solution)
        self.assertLessEqual(solution.orientation_error_rad, .10)
        self.assertLessEqual(solution.position_error_m, .003)

    def test_contact_target_tracks_observation_not_predefined_hinge_geometry(self):
        reference = np.array([2., 1., .8])
        contact = reference + [.01, -.01, -.02]
        observed = reference + [.02, .015, 0.]
        target = bounded_contact_target(observed, reference, contact, contact, .15, .03)
        np.testing.assert_allclose(target, contact + observed-reference)

    def test_contact_follow_step_is_bounded(self):
        target = bounded_contact_target([.1, 0., 0.], [0., 0., 0.],
                                        [0., 0., 0.], [0., 0., 0.], .15, .008)
        np.testing.assert_allclose(target, [.008, 0., 0.])

    def test_far_or_nonfinite_contact_candidate_is_rejected(self):
        for observed in ([1., 0., 0.], [float('nan'), 0., 0.]):
            with self.assertRaises(ValueError):
                bounded_contact_target(observed, [0., 0., 0.], [0., 0., 0.],
                                       [0., 0., 0.], .15, .008)

    def test_contact_follow_is_independent_of_world_origin(self):
        points = np.array([[.02, .01, 0.], [0., 0., 0.],
                           [.01, 0., 0.], [.01, 0., 0.]])
        first = bounded_contact_target(*points, .15, .008)
        offset = np.array([12., -4., 1.])
        second = bounded_contact_target(*(points+offset), .15, .008)
        np.testing.assert_allclose(first+offset, second)

    def test_fresh_callback_cannot_mask_old_image_time(self):
        node = self.make_press_node()
        point = PointStamped()
        point.header.stamp.sec = 1
        node._feedback_contact_observation = (9.9, point, 'yolo:primary:item:registered_depth')
        node._sim_time_sec = lambda: 10.
        node._feedback_config['follow_max_age_sec'] = 1.
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.time.monotonic', return_value=10.):
            self.assertIsNone(node._feedback_live_contact_observation())
            point.header.stamp.sec = 10
            self.assertIsNotNone(node._feedback_live_contact_observation())

    def test_visual_dropout_never_commands_arm_or_forward_motion(self):
        node = self.make_press_node()
        node._feedback_live_contact_observation = lambda: None
        with patch.object(node, '_feedback_command_point') as command, \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(ManipulationNode._feedback_follow_contact(node))
            command.assert_not_called()
            self.assertIn('Lost live RGB-D', stop.call_args[0][0])

    def test_contact_follow_compensates_current_base_motion(self):
        for base_x in (.4, .5):
            node = self.make_press_node()
            node._feedback_config.update(follow_max_shift_m=.15, follow_step_m=.008)
            node._feedback_follow_reference = (np.array([1., 0., .75]),
                                                np.array([1., 0., .75]))
            node._feedback_follow_last_command = -float('inf')
            node._feedback_press_origin = np.array([1.-base_x, 0., .78])
            node._feedback_press_depth = .03
            node._manipulation_frame = 'base_link'
            node._sim_door_initial_position = 0.
            node._sim_door_position = -.04
            node._sim_lever_position = -.1
            node._feedback_tool = lambda: np.array([1.-base_x, 0., .75])
            observation = PointStamped()
            observation.header.frame_id = 'odom'
            observation.point.x, observation.point.z = 1.02, .75
            node._feedback_live_contact_observation = lambda: (10., observation, 'yolo:primary:item:registered_depth')
            def transform(point, frame, **kw):
                result = PointStamped()
                result.header.frame_id = frame
                result.point.x = point.point.x + (base_x if frame == 'odom' else -base_x)
                result.point.y, result.point.z = point.point.y, point.point.z
                return result
            node._tf_buffer = NS(transform=transform)
            with patch.object(node, '_feedback_command_point', return_value=True) as command:
                self.assertTrue(ManipulationNode._feedback_follow_contact(node))
                np.testing.assert_allclose(command.call_args[0][0], [1.008-base_x, 0., .75])

    def test_occluded_following_requires_verified_contact(self):
        node = self.make_press_node()
        node._feedback_press_verified = False
        with patch.object(node, '_feedback_command_point') as command:
            self.assertFalse(node._feedback_begin_encoder_follow())
            command.assert_not_called()

    def test_shifted_visual_target_is_not_used_as_a_contact_goal(self):
        node = self.make_press_node()
        node._feedback_encoder_contact_follow = True
        node._feedback_live_contact_observation = lambda: (
            1., NS(point=NS(x=.58, y=.12, z=.8)), 'track:primary_yolo:lk:registered_depth')
        node._manipulation_frame = 'base_link'
        node._tf_buffer = NS(transform=lambda point, *a, **k: point)
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True), \
                patch.object(node, '_publish_sim_cmd_vel') as drive, \
                patch.object(node, '_feedback_command_point') as arm, \
                patch.object(node, '_feedback_begin_encoder_follow', return_value=False) as fallback:
            self.assertFalse(ManipulationNode._feedback_begin_contact_follow(node))
            fallback.assert_called_once()
            arm.assert_not_called()
            drive.assert_called_once_with(0., 0.)

    def test_occluded_contact_loss_does_not_make_an_arm_target(self):
        node = self.make_press_node()
        node._feedback_press_verified = True
        node._sim_lever_position = -.01
        with patch.object(node, '_feedback_command_point') as command:
            self.assertFalse(node._feedback_follow_encoder_contact())
            command.assert_not_called()

    def test_occluded_follow_does_not_accumulate_lateral_servo_deflection(self):
        node = self.make_press_node()
        node._feedback_press_verified = True
        node._sim_lever_position = -.1
        node._feedback_config['follow_max_shift_m'] = .15
        node._feedback_press_origin = np.array([.58, .2, .8])
        node._feedback_press_depth = .03
        node._feedback_encoder_follow_origin = np.array([.58, .2, .77])
        node._feedback_follow_last_command = -float('inf')
        node._feedback_tool = lambda: np.array([.58, .19, .77])
        origin = np.array([.58, .2, .77])
        node._feedback_sweep_state = (origin, origin.copy(), 0.)
        node._feedback_sweep_initial_door = 0.
        node._feedback_sweep_normal = np.array([1., 0., 0.])
        node._feedback_contact_orientation_reference = (0., 0., np.eye(3))
        node._sim_door_position = 0.
        node._manipulation_frame = 'base_link'
        node._tf_buffer = NS(transform=lambda point, *a, **k: point)
        with patch.object(node, '_feedback_command_point', return_value=True) as command:
            self.assertTrue(node._feedback_follow_encoder_contact())
            np.testing.assert_allclose(command.call_args[0][0], [.58, .2, .77])

    def test_arm_and_lever_cannot_mask_stale_door_feedback(self):
        node = object.__new__(ManipulationNode)
        node._feedback_config = {'max_age_sec': 1.0}
        node._feedback_arm_stamp = node._feedback_lever_stamp = 9.9
        node._feedback_door_stamp = None
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.time.monotonic', return_value=10.0):
            self.assertFalse(node._feedback_fresh())
            node._feedback_door_stamp = 9.9
            self.assertTrue(node._feedback_fresh())

    def test_contact_loss_on_rotating_door_stops_before_new_motion(self):
        node = self.make_press_node()
        node._sim_door_initial_position = 0.0
        node._sim_door_position = -.208
        node._sim_lever_position = -.009
        node._feedback_fresh = lambda: True
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True), \
                patch.object(node, '_publish_sim_cmd_vel') as drive, \
                patch.object(node, '_feedback_command_point') as arm, \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(node._feedback_latch_clear_push())
            self.assertIn('reobservation required', stop.call_args[0][0])
            arm.assert_not_called()
            self.assertTrue(all(call.args == (0.0, 0.0) for call in drive.call_args_list))

    def test_failed_final_stow_cannot_report_success(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact = True
        node._feedback_fresh = lambda: True
        node._command_sim_arm_pose = lambda *a: True
        node._wait_for_sim_arm_target = lambda **kw: False
        with patch.object(node, '_feedback_event') as event, \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(node._move_to_home())
            event.assert_not_called()
            self.assertIn('not confirmed', stop.call_args[0][0])

    def test_latch_probe_requires_fresh_feedback_before_driving(self):
        node = self.make_press_node()
        node._sim_door_initial_position = 0.0
        node._feedback_fresh = lambda: False
        with patch.object(node, '_publish_sim_cmd_vel') as drive:
            self.assertFalse(node._feedback_latch_clear_push())
            drive.assert_not_called()

    def test_lever_motion_alone_never_proves_latch_release(self):
        node = self.make_press_node()
        node._sim_door_initial_position = node._sim_door_position = 0.0
        node._sim_lever_position = -.1
        node._sim_latch_clear_min_angle = .25
        node._feedback_fresh = lambda: True
        node._feedback_press_origin = np.array([.58, .2, .8])
        node._feedback_press_depth = .03
        actual = np.array([.58, .2, .77])
        node._feedback_tool = lambda: actual.copy()
        clock = [10.]
        node._sim_time_sec = lambda: clock[0]
        drive = [0.0]
        node._publish_sim_cmd_vel = lambda x, y: drive.__setitem__(0, x)
        def command(goal, stage):
            self.assertEqual(drive[0], 0.0)
            actual[:] = goal
            return True
        node._feedback_command_point = command
        def wait(duration, **kw):
            clock[0] += duration
            return True
        node._wait_for_sim_duration = wait
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=lambda s: wait(s)), \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(node._feedback_latch_clear_push())
            self.assertIn('No measured response', stop.call_args[0][0])
        self.assertEqual(drive[0], 0.0)
        self.assertLessEqual(node._feedback_press_depth, .035 + 1.e-9)

    def test_latch_correction_requires_incremental_not_absolute_lever_motion(self):
        node = self.make_press_node()
        node._sim_door_initial_position = 0.
        node._sim_door_position = -.115
        node._sim_lever_position = -.135
        node._feedback_fresh = lambda: True
        clock = [10.]
        node._sim_time_sec = lambda: clock[0]
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        def tick(seconds):
            clock[0] += seconds
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=tick), \
                patch.object(node, '_publish_sim_cmd_vel') as drive, \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(node._feedback_confirm_latch_press_response(-.135, .115))
            self.assertIn('No measured response', stop.call_args[0][0])
            self.assertTrue(all(call.args == (0., 0.) for call in drive.call_args_list))

    def test_latch_correction_accepts_measured_lever_or_door_response(self):
        for lever, door in ((-.15, -.115), (-.135, -.12)):
            node = self.make_press_node()
            node._sim_door_initial_position = 0.
            node._sim_door_position = door
            node._sim_lever_position = lever
            node._feedback_fresh = lambda: True
            with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True), \
                    patch.object(node, '_publish_sim_cmd_vel') as drive:
                self.assertTrue(node._feedback_confirm_latch_press_response(-.135, .115))
                drive.assert_called_once_with(0., 0.)

    def test_latch_correction_rejects_stale_sensors(self):
        node = self.make_press_node()
        node._feedback_fresh = lambda: False
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True), \
                patch.object(node, '_publish_sim_cmd_vel'), \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(node._feedback_confirm_latch_press_response(-.135, .115))
            self.assertIn('Lost feedback', stop.call_args[0][0])

    def test_latch_correction_waits_for_moving_servo_without_extra_press(self):
        node = self.make_press_node()
        node._sim_door_initial_position = 0.
        node._sim_door_position = -.1
        node._sim_lever_position = -.135
        node._feedback_fresh = lambda: True
        clock = [10.]
        node._sim_time_sec = lambda: clock[0]
        node._feedback_tool = lambda: np.array([.58, .2, .8-.006*(clock[0]-10.)])
        def tick(seconds):
            clock[0] += seconds
            if clock[0] >= 11.5:
                node._sim_lever_position = -.15
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=tick), \
                patch.object(node, '_publish_sim_cmd_vel') as drive, \
                patch.object(node, '_feedback_command_point') as arm:
            self.assertTrue(node._feedback_confirm_latch_press_response(-.135, .1, True))
            arm.assert_not_called()
            drive.assert_called_once_with(0., 0.)

    def make_hold_node(self):
        node = self.make_press_node()
        node._feedback_lever_hold_delta = .135
        node._sim_lever_position = -.11
        node._sim_door_initial_position = 0.
        node._sim_door_position = -.12
        node._feedback_fresh = lambda: True
        node._feedback_press_origin = np.array([.58, .2, .85])
        node._feedback_press_depth = .05
        return node

    def test_lever_relaxation_pauses_base_before_bounded_press_correction(self):
        node = self.make_hold_node()
        order = []
        node._publish_sim_cmd_vel = lambda *args: order.append(('drive', args))
        node._feedback_command_point = lambda goal, stage: order.append(('arm', goal.copy())) or True
        with patch.object(node, '_feedback_confirm_latch_press_response', return_value=True) as confirm:
            self.assertTrue(node._feedback_retain_lever_press())
            self.assertEqual(order[0], ('drive', (0., 0.)))
            np.testing.assert_allclose(order[1][1], [.58, .2, .795])
            self.assertAlmostEqual(node._feedback_press_depth, .055)
            confirm.assert_called_once_with(-.11, .12, require_lever_motion=True)

    def test_steady_lever_does_not_add_press_travel(self):
        node = self.make_hold_node()
        node._sim_lever_position = -.133
        with patch.object(node, '_feedback_command_point') as command:
            self.assertTrue(node._feedback_retain_lever_press())
            command.assert_not_called()
            self.assertEqual(node._feedback_press_depth, .05)

    def test_hold_probe_is_below_measured_tip_even_with_loaded_servo_sag(self):
        node = self.make_hold_node()
        node._feedback_tool = lambda: np.array([.58, .2, .787])
        targets = []
        node._feedback_command_point = lambda goal, stage: targets.append(goal.copy()) or True
        with patch.object(node, '_publish_sim_cmd_vel'), \
                patch.object(node, '_feedback_confirm_latch_press_response', return_value=True):
            self.assertTrue(node._feedback_retain_lever_press())
        self.assertAlmostEqual(targets[-1][2], .782)
        self.assertAlmostEqual(node._feedback_press_depth, .068)

    def test_measured_sag_cannot_bypass_total_press_travel_limit(self):
        node = self.make_hold_node()
        node._feedback_press_depth = .095
        node._feedback_tool = lambda: np.array([.58, .2, .74])
        with patch.object(node, '_publish_sim_cmd_vel'), \
                patch.object(node, '_feedback_command_point') as command:
            self.assertFalse(node._feedback_retain_lever_press())
            command.assert_not_called()

    def test_hold_response_baseline_is_captured_after_ik_dispatch(self):
        node = self.make_hold_node()
        def plan_and_dispatch(goal, stage):
            node._sim_lever_position = -.09
            node._sim_door_position = -.125
            return True
        node._feedback_command_point = plan_and_dispatch
        with patch.object(node, '_publish_sim_cmd_vel'), \
                patch.object(node, '_feedback_confirm_latch_press_response', return_value=True) as confirm:
            self.assertTrue(node._feedback_retain_lever_press())
            confirm.assert_called_once_with(-.09, .125, require_lever_motion=True)

    def test_lost_or_stale_contact_cannot_trigger_retention_press(self):
        for lever, fresh in ((-.04, True), (-.11, False), (float('nan'), True)):
            node = self.make_hold_node()
            node._sim_lever_position = lever
            node._feedback_fresh = lambda: fresh
            with patch.object(node, '_feedback_command_point') as command:
                self.assertFalse(node._feedback_retain_lever_press())
                command.assert_not_called()

    def test_retention_never_exceeds_press_budget(self):
        node = self.make_hold_node()
        node._feedback_press_depth = .11
        node._feedback_press_origin[2] = .91
        with patch.object(node, '_publish_sim_cmd_vel'), \
                patch.object(node, '_feedback_command_point') as command:
            self.assertFalse(node._feedback_retain_lever_press())
            command.assert_not_called()

    def test_coasting_door_cannot_prove_retention_correction(self):
        node = self.make_hold_node()
        clock = [10.]
        node._sim_time_sec = lambda: clock[0]
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        def tick(seconds):
            clock[0] += seconds
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=tick), \
                patch.object(node, '_publish_sim_cmd_vel'):
            self.assertFalse(node._feedback_confirm_latch_press_response(
                -.11, .10, require_lever_motion=True))

    def test_probe_duration_starts_after_expensive_ik_not_before_it(self):
        node = self.make_hold_node()
        node._feedback_lever_hold_delta = .11
        clock = [10.]
        driven = [0.]
        command = [0.]
        node._sim_time_sec = lambda: clock[0]
        node._publish_sim_cmd_vel = lambda v, w: command.__setitem__(0, v)
        def plan():
            self.assertEqual(command[0], 0.)
            clock[0] += .8
            return True
        def tick(seconds):
            clock[0] += seconds
            if command[0] > 0.:
                driven[0] += seconds
        node._feedback_follow_contact = plan
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=tick):
            self.assertEqual(node._feedback_probe_contact_motion(), 'complete')
        self.assertGreaterEqual(driven[0], .35-1.e-8)
        self.assertLess(driven[0], .38)
        self.assertEqual(command[0], 0.)

    def test_probe_stops_when_contact_relaxes_without_solving_ik_while_driving(self):
        node = self.make_hold_node()
        node._feedback_lever_hold_delta = .11
        clock = [10.]
        command = [0.]
        node._sim_time_sec = lambda: clock[0]
        node._publish_sim_cmd_vel = lambda v, w: command.__setitem__(0, v)
        def tick(seconds):
            clock[0] += seconds
            node._sim_lever_position = -.09
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=tick):
            self.assertEqual(node._feedback_probe_contact_motion(), 'relaxed')
        self.assertEqual(command[0], 0.)

    def test_pregrasp_waypoints_do_not_accumulate_measured_sag(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact = True
        node._sim_time_sec = lambda: 1.0
        node._piper_kinematics = PiperActualKinematics()
        node._feedback_config = dict(clearance_min_m=.05, clearance_max_m=.15,
                                     tool_tolerance_m=.015, approach_step_m=.02)
        handle = PointStamped()
        handle.header.frame_id = 'base_link'
        handle.point.x, handle.point.y, handle.point.z = .8, .2, .77
        node._feedback_handle_points = [[.8, .2, .77]] * 3
        node._refine_handle_after_pre_grasp = lambda _: handle
        node._tf_buffer = NS(transform=lambda point, *a, **k: point)
        node._feedback_event = lambda *a, **k: None
        node._command_sim_gripper = lambda _: None
        start = np.array([.17, .15, .91])
        actual = start.copy()
        node._feedback_tool = lambda: actual.copy()
        waypoints = []
        def move(goal, stage):
            waypoints.append(goal.copy())
            actual[:] = goal - np.array([0., 0., .01])
            return True
        node._feedback_move_tool = move
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True):
            self.assertTrue(node._move_to_pre_grasp(handle))
        for i, point in enumerate(waypoints, 1):
            np.testing.assert_allclose(point, start+(waypoints[-1]-start)*i/len(waypoints))
        self.assertAlmostEqual(waypoints[-1][2], .77)

    def test_press_confirmation_is_independent_of_sensor_rate(self):
        for rate in (20, 100, 1000):
            samples = [(10.0 + i/rate, -.1) for i in range(int(rate*.2)+1)]
            self.assertTrue(stable_lever_motion(samples, 10., 0., -1., .08))
        self.assertFalse(stable_lever_motion([(10., -.1), (10.01, -.1),
                                             (10.02, -.1)], 10., 0., -1., .08))

    def test_missing_stale_future_and_nonfinite_feedback_rejected(self):
        for stamp in (None, 0.0, 11.0, float('nan')):
            self.assertFalse(fresh_sample(stamp, 10.0, 1.0))
        self.assertTrue(fresh_sample(9.5, 10.0, 1.0))

    def test_approach_clearance_grows_with_observed_spread(self):
        points = [[.58, .2, .8]] * 3
        _, first, _ = observed_clearance(points, .05, .15)
        noisy = [[.57, .2, .8], [.58, .2, .8], [.59, .2, .8]]
        _, second, _ = observed_clearance(noisy, .05, .15)
        self.assertAlmostEqual(first, .05)
        self.assertAlmostEqual(second, .07)

    def test_ambiguous_handle_is_not_clipped_to_a_safe_looking_margin(self):
        for points in ([[0, 0, 0]] * 2,
                       [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                       [[0, 0, 0], [0, 0, 0], [float('nan'), 0, 0]]):
            with self.assertRaises(ValueError):
                observed_clearance(points, .05, .15)

    def test_press_stops_on_feedback_before_legacy_seven_cm(self):
        depth, done = next_press_depth(.02, .01, .11, 0, -.09, -1, .08)
        self.assertTrue(done)
        self.assertEqual(depth, .02)

    def test_wrong_direction_is_not_press_confirmation(self):
        depth, done = next_press_depth(.02, .01, .11, 0, .09, -1, .08)
        self.assertFalse(done)
        self.assertAlmostEqual(depth, .03)

    def test_travel_limit_without_feedback_fails(self):
        with self.assertRaises(ValueError):
            next_press_depth(.11, .01, .11, 0, 0, -1, .08)

    def test_step_never_exceeds_travel_limit(self):
        depth, done = next_press_depth(.105, .01, .11, 0, -.01, -1, .08)
        self.assertFalse(done)
        self.assertEqual(depth, .11)

    def test_nonfinite_lever_never_passes(self):
        with self.assertRaises(ValueError):
            next_press_depth(.01, .01, .11, 0, float('nan'), -1, .08)

    def make_press_node(self):
        node = object.__new__(ManipulationNode)
        node._sim_lever_initial_position = 0.0
        node._sim_lever_position = 0.0
        node._sim_lever_press_min_angle = .08
        node._sim_lever_press_timeout = 8.0
        node._feedback_lever_samples = []
        node._sim_last_arm_target = np.zeros(6)
        node._feedback_config = dict(press_step_m=.01, press_max_travel_m=.11,
                                     press_sign=-1, tracking_limit_m=.04,
                                     lever_hold_tolerance_rad=.01)
        node._feedback_tool = lambda: np.array([.58, .2, .8])
        node._sim_time_sec = lambda: 1.0
        node._feedback_stop = lambda reason: False
        node._feedback_event = lambda *a, **k: None
        node._feedback_begin_contact_follow = lambda: True
        node._feedback_anchor_contact_wrist = lambda: True
        node._feedback_follow_contact = lambda: True
        node._feedback_follow_mode = 'vision'
        return node

    def test_stale_arm_aborts_without_a_press_command(self):
        node = self.make_press_node()
        node._feedback_tool = lambda: None
        with patch.object(node, '_feedback_command_point') as command:
            self.assertFalse(node._feedback_press(None))
            command.assert_not_called()

    def test_jammed_arm_cannot_keep_advancing(self):
        node = self.make_press_node()
        node._wait_for_sim_duration = lambda *a, **k: True
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True), \
                patch.object(node, '_feedback_command_point', return_value=True) as command, \
                patch.object(node, '_feedback_stop', return_value=False) as stop:
            self.assertFalse(node._feedback_press(None))
            self.assertLessEqual(command.call_count, 5)
            self.assertIn('tracking error', stop.call_args[0][0])

    def test_actual_feedback_changes_the_press_endpoint(self):
        endpoints = []
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        for lever_response in (5., 2.):
            node = self.make_press_node()
            clock = [10.]
            actual = np.array([.58, .2, .8])
            events = []
            node._feedback_tool = lambda: actual.copy()
            node._sim_time_sec = lambda: clock[0]
            node._feedback_event = lambda event, **kw: events.append((event, kw))
            def command(goal, stage):
                actual[:] = goal
                node._sim_lever_position = -lever_response * (.8-actual[2])
                return True
            node._feedback_command_point = command
            def wait(duration, **kw):
                for _ in range(35):
                    clock[0] += .01
                    node._feedback_lever_samples.append((clock[0], node._sim_lever_position))
                return True
            node._wait_for_sim_duration = wait
            with patch(module+'.rclpy.ok', return_value=True), \
                    patch(module+'.time.monotonic', side_effect=lambda: clock[0]):
                self.assertTrue(node._feedback_press(None))
            endpoints.append(events[-1][1]['commanded_depth_m'])
        self.assertLess(endpoints[0], endpoints[1])
        self.assertLess(endpoints[1], .07)

    def test_press_hold_retains_load_in_a_compliant_position_servo(self):
        node = self.make_press_node()
        clock = [10.]
        actual = np.array([.58, .2, .8])
        commands = []
        node._feedback_tool = lambda: actual.copy()
        node._sim_time_sec = lambda: clock[0]
        def command(goal, stage):
            commands.append((stage, goal.copy()))
            actual[:] = goal + [0., 0., .008]
            node._sim_lever_position = -4. * (.8 - actual[2])
            return True
        node._feedback_command_point = command
        def wait(duration, **kw):
            for _ in range(35):
                clock[0] += .01
                node._feedback_lever_samples.append((clock[0], node._sim_lever_position))
            return True
        node._wait_for_sim_duration = wait
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]):
            self.assertTrue(node._feedback_press(None))
        self.assertGreaterEqual(-node._sim_lever_position, node._sim_lever_press_min_angle)
        self.assertEqual(commands[-1][0], 'feedback_press_hold')
        np.testing.assert_allclose(commands[-1][1], commands[-2][1])
        retained = node._feedback_press_origin - [0., 0., node._feedback_press_depth]
        np.testing.assert_allclose(retained, commands[-1][1])


if __name__ == '__main__':
    unittest.main()
