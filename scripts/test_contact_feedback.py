#!/usr/bin/env python3
"""Feedback must prove motion; reaching a command limit cannot pass."""
import unittest
from unittest.mock import patch
import numpy as np
from types import SimpleNamespace as NS
from geometry_msgs.msg import PointStamped
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics

from fire_robot_manipulation.contact_feedback import (
    fresh_sample, observed_clearance, next_press_depth, stable_lever_motion)
from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode


class FeedbackGuardTests(unittest.TestCase):
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
            self.assertIn('travel limit', stop.call_args[0][0])
        self.assertEqual(drive[0], 0.0)
        self.assertLessEqual(node._feedback_press_depth, .11)

    def test_pregrasp_waypoints_do_not_accumulate_measured_sag(self):
        node = object.__new__(ManipulationNode)
        node._feedback_contact = True
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
                                     press_sign=-1, tracking_limit_m=.04)
        node._feedback_tool = lambda: np.array([.58, .2, .8])
        node._sim_time_sec = lambda: 1.0
        node._feedback_stop = lambda reason: False
        node._feedback_event = lambda *a, **k: None
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


if __name__ == '__main__':
    unittest.main()
