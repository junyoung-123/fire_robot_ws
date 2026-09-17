import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np

from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class HandleHeldSequenceTests(unittest.TestCase):
    def node(self, opening=True):
        node = object.__new__(ManipulationNode)
        for key, value in dict(_hold_handle_during_base_open=True, _sim_mode=True,
                _feedback_contact=True, _sim_release_handle_before_base_push=False,
                _sim_door_contact_only=True, _sim_push_follow_enabled=True,
                _press_handle_before_push=True, _door_open_motion='push',
                _post_open_backoff_enabled=True, _sim_door_position=0., _sim_door_initial_position=0.,
                _sim_lever_position=0., _sim_arm_clearance_retract=.18).items():
            setattr(node, key, value)
        calls = []
        node.get_logger = lambda: NS(info=lambda *a: None, error=lambda *a: None)
        node._set_manip_phase = lambda phase, *a: calls.append(phase)
        node._transform_handle_to_manipulation_frame = lambda point: point
        node._feedback_fresh = lambda: True
        node._move_to_pre_grasp = lambda point: True
        node._grasp_handle = lambda point: True
        node._press_handle = lambda point: True
        node._run_sim_latch_clear_push = lambda: True
        node._feedback_event = lambda event, **kw: calls.append(event)
        node._feedback_stop = lambda reason: False
        def full_open(*args):
            calls.append('full_open_attempt')
            if opening:
                node._sim_door_position = -2.06
            return opening
        node._open_door_motion = full_open
        node._command_sim_gripper = lambda value: calls.append('gripper_release')
        node._wait_for_sim_gripper_release = lambda: True
        node._feedback_tool = lambda: np.array([.75,.15,.75])
        node._feedback_contact_rotation = lambda: np.eye(3)
        node._sim_arm_joint_positions = dict(gripper_left_joint=.025,gripper_right_joint=-.025)
        node._feedback_retract_released_handle = lambda span: calls.append('measured_retract') or True
        node._piper_kinematics = NS(tool_front_extent_m=.1358,
            preferred_contact_posture=lambda point,*args: (point,.35))
        node._feedback_move_tool = lambda *a: True
        node._move_sim_arm_to_stow = lambda: calls.append('stow') or True
        node._move_to_home = lambda: True
        node._post_open_backoff = lambda: calls.append('backoff')
        return node, calls

    def test_hand_stays_closed_until_full_open_then_release_retract_stow(self):
        node, calls = self.node()
        self.assertTrue(node._execute_door_open_sequence(NS(), 'observed'))
        self.assertIn('HANDLE_HELD_BASE_OPEN', calls)
        self.assertNotIn('BASE_PUSH_OPEN', calls)
        self.assertLess(calls.index('full_open_attempt'), calls.index('gripper_release'))
        self.assertLess(calls.index('gripper_release'), calls.index('RETRACT_FROM_HANDLE'))
        self.assertLess(calls.index('stow'), calls.index('backoff'))

    def test_open_failure_cannot_release_and_pretend_sequence_complete(self):
        node, calls = self.node(opening=False)
        self.assertFalse(node._execute_door_open_sequence(NS(), 'observed'))
        self.assertNotIn('gripper_release', calls)
        self.assertNotIn('COMPLETE', calls)

    def test_contradictory_early_release_setting_fails_before_grasp(self):
        node, calls = self.node()
        node._sim_release_handle_before_base_push = True
        self.assertFalse(node._execute_door_open_sequence(NS(), 'observed'))
        self.assertNotIn('GRASP_HANDLE', calls)

    def test_handle_held_loop_stops_on_finger_contact_loss_before_driving(self):
        node, calls = self.node()
        node._feedback_press_verified = True
        node._feedback_follow_mode = 'encoder_lever_contact'
        node._feedback_encoder_follow_origin = np.array([.75,.15,.75])
        node._sim_time_sec = lambda: 10.
        node._sim_arm_joint_positions = dict(gripper_left_joint=.0, gripper_right_joint=-.02)
        node._publish_sim_cmd_vel = lambda v, w: calls.append(('velocity', v, w))
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        self.assertEqual([c for c in calls if isinstance(c, tuple)], [('velocity', 0., 0.)])

    def test_missing_lever_feedback_does_not_dispatch_motion(self):
        node, calls = self.node()
        node._feedback_press_verified = True
        node._feedback_fresh = lambda: False
        with patch.object(node, '_feedback_probe_contact_motion') as motion:
            self.assertFalse(node._feedback_handle_held_open(2.05))
            motion.assert_not_called()

    def test_held_mode_follows_both_axes_without_integrating_preload(self):
        node, calls = self.node()
        node._feedback_press_verified = True
        node._feedback_full_handle_follow = True
        node._feedback_config = dict(press_sign=-1., follow_max_shift_m=.15,
                                     tracking_limit_m=.04, follow_step_m=.008)
        node._sim_lever_position, node._sim_lever_initial_position = -.12, 0.
        node._sim_lever_press_min_angle = .08
        node._feedback_encoder_follow_origin = np.array([.7,.15,.75])
        node._feedback_tool = lambda: np.array([.66,.15,.75])
        node._sim_time_sec = lambda: 10.
        node._feedback_follow_last_command = 0.
        node._feedback_press_origin = np.array([.68,.15,.77])
        node._feedback_press_depth = .02
        node._feedback_sweep_state = (np.array([.66,.15,.75]), np.array([.66,.15,.75]), 0.)
        node._feedback_sweep_initial_door = 0.
        node._feedback_contact_orientation_reference = (0.,0.,np.eye(3))
        node._manipulation_frame = 'base_link'
        def transform(point, frame, **kw):
            if frame == 'base_link':
                point.point.x -= .01
            return point
        node._tf_buffer = NS(transform=transform)
        targets = []
        node._feedback_command_point = lambda goal, stage: targets.append(goal.copy()) or True
        self.assertTrue(node._feedback_follow_encoder_contact())
        expected_x = .66 - .01
        self.assertAlmostEqual(targets[-1][0], expected_x)
        self.assertAlmostEqual(node._feedback_press_origin[0], expected_x)
        np.testing.assert_allclose(node._feedback_sweep_state[0], [.66,.15,.75])
        self.assertAlmostEqual(targets[-1][2], .75)

    def test_initial_base_turn_follows_observed_panel_normal(self):
        node, calls = self.node()
        node._feedback_press_verified = True
        node._feedback_follow_mode = 'encoder_lever_contact'
        node._feedback_encoder_follow_origin = np.array([.65,.15,.75])
        node._feedback_config = dict(press_sign=-1., tool_tolerance_m=.015)
        node._sim_time_sec = lambda: 10.
        node._sim_arm_joint_positions = dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=.02, gripper_right_joint=-.025)
        node._sim_lever_position, node._sim_lever_initial_position = -.12, 0.
        node._sim_lever_press_min_angle = .08
        node._sim_door_position, node._sim_door_initial_position = -.25, 0.
        node._feedback_contact_orientation_reference = (0.,0.,np.eye(3))
        node._feedback_base_yaw = lambda: 0.
        node._publish_sim_cmd_vel = lambda *a: None
        angular = []
        node._feedback_probe_contact_motion = lambda value, *args: angular.append(value)
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        self.assertLess(angular[0], 0.)
        self.assertGreaterEqual(angular[0], -.08)

    def test_loaded_arm_settles_stopped_before_driving(self):
        node, calls = self.node()
        node._feedback_full_handle_follow = True
        node._feedback_config = dict(tracking_limit_m=.04, tool_tolerance_m=.015)
        node._feedback_last_cartesian_target = np.array([.77,.15,.75])
        node._sim_arm_joint_positions = dict.fromkeys(PIPER_JOINT_NAMES, 0.)
        node._piper_kinematics = NS(forward=lambda q: (None, np.eye(3)),
                                    _rotation_error=lambda a, b: np.zeros(3))
        node._publish_sim_cmd_vel = lambda v, w: calls.append(('velocity', v, w))
        node._feedback_move_tool = lambda goal, stage: calls.append('settle') or True
        self.assertTrue(node._feedback_settle_held_target())
        self.assertLess(calls.index(('velocity', 0., 0.)), calls.index('settle'))

    def test_excessive_tracking_error_is_not_recovered_by_relaxing_limit(self):
        node, calls = self.node()
        node._feedback_full_handle_follow = True
        node._feedback_config = dict(tracking_limit_m=.04, tool_tolerance_m=.015)
        node._feedback_last_cartesian_target = np.array([.80,.15,.75])
        with patch.object(node, '_feedback_move_tool') as move:
            self.assertFalse(node._feedback_settle_held_target())
            move.assert_not_called()

    def test_failed_settle_cannot_dispatch_base_probe(self):
        node, calls = self.node()
        node._feedback_retain_lever_press = lambda: True
        node._feedback_follow_contact = lambda: True
        node._feedback_settle_held_target = lambda: False
        node._publish_sim_cmd_vel = lambda v, w: calls.append(('velocity', v, w))
        self.assertIsNone(node._feedback_probe_contact_motion(.05))
        self.assertEqual(calls, [('velocity', 0., 0.)])

    def test_settling_stops_if_lever_contact_is_lost(self):
        node, calls = self.node()
        del node._feedback_move_tool
        node._feedback_full_handle_follow = True
        node._feedback_config = dict(press_sign=-1., tracking_limit_m=.04)
        node._sim_lever_position, node._sim_lever_initial_position = -.01, 0.
        node._sim_lever_press_min_angle = .08
        node._feedback_command_point = lambda *args: True
        node._sim_time_sec = lambda: 10.
        with patch('fire_robot_manipulation.physical_contact_manipulation_node.rclpy.ok', return_value=True):
            self.assertFalse(node._feedback_move_tool(np.array([.75,.15,.75]),
                                                      'encoder_contact_follow'))


if __name__ == '__main__':
    unittest.main()
