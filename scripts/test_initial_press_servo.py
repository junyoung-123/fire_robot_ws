"""Initial lever probing keeps pose bounds even when loaded tracking settles."""
import unittest
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from scripts.test_loaded_servo_response import LoadedServoResponseTests


class InitialPressServoTests(unittest.TestCase):
    def node(self):
        node = LoadedServoResponseTests().node()
        node._phase = 'PRESS_HANDLE'
        node._sim_lever_press_min_angle = .15
        node._sim_lever_position = -.09
        node._feedback_press_verified = False
        goal = np.array([.696923661857338, .1129203387791938, .740989426748778])
        node._feedback_last_cartesian_target = goal.copy()
        node._feedback_tool = lambda: goal + [0., 0., .014999]
        return node

    def test_recorded_saturated_press_corrects_same_cartesian_goal(self):
        node = self.node()
        goal = node._feedback_last_cartesian_target.copy()
        self.assertTrue(node._feedback_correct_hold_servo(pressing=True))
        command, stage = node._feedback_command_point.call_args.args
        np.testing.assert_array_equal(command, goal)
        self.assertEqual(stage, 'feedback_press')
        self.assertFalse(node._feedback_press_verified)
        node._publish_sim_cmd_vel.assert_called_with(0., 0.)

    def test_initial_press_correction_requires_observed_response_and_bounds(self):
        for invalid in ('fresh', 'grip', 'normal', 'height', 'norm', 'wrist', 'lever', 'bias', 'phase'):
            node = self.node()
            if invalid == 'fresh': node._feedback_fresh = lambda: False
            if invalid == 'grip': node._sim_arm_joint_positions['gripper_left_joint'] = .001
            if invalid == 'normal': node._feedback_tool = lambda: node._feedback_last_cartesian_target + [.016, 0., 0.]
            if invalid == 'height': node._feedback_tool = lambda: node._feedback_last_cartesian_target + [0., 0., .016]
            if invalid == 'norm': node._feedback_tool = lambda: node._feedback_last_cartesian_target + [.012, 0., .012]
            if invalid == 'wrist': node._piper_kinematics._rotation_error.return_value = np.array([.11, 0., 0.])
            if invalid == 'lever': node._sim_lever_position = -.001
            if invalid == 'bias': node._feedback_joint_bias = np.full(6, .19)
            if invalid == 'phase': node._phase = 'PRE_GRASP'
            self.assertFalse(node._feedback_correct_hold_servo(pressing=True), invalid)
            node._feedback_command_point.assert_not_called()

    def press_loop(self, responds):
        node = self.node()
        clock = [10.]
        actual = np.array([.69, .11, .76])
        node._feedback_tool = lambda: actual.copy()
        node._sim_time_sec = lambda: clock[0]
        node._feedback_anchor_contact_wrist = lambda: True
        node._sim_lever_press_timeout = 8.
        node._feedback_lever_samples = []
        node._feedback_config.update(press_step_m=.01)
        node._feedback_press_increment = Mock(side_effect=lambda *a: .0075 if not node._feedback_command_point.called else 0.)
        node._feedback_command_point.reset_mock()
        node._wait_for_sim_duration = lambda dt, **kw: clock.__setitem__(0, clock[0]+dt) or True
        def correct(**kwargs):
            if responds:
                node._sim_lever_position = -.16
                node._feedback_lever_samples = [(clock[0]-.1, -.16), (clock[0]-.05, -.16), (clock[0], -.16)]
            return True
        node._feedback_correct_hold_servo = Mock(side_effect=correct)
        with patch('rclpy.ok', return_value=True), patch('time.monotonic', side_effect=lambda: clock[0]), \
                patch('time.sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)):
            success = ManipulationNode._feedback_press(node, None)
        return node, success

    def test_stable_measured_lever_response_still_decides_press_success(self):
        node, success = self.press_loop(True)
        self.assertTrue(success)
        node._feedback_correct_hold_servo.assert_called_once_with(pressing=True)
        self.assertTrue(node._feedback_press_verified)

    def test_three_corrections_without_response_cannot_pass(self):
        node, success = self.press_loop(False)
        self.assertFalse(success)
        self.assertEqual(node._feedback_correct_hold_servo.call_count, 3)
        self.assertFalse(node._feedback_press_verified)
        self.assertIn('timeout', node._feedback_stop.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
