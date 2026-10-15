"""Frozen V13 W3/W5 residuals: no relaxed success or deeper target."""
import unittest
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from scripts.test_loaded_settle_reserve import LoadedSettleReserveTests


class LoadedServoResponseTests(unittest.TestCase):
    def node(self):
        node = LoadedSettleReserveTests().node()
        node._feedback_config.update(joint_bias_limit_rad=.18)
        node._feedback_ideal_arm_target = np.zeros(6)
        node._feedback_follow_mode = 'encoder_lever_contact'
        return node

    def response(self, node, correct):
        now = [10.]
        node._sim_time_sec = lambda: now[0]
        node._feedback_correct_hold_servo = Mock(side_effect=correct)
        node._sim_door_position = -.26
        node._sim_door_initial_position = 0.
        node._sim_lever_position = -.164168
        with patch('rclpy.ok', return_value=True), patch('time.sleep', side_effect=lambda dt: now.__setitem__(0, now[0]+dt)):
            result = node._feedback_confirm_latch_press_response(-.160, .26, require_lever_motion=True)
        return result

    def test_latch_hold_uses_bounded_servo_before_full_follow(self):
        node = self.node()
        node._phase = 'LATCH_CLEAR_PUSH'
        node._feedback_full_handle_follow = False
        original = node._feedback_last_cartesian_target.copy()
        def respond():
            node._sim_lever_position = -.170
            return True
        self.assertTrue(self.response(node, respond))
        node._feedback_correct_hold_servo.assert_called_once()
        np.testing.assert_array_equal(node._feedback_last_cartesian_target, original)

    def test_no_response_still_fails_after_bounded_attempts(self):
        node = self.node()
        node._phase = 'LATCH_CLEAR_PUSH'
        node._feedback_full_handle_follow = False
        self.assertFalse(self.response(node, lambda: True))
        self.assertEqual(node._feedback_correct_hold_servo.call_count, 3)
        self.assertIn('No measured response', node._feedback_stop.call_args.args[0])

    def test_noncontact_phase_cannot_enable_extra_servo(self):
        node = self.node()
        node._phase = 'PRE_GRASP'
        node._feedback_full_handle_follow = False
        self.assertFalse(self.response(node, lambda: True))
        node._feedback_correct_hold_servo.assert_not_called()

    def test_recorded_w5_settled_residual_corrects_same_goal(self):
        node = self.node()
        goal = np.array([.49084532, .05296678, .75332882])
        actual = np.array([.48148459, .05578650, .76502028])
        node._feedback_last_cartesian_target = goal.copy()
        node._feedback_press_origin = np.array([goal[0], goal[1], .76082992])
        node._feedback_press_depth = node._feedback_press_origin[2]-goal[2]
        node._measured_prepress_height = .76082992
        node._sim_door_position = -.292
        node._feedback_tool = lambda: actual.copy()
        commands = []
        def correct():
            commands.append(node._feedback_last_cartesian_target.copy())
            actual[:] = goal + .5*(actual-goal)
            return True
        node._feedback_correct_hold_servo = Mock(side_effect=correct)
        now = [0.]
        node._sim_time_sec = lambda: now[0]
        with patch('rclpy.ok', return_value=True), patch('time.sleep', side_effect=lambda dt: now.__setitem__(0, now[0]+dt)):
            self.assertTrue(ManipulationNode._feedback_move_tool(node, goal, 'encoder_contact_follow'))
        self.assertEqual(len(commands), 1)
        np.testing.assert_array_equal(commands[0], goal)
        self.assertLess(np.linalg.norm(actual-goal), .015)
        self.assertTrue(all(c.args == (0.,0.) for c in node._publish_sim_cmd_vel.call_args_list))

    def test_active_servo_rejects_lost_sensors_grip_or_pose_bounds(self):
        for invalid in ('fresh', 'grip', 'normal', 'height', 'wrist', 'lever', 'bias'):
            node = self.node()
            node._feedback_tool = lambda: node._feedback_last_cartesian_target + [0.,0.,.01]
            if invalid == 'fresh': node._feedback_fresh = lambda: False
            if invalid == 'grip': node._sim_arm_joint_positions['gripper_left_joint'] = .001
            if invalid == 'normal': node._feedback_tool = lambda: node._feedback_last_cartesian_target + [.016,0.,0.]
            if invalid == 'height': node._feedback_tool = lambda: node._feedback_last_cartesian_target + [0.,0.,.016]
            if invalid == 'wrist': node._piper_kinematics._rotation_error.return_value = np.array([.11,0.,0.])
            if invalid == 'lever': node._sim_lever_position = -.01
            if invalid == 'bias': node._feedback_joint_bias = np.full(6,.19)
            self.assertFalse(node._feedback_correct_hold_servo(), invalid)
            node._feedback_command_point.assert_not_called()


if __name__ == '__main__':
    unittest.main()
