import unittest
from unittest.mock import Mock, patch

from fire_robot_manipulation.physical_contact_manipulation_node import ManipulationNode


class FeedbackGripperClosureTests(unittest.TestCase):
    def run_close(self, left, right, fresh=True, moving=False):
        node = object.__new__(ManipulationNode)
        clock = [10.]
        node._sim_time_sec = lambda: clock[0]
        node._feedback_fresh = lambda: fresh
        node._sim_arm_joint_positions = {'gripper_left_joint': left, 'gripper_right_joint': right}
        node._command_sim_gripper = Mock()
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        def tick(seconds):
            clock[0] += seconds
            if moving:
                node._sim_arm_joint_positions['gripper_left_joint'] = .012 if int(clock[0]*20)%2 else .024
        module = 'fire_robot_manipulation.physical_contact_manipulation_node'
        with patch(module+'.rclpy.ok', return_value=True), \
                patch(module+'.time.monotonic', side_effect=lambda: clock[0]), \
                patch(module+'.time.sleep', side_effect=tick):
            result = node._feedback_close_gripper()
        node._command_sim_gripper.assert_called_once_with(0.)
        return result, node

    def test_two_stalled_fingers_nominate_contact(self):
        result, node = self.run_close(.019, -.028)
        self.assertTrue(result)
        self.assertEqual(node._feedback_event.call_args.args[0], 'gripper_contact_candidate')
        node._feedback_stop.assert_not_called()

    def test_empty_closure_is_not_a_grasp(self):
        self.assertFalse(self.run_close(.00062, 0.)[0])

    def test_one_sided_or_moving_contact_is_insufficient(self):
        self.assertFalse(self.run_close(.0006, -.029)[0])
        self.assertFalse(self.run_close(.019, -.028, moving=True)[0])

    def test_stale_or_nonfinite_encoders_stop(self):
        self.assertFalse(self.run_close(.019, -.028, fresh=False)[0])
        self.assertFalse(self.run_close(float('nan'), -.028)[0])


if __name__ == '__main__':
    unittest.main()
