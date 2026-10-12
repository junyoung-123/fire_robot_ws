import unittest
from unittest.mock import Mock, patch
import numpy as np
from geometry_msgs.msg import PointStamped
from fire_robot_manipulation.preserved_contact_policy import ManipulationNode


class GraspAlignmentGateTests(unittest.TestCase):
    def fixture(self, gap):
        node = ManipulationNode.__new__(ManipulationNode)
        node._contact_reference_frame = 'map'
        handle = PointStamped()
        handle.point.x, handle.point.z = .6, .76
        node._feedback_handle_anchor = handle
        node._transform_handle_to_manipulation_frame = Mock(return_value=handle)
        node._feedback_config = {'tool_tolerance_m': .015, 'approach_step_m': .02}
        node._sim_time_sec = Mock(return_value=1.)
        node._feedback_event = Mock()
        node._feedback_fresh = Mock(return_value=True)
        node._feedback_stop = Mock(return_value=False)
        node._feedback_close_gripper = Mock(return_value=True)
        node._feedback_tool = Mock(side_effect=[np.array([.6, 0., .76-gap]),
                                               np.array([.6, 0., .758])])
        node._feedback_move_tool = Mock(return_value=True)
        return node, handle

    def test_ten_mm_gap_requires_alignment_before_closing(self):
        node, handle = self.fixture(.01)
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok', return_value=True):
            self.assertTrue(node._feedback_grasp(handle))
        node._feedback_move_tool.assert_called_once()
        np.testing.assert_allclose(node._feedback_move_tool.call_args.args[0], [.6, 0., .76])
        node._feedback_close_gripper.assert_called_once()

    def test_failed_final_alignment_cannot_close_jaws(self):
        node, handle = self.fixture(.01)
        node._feedback_move_tool.return_value = False
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok', return_value=True):
            self.assertFalse(node._feedback_grasp(handle))
        node._feedback_close_gripper.assert_not_called()

    def test_four_mm_gap_can_nominate_grasp(self):
        node, handle = self.fixture(.004)
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok', return_value=True):
            self.assertTrue(node._feedback_grasp(handle))
        node._feedback_move_tool.assert_not_called()


if __name__ == '__main__':
    unittest.main()
