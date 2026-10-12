import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class ReleaseRecoveryOrderTests(unittest.TestCase):
    def fixture(self):
        node = RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._sim_arm_joint_positions = dict.fromkeys(PIPER_JOINT_NAMES, 0.)
        node._piper_kinematics = NS(tool_front_extent_m=.1358, tool_contact_offset_m=.1)
        node._feedback_config = {'tool_tolerance_m': .015}
        node._fingers_released = Mock(return_value=True)
        node._feedback_fresh = Mock(return_value=True)
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        node._publish_sim_cmd_vel = Mock()
        node._feedback_tool = Mock(return_value=np.array([.4, 0., .75]))
        node._feedback_move_tool = Mock(return_value=True)
        node._feedback_relax_released_wrist = Mock(return_value=True)
        return node

    def run_retreat(self, node):
        planner = Mock(return_value=(np.array([.5, 0., .75]), np.array([1., 0., 0.]), [
            (np.array([.4, 0., .75]), np.eye(3), np.zeros(6))]))
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.plan_released_retreat', planner):
            return node._feedback_retract_released_handle(.045)

    def test_restore_wrist_only_after_encoder_verified_axial_clearance(self):
        node = self.fixture()
        timeline = Mock()
        timeline.attach_mock(node._feedback_move_tool, 'withdraw')
        timeline.attach_mock(node._feedback_tool, 'measure')
        timeline.attach_mock(node._feedback_relax_released_wrist, 'restore')
        self.assertTrue(self.run_retreat(node))
        self.assertEqual([call[0] for call in timeline.mock_calls], ['withdraw', 'measure', 'restore'])

    def test_failed_withdrawal_cannot_restore_or_report_success(self):
        node = self.fixture()
        node._feedback_move_tool.return_value = False
        self.assertFalse(self.run_retreat(node))
        node._feedback_relax_released_wrist.assert_not_called()

    def test_commanded_clearance_without_measured_clearance_is_rejected(self):
        node = self.fixture()
        node._feedback_tool.return_value = np.array([.499, 0., .75])
        self.assertFalse(self.run_retreat(node))
        node._feedback_relax_released_wrist.assert_not_called()

    def test_closed_jaws_prevent_any_recovery_motion(self):
        node = self.fixture()
        node._fingers_released.return_value = False
        self.assertFalse(self.run_retreat(node))
        node._feedback_move_tool.assert_not_called()
        node._feedback_relax_released_wrist.assert_not_called()


if __name__ == '__main__':
    unittest.main()
