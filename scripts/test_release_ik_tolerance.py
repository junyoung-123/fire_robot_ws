import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np

from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES, PIPER_JOINT_LIMITS


class ReleaseIKToleranceTests(unittest.TestCase):
    def fixture(self):
        node = ManipulationNode.__new__(ManipulationNode)
        joints = np.mean(PIPER_JOINT_LIMITS, axis=1)
        node._sim_arm_joint_positions = dict(zip(PIPER_JOINT_NAMES, joints))
        node._feedback_joint_bias = np.zeros(6)
        node._feedback_tool = Mock(return_value=np.array([.5, 0., .75]))
        node._feedback_contact_rotation = Mock(return_value=np.eye(3))
        node._piper_kinematics = NS(solve=Mock(return_value=NS(positions=joints)))
        node._command_sim_arm_targets = Mock(return_value=True)
        node._feedback_stop = Mock(return_value=False)
        return node

    def test_every_released_servo_has_ik_residual_below_its_two_mm_gate(self):
        for stage in ('unload_finger_release', 'released_wrist_unload', 'released_wrist_retreat'):
            with self.subTest(stage=stage):
                node = self.fixture()
                self.assertTrue(node._feedback_command_point(np.array([.49, 0., .75]), stage))
                tolerance = node._piper_kinematics.solve.call_args.kwargs['position_tolerance_m']
                self.assertEqual(tolerance, .001)
                self.assertLess(tolerance, .002)

    def test_existing_approach_ik_tolerance_is_unchanged(self):
        node = self.fixture()
        self.assertTrue(node._feedback_command_point(np.array([.49, 0., .75]), 'observed_pre_grasp'))
        self.assertEqual(node._piper_kinematics.solve.call_args.kwargs['position_tolerance_m'], .003)

    def test_unreachable_released_goal_is_rejected(self):
        node = self.fixture()
        node._piper_kinematics.solve.return_value = None
        self.assertFalse(node._feedback_command_point(np.array([.49, 0., .75]), 'released_wrist_retreat'))
        node._command_sim_arm_targets.assert_not_called()


if __name__ == '__main__':
    unittest.main()
