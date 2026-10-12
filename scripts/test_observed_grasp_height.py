import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class ObservedGraspHeightTests(unittest.TestCase):
    def fixture(self, height):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._piper_kinematics=NS(forward=Mock(return_value=(np.array([.6,0.,height]),np.eye(3))))
        node._feedback_base_yaw=Mock(return_value=0.)
        node._feedback_fresh=Mock(return_value=True)
        node._feedback_event=Mock()
        node._feedback_stop=Mock(return_value=False)
        node._sim_door_position=0.
        node._feedback_arm_stamp=1.
        return node

    def run_press(self, node, observed_height):
        with patch('fire_robot_manipulation.preserved_contact_policy.ManipulationNode._feedback_press',return_value=True) as press:
            result=node._feedback_press(NS(point=NS(z=observed_height)))
            return result,press.called

    def test_grasp_low_by_twelve_mm_cannot_become_an_encoder_baseline(self):
        node=self.fixture(.752)
        success,called=self.run_press(node,.764)
        self.assertFalse(success)
        self.assertFalse(called)

    def test_camera_height_and_actual_fk_are_both_preserved(self):
        node=self.fixture(.761)
        success,called=self.run_press(node,.764)
        self.assertTrue(success)
        self.assertTrue(called)
        self.assertEqual(node._observed_prepress_height,.764)
        self.assertEqual(node._measured_prepress_height,.761)

    def test_invalid_depth_height_does_not_command_press(self):
        for height in (float('nan'),float('inf')):
            node=self.fixture(.764)
            success,called=self.run_press(node,height)
            self.assertFalse(success)
            self.assertFalse(called)


if __name__=='__main__':
    unittest.main()
