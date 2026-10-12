import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class ReleasedWristUnloadTests(unittest.TestCase):
    def fixture(self):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=.035,gripper_right_joint=-.035)
        node._sim_gripper_release_half_width=.034
        node._sim_door_position=0.
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._feedback_base_yaw=lambda:0.
        node._measured_prepress_height=.75
        node._observed_prepress_height=.75
        node._feedback_config={'press_max_travel_m':.11}
        node._publish_sim_cmd_vel=Mock()
        node._feedback_fresh=Mock(return_value=True)
        node._feedback_stop=Mock(return_value=False)
        node._feedback_event=Mock()
        pose=[np.array([.6,0.,.735])]
        rotation=[measured_wrist_rotation(np.eye(3),0.,0.,-.28)]
        node._feedback_tool=lambda:pose[0].copy()
        node._piper_kinematics=NS(forward=lambda q:(pose[0].copy(),rotation[0].copy()),
            _rotation_error=lambda a,b:a[:,0]-b[:,0])
        changes=[]
        def move(point,stage):
            changes.append(point-pose[0])
            pose[0]=point.copy()
            rotation[0]=node._feedback_released_rotation.copy()
            return True
        node._feedback_move_tool=Mock(side_effect=move)
        return node,changes

    def test_unwind_returns_to_measured_prepress_height_in_bounded_steps_without_base_motion(self):
        node,changes=self.fixture()
        self.assertTrue(node._feedback_relax_released_wrist())
        self.assertGreater(len(changes),1)
        self.assertLessEqual(max(np.linalg.norm(change) for change in changes),.006)
        self.assertAlmostEqual(node._feedback_tool()[2],.75)
        np.testing.assert_allclose(node._feedback_released_rotation,np.eye(3),atol=1.e-10)
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_missing_or_implausible_height_cannot_be_used_as_target(self):
        for height in (float('nan'),1.5):
            node,_=self.fixture(); node._measured_prepress_height=height
            self.assertFalse(node._feedback_relax_released_wrist())
            node._feedback_move_tool.assert_not_called()

    def test_observed_handle_height_is_used_instead_of_lower_fk_grasp_height(self):
        node,_=self.fixture()
        node._observed_prepress_height=.754
        self.assertTrue(node._feedback_relax_released_wrist())
        self.assertAlmostEqual(node._feedback_tool()[2],.754)

    def test_missing_camera_height_cannot_be_replaced_with_a_world_constant(self):
        node,_=self.fixture()
        del node._observed_prepress_height
        self.assertFalse(node._feedback_relax_released_wrist())
        node._feedback_move_tool.assert_not_called()

    def test_closed_finger_or_stale_feedback_prohibits_unloading(self):
        for closed in (True,False):
            node,_=self.fixture()
            if closed: node._sim_arm_joint_positions['gripper_left_joint']=.02
            else: node._feedback_fresh.return_value=False
            self.assertFalse(node._feedback_relax_released_wrist())
            node._feedback_move_tool.assert_not_called()

    def test_failed_measured_arm_move_is_not_success(self):
        node,_=self.fixture(); node._feedback_move_tool.side_effect=None
        node._feedback_move_tool.return_value=False
        self.assertFalse(node._feedback_relax_released_wrist())


if __name__=='__main__':
    unittest.main()
