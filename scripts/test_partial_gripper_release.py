import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np

from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class PartialGripperReleaseTests(unittest.TestCase):
    def fixture(self,left=.02,right=-.035):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._phase='RELEASE_HANDLE'
        node._sim_gripper_release_half_width=.034
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=left,gripper_right_joint=right)
        node._feedback_config={'tool_tolerance_m':.015}
        node._piper_kinematics=NS(tool_front_extent_m=.1358,tool_contact_offset_m=.1)
        node._feedback_fresh=Mock(return_value=True)
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        node._feedback_stop=Mock(return_value=False)
        node._feedback_move_tool=Mock(return_value=True)
        clock=[0.]
        node._sim_time_sec=lambda:clock[0]
        def sleep(_): clock[0]+=.05
        return node,sleep

    def run_release(self,node,sleep,planner):
        with patch('rclpy.ok',return_value=True),patch('time.sleep',side_effect=sleep),patch(
            'fire_robot_manipulation.robot_observation_manipulation_node.plan_released_retreat',planner):
            return node._wait_for_sim_gripper_release()

    def planner(self):
        return Mock(return_value=(np.zeros(3),np.array([1.,0.,0.]),[
            (np.array([.594,0.,.75]),np.eye(3),np.zeros(6))]))

    def test_asymmetric_release_withdraws_stopped_but_requires_both_actual_fingers_open(self):
        node,sleep=self.fixture()
        def freed(point,stage):
            self.assertEqual(stage,'unload_finger_release')
            node._sim_arm_joint_positions['gripper_left_joint']=.035
            return True
        node._feedback_move_tool.side_effect=freed
        self.assertTrue(self.run_release(node,sleep,self.planner()))
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
        node._feedback_move_tool.assert_called_once()

    def test_partial_open_alone_cannot_count_as_release_success(self):
        node,sleep=self.fixture()
        self.assertFalse(self.run_release(node,sleep,self.planner()))
        self.assertIn('withdrawal bound',node._feedback_stop.call_args.args[0])

    def test_no_opposing_finger_release_means_no_arm_withdrawal(self):
        node,sleep=self.fixture(.024,-.024)
        planner=self.planner()
        self.assertFalse(self.run_release(node,sleep,planner))
        planner.assert_not_called()
        node._feedback_move_tool.assert_not_called()

    def test_stale_observations_cannot_start_unload(self):
        node,sleep=self.fixture()
        node._feedback_fresh.return_value=False
        planner=self.planner()
        self.assertFalse(self.run_release(node,sleep,planner))
        planner.assert_not_called()

    def test_already_open_fingers_do_not_dispatch_extra_withdrawal(self):
        node,sleep=self.fixture(.035,-.035)
        planner=self.planner()
        self.assertTrue(self.run_release(node,sleep,planner))
        planner.assert_not_called()

    def test_failed_arm_step_never_reports_release(self):
        node,sleep=self.fixture()
        node._feedback_move_tool.return_value=False
        self.assertFalse(self.run_release(node,sleep,self.planner()))


if __name__=='__main__':
    unittest.main()
