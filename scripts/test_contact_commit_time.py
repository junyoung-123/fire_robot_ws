import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np
from geometry_msgs.msg import PointStamped
from rclpy.time import Time
from tf2_ros import TransformException
from fire_robot_manipulation.minimal_observed_angle_node import MinimalObservedAngle
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


MODULE='fire_robot_manipulation.robot_observation_manipulation_node'


class ContactCommitTimeTests(unittest.TestCase):
    def fixture(self,stamp=20.):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._arm_observation_history=deque([(stamp,{n:.1 for n in PIPER_JOINT_NAMES})])
        node._piper_kinematics=NS(forward=Mock(return_value=(np.array([.5,.1,.74]),np.eye(3))))
        node._sim_time_sec=Mock(return_value=20.02)
        node._feedback_fresh=Mock(return_value=True)
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        point=PointStamped();point.header.frame_id='base_link'
        point.header.stamp=Time(seconds=20.).to_msg()
        measured=PointStamped();measured.header.frame_id='base_link'
        return node,point,measured

    def run_scope(self):
        return patch(MODULE+'.rclpy.ok',return_value=True)

    def test_delayed_tf_is_reacquired_with_original_times_and_zero_chassis_command(self):
        node,point,measured=self.fixture()
        result=(NS(x=1.,y=2.,z=3.),NS(x=1.,y=2.,z=3.))
        with self.run_scope(),patch(MODULE+'.time.sleep'), \
                patch.object(MinimalObservedAngle,'_feedback_commit_probe_points',
                    side_effect=[TransformException('latest pose is 19.95'),result]) as transform:
            self.assertEqual(node._feedback_commit_probe_points(point,measured),result)
        self.assertEqual(transform.call_count,2)
        self.assertEqual(point.header.stamp,Time(seconds=20.).to_msg())
        self.assertEqual(measured.header.stamp,Time(seconds=20.).to_msg())
        self.assertEqual([measured.point.x,measured.point.y,measured.point.z],[.5,.1,.74])
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_future_arm_callback_waits_for_clock_without_retimestampping(self):
        node,point,measured=self.fixture(stamp=20.04)
        node._sim_time_sec.side_effect=[20.02,20.05]
        with self.run_scope(),patch(MODULE+'.time.sleep'), \
                patch.object(MinimalObservedAngle,'_feedback_commit_probe_points',return_value=('target','actual')) as transform:
            self.assertEqual(node._feedback_commit_probe_points(point,measured),('target','actual'))
        transform.assert_called_once()
        self.assertEqual(measured.header.stamp,Time(seconds=20.04).to_msg())

    def test_available_command_time_encoders_are_used_without_future_tf(self):
        node,point,measured=self.fixture(stamp=19.953)
        node._arm_observation_history.append((20.003,{n:.2 for n in PIPER_JOINT_NAMES}))
        with self.run_scope(),patch.object(MinimalObservedAngle,'_feedback_commit_probe_points',
                                         return_value=('target','actual')):
            self.assertEqual(node._feedback_commit_probe_points(point,measured),('target','actual'))
        self.assertEqual(measured.header.stamp,Time(seconds=19.953).to_msg())
        self.assertEqual(point.header.stamp,Time(seconds=20.).to_msg())
        node._piper_kinematics.forward.assert_called_once_with([.1]*6)

    def test_missing_or_stale_own_sample_cannot_commit(self):
        for missing in (True,False):
            node,point,measured=self.fixture(stamp=19.8)
            if missing: node._arm_observation_history.clear()
            with self.run_scope(),patch.object(MinimalObservedAngle,'_feedback_commit_probe_points') as transform, \
                    self.assertRaises(ValueError):
                node._feedback_commit_probe_points(point,measured)
            transform.assert_not_called()

    def test_historical_tf_wait_is_bounded_and_never_refreshes_original_stamps(self):
        node,point,measured=self.fixture()
        node._sim_time_sec.side_effect=[20.02,20.16]
        with self.run_scope(),patch(MODULE+'.time.sleep'), \
                patch(MODULE+'.time.monotonic',side_effect=[100.,100.,102.]), \
                patch.object(MinimalObservedAngle,'_feedback_commit_probe_points',side_effect=TransformException('pending')) as transform, \
                self.assertRaisesRegex(ValueError,'deadline'):
            node._feedback_commit_probe_points(point,measured)
        transform.assert_called_once()
        self.assertEqual(point.header.stamp,Time(seconds=20.).to_msg())

    def test_delayed_historical_transform_is_not_fresh_control_input(self):
        node,point,measured=self.fixture()
        node._lever_tracking=True
        node._control_plane_confirmations=[dict(stamp=20.01,angle=.25),dict(stamp=20.18,angle=.27)]
        node._sim_time_sec.side_effect=[20.02,20.20]
        with self.run_scope(),patch(MODULE+'.time.sleep'), \
                patch.object(MinimalObservedAngle,'_feedback_commit_probe_points',
                    side_effect=[TransformException('pending'),('target','actual')]):
            self.assertEqual(node._feedback_commit_probe_points(point,measured),('target','actual'))
        self.assertEqual(node._feedback_commit_panel_angle,.25)
        self.assertEqual(measured.header.stamp,Time(seconds=20.).to_msg())
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_lost_independent_feedback_cannot_authorize_commit(self):
        node,point,measured=self.fixture();node._feedback_fresh.return_value=False
        with self.run_scope(),patch.object(MinimalObservedAngle,'_feedback_commit_probe_points') as transform, \
                self.assertRaisesRegex(ValueError,'independent robot feedback'):
            node._feedback_commit_probe_points(point,measured)
        transform.assert_not_called()


if __name__=='__main__':
    unittest.main()
