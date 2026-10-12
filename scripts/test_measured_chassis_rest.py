import math
import time
import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import ast
from pathlib import Path
import numpy as np
from fire_robot_manipulation.contact_feedback import measured_chassis_rest
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class MeasuredChassisRestTests(unittest.TestCase):
    def test_commanded_stop_does_not_prove_measured_rest(self):
        self.assertFalse(measured_chassis_rest([1.,0.,0.,0.],[1.2,0.,0.,-.01],[.67,.25,.74],.0004))
        self.assertFalse(measured_chassis_rest([1.,0.,0.,0.],[1.2,.002,0.,0.],[.67,.25,.74],.0004))

    def test_fresh_separated_static_poses_confirm_rest(self):
        self.assertTrue(measured_chassis_rest([1.,0.,0.,0.],[1.2,.00005,0.,.0001],[.67,.25,.74],.0004))
        self.assertFalse(measured_chassis_rest([1.,0.,0.,0.],[1.01,0.,0.,0.],[.67,.25,.74],.0004))

    def test_yaw_wrap_and_invalid_data(self):
        self.assertTrue(measured_chassis_rest([1.,0.,0.,math.pi],[1.2,0.,0.,-math.pi],[.67,.25,.74],.0004))
        with self.assertRaises(ValueError):
            measured_chassis_rest([1.,0.,0.,0.],[.9,0.,0.,0.],[.67,.25,.74],.0004)


class OwnWheelContactMotionTests(unittest.TestCase):
    def fixture(self):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._contact_wheel_samples=deque(maxlen=30)
        node._sim_time_sec=Mock(return_value=20.)
        node._feedback_base_pose_sample=Mock(return_value=(19.98,np.array([.1,.2,.3])))
        return node

    def message(self,stamp=19.98,frame='odom',child='base_footprint',yaw=.4):
        sec=int(stamp)
        return NS(header=NS(stamp=NS(sec=sec,nanosec=round((stamp-sec)*1.e9)),frame_id=frame),
            child_frame_id=child,pose=NS(pose=NS(position=NS(x=7.,y=-8.),
                orientation=NS(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2)))),
            twist=NS(twist=NS(linear=NS(x=.02),angular=NS(z=.04))))

    def test_map_correction_cannot_invent_chassis_motion(self):
        node=self.fixture()
        node._contact_wheel_callback(self.message())
        stamp,pose=node._feedback_probe_pose_sample()
        self.assertAlmostEqual(stamp,19.98)
        np.testing.assert_allclose(pose,[7.,-8.,.4])
        node._feedback_base_pose_sample.assert_called_once()
        node._feedback_base_pose_sample.return_value=(20.,np.array([.2,.4,.6]))
        np.testing.assert_allclose(node._feedback_probe_pose_sample()[1],pose)

    def test_bad_frame_invalid_quaternion_or_nan_cannot_drive(self):
        for msg in (self.message(frame='map'),self.message(child='door'),self.message(yaw=float('nan'))):
            node=self.fixture(); node._contact_wheel_callback(msg)
            with self.assertRaises(ValueError): node._feedback_wheel_pose_sample()
        node=self.fixture();msg=self.message();msg.pose.pose.orientation.w=2.
        node._contact_wheel_callback(msg)
        self.assertEqual(len(node._contact_wheel_samples),0)

    def test_future_callback_does_not_hide_fresh_original_stamp(self):
        node=self.fixture()
        for stamp in (19.98,20.02,19.96): node._contact_wheel_callback(self.message(stamp))
        self.assertAlmostEqual(node._feedback_wheel_pose_sample()[0],19.98)
        node._sim_time_sec.return_value=20.18
        with self.assertRaises(ValueError): node._feedback_wheel_pose_sample()

    def test_old_wall_callback_is_not_retimestampped_or_accepted(self):
        node=self.fixture()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',return_value=time.monotonic()-2.):
            node._contact_wheel_callback(self.message())
        with self.assertRaises(ValueError): node._feedback_wheel_pose_sample()

    def rest_fixture(self):
        node=self.fixture()
        node._feedback_full_handle_follow=True
        node._feedback_config={'follow_step_m':.008}
        node._publish_sim_cmd_vel=Mock()
        node._feedback_fresh=Mock(return_value=True)
        node._feedback_tool=Mock(return_value=np.array([.55,.05,.74]))
        node._feedback_follow_contact=Mock(return_value=True)
        node._feedback_event=Mock()
        node._feedback_stop=Mock(return_value=False)
        return node

    def test_stopped_rest_reacquires_original_stamps_after_callback_ordering_gap(self):
        for missing in ('future','old_callback'):
            node=self.rest_fixture()
            clock=[20.]
            node._sim_time_sec.side_effect=lambda:clock[0]
            if missing=='future':
                node._contact_wheel_callback(self.message(20.02))
            else:
                node._contact_wheel_samples.append((19.98,time.monotonic()-2.,np.array([7.,-8.,.4])))
            def fresh_callback(_):
                clock[0]+=.05
                node._contact_wheel_callback(self.message(clock[0]))
            with patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True), \
                 patch('fire_robot_manipulation.robot_observation_manipulation_node.time.sleep',side_effect=fresh_callback):
                self.assertTrue(node._feedback_wait_contact_chassis_rest())
            node._feedback_stop.assert_not_called()
            self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
            events=[call.args[0] for call in node._feedback_event.call_args_list]
            self.assertIn('chassis_rest_wheel_reobservation',events)
            self.assertIn('chassis_rest_confirmed',events)
            self.assertAlmostEqual(node._contact_wheel_samples[-1][0],clock[0])

    def test_missing_wheels_keep_stop_and_original_rest_deadline(self):
        node=self.rest_fixture()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True), \
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',side_effect=[100.,100.,100.,131.]), \
             patch('fire_robot_manipulation.robot_observation_manipulation_node.time.sleep'):
            self.assertFalse(node._feedback_wait_contact_chassis_rest())
        node._feedback_stop.assert_called_once_with('Chassis did not measurably stop before held probe')
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))
        node._feedback_follow_contact.assert_not_called()

    def test_rest_uses_wheel_motion_but_keeps_original_tolerance(self):
        source=Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation/fire_robot_manipulation/robot_observation_manipulation_node.py'
        tree=ast.parse(source.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                    and n.name=='_feedback_wait_contact_chassis_rest')
        calls={ast.unparse(n.func) for n in ast.walk(method) if isinstance(n,ast.Call)}
        self.assertIn('self._feedback_wheel_pose_sample',calls)
        self.assertNotIn('self._feedback_base_pose_sample',calls)
        guard=next(n for n in ast.walk(method) if isinstance(n,ast.Call) and ast.unparse(n.func)=='measured_chassis_rest')
        self.assertEqual(ast.unparse(guard.args[3]),"0.05 * self._feedback_config['follow_step_m']")


if __name__=='__main__':
    unittest.main()
