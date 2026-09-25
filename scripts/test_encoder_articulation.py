import unittest
from types import SimpleNamespace
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from unittest.mock import patch
from rclpy.time import Time
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation
from fire_robot_manipulation.encoder_articulation import lever_from_wrist, articulation_from_wrist, inside_gripper_envelope, gripper_occludes_ray
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation


class EncoderArticulationTests(unittest.TestCase):
    def test_recovers_measured_roll_in_both_directions(self):
        for door in (-2.1,-.7,0.,.8,2.1):
            for base in (-1.,0.,.9):
                for lever in (-.3,0.,.3):
                    ref=measured_wrist_rotation(np.eye(3),.05,0.,.07)
                    actual=measured_wrist_rotation(ref,door,base,lever)
                    estimate,residual=lever_from_wrist(ref,actual,door,base)
                    self.assertAlmostEqual(estimate,lever)
                    self.assertLess(residual,1.e-6)

    def test_base_turn_is_not_lever_turn(self):
        ref=np.eye(3); actual=measured_wrist_rotation(ref,0.,.7,0.)
        self.assertAlmostEqual(lever_from_wrist(ref,actual,0.,.7)[0],0.)

    def test_independent_wrist_decomposition_without_door_truth(self):
        ref=measured_wrist_rotation(np.eye(3),.03,0.,.05)
        for door in (-2.2,-1.,0.,1.,2.2):
            for base in (-1.4,.1,1.1):
                for lever in (-.3,0.,.3):
                    actual=measured_wrist_rotation(ref,door,base,lever)
                    yaw,roll,residual=articulation_from_wrist(ref,actual,base)
                    self.assertAlmostEqual(yaw,door)
                    self.assertAlmostEqual(roll,lever)
                    self.assertLess(residual,1.e-6)

    def test_nonlever_motion_is_rejected(self):
        ref=np.eye(3); actual=measured_wrist_rotation(ref,.4,0.,0.)
        with self.assertRaises(ValueError): lever_from_wrist(ref,actual,0.,0.)

    def test_environment_joint_injection_ignored(self):
        state=SimpleNamespace(_sim_door_position=.1,_sim_lever_position=.2)
        for names in (['door_hinge'],['lever_joint'],['door_hinge','lever_joint'],
                      ['camera_door_rotation','lever_joint']):
            msg=JointState(); msg.name=names; msg.position=[1.]*len(names)
            RobotObservationManipulation._sim_door_state_callback(state,msg)
            self.assertEqual(state._sim_door_position,.1)
            self.assertEqual(state._sim_lever_position,.2)

    def test_nonfinite_sensor_reading_rejected(self):
        with self.assertRaises(ValueError):
            lever_from_wrist(np.eye(3),np.eye(3),float('nan'),0.)

    def test_pregrasp_has_no_lever_measurement(self):
        state=SimpleNamespace(_sim_door_initial_position=.1,_sim_door_position=.1,
            _sim_lever_position=None,_sim_lever_initial_position=None)
        self.assertTrue(RobotObservationManipulation._pregrasp_observation_consistent(state))
        self.assertIsNone(state._sim_lever_position)

    def test_pregrasp_camera_disturbance_is_rejected(self):
        state=SimpleNamespace(_sim_door_initial_position=.1,_sim_door_position=.15)
        self.assertFalse(RobotObservationManipulation._pregrasp_observation_consistent(state))

    def test_gripper_mask_does_not_hide_remote_points(self):
        for point,expected in (([0.,0.,0.],True),([0.,0.,.3],False),([.3,0.,0.],False)):
            self.assertEqual(inside_gripper_envelope(point,[0.,0.,0.],np.eye(3),.1,.1358,.012),expected)

    def test_calibrated_gripper_axes(self):
        self.assertTrue(inside_gripper_envelope([0.,.07,-.07],[0.,0.,0.],np.eye(3),.1,.1358,0.))
        self.assertFalse(inside_gripper_envelope([.07,0.,-.07],[0.,0.,0.],np.eye(3),.1,.1358,0.))

    def test_occlusion_segment_not_infinite_line(self):
        args=([0.,0.,0.],np.eye(3),.1,.1358,.012)
        self.assertTrue(gripper_occludes_ray([0.,0.,-.5],[0.,0.,.1],*args))
        self.assertFalse(gripper_occludes_ray([0.,0.,-.5],[0.,0.,-.2],*args))
        self.assertFalse(gripper_occludes_ray([.3,0.,-.5],[.3,0.,.1],*args))

    def test_base_yaw_queries_clock_time_not_latest(self):
        requested=[]
        def lookup(target,source,stamp):
            requested.append(stamp.nanoseconds)
            msg=TransformStamped(); msg.header.stamp=(Time(seconds=10.01) if stamp.nanoseconds==0 else stamp).to_msg()
            msg.transform.rotation.w=1.
            return msg
        state=SimpleNamespace(_sim_time_sec=lambda:10.,_manipulation_frame='base_link',
            _tf_buffer=SimpleNamespace(lookup_transform=lookup),_feedback_config={'max_age_sec':1.})
        with patch('rclpy.ok',return_value=True):
            self.assertEqual(RobotObservationManipulation._feedback_base_yaw(state),0.)
        self.assertEqual(requested,[0,10000000000])

    def test_held_press_does_not_accumulate_its_own_encoder_measurement(self):
        state=SimpleNamespace(_phase='HANDLE_HELD_BASE_OPEN',
            _feedback_contact_orientation_reference=(0.,0.,np.eye(3)),
            _sim_door_position=.5,_feedback_base_yaw=lambda:.1,_held_press_roll=-.2,
            _sim_lever_position=-.2)
        first=RobotObservationManipulation._feedback_contact_rotation(state)
        state._sim_lever_position=-.25
        second=RobotObservationManipulation._feedback_contact_rotation(state)
        np.testing.assert_allclose(first,second)
        yaw,roll,_=articulation_from_wrist(np.eye(3),second,.1)
        self.assertAlmostEqual(yaw,.5)
        self.assertAlmostEqual(roll,-.2)


if __name__=='__main__': unittest.main()
