import unittest
import time
from types import SimpleNamespace
from collections import deque
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from unittest.mock import patch, Mock
from rclpy.time import Time
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation
from fire_robot_manipulation.encoder_articulation import lever_from_wrist, articulation_from_wrist, fused_grasped_panel_angle, inside_gripper_envelope, gripper_occludes_ray
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES
from fire_robot_manipulation.contact_feedback import ObservedSceneGate


class EncoderArticulationTests(unittest.TestCase):
    def test_measured_fk_limits_a_supported_panel_noise_step(self):
        ref=np.eye(3)
        measured=measured_wrist_rotation(ref,.25,.8,-.35)
        angle,disagreement,roll,_=fused_grasped_panel_angle(.253+.055,ref,measured,.8,.003)
        self.assertAlmostEqual(angle,.253+.011)
        self.assertAlmostEqual(disagreement,.055)
        self.assertAlmostEqual(roll,-.35)

    def test_panel_grasp_disagreement_is_not_smoothed_away(self):
        measured=measured_wrist_rotation(np.eye(3),.25,.8,-.35)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            fused_grasped_panel_angle(.7,np.eye(3),measured,.8,0.)

    def test_encoder_increment_does_not_preserve_absolute_grasp_bias(self):
        measured=measured_wrist_rotation(np.eye(3),.46,.8,-.35)
        angle,_,_,_=fused_grasped_panel_angle(.35,np.eye(3),measured,.8,0.,
                                           previous_estimate=(.33,.44))
        self.assertAlmostEqual(angle,.35)

    def test_increment_filter_limits_a_single_panel_noise_step(self):
        measured=measured_wrist_rotation(np.eye(3),.25,.8,-.35)
        angle,_,_,_=fused_grasped_panel_angle(.305,np.eye(3),measured,.8,0.,
                                           previous_estimate=(.25,.25))
        self.assertAlmostEqual(angle,.261)

    def test_panel_wrap_is_compared_without_a_two_pi_jump(self):
        measured=measured_wrist_rotation(np.eye(3),3.12,0.,-.35)
        angle,_,_,_=fused_grasped_panel_angle(-3.10,np.eye(3),measured,0.,0.)
        self.assertTrue(3.12<angle<3.14)

    def test_compliant_wrist_prediction_cannot_leave_measured_panel_interval(self):
        measured=measured_wrist_rotation(np.eye(3),-.35,0.,-.25)
        angle,disagreement,_,_=fused_grasped_panel_angle(-.25,np.eye(3),measured,0.,0.,
            previous_estimate=(-.25,-.25),panel_uncertainty_rad=.008)
        self.assertAlmostEqual(disagreement,.10)
        self.assertLessEqual(abs(angle+.25),.008+1.e-12)

    def test_bounded_fusion_still_rejects_grasp_disagreement_or_missing_quality(self):
        measured=measured_wrist_rotation(np.eye(3),-.45,0.,-.25)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            fused_grasped_panel_angle(-.25,np.eye(3),measured,0.,0.,panel_uncertainty_rad=.008)
        measured=measured_wrist_rotation(np.eye(3),-.25,0.,-.25)
        for uncertainty in (float('nan'),-.01,.04):
            with self.subTest(uncertainty=uncertainty), self.assertRaisesRegex(ValueError,'confidence'):
                fused_grasped_panel_angle(-.25,np.eye(3),measured,0.,0.,panel_uncertainty_rad=uncertainty)

    def test_control_fusion_keeps_the_independent_panel_message_unchanged(self):
        node=object.__new__(RobotObservationManipulation)
        node._lever_tracking=True
        node._feedback_joint_msg_stamp=(10,0)
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=.026,gripper_right_joint=-.026)
        measured=measured_wrist_rotation(np.eye(3),.25,.3,-.35)
        node._piper_kinematics=SimpleNamespace(forward=Mock(return_value=(np.zeros(3),measured)))
        tf=TransformStamped();tf.header.stamp.sec=10
        tf.transform.rotation.z=float(np.sin(.5/2));tf.transform.rotation.w=float(np.cos(.5/2))
        node._tf_buffer=SimpleNamespace(lookup_transform=Mock(return_value=tf))
        node._contact_reference_frame='odom'
        node._lever_reference=(np.eye(3),.01,.2)
        node._encoder_panel_gate=ObservedSceneGate()
        node._control_plane_confirmations=deque(maxlen=40)
        node._plane_confirmations=deque([dict(stamp=10.,received=time.monotonic(),uncertainty=.008)],maxlen=40)
        node._sim_time_sec=Mock(return_value=10.)
        node._feedback_event=Mock();node._publish_sim_cmd_vel=Mock()
        msg=JointState();msg.header.stamp.sec=10;msg.name=['camera_door_rotation'];msg.position=[.31]
        with patch.object(ManipulationNode,'_sim_door_state_callback') as callback:
            node._sim_door_state_callback(msg)
        filtered=callback.call_args.args[0]
        self.assertAlmostEqual(msg.position[0],.31)
        self.assertAlmostEqual(filtered.position[0],.31)
        self.assertIsNot(msg,filtered)
        self.assertTrue(node._encoder_panel_agreement)

    def test_progress_and_reversal_use_one_estimator_but_completion_keeps_raw_panels(self):
        node=object.__new__(RobotObservationManipulation)
        node._sim_door_initial_position=0.
        node._sim_time_sec=Mock(return_value=10.)
        node._plane_confirmations=[dict(stamp=stamp,received=100.,angle=.34,
            uncertainty=.01,sensor='lidar') for stamp in (9.8,9.9,10.)]
        node._control_plane_confirmations=[dict(stamp=stamp,received=100.,angle=.24)
            for stamp in (9.8,9.9,10.)]
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',return_value=100.):
            self.assertAlmostEqual(node._feedback_progress_angle(),.24)
        self.assertTrue(all(sample['angle']==.34 for sample in node._plane_confirmations))

    def fusion_fixture(self):
        node=object.__new__(RobotObservationManipulation)
        node._lever_tracking=True
        node._feedback_joint_msg_stamp=(10,200000000)
        node._sim_arm_joint_positions=dict.fromkeys(PIPER_JOINT_NAMES,0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=.026,gripper_right_joint=-.026)
        node._piper_kinematics=SimpleNamespace(forward=Mock(return_value=(np.zeros(3),np.eye(3))))
        tf=TransformStamped();tf.header.stamp.sec=10;tf.transform.rotation.w=1.
        node._tf_buffer=SimpleNamespace(lookup_transform=Mock(return_value=tf))
        node._contact_reference_frame='odom'
        node._lever_reference=(np.eye(3),0.,0.)
        node._encoder_panel_gate=ObservedSceneGate()
        node._control_plane_confirmations=deque(maxlen=40)
        node._plane_confirmations=deque([dict(stamp=10.,received=time.monotonic(),uncertainty=.008)],maxlen=40)
        node._sim_time_sec=Mock(return_value=10.2)
        node._feedback_event=Mock();node._publish_sim_cmd_vel=Mock()
        return node

    def interval_fixture(self, angles, uncertainties):
        node=object.__new__(RobotObservationManipulation)
        node._sim_door_initial_position=0.
        node._sim_time_sec=Mock(return_value=10.)
        node._control_plane_confirmations=[dict(stamp=stamp,received=100.,angle=angle)
            for stamp,angle in zip((9.8,9.9,10.),angles)]
        node._plane_confirmations=[dict(stamp=sample['stamp'],received=100.,angle=sample['angle'],
            uncertainty=uncertainty,sensor='lidar')
            for sample,uncertainty in zip(node._control_plane_confirmations,uncertainties)]
        return node

    def test_reversal_uses_confirmed_bounds_not_one_noisy_frame(self):
        peak=self.interval_fixture((.29,.29,.29),(.03,.03,.03))
        current=self.interval_fixture((.24,.23,.17),(.03,.03,.03))
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',return_value=100.):
            peak_lower,_=peak._feedback_progress_interval()
            _,current_upper=current._feedback_progress_interval()
        self.assertLessEqual(peak_lower-current_upper,.08)

    def test_sustained_real_reversal_still_exceeds_existing_limit(self):
        peak=self.interval_fixture((.5,.5,.5),(.02,.02,.02))
        current=self.interval_fixture((.3,.3,.3),(.02,.02,.02))
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',return_value=100.):
            peak_lower,_=peak._feedback_progress_interval()
            _,current_upper=current._feedback_progress_interval()
        self.assertGreater(peak_lower-current_upper,.08)

    def test_missing_or_uncertain_raw_sensor_cannot_validate_progress(self):
        node=self.interval_fixture((.3,.3,.3),(.04,.04,.04))
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.time.monotonic',return_value=100.):
            self.assertIsNone(node._feedback_progress_interval())
            node._plane_confirmations=[]
            self.assertIsNone(node._feedback_progress_interval())

    def test_delayed_panel_uses_matching_encoder_history_not_latest(self):
        node=self.fusion_fixture()
        aligned=node._sim_arm_joint_positions.copy();aligned[PIPER_JOINT_NAMES[0]]=.1
        node._arm_observation_history=[(10.,aligned),(10.2,node._sim_arm_joint_positions.copy())]
        msg=JointState();msg.header.stamp.sec=10;msg.name=['camera_door_rotation'];msg.position=[0.]
        with patch.object(ManipulationNode,'_sim_door_state_callback') as callback:
            node._sim_door_state_callback(msg)
        callback.assert_called_once()
        self.assertEqual(node._piper_kinematics.forward.call_args.args[0][0],.1)
        self.assertEqual(node._tf_buffer.lookup_transform.call_args.args[2].nanoseconds,0)
        self.assertTrue(node._encoder_panel_agreement)

    def test_missing_independent_confidence_pauses_without_accepting_encoder_prediction(self):
        node=self.fusion_fixture();node._plane_confirmations.clear()
        node._arm_observation_history=[(10.,node._sim_arm_joint_positions.copy())]
        msg=JointState();msg.header.stamp.sec=10;msg.name=['camera_door_rotation'];msg.position=[0.]
        with patch.object(ManipulationNode,'_sim_door_state_callback') as callback:
            node._sim_door_state_callback(msg)
        callback.assert_not_called()
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)
        self.assertIsNone(node._encoder_panel_gate.pending)

    def test_encoder_milliseconds_ahead_of_tf_never_requests_a_future_pose(self):
        node=self.fusion_fixture()
        node._arm_observation_history=[(10.003,node._sim_arm_joint_positions.copy())]
        msg=JointState();msg.header.stamp.sec=10;msg.name=['camera_door_rotation'];msg.position=[0.]
        with patch.object(ManipulationNode,'_sim_door_state_callback') as callback:
            node._sim_door_state_callback(msg)
        callback.assert_called_once()
        self.assertTrue(node._encoder_panel_agreement)
        self.assertTrue(all(call.args[2].nanoseconds<=10000000000
                            for call in node._tf_buffer.lookup_transform.call_args_list))

    def test_alignment_delay_stops_motion_without_fabricating_a_contradiction(self):
        node=self.fusion_fixture()
        node._arm_observation_history=[]
        msg=JointState();msg.header.stamp.sec=10;msg.name=['camera_door_rotation'];msg.position=[0.]
        with patch.object(ManipulationNode,'_sim_door_state_callback') as callback:
            node._sim_door_state_callback(msg)
        callback.assert_not_called()
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)
        self.assertIsNone(node._encoder_panel_gate.pending)
        self.assertFalse(node._encoder_panel_agreement)

    def test_duplicate_panel_cannot_rewrite_the_encoder_filter_baseline(self):
        node=self.fusion_fixture();node._grasp_panel_filter=(.2,.3,10.)
        msg=JointState();msg.header.stamp.sec=10;msg.name=['camera_door_rotation'];msg.position=[.5]
        with patch.object(ManipulationNode,'_sim_door_state_callback') as callback:
            node._sim_door_state_callback(msg)
        callback.assert_not_called()
        self.assertEqual(node._grasp_panel_filter,(.2,.3,10.))

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
        state=SimpleNamespace(_sim_time_sec=lambda:10.,_manipulation_frame='base_link',_contact_reference_frame='map',
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
