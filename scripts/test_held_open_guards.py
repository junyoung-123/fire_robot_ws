import unittest
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
from fire_robot_manipulation.preserved_contact_policy import ManipulationNode
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES
from fire_robot_manipulation.contact_feedback import ObservedContactProgress


class ObservedContactProgressTests(unittest.TestCase):
    def test_measured_correction_keeps_bounded_recovery_alive(self):
        watch = ObservedContactProgress(0., .4)
        target = np.array([.5, -.15, .75])
        for stamp in range(0, 51, 5):
            watch.observe(float(stamp), .4, [.5, .05-stamp*.002, .75], target)
        self.assertEqual(watch.last_progress, 50.)
        self.assertEqual(watch.last_opening, 0.)

    def test_target_change_without_robot_motion_is_not_progress(self):
        watch = ObservedContactProgress(0., .4)
        actual = [.5, .05, .75]
        for stamp in range(0, 21, 5):
            watch.observe(float(stamp), .4, actual, [.5, -.15+stamp*.005, .75])
        with self.assertRaisesRegex(ValueError, 'no measured'):
            watch.observe(21., .4, actual, actual)

    def test_measured_opening_resets_recovery_budget(self):
        watch = ObservedContactProgress(0., .4)
        for stamp in range(0, 100, 10):
            watch.observe(float(stamp), .4+stamp*.002, [.5, 0., .75], [.5, 0., .75])
        self.assertEqual(watch.last_opening, 90.)

    def test_posture_progress_cannot_replace_opening_indefinitely(self):
        watch = ObservedContactProgress(0., .4)
        for stamp in range(0, 76, 5):
            watch.observe(float(stamp), .4, [.5, .1-stamp*.002, .75], [.5, -.15, .75])
        with self.assertRaisesRegex(ValueError, 'bounded budget'):
            watch.observe(76., .4, [.5, -.052, .75], [.5, -.15, .75])

    def test_unchanged_encoders_and_angle_still_fail(self):
        watch = ObservedContactProgress(0., .4)
        watch.observe(0., .4, [.5, .05, .75], [.5, -.15, .75])
        with self.assertRaisesRegex(ValueError, 'no measured'):
            watch.observe(21., .4, [.5, .05, .75], [.5, -.15, .75])

    def test_measured_heading_correction_counts_with_stationary_tcp(self):
        watch=ObservedContactProgress(0.,.4)
        for stamp in range(0,51,5):
            watch.observe(float(stamp),.4,[.6,-.1,.75],[.6,-.1,.75],
                          actual_heading=-.3-stamp*.01,preferred_heading=-.9)
        self.assertEqual(watch.last_progress,50.)
        self.assertEqual(watch.last_opening,0.)

    def test_heading_target_changes_cannot_manufacture_progress(self):
        watch=ObservedContactProgress(0.,.4)
        for stamp in range(0,21,5):
            watch.observe(float(stamp),.4,[.6,-.1,.75],[.6,-.1,.75],
                          actual_heading=-.3,preferred_heading=-.9+stamp*.01)
        with self.assertRaisesRegex(ValueError,'no measured'):
            watch.observe(21.,.4,[.6,-.1,.75],[.6,-.1,.75],
                          actual_heading=-.3,preferred_heading=-.3)

    def test_heading_correction_cannot_replace_opening_indefinitely(self):
        watch=ObservedContactProgress(0.,.4)
        for stamp in range(0,76,5):
            watch.observe(float(stamp),.4,[.6,-.1,.75],[.6,-.1,.75],
                          actual_heading=-.1-stamp*.01,preferred_heading=-1.)
        with self.assertRaisesRegex(ValueError,'bounded budget'):
            watch.observe(76.,.4,[.6,-.1,.75],[.6,-.1,.75],
                          actual_heading=-.86,preferred_heading=-1.)

    def test_incomplete_heading_observation_is_rejected(self):
        watch=ObservedContactProgress(0.,.4)
        with self.assertRaisesRegex(ValueError,'Invalid contact heading'):
            watch.observe(0.,.4,[.6,-.1,.75],[.6,-.1,.75],actual_heading=-.3)


class HeldOpenGuardTests(unittest.TestCase):
    def fixture(self, angle):
        node = ManipulationNode.__new__(ManipulationNode)
        node._feedback_press_verified = True
        node._feedback_fresh = Mock(return_value=True)
        node._sim_arm_joint_positions = dict.fromkeys(PIPER_JOINT_NAMES, 0.)
        node._sim_arm_joint_positions.update(gripper_left_joint=.026, gripper_right_joint=-.026)
        node._feedback_begin_encoder_follow = Mock(return_value=True)
        node._feedback_encoder_follow_origin = np.array([.6, .05, .75])
        node._feedback_tool = Mock(return_value=node._feedback_encoder_follow_origin.copy())
        node._feedback_contact_rotation = Mock(return_value=np.eye(3))
        node._piper_kinematics = NS(tool_front_extent_m=.1358,
            preferred_contact_posture=Mock(return_value=(np.array([.5, .05, .75]), .3)),
            contact_velocity_barriers=Mock(return_value=(np.empty((0,3)), np.empty(0), .3)))
        node._feedback_config = {'tool_tolerance_m': .015, 'press_sign': -1.}
        node._sim_time_sec = Mock(return_value=1.)
        node._feedback_base_yaw = Mock(return_value=0.)
        node._sim_door_position = angle
        node._feedback_door_msg_stamp = None
        node._feedback_joint_msg_stamp = None
        node._sim_door_initial_position = 0.
        node._sim_lever_position = -.3
        node._sim_lever_initial_position = 0.
        node._sim_lever_press_min_angle = .08
        node._feedback_event = Mock()
        node._feedback_stop = Mock(return_value=False)
        node._publish_sim_cmd_vel = Mock()
        node._feedback_opening_complete = Mock(return_value=True)
        return node

    def test_sensor_confirmed_full_angle_executes_guard_and_completes(self):
        node = self.fixture(2.1)
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok', return_value=True):
            self.assertTrue(node._feedback_handle_held_open(2.05))
        node._feedback_opening_complete.assert_called_once_with(2.1, 2.05)

    def test_observed_reverse_stops_before_next_base_probe(self):
        node = self.fixture(.4)
        calls = 0
        def tool():
            nonlocal calls
            calls += 1
            if calls == 2:
                node._sim_door_position = .3
            return np.array([.6, .05, .75])
        node._feedback_tool.side_effect = tool
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok', return_value=True):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        node._feedback_stop.assert_called_once_with('Observed door reversed during held opening')
        node._feedback_opening_complete.assert_not_called()

    def test_missing_opposed_grasp_prevents_posture_selection(self):
        node = self.fixture(.4)
        node._sim_arm_joint_positions['gripper_left_joint'] = 0.
        self.assertFalse(node._feedback_handle_held_open(2.05))
        node._feedback_begin_encoder_follow.assert_not_called()

    def test_stopped_observation_speed_does_not_decay_active_request(self):
        node = self.fixture(.4)
        node._feedback_config['opening_speed_mps'] = .012
        node._feedback_contact_orientation_reference = (0., 0., np.eye(3))
        node._manipulation_frame = 'base_link'
        node._contact_reference_frame = 'map'
        node._tf_buffer = Mock()
        node._tf_buffer.transform.side_effect = lambda message, *args, **kwargs: message
        node._feedback_probe_contact_motion = Mock(return_value=None)
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok', return_value=True), \
                patch('fire_robot_manipulation.preserved_contact_policy.robust_contact_speed', return_value=0.):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        self.assertEqual(node._feedback_probe_contact_motion.call_args.args[2], .012)

    def test_observed_stance_payload_tracks_opening_and_posture_together(self):
        node=self.fixture(.25)
        node._piper_kinematics.measured_steady_contact_posture=Mock(side_effect=ValueError('heading is not ready'))
        node._feedback_prepare_contact_stance=True
        node._observed_latch_geometry=[(i*.1,i*.02,.6,.05) for i in range(12)]
        node._feedback_config['opening_speed_mps']=.03
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._manipulation_frame='base_link'
        node._contact_reference_frame='odom'
        node._tf_buffer=Mock()
        node._tf_buffer.transform.side_effect=lambda message,*args,**kwargs:message
        node._feedback_probe_contact_motion=Mock(return_value=None)
        payloads=[]
        def event(name,**values):
            json.dumps(values,allow_nan=False)
            payloads.append((name,values))
        node._feedback_event.side_effect=event
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok',return_value=True), \
             patch('fire_robot_manipulation.preserved_contact_policy.observed_contact_curvature',
                   return_value={'curvature_per_m':-1.3}):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        stance=next(values for name,values in payloads if name=='observed_contact_stance')
        self.assertIs(type(stance['ready']),bool)
        self.assertFalse(stance['ready'])
        self.assertEqual(node._feedback_probe_contact_motion.call_args.args[2],.03)
        self.assertEqual(stance['strategy'],'joint_constrained_simultaneous_stance_and_opening')

    def test_heading_is_a_soft_goal_before_stance_ready_not_an_early_hard_gate(self):
        node=self.fixture(.25)
        node._piper_kinematics.measured_steady_contact_posture=Mock(side_effect=ValueError('heading is not ready'))
        node._feedback_prepare_contact_stance=True
        node._observed_latch_geometry=[(i*.1,i*.02,.6,.05) for i in range(12)]
        node._feedback_config['opening_speed_mps']=.03
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._manipulation_frame='base_link'
        node._contact_reference_frame='odom'
        node._tf_buffer=Mock()
        node._tf_buffer.transform.side_effect=lambda message,*args,**kwargs:message
        node._feedback_probe_contact_motion=Mock(return_value=None)
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok',return_value=True), \
             patch('fire_robot_manipulation.preserved_contact_policy.observed_contact_curvature',
                   return_value={'curvature_per_m':-1.3}), \
             patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   return_value=(.02,.04,np.array([.5,.05,.75]),.10)) as command:
            self.assertFalse(node._feedback_handle_held_open(2.05))
        curvature=command.call_args.kwargs['observed_curvature']
        self.assertAlmostEqual(command.call_args.kwargs['heading_priority_weight'],1./curvature**2)
        self.assertIsNone(command.call_args.kwargs['heading_tolerance_rad'])
        self.assertTrue(command.call_args.kwargs['limit_posture_drift'])
        self.assertTrue(command.call_args.kwargs['measured_heading_reference'])

    def test_observed_stance_cannot_advance_without_a_reliable_latch_trajectory(self):
        node=self.fixture(.25)
        node._feedback_prepare_contact_stance=True
        node._observed_latch_geometry=[]
        node._feedback_probe_contact_motion=Mock()
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok',return_value=True):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        node._feedback_probe_contact_motion.assert_not_called()
        node._feedback_stop.assert_called_once_with(
            'No independently observed latch trajectory for safe opening stance')

    def test_attainable_but_shortened_stance_does_not_replace_selected_reach(self):
        node=self.fixture(.25)
        node._piper_kinematics.preferred_contact_posture.return_value=(np.array([.55,-.23,.75]),.3)
        node._piper_kinematics.measured_steady_contact_posture=Mock(
            return_value=(np.array([.6,.05,.75]),.3))
        node._feedback_prepare_contact_stance=True
        node._observed_latch_geometry=[(i*.1,i*.02,.6,.05) for i in range(12)]
        node._feedback_config['opening_speed_mps']=.03
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._manipulation_frame='base_link'
        node._contact_reference_frame='odom'
        node._tf_buffer=Mock()
        node._tf_buffer.transform.side_effect=lambda message,*args,**kwargs:message
        node._feedback_probe_contact_motion=Mock(return_value=None)
        with patch('fire_robot_manipulation.preserved_contact_policy.rclpy.ok',return_value=True), \
             patch('fire_robot_manipulation.preserved_contact_policy.observed_contact_curvature',
                   return_value={'curvature_per_m':-1.3}):
            self.assertFalse(node._feedback_handle_held_open(2.05))
        node._piper_kinematics.measured_steady_contact_posture.assert_not_called()
        self.assertEqual(node._feedback_probe_contact_motion.call_args.args[2],.03)
        self.assertNotIn('measured_steady_stance_adopted',
                         [call.args[0] for call in node._feedback_event.call_args_list])


if __name__ == '__main__':
    unittest.main()
