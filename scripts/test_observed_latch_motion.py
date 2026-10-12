import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS
import numpy as np
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.minimal_observed_angle_node import MinimalObservedAngle


MODULE='fire_robot_manipulation.robot_observation_manipulation_node'


class ObservedLatchMotionTests(unittest.TestCase):
    def fixture(self):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._feedback_press_depth=.025
        node._feedback_press_origin=np.array([.5,.1,.765])
        node._phase='LATCH_CLEAR_PUSH'
        node._feedback_stop=Mock(return_value=False)
        node._sim_lever_position=-.2
        node._sim_lever_initial_position=0.
        node._feedback_config={'press_sign':-1.,'press_roll_lead_rad':.06,'opening_speed_mps':.012}
        node._observed_latch_geometry=[(t,0.,0.,0.) for t in range(12)]
        node._feedback_full_handle_follow=False
        node._feedback_last_cartesian_target=np.array([.5,.1,.74])
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._sim_door_position=-.21
        node._feedback_base_yaw=Mock(return_value=0.)
        node._feedback_fresh=Mock(return_value=True)
        node._feedback_tool=Mock(return_value=np.array([.5,.1,.74]))
        node._publish_sim_cmd_vel=Mock()
        node._feedback_begin_encoder_follow=Mock(return_value=True)
        node._feedback_event=Mock()
        node._feedback_panel_barrier_geometry=Mock(return_value=(np.array([.5,.1,.74]),np.array([1.,0.,0.]),.03))
        node._sim_arm_joint_positions={name:0. for name in ('joint1','joint2','joint3','joint4','joint5','joint6')}
        node._piper_kinematics=NS(contact_velocity_barriers=Mock(return_value=(np.zeros((1,3)),np.ones(1),.3)))
        node._feedback_scene_constraints=Mock(return_value=(np.zeros((2,3)),np.ones(2)))
        node._feedback_contact_velocity=Mock(return_value=(.011,-.04,np.array([.5,.1,.74]),.09))
        return node

    def fit(self):
        return dict(curvature_per_m=-1.5,radius_m=2/3,radius_uncertainty_m=.02)

    def test_unobserved_curvature_never_invents_a_turn(self):
        node=self.fixture()
        with patch(MODULE+'.observed_contact_curvature',return_value=None):
            self.assertEqual(node._feedback_latch_motion(),(.02,0.,None))
        node._feedback_contact_velocity.assert_not_called()
        node._feedback_begin_encoder_follow.assert_not_called()

    def test_supported_motion_uses_arm_panel_and_scan_barriers(self):
        node=self.fixture()
        with patch(MODULE+'.observed_contact_curvature',return_value=self.fit()):
            self.assertEqual(node._feedback_latch_motion(),(.011,-.04,.012))
        self.assertTrue(node._feedback_full_handle_follow)
        node._feedback_begin_encoder_follow.assert_called_once()
        self.assertEqual(node._feedback_held_height,.74)
        options=node._feedback_contact_velocity.call_args.kwargs
        self.assertEqual(options['observed_curvature'],-1.5)
        self.assertEqual(options['margin'],.03)
        self.assertIn('arm_constraints',options)
        self.assertIn('scene_constraints',options)
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)

    def test_latch_roll_does_not_ratchet_with_encoder_residuals(self):
        node=self.fixture()
        with patch(MODULE+'.observed_contact_curvature',return_value=self.fit()):
            node._feedback_latch_motion()
            baseline=node._latch_press_roll
            node._sim_lever_position=-.6
            node._feedback_latch_motion()
            self.assertEqual(node._latch_press_roll,baseline)
            node._feedback_press_depth+=.005
            node._feedback_latch_motion()
            self.assertAlmostEqual(node._latch_press_roll,-.66)
        node._feedback_begin_encoder_follow.assert_called_once()

    def test_stale_feedback_and_failed_handoff_cannot_authorize_motion(self):
        for stale in (True,False):
            node=self.fixture()
            node._feedback_fresh.return_value=not stale
            node._feedback_begin_encoder_follow.return_value=False
            with patch(MODULE+'.observed_contact_curvature',return_value=self.fit()), self.assertRaises(ValueError):
                node._feedback_latch_motion()
            node._feedback_contact_velocity.assert_not_called()

    def test_infeasible_current_constraints_are_not_bypassed(self):
        node=self.fixture()
        node._feedback_contact_velocity.side_effect=ValueError('unsafe current scan')
        with patch(MODULE+'.observed_contact_curvature',return_value=self.fit()), self.assertRaisesRegex(ValueError,'unsafe current scan'):
            node._feedback_latch_motion()
        self.assertTrue(all(call.args==(0.,0.) for call in node._publish_sim_cmd_vel.call_args_list))

    def test_explicit_press_refreshes_roll_before_dispatch_not_next_probe(self):
        node=self.fixture()
        node._latch_press_roll=-.15
        goal=np.array([.5,.1,.735])
        with patch.object(MinimalObservedAngle,'_feedback_command_point',return_value=True) as command:
            self.assertTrue(node._feedback_command_point(goal,'feedback_latch_press'))
            self.assertAlmostEqual(node._latch_press_roll,-.26)
            node._feedback_press_depth=.03
            node._sim_lever_position=-.25
            with patch(MODULE+'.observed_contact_curvature',return_value=None):
                node._feedback_latch_motion()
            self.assertAlmostEqual(node._latch_press_roll,-.26)
            command.assert_called_once()

    def test_stale_explicit_press_never_dispatches_or_updates_roll(self):
        node=self.fixture(); node._latch_press_roll=-.15
        node._feedback_fresh.return_value=False
        with patch.object(MinimalObservedAngle,'_feedback_command_point',return_value=True) as command:
            self.assertFalse(node._feedback_command_point(np.array([.5,.1,.735]),'feedback_latch_press'))
        command.assert_not_called()
        self.assertEqual(node._latch_press_roll,-.15)


if __name__=='__main__':
    unittest.main()
