import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from tf2_ros import TransformException
import numpy as np
from fire_robot_manipulation.contact_feedback import measured_normal_preload_release, tracking_bounded_normal_advance
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class NormalPreloadReleaseTests(unittest.TestCase):
    def test_normal_advance_uses_remaining_error_not_nominal_time_only(self):
        target=np.array([.5,0.,.75]);actual=np.array([.488,0.,.756])
        step=tracking_bounded_normal_advance(target,actual,[1.,0.,0.],.0042,.008,.015)
        self.assertLess(step,.0001)
        self.assertAlmostEqual(np.linalg.norm(actual-(target+[step,0.,0.])),.0135)

    def test_normal_advance_does_not_move_an_anchor_or_relax_tolerance(self):
        target=np.array([.5,0.,.75]);actual=target.copy()
        self.assertEqual(tracking_bounded_normal_advance(target,actual,[1.,0.,0.],.0042,.008,.015),.0042)
        np.testing.assert_allclose(target,[.5,0.,.75])
        with self.assertRaises(ValueError):
            tracking_bounded_normal_advance(target,[.484,0.,.75],[1.,0.,0.],.0042,.008,.015)

    def test_exhausted_or_transverse_budget_cannot_authorize_more_normal_push(self):
        for actual in ([.486,0.,.75],[.5,.014,.75]):
            self.assertEqual(tracking_bounded_normal_advance([.5,0.,.75],actual,[1.,0.,0.],.0042,.008,.015),0.)

    def test_normal_advance_is_panel_rotation_equivariant(self):
        target=np.array([.5,0.,.75]);normal=np.array([.8,-.6,0.])
        actual=target-.012*normal
        self.assertAlmostEqual(tracking_bounded_normal_advance(target,actual,normal,.0042,.008,.015),.0015)

    def test_normal_advance_rejects_invalid_geometry(self):
        for normal,step in (([2.,0.,0.],.004),([0.,0.,1.],.004),([1.,0.,0.],-.004),([1.,0.,0.],.009)):
            with self.assertRaises(ValueError):
                tracking_bounded_normal_advance([.5,0.,.75],[.5,0.,.75],normal,step,.008,.015)

    def release(self, actual, used=0., normal=(1.,0.,0.)):
        return measured_normal_preload_release([.5,0.,.75],actual,normal,.04,.008,.015,used)

    def test_unload_keeps_cartesian_reserve_and_never_changes_vertical_press(self):
        goal,travel=self.release([.486,.002,.751])
        self.assertAlmostEqual(np.linalg.norm(goal-[.486,.002,.751]),.9*.015)
        self.assertGreater(goal[0],.486)
        self.assertEqual(goal[2],.75)
        self.assertLess(travel,.001)

    def test_cumulative_budget_cannot_integrate_a_loaded_residual_indefinitely(self):
        goal,travel=self.release([.4851,0.,.75],used=.007)
        self.assertAlmostEqual(travel,.001)
        np.testing.assert_allclose(goal,[.499,0.,.75])
        goal,travel=self.release([.486,0.,.75],used=.008)
        self.assertEqual(travel,0.)
        np.testing.assert_allclose(goal,[.5,0.,.75])

    def test_does_not_push_to_chase_positive_error_or_rebase_small_error(self):
        for actual in ([.51,0.,.75],[.495,0.,.75]):
            point,travel=self.release(actual)
            np.testing.assert_allclose(point,[.5,0.,.75])
            self.assertEqual(travel,0.)

    def test_existing_r209_failure_is_still_rejected(self):
        with self.assertRaises(ValueError):
            self.release([.5-.015290006845457133,0.,.75-.0006383672159829823])

    def test_rotated_panel_unloads_only_along_its_observed_normal(self):
        normal=np.array([.8,-.6,0.])
        actual=np.array([.5,0.,.75])-.014*normal
        point,travel=self.release(actual,normal=normal)
        np.testing.assert_allclose(point-np.array([.5,0.,.75]),-.0005*normal)

    def test_invalid_budget_and_vertical_or_total_error_fail_closed(self):
        for actual,used in (([.486,0.,.77],0.),([.49,.04,.75],0.),
                            ([.49,0.,.75],-.001),([.49,0.,.75],.009)):
            with self.subTest(actual=actual,used=used),self.assertRaises(ValueError):
                self.release(actual,used)

    def node(self):
        guard=patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True)
        guard.start(); self.addCleanup(guard.stop)
        node=object.__new__(RobotObservationManipulation)
        node._feedback_config=dict(follow_step_m=.008,tool_tolerance_m=.015,
                                   tracking_limit_m=.04,press_sign=-1.)
        node._feedback_full_handle_follow=True
        node._feedback_tool=lambda:np.array([.486,0.,.75])
        node._feedback_last_cartesian_target=np.array([.5,0.,.75])
        node._feedback_fresh=lambda:True
        node._feedback_contact_orientation_reference=(0.,0.,np.eye(3))
        node._sim_door_position=0.;node._feedback_base_yaw=lambda:0.
        node._feedback_wait_contact_chassis_rest=Mock(return_value=True)
        node._lever_estimate_valid=True;node._encoder_panel_agreement=True
        node._sim_arm_joint_positions={name:0. for name in PIPER_JOINT_NAMES}
        node._sim_arm_joint_positions.update(gripper_left_joint=.026,gripper_right_joint=-.026)
        node._piper_kinematics=NS(forward=lambda q:(np.zeros(3),np.eye(3)),
                                _rotation_error=lambda a,b:np.zeros(3))
        node._feedback_contact_rotation=lambda:np.eye(3)
        node._sim_lever_position=-.2;node._sim_lever_initial_position=0.
        node._sim_lever_press_min_angle=.08
        node._feedback_probe_pose_sample=lambda:(10.,np.zeros(3))
        node._sim_time_sec=lambda:10.05
        node._publish_sim_cmd_vel=Mock()
        node._feedback_qualify_probe_reference=Mock()
        node._manipulation_frame='base_link';node._contact_reference_frame='contact_scan'
        node._tf_buffer=NS(transform=lambda p,*a,**kw:p)
        node._feedback_command_point=Mock(return_value=True)
        node._feedback_sweep_state=(np.array([.5,0.,.75]),np.array([.486,0.,.75]),1.8)
        node._feedback_press_origin=np.array([.5,0.,.77])
        node._feedback_event=Mock();node._feedback_stop=Mock(return_value=False)
        return node

    def test_adapter_retains_corrected_reference_only_after_accepted_command(self):
        node=self.node()
        self.assertTrue(node._feedback_relieve_normal_preload())
        node._feedback_wait_contact_chassis_rest.assert_called_once()
        np.testing.assert_allclose(node._feedback_sweep_state[0],[.4995,0.,.75])
        np.testing.assert_allclose(node._feedback_press_origin,[.4995,0.,.77])
        self.assertAlmostEqual(node._feedback_normal_unload_used,.0005)
        self.assertEqual(node._feedback_sweep_state[2],1.8)

    def test_lost_panel_grasp_or_rest_cannot_dispatch_correction(self):
        for field,value in (('_encoder_panel_agreement',False),('_lever_estimate_valid',False),
                            ('_sim_lever_position',0.)):
            node=self.node();setattr(node,field,value)
            self.assertFalse(node._feedback_relieve_normal_preload())
            node._feedback_command_point.assert_not_called()
        node=self.node();node._feedback_wait_contact_chassis_rest.return_value=False
        self.assertFalse(node._feedback_relieve_normal_preload())
        node._feedback_command_point.assert_not_called()

    def test_rejected_command_does_not_spend_budget_or_change_anchor(self):
        node=self.node();node._feedback_command_point.return_value=False
        self.assertFalse(node._feedback_relieve_normal_preload())
        self.assertEqual(getattr(node,'_feedback_normal_unload_used',0.),0.)
        np.testing.assert_allclose(node._feedback_sweep_state[0],[.5,0.,.75])

    def test_delayed_tf_waits_stopped_without_changing_requested_timestamp(self):
        node=self.node(); stamps=[]
        def transform(point,*args,**kwargs):
            stamps.append((point.header.stamp.sec,point.header.stamp.nanosec))
            if len(stamps)==1:
                raise TransformException('future by 0.05 sec')
            return point
        node._tf_buffer.transform=transform
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True):
            self.assertTrue(node._feedback_relieve_normal_preload())
        self.assertEqual(stamps,[(10,0),(10,0)])
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_expired_original_pose_never_dispatches_unloading(self):
        node=self.node();node._sim_time_sec=lambda:10.16
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.rclpy.ok',return_value=True):
            self.assertFalse(node._feedback_relieve_normal_preload())
        node._feedback_command_point.assert_not_called()

    def test_r212_early_sample_does_not_spend_late_settling_budget(self):
        target=np.array([.5238950606255292,.08960386807605841,.7557517239963584])
        actual=np.array([.5189961450289248,.09826739782303504,.762980943572061])
        normal=np.array([.9352416401633296,-.354009992097123,0.])
        released,travel=measured_normal_preload_release(target,actual,normal,.04,.008,.015,0.)
        self.assertEqual(travel,0.)
        np.testing.assert_allclose(released,target)

    def test_combined_error_cannot_be_fixed_by_erasing_normal_alone(self):
        point,travel=self.release([.493,.014,.75])
        self.assertEqual(travel,0.)
        np.testing.assert_allclose(point,[.5,0.,.75])


if __name__=='__main__':
    unittest.main()
