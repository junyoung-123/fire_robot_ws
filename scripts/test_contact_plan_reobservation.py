import unittest
import numpy as np
from unittest.mock import Mock, patch
from fire_robot_manipulation.contact_feedback import SceneObservationPending
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.contact_lookahead import ContactPlan
from fire_robot_manipulation.piper_actual_kinematics import PIPER_JOINT_NAMES


class ContactPlanReobservationTests(unittest.TestCase):
    def lookahead_node(self):
        node=self.node()
        node._phase='HANDLE_HELD_BASE_OPEN'
        node._contact_scene_plan_inputs=(np.ones((3,2)),np.zeros(3,bool),np.full(3,.05))
        node._sim_arm_joint_positions={name:0. for name in PIPER_JOINT_NAMES}
        node._piper_kinematics=Mock()
        args=([.6,.05,.75],[1.,0.,0.],[.53,-.2,.75],
            [[-.2,-.2],[-.2,.2],[.3,.2],[.3,-.2]])
        kwargs=dict(observed_curvature=-1.5,opening_speed=.03,margin=.05,
            panel_anchor=np.array([.65,.05,.75]),panel_normal=np.array([1.,0.,0.]),
            limit_posture_drift=True,allow_posture_recovery=False,
            arm_constraints=(np.zeros((1,3)),np.ones(1)),
            scene_constraints=(np.zeros((1,2)),np.ones(1)))
        return node,args,kwargs

    def test_stopped_prediction_cannot_execute_pre_search_barriers(self):
        node,args,kwargs=self.lookahead_node()
        plan=ContactPlan((0.,.04),.1,.05,.25,10)
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.plan_contact_motion',
                   return_value=plan), patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command') as qp:
            with self.assertRaises(SceneObservationPending):
                node._feedback_contact_velocity(*args,**kwargs)
            qp.assert_not_called()
            node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_prediction_is_only_nominal_and_retains_reacquired_guards(self):
        node,args,kwargs=self.lookahead_node()
        plan=ContactPlan((0.,.04),.1,.05,.25,10)
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.plan_contact_motion',
                   return_value=plan), patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   return_value=(0.,.02,None,.05)) as qp:
            with self.assertRaises(SceneObservationPending):
                node._feedback_contact_velocity(*args,**kwargs)
            fresh=dict(kwargs,arm_constraints=(np.ones((1,3)),np.zeros(1)),
                scene_constraints=(np.ones((1,2)),np.zeros(1)))
            node._feedback_contact_velocity(*args,**fresh)
            self.assertIs(qp.call_args.kwargs['arm_constraints'],fresh['arm_constraints'])
            self.assertIs(qp.call_args.kwargs['scene_constraints'],fresh['scene_constraints'])
            self.assertEqual(qp.call_args.kwargs['nominal_velocity'],plan.velocity)
            self.assertFalse(qp.call_args.kwargs['limit_posture_drift'])

    def test_failed_prediction_cannot_remove_soft_or_physical_constraints(self):
        node,args,kwargs=self.lookahead_node()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.plan_contact_motion',
                   side_effect=ValueError('No observed path')) as prediction, patch(
                   'fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   return_value=(0.,0.,None,.05)) as qp:
            with self.assertRaises(SceneObservationPending):
                node._feedback_contact_velocity(*args,**kwargs)
            node._feedback_contact_velocity(*args,**kwargs)
            self.assertEqual(prediction.call_count,1)
            self.assertEqual(qp.call_args.kwargs,kwargs)

    def test_preparation_can_recover_without_dropping_physical_barriers(self):
        node=self.node()
        failure=ValueError('No panel-clear contact motion within calibrated velocity bounds')
        safe=(-.003,-.002,None,.08)
        kwargs=dict(measured_heading_reference=True,heading_tolerance_rad=None,
            limit_posture_drift=True,allow_posture_recovery=True,
            arm_constraints=(np.zeros((1,3)),np.ones(1)),
            scene_constraints=(np.zeros((1,2)),np.ones(1)),margin=.08)
        with patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   side_effect=[failure,safe]) as planner:
            self.assertEqual(node._feedback_contact_velocity(**kwargs),safe)
        expected=dict(kwargs,limit_posture_drift=False)
        for key,value in expected.items():
            if key in ('arm_constraints','scene_constraints'):
                self.assertIs(planner.call_args.kwargs[key],value)
            else:
                self.assertEqual(planner.call_args.kwargs[key],value)
        self.assertEqual(planner.call_count,2)

    def test_attained_stance_or_missing_physical_barriers_cannot_use_fallback(self):
        for changes in (dict(measured_heading_reference=False),dict(heading_tolerance_rad=.15),
                        dict(arm_constraints=None),dict(scene_constraints=None),
                        dict(allow_posture_recovery=False)):
            node=self.node()
            kwargs=dict(measured_heading_reference=True,heading_tolerance_rad=None,
                limit_posture_drift=True,allow_posture_recovery=True,
                arm_constraints=(np.zeros((1,3)),np.ones(1)),
                scene_constraints=(np.zeros((1,2)),np.ones(1)))
            kwargs.update(changes)
            with patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                       side_effect=ValueError('No panel-clear contact motion within calibrated velocity bounds')) as planner:
                with self.assertRaises(SceneObservationPending):
                    node._feedback_contact_velocity(**kwargs)
                self.assertEqual(planner.call_count,1)

    def node(self):
        node = RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._last_scene_scan_stamp = 1.
        node._last_scanned_body_gap = .16
        node._sim_time_sec = Mock(return_value=1.)
        node._publish_sim_cmd_vel = Mock()
        node._feedback_event = Mock()
        return node

    def step(self, node, stamp):
        node._last_scene_scan_stamp = stamp
        node._sim_time_sec.return_value = stamp+.01
        return node._feedback_contact_velocity()

    def test_one_infeasible_scan_stops_and_needs_three_fresh_clear_plans(self):
        node = self.node()
        failure = ValueError('No panel-clear contact motion within calibrated velocity bounds')
        safe = (.01, .02, None, .16)
        with patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   side_effect=[failure, safe, safe, safe]):
            with self.assertRaises(SceneObservationPending):
                self.step(node, 1.)
            for stamp in (1.1, 1.2):
                with self.assertRaises(SceneObservationPending):
                    self.step(node, stamp)
            self.assertEqual(self.step(node, 1.3), safe)
        self.assertEqual(node._publish_sim_cmd_vel.call_count, 4)

    def test_persistent_infeasibility_still_fails(self):
        node = self.node()
        with patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   side_effect=ValueError('No panel-clear contact motion within calibrated velocity bounds')):
            for stamp in (1., 1.1,1.2):
                with self.assertRaises(SceneObservationPending):
                    self.step(node, stamp)
            with self.assertRaisesRegex(ValueError, 'remains ambiguous'):
                self.step(node, 3.1)

    def test_invalid_geometry_is_not_reclassified_as_noise(self):
        node = self.node()
        with patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   side_effect=ValueError('Invalid independently measured panel barrier')):
            with self.assertRaisesRegex(ValueError, 'Invalid independently measured'):
                self.step(node, 1.)

    def test_stale_safe_plan_cannot_start_motion(self):
        node = self.node()
        node._sim_time_sec.return_value = 1.4
        with patch('fire_robot_manipulation.preserved_contact_policy.contact_posture_command',
                   return_value=(.01,.02,None,.16)):
            with self.assertRaises(SceneObservationPending):
                node._feedback_contact_velocity()
            node._publish_sim_cmd_vel.assert_called_with(0.,0.)
            node._sim_time_sec.return_value = 3.5
            with self.assertRaisesRegex(ValueError,'no fresh scan'):
                node._feedback_contact_velocity()


if __name__ == '__main__':
    unittest.main()
