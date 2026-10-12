import math
import unittest
from unittest.mock import patch
import numpy as np
from fire_robot_manipulation.contact_lookahead import (
    plan_contact_motion, contact_prediction_score, continue_contact_plan,ContactPlan)
from fire_robot_manipulation.contact_feedback import contact_posture_command, measured_wrist_rotation
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_BODY_OUTLINE


class ContactPredictionObjectiveTests(unittest.TestCase):
    def test_legacy_without_observed_heading_keeps_exact_ranking(self):
        self.assertEqual(contact_prediction_score(.25,.1,.24),.25-.8*.1-2.*(.28-.24))

    def test_recorded_wrong_way_extension_loses_to_observed_stance_correction(self):
        desired=math.asin(-1.5669611723528853*.5426005447509311)
        extension=contact_prediction_score(.27572906413518933,.38682077808333515,
            .27930513957648617,-.48265539052837997,desired)
        correction=contact_prediction_score(.02581094312662076,.05533984645025294,
            .2402035128829302,-.8872914247777455,desired)
        self.assertGreater(correction,extension)
        self.assertLess(contact_prediction_score(.25,.1,.20,-.8,desired),
                        contact_prediction_score(.25,.1,.28,-.8,desired))

    def test_heading_wrap_and_invalid_inputs(self):
        a=contact_prediction_score(.1,.01,.3,-math.pi+.02,math.pi-.02)
        b=contact_prediction_score(.1,.01,.3,.04,0.)
        self.assertAlmostEqual(a,b)
        for args in ((float('nan'),.1,.3),(.1,-.1,.3),(.1,.1,.3,0.,None),
                     (.1,.1,.3,float('inf'),0.)):
            with self.assertRaises(ValueError): contact_prediction_score(*args)

    def test_same_terminal_state_prefers_less_intermediate_extension(self):
        terminal=dict(opened=.1,hand_error=.04,minimum_condition=.26,
                      relative_heading=-.9,desired_heading=-.9)
        self.assertGreater(contact_prediction_score(**terminal,path_error=.02),
                           contact_prediction_score(**terminal,path_error=.12))
        self.assertEqual(contact_prediction_score(**terminal,path_error=0.),
                         contact_prediction_score(**terminal))
        for value in (-.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                contact_prediction_score(**terminal,path_error=value)


class ContactLookaheadTests(unittest.TestCase):
    def setUp(self):
        self.arm=PiperActualKinematics()
        self.normal=np.array([math.cos(.25),-math.sin(.25),0.])
        rotation=measured_wrist_rotation(self.arm.approach_rotation,-.25,0.,-.32)
        result=self.arm.solve([.59,.04,.735],target_rotation=rotation,
            position_tolerance_m=.001,minimum_joint_margin_rad=.20)
        self.assertIsNotNone(result)
        self.q=result.positions
        self.point=self.arm.forward(self.q)[0]
        self.cloud=np.array([[1.,.5],[1.,-.5],[-1.,.5],[-1.,-.5]])

    def plan(self, **overrides):
        args=dict(kinematics=self.arm,joints=self.q,normal=self.normal,curvature=-1.55,
            scene_points=self.cloud,moving_panel=np.zeros(4,bool),scene_margin=.05,
            panel_anchor=self.point+.035*self.normal,panel_normal=self.normal,
            panel_margin=.05,preferred=[.53,-.2,.735],depth=2,beam_width=3)
        args.update(overrides)
        return plan_contact_motion(**args)

    def test_prediction_is_bounded_and_retains_joint_clearance(self):
        plan=self.plan()
        self.assertGreater(plan.evaluated,0)
        self.assertGreaterEqual(plan.minimum_condition,.20)
        self.assertGreaterEqual(plan.minimum_clearance_m,.03)
        self.assertTrue(-.04<=plan.velocity[0]<=.06)
        self.assertLessEqual(abs(plan.velocity[1]),.08)
        self.assertEqual(len(plan.arcs),2)
        self.assertEqual(plan.arcs[0]['projected_velocity'],plan.projected_velocity)

    def test_next_arc_requires_matching_actual_tool_panel_and_curvature(self):
        arcs=(dict(velocity=(.02,.04),projected_velocity=(.01,.03),
                   tool_xyz=[.53,-.2,.735],normal_xyz=[1.,0.,0.]),
              dict(velocity=(.04,-.04),projected_velocity=(.03,-.02),
                   tool_xyz=[.53,-.2,.735],normal_xyz=[1.,0.,0.]))
        plan=ContactPlan((.02,.04),.1,.05,.25,48,(.01,.03),1.,arcs,0,-1.5)
        next_plan=continue_contact_plan(plan,[.531,-.2,.735],[1.,0.,0.],-1.51,.02)
        self.assertIsNotNone(next_plan)
        self.assertEqual(next_plan.arc_index,1)
        self.assertEqual(next_plan.projected_velocity,(.03,-.02))
        self.assertIsNone(continue_contact_plan(next_plan,[.53,-.2,.735],[1.,0.,0.],-1.5,.02))
        self.assertEqual(plan.arc_index,0)
        for point,normal,curve,sigma in (([.55,-.2,.735],[1.,0.,0.],-1.5,.02),
                ([.53,-.2,.735],[math.cos(.1),math.sin(.1),0.],-1.5,.02),
                ([.53,-.2,.735],[1.,0.,0.],-1.6,.02),
                ([.53,-.2,.735],[1.,0.,0.],-1.5,None)):
            with self.subTest(point=point,normal=normal,curve=curve,sigma=sigma):
                self.assertIsNone(continue_contact_plan(plan,point,normal,curve,sigma))

    def test_observed_heading_objective_keeps_all_physical_feasibility_guards(self):
        plan=self.plan(controller_options={'measured_heading_reference':True})
        self.assertGreaterEqual(plan.minimum_condition,.20)
        self.assertGreaterEqual(plan.minimum_clearance_m,.03)
        with self.assertRaisesRegex(ValueError,'observed articulation reach'):
            self.plan(preferred=[1.,-.2,.735],
                      controller_options={'measured_heading_reference':True})

    def test_hard_collision_cannot_be_a_prediction(self):
        with self.assertRaisesRegex(ValueError,'hard chassis'):
            self.plan(scene_points=np.array([[.32,0.],[1.,1.],[-1.,-1.]]),
                      moving_panel=np.zeros(3,bool))

    def test_unbounded_curvature_or_missing_support_rejected(self):
        for kwargs in ({'curvature':float('nan')},{'scene_margin':.02},
                       {'scene_points':np.array([[1.,1.],[2.,2.]])}):
            with self.assertRaises(ValueError):
                self.plan(**kwargs)

    def test_nominal_prediction_never_bypasses_fresh_physical_barriers(self):
        rows=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
        bounds=np.array([.002,.002,.005,.005])
        v,w,_,_=contact_posture_command(self.point,self.normal,[.53,-.2,.735],
            PIPER_BODY_OUTLINE,margin=.05,nominal_velocity=[.04,.08],
            allow_posture_recovery=True,scene_constraints=(rows,bounds))
        self.assertLessEqual(np.max(rows@np.array([v,w])-bounds),1.e-9)

    def test_generated_plan_is_accepted_by_the_actual_velocity_controller(self):
        plan = self.plan()
        v, w, _, _ = contact_posture_command(
            self.point, self.normal, [.53, -.2, .735], PIPER_BODY_OUTLINE,
            margin=.05, nominal_velocity=plan.velocity, allow_posture_recovery=True)
        self.assertTrue(-.04 <= v <= .06)
        self.assertLessEqual(abs(w), .08)

    def test_first_prediction_matches_actual_arm_and_scene_projection(self):
        from fire_robot_manipulation.contact_feedback import scanned_body_constraints
        plan = self.plan()
        arm_rows, arm_bounds, _ = self.arm.contact_velocity_barriers(
            self.q, self.normal, 0., contact_curvature=-1.55)
        scene_rows, scene_bounds, _ = scanned_body_constraints(
            self.cloud, PIPER_BODY_OUTLINE, margin=.05,
            point_opening_motion=np.zeros_like(self.cloud), motion_uncertainty=np.zeros(4))
        v, w, _, _ = contact_posture_command(
            self.point, self.normal, [.53, -.2, .735], PIPER_BODY_OUTLINE,
            margin=.05, opening_speed=.03, nominal_velocity=plan.velocity,
            observed_curvature=-1.55, panel_anchor=self.point+.035*self.normal,
            panel_normal=self.normal, arm_constraints=(arm_rows, arm_bounds),
            scene_constraints=(scene_rows, scene_bounds),
            limit_posture_drift=False, allow_posture_recovery=True,
            prefer_nominal_recovery=plan.velocity[0]<0.)
        np.testing.assert_allclose(plan.projected_velocity, [v, w], atol=1.e-10)

    def test_blocked_projected_velocity_cannot_predict_nominal_opening(self):
        rows = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        with patch('fire_robot_manipulation.contact_lookahead.scanned_body_constraints',
                   return_value=(rows, np.zeros(4), .5)):
            plan = self.plan()
        np.testing.assert_allclose(plan.projected_velocity, [0., 0.], atol=1.e-10)
        self.assertAlmostEqual(plan.predicted_opening_rad, 0.)

    def test_observation_uncertainty_is_required_to_be_bounded(self):
        for uncertainty in ([0., 0., 0.], [0., 0., 0., -1.], [0., 0., 0., float('nan')]):
            with self.assertRaises(ValueError):
                self.plan(moving_uncertainty=uncertainty)
        with self.assertRaises(ValueError):
            self.plan(controller_options={'simulator_hinge': [0., 0.]})

    def test_invalid_nominal_request_rejected(self):
        with self.assertRaisesRegex(ValueError,'nominal velocity'):
            contact_posture_command(self.point,self.normal,[.53,-.2,.735],
                PIPER_BODY_OUTLINE,nominal_velocity=[.07,0.])

    def test_stance_preparation_can_turn_forward_without_forcing_door_advance(self):
        point=np.array([.53,.14,.735])
        normal=np.array([math.cos(-.3),math.sin(-.3),0.])
        v,w,_,_=contact_posture_command(point,normal,[.53,-.2,.735],PIPER_BODY_OUTLINE,
            margin=.05,opening_speed=.03,observed_curvature=-1.5,
            measured_heading_reference=True,heading_priority_weight=1./1.5**2,
            allow_posture_recovery=True,nominal_velocity=[.02,.08])
        self.assertGreaterEqual(v,0.)
        self.assertGreater(w,0.)
        self.assertLessEqual(float(normal@np.array([v-w*point[1],w*point[0],0.])),.002)

    def test_partial_horizon_is_not_reported_as_feasible(self):
        from fire_robot_manipulation.contact_lookahead import _advance
        calls = 0
        def truncate(*args, **kwargs):
            nonlocal calls
            calls += 1
            return _advance(*args, **kwargs) if calls <= 11 else None
        with patch('fire_robot_manipulation.contact_lookahead._advance', side_effect=truncate):
            with self.assertRaisesRegex(ValueError, 'full lookahead horizon'):
                self.plan()

    def test_temporarily_worse_setup_turn_is_not_pruned_before_feasibility_check(self):
        def model(_arm, state, velocity, *args):
            value = list(state)
            if state[11] is None:
                value[11] = velocity
                value[8] = .1 if velocity[1] < 0. else 1.
            else:
                if state[11][1] >= 0.:
                    return None
                value[8] += .1
            return tuple(value)
        with patch('fire_robot_manipulation.contact_lookahead._advance', side_effect=model):
            plan = self.plan(beam_width=6)
        self.assertLess(plan.velocity[1], 0.)
        self.assertAlmostEqual(plan.predicted_opening_rad, .2)

    def test_seed_only_cannot_jump_to_another_ik_branch(self):
        result=self.arm.solve(self.point,seed=self.q,target_rotation=self.arm.forward(self.q)[1],
            seed_only=True,minimum_joint_margin_rad=.20)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.positions,self.q,atol=1.e-5)
        self.assertIsNone(self.arm.solve(self.point,seed=None,seed_only=True))


if __name__=='__main__':
    unittest.main()
