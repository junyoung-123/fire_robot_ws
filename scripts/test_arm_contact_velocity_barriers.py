import unittest

import numpy as np

from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_BODY_OUTLINE
from fire_robot_manipulation.contact_feedback import contact_posture_command, chassis_supplied_opening_speed


class ArmVelocityBarrierTests(unittest.TestCase):
    def setUp(self):
        self.kinematics = PiperActualKinematics()
        self.joints = np.array([-.7522675, 2.23194639, -1.60807725, .65629871, -.65415069, .66780722])
        self.normal = np.array([.40583488, -.91394642, 0.])
        self.normal /= np.linalg.norm(self.normal)

    def test_barriers_are_finite_and_use_only_robot_measurements(self):
        rows, bounds, condition = self.kinematics.contact_velocity_barriers(self.joints, self.normal, -.015)
        self.assertEqual(rows.shape, (13, 3))
        self.assertEqual(bounds.shape, (13,))
        self.assertTrue(np.all(np.isfinite(rows)))
        self.assertTrue(np.all(np.isfinite(bounds)))
        self.assertAlmostEqual(condition, self.kinematics.translation_condition_ratio(self.joints))

    def test_loaded_workspace_recovery_obeys_same_panel_and_arm_barriers(self):
        point, _ = self.kinematics.forward(self.joints)
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(self.joints, self.normal, -.015)
        command = contact_posture_command(point, self.normal, [.505, -.335, point[2]],
            PIPER_BODY_OUTLINE, margin=.18, door_yaw_rate=-.015,
            limit_posture_drift=True, allow_posture_recovery=True, arm_constraints=(rows, bounds))
        v, w = command[:2]
        u = chassis_supplied_opening_speed(point, self.normal, v, w, .012)
        self.assertTrue(np.all(rows@np.array([v, w, u]) <= bounds+1.e-8))
        self.assertGreaterEqual(v, -.04)
        self.assertLessEqual(v, .06)
        self.assertLessEqual(abs(w), .08)

    def test_infeasible_arm_barrier_never_becomes_an_unchecked_command(self):
        point, _ = self.kinematics.forward(self.joints)
        with self.assertRaisesRegex(ValueError, 'No panel-clear'):
            contact_posture_command(point, self.normal, point, PIPER_BODY_OUTLINE,
                arm_constraints=(np.zeros((1, 3)), np.array([-1.])))

    def test_observed_rotation_is_coupled_to_this_steps_actual_advance(self):
        rows0, bounds0, _ = self.kinematics.contact_velocity_barriers(self.joints, self.normal, 0.)
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            self.joints, self.normal, 0., contact_curvature=-1.3854873144)
        np.testing.assert_allclose(rows[:, :2], rows0[:, :2])
        np.testing.assert_allclose(bounds, bounds0)
        self.assertGreater(np.linalg.norm(rows[:, 2]-rows0[:, 2]), .01)

    def test_observed_rotation_and_loaded_recovery_satisfy_all_barriers(self):
        point, _ = self.kinematics.forward(self.joints)
        curvature = -1.3854873144
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            self.joints, self.normal, 0., contact_curvature=curvature)
        v, w, _, gap = contact_posture_command(point, self.normal, [.505, -.335, point[2]],
            PIPER_BODY_OUTLINE, margin=.18, observed_curvature=curvature,
            limit_posture_drift=True, allow_posture_recovery=True, arm_constraints=(rows, bounds))
        u = chassis_supplied_opening_speed(point, self.normal, v, w, .012)
        self.assertGreaterEqual(u, 0.)
        self.assertLessEqual(u, .012)
        self.assertTrue(np.all(rows@np.array([v, w, u]) <= bounds+1.e-8))
        tangent = np.array([-self.normal[1], self.normal[0]])
        gaps = (point[:2]-PIPER_BODY_OUTLINE)@self.normal[:2]
        derivatives = (point[:2]-PIPER_BODY_OUTLINE)@tangent
        corners = PIPER_BODY_OUTLINE[:, 1]*self.normal[0]-PIPER_BODY_OUTLINE[:, 0]*self.normal[1]
        gap_rate = u+curvature*u*derivatives-v*self.normal[0]+w*corners
        self.assertTrue(np.all(gap_rate >= -.25*(gaps-.18)-1.e-8))

    def test_exact_recorded_recovery_has_zero_normal_arm_advance(self):
        joints = np.array([-.7522675017481499, 2.2319463913030555, -1.6080772464339004,
                           .6562987071865404, -.6541506925084927, .6678072199997416])
        normal = np.array([.4058348784312551, -.9139464160708156, 0.])
        normal /= np.linalg.norm(normal)
        point, _ = self.kinematics.forward(joints)
        curvature = -1.3854873144306732
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            joints, normal, 0., contact_curvature=curvature)
        v, w, _, _ = contact_posture_command(point, normal,
            [.5046671691528666, -.3351612662707947, .7325024634908124],
            PIPER_BODY_OUTLINE, margin=.180840687312025, observed_curvature=curvature,
            limit_posture_drift=True, allow_posture_recovery=True, arm_constraints=(rows, bounds))
        u = chassis_supplied_opening_speed(point, normal, v, w, .012)
        self.assertAlmostEqual(u, 0.)
        self.assertTrue(np.all(rows@np.array([v, w, u]) <= bounds+1.e-8))

    def test_invalid_measurements_and_constraint_shapes_are_rejected(self):
        for joints, normal, rate in ((np.full(6, np.nan), self.normal, 0.),
                                     (self.joints, [2., 0., 0.], 0.),
                                     (self.joints, self.normal, float('nan'))):
            with self.assertRaises(ValueError):
                self.kinematics.contact_velocity_barriers(joints, normal, rate)
        point, _ = self.kinematics.forward(self.joints)
        with self.assertRaisesRegex(ValueError, 'arm velocity'):
            contact_posture_command(point, self.normal, point, PIPER_BODY_OUTLINE,
                arm_constraints=(np.zeros((1, 2)), np.array([0.])))

    def test_recorded_r74_posture_goal_is_soft_but_physical_barriers_remain_hard(self):
        joints = [-.7681114636917756, 2.096684907564735, -1.3685122376827885,
                  .4004422365249977, -.6907467181302993, .8970295537473475]
        normal = np.array([np.cos(-1.0645916495427128), np.sin(-1.0645916495427128), 0.])
        curvature = -1.489448208968612
        point, _ = self.kinematics.forward(joints)
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            joints, normal, 0., contact_curvature=curvature)
        v, w, _, _ = contact_posture_command(point, normal,
            [.4707268191216811, -.3351612662707947, .7323585217420128],
            PIPER_BODY_OUTLINE, margin=.16947704761619994, observed_curvature=curvature,
            limit_posture_drift=False, allow_posture_recovery=True, arm_constraints=(rows, bounds))
        u = chassis_supplied_opening_speed(point, normal, v, w, .012)
        self.assertGreater(u, .002)
        self.assertTrue(np.all(rows@np.array([v, w, u]) <= bounds+1.e-8))
        tangent = np.array([-normal[1], normal[0]])
        gaps = (point[:2]-PIPER_BODY_OUTLINE)@normal[:2]
        derivatives = (point[:2]-PIPER_BODY_OUTLINE)@tangent
        corners = PIPER_BODY_OUTLINE[:, 1]*normal[0]-PIPER_BODY_OUTLINE[:, 0]*normal[1]
        self.assertTrue(np.all(u+curvature*u*derivatives-v*normal[0]+w*corners
                              >= -.25*(gaps-.16947704761619994)-1.e-8))

    def test_pre_recovery_r74_pose_cannot_bypass_physical_constraints_to_open(self):
        joints = [-.7442320893129418, 2.0729120798480247, -1.367402060734662,
                  .493774097284926, -.69736848467986, .8275234517808918]
        normal = np.array([np.cos(-1.0645916495427128), np.sin(-1.0645916495427128), 0.])
        point, _ = self.kinematics.forward(joints)
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            joints, normal, 0., contact_curvature=-1.489448208968612)
        v, w, _, _ = contact_posture_command(point, normal,
            [.4707268191216811, -.3351612662707947, .7323585217420128],
            PIPER_BODY_OUTLINE, margin=.16947704761619994,
            observed_curvature=-1.489448208968612, limit_posture_drift=False,
            allow_posture_recovery=True, arm_constraints=(rows, bounds))
        u = chassis_supplied_opening_speed(point, normal, v, w, .012)
        self.assertAlmostEqual(u, 0.)
        self.assertTrue(np.all(rows@np.array([v, w, u]) <= bounds+1.e-8))

    def test_r75_opening_priority_preserves_all_physical_barriers(self):
        joints = [-.5329687508101993, 1.997889710427188, -1.237219439352815,
                  .4435811420295233, -.7360815058818013, .8837803362686204]
        normal = np.array([np.cos(-.8457968682894639), np.sin(-.8457968682894639), 0.])
        point, _ = self.kinematics.forward(joints)
        curvature = -1.5197769872926536
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            joints, normal, 0., contact_curvature=curvature)
        v, w, _, _ = contact_posture_command(point, normal,
            [.4628, -.3351612662707947, .7326763525620835], PIPER_BODY_OUTLINE,
            margin=.16974303164639631, observed_curvature=curvature,
            opening_speed=.03, opening_priority_weight=20., limit_posture_drift=False,
            allow_posture_recovery=True, arm_constraints=(rows, bounds))
        u = chassis_supplied_opening_speed(point, normal, v, w, .03)
        self.assertGreater(u, .02)
        self.assertTrue(np.all(rows@np.array([v, w, u]) <= bounds+1.e-8))
        tangent = np.array([-normal[1], normal[0]])
        gaps = (point[:2]-PIPER_BODY_OUTLINE)@normal[:2]
        derivatives = (point[:2]-PIPER_BODY_OUTLINE)@tangent
        corners = PIPER_BODY_OUTLINE[:, 1]*normal[0]-PIPER_BODY_OUTLINE[:, 0]*normal[1]
        self.assertTrue(np.all(u+curvature*u*derivatives-v*normal[0]+w*corners
                              >= -.25*(gaps-.16974303164639631)-1.e-8))
        self.assertTrue(-.04 <= v <= .06 and abs(w) <= .08)

    def test_opening_priority_cannot_override_an_infeasible_hard_barrier(self):
        point, _ = self.kinematics.forward(self.joints)
        with self.assertRaisesRegex(ValueError, 'No panel-clear'):
            contact_posture_command(point, self.normal, point, PIPER_BODY_OUTLINE,
                opening_speed=.03, opening_priority_weight=20.,
                arm_constraints=(np.zeros((1, 3)), np.array([-1.])))
        for weight in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                contact_posture_command(point, self.normal, point, PIPER_BODY_OUTLINE,
                    opening_priority_weight=weight)

    def test_stance_preparation_has_zero_opening_and_keeps_hard_limits(self):
        point, _ = self.kinematics.forward(self.joints)
        curvature = -1.3854873144
        rows, bounds, _ = self.kinematics.contact_velocity_barriers(
            self.joints, self.normal, 0., contact_curvature=curvature)
        v, w, _, _ = contact_posture_command(point, self.normal, [.505, -.335, point[2]],
            PIPER_BODY_OUTLINE, margin=.18, opening_speed=0., observed_curvature=curvature,
            heading_priority_weight=.05, allow_posture_recovery=True, arm_constraints=(rows,bounds))
        u = chassis_supplied_opening_speed(point, self.normal, v, w, 0.)
        self.assertEqual(u, 0.)
        self.assertTrue(np.all(rows@np.array([v,w,u]) <= bounds+1.e-8))
        self.assertTrue(-.04 <= v <= .06 and abs(w)<=.08)
        for weight in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                contact_posture_command(point,self.normal,point,PIPER_BODY_OUTLINE,
                    heading_priority_weight=weight)



if __name__ == '__main__':
    unittest.main()
