import math
import unittest

import numpy as np

from fire_robot_manipulation.contact_feedback import (
    contact_posture_command, observed_panel_barrier_geometry, qualified_panel_support)
from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE


class IndependentPanelBarrierTests(unittest.TestCase):
    def support(self,stamp,**changes):
        sample=dict(stamp=stamp,center=[.4,0.,.8],normal=[1.,0.,0.],
                    uncertainty=.02,residual=.02)
        sample.update(changes)
        return sample

    def test_bad_latest_exposure_cannot_use_older_valid_support(self):
        samples=[self.support(t) for t in (1.,1.1,1.2)]
        self.assertEqual(len(qualified_panel_support(samples)),3)
        self.assertEqual(qualified_panel_support(samples+[self.support(1.3,residual=.03001)]),[])

    def test_reobservation_requires_a_new_qualified_suffix(self):
        samples=[self.support(1.),self.support(1.1,residual=.031)]
        samples.extend(self.support(t) for t in (1.2,1.3,1.4))
        self.assertEqual([v['stamp'] for v in qualified_panel_support(reversed(samples))],[1.2,1.3,1.4])

    def test_quality_thresholds_and_malformed_exposures_are_not_relaxed(self):
        for changes in (dict(residual=None),dict(uncertainty=None),dict(center=None),
                        dict(normal=None),dict(uncertainty=math.radians(2.001)),
                        dict(residual=float('nan')),dict(normal=[2.,0.,0.])):
            with self.subTest(changes=changes):
                self.assertEqual(qualified_panel_support([self.support(1.),self.support(1.1,**changes)]),[])
        self.assertEqual(len(qualified_panel_support([
            self.support(1.,residual=.03,uncertainty=math.radians(2.))])),1)

    def test_lidar_scatter_requires_a_separate_bounded_plane_offset_measurement(self):
        sample=self.support(1.,residual=.034,range_uncertainty=.015)
        self.assertEqual(len(qualified_panel_support([sample])),1)
        for changes in (dict(range_uncertainty=.03001),dict(range_uncertainty=float('nan')),
                        dict(range_uncertainty='0.01'),dict(residual=.06001),
                        dict(range_uncertainty=None)):
            with self.subTest(changes=changes):
                self.assertEqual(qualified_panel_support([dict(sample,**changes)]),[])

    def fixture(self):
        point = np.array([.788, .141, .738])
        hand = np.array([math.cos(-.760), math.sin(-.760), 0.])
        plane = np.array([math.cos(-.663), math.sin(-.663), 0.])
        anchor = point+.060*plane
        tangent = np.array([-plane[1], plane[0], 0.])
        centers = np.array([anchor-.15*tangent]*3)
        return point, hand, plane, anchor, centers

    def geometry(self, **updates):
        point, hand, plane, _, centers = self.fixture()
        args = dict(point=point, expected_normal=hand, body_outline=PIPER_BODY_OUTLINE,
                    centers=centers, normals=np.array([plane, -plane, plane]),
                    stamps=[1., 1.1, 1.2], uncertainties=[.007]*3,
                    residuals=[.003]*3, now=1.22, calibration_tolerance=.015)
        args.update(updates)
        return observed_panel_barrier_geometry(**args)

    def test_depth_plane_does_not_pass_through_the_protruding_grasp(self):
        point, _, plane, anchor, _ = self.fixture()
        measured, normal, margin = self.geometry()
        np.testing.assert_allclose(measured, anchor)
        np.testing.assert_allclose(normal, plane)
        self.assertAlmostEqual(float((measured-point)@normal), .060)
        self.assertGreater(margin, .03+.015+.009)
        self.assertGreater(float(((measured[:2]-PIPER_BODY_OUTLINE)@normal[:2]).min()), margin)

    def test_range_angle_and_age_errors_increase_the_required_margin(self):
        baseline = self.geometry()[2]
        self.assertGreater(self.geometry(residuals=[.006]*3)[2], baseline)
        self.assertGreater(self.geometry(uncertainties=[.014]*3)[2], baseline)
        self.assertGreater(self.geometry(now=1.28)[2], baseline)

    def test_lidar_plane_quality_does_not_reduce_raw_range_collision_reserve(self):
        margin=self.geometry(residuals=[.034]*3,range_uncertainties=[.015]*3)[2]
        lower_noise=self.geometry(residuals=[.03]*3)[2]
        self.assertGreater(margin,lower_noise)
        self.assertGreaterEqual(margin-lower_noise,3.*(.034-.03)-1.e-12)
        with self.assertRaises(ValueError):
            self.geometry(residuals=[.034]*3,range_uncertainties=[.031]*3)

    def test_stale_future_and_unconfirmed_planes_are_rejected(self):
        for update in (dict(now=1.6), dict(now=1.19), dict(stamps=[1., 1.05, 1.1]),
                       dict(stamps=[1., 1., 1.2])):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.geometry(**update)

    def test_historical_support_may_span_callbacks_but_latest_plane_stays_fresh(self):
        self.geometry(stamps=[.85, 1., 1.2], now=1.22)
        with self.assertRaises(ValueError):
            self.geometry(stamps=[.9, 1., 1.1], now=1.36)

    def test_inconsistent_or_wrong_panel_cannot_supply_clearance(self):
        point, _, plane, _, centers = self.fixture()
        wrong = centers.copy(); wrong[0] += .25*plane
        with self.assertRaisesRegex(ValueError, 'temporally'):
            self.geometry(centers=wrong)
        with self.assertRaisesRegex(ValueError, 'beyond'):
            self.geometry(centers=np.array([point-.08*plane]*3))
        with self.assertRaisesRegex(ValueError, 'articulation'):
            self.geometry(normals=np.array([[1., 0., 0.]]*3))

    def test_old_barrier_is_unchanged_without_an_independent_plane(self):
        point, hand, _, _, _ = self.fixture()
        args = dict(point=point, normal=hand, preferred=[.582, -.282, .738],
                    body_outline=PIPER_BODY_OUTLINE, observed_curvature=-1.655,
                    heading_priority_weight=20., measured_heading_reference=True,
                    opening_speed=.03, allow_posture_recovery=True, margin=.06)
        old = contact_posture_command(**args)
        explicit = contact_posture_command(**args, panel_anchor=point, panel_normal=hand)
        np.testing.assert_allclose(old[:2], explicit[:2])
        self.assertAlmostEqual(old[3], explicit[3])

    def test_generalized_plane_barrier_matches_a_finite_measured_step(self):
        point, hand, _, _, _ = self.fixture()
        anchor, plane, margin = self.geometry()
        curvature = -1.655
        v, w, _, gap = contact_posture_command(
            point, hand, [.582, -.282, .738], PIPER_BODY_OUTLINE,
            margin=margin, observed_curvature=curvature, opening_speed=.03,
            heading_priority_weight=20., measured_heading_reference=True,
            allow_posture_recovery=True, panel_anchor=anchor, panel_normal=plane)
        supplied = hand[0]*v+(hand[1]*point[0]-hand[0]*point[1])*w
        u = float(np.clip(supplied, 0., .03))
        dt = 1.e-4
        angle = curvature*u*dt
        c, s = math.cos(angle), math.sin(angle)
        panel_rotation = np.array([[c, -s], [s, c]])
        new_plane = panel_rotation@plane[:2]
        offset = float((anchor-point)@plane)
        new_anchor = point[:2]+u*hand[:2]*dt+offset*new_plane
        c, s = math.cos(w*dt), math.sin(w*dt)
        inverse = np.array([[c, s], [-s, c]])
        new_anchor = inverse@(new_anchor-[v*dt, 0.])
        new_plane = inverse@new_plane
        before = (anchor[:2]-PIPER_BODY_OUTLINE)@plane[:2]
        after = (new_anchor-PIPER_BODY_OUTLINE)@new_plane
        self.assertGreaterEqual(float(((after-before)/dt+.25*(before-margin)).min()), -1.e-5)
        self.assertAlmostEqual(gap, float(before.min()))

    def test_an_anchor_without_its_normal_is_not_silently_used(self):
        point, hand, _, anchor, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'together'):
            contact_posture_command(point, hand, point, PIPER_BODY_OUTLINE, panel_anchor=anchor)


if __name__ == '__main__':
    unittest.main()
