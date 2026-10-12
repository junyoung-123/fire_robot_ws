import math
import unittest
import numpy as np
from fire_robot_manipulation.contact_scan_registration import (
    StaticScanRegistration, CorrectionFilter, transform_cloud, compose_pose, inverse_pose, exclude_panel,
    exclude_observed_panel)


class StaticScanTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(-2., 3., 250)
        y = np.linspace(-2., 2., 200)
        self.points = np.vstack((np.column_stack((x, np.full_like(x, -2.))),
                                np.column_stack((x, np.full_like(x, 2.))),
                                np.column_stack((np.full_like(y, 3.), y))))

    def test_motion_is_recovered_from_measured_static_geometry(self):
        target = np.array([.035, -.021, .012])
        cloud = transform_cloud(self.points, inverse_pose(target))
        fit = StaticScanRegistration(self.points).estimate(cloud, [0., 0., 0.])
        np.testing.assert_allclose(fit.pose, target, atol=2.e-4)

    def test_noise_and_unmatched_returns_do_not_become_pose_truth(self):
        rng = np.random.default_rng(817)
        target = np.array([.04, -.025, -.016])
        reference = self.points+rng.normal(0., .008, self.points.shape)
        cloud = transform_cloud(self.points, inverse_pose(target))+rng.normal(0., .02, self.points.shape)
        cloud = np.vstack((cloud, rng.uniform(-1., 1., (80, 2))))
        fit = StaticScanRegistration(reference).estimate(cloud, [0., 0., 0.])
        self.assertLess(np.linalg.norm(fit.pose[:2]-target[:2]), .015)
        self.assertLess(abs(fit.pose[2]-target[2]), math.radians(.3))

    def test_parallel_corridor_without_longitudinal_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unobservable'):
            StaticScanRegistration(self.points[:500]).estimate(self.points[:500], [0., 0., 0.])

    def test_too_few_or_nonfinite_points_are_rejected(self):
        for points in (np.zeros((12, 2)), np.full((100, 2), np.nan)):
            with self.assertRaises(ValueError):
                StaticScanRegistration(points)

    def test_new_child_frame_preserves_sensor_mount_offset(self):
        wheel = np.array([4., -1., .4])
        correction = np.array([.02, -.01, .015])
        laser = np.array([.25, .0, .0])
        estimated_laser = compose_pose(correction, compose_pose(wheel, laser))
        recovered = compose_pose(estimated_laser, inverse_pose(compose_pose(wheel, laser)))
        np.testing.assert_allclose(recovered, correction, atol=1.e-10)
        child = inverse_pose(recovered)
        corrected_base = compose_pose(inverse_pose(child), wheel)
        np.testing.assert_allclose(corrected_base, compose_pose(correction, wheel), atol=1.e-10)

    def test_only_observed_panel_strip_is_excluded(self):
        points = np.array([[1., 0.], [1., .5], [1.5, .5], [-1., 0.]])
        filtered = exclude_panel(points, [[1., -.1], [1., .6]])
        np.testing.assert_equal(filtered, points[2:])

    def short_panel(self,**changes):
        panel=dict(valid=True,segment_xy=[[1.,0.],[1.,.05]],center=[1.,0.,.8],
                   normal=[1.,0.,0.],orientation_uncertainty_rad=.01,residual_m=.003)
        panel.update(changes)
        return panel

    def test_occluded_qualified_panel_excludes_returns_without_inventing_support(self):
        points=np.array([[1.,0.],[1.,.8],[1.5,.8],[-1.,0.]])
        np.testing.assert_equal(exclude_observed_panel(points,self.short_panel()),points[2:])

    def test_short_bad_plane_cannot_mask_localization_returns(self):
        for changes in (dict(valid=False),dict(normal=[0.,0.,1.]),dict(residual_m=.031),
                        dict(center=None),dict(orientation_uncertainty_rad=math.radians(2.1))):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                exclude_observed_panel(self.points,self.short_panel(**changes))

    def test_full_segment_keeps_existing_mask_and_static_observability_guard(self):
        panel=self.short_panel(segment_xy=[[1.,-.1],[1.,.6]])
        np.testing.assert_equal(exclude_observed_panel(self.points,panel),
                               exclude_panel(self.points,panel['segment_xy']))
        with self.assertRaises(ValueError):
            StaticScanRegistration(exclude_observed_panel(np.array([[1.,0.]]*100),self.short_panel()))

    def test_scan_noise_is_not_injected_as_high_frequency_robot_motion(self):
        tracker = CorrectionFilter([0., 0., 0.])
        tracker.update([0., 0., 0.], 1.)
        pose = tracker.update([.004, 0., 0.], 1.+1./12.)
        self.assertLess(pose[0], .0002)
        with self.assertRaises(ValueError):
            tracker.update([0., 0., 0.], 1.)

    def test_angle_wrap_is_continuous(self):
        tracker = CorrectionFilter([0., 0., math.pi-.01])
        tracker.update([0., 0., math.pi-.01], 0.)
        pose = tracker.update([0., 0., -math.pi+.01], 1.)
        self.assertLess(abs(abs(pose[2])-math.pi), .011)


if __name__ == '__main__':
    unittest.main()
