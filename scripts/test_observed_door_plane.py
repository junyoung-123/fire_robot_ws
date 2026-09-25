import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.observed_door_plane import fit_panel, PlaneRotation, rotation_matrix


class TestObservedDoorPlane(unittest.TestCase):
    def test_planes_with_noise_and_occluders(self):
        rng = np.random.default_rng(7)
        for degrees in (0, -45, -90, -118):
            angle = math.radians(degrees)
            normal = np.array([math.cos(angle), math.sin(angle), 0])
            tangent = np.array([-normal[1], normal[0], 0])
            anchor = np.array([.4, -.3, .9])
            points = (anchor+rng.uniform(-.4,.4,(600,1))*tangent+
                      rng.uniform(-.4,.4,(600,1))*[0,0,1]+rng.normal(0,.001,(600,3)))
            points = np.vstack([points,anchor+rng.uniform(-.4,.4,(130,3))])
            result = fit_panel(points, anchor, rng)
            self.assertGreater(abs(result['normal']@normal), .999)

    def test_plane_sign_does_not_limit_angle_to_90_degrees(self):
        tracker = PlaneRotation(baseline_samples=2)
        result = None
        for i, angle in enumerate([0.,0.]+list(np.linspace(0,-2.06,150))):
            result = tracker.update(((-1)**i)*np.array([math.cos(angle),math.sin(angle),0]),i*.05)
        self.assertAlmostEqual(result,-2.06,places=5)

    def test_bad_support_rejected(self):
        with self.assertRaises(ValueError):
            fit_panel(np.zeros((200,3)),np.zeros(3),np.random.default_rng(0))

    def test_narrow_well_sampled_patch_has_bounded_orientation_error(self):
        rng=np.random.default_rng(155)
        points=np.column_stack((rng.normal(0,.0015,800),
            rng.uniform(-.06,.06,800),rng.uniform(-.3,.3,800)))
        result=fit_panel(points,[0,0,0],rng)
        self.assertLess(result['support_std_m'],.045)
        error=math.acos(min(1.,abs(result['normal'][0])))
        self.assertLess(error,result['orientation_uncertainty_rad'])
        self.assertLess(result['orientation_uncertainty_rad'],math.radians(2.))

    def test_noisy_almost_line_does_not_get_a_false_plane(self):
        rng=np.random.default_rng(155)
        points=np.column_stack((rng.normal(0,.002,800),
            rng.uniform(-.003,.003,800),rng.uniform(-.3,.3,800)))
        with self.assertRaises(ValueError): fit_panel(points,[0,0,0],rng)

    def test_prior_requires_current_depth_support(self):
        rng=np.random.default_rng(91)
        points=np.column_stack((np.ones(600)*.3,rng.uniform(-.3,.3,600),rng.uniform(-.3,.3,600)))
        fit=fit_panel(points,[.3,0,0],rng,prior=(np.array([1.,0,0]),np.array([.3,0,0])))
        self.assertGreater(abs(fit['normal'][0]),.999)
        with self.assertRaises(ValueError):
            fit_panel(np.zeros((200,3)),np.zeros(3),rng,
                      prior=(np.array([1.,0,0]),np.array([.3,0,0])))

    def test_large_jump_after_occlusion_is_rejected(self):
        tracker = PlaneRotation(baseline_samples=2)
        tracker.update([1,0,0],0.)
        tracker.update([1,0,0],.1)
        with self.assertRaises(ValueError):
            tracker.update([0,1,0],2.)

    def test_robot_motion_is_not_door_rotation(self):
        rng = np.random.default_rng(23)
        anchor = np.array([.4, -.3, .9])
        normal = np.array([math.cos(.7), math.sin(.7), 0.])
        tangent = np.array([-normal[1], normal[0], 0.])
        world = (anchor+rng.uniform(-.4,.4,(600,1))*tangent+
                 rng.uniform(-.4,.4,(600,1))*[0,0,1])
        tracker = PlaneRotation(baseline_samples=2)
        for i, yaw in enumerate(np.linspace(-1.2, 1.2, 50)):
            rotation = rotation_matrix([0,0,math.sin(yaw/2),math.cos(yaw/2)])
            translation = np.array([i*.01,-i*.004,.2])
            camera = (world-translation)@rotation
            measured_world = camera@rotation.T+translation
            fit = fit_panel(measured_world, anchor, rng)
            angle = tracker.update(fit['normal'],i*.1)
            if angle is not None:
                self.assertAlmostEqual(angle,0.,places=8)

    def test_positive_rotation_beyond_ninety(self):
        tracker = PlaneRotation(baseline_samples=2)
        for i, angle in enumerate([0.,0.]+list(np.linspace(0,2.10,150))):
            observed = ((-1)**i)*np.array([math.cos(angle),math.sin(angle),0.])
            result = tracker.update(observed,i*.05)
        self.assertAlmostEqual(result,2.10,places=6)


if __name__ == '__main__':
    unittest.main()
