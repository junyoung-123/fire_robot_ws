import math
import unittest
import numpy as np
from fire_robot_fsm.observed_wall_normal import fit_wall_normal


class WallNormalTests(unittest.TestCase):
    def scan(self, normal):
        angles = np.linspace(-math.pi, math.pi, 720)
        cosine = np.cos(angles-normal)
        ranges = np.full(len(angles), np.inf)
        ranges[cosine>.2] = 1.1/cosine[cosine>.2]
        rng = np.random.default_rng(19)
        ranges += rng.normal(0, .015, len(angles))
        return ranges, angles[0], angles[1]-angles[0]

    def test_observed_wall_both_sides_and_rotations(self):
        for angle in (-2.2, -1.57, -.3, .0, .4, 1.57, 2.2):
            ranges, minimum, step = self.scan(angle)
            result = fit_wall_normal(ranges, minimum, step, .15, 12.,
                                     (1.1*math.cos(angle), 1.1*math.sin(angle)))
            self.assertIsNotNone(result)
            self.assertAlmostEqual(math.atan2(math.sin(result-angle), math.cos(result-angle)),
                                   0, delta=math.radians(3))

    def test_foreground_outliers_do_not_tilt_wall(self):
        ranges, minimum, step = self.scan(.2)
        ranges[350:361] = .65
        normal = fit_wall_normal(ranges, minimum, step, .15, 12., (1.1, .2))
        self.assertIsNotNone(normal)
        self.assertAlmostEqual(normal, .2, delta=math.radians(3))

    def test_absent_or_unrelated_wall_cannot_confirm_alignment(self):
        ranges, minimum, step = self.scan(0)
        self.assertIsNone(fit_wall_normal(ranges, minimum, step, .15, 12., (4., 4.)))
        self.assertIsNone(fit_wall_normal([float('nan')]*720, minimum, step, .15, 12., (1., 0.)))

    def test_dense_foreground_occlusion_preserves_observed_wall(self):
        ranges, minimum, step = self.scan(0)
        angles = minimum + np.arange(len(ranges))*step
        foreground = (angles > -.65) & (angles < -.05)
        ranges[foreground] = .60 / np.cos(angles[foreground])
        result = fit_wall_normal(ranges, minimum, step, .15, 12., (1.1, 0.))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0., delta=math.radians(3))

    def test_incorrect_visual_depth_requires_reobservation(self):
        ranges, minimum, step = self.scan(0)
        self.assertIsNone(fit_wall_normal(ranges, minimum, step, .15, 12., (.55, 0.)))


if __name__ == '__main__':
    unittest.main()
