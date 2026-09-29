import math
import unittest
from fire_robot_fsm.observed_exit_aperture import find_aperture


class ApertureTests(unittest.TestCase):
    def scan(self, width=1.2):
        angles = [-.6+i*.002 for i in range(601)]
        return [float('inf') if abs(3*math.tan(a)) < width/2 else 3/math.cos(a)
                for a in angles]

    def find(self, ranges, anchor=(3., .6)):
        return find_aperture(ranges, -.6, .002, .15, 12., anchor)

    def test_measured_gap_center_not_green_jamb(self):
        result = self.find(self.scan())
        self.assertIsNotNone(result)
        x, y, yaw, width = result
        self.assertAlmostEqual(x, 3., delta=.02)
        self.assertAlmostEqual(y, 0., delta=.02)
        self.assertAlmostEqual(yaw, 0., delta=.02)
        self.assertAlmostEqual(width, 1.2, delta=.04)

    def test_closed_and_invalid_scan_not_free(self):
        self.assertIsNone(self.find([3.]*601))
        for missing in (float('nan'), -float('inf')):
            self.assertIsNone(self.find([missing if math.isinf(r) else r for r in self.scan()]))

    def test_unbounded_narrow_and_unrelated_gaps_rejected(self):
        self.assertIsNone(self.find([float('inf')]*601))
        self.assertIsNone(self.find(self.scan(.4)))
        self.assertIsNone(self.find(self.scan(), anchor=(8., 4.)))

    def test_rotated_sensor(self):
        result = find_aperture(self.scan(), -.6+.4, .002, .15, 12.,
                               (3*math.cos(.4), 3*math.sin(.4)))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[2], .4, delta=.02)


if __name__ == '__main__':
    unittest.main()
