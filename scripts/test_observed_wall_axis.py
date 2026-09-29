#!/usr/bin/env python3
import math
import unittest

import cv2
import numpy as np
from fire_robot_navigation.observed_wall_axis import estimate_wall_axis


class WallAxisTest(unittest.TestCase):
    def test_unknown_has_no_axis(self):
        self.assertIsNone(estimate_wall_axis(np.zeros((80, 140), bool), 0.05))

    def test_observed_walls_under_rotation_and_clutter(self):
        rng = np.random.default_rng(20260928)
        for angle in (-32, -13, 0, 19, 44):
            image = np.zeros((260, 360), np.uint8)
            cv2.line(image, (50, 90), (310, 90), 255, 1)
            cv2.line(image, (80, 170), (280, 170), 255, 1)
            for _ in range(18):
                x, y = rng.integers([70, 105], [285, 145])
                cv2.rectangle(image, (int(x), int(y)), (int(x+6), int(y+6)), 255, 1)
            transform = cv2.getRotationMatrix2D((180, 130), angle, 1)
            image = cv2.warpAffine(image, transform, (360, 260), flags=cv2.INTER_NEAREST)
            yaw = estimate_wall_axis(image > 0, .05)
            error = math.atan2(math.sin(2*(yaw+math.radians(angle))),
                               math.cos(2*(yaw+math.radians(angle))))/2
            self.assertLess(abs(math.degrees(error)), 2.0)


if __name__ == '__main__':
    unittest.main()
