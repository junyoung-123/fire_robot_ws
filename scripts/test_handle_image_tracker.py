#!/usr/bin/env python3
import unittest
import cv2
import numpy as np
from fire_robot_perception.handle_image_tracker import HandleImageTracker


class ImageTrackerTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 160, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        self.image[30:60, 45:100] = rng.integers(0, 255, (30, 55, 3), dtype=np.uint8)
        self.tracker = HandleImageTracker()

    def test_cannot_track_without_yolo_seed(self):
        self.assertIsNone(self.tracker.advance(self.image, 1.))

    def test_tracks_image_motion_not_a_cached_box(self):
        self.assertTrue(self.tracker.seed(self.image, (42, 27, 103, 63), .8, 0.))
        shifted = cv2.warpAffine(self.image, np.float32([[1, 0, 3], [0, 1, 2]]), (160, 100))
        box, confidence = self.tracker.advance(shifted, .2)
        np.testing.assert_allclose(box, [45, 29, 106, 65], atol=1)
        self.assertLessEqual(confidence, .8)

    def test_lost_or_expired_track_is_rejected(self):
        self.tracker.seed(self.image, (42, 27, 103, 63), .8, 0.)
        self.assertIsNone(self.tracker.advance(np.zeros_like(self.image), .2))
        self.assertIsNone(self.tracker.advance(self.image, 16.))


if __name__ == '__main__':
    unittest.main()
