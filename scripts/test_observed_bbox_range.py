#!/usr/bin/env python3
import unittest
from types import SimpleNamespace
from fire_robot_perception.door_detection_node import DoorDetectionNode


class ClippedRangeTest(unittest.TestCase):
    def test_clipped_box_keeps_scan_instead_of_false_height_range(self):
        node = SimpleNamespace(
            _visual_range_requires_complete_bbox=True,
            _range_distance_at_pixel=lambda *args: 0.82,
            _visual_distance_from_bbox=lambda *args: 1.95)
        source = SimpleNamespace(img_h=240, yaw_offset=1.2)
        for center, height in [(120, 240), (50, 100), (190, 100)]:
            result = DoorDetectionNode._door_distance_at_pixel(node, 160, center, height, source)
            self.assertEqual(result, .82)

    def test_clipped_without_scan_does_not_invent_range(self):
        node = SimpleNamespace(
            _visual_range_requires_complete_bbox=True,
            _range_distance_at_pixel=lambda *args: None,
            _visual_distance_from_bbox=lambda *args: 1.95)
        source = SimpleNamespace(img_h=240, yaw_offset=1.2)
        self.assertIsNone(DoorDetectionNode._door_distance_at_pixel(node, 160, 120, 240, source))


if __name__ == '__main__':
    unittest.main()
