#!/usr/bin/env python3
"""The manipulation depth path cannot fall back to a wall or height guess."""
import time
import unittest
from types import SimpleNamespace as NS
import numpy as np
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from fire_robot_perception.door_detection_node import DoorDetectionNode


class RegisteredDepthTests(unittest.TestCase):
    def setUp(self):
        self.node = object.__new__(DoorDetectionNode)
        self.node.get_logger = lambda: NS(info=lambda *a, **k: None)
        self.node._frame = 'base_link'
        self.header = Header(frame_id='camera_color_optical_frame')
        self.header.stamp.sec = 10
        self.source = NS(name='front', has_info=True, img_h=20, img_w=20,
                         cx=10., cy=10., fx=10., fy=10.)
        self.obs = NS(x=12, y=11)
        self.node._handle_depth_samples = [
            (np.ones((20, 20)), self.header, time.monotonic())]
        def transform(point, frame, **kwargs):
            result = PointStamped()
            result.point.x = .25 + point.point.z
            result.point.y = .02 - point.point.x
            result.point.z = .8 - point.point.y
            return result
        self.node._tf_buffer = NS(transform=transform)

    def result(self):
        return self.node._registered_handle_point(self.obs, self.source, self.header)

    def test_camera_extrinsic_is_applied_to_all_three_coordinates(self):
        np.testing.assert_allclose(self.result(), [1.25, -.18, .7])

    def test_no_camera_info_no_coordinate(self):
        self.source.has_info = False
        self.assertIsNone(self.result())

    def test_missing_or_stale_depth_no_coordinate(self):
        self.node._handle_depth_samples = []
        self.assertIsNone(self.result())
        self.node._handle_depth_samples = [(np.ones((20, 20)), self.header, 0.)]
        self.assertIsNone(self.result())

    def test_wrong_timestamp_cannot_use_latest_depth(self):
        other = Header(frame_id=self.header.frame_id)
        other.stamp.sec = 9
        self.node._handle_depth_samples = [(np.ones((20, 20)), other, time.monotonic())]
        self.assertIsNone(self.result())

    def test_missing_tf_does_not_use_raw_pixel_coordinate(self):
        def fail(*a, **k):
            raise RuntimeError('TF unavailable')
        self.node._tf_buffer.transform = fail
        self.assertIsNone(self.result())

    def test_invalid_depth_and_depth_edges_rejected(self):
        depth = self.node._handle_depth_samples[0][0]
        depth[:] = float('nan')
        self.assertIsNone(self.result())
        depth[:] = 1.
        depth[:, 12:] = 1.5
        self.assertIsNone(self.result())


if __name__ == '__main__':
    unittest.main()
