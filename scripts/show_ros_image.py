#!/usr/bin/env python3
"""Small OpenCV ROS image viewer for recording layouts."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--window-name", default="Front Camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


class ImageWindow(Node):
    def __init__(self, topic: str, window_name: str, width: int, height: int):
        super().__init__("front_camera_window")
        self._bridge = CvBridge()
        self._window_name = window_name
        self._width = max(160, width)
        self._height = max(120, height)
        self._last_frame: np.ndarray | None = None
        self._last_stamp = 0.0

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self._width, self._height)

        self.create_subscription(Image, topic, self._image_callback, 10)
        self.create_timer(1.0 / 20.0, self._draw)
        self.get_logger().info(f"Showing {topic} in window '{window_name}'")

    def _image_callback(self, msg: Image) -> None:
        try:
            self._last_frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._last_stamp = time.monotonic()
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert image: {exc}")

    def _draw(self) -> None:
        if self._last_frame is None:
            frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Waiting for camera image...",
                (24, self._height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
        else:
            frame = self._last_frame
            age = time.monotonic() - self._last_stamp
            if age > 1.0:
                cv2.putText(
                    frame,
                    f"stale {age:.1f}s",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = ImageWindow(args.topic, args.window_name, args.width, args.height)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
