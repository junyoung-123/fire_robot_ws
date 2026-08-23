#!/usr/bin/env python3
"""Capture /door_detection/debug when a blue door handle is observed."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from fire_robot_interfaces.msg import DoorInfo
from rclpy.node import Node
from sensor_msgs.msg import Image


class HandleDebugCapture(Node):
    def __init__(self, output_dir: Path, timeout_sec: float):
        super().__init__('handle_debug_capture')
        self.output_dir = output_dir
        self.timeout_time = time.monotonic() + timeout_sec
        self.done = False
        self.latest_image = None
        self.result = {
            'captured': False,
            'reason': 'timeout',
        }
        self.create_subscription(Image, '/door_detection/debug',
                                 self._image_callback, 10)
        self.create_subscription(DoorInfo, '/detected_door',
                                 self._door_callback, 10)

    def _image_callback(self, msg: Image):
        if msg.encoding not in ('bgr8', 'rgb8'):
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.latest_image = image.copy()

    def _door_callback(self, msg: DoorInfo):
        if msg.door_color != 'blue' or not msg.handle_detected:
            return

        image_path = self.output_dir / 'blue_handle_debug.png'
        image_written = False
        if self.latest_image is not None:
            image_written = cv2.imwrite(str(image_path), self.latest_image)

        self.result = {
            'captured': True,
            'door_id': msg.door_id,
            'door_color': msg.door_color,
            'handle_detected': bool(msg.handle_detected),
            'handle_detection_method': msg.handle_detection_method,
            'handle_confidence': float(msg.handle_confidence),
            'handle_frame': msg.handle_position.header.frame_id,
            'handle_x': float(msg.handle_position.point.x),
            'handle_y': float(msg.handle_position.point.y),
            'handle_z': float(msg.handle_position.point.z),
            'door_x': float(msg.door_pose.pose.position.x),
            'door_y': float(msg.door_pose.pose.position.y),
            'debug_image': str(image_path) if image_written else '',
        }
        self._finish()

    def _finish(self):
        if self.done:
            return
        self.done = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.output_dir / 'handle_detection_summary.json'
        summary_path.write_text(
            json.dumps(self.result, indent=2), encoding='utf-8')
        print(json.dumps(self.result, indent=2), flush=True)
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--timeout-sec', type=float, default=30.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    rclpy.init()
    node = HandleDebugCapture(output_dir, args.timeout_sec)
    while rclpy.ok() and not node.done and time.monotonic() < node.timeout_time:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.done:
        node._finish()


if __name__ == '__main__':
    main()
