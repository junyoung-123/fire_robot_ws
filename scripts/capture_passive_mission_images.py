#!/usr/bin/env python3
"""Save unmodified ROS camera frames at FSM changes; publish no commands."""
import argparse
import json
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from fire_robot_interfaces.msg import RobotState


CAMERAS = {
    'front': '/camera/color/image_raw',
    'front_left': '/camera/front_left/image_raw',
    'front_right': '/camera/front_right/image_raw',
}


class PassiveCapture(Node):
    def __init__(self, output):
        super().__init__('passive_mission_images')
        self.output = output
        self.bridge = CvBridge()
        self.frames = {}
        self.records = []
        self.last_key = None
        self.pending = None
        self.complete_at = None
        for name, topic in CAMERAS.items():
            self.create_subscription(Image, topic,
                                     lambda msg, name=name: self.image(msg, name),
                                     qos_profile_sensor_data)
        self.create_subscription(RobotState, '/robot_state', self.state, 10)
        self.create_timer(.2, self.capture)

    def image(self, msg, name):
        self.frames[name] = (time.monotonic(), msg)

    def state(self, msg):
        key = (int(msg.state), msg.target_door_id)
        if key != self.last_key:
            self.pending = {'state': key[0], 'target_id': key[1],
                            'description': msg.state_description,
                            'received_monotonic': time.monotonic()}
            self.last_key = key
            self.capture()
        if msg.state == RobotState.MISSION_COMPLETE and self.complete_at is None:
            self.complete_at = time.monotonic()

    def capture(self):
        if self.pending is None:
            return
        now = time.monotonic()
        fresh = {name: pair for name, pair in self.frames.items() if now-pair[0] <= 2.}
        if len(fresh) < 3 and now-self.pending['received_monotonic'] < 5.:
            return
        record = {**self.pending, 'frames': {}}
        for name, (received, msg) in fresh.items():
            filename = f'{len(self.records):03d}_state{record["state"]}_{name}.png'
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if not cv2.imwrite(str(self.output / filename), raw):
                raise RuntimeError('Cannot save camera evidence: ' + filename)
            record['frames'][name] = {
                'file': filename, 'topic': CAMERAS[name],
                'image_stamp_sec': msg.header.stamp.sec + msg.header.stamp.nanosec * 1.e-9,
                'frame_id': msg.header.frame_id, 'receipt_age_sec': now-received,
                'width': msg.width, 'height': msg.height,
            }
        self.records.append(record)
        self.pending = None
        (self.output / 'frames.json').write_text(json.dumps({
            'scope': 'Unmodified ROS camera frames; state-associated, not synchronized across cameras.',
            'publishes_commands': False, 'records': self.records,
        }, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=2200.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = PassiveCapture(args.output)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
            if node.complete_at is not None and time.monotonic()-node.complete_at >= 3.:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.capture()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
