#!/usr/bin/env python3
"""Capture Gazebo-rendered before/after frames for the door opening demo."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from fire_robot_interfaces.srv import OpenDoor


class ManipulationProofCapture(Node):
    def __init__(self, image_topic: str, joint_topic: str):
        super().__init__('manipulation_proof_capture')
        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_hinge = None
        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.create_subscription(JointState, joint_topic, self._joint_cb, 10)
        self.client = self.create_client(OpenDoor, '/open_door')

    def _image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as exc:
            self.get_logger().warn(f'image decode failed: {exc}')

    def _joint_cb(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if 'door_hinge' in name:
                self.latest_hinge = float(position)
                return
        if msg.position:
            self.latest_hinge = float(msg.position[0])

    def wait_for_image(self, timeout_sec: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_image is not None:
                return self.latest_image.copy()
        raise TimeoutError('proof camera image was not received')

    def wait_for_service(self, timeout_sec: float):
        if not self.client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError('/open_door service was not available')

    def call_open_door(self, timeout_sec: float) -> bool:
        req = OpenDoor.Request()
        req.door_id = 'manipulation_demo'
        req.handle_position.header.frame_id = 'base_link'
        req.handle_position.point.x = 0.60
        req.handle_position.point.y = 0.00
        req.handle_position.point.z = 0.80
        req.handle_detected = False
        req.handle_detection_method = 'manual_demo'
        req.handle_confidence = 0.0
        future = self.client.call_async(req)
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if future.done():
                result = future.result()
                return bool(result and result.success)
        raise TimeoutError('/open_door service call timed out')

    def call_open_door_with_video(
            self,
            timeout_sec: float,
            video_output: Path,
            fps: float,
            min_open_angle_rad: float,
            post_open_sec: float) -> tuple[bool, float]:
        req = OpenDoor.Request()
        req.door_id = 'manipulation_demo'
        req.handle_position.header.frame_id = 'base_link'
        req.handle_position.point.x = 0.60
        req.handle_position.point.y = 0.00
        req.handle_position.point.z = 0.80
        req.handle_detected = False
        req.handle_detection_method = 'manual_demo'
        req.handle_confidence = 0.0
        future = self.client.call_async(req)

        first = self.wait_for_image(3.0)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(video_output),
            cv2.VideoWriter_fourcc(*'mp4v'),
            max(0.5, fps),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f'failed to open video writer: {video_output}')

        interval = 1.0 / max(0.5, fps)
        deadline = time.monotonic() + timeout_sec
        last_frame_at = 0.0
        opened_at = None
        service_success = False
        try:
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=0.03)
                now = time.monotonic()
                if now - last_frame_at >= interval and self.latest_image is not None:
                    frame = self.latest_image.copy()
                    hinge = self.latest_hinge if self.latest_hinge is not None else 0.0
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (12, 19, 32), -1)
                    cv2.putText(
                        frame,
                        f'Gazebo proof camera | door_hinge={hinge:.3f} rad '
                        f'({math.degrees(hinge):.1f} deg)',
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.92,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(frame)
                    last_frame_at = now

                if future.done() and not service_success:
                    result = future.result()
                    service_success = bool(result and result.success)
                hinge = self.latest_hinge if self.latest_hinge is not None else 0.0
                if hinge >= min_open_angle_rad and opened_at is None:
                    opened_at = now
                if opened_at is not None and now - opened_at >= post_open_sec:
                    return service_success, hinge
            hinge = self.latest_hinge if self.latest_hinge is not None else float('nan')
            raise TimeoutError(
                f'video capture timed out before open confirmation '
                f'(service_success={service_success}, hinge={hinge:.3f})')
        finally:
            writer.release()

    def wait_for_hinge(self, min_angle_rad: float, timeout_sec: float) -> float:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_hinge is not None and self.latest_hinge >= min_angle_rad:
                return self.latest_hinge
        if self.latest_hinge is None:
            raise TimeoutError('door_hinge feedback was not received')
        raise TimeoutError(
            f'door_hinge did not reach {min_angle_rad:.2f} rad '
            f'(last={self.latest_hinge:.3f} rad)')


def annotate(image: np.ndarray, title: str, hinge_text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 78), (12, 19, 32), -1)
    cv2.putText(out, title, (24, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, hinge_text, (24, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (180, 220, 255), 2, cv2.LINE_AA)
    return out


def make_collage(before: np.ndarray, after: np.ndarray, output: Path,
                 before_hinge: float, after_hinge: float,
                 service_success: bool):
    h = min(before.shape[0], after.shape[0])
    before = before[:h]
    after = after[:h]
    before = annotate(before, 'Before: closed blue hinged door',
                      f'door_hinge={before_hinge:.3f} rad '
                      f'({math.degrees(before_hinge):.1f} deg)')
    after = annotate(after, 'After: Gazebo door opened by /open_door',
                     f'door_hinge={after_hinge:.3f} rad '
                     f'({math.degrees(after_hinge):.1f} deg), '
                     f'service_success={service_success}')
    gap = np.full((h, 16, 3), 245, dtype=np.uint8)
    collage = np.hstack([before, gap, after])
    footer = np.full((88, collage.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        footer,
        'Gazebo proof camera frames + /door_joint_states feedback. '
        'This validates rendered door rotation, command transport, and service result.',
        (24, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (25, 35, 55),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        'Scope: controlled manipulation demo; full-contact handle friction still needs real robot validation.',
        (24, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (70, 80, 100),
        2,
        cv2.LINE_AA,
    )
    final = np.vstack([collage, footer])
    cv2.imwrite(str(output), final)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-topic', default='/proof/overhead/image')
    parser.add_argument('--joint-topic', default='/door_joint_states')
    parser.add_argument('--min-open-angle-rad', type=float, default=2.00)
    parser.add_argument('--image-timeout-sec', type=float, default=18.0)
    parser.add_argument('--service-timeout-sec', type=float, default=24.0)
    parser.add_argument('--feedback-timeout-sec', type=float, default=10.0)
    parser.add_argument('--video-output', default='')
    parser.add_argument('--video-fps', type=float, default=6.0)
    parser.add_argument('--video-post-open-sec', type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = ManipulationProofCapture(args.image_topic, args.joint_topic)
    try:
        before = node.wait_for_image(args.image_timeout_sec)
        before_hinge = node.latest_hinge if node.latest_hinge is not None else 0.0
        cv2.imwrite(str(output_dir / 'gazebo_proof_before.png'), before)
        node.wait_for_service(12.0)
        if args.video_output:
            video_output = Path(args.video_output)
            video_output.parent.mkdir(parents=True, exist_ok=True)
            service_success, after_hinge = node.call_open_door_with_video(
                args.service_timeout_sec + args.feedback_timeout_sec + 8.0,
                video_output,
                args.video_fps,
                args.min_open_angle_rad,
                args.video_post_open_sec)
        else:
            service_success = node.call_open_door(args.service_timeout_sec)
            after_hinge = node.wait_for_hinge(
                args.min_open_angle_rad, args.feedback_timeout_sec)
        after = node.wait_for_image(args.image_timeout_sec)
        cv2.imwrite(str(output_dir / 'gazebo_proof_after.png'), after)
        make_collage(
            before, after, output_dir / 'gazebo_door_open_before_after.png',
            before_hinge, after_hinge, service_success)
        (output_dir / 'summary.txt').write_text(
            f'service_success={service_success}\n'
            f'before_hinge_rad={before_hinge:.6f}\n'
            f'before_hinge_deg={math.degrees(before_hinge):.3f}\n'
            f'after_hinge_rad={after_hinge:.6f}\n'
            f'after_hinge_deg={math.degrees(after_hinge):.3f}\n'
            f'image_topic={args.image_topic}\n',
            encoding='utf-8')
        print(output_dir / 'gazebo_door_open_before_after.png')
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
