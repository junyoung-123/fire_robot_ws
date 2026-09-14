#!/usr/bin/env python3
"""Run a Gazebo contact-only door-opening check.

The door is considered opened only when the Gazebo door_hinge joint moves.
The test never publishes to a door hinge command topic; only the simulated arm
and gripper joint position topics are commanded through manipulation_node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import rclpy
import tf2_geometry_msgs
import tf2_ros
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_interfaces.srv import OpenDoor
from fire_robot_manipulation.piper_actual_kinematics import (
    PiperActualKinematics, PIPER_JOINT_LIMITS)
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from PIL import Image as PilImage
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String


@dataclass
class ContactRecord:
    start_wall_time: float = field(default_factory=time.monotonic)
    door_samples: list[tuple[float, float]] = field(default_factory=list)
    lever_samples: list[tuple[float, float]] = field(default_factory=list)
    base_samples: list[tuple[float, float, float]] = field(default_factory=list)
    phases: list[tuple[float, str]] = field(default_factory=list)
    feedback_events: list[dict] = field(default_factory=list)
    latest_image: Image | None = None
    latest_front_image: Image | None = None
    latest_overhead_image: Image | None = None
    latest_detection_image: Image | None = None
    yolo_detection_image: Image | None = None
    capture_next_yolo_debug: bool = False
    latest_yolo_door: DoorInfo | None = None
    latest_yolo_handle_odom: PointStamped | None = None
    latest_odom: Odometry | None = None
    yolo_observations: list[dict[str, object]] = field(default_factory=list)
    latest_phase: str = "WAITING"
    saved_images: list[str] = field(default_factory=list)


class ContactProbe(Node):
    def __init__(self, record: ContactRecord):
        super().__init__("physical_contact_probe")
        self.record = record
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(
            JointState, "/door_joint_states", self._door_cb, 20)
        self.create_subscription(
            String, "/manipulation_phase", self._phase_cb, 20)
        self.create_subscription(
            String, "/manipulation_feedback", self._feedback_cb, 20)
        self.create_subscription(
            Image, "/proof/perspective/image", self._image_cb, 10)
        self.create_subscription(
            Image, '/camera/color/image_raw',
            lambda msg: setattr(self.record, 'latest_front_image', msg), 5)
        self.create_subscription(
            Image, "/proof/overhead/image", self._overhead_image_cb, 5)
        self.create_subscription(
            Image, "/door_detection/debug", self._detection_image_cb, 10)
        self.create_subscription(
            DoorInfo, "/detected_door", self._detected_door_cb, 20)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 20)
        self.client = self.create_client(OpenDoor, "/open_door")
        self.arm_publishers = {
            f"joint{i}": self.create_publisher(
                Float64, f"/fire_robot/sim_arm/joint{i}_cmd", 10)
            for i in range(1, 7)
        }
        self.arm_publishers["gripper_left_joint"] = self.create_publisher(
            Float64, "/fire_robot/sim_arm/gripper_left_joint_cmd", 10)
        self.arm_publishers["gripper_right_joint"] = self.create_publisher(
            Float64, "/fire_robot/sim_arm/gripper_right_joint_cmd", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.manual_cmd_vel_pub = self.create_publisher(
            Twist, "/cmd_vel_manual", 10)
        self._video_path: Path | None = None
        self._video_writer = None
        self._video_fps = 10.0
        self._last_video_frame_at = 0.0
        self._keyframe_dir: Path | None = None
        self._last_keyframe_phase = ""

    def _stamp(self) -> float:
        return time.monotonic() - self.record.start_wall_time

    def _door_cb(self, msg: JointState):
        for name, values in (
                ("door_hinge", self.record.door_samples),
                ("lever_joint", self.record.lever_samples)):
            try:
                index = msg.name.index(name)
                angle = float(msg.position[index])
            except (ValueError, IndexError):
                continue
            values.append((self._stamp(), angle))

    def _phase_cb(self, msg: String):
        self.record.phases.append((self._stamp(), msg.data))
        self.record.latest_phase = msg.data.split(":", 1)[0]

    def _feedback_cb(self, msg: String):
        try:
            event = json.loads(msg.data)
            event['wall_elapsed_sec'] = self._stamp()
            self.record.feedback_events.append(event)
        except (ValueError, TypeError):
            pass

    def _image_cb(self, msg: Image):
        self.record.latest_image = msg
        self._record_video_frame(msg)

    def _overhead_image_cb(self, msg: Image):
        self.record.latest_overhead_image = msg

    def _detection_image_cb(self, msg: Image):
        self.record.latest_detection_image = msg
        if self.record.capture_next_yolo_debug:
            self.record.yolo_detection_image = msg
            self.record.capture_next_yolo_debug = False

    def _detected_door_cb(self, msg: DoorInfo):
        method = str(msg.handle_detection_method or '')
        if (
                msg.door_color != 'blue'
                or not msg.handle_detected
                or not method.startswith(('yolo:primary', 'track:primary_yolo'))):
            return
        self.record.latest_yolo_door = msg
        self.record.capture_next_yolo_debug = True
        fixed_handle = _door_handle_in_odom(msg, self.tf_buffer)
        if fixed_handle is not None:
            self.record.latest_yolo_handle_odom = fixed_handle
        point = msg.handle_position.point
        observation = {
            'time_sec': self._stamp(),
            'door_id': str(msg.door_id),
            'method': method,
            'confidence': float(msg.handle_confidence),
            'frame_id': str(msg.handle_position.header.frame_id),
            'xyz': [float(point.x), float(point.y), float(point.z)],
        }
        if fixed_handle is not None:
            fixed = fixed_handle.point
            observation['odom_xyz'] = [
                float(fixed.x), float(fixed.y), float(fixed.z)]
        self.record.yolo_observations.append(observation)

    def _odom_cb(self, msg: Odometry):
        self.record.latest_odom = msg
        self.record.base_samples.append((
            self._stamp(),
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y)))

    def start_recording(self, path: Path, fps: float, keyframe_dir: Path):
        self._video_path = path
        self._video_fps = max(1.0, float(fps))
        self._keyframe_dir = keyframe_dir
        self._keyframe_dir.mkdir(parents=True, exist_ok=True)

    def stop_recording(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def _record_video_frame(self, msg: Image):
        if self._video_path is None:
            return
        now = time.monotonic()
        if now - self._last_video_frame_at < 1.0 / self._video_fps:
            return
        rgb = _ros_image_to_rgb(msg)
        if rgb is None:
            return
        proof_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame = self._compose_video_frame(proof_frame)
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._video_writer = cv2.VideoWriter(
                str(self._video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._video_fps,
                (width, height))
            if not self._video_writer.isOpened():
                raise RuntimeError(
                    f"Failed to open video writer: {self._video_path}")
        self._video_writer.write(frame)
        self._last_video_frame_at = now

        phase = self.record.latest_phase
        if self._keyframe_dir is not None and phase != self._last_keyframe_phase:
            cv2.imwrite(str(self._keyframe_dir / f"{phase.lower()}.png"), frame)
            self._last_keyframe_phase = phase

    def _compose_video_frame(self, proof_frame: np.ndarray) -> np.ndarray:
        panel_width = 640
        panel_height = 360
        frame = np.full((500, panel_width * 2, 3), 242, dtype=np.uint8)
        left = cv2.resize(proof_frame, (panel_width, panel_height))
        debug_msg = (
            self.record.yolo_detection_image
            or self.record.latest_detection_image)
        debug_rgb = _ros_image_to_rgb(debug_msg)
        if debug_rgb is None:
            right = np.full_like(left, 225)
            cv2.putText(
                right, 'Waiting for /door_detection/debug', (80, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 45, 45), 2,
                cv2.LINE_AA)
        else:
            right = cv2.resize(
                cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR),
                (panel_width, panel_height))
        frame[92:92 + panel_height, :panel_width] = left
        frame[92:92 + panel_height, panel_width:] = right

        door_angle = (
            self.record.door_samples[-1][1]
            if self.record.door_samples else 0.0)
        lever_angle = (
            self.record.lever_samples[-1][1]
            if self.record.lever_samples else 0.0)
        base_distance = 0.0
        if len(self.record.base_samples) >= 2:
            _, x0, y0 = self.record.base_samples[0]
            _, x1, y1 = self.record.base_samples[-1]
            base_distance = math.hypot(x1 - x0, y1 - y0)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (12, 18, 30), -1)
        cv2.putText(
            frame,
            f"Phase: {self.record.latest_phase}",
            (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.86,
            (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"lever={math.degrees(lever_angle):.1f} deg | "
            f"door={math.degrees(door_angle):.1f} deg | "
            f"base displacement={base_distance:.2f} m",
            (22, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
            (120, 220, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 452), (frame.shape[1], 500), (12, 18, 30), -1)
        cv2.putText(
            frame, 'EXTERNAL PROOF CAMERA', (18, 444),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2,
            cv2.LINE_AA)
        cv2.putText(
            frame, 'YOLO DETECTION SNAPSHOT (SAME RUN)',
            (panel_width + 18, 444),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2,
            cv2.LINE_AA)
        handle_text = 'YOLO: waiting for primary detection'
        if self.record.latest_yolo_door is not None:
            door = self.record.latest_yolo_door
            point = door.handle_position.point
            handle_text = (
                f"YOLO: {door.handle_detection_method} "
                f"conf={door.handle_confidence:.2f} | "
                f"handle[{door.handle_position.header.frame_id}]="
                f"({point.x:.2f},{point.y:.2f},{point.z:.2f})")
        cv2.putText(
            frame,
            handle_text,
            (22, 482), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
            (255, 255, 255), 2, cv2.LINE_AA)
        return frame


def _ros_image_to_rgb(msg: Image | None) -> np.ndarray | None:
    if msg is None:
        return None
    encoding = (msg.encoding or "").lower()
    channels = 3
    if encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    data = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if channels == 1:
            image = data.reshape((msg.height, msg.width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = data.reshape((msg.height, msg.width, channels))
    except ValueError:
        return None
    if encoding == "bgr8":
        image = image[..., ::-1]
    elif encoding == "bgra8":
        image = image[..., [2, 1, 0, 3]]
    if channels == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def _save_ros_image(msg: Image | None, path: Path) -> bool:
    image = _ros_image_to_rgb(msg)
    if image is None:
        return False
    PilImage.fromarray(image, mode="RGB").save(path)
    return True


def _door_handle_in_odom(
        door: DoorInfo, tf_buffer: tf2_ros.Buffer) -> PointStamped | None:
    source_frame = str(door.handle_position.header.frame_id or '')
    source = door.handle_position.point
    if source_frame == 'odom':
        fixed = PointStamped()
        fixed.header = door.handle_position.header
        fixed.point.x = float(source.x)
        fixed.point.y = float(source.y)
        fixed.point.z = float(source.z)
        return fixed
    if not source_frame:
        return None
    observed = PointStamped()
    observed.header = door.handle_position.header
    # The observation is retained while the base approaches the door. Use the
    # complete TF (including the base_link height), not a planar x/y shortcut.
    observed.header.stamp = Time().to_msg()
    observed.point = source
    try:
        return tf_buffer.transform(
            observed, 'odom', timeout=Duration(seconds=0.10))
    except Exception:
        return None


def _plot_angles(record: ContactRecord, out_path: Path, min_angle_rad: float):
    if record.door_samples:
        times = [t for t, _ in record.door_samples]
        angles = [a for _, a in record.door_samples]
    else:
        times = [0.0]
        angles = [0.0]

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.plot(times, angles, color="#0b5bd3", linewidth=2.0, label="door_hinge")
    ax.axhline(min_angle_rad, color="#c62525", linestyle="--",
               linewidth=1.3, label=f"pass threshold +/-{min_angle_rad:.2f} rad")
    ax.axhline(-min_angle_rad, color="#c62525", linestyle="--",
               linewidth=1.3)
    ax.set_title("Gazebo physical-contact door hinge angle")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hinge angle (rad)")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _spin_for(node: ContactProbe, seconds: float):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def _publish_arm_target(node: ContactProbe, targets: dict[str, float],
                        seconds: float):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for name, value in targets.items():
            publisher = node.arm_publishers.get(name)
            if publisher is None:
                continue
            msg = Float64()
            msg.data = float(value)
            publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)


def _run_full_open_sweep(node: ContactProbe):
    # Door hinge remains uncommanded. This only moves the simplified PIPER arm
    # around the already-open door to keep applying contact force.
    _publish_arm_target(node, {
        "gripper_left_joint": 0.0,
        "gripper_right_joint": 0.0,
    }, 0.4)
    sweep = [
        ({"joint1": 0.55, "joint2": 1.20, "joint3": -0.18,
          "joint4": 0.0, "joint5": -0.04, "joint6": 0.0}, 0.7),
        ({"joint1": 0.75, "joint2": 1.20, "joint3": -0.20,
          "joint4": 0.0, "joint5": -0.06, "joint6": 0.0}, 0.7),
        ({"joint1": 0.95, "joint2": 1.16, "joint3": -0.18,
          "joint4": 0.0, "joint5": -0.08, "joint6": 0.0}, 0.7),
        ({"joint1": 1.15, "joint2": 1.05, "joint3": -0.10,
          "joint4": 0.0, "joint5": -0.08, "joint6": 0.0}, 0.9),
        ({"joint1": 1.30, "joint2": 0.92, "joint3": -0.02,
          "joint4": 0.0, "joint5": -0.06, "joint6": 0.0}, 1.0),
    ]
    for targets, seconds in sweep:
        _publish_arm_target(node, targets, seconds)


def _publish_cmd_vel(node: ContactProbe, linear_x: float, angular_z: float,
                     seconds: float, manual: bool = False):
    publisher = (
        node.manual_cmd_vel_pub if manual else node.cmd_vel_pub)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    msg = Twist()
    publisher.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)


def _publish_manual_cmd_vel_for_distance(
        node: ContactProbe, distance_m: float,
        linear_x: float, timeout_sec: float) -> float:
    if distance_m <= 0.0 or node.record.latest_odom is None:
        return 0.0
    start = node.record.latest_odom.pose.pose.position
    start_x = float(start.x)
    start_y = float(start.y)
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    travelled = 0.0
    while time.monotonic() < deadline and travelled < distance_m:
        msg = Twist()
        msg.linear.x = float(linear_x)
        node.manual_cmd_vel_pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.record.latest_odom is not None:
            current = node.record.latest_odom.pose.pose.position
            travelled = math.hypot(
                float(current.x) - start_x,
                float(current.y) - start_y)
    stop = Twist()
    node.manual_cmd_vel_pub.publish(stop)
    rclpy.spin_once(node, timeout_sec=0.2)
    return float(travelled)


def _run_mobile_full_open_follow(node: ContactProbe):
    _publish_arm_target(node, {
        "joint1": 0.55, "joint2": 1.20, "joint3": -0.18,
        "joint4": 0.0, "joint5": -0.04, "joint6": 0.0,
        "gripper_left_joint": 0.0, "gripper_right_joint": 0.0,
    }, 0.5)
    _publish_cmd_vel(node, 0.12, -0.22, 3.2)
    _publish_arm_target(node, {
        "joint1": 0.95, "joint2": 1.14, "joint3": -0.16,
        "joint4": 0.0, "joint5": -0.08, "joint6": 0.0,
    }, 0.5)
    _publish_cmd_vel(node, 0.10, -0.34, 3.0)
    _publish_cmd_vel(node, 0.0, 0.0, 0.2)


def _publish_ign_empty(topic: str) -> bool:
    try:
        result = subprocess.run(
            ["ign", "topic", "-t", topic, "-m", "ignition.msgs.Empty", "-p", ""],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0)
    except Exception:
        return False
    return result.returncode == 0


def _wait_for_ready(node: ContactProbe, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        has_door = bool(node.record.door_samples)
        has_service = node.client.service_is_ready()
        if not has_service:
            node.client.wait_for_service(timeout_sec=0.05)
        if has_door and node.client.service_is_ready():
            return True
    return False


def _wait_for_yolo_observation(
        node: ContactProbe, timeout_sec: float,
        minimum_observations: int,
        minimum_confidence: float,
        search_approach_m: float,
        search_speed_mps: float) -> tuple[DoorInfo, PointStamped] | None:
    deadline = time.monotonic() + timeout_sec
    required = max(1, int(minimum_observations))
    start_xy: tuple[float, float] | None = None
    valid: list[dict[str, object]] = []
    seen_count = 0
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            odom = node.record.latest_odom
            if odom is not None and start_xy is None:
                position = odom.pose.pose.position
                start_xy = (float(position.x), float(position.y))

            observations = node.record.yolo_observations
            for observation in observations[seen_count:]:
                seen_count += 1
                xyz = observation.get('xyz', [])
                confidence = float(observation.get('confidence', 0.0))
                if len(xyz) != 3 or confidence < minimum_confidence:
                    continue
                x, y, z = (float(value) for value in xyz)
                if not (0.30 <= x <= 1.20 and abs(y) <= 0.60
                        and 0.45 <= z <= 1.05):
                    continue
                valid.append(observation)
                valid = valid[-required:]

            if len(valid) >= required:
                xs = [float(item['xyz'][0]) for item in valid]
                ys = [float(item['xyz'][1]) for item in valid]
                zs = [float(item['xyz'][2]) for item in valid]
                stable = (
                    max(xs) - min(xs) <= 0.14
                    and max(ys) - min(ys) <= 0.12
                    and max(zs) - min(zs) <= 0.12)
                if (
                        stable
                        and node.record.latest_yolo_door is not None
                        and node.record.latest_yolo_handle_odom is not None):
                    return (
                        node.record.latest_yolo_door,
                        node.record.latest_yolo_handle_odom)

            travelled = 0.0
            if start_xy is not None and odom is not None:
                position = odom.pose.pose.position
                travelled = math.hypot(
                    float(position.x) - start_xy[0],
                    float(position.y) - start_xy[1])
            command = Twist()
            if travelled < max(0.0, search_approach_m):
                command.linear.x = max(0.0, search_speed_mps)
            node.manual_cmd_vel_pub.publish(command)
    finally:
        node.manual_cmd_vel_pub.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.2)
    return None


def _start_launch(log_path: Path, headless: bool,
                  door_open_motion: str,
                  min_angle_rad: float,
                  use_yolo_observation: bool,
                  feedback_contact: bool = False) -> subprocess.Popen:
    headless_arg = "true" if headless else "false"
    perception_arg = "true" if use_yolo_observation else "false"
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError(
            "ros2 was not found. Source ROS2 and this workspace before "
            "running the physical-contact validation.")
    command = [
        ros2, "launch", "fire_robot_bringup",
        "physical_contact_door_test.launch.py",
        f"headless:={headless_arg}",
        "use_rviz:=false",
        f"enable_perception:={perception_arg}",
        f"require_yolo_handle:={perception_arg}",
        f"door_open_motion:={door_open_motion}",
        f"contact_min_angle_rad:={float(min_angle_rad):.6f}",
        f"feedback_contact_enabled:={str(feedback_contact).lower()}",
    ]
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )


def _approach_visible_handle(node, args):
    """Keep the observed handle in view; never reuse pre-motion base XYZ."""
    started = time.monotonic()
    first_base = node.record.latest_odom.pose.pose.position
    start_xy = (first_base.x, first_base.y)
    stable = 0
    used_time = -1.0
    locked_xyz = np.asarray(node.record.yolo_observations[-1]['odom_xyz'])
    kinematics = PiperActualKinematics()
    ik_sample_time = -1.0
    reachable = False
    ik_worker = ThreadPoolExecutor(max_workers=1)
    ik_future = None
    ik_target = None
    try:
        while time.monotonic() - started < 80.0:
            rclpy.spin_once(node, timeout_sec=.05)
            observed = node.record.latest_yolo_door
            samples = node.record.yolo_observations
            command = Twist()
            if not samples or observed is None:
                node.manual_cmd_vel_pub.publish(command)
                continue
            sample = samples[-1]
            tracked_xyz = sample.get('odom_xyz')
            if (node._stamp() - sample['time_sec'] > .8
                    or observed.handle_confidence < args.tracking_min_yolo_confidence
                    or tracked_xyz is None
                    or np.linalg.norm(np.asarray(tracked_xyz) - locked_xyz) > .10):
                node.manual_cmd_vel_pub.publish(command)
                continue
            current = node.record.latest_odom.pose.pose.position
            traveled = math.hypot(current.x-start_xy[0], current.y-start_xy[1])
            if traveled > args.max_observation_approach_m:
                return None, traveled
            # The camera extrinsic comes from TF, not a guessed robot offset.
            camera = node.tf_buffer.lookup_transform(
                'base_link', 'camera_link', Time(), timeout=Duration(seconds=.1))
            point = observed.handle_position.point
            bearing = math.atan2(point.y - kinematics.mount_xyz[1],
                                 point.x - kinematics.mount_xyz[0])
            xyz = np.asarray((point.x, point.y, point.z))
            if ik_future is not None and ik_future.done():
                solution = ik_future.result()
                reachable = (solution is not None
                             and np.linalg.norm(xyz-ik_target) < .04
                             and np.all(solution.positions > PIPER_JOINT_LIMITS[:, 0]+.08)
                             and np.all(solution.positions < PIPER_JOINT_LIMITS[:, 1]-.08))
                ik_future = None
            if ik_future is None and sample['time_sec'] != ik_sample_time:
                ik_sample_time = sample['time_sec']
                ik_target = xyz.copy()
                ik_future = ik_worker.submit(kinematics.solve, xyz, max_iterations=100)
            if (abs(bearing) < .06 and reachable
                    and observed.handle_confidence >= args.minimum_yolo_confidence):
                node.manual_cmd_vel_pub.publish(command)
                if sample['time_sec'] != used_time:
                    stable += 1
                    used_time = sample['time_sec']
                if stable >= 3:
                    return observed, traveled
            else:
                stable = 0
                command.angular.z = float(np.clip(0.6 * bearing, -.16, .16))
                if abs(bearing) < .12 and not reachable:
                    if point.x < .4:
                        return None, traveled
                    command.linear.x = .04
                node.manual_cmd_vel_pub.publish(command)
        return None, 0.0
    finally:
        node.manual_cmd_vel_pub.publish(Twist())
        ik_worker.shutdown(wait=True, cancel_futures=True)


def _stop_launch(proc: subprocess.Popen | None):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=8)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=5)
            except Exception:
                pass


def run(args: argparse.Namespace) -> int:
    if args.feedback_contact and (not args.use_yolo_observation
                                  or args.attach_grasp_joint
                                  or args.extra_sweep or args.mobile_follow):
        raise ValueError('Feedback proof requires live YOLO and no assisted opening')
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser() / f"physical_contact_door_test_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_log = output_dir / "launch.log"

    proc: subprocess.Popen | None = None
    if not args.attach:
        proc = _start_launch(
            launch_log, args.headless, args.door_open_motion,
            args.min_angle_rad, args.use_yolo_observation, args.feedback_contact)
        time.sleep(args.launch_settle_sec)

    rclpy.init()
    record = ContactRecord()
    node = ContactProbe(record)
    video_path = output_dir / "gazebo_lever_press_base_push.mp4"
    keyframe_dir = output_dir / "phases"
    result: dict[str, object] = {
        "test": "physical_contact_door_test",
        "door_hinge_command_topic_used": False,
        "arm_joint_commands_used": True,
        "mobile_base_commands_used": True,
        "door_open_motion": args.door_open_motion,
        "extra_sweep": args.extra_sweep,
        "attach_grasp_joint": args.attach_grasp_joint,
        "min_angle_rad": args.min_angle_rad,
        "use_yolo_observation": args.use_yolo_observation,
        "handle_model_role_required": (
            "primary" if args.use_yolo_observation else "fixture"),
        "handle_frame": "base_link",
        "handle_xyz": [args.handle_x, args.handle_y, args.handle_z],
        "pass": False,
        "feedback_contact_requested": args.feedback_contact,
        "feedback_scope": "instrumented Gazebo joints, not real lever sensing",
        "base_approach_policy": "observed handle + actual PIPER IK" if args.feedback_contact else "legacy",
    }

    try:
        ready = _wait_for_ready(node, args.ready_timeout_sec)
        result["ready"] = ready
        node.start_recording(video_path, args.video_fps, keyframe_dir)
        _spin_for(node, 1.0)
        before_path = output_dir / "before_perspective.png"
        if _save_ros_image(record.latest_image, before_path):
            record.saved_images.append(str(before_path))
        before_overhead_path = output_dir / "before_overhead.png"
        if _save_ros_image(record.latest_overhead_image, before_overhead_path):
            record.saved_images.append(str(before_overhead_path))
        before_debug_path = output_dir / "before_detection_debug.png"
        if _save_ros_image(record.latest_detection_image, before_debug_path):
            record.saved_images.append(str(before_debug_path))

        if not ready:
            result["error"] = "Timed out waiting for /door_joint_states and /open_door."
        else:
            initial = record.door_samples[-1][1] if record.door_samples else 0.0
            initial_lever = (
                record.lever_samples[-1][1] if record.lever_samples else 0.0)
            req = OpenDoor.Request()
            request_ready = True
            if args.use_yolo_observation:
                observed = _wait_for_yolo_observation(
                    node, args.yolo_timeout_sec,
                    args.minimum_yolo_observations,
                    args.minimum_yolo_confidence,
                    args.yolo_search_approach_m,
                    args.yolo_search_speed_mps)
                if observed is None:
                    request_ready = False
                    result["error"] = (
                        "No repeated yolo:primary handle observation was "
                        "received before timeout.")
                else:
                    observed_door, _fixed_handle = observed
                    _spin_for(node, 0.5)
                    yolo_path = output_dir / "yolo_primary_detection.png"
                    if _save_ros_image(
                            record.yolo_detection_image, yolo_path):
                        record.saved_images.append(str(yolo_path))

                    source = observed_door.handle_position.point
                    approach_distance = 0.0
                    if (
                            observed_door.handle_position.header.frame_id
                            == 'base_link'):
                        approach_distance = min(
                            args.max_observation_approach_m,
                            max(0.0, float(source.x)
                                - args.target_handle_x_m))
                    if args.feedback_contact:
                        observed_fresh, actual_approach = _approach_visible_handle(node, args)
                        if observed_fresh is None:
                            request_ready = False
                            result['error'] = 'No fresh visible handle at approach completion'
                        else:
                            observed_door = observed_fresh
                    elif approach_distance > 0.01:
                        actual_approach = _publish_manual_cmd_vel_for_distance(
                            node, approach_distance,
                            args.observation_approach_speed_mps,
                            args.observation_approach_timeout_sec)
                        _spin_for(node, 0.7)
                    else:
                        actual_approach = 0.0

                    # Reacquire after the base motion and command the arm from
                    # the newest camera/LiDAR observation in base_link. This
                    # prevents a stale pre-approach coordinate from being
                    # transformed through a changed robot pose.
                    latest_door = record.latest_yolo_door
                    if (not args.feedback_contact
                            and
                            latest_door is not None
                            and latest_door.handle_position.header.frame_id
                            == 'base_link'
                            and latest_door.handle_detected
                            and str(latest_door.handle_detection_method).startswith(
                                'yolo:primary')
                            and float(latest_door.handle_confidence)
                            >= args.minimum_yolo_confidence):
                        observed_door = latest_door

                    current_handle = PointStamped()
                    current_handle.header = observed_door.handle_position.header
                    current_handle.header.stamp.sec = 0
                    current_handle.header.stamp.nanosec = 0
                    current_handle.point = observed_door.handle_position.point
                    req.door_id = str(observed_door.door_id)
                    req.handle_position = current_handle
                    req.handle_detected = True
                    req.handle_detection_method = str(
                        observed_door.handle_detection_method)
                    req.handle_confidence = float(
                        observed_door.handle_confidence)
                    result.update({
                        "handle_source": "door_detection_node",
                        "handle_method": req.handle_detection_method,
                        "handle_confidence": req.handle_confidence,
                        "handle_frame": req.handle_position.header.frame_id,
                        "handle_xyz": [
                            float(req.handle_position.point.x),
                            float(req.handle_position.point.y),
                            float(req.handle_position.point.z),
                        ],
                        "initial_observed_handle_frame": str(
                            observed_door.handle_position.header.frame_id),
                        "initial_observed_handle_xyz": [
                            float(source.x), float(source.y), float(source.z)],
                        "observation_approach_distance_m": (
                            None if args.feedback_contact else approach_distance),
                        "observation_approach_actual_m": actual_approach,
                        "yolo_observation_count": len(
                            record.yolo_observations),
                    })
            else:
                req.door_id = "physical_contact_test_door"
                req.handle_position = PointStamped()
                req.handle_position.header.frame_id = "base_link"
                req.handle_position.point.x = float(args.handle_x)
                req.handle_position.point.y = float(args.handle_y)
                req.handle_position.point.z = float(args.handle_z)
                req.handle_detected = True
                req.handle_detection_method = "test_fixture_contact_handle"
                req.handle_confidence = 1.0

            future = node.client.call_async(req) if request_ready else None
            deadline = time.monotonic() + args.service_timeout_sec
            attach_sent = False
            while (
                    rclpy.ok()
                    and future is not None
                    and not future.done()
                    and time.monotonic() < deadline):
                rclpy.spin_once(node, timeout_sec=0.1)
                if args.attach_grasp_joint and not attach_sent:
                    latest_phase = record.phases[-1][1] if record.phases else ""
                    if (
                            latest_phase.startswith("GRASP_HANDLE")
                            or latest_phase.startswith("PRESS_HANDLE")
                            or latest_phase.startswith("PULL_OPEN")
                            or latest_phase.startswith("PUSH_OPEN")):
                        attach_sent = _publish_ign_empty(
                            "/fire_robot/sim_arm/handle_attach")
                        result["attach_request_sent"] = attach_sent
                        result["attach_request_phase"] = latest_phase
            _spin_for(node, args.post_wait_sec)
            if args.extra_sweep:
                _run_full_open_sweep(node)
                _spin_for(node, args.post_wait_sec)
            if args.mobile_follow:
                _run_mobile_full_open_follow(node)
                _spin_for(node, args.post_wait_sec)

            service_success = False
            service_message = (
                "service call timed out" if request_ready
                else "service was not called")
            if future is not None and future.done():
                response = future.result()
                if response is not None:
                    service_success = bool(response.success)
                    service_message = str(response.message)

            angles = [a for _, a in record.door_samples]
            lever_angles = [a for _, a in record.lever_samples]
            final = angles[-1] if angles else initial
            max_delta = max((abs(a - initial) for a in angles), default=0.0)
            final_lever = (
                lever_angles[-1] if lever_angles else initial_lever)
            max_lever_delta = max(
                (abs(a - initial_lever) for a in lever_angles), default=0.0)
            base_displacement = 0.0
            if len(record.base_samples) >= 2:
                _, x0, y0 = record.base_samples[0]
                _, x1, y1 = record.base_samples[-1]
                base_displacement = math.hypot(x1 - x0, y1 - y0)
            result.update({
                "service_success": service_success,
                "service_message": service_message,
                "initial_angle_rad": initial,
                "final_angle_rad": final,
                "max_delta_rad": max_delta,
                "initial_lever_angle_rad": initial_lever,
                "final_lever_angle_rad": final_lever,
                "max_lever_delta_rad": max_lever_delta,
                "base_displacement_m": base_displacement,
                "yolo_observations": record.yolo_observations,
                "sample_count": len(record.door_samples),
                "phases": record.phases,
                "pass": (
                    service_success
                    and max_delta >= args.min_angle_rad
                    and abs(final - initial) >= args.min_angle_rad - 0.12
                    and max_lever_delta >= args.min_lever_press_rad),
            })

        _spin_for(node, 1.0)
        node.stop_recording()
        after_path = output_dir / "after_perspective.png"
        if _save_ros_image(record.latest_image, after_path):
            record.saved_images.append(str(after_path))
        after_overhead_path = output_dir / "after_overhead.png"
        if _save_ros_image(record.latest_overhead_image, after_overhead_path):
            record.saved_images.append(str(after_overhead_path))
        after_debug_path = output_dir / "after_detection_debug.png"
        if _save_ros_image(record.latest_detection_image, after_debug_path):
            record.saved_images.append(str(after_debug_path))
        raw_front_path = output_dir / 'after_front_raw.png'
        if _save_ros_image(record.latest_front_image, raw_front_path):
            record.saved_images.append(str(raw_front_path))
        angle_plot = output_dir / "door_hinge_angle.png"
        _plot_angles(record, angle_plot, args.min_angle_rad)
        record.saved_images.append(str(angle_plot))
        result["artifacts"] = {
            "output_dir": str(output_dir),
            "launch_log": str(launch_log),
            "images": record.saved_images,
            "angle_plot": str(angle_plot),
            "video": str(video_path),
            "phase_keyframes": str(keyframe_dir),
        }
        result['feedback_events'] = record.feedback_events
        if args.feedback_contact:
            names = {e.get('event') for e in record.feedback_events}
            result['feedback_contract_pass'] = (
                {'approach_clearance', 'approach_reached', 'grasp_candidate',
                 'press_verified', 'latch_released', 'home_verified'} <= names
                and 'stopped' not in names)
            result['pass'] = bool(result['pass'] and result['feedback_contract_pass'])
    finally:
        node.stop_recording()
        node.destroy_node()
        rclpy.shutdown()
        if not args.attach:
            _stop_launch(proc)

    report_path = output_dir / "result.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("pass") else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="artifacts/validation/physical_contact",
    )
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--launch-settle-sec", type=float, default=8.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=25.0)
    parser.add_argument("--service-timeout-sec", type=float, default=240.0)
    parser.add_argument("--post-wait-sec", type=float, default=2.0)
    parser.add_argument("--min-angle-rad", type=float, default=2.05)
    parser.add_argument("--min-lever-press-rad", type=float, default=0.08)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--use-yolo-observation", action="store_true")
    parser.add_argument("--feedback-contact", action="store_true")
    parser.add_argument("--yolo-timeout-sec", type=float, default=40.0)
    parser.add_argument("--minimum-yolo-observations", type=int, default=2)
    parser.add_argument("--minimum-yolo-confidence", type=float, default=0.45)
    parser.add_argument("--tracking-min-yolo-confidence", type=float, default=0.20)
    parser.add_argument("--yolo-search-approach-m", type=float, default=0.45)
    parser.add_argument("--yolo-search-speed-mps", type=float, default=0.04)
    parser.add_argument("--target-handle-x-m", type=float, default=0.58)
    parser.add_argument(
        "--observation-approach-speed-mps", type=float, default=0.08)
    parser.add_argument(
        "--observation-approach-timeout-sec", type=float, default=40.0)
    parser.add_argument(
        "--max-observation-approach-m", type=float, default=0.85)
    parser.add_argument(
        "--door-open-motion", choices=("pull", "push"), default="push")
    parser.add_argument("--extra-sweep", action="store_true")
    parser.add_argument("--attach-grasp-joint", action="store_true")
    parser.add_argument("--mobile-follow", action="store_true")
    parser.add_argument("--handle-x", type=float, default=0.43)
    parser.add_argument("--handle-y", type=float, default=0.24)
    parser.add_argument("--handle-z", type=float, default=0.66)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
