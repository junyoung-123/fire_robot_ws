#!/usr/bin/env python3
"""Run a Gazebo contact-only door-opening check.

The door is considered opened only when the Gazebo door_hinge joint moves.
The test never publishes to a door hinge command topic; only the simulated arm
and gripper joint position topics are commanded through manipulation_node.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from fire_robot_interfaces.srv import OpenDoor
from geometry_msgs.msg import PointStamped, Twist
from PIL import Image as PilImage
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64, String


@dataclass
class ContactRecord:
    start_wall_time: float = field(default_factory=time.monotonic)
    door_samples: list[tuple[float, float]] = field(default_factory=list)
    phases: list[tuple[float, str]] = field(default_factory=list)
    latest_image: Image | None = None
    saved_images: list[str] = field(default_factory=list)


class ContactProbe(Node):
    def __init__(self, record: ContactRecord):
        super().__init__("physical_contact_probe")
        self.record = record
        self.create_subscription(
            JointState, "/door_joint_states", self._door_cb, 20)
        self.create_subscription(
            String, "/manipulation_phase", self._phase_cb, 20)
        self.create_subscription(
            Image, "/proof/overhead/image", self._image_cb, 5)
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

    def _stamp(self) -> float:
        return time.monotonic() - self.record.start_wall_time

    def _door_cb(self, msg: JointState):
        try:
            index = msg.name.index("door_hinge")
        except ValueError:
            return
        try:
            angle = float(msg.position[index])
        except (IndexError, ValueError):
            return
        self.record.door_samples.append((self._stamp(), angle))

    def _phase_cb(self, msg: String):
        self.record.phases.append((self._stamp(), msg.data))

    def _image_cb(self, msg: Image):
        self.record.latest_image = msg


def _save_ros_image(msg: Image | None, path: Path) -> bool:
    if msg is None:
        return False
    encoding = (msg.encoding or "").lower()
    channels = 3
    mode = "RGB"
    if encoding in ("rgba8", "bgra8"):
        channels = 4
        mode = "RGBA"
    elif encoding in ("mono8", "8uc1"):
        channels = 1
        mode = "L"

    data = np.frombuffer(msg.data, dtype=np.uint8)
    if channels == 1:
        image = data.reshape((msg.height, msg.width))
    else:
        image = data.reshape((msg.height, msg.width, channels))
    if encoding in ("bgr8", "bgra8"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    PilImage.fromarray(image, mode=mode).save(path)
    return True


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
                     seconds: float):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        node.cmd_vel_pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
    msg = Twist()
    node.cmd_vel_pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)


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


def _start_launch(log_path: Path, headless: bool,
                  door_open_motion: str,
                  min_angle_rad: float) -> subprocess.Popen:
    headless_arg = "true" if headless else "false"
    command = (
        "cd /home/junyoung/fire_robot_ws_test && "
        "source /opt/ros/humble/setup.bash && "
        "source install/setup.bash && "
        "ros2 launch fire_robot_bringup physical_contact_door_test.launch.py "
        f"headless:={headless_arg} use_rviz:=false "
        f"door_open_motion:={door_open_motion} "
        f"contact_min_angle_rad:={float(min_angle_rad):.6f}"
    )
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        ["bash", "-lc", command],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )


def _stop_launch(proc: subprocess.Popen | None):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=8)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass


def run(args: argparse.Namespace) -> int:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser() / f"physical_contact_door_test_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_log = output_dir / "launch.log"

    proc: subprocess.Popen | None = None
    if not args.attach:
        proc = _start_launch(
            launch_log, args.headless, args.door_open_motion,
            args.min_angle_rad)
        time.sleep(args.launch_settle_sec)

    rclpy.init()
    record = ContactRecord()
    node = ContactProbe(record)
    result: dict[str, object] = {
        "test": "physical_contact_door_test",
        "door_hinge_command_topic_used": False,
        "arm_joint_commands_used": True,
        "mobile_base_commands_used": args.mobile_follow,
        "door_open_motion": args.door_open_motion,
        "extra_sweep": args.extra_sweep,
        "attach_grasp_joint": args.attach_grasp_joint,
        "min_angle_rad": args.min_angle_rad,
        "handle_frame": "base_link",
        "handle_xyz": [args.handle_x, args.handle_y, args.handle_z],
        "pass": False,
    }

    try:
        ready = _wait_for_ready(node, args.ready_timeout_sec)
        result["ready"] = ready
        before_path = output_dir / "before_overhead.png"
        if _save_ros_image(record.latest_image, before_path):
            record.saved_images.append(str(before_path))

        if not ready:
            result["error"] = "Timed out waiting for /door_joint_states and /open_door."
        else:
            initial = record.door_samples[-1][1] if record.door_samples else 0.0
            req = OpenDoor.Request()
            req.door_id = "physical_contact_test_door"
            req.handle_position = PointStamped()
            req.handle_position.header.frame_id = "base_link"
            req.handle_position.point.x = float(args.handle_x)
            req.handle_position.point.y = float(args.handle_y)
            req.handle_position.point.z = float(args.handle_z)
            req.handle_detected = True
            req.handle_detection_method = "test_fixture_contact_handle"
            req.handle_confidence = 1.0

            future = node.client.call_async(req)
            deadline = time.monotonic() + args.service_timeout_sec
            attach_sent = False
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
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
            service_message = "service call timed out"
            if future.done():
                response = future.result()
                if response is not None:
                    service_success = bool(response.success)
                    service_message = str(response.message)

            angles = [a for _, a in record.door_samples]
            final = angles[-1] if angles else initial
            max_delta = max((abs(a - initial) for a in angles), default=0.0)
            result.update({
                "service_success": service_success,
                "service_message": service_message,
                "initial_angle_rad": initial,
                "final_angle_rad": final,
                "max_delta_rad": max_delta,
                "sample_count": len(record.door_samples),
                "phases": record.phases,
                "pass": service_success and max_delta >= args.min_angle_rad,
            })

        after_path = output_dir / "after_overhead.png"
        if _save_ros_image(record.latest_image, after_path):
            record.saved_images.append(str(after_path))
        angle_plot = output_dir / "door_hinge_angle.png"
        _plot_angles(record, angle_plot, args.min_angle_rad)
        record.saved_images.append(str(angle_plot))
        result["artifacts"] = {
            "output_dir": str(output_dir),
            "launch_log": str(launch_log),
            "images": record.saved_images,
            "angle_plot": str(angle_plot),
        }
    finally:
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
        default="/mnt/c/Users/황준영/Documents/졸업작품/검증결과",
    )
    parser.add_argument("--attach", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--launch-settle-sec", type=float, default=8.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=25.0)
    parser.add_argument("--service-timeout-sec", type=float, default=35.0)
    parser.add_argument("--post-wait-sec", type=float, default=2.0)
    parser.add_argument("--min-angle-rad", type=float, default=0.75)
    parser.add_argument(
        "--door-open-motion", choices=("pull", "push"), default="push")
    parser.add_argument("--extra-sweep", action="store_true")
    parser.add_argument("--attach-grasp-joint", action="store_true")
    parser.add_argument("--mobile-follow", action="store_true")
    parser.add_argument("--handle-x", type=float, default=0.43)
    parser.add_argument("--handle-y", type=float, default=0.24)
    parser.add_argument("--handle-z", type=float, default=0.83)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
