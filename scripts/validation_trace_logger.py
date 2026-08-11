#!/usr/bin/env python3
import argparse
import csv
import math
import signal
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from fire_robot_interfaces.msg import DoorInfo, RobotState
from nav_msgs.msg import Path as NavPath


STATE_NAMES = {
    RobotState.IDLE: "IDLE",
    RobotState.EXPLORING: "EXPLORING",
    RobotState.NAVIGATING: "NAVIGATING",
    RobotState.OPENING_DOOR: "OPENING_DOOR",
    RobotState.DOOR_OPENED: "DOOR_OPENED",
    RobotState.EXITING: "EXITING",
    RobotState.MISSION_COMPLETE: "MISSION_COMPLETE",
    RobotState.EMERGENCY_STOP: "EMERGENCY_STOP",
}


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ValidationTraceLogger(Node):
    def __init__(self, output_dir: Path, sample_period: float):
        super().__init__("validation_trace_logger")
        self.output_dir = output_dir
        self.sample_period = max(0.1, sample_period)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.poses = []
        self.targets = []
        self.detections = []
        self.paths = []
        self.path_seq = 0
        self.states = []
        self.open_events = []
        self.latest_pose = None
        self.latest_state = None
        self.latest_target_id = ""
        self.start_wall = time.monotonic()

        self.create_subscription(DoorInfo, "/target_door", self.target_cb, 10)
        self.create_subscription(DoorInfo, "/detected_door", self.detected_cb, 10)
        self.create_subscription(NavPath, "/plan",
                                 lambda msg: self.path_cb(msg, "global_plan"), 10)
        self.create_subscription(NavPath, "/local_plan",
                                 lambda msg: self.path_cb(msg, "local_plan"), 10)
        self.create_subscription(RobotState, "/robot_state", self.state_cb, 10)
        self.create_timer(self.sample_period, self.sample_pose)

    def stamp_seconds(self):
        now = self.get_clock().now().nanoseconds / 1.0e9
        if now > 0.0:
            return now
        return time.monotonic() - self.start_wall

    def pose_xy_yaw(self, pose_stamped):
        p = pose_stamped.pose.position
        return float(p.x), float(p.y), yaw_from_quat(pose_stamped.pose.orientation)

    def point_xy(self, point_stamped):
        return float(point_stamped.point.x), float(point_stamped.point.y)

    def target_cb(self, msg: DoorInfo):
        t = self.stamp_seconds()
        x = y = yaw = float("nan")
        if msg.door_pose.header.frame_id == "map":
            x, y, yaw = self.pose_xy_yaw(msg.door_pose)
        hx = hy = float("nan")
        if msg.handle_position.header.frame_id == "map":
            hx, hy = self.point_xy(msg.handle_position)
        self.latest_target_id = msg.door_id
        self.targets.append({
            "t": t,
            "id": msg.door_id,
            "color": msg.door_color,
            "x": x,
            "y": y,
            "yaw": yaw,
            "handle_x": hx,
            "handle_y": hy,
            "confidence": float(msg.confidence),
        })

    def detected_cb(self, msg: DoorInfo):
        if msg.door_color not in ("blue", "red", "green"):
            return
        t = self.stamp_seconds()
        x = y = float("nan")
        if msg.door_pose.header.frame_id == "map":
            x, y, _ = self.pose_xy_yaw(msg.door_pose)
        hx = hy = float("nan")
        if msg.handle_position.header.frame_id == "map":
            hx, hy = self.point_xy(msg.handle_position)
        self.detections.append({
            "t": t,
            "id": msg.door_id,
            "color": msg.door_color,
            "x": x,
            "y": y,
            "handle_x": hx,
            "handle_y": hy,
            "confidence": float(msg.confidence),
        })

    def path_cb(self, msg: NavPath, topic_name: str):
        if not msg.poses:
            return
        self.path_seq += 1
        t = self.stamp_seconds()
        stride = max(1, len(msg.poses) // 160)
        for idx, pose_stamped in enumerate(msg.poses[::stride]):
            if pose_stamped.header.frame_id not in ("", "map"):
                continue
            x, y, _ = self.pose_xy_yaw(pose_stamped)
            self.paths.append({
                "t": t,
                "topic": topic_name,
                "seq": self.path_seq,
                "index": idx * stride,
                "x": x,
                "y": y,
            })

    def state_cb(self, msg: RobotState):
        t = self.stamp_seconds()
        description = (msg.state_description or "").strip()
        state_name = STATE_NAMES.get(int(msg.state), description or str(int(msg.state)))
        target_id = msg.target_door_id or self.latest_target_id
        previous_state = self.latest_state
        previous_target_id = self.latest_target_id
        if (self.latest_state, self.latest_target_id) != (state_name, target_id):
            self.states.append({
                "t": t,
                "state": state_name,
                "target_id": target_id,
                "description": msg.state_description,
            })
        self.latest_state = state_name
        self.latest_target_id = target_id

        if (previous_state == "OPENING_DOOR"
                and state_name != "OPENING_DOOR"
                and previous_target_id):
            self.record_event("DOOR_OPENED", previous_target_id, t)
        if state_name in ("DOOR_OPENED", "MISSION_COMPLETE"):
            self.record_event(state_name, target_id, t)

    def record_event(self, event_name: str, target_id: str, t: float):
        pose = self.latest_pose
        if pose is None:
            return
        if (self.open_events
                and self.open_events[-1]["event"] == event_name
                and self.open_events[-1]["target_id"] == target_id
                and t - self.open_events[-1]["t"] < 2.0):
            return
        self.open_events.append({
            "t": t,
            "event": event_name,
            "target_id": target_id,
            "x": pose["x"],
            "y": pose["y"],
            "yaw": pose["yaw"],
        })

    def sample_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "base_link", Time(), timeout=Duration(seconds=0.02))
        except TransformException:
            return
        tr = tf.transform.translation
        yaw = yaw_from_quat(tf.transform.rotation)
        row = {
            "t": self.stamp_seconds(),
            "x": float(tr.x),
            "y": float(tr.y),
            "yaw": yaw,
            "state": self.latest_state or "",
            "target_id": self.latest_target_id,
        }
        self.latest_pose = row
        self.poses.append(row)

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_csv("poses.csv", self.poses,
                       ["t", "x", "y", "yaw", "state", "target_id"])
        self.write_csv("targets.csv", self.targets,
                       ["t", "id", "color", "x", "y", "yaw",
                        "handle_x", "handle_y", "confidence"])
        self.write_csv("detections.csv", self.detections,
                       ["t", "id", "color", "x", "y",
                        "handle_x", "handle_y", "confidence"])
        self.write_csv("plans.csv", self.paths,
                       ["t", "topic", "seq", "index", "x", "y"])
        self.write_csv("states.csv", self.states,
                       ["t", "state", "target_id", "description"])
        self.write_csv("events.csv", self.open_events,
                       ["t", "event", "target_id", "x", "y", "yaw"])
        self.plot_trajectory()
        self.plot_alignment()
        summary = self.output_dir / "summary.txt"
        open_count = sum(1 for e in self.open_events if e["event"] == "DOOR_OPENED")
        complete = any(e["event"] == "MISSION_COMPLETE" for e in self.open_events)
        summary.write_text(
            f"poses={len(self.poses)}\n"
            f"targets={len(self.targets)}\n"
            f"detections={len(self.detections)}\n"
            f"path_points={len(self.paths)}\n"
            f"door_open_events={open_count}\n"
            f"mission_complete={complete}\n"
        )

    def write_csv(self, name, rows, fields):
        path = self.output_dir / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})

    def plot_trajectory(self):
        fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
        ax.set_title("Validation Trace: Robot Path, Targets, Door Open Events")
        ax.set_xlabel("map x [m]")
        ax.set_ylabel("map y [m]")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

        self.scatter_detections(ax)
        pose_segments = self.filtered_pose_segments()
        if pose_segments:
            labeled = False
            for segment in pose_segments:
                if len(segment) < 2:
                    continue
                xs = [p["x"] for p in segment]
                ys = [p["y"] for p in segment]
                ax.plot(
                    xs, ys, color="#222222", linewidth=2.0,
                    label=None if labeled else "robot path")
                labeled = True
            plot_poses = [p for segment in pose_segments if len(segment) >= 2 for p in segment]
            step = max(1, len(plot_poses) // 28)
            for p in plot_poses[::step]:
                ax.arrow(
                    p["x"], p["y"],
                    0.22 * math.cos(p["yaw"]),
                    0.22 * math.sin(p["yaw"]),
                    head_width=0.06, head_length=0.08,
                    color="#222222", alpha=0.35,
                    length_includes_head=True)

        self.scatter_targets(ax)
        self.plot_latest_plans(ax)
        self.scatter_events(ax)
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "trajectory.png")
        plt.close(fig)

    def filtered_pose_segments(self):
        if not self.poses:
            return []
        max_step_m = 0.90
        max_speed_mps = 2.00
        segments = []
        current = [self.poses[0]]
        dropped = 0
        for pose in self.poses[1:]:
            prev = current[-1]
            dt = max(0.0, float(pose["t"]) - float(prev["t"]))
            dist = math.hypot(float(pose["x"]) - float(prev["x"]),
                              float(pose["y"]) - float(prev["y"]))
            too_far = dist > max_step_m
            too_fast = dt > 0.0 and dist / dt > max_speed_mps
            if too_far or too_fast:
                segments.append(current)
                current = [pose]
                dropped += 1
            else:
                current.append(pose)
        segments.append(current)
        if dropped:
            self.get_logger().warn(
                f'Trajectory plot split at {dropped} impossible TF jumps.')
        return segments

    def plot_alignment(self):
        opens = [e for e in self.open_events if e["event"] == "DOOR_OPENED"]
        fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
        ax.set_title("Door Alignment Trace: Open Poses vs Target Parking/Handle")
        ax.set_xlabel("map x [m]")
        ax.set_ylabel("map y [m]")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        self.scatter_detections(ax, alpha=0.08)

        for idx, event in enumerate(opens, start=1):
            target = self.latest_target_before(event["t"], event["target_id"])
            ax.scatter(event["x"], event["y"], marker="*", s=160,
                       color="#111111", zorder=5)
            ax.text(event["x"], event["y"] + 0.12, f"open {idx}",
                    fontsize=8, ha="center")
            ax.arrow(event["x"], event["y"],
                     0.35 * math.cos(event["yaw"]),
                     0.35 * math.sin(event["yaw"]),
                     head_width=0.08, head_length=0.10,
                     color="#111111", length_includes_head=True)
            if target is not None:
                tx, ty = target["x"], target["y"]
                hx, hy = target["handle_x"], target["handle_y"]
                if math.isfinite(tx) and math.isfinite(ty):
                    ax.scatter(tx, ty, marker="x", s=90,
                               color="#1f77b4", zorder=4)
                    ax.plot([event["x"], tx], [event["y"], ty],
                            color="#1f77b4", linestyle="--", linewidth=1.0)
                if math.isfinite(hx) and math.isfinite(hy):
                    ax.scatter(hx, hy, marker="s", s=65,
                               color="#2ca02c", zorder=4)
                    ax.plot([event["x"], hx], [event["y"], hy],
                            color="#2ca02c", linestyle=":", linewidth=1.0)

        ax.scatter([], [], marker="*", s=120, color="#111111", label="open pose")
        ax.scatter([], [], marker="x", s=70, color="#1f77b4", label="parking target")
        ax.scatter([], [], marker="s", s=55, color="#2ca02c", label="handle estimate")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(self.output_dir / "door_alignment.png")
        plt.close(fig)

    def plot_latest_plans(self, ax):
        styles = {
            "global_plan": ("#2ca02c", 1.8, 0.70, "latest global plan"),
            "local_plan": ("#ff7f0e", 1.4, 0.55, "latest local plan"),
        }
        for topic_name, (color, width, alpha, label) in styles.items():
            rows = [r for r in self.paths if r["topic"] == topic_name]
            if not rows:
                continue
            latest_seq = max(r["seq"] for r in rows)
            latest = [r for r in rows if r["seq"] == latest_seq]
            latest.sort(key=lambda r: r["index"])
            ax.plot([r["x"] for r in latest],
                    [r["y"] for r in latest],
                    color=color, linewidth=width, alpha=alpha,
                    label=label)

    def latest_target_before(self, t, target_id):
        best = None
        for target in self.targets:
            if target["t"] > t:
                continue
            if target_id and target["id"] != target_id:
                continue
            if target["color"] not in ("blue", "green"):
                continue
            best = target
        if best is not None:
            return best
        for target in self.targets:
            if target["t"] <= t and target["color"] in ("blue", "green"):
                best = target
        return best

    def scatter_detections(self, ax, alpha=0.12):
        colors = {"blue": "#1f77b4", "red": "#d62728", "green": "#2ca02c"}
        labels = {"blue": "detected blue", "red": "detected red",
                  "green": "detected green"}
        used = set()
        for color, marker in (("blue", "o"), ("red", "s"), ("green", "^")):
            rows = [d for d in self.detections
                    if d["color"] == color
                    and math.isfinite(d["x"])
                    and math.isfinite(d["y"])]
            if not rows:
                continue
            ax.scatter(
                [r["x"] for r in rows],
                [r["y"] for r in rows],
                s=12, marker=marker, color=colors[color], alpha=alpha,
                label=labels[color] if color not in used else None)
            used.add(color)

    def scatter_targets(self, ax):
        styles = {
            "blue": ("#1f77b4", "P", "blue nav target"),
            "green": ("#2ca02c", "X", "exit target"),
            "explore": ("#9467bd", ".", "explore waypoint"),
        }
        used = set()
        for target in self.targets:
            color_name = target["color"]
            if color_name not in styles:
                continue
            if not (math.isfinite(target["x"]) and math.isfinite(target["y"])):
                continue
            color, marker, label = styles[color_name]
            ax.scatter(target["x"], target["y"], marker=marker, s=45,
                       color=color, alpha=0.85,
                       label=label if color_name not in used else None)
            used.add(color_name)

    def scatter_events(self, ax):
        used_open = False
        used_done = False
        for event in self.open_events:
            if event["event"] == "DOOR_OPENED":
                label = None if used_open else "door opened pose"
                ax.scatter(event["x"], event["y"], marker="*", s=140,
                           color="#111111", label=label, zorder=6)
                used_open = True
            elif event["event"] == "MISSION_COMPLETE":
                label = None if used_done else "mission complete"
                ax.scatter(event["x"], event["y"], marker="D", s=90,
                           color="#17becf", label=label, zorder=6)
                used_done = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-period", type=float, default=0.35)
    args = parser.parse_args()

    rclpy.init()
    node = ValidationTraceLogger(Path(args.output_dir), args.sample_period)
    stop = {"value": False}

    def _stop(_signum, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while rclpy.ok() and not stop["value"]:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
