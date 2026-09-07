#!/usr/bin/env python3
"""Render validation paths with the obstacle footprints from each Gazebo world."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


WORLD_FILES = {
    1: "obstacle_wall_doors_v5.world",
    2: "obstacle_door_layout_alt_v1.world",
    3: "obstacle_door_layout_world3_v1.world",
    4: "obstacle_no_blue_world4_v1.world",
    5: "obstacle_all_blue_world5_v1.world",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def floats(text: str | None, count: int) -> list[float] | None:
    if not text:
        return None
    try:
        values = [float(value) for value in text.split()]
    except ValueError:
        return None
    return values if len(values) >= count else None


def obstacle_footprints(world_path: Path) -> list[tuple[str, list[tuple[float, float]]]]:
    root = ET.parse(world_path).getroot()
    footprints = []
    for model in root.findall(".//world/model"):
        name = model.get("name", "")
        lower_name = name.lower()
        if not any(token in lower_name for token in ("obstacle", "block", "clutter", "_obs_")):
            continue
        pose = floats(model.findtext("pose"), 6)
        collision = model.find("./link/collision")
        size = floats(collision.findtext("./geometry/box/size") if collision is not None else None, 3)
        if pose is None or size is None:
            continue
        cx, cy, yaw = pose[0], pose[1], pose[5]
        collision_pose = floats(collision.findtext("pose"), 6) if collision is not None else None
        if collision_pose is not None:
            dx, dy = collision_pose[0], collision_pose[1]
            cx += math.cos(yaw) * dx - math.sin(yaw) * dy
            cy += math.sin(yaw) * dx + math.cos(yaw) * dy
            yaw += collision_pose[5]
        half_x, half_y = size[0] / 2.0, size[1] / 2.0
        corners = []
        for local_x, local_y in ((-half_x, -half_y), (half_x, -half_y),
                                 (half_x, half_y), (-half_x, half_y)):
            corners.append((
                cx + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
                cy + math.sin(yaw) * local_x + math.cos(yaw) * local_y,
            ))
        footprints.append((name, corners))
    return footprints


def finite_xy(row: dict[str, str], x_key: str = "x", y_key: str = "y"):
    try:
        x, y = float(row[x_key]), float(row[y_key])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def render_world(ax, world: int, validation_dir: Path, worlds_dir: Path):
    trace_dir = validation_dir / f"world{world}_trace"
    poses = [point for row in read_csv(trace_dir / "poses.csv")
             if (point := finite_xy(row)) is not None]
    events = read_csv(trace_dir / "events.csv")
    targets = read_csv(trace_dir / "targets.csv")
    footprints = obstacle_footprints(worlds_dir / WORLD_FILES[world])

    for index, (name, corners) in enumerate(footprints, start=1):
        ax.add_patch(Polygon(
            corners, closed=True, facecolor="#f4a261", edgecolor="#9c4f14",
            linewidth=1.1, alpha=0.58, label="Gazebo obstacle footprint" if index == 1 else None))
        center_x = sum(point[0] for point in corners) / 4.0
        center_y = sum(point[1] for point in corners) / 4.0
        width = math.dist(corners[0], corners[1])
        depth = math.dist(corners[1], corners[2])
        ax.text(center_x, center_y, f"{width:.2f}x{depth:.2f}m",
                fontsize=5.8, ha="center", va="center", color="#4d260b")

    if poses:
        ax.plot([point[0] for point in poses], [point[1] for point in poses],
                color="#202124", linewidth=1.7, label="robot path", zorder=4)
        ax.scatter(poses[0][0], poses[0][1], marker="o", s=35,
                   color="#6a4c93", label="start", zorder=6)

    opened_label = True
    complete_label = True
    for event in events:
        point = finite_xy(event)
        if point is None:
            continue
        if event.get("event") == "DOOR_OPENED":
            ax.scatter(*point, marker="*", s=90, color="#111111", zorder=7,
                       label="door opened" if opened_label else None)
            opened_label = False
        elif event.get("event") == "MISSION_COMPLETE":
            ax.scatter(*point, marker="D", s=48, color="#00a6a6", zorder=7,
                       label="mission complete" if complete_label else None)
            complete_label = False

    latest_plans = {}
    for target in targets:
        if target.get("color") not in ("blue", "green"):
            continue
        point = finite_xy(target)
        if point is not None:
            latest_plans[target["color"]] = point
    for color, marker, label in (("blue", "P", "latest blue target"),
                                 ("green", "X", "exit target")):
        if color in latest_plans:
            ax.scatter(*latest_plans[color], marker=marker, s=55,
                       color="#277da1" if color == "blue" else "#2a9d4b",
                       label=label, zorder=6)

    status_path = validation_dir / f"world{world}.log.status"
    status = status_path.read_text(errors="ignore").strip() if status_path.exists() else "missing"
    ax.set_title(f"World {world} - {status}")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    parser.add_argument("worlds_dir", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()

    fig, axes = plt.subplots(5, 1, figsize=(15, 18), dpi=160)
    for world, ax in enumerate(axes, start=1):
        render_world(ax, world, args.validation_dir, args.worlds_dir)
    fig.suptitle("Full Validation: Robot Paths and Gazebo Obstacle Size/Position", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png)
    plt.close(fig)
    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
