#!/usr/bin/env python3
"""Convert visual-only blue SDF doors into hinged Gazebo doors.

The navigation stack still observes door candidates from camera/LiDAR data.
This helper only upgrades simulation assets so an accepted blue-door open
request can rotate the corresponding Gazebo door joint.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORLD_DIR = ROOT / "src" / "fire_robot_bringup" / "worlds"

DOOR_HANDLE_PATTERN = re.compile(
    r'(?P<indent>[ \t]*)<model name="door_blue(?P<idx>\d+)">(?P<door>.*?)'
    r'</model>\s*'
    r'(?P=indent)<model name="handle_blue(?P=idx)">(?P<handle>.*?)'
    r'</model>',
    re.DOTALL,
)


def parse_pose(block: str) -> list[float]:
    match = re.search(r"<pose>([^<]+)</pose>", block)
    if not match:
        raise ValueError("model block has no pose")
    values = [float(v) for v in match.group(1).split()]
    if len(values) != 6:
        raise ValueError(f"expected 6 pose values, got {values}")
    return values


def parse_panel_size(block: str) -> list[float]:
    match = re.search(r"<box>\s*<size>([^<]+)</size>\s*</box>", block)
    if not match:
        return [0.06, 0.80, 2.00]
    values = [float(v) for v in match.group(1).split()]
    if len(values) != 3:
        return [0.06, 0.80, 2.00]
    return values


def parse_material(block: str) -> str:
    match = re.search(r"<material>(.*?)</material>", block, re.DOTALL)
    if not match:
        return (
            "<material><ambient>0.04 0.14 0.92 1</ambient>"
            "<diffuse>0.04 0.14 0.92 1</diffuse></material>"
        )
    return "<material>" + match.group(1).strip() + "</material>"


def fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def topic_for_pose(door_pose: list[float], open_sign: float) -> str:
    x_code = int(round(door_pose[0] * 100.0))
    y_code = int(round(abs(door_pose[1]) * 100.0))
    y_sign = "p" if door_pose[1] >= 0.0 else "n"
    sign_code = "p" if open_sign >= 0.0 else "n"
    return f"/fire_robot/door/blue/x{x_code}/y{y_sign}{y_code}/s{sign_code}/cmd"


def hinge_and_open_sign(door_pose: list[float],
                        size: list[float]) -> tuple[float, float, float]:
    wall_side = 1.0 if door_pose[1] >= 0.0 else -1.0
    if size[0] >= size[1]:
        # Current wall-door worlds place side doors flush with the wall.  The
        # long horizontal dimension is x, so the hinge is the rear x edge and
        # the open sign pushes the handle inward from the wall side.
        return -size[0] / 2.0, 0.0, -wall_side

    # Older corridor worlds used visual panels protruding in y.  Keep their
    # hinge on the wall-side y edge to preserve the initial appearance.
    return 0.0, wall_side * size[1] / 2.0, wall_side


def build_hinged_model(match: re.Match[str]) -> str:
    indent = match.group("indent")
    idx = match.group("idx")
    door_block = match.group("door")
    handle_block = match.group("handle")

    door_pose = parse_pose(door_block)
    handle_pose = parse_pose(handle_block)
    size = parse_panel_size(door_block)
    material = parse_material(door_block)

    handle_rel = [handle_pose[i] - door_pose[i] for i in range(3)]
    hinge_x, hinge_y, open_sign = hinge_and_open_sign(door_pose, size)
    topic = topic_for_pose(door_pose, open_sign)

    pose_text = " ".join(fmt(v) for v in door_pose)
    size_text = " ".join(fmt(v) for v in size)
    handle_side = 1.0 if handle_rel[1] >= 0.0 else -1.0
    handle_z = handle_rel[2]
    plate_y = handle_side * 0.032
    lever_y = handle_side * 0.078
    collision_y = handle_side * 0.074
    hinge_pose_text = f"{fmt(hinge_x)} {fmt(hinge_y)} 0 0 0 0"

    i = indent
    return f"""{i}<model name="door_blue{idx}">
{i}  <static>false</static>
{i}  <pose>{pose_text}</pose>
{i}  <link name="panel">
{i}    <inertial>
{i}      <mass>2.0</mass>
{i}      <inertia>
{i}        <ixx>0.90</ixx><ixy>0</ixy><ixz>0</ixz>
{i}        <iyy>0.90</iyy><iyz>0</iyz><izz>0.12</izz>
{i}      </inertia>
{i}    </inertial>
{i}    <collision name="panel_collision">
{i}      <geometry><box><size>{size_text}</size></box></geometry>
{i}      <surface>
{i}        <friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction>
{i}      </surface>
{i}    </collision>
{i}    <visual name="panel">
{i}      <geometry><box><size>{size_text}</size></box></geometry>
{i}      {material}
{i}    </visual>
{i}    <collision name="handle_collision">
{i}      <pose>{fmt(handle_rel[0] + 0.06)} {fmt(collision_y)} {fmt(handle_z)} 0 0 0</pose>
{i}      <geometry><box><size>0.32 0.07 0.055</size></box></geometry>
{i}    </collision>
{i}    <visual name="handle_backplate">
{i}      <pose>{fmt(handle_rel[0] - 0.06)} {fmt(plate_y)} {fmt(handle_z)} 0 0 0</pose>
{i}      <geometry><box><size>0.08 0.022 0.28</size></box></geometry>
{i}      <material><ambient>0.75 0.60 0.05 1</ambient><diffuse>0.75 0.60 0.05 1</diffuse></material>
{i}    </visual>
{i}    <visual name="handle">
{i}      <pose>{fmt(handle_rel[0] + 0.06)} {fmt(lever_y)} {fmt(handle_z)} 0 0 0</pose>
{i}      <geometry><box><size>0.32 0.052 0.052</size></box></geometry>
{i}      <material><ambient>0.75 0.60 0.05 1</ambient><diffuse>0.75 0.60 0.05 1</diffuse></material>
{i}    </visual>
{i}  </link>
{i}  <joint name="hinge" type="revolute">
{i}    <pose>{hinge_pose_text}</pose>
{i}    <parent>world</parent>
{i}    <child>panel</child>
{i}    <axis>
{i}      <xyz>0 0 1</xyz>
{i}      <limit><lower>-2.10</lower><upper>2.10</upper><effort>40</effort><velocity>1.5</velocity></limit>
{i}      <dynamics><damping>1.0</damping><friction>0.05</friction></dynamics>
{i}    </axis>
{i}  </joint>
{i}  <plugin filename="ignition-gazebo-joint-position-controller-system"
{i}          name="gz::sim::systems::JointPositionController">
{i}    <joint_name>hinge</joint_name>
{i}    <topic>{topic}</topic>
{i}    <p_gain>8.0</p_gain>
{i}    <d_gain>0.4</d_gain>
{i}    <cmd_max>60</cmd_max>
{i}    <cmd_min>-60</cmd_min>
{i}  </plugin>
{i}</model>"""


def convert_world(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if "fire_robot/door/blue" in text:
        new_text, count = normalize_converted_world(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
        print(f"{path.name}: normalized {count} hinged blue doors")
        return 0

    converted = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal converted
        converted += 1
        return build_hinged_model(match)

    new_text = DOOR_HANDLE_PATTERN.sub(replace, text)
    if converted:
        path.write_text(new_text, encoding="utf-8")
    print(f"{path.name}: converted {converted} blue doors")
    return converted


def normalize_converted_world(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r'(?P<model><model name="door_blue(?P<idx>\d+)">.*?'
        r'<topic>/fire_robot/door/blue/.*?</topic>.*?</model>)',
        re.DOTALL,
    )
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        block = match.group("model")
        pose = parse_pose(block)
        size = parse_panel_size(block)
        hinge_x, hinge_y, open_sign = hinge_and_open_sign(pose, size)
        hinge_pose = f"<pose>{fmt(hinge_x)} {fmt(hinge_y)} 0 0 0 0</pose>"
        topic = f"<topic>{topic_for_pose(pose, open_sign)}</topic>"
        block = re.sub(
            r'(<joint name="hinge" type="revolute">\s*)<pose>[^<]+</pose>',
            r'\1' + hinge_pose,
            block,
            count=1,
            flags=re.DOTALL,
        )
        block = re.sub(
            r'<topic>/fire_robot/door/blue/.*?</topic>',
            topic,
            block,
            count=1,
            flags=re.DOTALL,
        )
        count += 1
        return block

    return pattern.sub(replace, text), count


def main() -> None:
    total = 0
    for world_path in sorted(WORLD_DIR.glob("*.world")):
        total += convert_world(world_path)
    print(f"total converted: {total}")


if __name__ == "__main__":
    main()
