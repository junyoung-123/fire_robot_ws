#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def parse_log(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"validation log not found: {path}")
    text = path.read_text(errors="ignore")
    matched = re.findall(
        r"Matched observed door (?P<door>[^ ]+) to (?P<topic>/fire_robot/door/blue/[^ ]+)",
        text,
    )
    request = re.findall(
        r"Open door request: id=(?P<door>[^,]+), handle_method=(?P<method>[^,]+), "
        r"handle_detected=(?P<detected>[^,]+), handle_conf=(?P<conf>[^\n]+)",
        text,
    )
    handle = re.findall(
        r"문 개방 요청 손잡이: id=(?P<door>[^,]+), method=(?P<method>[^,]+), "
        r"conf=(?P<conf>[^,]+), frame=(?P<frame>[^,]+), "
        r"pos=\((?P<x>[-0-9.]+),(?P<y>[-0-9.]+),(?P<z>[-0-9.]+)\)",
        text,
    )
    stages = re.findall(r"Manipulation phase: ([A-Z_]+:[^\n]+)", text)
    target_angle = re.findall(r"Gazebo door topic=.*target_angle=([-0-9.]+)", text)
    success = re.findall(r"문 개방 성공: ([^ ]+)", text)

    if not success:
        raise ValueError("log has no successful FSM door-opening event")
    chosen_door = success[0]
    chosen_match = next((m for m in matched if chosen_door in m[0]), None)
    chosen_handle = next((h for h in handle if chosen_door in h[0]), None)
    chosen_request = next((r for r in request if chosen_door in r[0]), None)
    missing = [
        name for name, value in (
            ("Gazebo door match", chosen_match),
            ("manipulation request", chosen_request),
            ("FSM handle request", chosen_handle),
            ("hinge target angle", target_angle),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "log is insufficient for manipulation evidence: " + ", ".join(missing))
    chosen_stages = [s.split(":", 1)[0] for s in stages if chosen_door in s]
    required_stages = {"LOCALIZE_HANDLE", "PRESS_HANDLE", "PUSH_OPEN", "COMPLETE"}
    missing_stages = sorted(required_stages - set(chosen_stages))
    if missing_stages:
        raise ValueError(
            "log is missing required manipulation stages: "
            + ", ".join(missing_stages))

    compact_stage = {
        "LOCALIZE_HANDLE": "LOCALIZE",
        "GRASP_HANDLE": "GRASP",
        "PRESS_HANDLE": "PRESS",
        "PUSH_OPEN": "PUSH",
        "RETURN_HOME": "HOME",
        "POST_OPEN_BACKOFF": "BACKOFF",
    }
    compact_stages = [compact_stage.get(stage, stage) for stage in chosen_stages[:8]]

    return {
        "door": chosen_door,
        "topic": chosen_match[1],
        "method": chosen_request[1],
        "detected": chosen_request[2],
        "confidence": chosen_request[3].strip(),
        "handle_xyz": (
            f"({float(chosen_handle[4]):.2f}, {float(chosen_handle[5]):.2f}, "
            f"{float(chosen_handle[6]):.2f})"
        ),
        "target_angle": target_angle[0],
        "stages": " -> ".join(compact_stages),
    }


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          fill: tuple[int, int, int], width: int = 5) -> None:
    draw.line([start, end], fill=fill, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 16
    left = (
        int(end[0] - length * math.cos(ang - 0.5)),
        int(end[1] - length * math.sin(ang - 0.5)),
    )
    right = (
        int(end[0] - length * math.cos(ang + 0.5)),
        int(end[1] - length * math.sin(ang + 0.5)),
    )
    draw.polygon([end, left, right], fill=fill)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
          title: str, subtitle: str, door_angle: float) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=12, fill=(248, 250, 252), outline=(203, 213, 225), width=2)
    draw.text((x1 + 22, y1 + 18), title, fill=(15, 23, 42), font=font(27, True))
    draw.text((x1 + 22, y1 + 54), subtitle, fill=(71, 85, 105), font=font(17))

    base = (x1 + 175, y1 + 235)
    draw.rounded_rectangle((base[0] - 58, base[1] - 38, base[0] + 58, base[1] + 38),
                           radius=14, fill=(51, 65, 85), outline=(15, 23, 42), width=2)
    draw.text((base[0] - 34, base[1] - 12), "BASE", fill=(248, 250, 252), font=font(16, True))

    hinge = (x1 + 360, y1 + 125)
    closed_end = (x1 + 360, y1 + 285)
    draw.line([hinge, closed_end], fill=(30, 64, 175), width=13)
    draw.ellipse((hinge[0] - 9, hinge[1] - 9, hinge[0] + 9, hinge[1] + 9), fill=(15, 23, 42))

    length = 170
    opened_end = (
        int(hinge[0] + length * math.sin(door_angle)),
        int(hinge[1] + length * math.cos(door_angle)),
    )
    if door_angle > 0.05:
        draw.line([hinge, opened_end], fill=(37, 99, 235), width=13)
        draw.arc((hinge[0] - 88, hinge[1] - 12, hinge[0] + 88, hinge[1] + 164),
                 start=90, end=90 + int(math.degrees(door_angle)), fill=(37, 99, 235), width=4)

    handle = (x1 + 360, y1 + 225)
    draw.rounded_rectangle((handle[0] - 4, handle[1] - 32, handle[0] + 12, handle[1] + 32),
                           radius=4, fill=(202, 138, 4))

    shoulder = (base[0] + 42, base[1] - 20)
    elbow = (x1 + 255, y1 + 190)
    wrist = (x1 + 325, y1 + 216 if door_angle < 0.05 else y1 + 210)
    draw.line([shoulder, elbow, wrist], fill=(14, 116, 144), width=8)
    draw.ellipse((shoulder[0] - 8, shoulder[1] - 8, shoulder[0] + 8, shoulder[1] + 8), fill=(8, 145, 178))
    draw.ellipse((elbow[0] - 7, elbow[1] - 7, elbow[0] + 7, elbow[1] + 7), fill=(8, 145, 178))
    draw.ellipse((wrist[0] - 7, wrist[1] - 7, wrist[0] + 7, wrist[1] + 7), fill=(8, 145, 178))
    arrow(draw, wrist, (handle[0] - 9, handle[1]), (202, 138, 4), 4)


def render(log_path: Path, out_path: Path) -> None:
    info = parse_log(log_path)
    img = Image.new("RGB", (1800, 1080), (241, 245, 249))
    draw = ImageDraw.Draw(img)

    draw.text((52, 38), "Robot Arm Door Opening Evidence",
              fill=(15, 23, 42), font=font(42, True))
    draw.text((54, 92),
              "Log-based schematic from the Gazebo/ROS2 manipulation FSM, not a raw camera screenshot.",
              fill=(71, 85, 105), font=font(21))

    panel(draw, (55, 150, 585, 545), "1. Align + Localize", "Robot stops square in front of the blue door", 0.0)
    panel(draw, (635, 150, 1165, 545), "2. Press Lever", "Arm reaches handle and presses the lever", 0.0)
    panel(draw, (1215, 150, 1745, 545), "3. Push Door", "Hinge command opens the simulated door", 2.05)

    box = (55, 620, 1745, 1005)
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    y = 650
    lines = [
        ("Validated door", info["door"]),
        ("Gazebo hinge topic", info["topic"]),
        ("Handle source", f'{info["method"]}, detected={info["detected"]}, confidence={info["confidence"]}'),
        ("Handle map position", info["handle_xyz"]),
        ("Door opening command", f'{info["target_angle"]} rad (~120 deg)'),
        ("Manipulation FSM", info["stages"]),
        ("Source log", str(log_path)),
    ]
    value_font = font(21)
    for key, value in lines:
        draw.text((88, y), key, fill=(15, 23, 42), font=font(21, True))
        value_lines = []
        rest = value
        while len(rest) > 96:
            cut = rest.rfind(" ", 0, 96)
            if cut < 45:
                cut = 96
            value_lines.append(rest[:cut].rstrip())
            rest = rest[cut:].lstrip()
        value_lines.append(rest)
        for idx, line in enumerate(value_lines):
            draw.text((430, y + idx * 28), line, fill=(51, 65, 85), font=value_font)
        y += 43 + (len(value_lines) - 1) * 28

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(Path(args.log), Path(args.out))


if __name__ == "__main__":
    main()
