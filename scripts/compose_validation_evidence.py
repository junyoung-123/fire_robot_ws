#!/usr/bin/env python3
"""Compose world validation trace images into a compact evidence sheet."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXPECTED_OPENED = {1: 3, 2: 3, 3: 4, 4: 0, 5: 6}


def _thumb(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert('RGB')
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.LANCZOS)


def _log_summary(log_path: Path, status_path: Path) -> list[str]:
    text = log_path.read_text(errors='ignore') if log_path.exists() else ''
    status = status_path.read_text(errors='ignore').strip() if status_path.exists() else 'missing'
    matched = sorted(set(re.findall(r'to (/fire_robot/door/blue/[^ ]+)', text)))
    rejected = len(re.findall(r'Rejected Gazebo', text))
    opens = len(re.findall(r'문 개방 성공', text))
    duplicate = len(re.findall(r'중복 문 개방 요청 완료 처리', text))
    mission = 'yes' if 'MISSION_COMPLETE' in text else 'no'
    return [
        f'status: {status}',
        f'mission_complete: {mission}',
        f'open_success_logs: {opens}',
        f'unique_blue_matches: {len(matched)} / {EXPECTED_OPENED.get(int(log_path.stem[-1]), "?")}',
        f'rejected_matches: {rejected}',
        f'idempotent_duplicates: {duplicate}',
    ]


def main() -> int:
    if len(sys.argv) != 3:
        print('usage: compose_validation_evidence.py <validation_dir> <output_png>')
        return 2

    base = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    traj_w = 760
    align_w = 700
    left_w = 270
    row_gap = 22
    pad = 18
    rows: list[Image.Image] = []

    for world in range(1, 6):
        trace_dir = base / f'world{world}_trace'
        trajectory = _thumb(trace_dir / 'trajectory.png', traj_w)
        alignment = _thumb(trace_dir / 'door_alignment.png', align_w)
        row_h = max(trajectory.height, alignment.height, 230) + pad * 2
        row_w = left_w + traj_w + align_w + pad * 4
        row = Image.new('RGB', (row_w, row_h), 'white')
        draw = ImageDraw.Draw(row)
        draw.rectangle((0, 0, row_w - 1, row_h - 1), outline=(210, 210, 210), width=2)
        draw.text((pad, pad), f'World {world}', fill=(10, 10, 10), font=title_font)
        draw.text((pad, pad + 22), '\n'.join(_log_summary(
            base / f'world{world}.log',
            base / f'world{world}.log.status',
        )), fill=(35, 35, 35), font=font)
        x = left_w + pad
        row.paste(trajectory, (x, pad))
        x += traj_w + pad
        row.paste(alignment, (x, pad))
        rows.append(row)

    total_w = max(row.width for row in rows)
    total_h = sum(row.height for row in rows) + row_gap * (len(rows) - 1)
    sheet = Image.new('RGB', (total_w, total_h), (245, 247, 250))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height + row_gap

    sheet.save(output)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
