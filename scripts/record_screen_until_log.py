#!/usr/bin/env python3
"""Record the Windows desktop until a ROS log contains a completion marker."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--log", required=True, help="Log file to watch")
    parser.add_argument("--pattern", default="MISSION_COMPLETE")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=1500.0)
    parser.add_argument("--post-seconds", type=float, default=8.0)
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--status-file", default="")
    return parser.parse_args()


def read_log_has_pattern(log_path: Path, pattern: str) -> bool:
    try:
        return pattern in log_path.read_text(errors="ignore")
    except FileNotFoundError:
        return False


def write_status(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n", encoding="utf-8")


def grab_screen_with_retry(status_path: Path | None, retries: int = 6):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return ImageGrab.grab(all_screens=True)
        except OSError as exc:
            last_error = exc
            write_status(status_path, f"screen grab retry {attempt}/{retries}: {exc}")
            time.sleep(0.5)
    raise last_error if last_error is not None else OSError("screen grab failed")


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    log_path = Path(args.log)
    status_path = Path(args.status_file) if args.status_file else None

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_interval = 1.0 / max(args.fps, 0.1)

    first = grab_screen_with_retry(status_path)
    width, height = first.size
    scale = min(1.0, float(args.max_width) / float(width)) if args.max_width > 0 else 1.0
    frame_size = (int(width * scale), int(height * scale))

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        frame_size,
    )
    if not writer.isOpened():
        raise SystemExit(f"failed to open video writer: {output}")

    started = time.monotonic()
    pattern_seen_at: float | None = None
    frame_count = 0
    write_status(status_path, f"recording started: {output}")

    try:
        while True:
            now = time.monotonic()
            try:
                image = grab_screen_with_retry(status_path, retries=3)
            except OSError as exc:
                write_status(status_path, f"screen grab skipped: {exc}")
                time.sleep(frame_interval)
                continue
            if scale != 1.0:
                image = image.resize(frame_size)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            writer.write(frame)
            frame_count += 1

            if pattern_seen_at is None and read_log_has_pattern(log_path, args.pattern):
                pattern_seen_at = now
                write_status(status_path, f"pattern seen: {args.pattern}")

            elapsed = now - started
            if pattern_seen_at is not None and now - pattern_seen_at >= args.post_seconds:
                break
            if elapsed >= args.max_seconds:
                write_status(status_path, f"max seconds reached: {args.max_seconds}")
                break

            sleep_for = frame_interval - (time.monotonic() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        write_status(status_path, "recording interrupted")
    finally:
        writer.release()

    write_status(status_path, f"recording finished: frames={frame_count}, output={output}")
    print(output)
    print(f"frames={frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
