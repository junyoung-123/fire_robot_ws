#!/usr/bin/env python3
"""Reject a validation trace when the simulated base tipped excessively."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--max-deg", type=float, default=20.0)
    args = parser.parse_args()

    values: dict[str, str] = {}
    for line in args.summary.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    roll = float(values.get("max_abs_roll_deg", "inf"))
    pitch = float(values.get("max_abs_pitch_deg", "inf"))
    limit = max(0.0, args.max_deg)
    passed = roll <= limit and pitch <= limit
    print(
        f"attitude={'PASS' if passed else 'FAIL'} "
        f"max_roll={roll:.3f}deg max_pitch={pitch:.3f}deg limit={limit:.3f}deg")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
