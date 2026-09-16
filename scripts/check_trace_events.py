#!/usr/bin/env python3
"""Require explicit success events in the trace used to draw evidence images."""
import argparse
import csv
import math
from pathlib import Path


def check(path, expected):
    if not path.is_file():
        return False, 'events.csv is missing'
    with path.open(newline='') as stream:
        events = list(csv.DictReader(stream))
    opened = [row for row in events if row['event'] == 'DOOR_OPENED']
    finished = [row for row in events if row['event'] == 'MISSION_COMPLETE']
    ids = {row['target_id'] for row in opened if row['target_id']}
    positions_ok = all(all(math.isfinite(float(row[key])) for key in ('x', 'y', 'yaw'))
                       for row in opened + finished)
    passed = (len(opened) == expected and len(ids) == expected
              and bool(finished) and positions_ok)
    return passed, f'opened_events={len(opened)} unique_ids={len(ids)} expected={expected} mission_complete={bool(finished)} finite_poses={positions_ok}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('events', type=Path)
    parser.add_argument('--expected-open-count', required=True, type=int)
    args = parser.parse_args()
    passed, detail = check(args.events, args.expected_open_count)
    print(('PASS ' if passed else 'FAIL ') + detail)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
