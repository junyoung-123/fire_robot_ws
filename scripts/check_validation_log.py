#!/usr/bin/env python3
"""Check a headless validation log against mission-level expectations."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


MISSION_RE = re.compile(r'MISSION_COMPLETE|State: MISSION_COMPLETE|미션 완료')
OPEN_RE = re.compile(r'문 개방 성공')
MATCH_RE = re.compile(
    r'Matched observed door .* to (?P<topic>/fire_robot/door/blue/[^\s]+)')
REJECT_RE = re.compile(r'Rejected Gazebo door match')
EMERGENCY_RE = re.compile(r'EMERGENCY_STOP|Traceback|IndexError|process has died')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', type=Path)
    parser.add_argument('--expected-open-count', type=int, required=True)
    parser.add_argument('--max-rejected', type=int, default=0)
    parser.add_argument('--allow-extra-open', action='store_true')
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f'FAIL log_not_found path={args.log_file}')
        return 2

    text = args.log_file.read_text(encoding='utf-8', errors='ignore')
    mission_complete = bool(MISSION_RE.search(text))
    emergency = bool(EMERGENCY_RE.search(text))
    open_count = len(OPEN_RE.findall(text))
    rejected_count = len(REJECT_RE.findall(text))
    matched_topics = MATCH_RE.findall(text)
    unique_topics = sorted(set(matched_topics))
    matched_count = len(unique_topics)

    failures: list[str] = []
    if not mission_complete:
        failures.append('mission_complete_missing')
    if emergency:
        failures.append('emergency_or_crash')
    if args.allow_extra_open:
        if matched_count < args.expected_open_count:
            failures.append(
                f'matched_blue_doors={matched_count} < expected={args.expected_open_count}')
    elif matched_count != args.expected_open_count:
        failures.append(
            f'matched_blue_doors={matched_count} != expected={args.expected_open_count}')
    if open_count < args.expected_open_count:
        failures.append(
            f'open_success_count={open_count} < expected={args.expected_open_count}')
    if rejected_count > args.max_rejected:
        failures.append(f'rejected_gazebo_matches={rejected_count} > max={args.max_rejected}')

    status = 'PASS' if not failures else 'FAIL'
    print(
        f'{status} mission_complete={mission_complete} '
        f'open_success_count={open_count} matched_blue_doors={matched_count} '
        f'rejected_gazebo_matches={rejected_count}')
    if unique_topics:
        print('matched_topics=' + ','.join(unique_topics))
    if failures:
        print('failures=' + ','.join(failures))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
