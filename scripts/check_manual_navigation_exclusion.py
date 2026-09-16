#!/usr/bin/env python3
"""Detect newly dispatched Nav2 goals during an exclusive FSM escape interval."""
import argparse
import re
from pathlib import Path

STAMP = re.compile(r'\[(\d+\.\d+)\]')
START = re.compile(r'Local obstacle escape \d+/\d+:')


def check_text(text):
    active_since = None
    violations = []
    intervals = 0
    for number, line in enumerate(text.splitlines(), 1):
        match = STAMP.search(line)
        if not match:
            continue
        timestamp = float(match.group(1))
        if '[state_machine_node]' in line and START.search(line):
            active_since = timestamp
            intervals += 1
        elif '[state_machine_node]' in line and 'Local obstacle escape finished' in line:
            active_since = None
        elif (active_since is not None and '[bt_navigator]' in line
              and 'Begin navigating from current location' in line
              and timestamp-active_since > .5):
            # Allow delivery of an already-in-flight request while cancel starts.
            violations.append(dict(line=number, seconds_into_escape=timestamp-active_since))
    return dict(intervals=intervals, new_goals_during_escape=violations,
                unfinished_escape=active_since is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    result = check_text(args.log.read_text(errors='replace'))
    passed = not result['new_goals_during_escape']
    print(('PASS ' if passed else 'FAIL ') + str(result))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
