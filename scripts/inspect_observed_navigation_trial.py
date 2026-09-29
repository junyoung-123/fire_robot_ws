#!/usr/bin/env python3
"""Read-only live diagnosis. Never publishes or changes an acceptance result."""
import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def rows(path):
    with path.open() as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def inspect(run):
    evidence = run/'evidence'
    events = []
    for line in (evidence/'events.jsonl').read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    truth = rows(evidence/'evaluation_poses.csv')
    estimated = rows(evidence/'poses.csv')
    states = [e['state'] for e in events if e['kind'] == 'state']
    data = dict(run=run.name, state=states[-1] if states else None,
                latest_estimate=estimated[-1] if estimated else None,
                latest_truth_evaluation_only=truth[-1] if truth else None, visits=[])
    doors = []
    for model in ET.parse(run/'open_exit.world').getroot().findall('./world/model'):
        if model.get('name', '').startswith('door_'):
            pose = list(map(float, model.findtext('pose').split()))
            doors.append((model.get('name'), pose))
    for event in events:
        if event['kind'] != 'visit' or not truth:
            continue
        t = event['payload']['sim_time']
        p = min(truth, key=lambda row: abs(row['sim_time']-t))
        name, pose = min(doors, key=lambda door: math.hypot(door[1][0]-p['x'], door[1][1]-p['y']))
        dx, dy = pose[0]-p['x'], pose[1]-p['y']
        normal = math.copysign(math.pi/2, dy)
        error = math.degrees(abs(math.atan2(math.sin(normal-p['yaw']), math.cos(normal-p['yaw']))))
        data['visits'].append(dict(t=t, door=name, distance=round(math.hypot(dx, dy), 3),
                                   longitudinal=round(abs(dx), 3), yaw_error_deg=round(error, 2),
                                   geometric_match=('blue' in name and abs(dx) <= .7 and
                                                    math.hypot(dx, dy) <= 1.6 and error <= 15)))
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    inspect(parser.parse_args().run)
