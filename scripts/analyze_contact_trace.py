#!/usr/bin/env python3
"""Offline evaluation: compare measured odometry to truth, never control."""
import argparse
import csv
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('evidence', type=Path)
args = parser.parse_args()
truth = list(csv.DictReader((args.evidence/'evaluation_poses.csv').open()))
events = [json.loads(line) for line in (args.evidence/'events.jsonl').open()]
wheels = [e for e in events if e['kind'] == 'wheel_odom']
samples = [e for e in events if e['kind'] == 'contact_feedback'
           and e['payload']['event'] == 'handle_held_base_sample']
if samples:
    first = samples[0]['sim_time']
    def nearest(rows, t, key):
        return min(rows, key=lambda r: abs(float(r[key])-t))
    base0 = nearest(wheels, first, 'sim_time')['pose'][2]
    truth0 = float(nearest(truth, first, 'sim_time')['yaw'])
    wheel_start = nearest(wheels, first, 'sim_time')['pose']
    truth_start = nearest(truth, first, 'sim_time')
    alignment = truth0-base0
    for e in samples[::max(1, len(samples)//15)]:
        t, p = e['sim_time'], e['payload']
        wheel = nearest(wheels, t, 'sim_time')['pose'][2]
        actual = float(nearest(truth, t, 'sim_time')['yaw'])
        measured_pose = nearest(wheels, t, 'sim_time')['pose']
        truth_pose = nearest(truth, t, 'sim_time')
        dx, dy = measured_pose[0]-wheel_start[0], measured_pose[1]-wheel_start[1]
        ex = math.cos(alignment)*dx-math.sin(alignment)*dy-float(truth_pose['x'])+float(truth_start['x'])
        ey = math.sin(alignment)*dx+math.cos(alignment)*dy-float(truth_pose['y'])+float(truth_start['y'])
        wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
        print(json.dumps({'t': t, 'door_deg': math.degrees(p['door_delta_rad']),
            'wheel_delta_deg': math.degrees(wrap(wheel-base0)),
            'truth_delta_deg': math.degrees(wrap(actual-truth0)),
            'yaw_error_deg': math.degrees(wrap(wheel-base0-actual+truth0)),
            'translation_error_m': math.hypot(ex,ey),
            'v': p['linear_x'], 'w': p['angular_z'],
            'hand': p['actual_tool_xyz'], 'target': p['preferred_tool_xyz']}))
