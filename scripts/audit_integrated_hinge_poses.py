#!/usr/bin/env python3
"""Read Gazebo poses for evidence; never publish a door or robot command."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def yaw(quaternion):
    x, y, z, w = (quaternion.get(k, 0. if k != 'w' else 1.) for k in ('x', 'y', 'z', 'w'))
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('world', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    world = ET.parse(args.world).find('world')
    topic = '/world/' + world.get('name') + '/pose/info'
    raw = subprocess.check_output(['ign', 'topic', '-e', '-n', '1', '-t', topic,
                                   '--json-output'], text=True, timeout=15)
    message = json.loads(raw)
    observed = {p['name']: p for p in message.get('pose', []) if 'name' in p}
    records = []
    for model in world.findall('model'):
        if not model.get('name', '').startswith('door_blue'):
            continue
        initial = [float(x) for x in model.findtext('pose', '0 0 0 0 0 0').split()]
        pose = observed.get(model.get('name'))
        angle = None
        if pose:
            difference = yaw(pose.get('orientation', {})) - initial[5]
            angle = math.degrees(math.atan2(math.sin(difference), math.cos(difference)))
        records.append({'model': model.get('name'), 'observed_rotation_deg': angle,
                        'pose': pose})
    report = {'wall_time_unix': time.time(), 'source_topic': topic,
              'gazebo_header': message.get('header'), 'doors': records,
              'scope': 'Independent Gazebo model-pose audit. Not a control input or contact-force test.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
