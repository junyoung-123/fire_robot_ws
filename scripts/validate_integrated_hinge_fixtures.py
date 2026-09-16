#!/usr/bin/env python3
"""Isolated Gazebo fixture test. Ground-truth commands are NOT autonomous goals."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from fire_robot_manipulation.sim_door_feedback import model_metadata, rotation_sample

WORLDS = ('obstacle_wall_doors_v5.world', 'obstacle_door_layout_alt_v1.world',
          'obstacle_door_layout_world3_v1.world', 'obstacle_all_blue_world5_v1.world')


def pose_message(world):
    return json.loads(subprocess.check_output([
        'ign', 'topic', '-e', '-n', '1', '--json-output',
        '-t', '/world/' + world + '/pose/info'], text=True, timeout=4))


def command(topic, angle):
    subprocess.run(['ign', 'topic', '-t', topic, '-m', 'ignition.msgs.Double',
                    '-p', f'data: {angle:.7f}'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)


def run(world_path, output):
    metadata = model_metadata(world_path)
    world = next(iter(metadata.values()))['world']
    record = dict(world=world_path.name, world_sha256=hashlib.sha256(world_path.read_bytes()).hexdigest(),
                  scope='Hinge-command fixture test, not autonomous navigation or PIPER force opening.',
                  samples=[], passed=False)
    with (output / (world_path.stem + '.log')).open('w') as log:
        process = subprocess.Popen(['ign', 'gazebo', '-r', '-s', str(world_path.resolve())],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   env={**os.environ, 'LIBGL_ALWAYS_SOFTWARE': '1',
                                        'MESA_GL_VERSION_OVERRIDE': '3.3'})
        try:
            deadline = time.monotonic() + 25.
            while time.monotonic() < deadline:
                try:
                    message = pose_message(world)
                    for item in metadata.values():
                        rotation_sample(message, item['model'], item['yaw'])
                    break
                except (subprocess.SubprocessError, ValueError, KeyError):
                    time.sleep(.2)
            else:
                raise RuntimeError('Gazebo fixture not ready')
            targets = {topic: (1 if '/yp' in topic else -1) * 2.09439510239 for topic in metadata}
            record['targets'] = targets
            record['initial'] = {topic: rotation_sample(message, item['model'], item['yaw'])[1]
                                 for topic, item in metadata.items()}
            deadline, next_command, stable = time.monotonic() + 25., 0., None
            while time.monotonic() < deadline:
                if time.monotonic() >= next_command:
                    for topic, angle in targets.items():
                        command(topic, angle)
                    next_command = time.monotonic() + 1.
                message = pose_message(world)
                measured = {topic: rotation_sample(message, item['model'], item['yaw'])
                            for topic, item in metadata.items()}
                record['samples'].append(measured)
                reached = all(abs(value[1]-targets[topic]) <= .04 for topic,value in measured.items())
                stamp = min(value[0] for value in measured.values())
                if reached:
                    stable = stamp if stable is None else stable
                    if stamp - stable >= .2:
                        record['passed'] = True
                        break
                else:
                    stable = None
                time.sleep(.2)
        except Exception as exc:
            record['error'] = str(exc)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    (output / (world_path.stem + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    final = record['samples'][-1] if record['samples'] else {}
    print(world_path.name, 'PASS' if record['passed'] else 'FAIL',
          {k: round(math.degrees(v[1]), 2) for k,v in final.items()}, flush=True)
    return record['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in WORLDS:
        if not run(Path('src/fire_robot_bringup/worlds') / name, args.output):
            raise SystemExit(1)
