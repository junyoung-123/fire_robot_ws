#!/usr/bin/env python3
"""Run one isolated trial; only terminate process groups created by this runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from make_open_exit_world import generate


class ReadinessWatch:
    """Distinguish startup failure from a bounded, recoverable runtime outage."""

    def __init__(self):
        self.seen = set()
        self.missing_since = {}

    def check(self, elapsed, audit):
        for key, status in (
                ('navigate_action_server_present', 'NAV2'),
                ('map_messages_recorded', 'SLAM'),
                ('map_to_base_available', 'SLAM_TRANSFORM')):
            if audit.get(key, False):
                self.seen.add(key)
                self.missing_since.pop(key, None)
                continue
            since = self.missing_since.setdefault(key, elapsed)
            if key not in self.seen and elapsed > 180:
                return status + '_STARTUP_TIMEOUT'
            if key in self.seen and elapsed - since > 60:
                return status + '_RUNTIME_TIMEOUT'
        return None


def stop_group(process):
    for sig, wait in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        time.sleep(wait)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=1500)
    parser.add_argument('--open-exit', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ws = Path(__file__).resolve().parent.parent
    world_file = ws/'src/fire_robot_bringup/worlds'/args.world
    env = dict(os.environ, ROS_DOMAIN_ID=str(100 + os.getpid() % 100),
               IGN_PARTITION=f'observed_nav_{os.getpid()}', GZ_PARTITION=f'observed_nav_{os.getpid()}',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    manifest = {'world': args.world, 'scope': 'navigation_visits_only_no_physical_opening',
                'baseline_commit': '27359f7', 'prior_map': None, 'odometry': 'wheel',
                'localization': 'online_slam', 'world_sha256': hashlib.sha256(world_file.read_bytes()).hexdigest(),
                'ros_domain': env['ROS_DOMAIN_ID'], 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    manifest['transport_partition'] = env['IGN_PARTITION']
    manifest['evaluation_only_truth_topic'] = '/evaluation/ground_truth_odom'
    manifest['source_sha256'] = {
        str(path.relative_to(ws)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ws/'src').rglob('*'))
        if path.is_file() and path.suffix in ('.py', '.yaml', '.world', '.xacro')
    }
    manifest['harness_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ws/'scripts').glob('*observed_navigation*.py'))
    }
    launch_world = args.world
    if args.open_exit:
        generated_world = args.output.resolve()/'open_exit.world'
        manifest['exit_fixture'] = generate(world_file, generated_world)
        launch_world = str(generated_world)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    commands = [
        (['ros2', 'launch', 'fire_robot_bringup', 'observed_navigation.launch.py',
          'world:='+launch_world], 'simulation.log'),
        (['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
          '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry', '--ros-args',
          '-r', '__node:=evaluation_truth_bridge',
          '-r', '/odom:=/evaluation/ground_truth_odom'], 'evaluation_bridge.log'),
        (['python3', str(ws/'scripts/observed_navigation_recorder.py'),
          '--output', str(args.output/'evidence')], 'recorder.log'),
    ]
    processes, streams = [], []
    status = 'interrupted'
    start = time.monotonic()
    readiness = ReadinessWatch()
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for command, log in commands:
            stream = (args.output/log).open('w')
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=ws, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True))
        print(json.dumps({'output': str(args.output), 'ros_domain': env['ROS_DOMAIN_ID']}), flush=True)
        while time.monotonic()-start < args.timeout:
            audit_file = args.output/'evidence/input_audit.json'
            if audit_file.exists():
                try:
                    audit = json.loads(audit_file.read_text())
                except json.JSONDecodeError:
                    audit = None
                if audit is not None:
                    readiness_failure = readiness.check(time.monotonic()-start, audit)
                    if readiness_failure:
                        status = readiness_failure
                        break
            if (args.output/'stop_requested').exists():
                status = 'STOPPED_FOR_DIAGNOSIS'
                break
            state_file = args.output/'evidence/terminal_state.json'
            if state_file.exists():
                status = json.loads(state_file.read_text())['state']
                break
            if any(p.poll() is not None for p in processes):
                status = 'PROCESS_EXIT'
                break
            time.sleep(2)
        else:
            status = 'TIMEOUT'
    except KeyboardInterrupt:
        status = 'STOPPED_FOR_DIAGNOSIS'
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Stop actuation/simulation first; keep evidence subscribers alive until
        # the control process group has exited, including failure/timeouts.
        for process in processes:
            stop_group(process)
        for stream in streams:
            stream.close()
        result = {'status': status, 'wall_seconds': time.monotonic()-start,
                  'physical_opening_tested': False,
                  'note': 'MISSION_COMPLETE alone is not a validation PASS; review visits and trace.'}
        (args.output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
