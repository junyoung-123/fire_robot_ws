#!/usr/bin/env python3
"""Free-space actuator calibration only; never a door-opening PASS."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/fire_robot_bringup/scripts'))
from run_physical_contact_door_test import _stop_launch
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_NAMES


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = Node('free_space_gripper_probe')
    latest = {}
    samples = []
    stage = ['startup']
    def receive(msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.e-9
        latest.update(zip(msg.name, msg.position))
        latest['time'] = stamp
        if samples and stamp - samples[-1]['sim_time'] < .1:
            return
        if all(name in latest for name in ('gripper_left_joint', 'gripper_right_joint')):
            samples.append(dict(sim_time=stamp, stage=stage[0],
                                left=latest['gripper_left_joint'], right=latest['gripper_right_joint']))
    node.create_subscription(JointState, '/joint_states', receive, 10)
    pubs = {name: node.create_publisher(Float64, f'/fire_robot/sim_arm/{name}_cmd', 10)
            for name in (*PIPER_JOINT_NAMES, 'gripper_left_joint', 'gripper_right_joint')}
    log = (output / 'launch.log').open('w')
    proc = subprocess.Popen([
        'ros2', 'launch', 'fire_robot_bringup', 'physical_contact_door_test.launch.py',
        'headless:=true', 'use_rviz:=false', 'enable_perception:=false',
        'require_yolo_handle:=true', 'feedback_contact_enabled:=true'],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report = {'scope': 'Free-space gripper calibration, not a door test', 'stages': []}
    try:
        deadline = time.monotonic() + 60.
        while 'time' not in latest and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
        if 'time' not in latest:
            raise RuntimeError('No joint feedback')
        # This is a robot-local calibration pose clear of the panel, not a
        # supplied handle/world coordinate and not an autonomous mission.
        kin = PiperActualKinematics()
        pose = kin.solve([.50, .15, .85], position_tolerance_m=.003)
        if pose is None:
            raise RuntimeError('Calibration pose is not reachable')
        for name, width, duration in (('roll_and_open', .07, 8.), ('close_free', 0., 4.), ('open_free', .07, 5.)):
            stage[0] = name
            start = latest['time']
            deadline = time.monotonic() + 60.
            last_publish = 0.
            while latest['time'] - start < duration:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Calibration clock stalled')
                rclpy.spin_once(node, timeout_sec=.02)
                if time.monotonic()-last_publish >= .05:
                    for joint, value in zip(PIPER_JOINT_NAMES, pose.positions):
                        pubs[joint].publish(Float64(data=float(value)))
                    pubs['gripper_left_joint'].publish(Float64(data=width/2.))
                    pubs['gripper_right_joint'].publish(Float64(data=-width/2.))
                    last_publish = time.monotonic()
            measured = np.asarray([latest[j] for j in PIPER_JOINT_NAMES])
            point, rotation = kin.forward(measured)
            report['stages'].append(dict(stage=name, requested_width=width,
                left=latest['gripper_left_joint'], right=latest['gripper_right_joint'],
                tool_xyz=point.tolist(), tool_rotation=rotation.tolist()))
        report['samples'] = samples
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _stop_launch(proc)
        log.close()
        (output / 'gripper_result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'samples'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
