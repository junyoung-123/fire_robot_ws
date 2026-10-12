#!/usr/bin/env python3
"""Diagnostic fixture client; targets are copied only from measured DoorInfo."""
import argparse
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_interfaces.srv import OpenDoor


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--timeout',type=float,default=3600.)
args, ros_args = parser.parse_known_args()
rclpy.init(args=ros_args)
node = Node('observed_contact_probe_client')
client = node.create_client(OpenDoor, '/open_door')
latest = None


def observation(msg):
    global latest
    if (msg.door_color == 'blue' and msg.handle_detected and msg.handle_confidence >= .45
            and msg.handle_detection_method.startswith('yolo:primary')
            and msg.handle_detection_method.endswith(':registered_depth')):
        latest = msg


node.create_subscription(DoorInfo, '/detected_door', observation, 10)
started = time.monotonic()
future = None
result = {'success': False, 'message': 'No measured handle/service before diagnostic timeout'}
try:
    while time.monotonic()-started < args.timeout and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=.05)
        if future is None and latest is not None and client.service_is_ready():
            stamp = latest.handle_position.header.stamp
            measured = stamp.sec+stamp.nanosec*1e-9
            now = node.get_clock().now().nanoseconds*1e-9
            if not (-.1 < now-measured < .8):
                continue
            request = OpenDoor.Request()
            for key in ('door_id', 'handle_position', 'handle_detected',
                        'handle_detection_method', 'handle_confidence'):
                setattr(request, key, getattr(latest, key))
            future = client.call_async(request)
            print('Diagnostic request: ' + request.door_id, flush=True)
        if future is not None and future.done():
            response = future.result()
            result = {'success': response.success, 'message': response.message}
            break
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
finally:
    node.destroy_node()
    rclpy.shutdown()
