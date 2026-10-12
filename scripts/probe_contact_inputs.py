#!/usr/bin/env python3
"""Passive short sensor snapshot for contact handoff diagnosis."""
import json
import math
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from fire_robot_interfaces.msg import DoorInfo


rclpy.init()
node = Node('contact_input_probe')
values = {}


def handle(msg, name):
    p = msg.handle_position.point
    values[name] = {'xyz': [p.x, p.y, p.z], 'frame': msg.handle_position.header.frame_id,
        'stamp': msg.handle_position.header.stamp.sec + msg.handle_position.header.stamp.nanosec*1e-9,
        'method': msg.handle_detection_method, 'confidence': msg.handle_confidence}


def scan(msg):
    front = [v for i, v in enumerate(msg.ranges)
             if abs(msg.angle_min+i*msg.angle_increment) < .35
             and math.isfinite(v) and v > msg.range_min]
    values['front_scan'] = {'min': min(front) if front else None,
        'stamp': msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9}


for name in ('handle_observation', 'target_observation'):
    node.create_subscription(DoorInfo, '/contact/'+name, lambda m, n=name: handle(m, n), 10)
node.create_subscription(LaserScan, '/scan', scan, 5)
start = time.monotonic()
while time.monotonic()-start < 12:
    rclpy.spin_once(node, timeout_sec=.1)
print(json.dumps(values, indent=2))
node.destroy_node()
rclpy.shutdown()
