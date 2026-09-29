#!/usr/bin/env python3
"""Passive evidence recorder. Does not publish commands or simulator truth."""
import argparse
import csv
import json
import math
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from fire_robot_interfaces.msg import DoorInfo, RobotState
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException


class Recorder(Node):
    def __init__(self, directory):
        super().__init__('observed_navigation_recorder', parameter_overrides=[
            Parameter('use_sim_time', value=True)])
        self.out = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.bridge = CvBridge()
        self.last_map = -1e9
        self.last_camera = -1e9
        self.map_count = 0
        self.camera_count = 0
        self.start = time.monotonic()
        self.csv_file = (directory / 'poses.csv').open('w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['sim_time', 'wall_elapsed', 'x', 'y', 'yaw'])
        self.events = (directory / 'events.jsonl').open('w', buffering=1)
        self.truth_file = (directory / 'evaluation_poses.csv').open('w', newline='')
        self.truth_writer = csv.writer(self.truth_file)
        self.truth_writer.writerow(['sim_time', 'x', 'y', 'yaw'])
        self.last_truth = -1e9
        self.last_scan = -1e9
        self.last_costmap = -1e9
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, qos)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.costmap_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.camera_cb, qos_profile_sensor_data)
        self.create_subscription(RobotState, '/robot_state', self.state_cb, 10)
        self.create_subscription(Odometry, '/evaluation/ground_truth_odom', self.truth_cb, 10)
        self.create_subscription(String, '/navigation_visit', self.visit_cb, 10)
        self.create_subscription(DoorInfo, '/detected_door', lambda m: self.door_cb(m, 'detection'), 10)
        self.create_subscription(DoorInfo, '/target_door', lambda m: self.door_cb(m, 'target'), 10)
        self.create_timer(0.5, self.pose_cb)
        self.create_timer(15.0, self.audit)

    def seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def record(self, kind, **data):
        self.events.write(json.dumps({'kind': kind, 'sim_time': self.seconds(), **data}) + '\n')

    def pose_cb(self):
        try:
            tf = self.buffer.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return
        p, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.writer.writerow([self.seconds(), time.monotonic()-self.start, p.x, p.y, yaw])
        self.csv_file.flush()

    def map_cb(self, msg):
        if self.seconds()-self.last_map < 20:
            return
        self.last_map = self.seconds()
        data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        name = f'map_{self.map_count:03d}.npz'
        np.savez_compressed(self.out/name, data=data, resolution=msg.info.resolution,
                            origin=[msg.info.origin.position.x, msg.info.origin.position.y],
                            sim_time=self.seconds())
        self.record('map', file=name, known=int((data >= 0).sum()),
                    occupied=int((data >= 65).sum()), width=msg.info.width, height=msg.info.height)
        self.map_count += 1

    def truth_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if t-self.last_truth < 0.1:
            return
        self.last_truth = t
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.truth_writer.writerow([t, p.x, p.y, yaw])
        self.truth_file.flush()

    def scan_cb(self, msg):
        if self.seconds()-self.last_scan < 5:
            return
        self.last_scan = self.seconds()
        saved = dict(ranges=msg.ranges,
                            angle_min=msg.angle_min, angle_increment=msg.angle_increment,
                            range_min=msg.range_min, range_max=msg.range_max,
                            frame=msg.header.frame_id, sim_time=self.seconds())
        try:
            tf = self.buffer.lookup_transform('map', msg.header.frame_id, Time.from_msg(msg.header.stamp))
            p, q = tf.transform.translation, tf.transform.rotation
            saved['sensor_pose'] = [p.x, p.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]
        except TransformException:
            pass
        np.savez_compressed(self.out/'scan_latest.npz', **saved)
        np.savez_compressed(self.out/f'scan_{int(self.seconds()*1000):08d}.npz', **saved)

    def costmap_cb(self, msg):
        if self.seconds()-self.last_costmap < 10:
            return
        self.last_costmap = self.seconds()
        np.savez_compressed(self.out/'costmap_latest.npz',
                            data=np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width),
                            resolution=msg.info.resolution,
                            origin=[msg.info.origin.position.x, msg.info.origin.position.y],
                            sim_time=self.seconds())

    def odom_cb(self, msg):
        if not hasattr(self, 'last_odom') or self.seconds()-self.last_odom >= 2:
            self.last_odom = self.seconds()
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            self.record('wheel_odom', pose=[p.x, p.y, yaw],
                        velocity=[msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def camera_cb(self, msg):
        if self.seconds()-self.last_camera < 30:
            return
        self.last_camera = self.seconds()
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        name = f'camera_{self.camera_count:03d}.jpg'
        cv2.imwrite(str(self.out/name), frame)
        self.record('camera', file=name)
        self.camera_count += 1

    def door_cb(self, msg, kind):
        p = msg.door_pose.pose.position
        h = msg.handle_position.point
        self.record(kind, id=msg.door_id, color=msg.door_color,
                    frame=msg.door_pose.header.frame_id, xy=[p.x, p.y],
                    handle=[h.x, h.y], confidence=float(msg.confidence))

    def visit_cb(self, msg):
        self.record('visit', payload=json.loads(msg.data))

    def state_cb(self, msg):
        state = {getattr(RobotState, name): name for name in (
            'IDLE', 'EXPLORING', 'NAVIGATING', 'OPENING_DOOR', 'DOOR_OPENED',
            'EXITING', 'MISSION_COMPLETE', 'EMERGENCY_STOP')}.get(msg.state, str(msg.state))
        self.record('state', state=state)
        if state in ('MISSION_COMPLETE', 'EMERGENCY_STOP'):
            (self.out/'terminal_state.json').write_text(json.dumps({'state': state}))

    def audit(self):
        graph = []
        for name, ns in self.get_node_names_and_namespaces():
            try:
                topics = self.get_subscriber_names_and_types_by_node(name, ns)
            except Exception:
                topics = []
            graph.append({'node': name, 'namespace': ns, 'subscriptions': topics})
        (self.out/'ros_graph.json').write_text(json.dumps(graph, indent=2))
        audit = {
            'sim_time': self.seconds(),
            'map_messages_recorded': self.map_count,
            'map_to_base_available': self.buffer.can_transform('map', 'base_link', Time()),
            'navigate_action_server_present': bool(self.get_publishers_info_by_topic('/navigate_to_pose/_action/status')),
            'map_publishers': sorted(info.node_name for info in self.get_publishers_info_by_topic('/map')),
            'odom_publishers': sorted(info.node_name for info in self.get_publishers_info_by_topic('/odom')),
            'evaluation_truth_subscribers': sorted(
                info.node_name for info in self.get_subscriptions_info_by_topic('/evaluation/ground_truth_odom')),
        }
        (self.out/'input_audit.json').write_text(json.dumps(audit, indent=2))
        self.record('input_audit', audit=audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Recorder(args.output)
    stop = False

    def interrupted(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.csv_file.close()
        node.truth_file.close()
        node.events.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
