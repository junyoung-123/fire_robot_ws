"""Motion-gate SLAM scans using wheel odometry, preserving raw safety scans."""
from collections import deque
import math

from nav_msgs.msg import OccupancyGrid, Odometry
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class MotionGate:
    def __init__(self, translation=.03, rotation=.035):
        self.translation = translation
        self.rotation = rotation
        self.previous = None

    def accept(self, pose):
        if not all(math.isfinite(v) for v in pose):
            return False
        if self.previous is not None:
            dx, dy = pose[0]-self.previous[0], pose[1]-self.previous[1]
            yaw = math.atan2(math.sin(pose[2]-self.previous[2]),
                             math.cos(pose[2]-self.previous[2]))
            if math.hypot(dx, dy) < self.translation and abs(yaw) < self.rotation:
                return False
        self.previous = pose
        return True


class MotionScanFilter(Node):
    def __init__(self):
        super().__init__('motion_scan_filter')
        self.gate = MotionGate()
        self.odometry = deque(maxlen=200)
        self.last_scan_time = None
        self.map_ready = False
        self.tf_ready = False
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.publisher = self.create_publisher(LaserScan, '/slam_scan', qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 20)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/map', self.map_callback,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def map_callback(self, msg):
        if msg.info.width and msg.info.height:
            self.map_ready = True

    @staticmethod
    def stamp(header):
        return header.stamp.sec+header.stamp.nanosec*1e-9

    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        self.odometry.append((self.stamp(msg.header), (p.x, p.y, yaw)))

    def scan_callback(self, msg):
        stamp = self.stamp(msg.header)
        if self.last_scan_time is not None and stamp <= self.last_scan_time:
            return
        if not self.odometry or self.publisher.get_subscription_count() == 0:
            return
        odom_time, pose = min(self.odometry, key=lambda p: abs(p[0]-stamp))
        if abs(odom_time-stamp) > .15:
            return
        self.tf_ready = False
        if self.map_ready:
            try:
                transform = self.buffer.lookup_transform('map', 'odom', Time())
                age = stamp-self.stamp(transform.header)
                self.tf_ready = -.5 <= age <= .35
            except TransformException:
                pass
        # A map can arrive before the first pose has reached the TF relay.
        # Resume unchanged scans when TF expires too: motion can be stopped by
        # stale localization, so waiting for motion would prevent recovery.
        initial = not (self.map_ready and self.tf_ready)
        if initial and self.last_scan_time is not None and stamp-self.last_scan_time < .25:
            return
        if initial or self.gate.accept(pose):
            if initial:
                self.gate.previous = pose
            self.publisher.publish(msg)
            self.last_scan_time = stamp


def main(args=None):
    rclpy.init(args=args)
    node = MotionScanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
