from pathlib import Path
from collections import deque
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from .self_scan_geometry import horizontal_sections, expected_echo, mask_echoes, mask_confirmed_self_noise


class RobotSelfScanFilter(Node):
    def __init__(self):
        super().__init__('robot_self_scan_filter')
        mesh = Path(get_package_share_directory('fire_robot_description'))/'meshes/roas2/BASE_FRAME.stl'
        # The section follows the supplied URDF static frame and LiDAR mount.
        self.sections = horizontal_sections(mesh, .248-(-.065))
        self.expected = None
        self.signature = None
        self.history = deque(maxlen=11)
        self.last_stamp = None
        self.pub = self.create_publisher(LaserScan, '/scan', 10)
        self.create_subscription(LaserScan, '/scan_raw', self.scan, qos_profile_sensor_data)
        self.get_logger().info(f'Robot CAD self-filter: {len(self.sections)} horizontal surface segments')

    def scan(self, msg):
        signature = (len(msg.ranges), msg.angle_min, msg.angle_increment)
        if signature != self.signature:
            angles = msg.angle_min+np.arange(len(msg.ranges))*msg.angle_increment
            self.expected = expected_echo(self.sections, angles, (.25, 0), msg.range_min)
            self.signature = signature
            self.history.clear()
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1.e-9
        if self.last_stamp is not None and stamp<=self.last_stamp:
            self.history.clear()
        self.last_stamp=stamp
        self.history.append(np.asarray(msg.ranges,float))
        cleaned, _ = mask_echoes(msg.ranges, self.expected)
        cleaned=mask_confirmed_self_noise(cleaned,self.expected,list(self.history))
        msg.ranges = cleaned.tolist()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotSelfScanFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
