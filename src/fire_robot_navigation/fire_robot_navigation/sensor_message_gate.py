"""Bound ROS feedback traffic without changing simulation physics or stamps."""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


class SensorMessageGate(Node):
    def __init__(self):
        super().__init__('sensor_message_gate')
        self.last = {}
        self.clock_pub = self.create_publisher(Clock, '/clock', qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(Clock, '/clock_raw', self.clock, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states_raw', self.joints, qos_profile_sensor_data)

    def accept(self, kind, stamp):
        current = stamp.sec*1000000000+stamp.nanosec
        previous = self.last.get(kind)
        if previous is not None and 0 <= current-previous < 20000000:
            return False
        self.last[kind] = current
        return True

    def clock(self, msg):
        if self.accept('clock', msg.clock):
            self.clock_pub.publish(msg)

    def joints(self, msg):
        if self.accept('joints', msg.header.stamp):
            self.joint_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorMessageGate()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
