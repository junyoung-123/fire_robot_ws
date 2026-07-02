"""Final cmd_vel safety filter for simulation drive output."""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelSafetyNode(Node):
    """Clamp unsafe velocity commands before they reach Gazebo."""

    def __init__(self):
        super().__init__('cmd_vel_safety_node')

        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/cmd_vel_safe')
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('max_linear_x', 0.25)
        self.declare_parameter('max_angular_z', 1.0)

        self._allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self._max_linear_x = float(self.get_parameter('max_linear_x').value)
        self._max_angular_z = abs(float(self.get_parameter('max_angular_z').value))
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)

        self._pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Twist, input_topic, self._on_cmd_vel, 10)
        self._last_clamp_log_sec = 0.0

        self.get_logger().info(
            f'CmdVelSafetyNode started | {input_topic} -> {output_topic} '
            f'allow_reverse={self._allow_reverse}')

    def _on_cmd_vel(self, msg: Twist):
        safe = Twist()
        safe.linear.x = max(-self._max_linear_x, min(self._max_linear_x, msg.linear.x))
        if not self._allow_reverse and safe.linear.x < 0.0:
            self._log_reverse_clamp(msg.linear.x)
            safe.linear.x = 0.0
        safe.linear.y = 0.0
        safe.linear.z = 0.0
        safe.angular.x = 0.0
        safe.angular.y = 0.0
        safe.angular.z = max(-self._max_angular_z, min(self._max_angular_z, msg.angular.z))
        self._pub.publish(safe)

    def _log_reverse_clamp(self, original_x: float):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self._last_clamp_log_sec >= 1.0:
            self.get_logger().warn(
                f'Blocked reverse cmd_vel linear.x={original_x:.3f}')
            self._last_clamp_log_sec = now_sec


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
