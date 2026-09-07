"""Final cmd_vel safety filter for simulation drive output."""

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class CmdVelSafetyNode(Node):
    """Clamp unsafe velocity commands before they reach Gazebo."""

    def __init__(self):
        super().__init__('cmd_vel_safety_node')

        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('manual_input_topic', '/cmd_vel_manual')
        self.declare_parameter('output_topic', '/cmd_vel_safe')
        self.declare_parameter('manual_hold_sec', 0.75)
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('max_blocked_reverse_linear_x', 0.0)
        self.declare_parameter('max_linear_x', 0.25)
        self.declare_parameter('max_angular_z', 1.0)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('scan_timeout_sec', 0.75)
        self.declare_parameter('front_stop_distance_m', 0.48)
        self.declare_parameter('rear_stop_distance_m', 0.0)
        self.declare_parameter('rotation_stop_distance_m', 0.0)
        self.declare_parameter('drive_stop_half_angle_deg', 35.0)
        self.declare_parameter('lidar_stop_manual_commands', False)

        self._allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self._max_blocked_reverse_linear_x = abs(float(
            self.get_parameter('max_blocked_reverse_linear_x').value))
        self._max_linear_x = float(self.get_parameter('max_linear_x').value)
        self._max_angular_z = abs(float(self.get_parameter('max_angular_z').value))
        self._scan_timeout_sec = max(
            0.0, float(self.get_parameter('scan_timeout_sec').value))
        self._front_stop_distance_m = max(
            0.0, float(self.get_parameter('front_stop_distance_m').value))
        self._rear_stop_distance_m = max(
            0.0, float(self.get_parameter('rear_stop_distance_m').value))
        self._rotation_stop_distance_m = max(
            0.0, float(self.get_parameter('rotation_stop_distance_m').value))
        self._lidar_stop_manual_commands = bool(
            self.get_parameter('lidar_stop_manual_commands').value)
        self._drive_stop_half_angle = math.radians(max(
            1.0, float(self.get_parameter('drive_stop_half_angle_deg').value)))
        input_topic = str(self.get_parameter('input_topic').value)
        manual_input_topic = str(self.get_parameter('manual_input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        scan_topic = str(self.get_parameter('scan_topic').value)
        self._manual_hold_sec = max(
            0.0, float(self.get_parameter('manual_hold_sec').value))
        self._manual_active_until_sec = 0.0

        self._pub = self.create_publisher(Twist, output_topic, 10)
        self.create_subscription(Twist, input_topic, self._on_nav_cmd_vel, 10)
        self.create_subscription(
            Twist, manual_input_topic, self._on_manual_cmd_vel, 10)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self._latest_scan: LaserScan | None = None
        self._latest_scan_time_sec = 0.0
        self._last_clamp_log_sec = 0.0
        self._last_obstacle_clamp_log_sec = 0.0

        self.get_logger().info(
            f'CmdVelSafetyNode started | nav={input_topic}, '
            f'manual={manual_input_topic} -> {output_topic} '
            f'manual_hold={self._manual_hold_sec:.2f}s, '
            f'allow_reverse={self._allow_reverse}, '
            f'micro_reverse={self._max_blocked_reverse_linear_x:.3f}m/s, '
            f'LiDAR stops(front/rear/turn)='
            f'{self._front_stop_distance_m:.2f}/'
            f'{self._rear_stop_distance_m:.2f}/'
            f'{self._rotation_stop_distance_m:.2f}m, '
            f'manual_lidar_stop={self._lidar_stop_manual_commands}')

    def _on_scan(self, msg: LaserScan):
        self._latest_scan = msg
        self._latest_scan_time_sec = self._now_sec()

    def _on_nav_cmd_vel(self, msg: Twist):
        if self._now_sec() <= self._manual_active_until_sec:
            return
        self._publish_safe(msg, apply_lidar_stop=True)

    def _on_manual_cmd_vel(self, msg: Twist):
        self._manual_active_until_sec = self._now_sec() + self._manual_hold_sec
        self._publish_safe(
            msg, apply_lidar_stop=self._lidar_stop_manual_commands)

    def _publish_safe(self, msg: Twist, *, apply_lidar_stop: bool):
        safe = Twist()
        safe.linear.x = max(-self._max_linear_x, min(self._max_linear_x, msg.linear.x))
        if not self._allow_reverse and safe.linear.x < 0.0:
            micro_reverse = min(safe.linear.x, 0.0)
            if abs(micro_reverse) <= self._max_blocked_reverse_linear_x:
                safe.linear.x = micro_reverse
            else:
                self._log_reverse_clamp(msg.linear.x)
                safe.linear.x = 0.0
        safe.linear.y = 0.0
        safe.linear.z = 0.0
        safe.angular.x = 0.0
        safe.angular.y = 0.0
        safe.angular.z = max(-self._max_angular_z, min(self._max_angular_z, msg.angular.z))
        if apply_lidar_stop:
            self._apply_lidar_collision_stop(safe)
        self._pub.publish(safe)

    def _apply_lidar_collision_stop(self, safe: Twist):
        scan = self._latest_scan
        if scan is None:
            return
        if (
                self._scan_timeout_sec > 0.0
                and self._now_sec() - self._latest_scan_time_sec
                > self._scan_timeout_sec):
            return

        front = self._sector_min(
            scan, -self._drive_stop_half_angle, self._drive_stop_half_angle)
        rear = min(
            self._sector_min(scan, math.pi - self._drive_stop_half_angle, math.pi),
            self._sector_min(scan, -math.pi, -math.pi + self._drive_stop_half_angle))
        surround = self._sector_min(scan, -math.pi, math.pi)
        blocked: list[str] = []
        if safe.linear.x > 0.0 and front < self._front_stop_distance_m:
            safe.linear.x = 0.0
            blocked.append(f'front={front:.2f}m')
        elif safe.linear.x < 0.0 and rear < self._rear_stop_distance_m:
            safe.linear.x = 0.0
            blocked.append(f'rear={rear:.2f}m')
        if (
                abs(safe.angular.z) > 1e-3
                and surround < self._rotation_stop_distance_m):
            safe.angular.z = 0.0
            blocked.append(f'turn_clearance={surround:.2f}m')
        if blocked:
            self._log_obstacle_clamp(', '.join(blocked))

    @staticmethod
    def _sector_min(scan: LaserScan, start_angle: float, end_angle: float) -> float:
        minimum = float('inf')
        angle = float(scan.angle_min)
        for distance in scan.ranges:
            if start_angle <= angle <= end_angle and math.isfinite(distance):
                if scan.range_min <= distance <= scan.range_max:
                    minimum = min(minimum, float(distance))
            angle += float(scan.angle_increment)
        return minimum

    def _log_obstacle_clamp(self, reason: str):
        now_sec = self._now_sec()
        if now_sec - self._last_obstacle_clamp_log_sec >= 1.0:
            self.get_logger().warn(f'LiDAR collision stop: {reason}')
            self._last_obstacle_clamp_log_sec = now_sec

    def _log_reverse_clamp(self, original_x: float):
        now_sec = self._now_sec()
        if now_sec - self._last_clamp_log_sec >= 1.0:
            self.get_logger().warn(
                f'Blocked reverse cmd_vel linear.x={original_x:.3f}')
            self._last_clamp_log_sec = now_sec

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
