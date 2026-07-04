import copy
import math

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class InitialStaticMapNode(Node):
    """Latch the first usable SLAM/static map as a fixed navigation map.

    slam_toolbox may keep refining /map while the robot moves. For this
    project, the global wall/room structure should be a stable reference.
    This node captures an initial /map snapshot and republishes it on
    /initial_static_map with TRANSIENT_LOCAL durability.
    """

    def __init__(self):
        super().__init__('initial_static_map_node')

        self.declare_parameter('source_map_topic', '/map')
        self.declare_parameter('static_map_topic', '/initial_static_map')
        self.declare_parameter('occupied_threshold', 55)
        self.declare_parameter('min_occupied_cells', 80)
        self.declare_parameter('capture_delay_sec', 5.0)
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('lock_after_capture', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('pad_forward_m', 0.0)
        self.declare_parameter('pad_backward_m', 0.0)
        self.declare_parameter('pad_lateral_m', 0.0)
        self.declare_parameter('unknown_value', -1)

        self._source_topic = str(self.get_parameter('source_map_topic').value)
        self._static_topic = str(self.get_parameter('static_map_topic').value)
        self._occupied_threshold = int(
            self.get_parameter('occupied_threshold').value)
        self._min_occupied_cells = int(
            self.get_parameter('min_occupied_cells').value)
        self._capture_delay_sec = max(
            0.0, float(self.get_parameter('capture_delay_sec').value))
        self._lock_after_capture = bool(
            self.get_parameter('lock_after_capture').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._pad_forward_m = max(0.0, float(self.get_parameter('pad_forward_m').value))
        self._pad_backward_m = max(0.0, float(self.get_parameter('pad_backward_m').value))
        self._pad_lateral_m = max(0.0, float(self.get_parameter('pad_lateral_m').value))
        self._unknown_value = int(self.get_parameter('unknown_value').value)
        rate_hz = max(0.1, float(self.get_parameter('publish_rate_hz').value))

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._map_sub = self.create_subscription(
            OccupancyGrid, self._source_topic, self._map_callback, qos)
        self._map_pub = self.create_publisher(
            OccupancyGrid, self._static_topic, qos)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._first_usable_time = None
        self._captured_map: OccupancyGrid | None = None
        self._captured_once = False

        self.create_timer(1.0 / rate_hz, self._publish_static_map)
        self.get_logger().info(
            f'InitialStaticMapNode started | source={self._source_topic}, '
            f'static={self._static_topic}, delay={self._capture_delay_sec:.1f}s')

    def _map_callback(self, msg: OccupancyGrid):
        if self._captured_once and self._lock_after_capture:
            return
        if not self._is_usable_map(msg):
            self._first_usable_time = None
            return

        now = self.get_clock().now()
        if self._first_usable_time is None:
            self._first_usable_time = now
            self.get_logger().info(
                'First usable map received. Waiting before static snapshot...')
            return

        elapsed = (now - self._first_usable_time).nanoseconds / 1e9
        if elapsed < self._capture_delay_sec:
            return

        static_map = self._expanded_static_map(msg)
        self._captured_map = static_map
        self._captured_once = True
        occupied = self._occupied_count(static_map)
        self.get_logger().info(
            f'Captured initial static map: '
            f'{msg.info.width}x{msg.info.height}, '
            f'res={msg.info.resolution:.3f} -> '
            f'{static_map.info.width}x{static_map.info.height}, '
            f'origin=({static_map.info.origin.position.x:.2f}, '
            f'{static_map.info.origin.position.y:.2f}), '
            f'occupied={occupied}')

    def _publish_static_map(self):
        if self._captured_map is None:
            return
        msg = copy.deepcopy(self._captured_map)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._map_pub.publish(msg)

    def _expanded_static_map(self, msg: OccupancyGrid) -> OccupancyGrid:
        if (self._pad_forward_m <= 0.0
                and self._pad_backward_m <= 0.0
                and self._pad_lateral_m <= 0.0):
            return copy.deepcopy(msg)

        robot_pose = self._robot_pose()
        if robot_pose is None:
            self.get_logger().warn(
                'Could not get initial robot pose for static map padding. '
                'Using raw SLAM snapshot.')
            return copy.deepcopy(msg)

        res = float(msg.info.resolution)
        old_w = int(msg.info.width)
        old_h = int(msg.info.height)
        old_ox = float(msg.info.origin.position.x)
        old_oy = float(msg.info.origin.position.y)
        old_max_x = old_ox + old_w * res
        old_max_y = old_oy + old_h * res

        rx, ry, yaw = robot_pose
        fx = math.cos(yaw)
        fy = math.sin(yaw)
        lx = -math.sin(yaw)
        ly = math.cos(yaw)

        min_x = old_ox
        min_y = old_oy
        max_x = old_max_x
        max_y = old_max_y
        for forward in (-self._pad_backward_m, self._pad_forward_m):
            for lateral in (-self._pad_lateral_m, self._pad_lateral_m):
                x = rx + fx * forward + lx * lateral
                y = ry + fy * forward + ly * lateral
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        new_ox = old_ox - math.ceil(max(0.0, old_ox - min_x) / res) * res
        new_oy = old_oy - math.ceil(max(0.0, old_oy - min_y) / res) * res
        new_w = max(old_w, int(math.ceil((max_x - new_ox) / res)))
        new_h = max(old_h, int(math.ceil((max_y - new_oy) / res)))

        x_off = int(round((old_ox - new_ox) / res))
        y_off = int(round((old_oy - new_oy) / res))
        if x_off == 0 and y_off == 0 and new_w == old_w and new_h == old_h:
            return copy.deepcopy(msg)

        expanded = copy.deepcopy(msg)
        expanded.info.width = new_w
        expanded.info.height = new_h
        expanded.info.origin.position.x = new_ox
        expanded.info.origin.position.y = new_oy
        expanded.data = [self._unknown_value] * (new_w * new_h)

        for yy in range(old_h):
            old_start = yy * old_w
            old_end = old_start + old_w
            new_start = (yy + y_off) * new_w + x_off
            expanded.data[new_start:new_start + old_w] = msg.data[old_start:old_end]

        return expanded

    def _robot_pose(self) -> tuple[float, float, float] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, Time(),
                timeout=Duration(seconds=0.1))
        except TransformException:
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(t.x), float(t.y), yaw

    def _is_usable_map(self, msg: OccupancyGrid) -> bool:
        if msg.info.width == 0 or msg.info.height == 0:
            return False
        if len(msg.data) != msg.info.width * msg.info.height:
            return False
        return self._occupied_count(msg) >= self._min_occupied_cells

    def _occupied_count(self, msg: OccupancyGrid) -> int:
        threshold = self._occupied_threshold
        return sum(1 for value in msg.data if value >= threshold)


def main(args=None):
    rclpy.init(args=args)
    node = InitialStaticMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
