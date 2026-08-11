import copy
import math

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class FixedObstacleMapNode(Node):
    """Accumulate fixed obstacle observations into a latched map overlay.

    Demo obstacles are treated as static structure. The global planner can use
    this overlay as a StaticLayer while the local costmap still uses live scan
    data for close-range collision checking.
    """

    def __init__(self):
        super().__init__('fixed_obstacle_map_node')

        self.declare_parameter('static_map_topic', '/initial_static_map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('obstacle_map_topic', '/fixed_obstacle_map')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_range_m', 0.85)
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('mark_radius_m', 0.16)
        self.declare_parameter('hit_count_threshold', 2)
        self.declare_parameter('robot_clear_radius_m', 0.75)
        self.declare_parameter('center_y_m', 0.0)
        self.declare_parameter('max_abs_y_m', 0.0)
        self.declare_parameter('occupied_value', 100)
        self.declare_parameter('unknown_value', 0)
        self.declare_parameter('publish_rate_hz', 1.0)

        self._static_map_topic = str(
            self.get_parameter('static_map_topic').value)
        self._scan_topic = str(self.get_parameter('scan_topic').value)
        self._obstacle_map_topic = str(
            self.get_parameter('obstacle_map_topic').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._min_range_m = max(
            0.0, float(self.get_parameter('min_range_m').value))
        self._max_range_m = max(
            self._min_range_m, float(self.get_parameter('max_range_m').value))
        self._mark_radius_m = max(
            0.0, float(self.get_parameter('mark_radius_m').value))
        self._hit_count_threshold = max(
            1, int(self.get_parameter('hit_count_threshold').value))
        self._robot_clear_radius_m = max(
            0.0, float(self.get_parameter('robot_clear_radius_m').value))
        self._center_y_m = float(self.get_parameter('center_y_m').value)
        self._max_abs_y_m = max(
            0.0, float(self.get_parameter('max_abs_y_m').value))
        self._occupied_value = int(self.get_parameter('occupied_value').value)
        self._unknown_value = int(self.get_parameter('unknown_value').value)
        rate_hz = max(
            0.2, float(self.get_parameter('publish_rate_hz').value))

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._map_sub = self.create_subscription(
            OccupancyGrid, self._static_map_topic, self._map_callback, map_qos)
        self._scan_sub = self.create_subscription(
            LaserScan, self._scan_topic, self._scan_callback,
            qos_profile_sensor_data)
        self._map_pub = self.create_publisher(
            OccupancyGrid, self._obstacle_map_topic, map_qos)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._template_map: OccupancyGrid | None = None
        self._obstacle_map: OccupancyGrid | None = None
        self._hit_counts: list[int] = []
        self._mark_radius_cells = 1
        self._last_occupied_count = 0

        self.create_timer(1.0 / rate_hz, self._publish_obstacle_map)
        self.get_logger().info(
            f'FixedObstacleMapNode started | static={self._static_map_topic}, '
            f'scan={self._scan_topic}, output={self._obstacle_map_topic}')

    def _map_callback(self, msg: OccupancyGrid):
        if msg.info.width == 0 or msg.info.height == 0:
            return
        if len(msg.data) != msg.info.width * msg.info.height:
            return

        if self._same_geometry(msg):
            self._template_map = copy.deepcopy(msg)
            return

        self._template_map = copy.deepcopy(msg)
        self._obstacle_map = copy.deepcopy(msg)
        self._obstacle_map.header.frame_id = self._map_frame
        self._obstacle_map.data = [
            self._unknown_value
            for _ in range(msg.info.width * msg.info.height)
        ]
        self._hit_counts = [0 for _ in self._obstacle_map.data]
        resolution = max(1.0e-6, float(msg.info.resolution))
        self._mark_radius_cells = max(
            1, int(math.ceil(self._mark_radius_m / resolution)))
        self._last_occupied_count = 0
        self.get_logger().info(
            f'Fixed obstacle overlay initialized: '
            f'{msg.info.width}x{msg.info.height}, '
            f'res={msg.info.resolution:.3f}, '
            f'origin=({msg.info.origin.position.x:.2f}, '
            f'{msg.info.origin.position.y:.2f}), '
            f'mark_radius={self._mark_radius_cells} cells')

    def _same_geometry(self, msg: OccupancyGrid) -> bool:
        if self._obstacle_map is None:
            return False
        info = self._obstacle_map.info
        return (
            info.width == msg.info.width
            and info.height == msg.info.height
            and abs(info.resolution - msg.info.resolution) < 1.0e-9
            and abs(info.origin.position.x - msg.info.origin.position.x) < 1.0e-9
            and abs(info.origin.position.y - msg.info.origin.position.y) < 1.0e-9
        )

    def _scan_callback(self, msg: LaserScan):
        if self._obstacle_map is None:
            return

        scan_pose = self._lookup_pose(msg.header.frame_id, msg.header.stamp)
        if scan_pose is None:
            return
        robot_pose = self._lookup_pose(self._base_frame, msg.header.stamp)

        sx, sy, syaw = scan_pose
        cyaw = math.cos(syaw)
        syaw_sin = math.sin(syaw)
        angle = float(msg.angle_min)
        min_range = max(self._min_range_m, float(msg.range_min))
        max_range = min(self._max_range_m, float(msg.range_max))
        updates = 0

        for distance in msg.ranges:
            if math.isfinite(distance) and min_range <= distance <= max_range:
                lx = math.cos(angle) * distance
                ly = math.sin(angle) * distance
                mx = sx + cyaw * lx - syaw_sin * ly
                my = sy + syaw_sin * lx + cyaw * ly
                if (self._max_abs_y_m > 0.0
                        and abs(my - self._center_y_m) > self._max_abs_y_m):
                    angle += float(msg.angle_increment)
                    continue
                updates += self._mark_world_point(mx, my, robot_pose)
            angle += float(msg.angle_increment)

        if updates > 0:
            self._last_occupied_count += updates

    def _lookup_pose(self, source_frame: str, stamp) -> tuple[float, float, float] | None:
        if not source_frame:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame, source_frame, Time.from_msg(stamp),
                timeout=Duration(seconds=0.05))
        except TransformException:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._map_frame, source_frame, Time(),
                    timeout=Duration(seconds=0.05))
            except TransformException:
                return None

        t = transform.transform.translation
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(t.x), float(t.y), yaw

    def _mark_world_point(
            self,
            x: float,
            y: float,
            robot_pose: tuple[float, float, float] | None) -> int:
        info = self._obstacle_map.info
        resolution = float(info.resolution)
        gx = int((x - float(info.origin.position.x)) / resolution)
        gy = int((y - float(info.origin.position.y)) / resolution)
        if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
            return 0

        updated = 0
        radius = self._mark_radius_cells
        radius_sq = radius * radius
        for dy in range(-radius, radius + 1):
            yy = gy + dy
            if yy < 0 or yy >= info.height:
                continue
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius_sq:
                    continue
                xx = gx + dx
                if xx < 0 or xx >= info.width:
                    continue
                wx = float(info.origin.position.x) + (xx + 0.5) * resolution
                wy = float(info.origin.position.y) + (yy + 0.5) * resolution
                if self._inside_robot_clearance(wx, wy, robot_pose):
                    continue
                index = yy * info.width + xx
                if self._obstacle_map.data[index] == self._occupied_value:
                    continue
                self._hit_counts[index] += 1
                if self._hit_counts[index] >= self._hit_count_threshold:
                    self._obstacle_map.data[index] = self._occupied_value
                    updated += 1
        return updated

    def _inside_robot_clearance(
            self,
            x: float,
            y: float,
            robot_pose: tuple[float, float, float] | None) -> bool:
        if robot_pose is None or self._robot_clear_radius_m <= 0.0:
            return False
        rx, ry, _ = robot_pose
        return math.hypot(x - rx, y - ry) < self._robot_clear_radius_m

    def _clear_robot_area(self, robot_pose: tuple[float, float, float] | None):
        if (self._obstacle_map is None
                or robot_pose is None
                or self._robot_clear_radius_m <= 0.0):
            return
        info = self._obstacle_map.info
        resolution = max(1.0e-6, float(info.resolution))
        rx, ry, _ = robot_pose
        cgx = int((rx - float(info.origin.position.x)) / resolution)
        cgy = int((ry - float(info.origin.position.y)) / resolution)
        radius_cells = max(
            1, int(math.ceil(self._robot_clear_radius_m / resolution)))
        radius_sq = self._robot_clear_radius_m * self._robot_clear_radius_m

        for dy in range(-radius_cells, radius_cells + 1):
            gy = cgy + dy
            if gy < 0 or gy >= info.height:
                continue
            for dx in range(-radius_cells, radius_cells + 1):
                gx = cgx + dx
                if gx < 0 or gx >= info.width:
                    continue
                wx = float(info.origin.position.x) + (gx + 0.5) * resolution
                wy = float(info.origin.position.y) + (gy + 0.5) * resolution
                if (wx - rx) * (wx - rx) + (wy - ry) * (wy - ry) > radius_sq:
                    continue
                index = gy * info.width + gx
                self._obstacle_map.data[index] = self._unknown_value
                self._hit_counts[index] = 0

    def _publish_obstacle_map(self):
        if self._obstacle_map is None:
            return
        robot_pose = self._lookup_pose(
            self._base_frame, self.get_clock().now().to_msg())
        self._clear_robot_area(robot_pose)
        msg = copy.deepcopy(self._obstacle_map)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        self._map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FixedObstacleMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
