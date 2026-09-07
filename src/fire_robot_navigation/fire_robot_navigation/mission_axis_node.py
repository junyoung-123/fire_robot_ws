import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class MissionAxisNode(Node):
    """Estimate the mission progress axis from the SLAM/static occupancy map.

    The node publishes /mission_axis as a PoseStamped in map frame:
      - pose.position: origin, normally the robot pose when the first usable map arrives
      - pose.orientation yaw: forward corridor/room axis, flipped to match robot heading
    """

    def __init__(self):
        super().__init__('mission_axis_node')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('occupied_threshold', 55)
        self.declare_parameter('min_occupied_cells', 80)
        self.declare_parameter('sample_stride', 3)
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('lock_origin_to_initial_robot_pose', True)
        self.declare_parameter('lock_axis_after_first_estimate', False)
        self.declare_parameter('fallback_yaw', 0.0)
        self.declare_parameter('max_robot_heading_deviation_deg', 0.0)
        self.declare_parameter('fallback_without_map_after_sec', 0.0)

        self._map_topic = str(self.get_parameter('map_topic').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self._min_occupied_cells = int(self.get_parameter('min_occupied_cells').value)
        self._sample_stride = max(1, int(self.get_parameter('sample_stride').value))
        rate_hz = max(0.1, float(self.get_parameter('publish_rate_hz').value))
        self._lock_origin = bool(
            self.get_parameter('lock_origin_to_initial_robot_pose').value)
        self._lock_axis = bool(
            self.get_parameter('lock_axis_after_first_estimate').value)
        self._fallback_yaw = float(self.get_parameter('fallback_yaw').value)
        self._max_robot_heading_deviation = math.radians(
            max(0.0, float(self.get_parameter('max_robot_heading_deviation_deg').value)))
        self._fallback_without_map_after_sec = max(
            0.0, float(self.get_parameter('fallback_without_map_after_sec').value))

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._map_sub = self.create_subscription(
            OccupancyGrid, self._map_topic, self._map_callback, qos)
        self._axis_pub = self.create_publisher(PoseStamped, '/mission_axis', qos)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._latest_map: OccupancyGrid | None = None
        self._node_start_time = self.get_clock().now()
        self._origin_xy: tuple[float, float] | None = None
        self._origin_from_robot_pose = False
        self._axis_yaw: float | None = None
        self._axis_aligned_to_robot_heading = False
        self._published_once = False

        self.create_timer(1.0 / rate_hz, self._publish_axis)
        self.get_logger().info(
            f'MissionAxisNode started | map={self._map_topic}, '
            f'base={self._base_frame}, frame={self._map_frame}')

    def _map_callback(self, msg: OccupancyGrid):
        self._latest_map = msg
        self._publish_axis()

    def _publish_axis(self):
        grid = self._latest_map
        if grid is None:
            self._publish_fallback_axis_without_map()
            return

        robot_pose = self._robot_pose()
        origin_xy = self._origin_xy
        if origin_xy is None:
            if self._lock_origin and robot_pose is not None:
                origin_xy = (robot_pose[0], robot_pose[1])
                self._origin_from_robot_pose = True
            else:
                origin_xy = (
                    grid.info.origin.position.x
                    + grid.info.width * grid.info.resolution * 0.5,
                    grid.info.origin.position.y
                    + grid.info.height * grid.info.resolution * 0.5,
                )
                self._origin_from_robot_pose = False
            self._origin_xy = origin_xy
        elif (
                self._lock_origin
                and not self._origin_from_robot_pose
                and robot_pose is not None):
            origin_xy = (robot_pose[0], robot_pose[1])
            self._origin_xy = origin_xy
            self._origin_from_robot_pose = True
            self.get_logger().info(
                'Mission axis origin corrected from map center to initial '
                f'robot pose: origin=({origin_xy[0]:.2f}, {origin_xy[1]:.2f})')

        if self._axis_yaw is None or not self._lock_axis:
            estimated = self._estimate_axis_yaw(grid)
            if estimated is None:
                estimated = self._fallback_yaw
            if robot_pose is not None:
                estimated = self._align_axis_to_robot_heading(estimated, robot_pose[2])
                estimated = self._limit_axis_to_robot_heading(estimated, robot_pose[2])
                self._axis_aligned_to_robot_heading = True
            else:
                self._axis_aligned_to_robot_heading = False
            self._axis_yaw = estimated
        elif (
                self._lock_axis
                and not self._axis_aligned_to_robot_heading
                and robot_pose is not None):
            estimated = self._align_axis_to_robot_heading(
                self._axis_yaw, robot_pose[2])
            estimated = self._limit_axis_to_robot_heading(estimated, robot_pose[2])
            self._axis_yaw = estimated
            self._axis_aligned_to_robot_heading = True
            self.get_logger().info(
                'Mission axis yaw corrected using initial robot heading: '
                f'yaw={math.degrees(self._axis_yaw):.1f}deg')

        self._publish_axis_pose(origin_xy, self._axis_yaw)

    def _publish_fallback_axis_without_map(self):
        if self._fallback_without_map_after_sec <= 0.0:
            return
        now = self.get_clock().now()
        elapsed = (now - self._node_start_time).nanoseconds / 1e9
        if elapsed < self._fallback_without_map_after_sec:
            return
        robot_pose = self._robot_pose()
        if robot_pose is None:
            return
        origin_xy = self._origin_xy
        if origin_xy is None:
            origin_xy = (robot_pose[0], robot_pose[1])
            self._origin_xy = origin_xy
            self._origin_from_robot_pose = True
        if self._axis_yaw is None:
            self._axis_yaw = self._limit_axis_to_robot_heading(
                self._fallback_yaw, robot_pose[2])
            self._axis_yaw = self._align_axis_to_robot_heading(
                self._axis_yaw, robot_pose[2])
            self._axis_aligned_to_robot_heading = True
            self.get_logger().warn(
                'Mission axis map was not available in time; publishing '
                'robot-pose fallback axis until the static map arrives.',
                once=True)
        self._publish_axis_pose(origin_xy, self._axis_yaw)

    def _publish_axis_pose(self, origin_xy: tuple[float, float], yaw: float):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.pose.position.x = float(origin_xy[0])
        msg.pose.position.y = float(origin_xy[1])
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self._axis_pub.publish(msg)
        if not self._published_once:
            self._published_once = True
            self.get_logger().info(
                f'Mission axis published: origin=({origin_xy[0]:.2f}, '
                f'{origin_xy[1]:.2f}), yaw={math.degrees(yaw):.1f}deg')

    def _robot_pose(self) -> tuple[float, float, float] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame, Time(),
                timeout=Duration(seconds=0.05))
        except TransformException:
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(t.x), float(t.y), yaw

    def _estimate_axis_yaw(self, grid: OccupancyGrid) -> float | None:
        data = np.asarray(grid.data, dtype=np.int16)
        if data.size != grid.info.width * grid.info.height:
            return None
        data = data.reshape((grid.info.height, grid.info.width))

        occupied = np.argwhere(data >= self._occupied_threshold)
        if occupied.shape[0] < self._min_occupied_cells:
            return None
        occupied = occupied[::self._sample_stride]
        if occupied.shape[0] < 2:
            return None

        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        xs = ox + (occupied[:, 1].astype(np.float64) + 0.5) * res
        ys = oy + (occupied[:, 0].astype(np.float64) + 0.5) * res
        points = np.column_stack((xs, ys))
        points -= points.mean(axis=0)

        cov = np.cov(points, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        return math.atan2(float(axis[1]), float(axis[0]))

    def _align_axis_to_robot_heading(self, axis_yaw: float, robot_yaw: float) -> float:
        # PCA gives an unoriented line and may pick the wall-normal direction
        # when the initial SLAM map is still narrow. Check the perpendicular
        # candidate too, then orient the selected axis toward the robot heading.
        candidates = (axis_yaw, axis_yaw + math.pi * 0.5)
        best = max(candidates, key=lambda yaw: abs(math.cos(yaw - robot_yaw)))
        return self._flip_to_robot_heading(best, robot_yaw)

    def _limit_axis_to_robot_heading(self, axis_yaw: float, robot_yaw: float) -> float:
        max_dev = self._max_robot_heading_deviation
        if max_dev <= 0.0:
            return axis_yaw
        error = self._normalize_angle(axis_yaw - robot_yaw)
        if abs(error) <= max_dev:
            return axis_yaw
        self.get_logger().warn(
            f'Mission axis PCA yaw differs from robot heading by '
            f'{math.degrees(error):.1f}deg; using robot heading for the locked axis.',
            once=True)
        return robot_yaw

    def _flip_to_robot_heading(self, axis_yaw: float, robot_yaw: float) -> float:
        dot = math.cos(axis_yaw - robot_yaw)
        if dot < 0.0:
            axis_yaw += math.pi
        return self._normalize_angle(axis_yaw)

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = MissionAxisNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
