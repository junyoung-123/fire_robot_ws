"""Camera + 2D LiDAR fusion for observation-based planning aids.

The node publishes:
  - /segmentation_map: a visualization OccupancyGrid in map frame
  - /segmentation_points: occupied points in map frame for Nav2 ObstacleLayer
  - /segmentation_visual: front-camera semantic overlay when SegFormer is enabled

No door or obstacle world coordinates are loaded from the Gazebo world. Every
occupied point is derived from the current /scan transform, optionally filtered
with the latest camera semantic label.
"""

from __future__ import annotations

import math
import struct
import threading
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

torch = None
PILImage = None
SegformerForSemanticSegmentation = None
SegformerImageProcessor = None
_HAS_SEGFORMER = False


_OBSTACLE_IDS = {
    0,   # wall
    1,   # building
    5,   # tree
    8,   # chair
    10,  # table
    12,  # cabinet
    13,  # sofa
    14,  # bookcase
    18,  # person
    22,  # desk
    36,  # furniture
    58,  # stairs
    59,  # stairway
    63,  # door
}

_FREE_IDS = {4, 6, 9, 52}  # floor, road, field, path


class SensorFusionNode(Node):
    """Fuse the observed 2D LiDAR scan with optional front-camera semantics."""

    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.declare_parameter(
            'segformer_model', 'nvidia/segformer-b0-finetuned-ade-512-512')
        self.declare_parameter('enable_segformer', False)
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lidar_frame', 'lidar_link')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_m', 20.0)
        self.declare_parameter('map_height_m', 10.0)
        self.declare_parameter('map_origin_x', -10.0)
        self.declare_parameter('map_origin_y', -5.0)
        self.declare_parameter('lock_origin_to_initial_robot_pose', True)
        self.declare_parameter('unknown_value', -1)
        self.declare_parameter('lidar_max_range_m', 8.0)
        self.declare_parameter('min_obstacle_mark_range_m', 0.85)
        self.declare_parameter('obstacle_dilation_radius_m', 0.06)
        self.declare_parameter('camera_obstacle_dilation_radius_m', 0.18)
        self.declare_parameter('robot_clear_radius_m', 0.85)

        self._model_name = str(self.get_parameter('segformer_model').value)
        self._enable_segformer = bool(
            self.get_parameter('enable_segformer').value)
        self._use_gpu = bool(self.get_parameter('use_gpu').value)
        rate = max(0.2, float(self.get_parameter('publish_rate').value))
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._lidar_frame = str(self.get_parameter('lidar_frame').value)
        self._resolution = max(
            0.01, float(self.get_parameter('map_resolution').value))
        self._width = max(
            1, int(math.ceil(float(self.get_parameter('map_width_m').value)
                             / self._resolution)))
        self._height = max(
            1, int(math.ceil(float(self.get_parameter('map_height_m').value)
                             / self._resolution)))
        self._origin_x = float(self.get_parameter('map_origin_x').value)
        self._origin_y = float(self.get_parameter('map_origin_y').value)
        self._lock_origin = bool(
            self.get_parameter('lock_origin_to_initial_robot_pose').value)
        self._origin_locked = not self._lock_origin
        self._unknown_value = int(self.get_parameter('unknown_value').value)
        self._lidar_max_range_m = float(
            self.get_parameter('lidar_max_range_m').value)
        self._min_mark_range_m = float(
            self.get_parameter('min_obstacle_mark_range_m').value)
        self._obstacle_dilation_radius_m = float(
            self.get_parameter('obstacle_dilation_radius_m').value)
        self._camera_obstacle_dilation_radius_m = float(
            self.get_parameter('camera_obstacle_dilation_radius_m').value)
        self._robot_clear_radius_m = float(
            self.get_parameter('robot_clear_radius_m').value)

        self.bridge = CvBridge()
        self._fx = self._fy = 500.0
        self._cx = 320.0
        self._cy = 240.0
        self._img_w = 640
        self._img_h = 480

        self._latest_image: np.ndarray | None = None
        self._seg_labels: np.ndarray | None = None
        self._latest_scan: LaserScan | None = None
        self._lock = threading.Lock()

        self._processor = None
        self._model = None
        self._device = 'cpu'
        if self._enable_segformer:
            self._load_segformer()
        else:
            self.get_logger().info('SegFormer disabled. Publishing LiDAR-derived observation map.')

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.create_subscription(
            Image, '/camera/color/image_raw', self.camera_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.camera_info_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.grid_pub = self.create_publisher(
            OccupancyGrid, '/segmentation_map', 10)
        self.points_pub = self.create_publisher(
            PointCloud2, '/segmentation_points', 10)
        self.seg_vis = self.create_publisher(
            Image, '/segmentation_visual', 10)

        if self._model is not None:
            self._seg_thread = threading.Thread(
                target=self._segmentation_loop, daemon=True)
            self._seg_thread.start()

        self.create_timer(1.0 / rate, self.publish_map)
        self.get_logger().info(
            f'SensorFusionNode started | frame={self._map_frame}, '
            f'grid={self._width}x{self._height}@{self._resolution:.2f}m')

    def _load_segformer(self):
        global torch, PILImage
        global SegformerForSemanticSegmentation, SegformerImageProcessor
        global _HAS_SEGFORMER
        if not _HAS_SEGFORMER:
            try:
                import torch as torch_module
                from PIL import Image as pil_image_module
                from transformers import (
                    SegformerForSemanticSegmentation as segformer_model_class,
                    SegformerImageProcessor as segformer_processor_class,
                )
                torch = torch_module
                PILImage = pil_image_module
                SegformerForSemanticSegmentation = segformer_model_class
                SegformerImageProcessor = segformer_processor_class
                _HAS_SEGFORMER = True
            except ImportError:
                self.get_logger().warn(
                    'SegFormer requested but torch/transformers is not installed. '
                    'Using LiDAR-only observation points.')
                return

        if not _HAS_SEGFORMER:
            self.get_logger().warn(
                'SegFormer requested but torch/transformers is not installed. '
                'Using LiDAR-only observation points.')
            return
        try:
            self.get_logger().info(f'Loading SegFormer: {self._model_name}')
            self._processor = SegformerImageProcessor.from_pretrained(
                self._model_name)
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                self._model_name)
            if self._use_gpu and torch.cuda.is_available():
                self._device = 'cuda'
            self._model = self._model.to(self._device).eval()
            self.get_logger().info('SegFormer loaded.')
        except Exception as exc:
            self._processor = None
            self._model = None
            self.get_logger().warn(
                f'SegFormer load failed ({exc}). Using LiDAR-only observation points.')

    def camera_info_callback(self, msg: CameraInfo):
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])
        self._img_w = int(msg.width)
        self._img_h = int(msg.height)

    def camera_callback(self, msg: Image):
        if self._model is None:
            return
        with self._lock:
            self._latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _segmentation_loop(self):
        while rclpy.ok():
            image = None
            with self._lock:
                if self._latest_image is not None:
                    image = self._latest_image.copy()
            if image is not None:
                self._run_segformer(image)
            time.sleep(0.5)

    def _run_segformer(self, image: np.ndarray):
        if self._processor is None or self._model is None:
            return
        try:
            pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = self._processor(images=pil, return_tensors='pt')
            inputs = {key: value.to(self._device)
                      for key, value in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            up = torch.nn.functional.interpolate(
                logits,
                size=(image.shape[0], image.shape[1]),
                mode='bilinear',
                align_corners=False,
            )
            seg = up.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
            with self._lock:
                self._seg_labels = seg
            self._publish_seg_visual(seg)
        except Exception as exc:
            self.get_logger().warn(f'SegFormer inference failed: {exc}')

    def publish_map(self):
        self._lock_grid_origin()
        grid = np.full(
            (self._height, self._width), self._unknown_value, dtype=np.int8)
        occupied_points: list[tuple[float, float, float]] = []

        with self._lock:
            seg = self._seg_labels.copy() if self._seg_labels is not None else None

        scan = self._latest_scan
        if scan is not None:
            self._fuse_scan(grid, occupied_points, scan, seg)

        self._clear_robot_area(grid)
        self._publish_grid(grid)
        self._publish_points(occupied_points)

    def _lock_grid_origin(self):
        if self._origin_locked:
            return
        pose = self._lookup_pose(self._base_frame, None)
        if pose is None:
            return
        rx, ry, _ = pose
        self._origin_x = rx - self._width * self._resolution * 0.5
        self._origin_y = ry - self._height * self._resolution * 0.5
        self._origin_locked = True
        self.get_logger().info(
            f'Segmentation map origin locked to initial robot pose: '
            f'origin=({self._origin_x:.2f}, {self._origin_y:.2f})')

    def _fuse_scan(
            self,
            grid: np.ndarray,
            occupied_points: list[tuple[float, float, float]],
            scan: LaserScan,
            seg: np.ndarray | None):
        source_frame = scan.header.frame_id or self._lidar_frame
        scan_pose = self._lookup_pose(source_frame, scan.header.stamp)
        if scan_pose is None:
            return

        sx, sy, syaw = scan_pose
        cos_yaw = math.cos(syaw)
        sin_yaw = math.sin(syaw)
        angle = float(scan.angle_min)
        min_range = max(float(scan.range_min), self._min_mark_range_m)
        max_range = min(float(scan.range_max), self._lidar_max_range_m)

        for distance in scan.ranges:
            if math.isfinite(distance) and min_range <= distance <= max_range:
                lx = math.cos(angle) * distance
                ly = math.sin(angle) * distance
                label = self._seg_label_for_lidar_point(lx, ly, seg)
                occupied = label in _OBSTACLE_IDS or label == -1
                free = label in _FREE_IDS
                mx = sx + cos_yaw * lx - sin_yaw * ly
                my = sy + sin_yaw * lx + cos_yaw * ly
                if occupied:
                    radius = self._camera_obstacle_dilation_radius_m if seg is not None else self._obstacle_dilation_radius_m
                    self._mark_grid_point(grid, mx, my, 100, radius)
                    occupied_points.append((mx, my, 0.15))
                elif free:
                    self._mark_grid_point(grid, mx, my, 0, 0.0)
            angle += float(scan.angle_increment)

    def _seg_label_for_lidar_point(
            self, lx: float, ly: float, seg: np.ndarray | None) -> int:
        if seg is None or lx <= 0.05:
            return -1
        seg_h, seg_w = seg.shape[:2]
        if seg_w <= 0 or seg_h <= 0:
            return -1

        image_u = self._fx * (-ly / lx) + self._cx
        image_v = self._img_h * 0.70
        u = int(image_u * (seg_w / max(1, self._img_w)))
        v = int(image_v * (seg_h / max(1, self._img_h)))
        if not (0 <= u < seg_w and 0 <= v < seg_h):
            return -1
        return int(seg[v, u])

    def _mark_grid_point(
            self, grid: np.ndarray, x: float, y: float, value: int,
            radius_m: float):
        gx = int((x - self._origin_x) / self._resolution)
        gy = int((y - self._origin_y) / self._resolution)
        if gx < 0 or gy < 0 or gx >= self._width or gy >= self._height:
            return
        radius_cells = max(0, int(math.ceil(radius_m / self._resolution)))
        if radius_cells == 0:
            grid[gy, gx] = value
            return
        radius_sq = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            yy = gy + dy
            if yy < 0 or yy >= self._height:
                continue
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_sq:
                    continue
                xx = gx + dx
                if 0 <= xx < self._width:
                    grid[yy, xx] = value

    def _clear_robot_area(self, grid: np.ndarray):
        if self._robot_clear_radius_m <= 0.0:
            return
        pose = self._lookup_pose(self._base_frame, None)
        if pose is None:
            return
        rx, ry, _ = pose
        radius_cells = max(
            1, int(math.ceil(self._robot_clear_radius_m / self._resolution)))
        cgx = int((rx - self._origin_x) / self._resolution)
        cgy = int((ry - self._origin_y) / self._resolution)
        radius_sq = self._robot_clear_radius_m * self._robot_clear_radius_m
        for dy in range(-radius_cells, radius_cells + 1):
            gy = cgy + dy
            if gy < 0 or gy >= self._height:
                continue
            for dx in range(-radius_cells, radius_cells + 1):
                gx = cgx + dx
                if gx < 0 or gx >= self._width:
                    continue
                wx = self._origin_x + (gx + 0.5) * self._resolution
                wy = self._origin_y + (gy + 0.5) * self._resolution
                if (wx - rx) * (wx - rx) + (wy - ry) * (wy - ry) <= radius_sq:
                    grid[gy, gx] = 0

    def _lookup_pose(self, source_frame: str, stamp) -> tuple[float, float, float] | None:
        if not source_frame:
            return None
        try:
            lookup_time = Time.from_msg(stamp) if stamp is not None else Time()
            transform = self._tf_buffer.lookup_transform(
                self._map_frame, source_frame, lookup_time,
                timeout=Duration(seconds=0.05))
        except TransformException:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._map_frame, source_frame, Time(),
                    timeout=Duration(seconds=0.05))
            except TransformException:
                return None
        return self._pose_from_transform(transform)

    def _pose_from_transform(
            self, transform: TransformStamped) -> tuple[float, float, float]:
        t = transform.transform.translation
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(t.x), float(t.y), yaw

    def _publish_grid(self, grid: np.ndarray):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.info.resolution = self._resolution
        msg.info.width = self._width
        msg.info.height = self._height
        msg.info.origin.position.x = self._origin_x
        msg.info.origin.position.y = self._origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().astype(np.int8).tolist()
        self.grid_pub.publish(msg)

    def _publish_points(self, points: list[tuple[float, float, float]]):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = b''.join(struct.pack('<fff', *point) for point in points)
        self.points_pub.publish(msg)

    def _publish_seg_visual(self, seg: np.ndarray):
        vis = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for obs_id in _OBSTACLE_IDS:
            vis[seg == obs_id] = [50, 50, 200]
        for free_id in _FREE_IDS:
            vis[seg == free_id] = [180, 220, 180]
        self.seg_vis.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
