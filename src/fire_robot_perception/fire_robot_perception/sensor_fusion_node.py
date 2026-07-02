"""
sensor_fusion_node.py

Camera + 2D LiDAR fusion for the Nav2 auxiliary segmentation map.

The LiDAR publishes on /scan using the ROS LaserScan message type. Camera
semantic segmentation is used when available to decide whether a LiDAR return is
traversable floor or an obstacle. If the semantic model is unavailable, LiDAR
returns are still mapped as obstacle endpoints so planning remains functional.
"""

import math
import struct
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

try:
    import torch
    from PIL import Image as PILImage
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor)
    _HAS_SEGFORMER = True
except ImportError:
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
_FREE_IDS = {4, 6, 9, 52}   # floor, road, field, path
CAMERA_HFOV_RAD = 1.204


@dataclass
class CameraSource:
    name: str
    image_topic: str
    info_topic: str
    yaw_offset: float
    fx: float = 250.0
    fy: float = 250.0
    cx: float = 160.0
    cy: float = 120.0
    img_w: int = 320
    img_h: int = 240
    latest_image: np.ndarray | None = None
    seg_labels: np.ndarray | None = None
    last_seg_time: float = 0.0


class SensorFusionNode(Node):
    """Publishes a fixed map-frame OccupancyGrid for pre-planning."""

    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.declare_parameter(
            'segformer_model', 'nvidia/segformer-b0-finetuned-ade-512-512')
        self.declare_parameter('enable_segformer', True)
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('segformer_torch_threads', 1)
        self.declare_parameter('max_segmentation_jobs_per_cycle', 1)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('use_depth_camera', False)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lidar_frame', 'lidar_link')
        self.declare_parameter('radar_frame', '')  # legacy alias
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_m', 40.0)
        self.declare_parameter('map_height_m', 20.0)
        self.declare_parameter('map_origin_x', -10.0)
        self.declare_parameter('map_origin_y', -10.0)
        self.declare_parameter('lidar_max_range_m', 8.0)
        self.declare_parameter('lidar_obstacle_memory_sec', 10.0)
        self.declare_parameter('radar_max_range_m', 0.0)  # legacy alias
        self.declare_parameter('radar_obstacle_memory_sec', 0.0)  # legacy alias
        self.declare_parameter('obstacle_dilation_radius_m', 0.12)
        self.declare_parameter('camera_obstacle_dilation_radius_m', 0.18)
        self.declare_parameter('min_obstacle_mark_range_m', 0.85)
        self.declare_parameter('robot_clear_radius_m', 0.85)
        self.declare_parameter('tf_timeout_sec', 0.08)
        self.declare_parameter('segmentation_min_interval_sec', 1.0)
        self.declare_parameter('camera_sources', [
            'front|/camera/color/image_raw|/camera/color/camera_info|0.0',
            'front_left|/camera/front_left/image_raw|/camera/front_left/camera_info|70.0',
            'front_right|/camera/front_right/image_raw|/camera/front_right/camera_info|-70.0',
        ])

        model_name = self.get_parameter('segformer_model').value
        enable_segformer = bool(self.get_parameter('enable_segformer').value)
        use_gpu = bool(self.get_parameter('use_gpu').value)
        torch_threads = int(self.get_parameter('segformer_torch_threads').value)
        self._max_seg_jobs_per_cycle = max(
            1, int(self.get_parameter('max_segmentation_jobs_per_cycle').value))
        rate = float(self.get_parameter('publish_rate').value)
        self._use_depth = bool(self.get_parameter('use_depth_camera').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        legacy_frame = str(self.get_parameter('radar_frame').value).strip()
        self._radar_frame = legacy_frame or str(self.get_parameter('lidar_frame').value)
        self._res = float(self.get_parameter('map_resolution').value)
        width_m = float(self.get_parameter('map_width_m').value)
        height_m = float(self.get_parameter('map_height_m').value)
        self._width = max(1, int(round(width_m / self._res)))
        self._height = max(1, int(round(height_m / self._res)))
        self._origin_x = float(self.get_parameter('map_origin_x').value)
        self._origin_y = float(self.get_parameter('map_origin_y').value)
        legacy_max_range = float(self.get_parameter('radar_max_range_m').value)
        self._radar_max_range = (
            legacy_max_range if legacy_max_range > 0.0
            else float(self.get_parameter('lidar_max_range_m').value))
        legacy_memory_sec = float(self.get_parameter('radar_obstacle_memory_sec').value)
        self._memory_sec = (
            legacy_memory_sec if legacy_memory_sec > 0.0
            else float(self.get_parameter('lidar_obstacle_memory_sec').value))
        self._dilation_cells = max(0, int(math.ceil(
            float(self.get_parameter('obstacle_dilation_radius_m').value)
            / self._res)))
        self._camera_dilation_cells = max(self._dilation_cells, int(math.ceil(
            float(self.get_parameter('camera_obstacle_dilation_radius_m').value)
            / self._res)))
        self._min_obstacle_mark_range_m = float(
            self.get_parameter('min_obstacle_mark_range_m').value)
        self._robot_clear_radius_m = float(
            self.get_parameter('robot_clear_radius_m').value)
        self._tf_timeout = float(self.get_parameter('tf_timeout_sec').value)
        self._seg_min_interval = float(
            self.get_parameter('segmentation_min_interval_sec').value)
        self._camera_sources = self._parse_camera_sources(
            self.get_parameter('camera_sources').value)

        self.bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._latest_radar: LaserScan | None = None
        self._latest_cloud: PointCloud2 | None = None
        self._obstacle_memory: dict[tuple[int, int], float] = {}
        self._lock = threading.Lock()

        self._processor = None
        self._model = None
        self._device = 'cpu'

        if _HAS_SEGFORMER and enable_segformer:
            try:
                if torch_threads > 0:
                    torch.set_num_threads(torch_threads)
                    try:
                        torch.set_num_interop_threads(1)
                    except RuntimeError:
                        pass
                self.get_logger().info(f'Loading SegFormer: {model_name}')
                self._processor = SegformerImageProcessor.from_pretrained(
                    model_name)
                self._model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name)
                if use_gpu and torch.cuda.is_available():
                    self._device = 'cuda'
                self._model = self._model.to(self._device).eval()
                self.get_logger().info('SegFormer loaded.')
            except Exception as e:
                self.get_logger().warn(
                    f'SegFormer load failed ({e}). LiDAR-only map mode.')
        elif not enable_segformer:
            self.get_logger().info(
                'SegFormer disabled by parameter. LiDAR-only map mode.')
        else:
            self.get_logger().warn(
                'transformers not installed. LiDAR-only map mode.')

        for source in self._camera_sources:
            self.create_subscription(
                Image, source.image_topic,
                lambda msg, src=source: self.camera_callback(msg, src), 10)
            self.create_subscription(
                CameraInfo, source.info_topic,
                lambda msg, src=source: self.camera_info_callback(msg, src),
                10)
        self.create_subscription(
            LaserScan, '/scan', self.radar_callback, 10)

        if self._use_depth:
            self.create_subscription(
                PointCloud2, '/camera/depth/points',
                self.depth_cloud_callback, 10)
            self.get_logger().info('Depth PointCloud2 fusion enabled.')

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.grid_pub = self.create_publisher(
            OccupancyGrid, '/segmentation_map', map_qos)
        self.points_pub = self.create_publisher(
            PointCloud2, '/segmentation_points', map_qos)
        self.seg_vis = self.create_publisher(Image, '/segmentation_visual', 10)

        self._seg_thread = threading.Thread(
            target=self._segmentation_loop, daemon=True)
        self._seg_thread.start()

        self.create_timer(1.0 / max(rate, 0.1), self.publish_map)
        self.get_logger().info(
            f'SensorFusionNode started | lidar={self._radar_frame} '
            f'map={self._map_frame} origin=({self._origin_x:.1f}, '
            f'{self._origin_y:.1f}) size={width_m:.1f}x{height_m:.1f}m '
            f'depth={"ON" if self._use_depth else "OFF"} '
            f'cameras={",".join(src.name for src in self._camera_sources)}')

    def _parse_camera_sources(self, raw_sources) -> list[CameraSource]:
        sources: list[CameraSource] = []
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        for raw in raw_sources:
            try:
                name, image_topic, info_topic, yaw_deg = str(raw).split('|', 3)
                sources.append(CameraSource(
                    name=name.strip(),
                    image_topic=image_topic.strip(),
                    info_topic=info_topic.strip(),
                    yaw_offset=math.radians(float(yaw_deg)),
                ))
            except Exception as exc:
                self.get_logger().warn(
                    f'Invalid camera source "{raw}": {exc}')
        if not sources:
            sources.append(CameraSource(
                name='front',
                image_topic='/camera/color/image_raw',
                info_topic='/camera/color/camera_info',
                yaw_offset=0.0,
            ))
        return sources

    def camera_info_callback(self, msg: CameraInfo, source: CameraSource):
        source.fx = msg.k[0]
        source.fy = msg.k[4]
        source.cx = msg.k[2]
        source.cy = msg.k[5]
        source.img_w = msg.width
        source.img_h = msg.height

    def camera_callback(self, msg: Image, source: CameraSource):
        with self._lock:
            source.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def radar_callback(self, msg: LaserScan):
        self._latest_radar = msg

    def depth_cloud_callback(self, msg: PointCloud2):
        self._latest_cloud = msg

    def _segmentation_loop(self):
        import time
        while rclpy.ok():
            if self._model is None:
                time.sleep(0.5)
                continue

            now_sec = time.time()
            jobs = []
            with self._lock:
                for source in self._camera_sources:
                    if source.latest_image is None:
                        continue
                    if now_sec - source.last_seg_time < self._seg_min_interval:
                        continue
                    source.last_seg_time = now_sec
                    jobs.append((source, source.latest_image.copy()))
                    if len(jobs) >= self._max_seg_jobs_per_cycle:
                        break

            for source, image in jobs:
                seg = self._run_segformer(image)
                if seg is not None:
                    with self._lock:
                        source.seg_labels = seg
            time.sleep(0.1)

    def _run_segformer(self, image: np.ndarray) -> np.ndarray | None:
        try:
            pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = self._processor(images=pil, return_tensors='pt')
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            up = torch.nn.functional.interpolate(
                logits,
                size=(image.shape[0], image.shape[1]),
                mode='bilinear',
                align_corners=False,
            )
            seg = up.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
            return seg
        except Exception as e:
            self.get_logger().warn(f'SegFormer error: {e}')
            return None

    def publish_map(self):
        grid = np.full((self._height, self._width), -1, dtype=np.int8)
        now_sec = self._now_sec()

        with self._lock:
            seg_sources = [
                {
                    'name': source.name,
                    'yaw': source.yaw_offset,
                    'fx': source.fx,
                    'cx': source.cx,
                    'img_w': source.img_w,
                    'img_h': source.img_h,
                    'seg': source.seg_labels.copy(),
                }
                for source in self._camera_sources
                if source.seg_labels is not None
            ]

        if self._latest_radar is not None:
            self._fuse_radar_with_seg(
                grid, self._latest_radar, seg_sources, now_sec)

        if self._use_depth and self._latest_cloud is not None:
            self._fuse_depth_cloud(grid, self._latest_cloud, now_sec)

        self._apply_obstacle_memory(grid, now_sec)
        self._publish_grid(grid)
        self._publish_obstacle_points(now_sec)

        if seg_sources:
            self._publish_seg_visual(seg_sources)

    def _fuse_radar_with_seg(self,
                             grid: np.ndarray,
                             scan: LaserScan,
                             seg_sources: list[dict],
                             now_sec: float):
        transform = self._lookup_map_transform(scan.header.frame_id)
        if transform is None:
            return
        sx, sy, yaw = transform
        start = self._world_to_grid(sx, sy)
        if start is None:
            return

        max_range = min(float(scan.range_max), self._radar_max_range)
        c = math.cos(yaw)
        s = math.sin(yaw)
        angle = float(scan.angle_min)

        for raw_r in scan.ranges:
            finite = (
                math.isfinite(raw_r)
                and scan.range_min <= raw_r <= scan.range_max
            )
            dist = min(float(raw_r), max_range) if finite else max_range
            if dist <= scan.range_min:
                angle += scan.angle_increment
                continue

            rx = dist * math.cos(angle)
            ry = dist * math.sin(angle)
            wx = sx + rx * c - ry * s
            wy = sy + rx * s + ry * c
            end = self._world_to_grid(wx, wy)
            if end is None:
                angle += scan.angle_increment
                continue

            hit = finite and float(raw_r) < max_range - 0.05
            self._mark_ray_free(grid, start, end, leave_endpoint=hit)
            if hit and dist >= self._min_obstacle_mark_range_m:
                label = self._get_seg_label(rx, ry, dist, seg_sources)
                if self._radar_hit_is_obstacle(label):
                    radius = (self._camera_dilation_cells
                              if label in _OBSTACLE_IDS
                              else self._dilation_cells)
                    self._remember_obstacle(end, now_sec, radius)
                    self._mark_obstacle(grid, *end, radius_cells=radius)

            angle += scan.angle_increment

    def _radar_hit_is_obstacle(self, label: int) -> bool:
        if label in _FREE_IDS:
            return False
        if label in _OBSTACLE_IDS:
            return True
        return True

    def _get_seg_label(self,
                       rx: float,
                       ry: float,
                       dist: float,
                       seg_sources: list[dict]) -> int:
        if not seg_sources or rx <= 0.0:
            return -1
        angle = math.atan2(ry, rx)
        best = None
        best_abs_delta = math.inf
        for source in seg_sources:
            delta = self._normalize_angle(angle - float(source['yaw']))
            abs_delta = abs(delta)
            if abs_delta < best_abs_delta:
                best = (source, delta)
                best_abs_delta = abs_delta
        if best is None or best_abs_delta > CAMERA_HFOV_RAD * 0.58:
            return -1

        source, delta = best
        seg = source['seg']
        u = int(float(source['cx']) - float(source['fx']) * math.tan(delta))
        v = int(int(source['img_h']) * 0.7)
        if not (0 <= u < int(source['img_w']) and 0 <= v < int(source['img_h'])):
            return -1
        return int(seg[v, u])

    def _fuse_depth_cloud(self,
                          grid: np.ndarray,
                          cloud: PointCloud2,
                          now_sec: float):
        transform = self._lookup_map_transform(self._base_frame)
        if transform is None:
            return
        bx, by, yaw = transform
        c = math.cos(yaw)
        s = math.sin(yaw)

        field_names = [f.name for f in cloud.fields]
        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            return
        x_off = next(f.offset for f in cloud.fields if f.name == 'x')
        y_off = next(f.offset for f in cloud.fields if f.name == 'y')
        z_off = next(f.offset for f in cloud.fields if f.name == 'z')

        for i in range(0, len(cloud.data), cloud.point_step):
            cx = struct.unpack_from('f', cloud.data, i + x_off)[0]
            cy = struct.unpack_from('f', cloud.data, i + y_off)[0]
            cz = struct.unpack_from('f', cloud.data, i + z_off)[0]
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cz)):
                continue
            if cz < 0.1 or cz > self._radar_max_range:
                continue
            rx = cz
            ry = -cx
            rz = -cy
            if rz < 0.05:
                continue
            wx = bx + rx * c - ry * s
            wy = by + rx * s + ry * c
            cell = self._world_to_grid(wx, wy)
            if cell is not None:
                self._remember_obstacle(cell, now_sec, self._dilation_cells)
                self._mark_obstacle(grid, *cell)

    def _lookup_map_transform(self, frame_id: str) -> tuple[float, float, float] | None:
        candidates = []
        clean = frame_id.strip() if frame_id else ''
        if clean:
            candidates.append(clean)
        for frame in (self._radar_frame, self._base_frame):
            if frame and frame not in candidates:
                candidates.append(frame)

        for frame in candidates:
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._map_frame,
                    frame,
                    Time(),
                    timeout=Duration(seconds=self._tf_timeout),
                )
            except TransformException:
                continue
            t = tf.transform.translation
            q = tf.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return float(t.x), float(t.y), math.atan2(siny_cosp, cosy_cosp)

        self.get_logger().warn(
            f'No TF from radar frame candidates {candidates} to {self._map_frame}',
            throttle_duration_sec=3.0)
        return None

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        gx = int((x - self._origin_x) / self._res)
        gy = int((y - self._origin_y) / self._res)
        if 0 <= gx < self._width and 0 <= gy < self._height:
            return gx, gy
        return None

    def _mark_ray_free(self,
                       grid: np.ndarray,
                       start: tuple[int, int],
                       end: tuple[int, int],
                       leave_endpoint: bool):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            if not (leave_endpoint and x == x1 and y == y1):
                grid[y, x] = 0
                self._obstacle_memory.pop((x, y), None)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _remember_obstacle(self,
                           cell: tuple[int, int],
                           now_sec: float,
                           radius_cells: int | None = None):
        r = self._dilation_cells if radius_cells is None else max(0, radius_cells)
        gx, gy = cell
        if r <= 0:
            self._obstacle_memory[(gx, gy)] = now_sec
            return
        for yy in range(max(0, gy - r), min(self._height, gy + r + 1)):
            for xx in range(max(0, gx - r), min(self._width, gx + r + 1)):
                if (xx - gx) * (xx - gx) + (yy - gy) * (yy - gy) <= r * r:
                    self._obstacle_memory[(xx, yy)] = now_sec

    def _apply_obstacle_memory(self, grid: np.ndarray, now_sec: float):
        if self._memory_sec <= 0.0:
            self._obstacle_memory.clear()
            return
        expired = [
            cell for cell, stamp in self._obstacle_memory.items()
            if now_sec - stamp > self._memory_sec
        ]
        for cell in expired:
            self._obstacle_memory.pop(cell, None)
        for gx, gy in self._obstacle_memory:
            if 0 <= gx < self._width and 0 <= gy < self._height:
                self._mark_obstacle(grid, gx, gy)

    def _mark_obstacle(self,
                       grid: np.ndarray,
                       gx: int,
                       gy: int,
                       radius_cells: int | None = None):
        r = self._dilation_cells if radius_cells is None else max(0, radius_cells)
        if r <= 0:
            grid[gy, gx] = 100
            return
        for yy in range(max(0, gy - r), min(self._height, gy + r + 1)):
            for xx in range(max(0, gx - r), min(self._width, gx + r + 1)):
                if (xx - gx) * (xx - gx) + (yy - gy) * (yy - gy) <= r * r:
                    grid[yy, xx] = 100

    def _publish_grid(self, grid: np.ndarray):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.info.resolution = self._res
        msg.info.width = self._width
        msg.info.height = self._height
        msg.info.origin.position.x = self._origin_x
        msg.info.origin.position.y = self._origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.grid_pub.publish(msg)

    def _publish_obstacle_points(self, now_sec: float):
        points = []
        base_pose = self._lookup_map_transform(self._base_frame)
        bx = by = None
        if base_pose is not None:
            bx, by, _ = base_pose
        for (gx, gy), stamp in self._obstacle_memory.items():
            if self._memory_sec > 0.0 and now_sec - stamp > self._memory_sec:
                continue
            x = self._origin_x + (gx + 0.5) * self._res
            y = self._origin_y + (gy + 0.5) * self._res
            if bx is not None and math.hypot(x - bx, y - by) < self._robot_clear_radius_m:
                continue
            points.append((x, y, 0.25))

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
        msg.data = b''.join(struct.pack('fff', *p) for p in points)
        self.points_pub.publish(msg)

    def _publish_seg_visual(self, seg_sources: list[dict]):
        panels = []
        for source in seg_sources:
            seg = source['seg']
            vis = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            vis[:] = [40, 40, 40]
            for obs_id in _OBSTACLE_IDS:
                vis[seg == obs_id] = [50, 50, 200]
            for free_id in _FREE_IDS:
                vis[seg == free_id] = [180, 220, 180]
            panels.append(vis)
        if panels:
            combined = cv2.hconcat(panels)
            self.seg_vis.publish(self.bridge.cv2_to_imgmsg(combined, 'bgr8'))

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
