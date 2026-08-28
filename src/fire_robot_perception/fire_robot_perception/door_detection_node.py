"""
door_detection_node.py

RGB 카메라 + Radar(+ 선택적 Depth 카메라)를 이용한 문 탐지 노드.

처리 파이프라인:
  1. YOLOv8 → 이미지 내 문(door) bounding box 탐지
  2. HSV 색상 분석 (bbox 내부) → "blue"(안전) / "red"(위험) 분류
  3. 거리 추정 (우선순위):
     a) Depth 카메라 활성화 시: bbox 중심 픽셀의 depth 값
     b) Radar /scan: bbox 방향의 LaserScan range 값
  4. 카메라 intrinsics + 거리 → 2D 위치 추정 (robot frame)
  5. DoorInfo / FireInfo 토픽 발행

모델: YOLOv8 (scripts/train_door_detector.py로 학습한 것,
             없으면 HSV-only fallback)
"""

import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import rclpy
import rclpy.duration
import tf2_ros
import tf2_geometry_msgs
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped

from fire_robot_interfaces.msg import DoorInfo, FireInfo

for _thread_env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_thread_env, '1')

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

# ── HSV 색상 범위 ────────────────────────────────────────
BLUE_LOWER  = np.array([100,  80,  50])
BLUE_UPPER  = np.array([130, 255, 255])
RED_LOWER1  = np.array([  0,  80,  50])
RED_UPPER1  = np.array([ 10, 255, 255])
RED_LOWER2  = np.array([160,  80,  50])
RED_UPPER2  = np.array([179, 255, 255])
# 초록: H 50-80 (100°-160°) — 비상구 탐지용
# Gazebo exit_marker ambient(0.0, 0.88, 0.05) → OpenCV H≈62
GREEN_LOWER = np.array([ 50, 100,  80])
GREEN_UPPER = np.array([ 80, 255, 255])

# bbox 내 색상 픽셀이 이 비율 이상이면 해당 색으로 판정
COLOR_RATIO_THRESHOLD = 0.20

CAMERA_HFOV_RAD = 1.204   # RealSense D435 기준 ~69도


@dataclass
class CameraSource:
    name: str
    image_topic: str
    info_topic: str
    yaw_offset: float
    fx: float = 234.0
    fy: float = 234.0
    cx: float = 160.0
    cy: float = 120.0
    img_w: int = 320
    img_h: int = 240
    has_info: bool = False


class DoorDetectionNode(Node):
    """YOLOv8 + HSV + Radar(+ Depth) 기반 파란/빨간 문 탐지 노드"""

    def __init__(self):
        super().__init__('door_detection_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('handle_model_path', '')
        self.declare_parameter('handle_confidence_threshold', 0.25)
        self.declare_parameter('handle_height_m', 0.9)
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('yolo_min_interval_sec', 0.50)
        self.declare_parameter('yolo_imgsz', 0)
        self.declare_parameter('torch_num_threads', 1)
        self.declare_parameter('reuse_yolo_detections', True)
        self.declare_parameter('image_process_min_interval_sec', 0.0)
        self.declare_parameter('front_image_process_min_interval_sec', -1.0)
        self.declare_parameter('side_image_process_min_interval_sec', -1.0)
        self.declare_parameter('front_yolo_min_interval_sec', -1.0)
        self.declare_parameter('side_yolo_min_interval_sec', -1.0)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('yolo_far_confidence_distance_m', 4.0)
        self.declare_parameter('yolo_far_min_confidence', 0.45)
        self.declare_parameter('yolo_side_far_min_confidence', 0.55)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('camera_hfov_deg', 69.0)
        self.declare_parameter('use_depth_camera', False)
        self.declare_parameter('door_height_m', 2.0)
        self.declare_parameter('max_detection_distance_m', 12.0)
        self.declare_parameter('door_approach_offset_m', 0.8)
        self.declare_parameter('nav_goal_max_abs_y_m', 0.0)
        self.declare_parameter('side_door_min_abs_y_m', 0.45)
        self.declare_parameter('side_door_standoff_m', 0.85)
        self.declare_parameter('side_handle_max_abs_y_m', 0.0)
        self.declare_parameter('min_door_aspect_ratio', 1.3)
        self.declare_parameter('side_range_max_disagreement_ratio', 2.2)
        self.declare_parameter('side_range_max_disagreement_m', 1.2)
        self.declare_parameter('side_lidar_short_visual_ratio', 0.70)
        self.declare_parameter('side_lidar_short_visual_margin_m', 0.55)
        self.declare_parameter('side_visual_fallback_max_distance_m', 4.5)
        self.declare_parameter('side_wall_projection_enabled', True)
        self.declare_parameter('side_wall_projection_min_abs_angle_deg', 8.0)
        self.declare_parameter('side_wall_projection_min_extend_m', 0.20)
        self.declare_parameter('front_lateral_lidar_prefer_angle_deg', 12.0)
        self.declare_parameter('hsv_fallback_max_door_distance_m', 5.5)
        self.declare_parameter('hsv_fallback_side_requires_range', True)
        self.declare_parameter('side_camera_requires_range', True)
        self.declare_parameter('yolo_reject_edge_clipped_doors', True)
        self.declare_parameter('yolo_edge_clip_min_width_ratio', 0.08)
        self.declare_parameter('yolo_edge_clip_min_height_ratio', 0.50)
        self.declare_parameter('yolo_side_edge_clip_allow_max_dist_m', 2.6)
        self.declare_parameter('yolo_side_edge_clip_allow_min_conf', 0.24)
        self.declare_parameter('yolo_reject_top_clipped_far_doors', True)
        self.declare_parameter('yolo_top_clip_min_distance_m', 4.0)
        self.declare_parameter('yolo_top_clip_max_bottom_ratio', 0.68)
        self.declare_parameter('yolo_top_clip_min_confidence', 0.82)
        self.declare_parameter('yolo_far_min_width_px', 0)
        self.declare_parameter('yolo_far_min_height_px', 0)
        self.declare_parameter('yolo_far_small_allow_min_confidence', 0.0)
        self.declare_parameter('log_detection_candidates', False)
        self.declare_parameter('hsv_fallback_min_width_px', 24)
        self.declare_parameter('hsv_fallback_min_height_px', 70)
        self.declare_parameter('hsv_fallback_green_min_height_px', 45)
        self.declare_parameter('hsv_fallback_green_min_width_px', 18)
        self.declare_parameter('hsv_fallback_green_min_area_ratio', 0.006)
        self.declare_parameter('hsv_fallback_green_min_fill_ratio', 0.16)
        self.declare_parameter('hsv_fallback_clipped_max_width_ratio', 0.56)
        self.declare_parameter('hsv_fallback_min_fill_ratio', 0.30)
        self.declare_parameter('blue_min_color_ratio', 0.24)
        self.declare_parameter('blue_min_dominance_margin', 0.07)
        self.declare_parameter('blue_max_red_ratio', 0.08)
        self.declare_parameter('blue_max_green_ratio', 0.10)
        self.declare_parameter('red_min_color_ratio', 0.20)
        self.declare_parameter('green_min_color_ratio', 0.16)
        self.declare_parameter('publish_map_frame', True)
        self.declare_parameter('camera_sources', [
            'front|/camera/color/image_raw|/camera/color/camera_info|0.0',
            'front_left|/camera/front_left/image_raw|/camera/front_left/camera_info|45.0',
            'front_right|/camera/front_right/image_raw|/camera/front_right/camera_info|-45.0',
        ])

        model_path       = self.get_parameter('model_path').value
        handle_model_path = self.get_parameter('handle_model_path').value
        self._handle_conf = float(self.get_parameter('handle_confidence_threshold').value)
        self._handle_height_m = float(self.get_parameter('handle_height_m').value)
        self._conf       = self.get_parameter('confidence_threshold').value
        self._yolo_min_interval_sec = max(
            0.0, float(self.get_parameter('yolo_min_interval_sec').value))
        self._yolo_imgsz = max(0, int(self.get_parameter('yolo_imgsz').value))
        self._torch_num_threads = max(
            0, int(self.get_parameter('torch_num_threads').value))
        self._reuse_yolo_detections = bool(
            self.get_parameter('reuse_yolo_detections').value)
        self._image_process_min_interval_sec = max(
            0.0, float(self.get_parameter('image_process_min_interval_sec').value))
        self._front_image_process_min_interval_sec = float(
            self.get_parameter('front_image_process_min_interval_sec').value)
        self._side_image_process_min_interval_sec = float(
            self.get_parameter('side_image_process_min_interval_sec').value)
        self._front_yolo_min_interval_sec = float(
            self.get_parameter('front_yolo_min_interval_sec').value)
        self._side_yolo_min_interval_sec = float(
            self.get_parameter('side_yolo_min_interval_sec').value)
        self._publish_debug_image = bool(
            self.get_parameter('publish_debug_image').value)
        self._yolo_far_confidence_distance_m = float(
            self.get_parameter('yolo_far_confidence_distance_m').value)
        self._yolo_far_min_confidence = float(
            self.get_parameter('yolo_far_min_confidence').value)
        self._yolo_side_far_min_confidence = float(
            self.get_parameter('yolo_side_far_min_confidence').value)
        self._frame      = self.get_parameter('frame_id').value
        hfov_deg         = self.get_parameter('camera_hfov_deg').value
        self._hfov       = math.radians(hfov_deg)
        self._use_depth  = self.get_parameter('use_depth_camera').value
        self._door_height_m = float(self.get_parameter('door_height_m').value)
        self._max_detection_distance = float(
            self.get_parameter('max_detection_distance_m').value)
        self._door_approach_offset = float(
            self.get_parameter('door_approach_offset_m').value)
        self._nav_goal_max_abs_y = float(
            self.get_parameter('nav_goal_max_abs_y_m').value)
        self._side_door_min_abs_y = float(
            self.get_parameter('side_door_min_abs_y_m').value)
        self._side_door_standoff = float(
            self.get_parameter('side_door_standoff_m').value)
        self._side_handle_max_abs_y = float(
            self.get_parameter('side_handle_max_abs_y_m').value)
        self._min_door_aspect_ratio = float(
            self.get_parameter('min_door_aspect_ratio').value)
        self._side_range_max_disagreement_ratio = float(
            self.get_parameter('side_range_max_disagreement_ratio').value)
        self._side_range_max_disagreement_m = float(
            self.get_parameter('side_range_max_disagreement_m').value)
        self._side_lidar_short_visual_ratio = float(
            self.get_parameter('side_lidar_short_visual_ratio').value)
        self._side_lidar_short_visual_margin_m = float(
            self.get_parameter('side_lidar_short_visual_margin_m').value)
        self._side_visual_fallback_max_distance_m = float(
            self.get_parameter('side_visual_fallback_max_distance_m').value)
        self._side_wall_projection_enabled = bool(
            self.get_parameter('side_wall_projection_enabled').value)
        self._side_wall_projection_min_abs_angle = math.radians(float(
            self.get_parameter('side_wall_projection_min_abs_angle_deg').value))
        self._side_wall_projection_min_extend_m = float(
            self.get_parameter('side_wall_projection_min_extend_m').value)
        self._front_lateral_lidar_prefer_angle = math.radians(float(
            self.get_parameter('front_lateral_lidar_prefer_angle_deg').value))
        self._hsv_fallback_max_door_distance_m = float(
            self.get_parameter('hsv_fallback_max_door_distance_m').value)
        self._hsv_fallback_side_requires_range = bool(
            self.get_parameter('hsv_fallback_side_requires_range').value)
        self._side_camera_requires_range = bool(
            self.get_parameter('side_camera_requires_range').value)
        self._yolo_reject_edge_clipped_doors = bool(
            self.get_parameter('yolo_reject_edge_clipped_doors').value)
        self._yolo_edge_clip_min_width_ratio = float(
            self.get_parameter('yolo_edge_clip_min_width_ratio').value)
        self._yolo_edge_clip_min_height_ratio = float(
            self.get_parameter('yolo_edge_clip_min_height_ratio').value)
        self._yolo_side_edge_clip_allow_max_dist_m = float(
            self.get_parameter('yolo_side_edge_clip_allow_max_dist_m').value)
        self._yolo_side_edge_clip_allow_min_conf = float(
            self.get_parameter('yolo_side_edge_clip_allow_min_conf').value)
        self._yolo_reject_top_clipped_far_doors = bool(
            self.get_parameter('yolo_reject_top_clipped_far_doors').value)
        self._yolo_top_clip_min_distance_m = float(
            self.get_parameter('yolo_top_clip_min_distance_m').value)
        self._yolo_top_clip_max_bottom_ratio = float(
            self.get_parameter('yolo_top_clip_max_bottom_ratio').value)
        self._yolo_top_clip_min_confidence = float(
            self.get_parameter('yolo_top_clip_min_confidence').value)
        self._yolo_far_min_width_px = int(
            self.get_parameter('yolo_far_min_width_px').value)
        self._yolo_far_min_height_px = int(
            self.get_parameter('yolo_far_min_height_px').value)
        self._yolo_far_small_allow_min_confidence = float(
            self.get_parameter('yolo_far_small_allow_min_confidence').value)
        self._hsv_fallback_min_width_px = int(
            self.get_parameter('hsv_fallback_min_width_px').value)
        self._hsv_fallback_min_height_px = int(
            self.get_parameter('hsv_fallback_min_height_px').value)
        self._hsv_fallback_green_min_height_px = int(
            self.get_parameter('hsv_fallback_green_min_height_px').value)
        self._hsv_fallback_green_min_width_px = int(
            self.get_parameter('hsv_fallback_green_min_width_px').value)
        self._hsv_fallback_green_min_area_ratio = float(
            self.get_parameter('hsv_fallback_green_min_area_ratio').value)
        self._hsv_fallback_green_min_fill_ratio = float(
            self.get_parameter('hsv_fallback_green_min_fill_ratio').value)
        self._hsv_fallback_clipped_max_width_ratio = float(
            self.get_parameter('hsv_fallback_clipped_max_width_ratio').value)
        self._hsv_fallback_min_fill_ratio = float(
            self.get_parameter('hsv_fallback_min_fill_ratio').value)
        self._blue_min_color_ratio = float(
            self.get_parameter('blue_min_color_ratio').value)
        self._blue_min_dominance_margin = float(
            self.get_parameter('blue_min_dominance_margin').value)
        self._blue_max_red_ratio = float(
            self.get_parameter('blue_max_red_ratio').value)
        self._blue_max_green_ratio = float(
            self.get_parameter('blue_max_green_ratio').value)
        self._red_min_color_ratio = float(
            self.get_parameter('red_min_color_ratio').value)
        self._green_min_color_ratio = float(
            self.get_parameter('green_min_color_ratio').value)
        self._log_detection_candidates = bool(
            self.get_parameter('log_detection_candidates').value)
        self._publish_map_frame = bool(
            self.get_parameter('publish_map_frame').value)
        self._camera_sources = self._parse_camera_sources(
            self.get_parameter('camera_sources').value)

        self.bridge = CvBridge()

        self._latest_scan:  LaserScan | None  = None
        self._latest_depth: np.ndarray | None = None   # (H, W) float32 [m]
        self._door_id_map:  dict[str, str]    = {}
        self._yolo_cache: dict[str, tuple[float, list[tuple[int, int, int, int, float]]]] = {}
        self._yolo_lock = Lock()
        self._last_image_process_time: dict[str, float] = {}

        # TF: base_link → map 변환 (감지 시점에 즉시 변환해 stale 좌표 방지)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        cb = ReentrantCallbackGroup()

        # Subscribers
        for source in self._camera_sources:
            self.create_subscription(
                Image, source.image_topic,
                lambda msg, src=source: self.image_callback(msg, src),
                10, callback_group=cb)
            self.create_subscription(
                CameraInfo, source.info_topic,
                lambda msg, src=source: self.camera_info_callback(msg, src),
                10)
        self.create_subscription(
            LaserScan, '/scan', self.radar_callback, 10,
            callback_group=cb)

        if self._use_depth:
            self.create_subscription(
                Image, '/camera/depth/image_rect_raw',
                self.depth_callback, 10, callback_group=cb)
            self.get_logger().info('Depth camera enabled.')

        # Publishers
        self.door_pub  = self.create_publisher(DoorInfo, '/detected_door', 10)
        self.fire_pub  = self.create_publisher(FireInfo, '/fire_info', 10)
        self.debug_pub = self.create_publisher(Image,    '/door_detection/debug', 10)

        # YOLOv8 로드
        self._configure_inference_threads()
        self._model = self._load_model(model_path, 'door')
        self._handle_model = self._load_model(handle_model_path, 'handle')

        self.get_logger().info(
            f'DoorDetectionNode started | YOLO={"OK" if self._model else "FALLBACK_HSV"}'
            f' | handle_YOLO={bool(self._handle_model)}'
            f' | imgsz={self._yolo_imgsz if self._yolo_imgsz > 0 else "auto"}'
            f' | depth={"ON" if self._use_depth else "OFF"}'
            f' | yolo_interval={self._yolo_min_interval_sec:.2f}s'
            f' | image_interval={self._image_process_min_interval_sec:.2f}s'
            f' | debug_image={"ON" if self._publish_debug_image else "OFF"}'
            f' | cameras={",".join(src.name for src in self._camera_sources)}')

    def _configure_inference_threads(self):
        if self._torch_num_threads <= 0:
            return
        try:
            import torch
            torch.set_num_threads(self._torch_num_threads)
            torch.set_num_interop_threads(max(1, min(2, self._torch_num_threads)))
            self.get_logger().info(
                f'YOLO inference torch threads limited to {self._torch_num_threads}.')
        except Exception as e:
            self.get_logger().warn(f'Failed to limit torch threads: {e}')

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

    # ── 모델 로드 ─────────────────────────────────────────
    def _load_model(self, model_path: str, purpose: str = 'door'):
        if not _HAS_YOLO:
            self.get_logger().warn(
                'ultralytics not installed. Using HSV-only detection.')
            return None
        path = Path(model_path)
        if path.exists():
            self.get_logger().info(f'Loading custom {purpose} YOLO model: {path}')
            return YOLO(str(path))
        # 경로 미지정 또는 파일 없음 → HSV 전용 모드
        # (COCO 사전학습 모델은 Door 클래스를 포함하지 않으므로 사용 불가)
        self.get_logger().warn(
            f'model_path "{model_path}" not found. Falling back to HSV-only detection.')
        return None

    # ── 콜백 ──────────────────────────────────────────────
    def camera_info_callback(self, msg: CameraInfo, source: CameraSource):
        source.fx    = msg.k[0]
        source.fy    = msg.k[4]
        source.cx    = msg.k[2]
        source.cy    = msg.k[5]
        source.img_w = msg.width
        source.img_h = msg.height
        source.has_info = True

    def radar_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def depth_callback(self, msg: Image):
        # 16UC1(mm) 또는 32FC1(m) 형식 → float32 미터 단위로 통일
        try:
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            else:
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                depth = depth_mm.astype(np.float32) / 1000.0
            self._latest_depth = depth
        except Exception as e:
            self.get_logger().warn(f'depth decode error: {e}')

    def image_callback(self, msg: Image, source: CameraSource):
        now = time.monotonic()
        process_interval = self._image_interval_for_source(source.name)
        if process_interval > 0.0:
            last = self._last_image_process_time.get(source.name)
            if last is not None and now - last < process_interval:
                return
            self._last_image_process_time[source.name] = now

        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._process(image, msg.header, source)

    def _representative_door_pixel(
            self,
            x1: int,
            x2: int,
            source: CameraSource,
            color: str) -> int:
        cx_pix = (x1 + x2) // 2
        # Use the bbox center as the physical door station. Picking the visible
        # side edge on oblique views can shift the mapped station toward the
        # jamb, so the FSM may stop beside a door instead of squarely in front
        # of it.
        return cx_pix

    def _image_interval_for_source(self, source_name: str) -> float:
        if source_name == 'front' and self._front_image_process_min_interval_sec >= 0.0:
            return self._front_image_process_min_interval_sec
        if source_name != 'front' and self._side_image_process_min_interval_sec >= 0.0:
            return self._side_image_process_min_interval_sec
        return self._image_process_min_interval_sec

    def _yolo_interval_for_source(self, source_name: str) -> float:
        if source_name == 'front' and self._front_yolo_min_interval_sec >= 0.0:
            return self._front_yolo_min_interval_sec
        if source_name != 'front' and self._side_yolo_min_interval_sec >= 0.0:
            return self._side_yolo_min_interval_sec
        return self._yolo_min_interval_sec

    # ── 메인 처리 ─────────────────────────────────────────
    def _process(self, image: np.ndarray, header, source: CameraSource):
        debug = image.copy() if self._publish_debug_image else None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        detections = self._detect_doors(image, source.name)  # list of (x1,y1,x2,y2,conf)
        handle_detections = self._detect_handles(image)
        red_positions: list[PointStamped] = []

        for (x1, y1, x2, y2, det_conf) in detections:
            roi = hsv[y1:y2, x1:x2]
            color = self._classify_color(roi)
            if color == 'unknown':
                continue

            handle_box = self._select_handle_for_door(
                (x1, y1, x2, y2), handle_detections)
            handle_cx_pix = (handle_box[0] + handle_box[2]) // 2 if handle_box else None
            cx_pix = self._representative_door_pixel(
                x1, x2, source, color)
            cy_pix = (y1 + y2) // 2
            dist   = self._door_distance_at_pixel(
                cx_pix, cy_pix, y2 - y1, source)
            if dist is None:
                continue
            if (self._model is not None
                    and not self._passes_yolo_quality_filter(
                        source, x1, y1, x2, y2, det_conf, dist, image.shape)):
                continue
            if (color in ('blue', 'red')
                    and self._hsv_fallback_max_door_distance_m > 0.0
                    and dist > self._hsv_fallback_max_door_distance_m):
                if self._log_detection_candidates:
                    mode = 'YOLO' if self._model is not None else 'HSV'
                    self.get_logger().info(
                        f'{mode} 후보 제외: source={source.name}, color={color}, '
                        f'dist={dist:.2f}m > {self._hsv_fallback_max_door_distance_m:.2f}m, '
                        f'bbox=({x1},{y1},{x2},{y2})',
                        throttle_duration_sec=2.0)
                continue

            door_id  = self._get_door_id(
                source.name, color, cx_pix, image.shape[1], dist)
            door_msg = self._build_door_info(
                header, source, door_id, color, cx_pix, dist,
                bbox_height=(y2 - y1), det_conf=det_conf,
                handle_cx_pix=handle_cx_pix)
            if self._log_detection_candidates and color in ('blue', 'green'):
                pose = door_msg.door_pose.pose.position
                handle = door_msg.handle_position.point
                center_pix = (x1 + x2) // 2
                angle_deg = math.degrees(self._robot_angle_for_pixel(cx_pix, source))
                center_angle_deg = math.degrees(
                    self._robot_angle_for_pixel(center_pix, source))
                measured_dbg = self._range_distance_at_pixel(cx_pix, cy_pix, source)
                visual_dbg = self._visual_distance_from_bbox(y2 - y1, source)
                measured_text = f'{measured_dbg:.2f}' if measured_dbg is not None else 'none'
                visual_text = f'{visual_dbg:.2f}' if visual_dbg is not None else 'none'
                self.get_logger().info(
                    f'문 후보 발행: id={door_msg.door_id}, source={source.name}, '
                    f'color={color}, dist={dist:.2f}m, conf={det_conf:.2f}, '
                    f'bbox=({x1},{y1},{x2},{y2}), rep_x={cx_pix}, '
                    f'center_x={center_pix}, angle={angle_deg:.1f}deg, '
                    f'center_angle={center_angle_deg:.1f}deg, '
                    f'lidar={measured_text}m, visual={visual_text}m, '
                    f'goal=({pose.x:.2f},{pose.y:.2f}), '
                    f'handle=({handle.x:.2f},{handle.y:.2f})',
                    throttle_duration_sec=1.5)
            self.door_pub.publish(door_msg)

            if color == 'red':
                red_point = PointStamped()
                red_point.header = door_msg.door_pose.header
                red_point.point = door_msg.door_pose.pose.position
                red_positions.append(red_point)

            # 디버그 드로잉 (BGR: blue, red, green)
            if debug is not None:
                _DBG = {'blue': (255, 100, 0), 'red': (0, 60, 255), 'green': (0, 200, 50)}
                col  = _DBG.get(color, (200, 200, 200))
                cv2.rectangle(debug, (x1, y1), (x2, y2), col, 3)
                if handle_box is not None:
                    hx1, hy1, hx2, hy2, handle_conf = handle_box
                    cv2.rectangle(debug, (hx1, hy1), (hx2, hy2), (0, 255, 255), 2)
                    cv2.putText(debug, f'handle {handle_conf:.2f}', (hx1, max(18, hy1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(debug, f'{source.name} {color} {det_conf:.2f} d={dist:.1f}m',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        self._publish_fire_info(header, red_positions)
        if debug is not None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))

    def _detect_handles(self, image: np.ndarray) -> list:
        """Detect lever handles once per frame with the dedicated model."""
        if self._handle_model is None:
            return []
        kwargs = {'conf': self._handle_conf, 'verbose': False}
        if self._yolo_imgsz > 0:
            kwargs['imgsz'] = self._yolo_imgsz
        with self._yolo_lock:
            results = self._handle_model(image, **kwargs)
        handles = []
        for result in results:
            for box in result.boxes:
                cls_name = self._handle_model.names[int(box.cls)]
                if 'handle' not in cls_name.lower():
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                handles.append((x1, y1, x2, y2, float(box.conf[0])))
        return handles

    @staticmethod
    def _select_handle_for_door(door_box, handles: list):
        """Return the highest-confidence handle whose center is in a door."""
        x1, y1, x2, y2 = door_box
        margin_x = max(4, int((x2 - x1) * 0.08))
        margin_y = max(4, int((y2 - y1) * 0.05))
        candidates = []
        for handle in handles:
            hx1, hy1, hx2, hy2, confidence = handle
            cx = (hx1 + hx2) / 2.0
            cy = (hy1 + hy2) / 2.0
            if (x1 - margin_x <= cx <= x2 + margin_x
                    and y1 - margin_y <= cy <= y2 + margin_y):
                candidates.append((confidence, handle))
        return max(
            candidates, default=(None, None), key=lambda item: item[0])[1]

    # ── 문 탐지 ───────────────────────────────────────────
    def _detect_doors(self, image: np.ndarray, source_name: str) -> list:
        """Return YOLO door boxes plus tight HSV color-panel fallbacks."""
        if self._model is not None:
            boxes = self._yolo_detect(image, source_name)
            color_boxes = self._hsv_fallback_detect(
                image, colors=('blue', 'red', 'green'))
            return self._merge_detection_boxes(boxes, color_boxes)
        return self._hsv_fallback_detect(image)

    def _merge_detection_boxes(self, primary: list, extra: list) -> list:
        merged = list(primary)
        for candidate in extra:
            overlaps = [
                box for box in merged
                if self._box_iou(candidate, box) > 0.30
            ]
            if not overlaps:
                merged.append(candidate)
                continue

            candidate_area = self._box_area(candidate)
            # Keep tight green HSV boxes even when YOLO already found a larger
            # door-shaped region. The tighter ROI lets the color classifier see
            # the exit panel instead of the surrounding wall.
            if any(candidate_area < self._box_area(box) * 0.75 for box in overlaps):
                merged.append(candidate)
        return merged

    def _box_area(self, box) -> int:
        x1, y1, x2, y2, _ = box
        return max(1, x2 - x1) * max(1, y2 - y1)

    def _box_iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2, _ = a
        bx1, by1, bx2, by2, _ = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _yolo_detect(self, image: np.ndarray, source_name: str) -> list:
        now = time.monotonic()
        yolo_interval = self._yolo_interval_for_source(source_name)
        if self._reuse_yolo_detections and yolo_interval > 0.0:
            cached = self._yolo_cache.get(source_name)
            if cached is not None:
                stamp, boxes = cached
                if now - stamp < yolo_interval:
                    return list(boxes)

        with self._yolo_lock:
            now = time.monotonic()
            if self._reuse_yolo_detections and yolo_interval > 0.0:
                cached = self._yolo_cache.get(source_name)
                if cached is not None and now - cached[0] < yolo_interval:
                    return list(cached[1])
            yolo_kwargs = {'conf': self._conf, 'verbose': False}
            if self._yolo_imgsz > 0:
                yolo_kwargs['imgsz'] = self._yolo_imgsz
            results = self._model(image, **yolo_kwargs)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_name = self._model.names[int(box.cls)]
                # 학습된 모델에서 'door' 클래스만, 기본 모델은 모든 클래스 허용
                if 'door' not in cls_name.lower() and len(self._model.names) > 10:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # 너무 작은 bbox 제외 (이미지 면적의 1% 미만)
                area = (x2 - x1) * (y2 - y1)
                if area < image.shape[0] * image.shape[1] * 0.01:
                    continue
                if not self._passes_door_shape_filter(x1, y1, x2, y2):
                    continue
                boxes.append((x1, y1, x2, y2, conf))
        if self._reuse_yolo_detections:
            self._yolo_cache[source_name] = (now, list(boxes))
        return boxes

    def _passes_yolo_quality_filter(self, source: CameraSource,
                                    x1: int, y1: int, x2: int, y2: int,
                                    conf: float, dist: float,
                                    image_shape) -> bool:
        if (self._yolo_reject_edge_clipped_doors
                and self._is_rejectable_yolo_edge_clipped_box(
                    x1, y1, x2, y2, image_shape)):
            if self._is_allowed_close_side_edge_clip(
                    source, conf, dist, x1, x2, image_shape):
                return True
            if self._log_detection_candidates:
                self.get_logger().info(
                    f'YOLO 후보 제외: source={source.name}, '
                    f'edge-clipped bbox=({x1},{y1},{x2},{y2})',
                    throttle_duration_sec=2.0)
            return False

        if self._is_rejectable_far_top_clipped_yolo_box(
                x1, y1, x2, y2, conf, dist, image_shape):
            if self._log_detection_candidates:
                self.get_logger().info(
                    f'YOLO 후보 제외: source={source.name}, far top-clipped '
                    f'dist={dist:.2f}m, conf={conf:.2f}, '
                    f'bbox=({x1},{y1},{x2},{y2})',
                    throttle_duration_sec=2.0)
            return False

        if self._is_rejectable_far_small_yolo_box(
                x1, y1, x2, y2, conf, dist):
            if self._log_detection_candidates:
                self.get_logger().info(
                    f'YOLO 후보 제외: source={source.name}, far small '
                    f'dist={dist:.2f}m, conf={conf:.2f}, '
                    f'bbox=({x1},{y1},{x2},{y2})',
                    throttle_duration_sec=2.0)
            return False

        if self._yolo_far_confidence_distance_m <= 0.0:
            return True
        if dist < self._yolo_far_confidence_distance_m:
            return True

        min_conf = self._yolo_far_min_confidence
        if source.name != 'front':
            min_conf = max(min_conf, self._yolo_side_far_min_confidence)

        if conf >= min_conf:
            return True

        if self._log_detection_candidates:
            self.get_logger().info(
                f'YOLO 후보 제외: source={source.name}, dist={dist:.2f}m, '
                f'conf={conf:.2f} < {min_conf:.2f}, '
                f'bbox=({x1},{y1},{x2},{y2})',
                throttle_duration_sec=2.0)
        return False

    def _is_rejectable_yolo_edge_clipped_box(
            self, x1: int, y1: int, x2: int, y2: int,
            image_shape) -> bool:
        img_h, img_w = image_shape[:2]
        if img_h <= 0 or img_w <= 0:
            return False

        touches_side = x1 <= 1 or x2 >= img_w - 2
        if not touches_side:
            return False

        width_ratio = (x2 - x1) / float(img_w)
        height_ratio = (y2 - y1) / float(img_h)
        if width_ratio < self._yolo_edge_clip_min_width_ratio:
            return False
        if height_ratio < self._yolo_edge_clip_min_height_ratio:
            return False

        # A tall bbox cut by the image side is usually a partial color plane while
        # turning. Wait until it is centered enough before assigning a map target.
        return True

    def _is_rejectable_far_top_clipped_yolo_box(
            self, x1: int, y1: int, x2: int, y2: int,
            conf: float, dist: float, image_shape) -> bool:
        if not self._yolo_reject_top_clipped_far_doors:
            return False
        if dist < self._yolo_top_clip_min_distance_m:
            return False
        if conf >= self._yolo_top_clip_min_confidence:
            return False
        img_h = image_shape[0] if len(image_shape) > 0 else 0
        if img_h <= 0:
            return False
        touches_top = y1 <= 1
        bottom_ratio = y2 / float(img_h)
        if not touches_top:
            return False
        return bottom_ratio <= self._yolo_top_clip_max_bottom_ratio

    def _is_rejectable_far_small_yolo_box(
            self, x1: int, y1: int, x2: int, y2: int,
            conf: float, dist: float) -> bool:
        if self._yolo_far_confidence_distance_m <= 0.0:
            return False
        if dist < self._yolo_far_confidence_distance_m:
            return False
        if (self._yolo_far_small_allow_min_confidence > 0.0
                and conf >= self._yolo_far_small_allow_min_confidence):
            return False
        min_w = max(0, self._yolo_far_min_width_px)
        min_h = max(0, self._yolo_far_min_height_px)
        if min_w <= 0 and min_h <= 0:
            return False
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return (min_w > 0 and width < min_w) or (min_h > 0 and height < min_h)

    def _is_allowed_close_side_edge_clip(
            self, source: CameraSource, conf: float, dist: float,
            x1: int, x2: int, image_shape) -> bool:
        if source.name == 'front':
            return False
        if self._yolo_side_edge_clip_allow_max_dist_m <= 0.0:
            return False
        return (
            dist <= self._yolo_side_edge_clip_allow_max_dist_m
            and conf >= self._yolo_side_edge_clip_allow_min_conf)

    def _hsv_fallback_detect(self, image: np.ndarray, colors: tuple[str, ...] | None = None) -> list:
        """YOLO 없을 때 HSV 기반 contour로 문 후보 검출"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        base_min_area = image.shape[0] * image.shape[1] * 0.02
        boxes = []
        color_masks = [
            ('blue', cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)),
            ('red', (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) |
                     cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))),
            ('green', cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)),
        ]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        for mask_color, mask in color_masks:
            if colors is not None and mask_color not in colors:
                continue
            min_area = base_min_area
            if mask_color == 'green':
                min_area = image.shape[0] * image.shape[1] * max(
                    0.001, self._hsv_fallback_green_min_area_ratio)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                roi_color = self._classify_color(hsv[y:y + h, x:x + w])
                if roi_color != mask_color:
                    continue
                if not self._passes_hsv_fallback_geometry(
                        mask_color, x, y, w, h, area, image.shape):
                    continue
                conf = min(1.0, area / (min_area * 5))
                boxes.append((x, y, x + w, y + h, conf))
        return boxes

    def _passes_hsv_fallback_geometry(
            self,
            color: str,
            x: int,
            y: int,
            w: int,
            h: int,
            contour_area: float,
            image_shape: tuple[int, ...]) -> bool:
        img_h, img_w = image_shape[:2]
        if color == 'green':
            min_width = self._hsv_fallback_green_min_width_px
            min_height = self._hsv_fallback_green_min_height_px
            min_fill = self._hsv_fallback_green_min_fill_ratio
        else:
            min_width = self._hsv_fallback_min_width_px
            min_height = self._hsv_fallback_min_height_px
            min_fill = self._hsv_fallback_min_fill_ratio

        if w < min_width or h < min_height:
            return False
        fill_ratio = contour_area / float(max(1, w * h))
        if fill_ratio < min_fill:
            return False
        if not self._passes_door_shape_filter(x, y, x + w, y + h):
            return False

        clipped_top_bottom = y <= 1 and (y + h) >= img_h - 1
        clipped_side = x <= 1 or (x + w) >= img_w - 1
        width_ratio = w / float(max(1, img_w))
        if (color in ('blue', 'red')
                and clipped_top_bottom
                and clipped_side
                and width_ratio > self._hsv_fallback_clipped_max_width_ratio):
            if self._log_detection_candidates:
                self.get_logger().info(
                    f'HSV candidate rejected: color={color}, edge-clipped bbox='
                    f'({x},{y},{x + w},{y + h}), width_ratio={width_ratio:.2f}',
                    throttle_duration_sec=2.0)
            return False

        return True
    def _passes_door_shape_filter(self, x1: int, y1: int,
                                  x2: int, y2: int) -> bool:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return (height / width) >= self._min_door_aspect_ratio

    # ── 색상 분류 ─────────────────────────────────────────
    def _color_ratios(self, roi_hsv: np.ndarray) -> tuple[float, float, float]:
        if roi_hsv.size == 0:
            return 0.0, 0.0, 0.0
        total = max(1, roi_hsv.shape[0] * roi_hsv.shape[1])
        blue_r = cv2.countNonZero(
            cv2.inRange(roi_hsv, BLUE_LOWER, BLUE_UPPER)) / total
        red_r = cv2.countNonZero(
            cv2.inRange(roi_hsv, RED_LOWER1, RED_UPPER1) |
            cv2.inRange(roi_hsv, RED_LOWER2, RED_UPPER2)) / total
        green_r = cv2.countNonZero(
            cv2.inRange(roi_hsv, GREEN_LOWER, GREEN_UPPER)) / total
        return blue_r, red_r, green_r

    def _panel_core_roi(self, roi_hsv: np.ndarray) -> np.ndarray:
        h, w = roi_hsv.shape[:2]
        if h < 12 or w < 8:
            return roi_hsv
        x_margin = int(w * (0.16 if w >= 36 else 0.08))
        y_margin = int(h * 0.06)
        x0 = min(max(0, x_margin), max(0, w - 1))
        x1 = max(x0 + 1, w - x_margin)
        y0 = min(max(0, y_margin), max(0, h - 1))
        y1 = max(y0 + 1, h - y_margin)
        return roi_hsv[y0:y1, x0:x1]

    def _classify_color(self, roi_hsv: np.ndarray) -> str:
        if roi_hsv.size == 0:
            return 'unknown'

        full_blue, full_red, full_green = self._color_ratios(roi_hsv)
        core_blue, core_red, core_green = self._color_ratios(
            self._panel_core_roi(roi_hsv))

        blue_min_full = max(COLOR_RATIO_THRESHOLD, self._blue_min_color_ratio)
        blue_min_core = max(0.16, self._blue_min_color_ratio * 0.62)
        blue_margin = min(0.12, self._blue_min_dominance_margin)
        core_blue_dominates = (
            core_blue >= blue_min_core
            and core_blue >= max(core_red, core_green) + blue_margin)
        full_blue_dominates = (
            full_blue >= blue_min_full
            and full_blue >= max(full_red, full_green)
            + self._blue_min_dominance_margin)

        if full_blue_dominates:
            if full_red > max(self._blue_max_red_ratio, 0.08):
                return 'unknown'
            if full_green > max(self._blue_max_green_ratio, 0.08):
                return 'unknown'
            return 'blue'

        if core_blue_dominates:
            if core_red > 0.12 or core_green > 0.12:
                return 'unknown'
            if full_red > 0.18 and full_red > full_blue + 0.16:
                return 'unknown'
            return 'blue'

        green_signal = max(full_green, core_green)
        if (green_signal > self._green_min_color_ratio
                and green_signal >= max(full_blue, core_blue, full_red, core_red)):
            return 'green'

        red_signal = max(full_red, core_red)
        blue_signal = max(full_blue, core_blue)
        if red_signal > self._red_min_color_ratio:
            if blue_signal >= red_signal - 0.04:
                return 'unknown'
            if core_blue >= 0.12 and core_blue >= core_red:
                return 'unknown'
            return 'red'
        return 'unknown'

    # ── 거리 추정 ─────────────────────────────────────────
    def _range_distance_at_pixel(self,
                                  cx_pix: int,
                                  cy_pix: int | None,
                                  source: CameraSource) -> float | None:
        """거리 추정: Depth 카메라 우선, 없으면 Radar /scan fallback."""
        if (self._use_depth and source.name == 'front'
                and self._latest_depth is not None):
            dist = self._depth_distance_at_pixel(cx_pix, cy_pix)
            if dist is not None:
                return dist
        robot_angle = self._robot_angle_for_pixel(cx_pix, source)
        return self._radar_distance_at_angle(robot_angle)

    def _door_distance_at_pixel(self,
                                cx_pix: int,
                                cy_pix: int,
                                bbox_height: int,
                                source: CameraSource) -> float | None:
        measured = self._range_distance_at_pixel(cx_pix, cy_pix, source)
        visual = self._visual_distance_from_bbox(bbox_height, source)

        # Side cameras see wall-mounted doors at an oblique angle. The apparent
        # bbox height is often too small, so visual range can jump several
        # meters. Use the 2D LiDAR ray as the anchor, but fall back to visual
        # range when the ray is clearly hitting an intervening obstacle/wall.
        if abs(source.yaw_offset) > math.radians(5.0):
            if self._side_camera_requires_range and measured is None:
                return None
            if (self._model is None
                    and self._hsv_fallback_side_requires_range
                    and measured is None):
                return None
            if measured is None:
                return visual
            if visual is not None:
                short_limit = max(
                    visual * self._side_lidar_short_visual_ratio,
                    visual - self._side_lidar_short_visual_margin_m)
                if measured < short_limit:
                    max_visual = self._side_visual_fallback_max_distance_m
                    if max_visual > 0.0 and visual > max_visual:
                        if self._log_detection_candidates:
                            self.get_logger().info(
                                f'Side camera visual fallback rejected '
                                f'(source={source.name}, lidar={measured:.2f}m, '
                                f'visual={visual:.2f}m > {max_visual:.2f}m)',
                                throttle_duration_sec=2.0)
                        return None
                    if self._log_detection_candidates:
                        self.get_logger().info(
                            f'Side camera LiDAR range looks occluded; '
                            f'using visual range instead '
                            f'(source={source.name}, lidar={measured:.2f}m, '
                            f'visual={visual:.2f}m)',
                            throttle_duration_sec=2.0)
                    return visual
                near = max(0.1, min(measured, visual))
                far = max(measured, visual)
                ratio = far / near
                diff = abs(measured - visual)
                if measured > max(visual * 1.35, visual + 0.75):
                    if self._log_detection_candidates:
                        self.get_logger().info(
                            f'Side camera LiDAR range looks too long for the '
                            f'visible door; using visual range '
                            f'(source={source.name}, lidar={measured:.2f}m, '
                            f'visual={visual:.2f}m)',
                            throttle_duration_sec=2.0)
                    return visual
                if (ratio > self._side_range_max_disagreement_ratio
                        and diff > self._side_range_max_disagreement_m):
                    return None
            return measured

        if measured is None:
            return visual
        if visual is not None:
            # A central obstacle can block the scan ray while the colored door is
            # still visible. Check this before the front-lateral scan preference,
            # otherwise a door can be projected onto the obstacle instead of the
            # wall station seen by the camera.
            if measured < max(visual * 0.65, visual - 1.0):
                return visual
            if source.name == 'front':
                front_angle = abs(self._robot_angle_for_pixel(cx_pix, source))
                if front_angle >= self._front_lateral_lidar_prefer_angle:
                    if measured > max(visual * 1.25, visual + 0.70):
                        if self._log_detection_candidates:
                            self.get_logger().info(
                                f'Front lateral LiDAR range looks too long for '
                                f'the visible door; using visual range '
                                f'(lidar={measured:.2f}m, '
                                f'visual={visual:.2f}m)',
                                throttle_duration_sec=2.0)
                        return visual
                    return measured
            if measured > self._max_detection_distance:
                return visual
        return measured

    def _visual_distance_from_bbox(self,
                                    bbox_height: int,
                                    source: CameraSource) -> float | None:
        if bbox_height <= 0 or source.fy <= 0.0:
            return None
        dist = (self._door_height_m * source.fy) / float(bbox_height)
        if 0.2 <= dist <= self._max_detection_distance:
            return float(dist)
        return None

    def _depth_distance_at_pixel(self,
                                   cx_pix: int,
                                   cy_pix: int | None) -> float | None:
        """bbox 중심 주변 패치의 중앙값 depth 반환 (단위: m)."""
        depth = self._latest_depth
        if depth is None:
            return None
        cy = cy_pix if cy_pix is not None else depth.shape[0] // 2
        # 11×11 패치 중앙값으로 노이즈 억제
        r0, r1 = max(0, cy - 5), min(depth.shape[0], cy + 6)
        c0, c1 = max(0, cx_pix - 5), min(depth.shape[1], cx_pix + 6)
        patch = depth[r0:r1, c0:c1]
        valid = patch[(patch > 0.1) & (patch < 15.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _robot_angle_for_pixel(self,
                                cx_pix: int,
                                source: CameraSource) -> float:
        """이미지 픽셀 x → 로봇 기준 수평각(+Y/좌측이 양수)."""
        pixel_angle = math.atan((source.cx - cx_pix) / source.fx)
        return source.yaw_offset + pixel_angle

    def _radar_distance_at_angle(self, robot_angle: float) -> float | None:
        """로봇 기준 수평각 → Radar /scan range."""
        if self._latest_scan is None:
            return None
        scan = self._latest_scan
        idx = int((robot_angle - scan.angle_min) / scan.angle_increment)
        n   = len(scan.ranges)
        window = []
        for offset in range(-2, 3):
            i = idx + offset
            if not (0 <= i < n):   # 범위 밖은 스킵 (wrap-around 방지)
                continue
            r = scan.ranges[i]
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                window.append(r)
        if not window:
            return None
        return float(np.median(window))

    # ── 메시지 빌드 ───────────────────────────────────────
    def _build_door_info(self, header, source: CameraSource,
                         door_id: str, color: str,
                         cx_pix: int, dist: float,
                         bbox_height: int, det_conf: float,
                         handle_cx_pix: int | None = None) -> DoorInfo:
        msg = DoorInfo()
        msg.header = header

        # 카메라 수평각 + 카메라 장착 yaw → robot frame (x=전방, y=좌).
        angle = self._robot_angle_for_pixel(cx_pix, source)
        handle_angle = self._robot_angle_for_pixel(
            handle_cx_pix if handle_cx_pix is not None else cx_pix, source)
        wall_projected_dist = self._side_wall_projected_distance(angle, color)
        if wall_projected_dist is not None:
            raw_handle_y = dist * math.sin(handle_angle)
            if (abs(raw_handle_y) >= self._side_door_min_abs_y
                    or wall_projected_dist
                    > dist + self._side_wall_projection_min_extend_m):
                dist = wall_projected_dist
        handle_x = dist * math.cos(handle_angle)
        handle_y = dist * math.sin(handle_angle)
        handle_y = self._clamp_side_handle_y(handle_y, color)
        nav_dist = max(0.3, dist - self._door_approach_offset)
        px = nav_dist * math.cos(angle)
        py = nav_dist * math.sin(angle)

        # base_link → map 프레임 변환 (감지 시점에 즉시 변환)
        pose_base = PoseStamped()
        pose_base.header.frame_id = self._frame
        pose_base.header.stamp    = header.stamp
        pose_base.pose.position.x = px
        pose_base.pose.position.y = py
        pose_base.pose.orientation.w = 1.0
        handle_base = PointStamped()
        handle_base.header = pose_base.header
        handle_base.point.x = handle_x
        handle_base.point.y = handle_y
        handle_base.point.z = self._handle_height_m

        # Shape the approach goal in base_link first. Clamping a side-door goal
        # after transforming to map can flip the side when the SLAM map axis is
        # not perfectly aligned with corridor lateral.
        shaped = DoorInfo()
        shaped.door_pose = pose_base
        shaped.handle_position = handle_base
        self._shape_navigation_pose_for_door(shaped, color, handle_y)
        pose_base = shaped.door_pose
        handle_base = shaped.handle_position

        if not self._publish_map_frame:
            msg.door_pose = pose_base
            msg.handle_position = handle_base
            msg.door_id    = door_id
            msg.door_color = color
            msg.is_open    = False
            msg.confidence = float(det_conf)
            msg.distance_from_fire = 0.0
            return msg

        try:
            pose_map = self._tf_buffer.transform(
                pose_base, 'map',
                timeout=rclpy.duration.Duration(seconds=0.1))
            handle_map = self._tf_buffer.transform(
                handle_base, 'map',
                timeout=rclpy.duration.Duration(seconds=0.1))
            msg.door_pose = pose_map
            msg.handle_position = handle_map
        except Exception as stamped_error:
            try:
                # Camera images and TF can be a few frames out of phase in
                # Gazebo/WSL. Use the latest available transform before
                # falling back to base_link so FSM can still reason in map.
                latest_stamp = rclpy.time.Time().to_msg()
                pose_base.header.stamp = latest_stamp
                handle_base.header.stamp = latest_stamp
                pose_map = self._tf_buffer.transform(
                    pose_base, 'map',
                    timeout=rclpy.duration.Duration(seconds=0.2))
                handle_map = self._tf_buffer.transform(
                    handle_base, 'map',
                    timeout=rclpy.duration.Duration(seconds=0.2))
                msg.door_pose = pose_map
                msg.handle_position = handle_map
            except Exception:
                self.get_logger().debug(
                    f'Door map transform unavailable, publishing {self._frame}: '
                    f'{stamped_error}')
                # SLAM 맵 초기화 전 → base_link 그대로 (Nav2 목표 전달 시 주의)
                msg.door_pose = pose_base
                msg.handle_position = handle_base

        msg.door_id    = door_id
        msg.door_color = color
        msg.is_open    = False
        msg.confidence = float(det_conf)
        msg.distance_from_fire = 0.0
        return msg

    def _clamp_side_handle_y(self, handle_y: float, color: str) -> float:
        """Keep wall-door handle estimates near the physical corridor wall band."""
        if color == 'green':
            return handle_y
        if abs(handle_y) < self._side_door_min_abs_y:
            return handle_y

        max_abs_y = self._side_handle_max_abs_y
        if max_abs_y <= 0.0 and self._nav_goal_max_abs_y > 0.0:
            max_abs_y = self._nav_goal_max_abs_y + self._side_door_standoff
        if max_abs_y <= self._side_door_min_abs_y:
            return handle_y

        if abs(handle_y) <= max_abs_y:
            return handle_y
        return math.copysign(max_abs_y, handle_y)

    def _side_wall_projected_distance(self,
                                      angle: float,
                                      color: str) -> float | None:
        """Project side-wall door detections onto the observed corridor wall band."""
        if not self._side_wall_projection_enabled or color == 'green':
            return None
        if abs(angle) < self._side_wall_projection_min_abs_angle:
            return None

        max_abs_y = self._side_handle_max_abs_y
        if max_abs_y <= 0.0 and self._nav_goal_max_abs_y > 0.0:
            max_abs_y = self._nav_goal_max_abs_y + self._side_door_standoff
        if max_abs_y <= self._side_door_min_abs_y:
            return None

        sin_a = math.sin(angle)
        if abs(sin_a) < 1e-3:
            return None
        dist = max_abs_y / abs(sin_a)
        forward = dist * math.cos(angle)
        if forward <= 0.20 or dist > self._max_detection_distance:
            return None
        return float(dist)

    def _shape_navigation_pose_for_door(self, msg: DoorInfo,
                                        color: str,
                                        lateral_hint_y: float):
        if color == 'green':
            self._set_pose_yaw(msg.door_pose, 0.0)
            self._clamp_navigation_pose(msg.door_pose, side_hint_y=lateral_hint_y)
            return
        if abs(lateral_hint_y) < self._side_door_min_abs_y:
            self._set_pose_yaw(msg.door_pose, 0.0)
            self._clamp_navigation_pose(msg.door_pose, side_hint_y=lateral_hint_y)
            return
        if msg.handle_position.header.frame_id != msg.door_pose.header.frame_id:
            self._clamp_navigation_pose(msg.door_pose, side_hint_y=lateral_hint_y)
            return

        side = 1.0 if lateral_hint_y > 0.0 else -1.0
        msg.door_pose.pose.position.x = msg.handle_position.point.x
        msg.door_pose.pose.position.y = (
            msg.handle_position.point.y - side * self._side_door_standoff)
        self._set_pose_yaw(msg.door_pose, side * math.pi / 2.0)
        self._clamp_navigation_pose(msg.door_pose, side_hint_y=lateral_hint_y)

    def _set_pose_yaw(self, pose: PoseStamped, yaw: float):
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

    def _clamp_navigation_pose(self, pose: PoseStamped,
                               side_hint_y: float | None = None):
        """Keep side-door Nav2 goals in the corridor approach lane."""
        if self._nav_goal_max_abs_y <= 0.0:
            return
        y = pose.pose.position.y
        limit = self._nav_goal_max_abs_y
        if side_hint_y is not None and abs(side_hint_y) > 0.08:
            side = 1.0 if side_hint_y > 0.0 else -1.0
            min_lane_abs_y = limit * 0.75
            lane_abs_y = min(limit, max(abs(y), min_lane_abs_y))
            pose.pose.position.y = side * lane_abs_y
            return
        pose.pose.position.y = max(-limit, min(limit, y))

    def _publish_fire_info(self, header, red_positions: list[PointStamped]):
        msg = FireInfo()
        msg.header           = header
        msg.red_door_count   = len(red_positions)
        msg.detected         = len(red_positions) > 0
        if red_positions:
            frame_id = red_positions[0].header.frame_id or self._frame
            same_frame = [
                p for p in red_positions
                if (p.header.frame_id or self._frame) == frame_id
            ]
            cx = float(sum(p.point.x for p in same_frame) / len(same_frame))
            cy = float(sum(p.point.y for p in same_frame) / len(same_frame))
            cz = float(sum(p.point.z for p in same_frame) / len(same_frame))

            fire_point = PointStamped()
            fire_point.header.frame_id = frame_id
            fire_point.header.stamp = header.stamp
            fire_point.point.x = cx
            fire_point.point.y = cy
            fire_point.point.z = cz

            if frame_id == 'map':
                msg.fire_position = fire_point
            else:
                try:
                    msg.fire_position = self._tf_buffer.transform(
                        fire_point, 'map',
                        timeout=rclpy.duration.Duration(seconds=0.1))
                except Exception:
                    msg.fire_position = fire_point
        self.fire_pub.publish(msg)

    def _get_door_id(self, source_name: str,
                     color: str, cx_pix: int, img_w: int,
                     dist: float | None = None) -> str:
        pixel_grid = cx_pix // max(1, img_w // 4)
        if dist is None or not math.isfinite(dist):
            range_grid = 'unknown'
        else:
            clipped = max(0.0, min(float(dist), self._max_detection_distance))
            range_grid = int(clipped // 2.0)
        key  = f'{source_name}_{color}_{pixel_grid}_{range_grid}'
        if key not in self._door_id_map:
            self._door_id_map[key] = f'door_{color}_{uuid.uuid4().hex[:6]}'
        return self._door_id_map[key]


def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
