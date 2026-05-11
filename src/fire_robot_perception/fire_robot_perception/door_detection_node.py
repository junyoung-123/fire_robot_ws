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
import uuid
from pathlib import Path

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
# Gazebo exit_marker ambient(0.05, 0.85, 0.15) → OpenCV H≈64
GREEN_LOWER = np.array([ 50, 100,  80])
GREEN_UPPER = np.array([ 80, 255, 255])

# bbox 내 색상 픽셀이 이 비율 이상이면 해당 색으로 판정
COLOR_RATIO_THRESHOLD = 0.20

CAMERA_HFOV_RAD = 1.204   # RealSense D435 기준 ~69도


class DoorDetectionNode(Node):
    """YOLOv8 + HSV + Radar(+ Depth) 기반 파란/빨간 문 탐지 노드"""

    def __init__(self):
        super().__init__('door_detection_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.40)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('camera_hfov_deg', 69.0)
        self.declare_parameter('use_depth_camera', False)

        model_path       = self.get_parameter('model_path').value
        self._conf       = self.get_parameter('confidence_threshold').value
        self._frame      = self.get_parameter('frame_id').value
        hfov_deg         = self.get_parameter('camera_hfov_deg').value
        self._hfov       = math.radians(hfov_deg)
        self._use_depth  = self.get_parameter('use_depth_camera').value

        self.bridge = CvBridge()

        # 카메라 내부 파라미터 기본값 (CameraInfo 수신 전)
        self._fx = 500.0
        self._fy = 500.0
        self._cx = 320.0
        self._cy = 240.0
        self._img_w = 640
        self._img_h = 480

        self._latest_scan:  LaserScan | None  = None
        self._latest_depth: np.ndarray | None = None   # (H, W) float32 [m]
        self._door_id_map:  dict[str, str]    = {}

        # TF: base_link → map 변환 (감지 시점에 즉시 변환해 stale 좌표 방지)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        cb = ReentrantCallbackGroup()

        # Subscribers
        self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10,
            callback_group=cb)
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.camera_info_callback, 10)
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
        self._model = self._load_model(model_path)

        self.get_logger().info(
            f'DoorDetectionNode started | YOLO={"OK" if self._model else "FALLBACK_HSV"}'
            f' | depth={"ON" if self._use_depth else "OFF"}')

    # ── 모델 로드 ─────────────────────────────────────────
    def _load_model(self, model_path: str):
        if not _HAS_YOLO:
            self.get_logger().warn(
                'ultralytics not installed. Using HSV-only detection.')
            return None
        path = Path(model_path)
        if path.exists():
            self.get_logger().info(f'Loading custom YOLO model: {path}')
            return YOLO(str(path))
        # 경로 미지정 또는 파일 없음 → HSV 전용 모드
        # (COCO 사전학습 모델은 Door 클래스를 포함하지 않으므로 사용 불가)
        self.get_logger().warn(
            f'model_path "{model_path}" not found. Falling back to HSV-only detection.')
        return None

    # ── 콜백 ──────────────────────────────────────────────
    def camera_info_callback(self, msg: CameraInfo):
        self._fx    = msg.k[0]
        self._fy    = msg.k[4]
        self._cx    = msg.k[2]
        self._cy    = msg.k[5]
        self._img_w = msg.width
        self._img_h = msg.height

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

    def image_callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self._process(image, msg.header)

    # ── 메인 처리 ─────────────────────────────────────────
    def _process(self, image: np.ndarray, header):
        debug = image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        detections = self._detect_doors(image)  # list of (x1,y1,x2,y2,conf)
        red_positions: list[PointStamped] = []

        for (x1, y1, x2, y2, det_conf) in detections:
            roi = hsv[y1:y2, x1:x2]
            color = self._classify_color(roi)
            if color == 'unknown':
                continue

            cx_pix = (x1 + x2) // 2
            cy_pix = (y1 + y2) // 2
            dist   = self._lidar_distance_at_pixel(cx_pix, cy_pix)
            if dist is None:
                continue

            door_id  = self._get_door_id(color, cx_pix, image.shape[1])
            door_msg = self._build_door_info(
                header, door_id, color, cx_pix, dist,
                bbox_height=(y2 - y1), det_conf=det_conf)
            self.door_pub.publish(door_msg)

            if color == 'red':
                red_positions.append(door_msg.door_pose.pose.position)

            # 디버그 드로잉 (BGR: blue, red, green)
            _DBG = {'blue': (255, 100, 0), 'red': (0, 60, 255), 'green': (0, 200, 50)}
            col  = _DBG.get(color, (200, 200, 200))
            cv2.rectangle(debug, (x1, y1), (x2, y2), col, 3)
            cv2.putText(debug, f'{color} {det_conf:.2f} d={dist:.1f}m',
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        self._publish_fire_info(header, red_positions)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))

    # ── 문 탐지 ───────────────────────────────────────────
    def _detect_doors(self, image: np.ndarray) -> list:
        """YOLOv8으로 문 bbox 반환. 모델 없으면 HSV contour fallback."""
        if self._model is not None:
            return self._yolo_detect(image)
        return self._hsv_fallback_detect(image)

    def _yolo_detect(self, image: np.ndarray) -> list:
        results = self._model(image, conf=self._conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_name = self._model.names[int(box.cls)]
                # 학습된 모델에서 'door' 클래스만, 기본 모델은 모든 클래스 허용
                if 'door' not in cls_name.lower() and len(self._model.names) > 10:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                # 너무 작은 bbox 제외 (이미지 면적의 1% 미만)
                area = (x2 - x1) * (y2 - y1)
                if area < image.shape[0] * image.shape[1] * 0.01:
                    continue
                boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _hsv_fallback_detect(self, image: np.ndarray) -> list:
        """YOLO 없을 때 HSV 기반 contour로 문 후보 검출"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue_mask  = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
        red_mask   = (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1) |
                      cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        combined   = blue_mask | red_mask | green_mask
        kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        min_area = image.shape[0] * image.shape[1] * 0.02
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            conf = min(1.0, cv2.contourArea(cnt) / (min_area * 5))
            boxes.append((x, y, x + w, y + h, conf))
        return boxes

    # ── 색상 분류 ─────────────────────────────────────────
    def _classify_color(self, roi_hsv: np.ndarray) -> str:
        if roi_hsv.size == 0:
            return 'unknown'
        total     = roi_hsv.shape[0] * roi_hsv.shape[1]
        blue_r    = cv2.countNonZero(
            cv2.inRange(roi_hsv, BLUE_LOWER, BLUE_UPPER)) / total
        red_r     = cv2.countNonZero(
            cv2.inRange(roi_hsv, RED_LOWER1, RED_UPPER1) |
            cv2.inRange(roi_hsv, RED_LOWER2, RED_UPPER2)) / total
        green_r   = cv2.countNonZero(
            cv2.inRange(roi_hsv, GREEN_LOWER, GREEN_UPPER)) / total
        best, ratio = max(
            [('blue', blue_r), ('red', red_r), ('green', green_r)],
            key=lambda x: x[1])
        return best if ratio > COLOR_RATIO_THRESHOLD else 'unknown'

    # ── 거리 추정 ─────────────────────────────────────────
    def _lidar_distance_at_pixel(self, cx_pix: int,
                                  cy_pix: int | None = None) -> float | None:
        """거리 추정: Depth 카메라 우선, 없으면 Radar /scan fallback."""
        if self._use_depth and self._latest_depth is not None:
            dist = self._depth_distance_at_pixel(cx_pix, cy_pix)
            if dist is not None:
                return dist
        return self._radar_distance_at_pixel(cx_pix)

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

    def _radar_distance_at_pixel(self, cx_pix: int) -> float | None:
        """이미지 픽셀 x → 수평각 → Radar /scan range."""
        if self._latest_scan is None:
            return None
        scan = self._latest_scan
        angle = math.atan((cx_pix - self._cx) / self._fx)
        idx = int((angle - scan.angle_min) / scan.angle_increment)
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
    def _build_door_info(self, header, door_id: str, color: str,
                         cx_pix: int, dist: float,
                         bbox_height: int, det_conf: float) -> DoorInfo:
        msg = DoorInfo()
        msg.header = header

        # 카메라 수평각 → robot frame (x = 전방, y = 좌)
        angle = math.atan((cx_pix - self._cx) / self._fx)
        px = dist * math.cos(angle)
        py = dist * math.sin(angle)

        # base_link → map 프레임 변환 (감지 시점에 즉시 변환)
        pose_base = PoseStamped()
        pose_base.header.frame_id = self._frame
        pose_base.header.stamp    = header.stamp
        pose_base.pose.position.x = px
        pose_base.pose.position.y = py
        pose_base.pose.orientation.w = 1.0
        try:
            pose_map = self._tf_buffer.transform(
                pose_base, 'map',
                timeout=rclpy.duration.Duration(seconds=0.1))
            msg.door_pose = pose_map
            msg.handle_position.header   = pose_map.header
            msg.handle_position.point.x  = pose_map.pose.position.x
            msg.handle_position.point.y  = pose_map.pose.position.y
            msg.handle_position.point.z  = 0.9
        except Exception:
            # SLAM 맵 초기화 전 → base_link 그대로 (Nav2 목표 전달 시 주의)
            msg.door_pose.header.frame_id         = self._frame
            msg.door_pose.header.stamp            = header.stamp
            msg.door_pose.pose.position.x         = px
            msg.door_pose.pose.position.y         = py
            msg.door_pose.pose.orientation.w      = 1.0
            msg.handle_position.header.frame_id   = self._frame
            msg.handle_position.point.x           = px
            msg.handle_position.point.y           = py
            msg.handle_position.point.z           = 0.9

        msg.door_id    = door_id
        msg.door_color = color
        msg.is_open    = False
        msg.confidence = float(det_conf)
        msg.distance_from_fire = 0.0
        return msg

    def _publish_fire_info(self, header, red_positions: list):
        msg = FireInfo()
        msg.header           = header
        msg.red_door_count   = len(red_positions)
        msg.detected         = len(red_positions) > 0
        if red_positions:
            cx = float(sum(p.x for p in red_positions) / len(red_positions))
            cy = float(sum(p.y for p in red_positions) / len(red_positions))
            # map 프레임으로 변환 시도 — 성공 시 FSM에서 거리 비교에 사용 가능
            pt_base = PointStamped()
            pt_base.header.frame_id = self._frame
            pt_base.header.stamp    = header.stamp
            pt_base.point.x = cx
            pt_base.point.y = cy
            pt_base.point.z = 0.0
            try:
                pt_map = self._tf_buffer.transform(
                    pt_base, 'map',
                    timeout=rclpy.duration.Duration(seconds=0.1))
                msg.fire_position = pt_map
            except Exception:
                msg.fire_position.header.frame_id = self._frame
                msg.fire_position.header.stamp    = header.stamp
                msg.fire_position.point.x = cx
                msg.fire_position.point.y = cy
                msg.fire_position.point.z = 0.0
        self.fire_pub.publish(msg)

    def _get_door_id(self, color: str, cx_pix: int, img_w: int) -> str:
        grid = cx_pix // (img_w // 4)
        key  = f'{color}_{grid}'
        if key not in self._door_id_map:
            self._door_id_map[key] = f'door_{color}_{uuid.uuid4().hex[:6]}'
        return self._door_id_map[key]


def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
