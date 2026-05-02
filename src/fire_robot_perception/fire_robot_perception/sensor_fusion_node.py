"""
sensor_fusion_node.py

RGB 카메라 + 2D LiDAR 융합으로 Nav2용 OccupancyGrid 생성.

융합 방식:
  1. SegFormer(ADE20K) → RGB 이미지의 픽셀별 시맨틱 레이블
  2. 각 LiDAR 포인트를 카메라 이미지에 투영 → 해당 픽셀의 시맨틱 레이블 획득
  3. 레이블이 '장애물'이면 → 그리드 셀 점유 / '바닥'이면 → 빈 공간
  4. OccupancyGrid로 발행 → Nav2 costmap에 반영

SegFormer 모델: nvidia/segformer-b0-finetuned-ade-512-512 (경량)
  - ADE20K 150 클래스 사전학습 → 추가 학습 불필요
  - B0: 경량 (3.7M params, ~실시간 가능)
"""

import math
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from nav_msgs.msg import OccupancyGrid

try:
    import torch
    from transformers import (SegformerImageProcessor,
                              SegformerForSemanticSegmentation)
    from PIL import Image as PILImage
    _HAS_SEGFORMER = True
except ImportError:
    _HAS_SEGFORMER = False

# ── ADE20K 클래스 → 장애물 여부 ──────────────────────────
# 통행 불가 클래스 (벽, 가구, 사람 등)
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
    63,  # door (문 앞은 장애물로 처리, 통과 의도 시 별도 처리)
}
# 통행 가능 클래스 (바닥 등)
_FREE_IDS = {4, 6, 9, 52}   # floor, road, field, path

# ── 맵 파라미터 ──────────────────────────────────────────
MAP_RES    = 0.05   # m/cell
MAP_W      = 200    # cells (10m)
MAP_H      = 200    # cells (10m)
MAP_OX     = -5.0   # origin X
MAP_OY     = -5.0   # origin Y
ROBOT_GX   = MAP_W // 2
ROBOT_GY   = MAP_H // 2


class SensorFusionNode(Node):
    """RGB + LiDAR → OccupancyGrid 노드"""

    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.declare_parameter('segformer_model',
                               'nvidia/segformer-b0-finetuned-ade-512-512')
        self.declare_parameter('use_gpu',     False)
        self.declare_parameter('publish_rate', 2.0)
        # LiDAR-카메라 외부 파라미터 (카메라와 LiDAR의 상대 위치, 기본: 동일 위치)
        self.declare_parameter('lidar_to_cam_x', 0.0)
        self.declare_parameter('lidar_to_cam_y', 0.0)

        model_name = self.get_parameter('segformer_model').value
        use_gpu    = self.get_parameter('use_gpu').value
        rate       = self.get_parameter('publish_rate').value

        self.bridge = CvBridge()

        # 카메라 내부 파라미터
        self._fx = self._fy = 500.0
        self._cx = self._cy = 320.0
        self._img_w = 640
        self._img_h = 480

        self._latest_image: np.ndarray | None = None
        self._seg_labels:   np.ndarray | None = None  # (H, W) int32
        self._latest_scan:  LaserScan | None  = None
        self._lock = threading.Lock()

        # SegFormer 로드
        self._processor = None
        self._model      = None
        self._device     = 'cpu'

        if _HAS_SEGFORMER:
            try:
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
                    f'SegFormer load failed ({e}). LiDAR-only mode.')
        else:
            self.get_logger().warn(
                'transformers not installed. LiDAR-only mode.')

        # Subscribers
        self.create_subscription(
            Image, '/camera/color/image_raw', self.camera_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.camera_info_callback, 10)
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        # Publishers
        self.grid_pub   = self.create_publisher(OccupancyGrid, '/segmentation_map', 10)
        self.seg_vis    = self.create_publisher(Image, '/segmentation_visual', 10)

        # SegFormer는 별도 스레드에서 0.5s 주기 실행
        self._seg_thread = threading.Thread(
            target=self._segmentation_loop, daemon=True)
        self._seg_thread.start()

        self.create_timer(1.0 / rate, self.publish_map)
        self.get_logger().info('SensorFusionNode started')

    # ── 콜백 ──────────────────────────────────────────────
    def camera_info_callback(self, msg: CameraInfo):
        self._fx    = msg.k[0]
        self._fy    = msg.k[4]
        self._cx    = msg.k[2]
        self._cy    = msg.k[5]
        self._img_w = msg.width
        self._img_h = msg.height

    def camera_callback(self, msg: Image):
        with self._lock:
            self._latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def lidar_callback(self, msg: LaserScan):
        self._latest_scan = msg

    # ── SegFormer 추론 루프 ───────────────────────────────
    def _segmentation_loop(self):
        import time
        while rclpy.ok():
            image = None
            with self._lock:
                if self._latest_image is not None:
                    image = self._latest_image.copy()
            if image is not None and self._model is not None:
                self._run_segformer(image)
            time.sleep(0.5)

    def _run_segformer(self, image: np.ndarray):
        try:
            pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            inputs = self._processor(images=pil, return_tensors='pt')
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            up = torch.nn.functional.interpolate(
                logits,
                size=(image.shape[0], image.shape[1]),
                mode='bilinear', align_corners=False,
            )
            seg = up.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
            with self._lock:
                self._seg_labels = seg
        except Exception as e:
            self.get_logger().warn(f'SegFormer error: {e}')

    # ── 맵 발행 ──────────────────────────────────────────
    def publish_map(self):
        grid = np.full((MAP_H, MAP_W), -1, dtype=np.int8)

        with self._lock:
            seg = self._seg_labels.copy() if self._seg_labels is not None else None

        # LiDAR 포인트를 카메라에 투영 → 시맨틱 레이블 얻기
        if self._latest_scan is not None:
            self._fuse_lidar_with_seg(grid, self._latest_scan, seg)

        self._publish_grid(grid)

        # 시각화
        if seg is not None:
            self._publish_seg_visual(seg)

    def _fuse_lidar_with_seg(self,
                              grid: np.ndarray,
                              scan: LaserScan,
                              seg: np.ndarray | None):
        """
        각 LiDAR 포인트를 이미지에 투영하여 시맨틱 레이블을 얻은 뒤 그리드 업데이트.
        투영 불가(화각 밖) 포인트는 LiDAR 측정값만으로 장애물 처리.
        """
        angle = scan.angle_min
        for r in scan.ranges:
            if not (math.isfinite(r) and scan.range_min <= r <= scan.range_max):
                angle += scan.angle_increment
                continue

            # LiDAR 포인트의 robot frame 위치 (x=전방, y=좌)
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)

            # 그리드 셀
            gx = ROBOT_GX + int(lx / MAP_RES)
            gy = ROBOT_GY - int(ly / MAP_RES)

            if not (0 <= gx < MAP_W and 0 <= gy < MAP_H):
                angle += scan.angle_increment
                continue

            # 카메라 이미지에 투영 (LiDAR와 카메라 광학 축이 평행하다고 가정)
            label = self._get_seg_label(lx, ly, r, seg)

            if label in _FREE_IDS:
                if grid[gy, gx] != 100:
                    grid[gy, gx] = 0
            elif label in _OBSTACLE_IDS:
                grid[gy, gx] = 100
            else:
                # 레이블 불명: 근거리(0.5m 이내)만 장애물, 원거리는 unknown(−1) 유지
                if r < 0.5:
                    grid[gy, gx] = 100

            angle += scan.angle_increment

    def _get_seg_label(self,
                       lx: float, ly: float, dist: float,
                       seg: np.ndarray | None) -> int:
        """LiDAR 포인트 (robot frame) → 카메라 이미지 픽셀 → 시맨틱 레이블"""
        if seg is None:
            return -1
        # 카메라는 로봇 전방(+X)을 바라봄
        if lx <= 0:
            return -1
        # 핀홀 투영: u = fx * (ly / lx) + cx
        u = int(self._fx * (-ly / lx) + self._cx)   # y는 카메라에서 반대방향
        # v는 바닥면 가정: 카메라 틸트 고려 없이 중하단 고정
        v = int(self._img_h * 0.7)
        if not (0 <= u < self._img_w and 0 <= v < self._img_h):
            return -1
        return int(seg[v, u])

    def _publish_grid(self, grid: np.ndarray):
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = MAP_RES
        msg.info.width      = MAP_W
        msg.info.height     = MAP_H
        msg.info.origin.position.x    = MAP_OX
        msg.info.origin.position.y    = MAP_OY
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.grid_pub.publish(msg)

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
