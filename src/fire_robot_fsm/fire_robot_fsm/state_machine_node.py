import math

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from enum import IntEnum
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from fire_robot_interfaces.msg import DoorInfo, FireInfo, RobotState
from fire_robot_interfaces.srv import OpenDoor


class State(IntEnum):
    IDLE             = 0
    EXPLORING        = 1   # 파란 문 탐색 (제자리 회전)
    NAVIGATING       = 2   # 파란 문으로 이동
    OPENING_DOOR     = 3   # 문 개방 중
    DOOR_OPENED      = 4   # 개방 완료 → 즉시 다음 문 탐색
    EXITING          = 5   # 모든 문 개방 완료, 비상구로 이동
    MISSION_COMPLETE = 6   # 비상구 도착, 미션 종료
    EMERGENCY_STOP   = 7
    EXPLORING_NAVIGATING = 8   # segment map 기반 탐색 waypoint 이동


class StateMachineNode(Node):
    """전체 시스템을 조율하는 유한 상태 머신(FSM) 노드

    미션 흐름:
      1. 빨간 문(화재) 감지 → 탐색 시작
      2. 파란 문 발견 → 이동 → 개방 → 다음 파란 문 탐색 (반복)
      3. explore_timeout_sec 동안 새 파란 문 없음 → 모든 문 개방 완료로 판단
      4. 비상구(exit_x, exit_y)로 이동 → 미션 완료

    파라미터:
      nav_timeout_sec    : 문 이동 타임아웃 (기본 60s)
      max_nav_retries    : 최대 재시도 횟수 (기본 3)
      explore_timeout_sec: 탐색 회전 후 비상구 이동 판단 시간 (기본 30s)
      exit_x / exit_y    : 비상구 map 프레임 좌표 (기본 10.0 / 0.0)
      exit_yaw           : 비상구 도착 방향 rad (기본 0.0)
    """

    def __init__(self):
        super().__init__('state_machine_node')

        self.declare_parameter('nav_timeout_sec',     60.0)
        self.declare_parameter('max_nav_retries',     3)
        self.declare_parameter('explore_timeout_sec', 30.0)
        self.declare_parameter('explore_linear_vel',  0.0)
        self.declare_parameter('explore_angular_vel', 0.3)
        self.declare_parameter('explore_obstacle_stop_m', 0.85)
        self.declare_parameter('explore_obstacle_slow_m', 1.40)
        self.declare_parameter('explore_avoid_angular_vel', 0.55)
        self.declare_parameter('explore_front_angle_deg', 28.0)
        self.declare_parameter('explore_side_angle_deg', 85.0)
        self.declare_parameter('explore_side_clearance_m', 0.65)
        self.declare_parameter('explore_scan_min_range_m', 0.85)
        self.declare_parameter('explore_color_lane_bias', True)
        self.declare_parameter('explore_forward_yaw', 0.0)
        self.declare_parameter('explore_heading_kp', 0.8)
        self.declare_parameter('explore_heading_soft_limit_deg', 45.0)
        self.declare_parameter('explore_heading_hard_limit_deg', 105.0)
        self.declare_parameter('explore_center_y', 0.0)
        self.declare_parameter('explore_y_soft_limit_m', 0.90)
        self.declare_parameter('explore_y_hard_limit_m', 1.35)
        self.declare_parameter('explore_center_lookahead_m', 1.20)
        self.declare_parameter('prefer_forward_doors', True)
        self.declare_parameter('mission_start_x', 0.0)
        self.declare_parameter('next_door_min_forward_m', 0.75)
        self.declare_parameter('opened_door_merge_dist_m', 1.0)
        self.declare_parameter('min_opened_doors_before_exit', 0)
        self.declare_parameter('complete_when_past_exit_x', False)
        self.declare_parameter('exit_complete_margin_m', 0.50)
        self.declare_parameter('final_scan_before_exit_sec', 0.0)
        self.declare_parameter('final_scan_angular_vel', 0.45)
        self.declare_parameter('pre_nav_scan_sec', 0.0)
        self.declare_parameter('pre_nav_scan_angular_vel', 0.0)
        self.declare_parameter('explore_nav_enabled', True)
        self.declare_parameter('explore_nav_step_m', 1.20)
        self.declare_parameter('explore_nav_lane_y_m', 0.75)
        self.declare_parameter('explore_nav_lane_width_m', 1.35)
        self.declare_parameter('explore_nav_scan_lookahead_m', 2.80)
        self.declare_parameter('explore_nav_color_bias_score', 0.35)
        self.declare_parameter('explore_nav_keep_lane_bias_score', 0.12)
        self.declare_parameter('explore_nav_front_obstacle_width_m', 1.35)
        self.declare_parameter('explore_nav_front_obstacle_bias_score', 3.0)
        self.declare_parameter('explore_nav_blue_door_bias_score', 2.2)
        self.declare_parameter('explore_nav_blue_lane_min_clearance_m', 0.80)
        self.declare_parameter('explore_nav_blue_lane_bias_max_dist_m', 3.0)
        self.declare_parameter('max_explore_nav_failures_before_exit', 8)
        self.declare_parameter('explore_nav_timeout_sec', 45.0)
        self.declare_parameter('min_target_door_abs_y_m', 0.0)
        self.declare_parameter('target_door_direct_nav_max_dist_m', 0.0)
        self.declare_parameter('nav_start_max_abs_y_m', 0.95)
        self.declare_parameter('nav_start_max_heading_error_deg', 85.0)
        self.declare_parameter('start_without_fire',  False)
        self.declare_parameter('exit_x',   10.0)
        self.declare_parameter('exit_y',    0.0)
        self.declare_parameter('exit_yaw',  0.0)
        self.declare_parameter('use_detected_exit', True)
        self.declare_parameter('detected_exit_min_x_margin_m', 3.0)
        self.declare_parameter('detected_exit_max_y_error_m', 1.5)

        self._nav_timeout_sec     = self.get_parameter('nav_timeout_sec').value
        self._max_nav_retries     = self.get_parameter('max_nav_retries').value
        self._explore_timeout_sec = self.get_parameter('explore_timeout_sec').value
        self._explore_linear_vel  = self.get_parameter('explore_linear_vel').value
        self._explore_angular_vel = self.get_parameter('explore_angular_vel').value
        self._explore_obstacle_stop_m = float(self.get_parameter('explore_obstacle_stop_m').value)
        self._explore_obstacle_slow_m = float(self.get_parameter('explore_obstacle_slow_m').value)
        self._explore_avoid_angular_vel = float(self.get_parameter('explore_avoid_angular_vel').value)
        self._explore_front_angle = math.radians(float(self.get_parameter('explore_front_angle_deg').value))
        self._explore_side_angle = math.radians(float(self.get_parameter('explore_side_angle_deg').value))
        self._explore_side_clearance_m = float(self.get_parameter('explore_side_clearance_m').value)
        self._explore_scan_min_range_m = float(self.get_parameter('explore_scan_min_range_m').value)
        self._explore_color_lane_bias = bool(self.get_parameter('explore_color_lane_bias').value)
        self._explore_forward_yaw = float(self.get_parameter('explore_forward_yaw').value)
        self._explore_heading_kp = float(self.get_parameter('explore_heading_kp').value)
        self._explore_heading_soft_limit = math.radians(
            float(self.get_parameter('explore_heading_soft_limit_deg').value))
        self._explore_heading_hard_limit = math.radians(
            float(self.get_parameter('explore_heading_hard_limit_deg').value))
        self._explore_center_y = float(self.get_parameter('explore_center_y').value)
        self._explore_y_soft_limit_m = float(self.get_parameter('explore_y_soft_limit_m').value)
        self._explore_y_hard_limit_m = float(self.get_parameter('explore_y_hard_limit_m').value)
        self._explore_center_lookahead_m = float(
            self.get_parameter('explore_center_lookahead_m').value)
        self._prefer_forward_doors = bool(self.get_parameter('prefer_forward_doors').value)
        self._mission_start_x = float(self.get_parameter('mission_start_x').value)
        self._next_door_min_forward_m = float(self.get_parameter('next_door_min_forward_m').value)
        self._opened_door_merge_dist_m = float(self.get_parameter('opened_door_merge_dist_m').value)
        self._min_opened_doors_before_exit = int(
            self.get_parameter('min_opened_doors_before_exit').value)
        self._complete_when_past_exit_x = bool(
            self.get_parameter('complete_when_past_exit_x').value)
        self._exit_complete_margin_m = float(
            self.get_parameter('exit_complete_margin_m').value)
        self._final_scan_before_exit_sec = float(
            self.get_parameter('final_scan_before_exit_sec').value)
        self._final_scan_angular_vel = float(
            self.get_parameter('final_scan_angular_vel').value)
        self._pre_nav_scan_sec = float(
            self.get_parameter('pre_nav_scan_sec').value)
        self._pre_nav_scan_angular_vel = float(
            self.get_parameter('pre_nav_scan_angular_vel').value)
        self._explore_nav_enabled = bool(
            self.get_parameter('explore_nav_enabled').value)
        self._explore_nav_step_m = float(
            self.get_parameter('explore_nav_step_m').value)
        self._explore_nav_lane_y_m = float(
            self.get_parameter('explore_nav_lane_y_m').value)
        self._explore_nav_lane_width_m = float(
            self.get_parameter('explore_nav_lane_width_m').value)
        self._explore_nav_scan_lookahead_m = float(
            self.get_parameter('explore_nav_scan_lookahead_m').value)
        self._explore_nav_color_bias_score = float(
            self.get_parameter('explore_nav_color_bias_score').value)
        self._explore_nav_keep_lane_bias_score = float(
            self.get_parameter('explore_nav_keep_lane_bias_score').value)
        self._explore_nav_front_obstacle_width_m = float(
            self.get_parameter('explore_nav_front_obstacle_width_m').value)
        self._explore_nav_front_obstacle_bias_score = float(
            self.get_parameter('explore_nav_front_obstacle_bias_score').value)
        self._explore_nav_blue_door_bias_score = float(
            self.get_parameter('explore_nav_blue_door_bias_score').value)
        self._explore_nav_blue_lane_min_clearance_m = float(
            self.get_parameter('explore_nav_blue_lane_min_clearance_m').value)
        self._explore_nav_blue_lane_bias_max_dist_m = float(
            self.get_parameter('explore_nav_blue_lane_bias_max_dist_m').value)
        self._max_explore_nav_failures_before_exit = int(
            self.get_parameter('max_explore_nav_failures_before_exit').value)
        self._explore_nav_timeout_sec = float(
            self.get_parameter('explore_nav_timeout_sec').value)
        self._min_target_door_abs_y_m = float(
            self.get_parameter('min_target_door_abs_y_m').value)
        self._target_door_direct_nav_max_dist_m = float(
            self.get_parameter('target_door_direct_nav_max_dist_m').value)
        self._nav_start_max_abs_y_m = float(self.get_parameter('nav_start_max_abs_y_m').value)
        self._nav_start_max_heading_error = math.radians(
            float(self.get_parameter('nav_start_max_heading_error_deg').value))
        self._start_without_fire  = self.get_parameter('start_without_fire').value
        self._exit_x   = self.get_parameter('exit_x').value
        self._exit_y   = self.get_parameter('exit_y').value
        self._exit_yaw = self.get_parameter('exit_yaw').value
        self._use_detected_exit = bool(self.get_parameter('use_detected_exit').value)
        self._detected_exit_min_x_margin_m = float(
            self.get_parameter('detected_exit_min_x_margin_m').value)
        self._detected_exit_max_y_error_m = float(
            self.get_parameter('detected_exit_max_y_error_m').value)

        cb_group = ReentrantCallbackGroup()

        self.door_sub = self.create_subscription(
            DoorInfo, '/detected_door', self.door_callback, 10,
            callback_group=cb_group)
        self.fire_sub = self.create_subscription(
            FireInfo, '/fire_info', self.fire_callback, 10,
            callback_group=cb_group)
        self.nav_done_sub = self.create_subscription(
            Bool, '/navigation_done', self.nav_done_callback, 10,
            callback_group=cb_group)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10,
            callback_group=cb_group)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.state_pub       = self.create_publisher(RobotState, '/robot_state', 10)
        self.target_door_pub = self.create_publisher(DoorInfo,   '/target_door', 10)
        self.cmd_vel_pub     = self.create_publisher(Twist,      '/cmd_vel',     10)

        self.open_door_client = self.create_client(
            OpenDoor, '/open_door', callback_group=cb_group)

        self.state              = State.IDLE
        self.detected_doors:    list[DoorInfo] = []
        self.fire_info:         FireInfo | None = None
        self.target_door:       DoorInfo | None = None
        self._exit_door:        DoorInfo | None = None   # 탐지된 초록 비상구
        self._latest_scan:      LaserScan | None = None
        self._last_lane_sign    = 0.0

        self._nav_done   = False
        self._nav_failed = False
        self._nav_start_time:     Time | None = None
        self._explore_start_time: Time | None = None

        self._failed_door_ids: set[str] = set()   # 네비게이션 실패 문
        self._opened_door_ids: set[str] = set()   # 개방 완료 문
        self._mission_progress_x = self._mission_start_x
        self._nav_retry_count = 0

        self._door_opening_in_progress = False
        self._final_scan_start_time: Time | None = None
        self._pre_nav_scan_start_time: Time | None = None
        self._pending_nav_scan_target_id: str | None = None
        self._explore_nav_goal_id: str = ''
        self._explore_nav_failure_count = 0

        self.timer = self.create_timer(0.5, self.fsm_loop)
        self.get_logger().info(
            f'StateMachineNode started | '
            f'nav_timeout={self._nav_timeout_sec}s  '
            f'max_retries={self._max_nav_retries}  '
            f'explore_timeout={self._explore_timeout_sec}s  '
            f'exit=({self._exit_x:.1f}, {self._exit_y:.1f})'
        )

    # ── 콜백 ──────────────────────────────────────────────
    def door_callback(self, msg: DoorInfo):
        # 초록 문(비상구) → 별도 저장, detected_doors에 추가하지 않음
        if msg.door_color == 'green':
            if self._exit_door is None:
                self.get_logger().info(
                    f'비상구 탐지: {msg.door_id} '
                    f'({msg.door_pose.pose.position.x:.1f}, '
                    f'{msg.door_pose.pose.position.y:.1f})')
            self._exit_door = msg
            return

        # 1차: 동일 door_id → 갱신
        existing = next(
            (d for d in self.detected_doors if d.door_id == msg.door_id), None)
        if existing:
            self._merge_door_observation(existing, msg)
            return

        # 2차: 탐색 회전 중 같은 문이 이미지 X 위치에 따라 다른 ID로 발급될 수 있음
        # map 프레임 기준 0.8 m 이내 + 같은 색이면 동일 문으로 판단 → 기존 항목 갱신
        px = msg.door_pose.pose.position.x
        py = msg.door_pose.pose.position.y
        nearby = next(
            (d for d in self.detected_doors
             if d.door_color == msg.door_color
             and math.hypot(d.door_pose.pose.position.x - px,
                            d.door_pose.pose.position.y - py) < 0.8),
            None)
        if nearby:
            self._merge_door_observation(nearby, msg)
            return
        self.detected_doors.append(msg)

    def fire_callback(self, msg: FireInfo):
        self.fire_info = msg

    def scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def nav_done_callback(self, msg: Bool):
        if self.state not in (State.NAVIGATING, State.EXITING, State.EXPLORING_NAVIGATING):
            return
        if msg.data:
            self.get_logger().info('Navigation succeeded.')
            self._nav_done   = True
            self._nav_failed = False
        else:
            self.get_logger().warn('Navigation failed.')
            self._nav_failed = True
            self._nav_done   = False

    def _merge_door_observation(self, existing: DoorInfo, msg: DoorInfo):
        if (existing.door_pose.header.frame_id == 'map'
                and msg.door_pose.header.frame_id == 'map'):
            ex = existing.door_pose.pose.position
            mp = msg.door_pose.pose.position
            jump = math.hypot(mp.x - ex.x, mp.y - ex.y)

            # Side-camera visual range can briefly jump when a door is partly
            # occluded. Keep the stable estimate unless the new observation is
            # clearly stronger.
            if jump > 2.0 and msg.confidence < max(0.35, existing.confidence * 0.9):
                return

            alpha = 0.25 if jump <= 2.0 else 0.10
            ex.x = (1.0 - alpha) * ex.x + alpha * mp.x
            ex.y = (1.0 - alpha) * ex.y + alpha * mp.y
            ex.z = mp.z
            existing.door_pose.header = msg.door_pose.header
            existing.door_pose.pose.orientation = msg.door_pose.pose.orientation

            eh = existing.handle_position.point
            mh = msg.handle_position.point
            eh.x = (1.0 - alpha) * eh.x + alpha * mh.x
            eh.y = (1.0 - alpha) * eh.y + alpha * mh.y
            eh.z = mh.z
            existing.handle_position.header = msg.handle_position.header
        else:
            existing.door_pose = msg.door_pose
            existing.handle_position = msg.handle_position

        existing.header = msg.header
        existing.confidence = max(existing.confidence, msg.confidence)
        existing.is_open = existing.is_open or msg.is_open
        existing.distance_from_fire = msg.distance_from_fire

    # ── FSM 루프 ──────────────────────────────────────────
    def fsm_loop(self):
        if   self.state == State.IDLE:             self._on_idle()
        elif self.state == State.EXPLORING:        self._on_exploring()
        elif self.state == State.NAVIGATING:       self._on_navigating()
        elif self.state == State.EXPLORING_NAVIGATING: self._on_exploring_navigating()
        elif self.state == State.OPENING_DOOR:     self._on_opening_door()
        elif self.state == State.DOOR_OPENED:      self._on_door_opened()
        elif self.state == State.EXITING:          self._on_exiting()
        elif self.state == State.MISSION_COMPLETE: self._on_mission_complete()
        elif self.state == State.EMERGENCY_STOP:   self._on_emergency_stop()

        self._publish_state()

    # ── 상태별 동작 ───────────────────────────────────────
    def _on_idle(self):
        if self._start_without_fire:
            self.get_logger().info(
                'start_without_fire enabled. Starting corridor exploration.')
            self._failed_door_ids.clear()
            self._opened_door_ids.clear()
            self._mission_progress_x = self._mission_start_x
            self._exit_door = None
            self._nav_retry_count = 0
            self._transition(State.EXPLORING)
            return

        if self.fire_info and self.fire_info.detected:
            self.get_logger().info(
                f'Fire detected ({self.fire_info.red_door_count} red door(s)). '
                'Starting mission.')
            self._failed_door_ids.clear()
            self._opened_door_ids.clear()
            self._mission_progress_x = self._mission_start_x
            self._exit_door = None
            self._nav_retry_count = 0
            self._transition(State.EXPLORING)

    def _on_exploring(self):
        # 개방 완료 + 실패 블랙리스트 제외한 파란 문 후보
        safe_doors = [
            d for d in self.detected_doors
            if d.door_color == 'blue'
            and d.door_id not in self._failed_door_ids
            and d.door_id not in self._opened_door_ids
        ]

        safe_doors = self._filter_forward_doors(safe_doors)

        if (len(self._opened_door_ids) >= self._min_opened_doors_before_exit
                and self._past_exit_line()):
            self.get_logger().info('비상구 라인 통과 확인. 미션 완료로 전환합니다.')
            self._transition(State.MISSION_COMPLETE)
            return

        if not safe_doors:
            # 탐색 타임아웃 → 모든 파란 문 개방 완료 판단
            if self._explore_start_time is not None:
                elapsed = (
                    self.get_clock().now() - self._explore_start_time
                ).nanoseconds / 1e9
                if elapsed >= self._explore_timeout_sec:
                    opened_count = len(self._opened_door_ids)
                    if opened_count < self._min_opened_doors_before_exit:
                        self.get_logger().warn(
                            f'{elapsed:.0f}s 탐색 후 새 파란 문 없음. '
                            f'아직 개방한 문이 {opened_count}/'
                            f'{self._min_opened_doors_before_exit}개라 탐색을 계속합니다.')
                        self._explore_start_time = self.get_clock().now()
                        self._rotate_to_scan()
                        return

                    if self._run_final_scan_before_exit():
                        return

                    self.get_logger().info(
                        f'{elapsed:.0f}s 탐색 후 새 파란 문 없음. '
                        f'총 {opened_count}개 문 개방 완료. '
                        '비상구로 이동합니다.')
                    self._transition(State.EXITING)
                    return

            if self._send_explore_waypoint():
                return

            # waypoint를 보낼 수 없을 때만 정지/회전 스캔
            self._rotate_to_scan()
            return

        # 새 파란 문 발견 → 탐색 타이머 리셋
        self._explore_start_time = self.get_clock().now()

        if not self._ready_to_start_navigation():
            self._rotate_to_scan()
            return

        # 화재 위치에서 가장 먼 파란 문 선택 (화재 반대편 출구 우선)
        # fire_position은 base_link 프레임 — door_pose는 map 프레임이므로
        # 프레임이 일치할 때만 거리 비교 사용, 아니면 confidence 기반 선택
        fire_pos = (self.fire_info.fire_position
                    if self.fire_info and self.fire_info.detected else None)
        fire_in_map = (fire_pos is not None
                       and fire_pos.header.frame_id == 'map')
        if self._prefer_forward_doors:
            self.target_door = self._nearest_forward_blue_door(safe_doors)
        elif fire_in_map:
            fx = fire_pos.point.x
            fy = fire_pos.point.y
            self.target_door = max(
                safe_doors,
                key=lambda d: math.hypot(
                    d.door_pose.pose.position.x - fx,
                    d.door_pose.pose.position.y - fy,
                )
            )
        else:
            self.target_door = max(safe_doors, key=lambda d: d.confidence)

        if self._defer_far_target_door(self.target_door):
            return

        if self._run_pre_nav_scan(self.target_door):
            return

        self.get_logger().info(
            f'목표 문: {self.target_door.door_id}  '
            f'개방 완료: {len(self._opened_door_ids)}개  '
            f'재시도: {self._nav_retry_count}/{self._max_nav_retries}')
        self.target_door_pub.publish(self.target_door)
        self._nav_done   = False
        self._nav_failed = False
        self._transition(State.NAVIGATING)

    def _on_navigating(self):
        if self._nav_start_time is not None:
            elapsed = (
                self.get_clock().now() - self._nav_start_time
            ).nanoseconds / 1e9
            if elapsed > self._nav_timeout_sec:
                self.get_logger().warn(f'Navigation timeout ({elapsed:.1f}s).')
                self._handle_nav_failure(reason='timeout')
                return

        if self._nav_done:
            self._nav_done = False
            self._door_opening_in_progress = False
            self._transition(State.OPENING_DOOR)
        elif self._nav_failed:
            self._nav_failed = False
            self._handle_nav_failure(reason='nav2_failure')

    def _on_exploring_navigating(self):
        if self._nav_start_time is not None:
            elapsed = (
                self.get_clock().now() - self._nav_start_time
            ).nanoseconds / 1e9
            if elapsed > self._explore_nav_timeout_sec:
                self.get_logger().warn(
                    f'탐색 waypoint 이동 timeout ({elapsed:.1f}s). 반대 차선으로 다시 스캔합니다.')
                if self._handle_explore_nav_failure_limit():
                    return
                self._flip_explore_lane_after_failure()
                self._nav_done = False
                self._nav_failed = False
                self._transition(State.EXPLORING)
                return

        if self._nav_done:
            self._nav_done = False
            self._explore_nav_failure_count = 0
            self.get_logger().info('탐색 waypoint 도착. LiDAR+camera 재스캔을 계속합니다.')
            self._alternate_explore_lane_after_success()
            self._transition(State.EXPLORING)
        elif self._nav_failed:
            self._nav_failed = False
            if self._handle_explore_nav_failure_limit():
                return
            self._flip_explore_lane_after_failure()
            self.get_logger().warn('탐색 waypoint 이동 실패. 반대 차선 우선으로 재스캔합니다.')
            self._transition(State.EXPLORING)

    def _handle_explore_nav_failure_limit(self) -> bool:
        self._explore_nav_failure_count += 1
        if (len(self._opened_door_ids) >= self._min_opened_doors_before_exit
                and self._explore_nav_failure_count
                >= self._max_explore_nav_failures_before_exit):
            self.get_logger().warn(
                f'탐색 waypoint 실패 {self._explore_nav_failure_count}회 누적. '
                '남은 문 후보 탐색을 중단하고 비상구로 전환합니다.')
            self._explore_nav_failure_count = 0
            self._transition(State.EXITING)
            return True
        return False

    def _handle_nav_failure(self, reason: str = ''):
        door_id = self.target_door.door_id if self.target_door else '(unknown)'
        self._nav_retry_count += 1
        self.get_logger().warn(
            f'Nav failure [{reason}] → {door_id}. '
            f'재시도 {self._nav_retry_count}/{self._max_nav_retries}')

        if self._nav_retry_count >= self._max_nav_retries:
            if self.target_door is not None:
                self._failed_door_ids.add(self.target_door.door_id)
                if self.target_door.door_color == 'blue':
                    failed_id = self.target_door.door_id
                    self.target_door = None
                    self._nav_retry_count = 0
                    self.get_logger().error(
                        f'Max retries 초과: {failed_id}. '
                        '해당 파란 문 후보를 제외하고 탐색을 계속합니다.')
                    self._transition(State.EXPLORING)
                    return
                self.target_door = None
            self.get_logger().error('Max retries 초과. Emergency stop.')
            self._transition(State.EMERGENCY_STOP)
            return

        self.get_logger().info('재탐색/자세 보정 후 같은 문 재시도 허용')
        self.target_door = None
        self._transition(State.EXPLORING)

    def _on_opening_door(self):
        if self.target_door is None or self._door_opening_in_progress:
            return
        if not self.open_door_client.service_is_ready():
            self.get_logger().warn('OpenDoor 서비스 대기 중...')
            return

        self._door_opening_in_progress = True
        req               = OpenDoor.Request()
        req.door_id       = self.target_door.door_id
        req.handle_position = self.target_door.handle_position
        future = self.open_door_client.call_async(req)
        future.add_done_callback(self._door_open_result)

    def _door_open_result(self, future):
        self._door_opening_in_progress = False
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'Door open service error: {e}')
            self._transition(State.EMERGENCY_STOP)
            return

        if result.success:
            self._mark_door_opened(self.target_door)
            self._nav_retry_count = 0   # 성공 시 재시도 카운터 리셋
            self.get_logger().info(
                f'문 개방 성공: {self.target_door.door_id}  '
                f'누적 개방: {len(self._opened_door_ids)}개')
            self._transition(State.DOOR_OPENED)
        else:
            self.get_logger().error(f'문 개방 실패: {result.message}')
            self._transition(State.EMERGENCY_STOP)

    def _on_door_opened(self):
        # 즉시 다음 파란 문 탐색으로 전환
        self.target_door = None
        self.get_logger().info(
            f'다음 파란 문 탐색 시작 (개방 완료: {len(self._opened_door_ids)}개)')
        self._transition(State.EXPLORING)

    def _send_explore_waypoint(self) -> bool:
        if not self._explore_nav_enabled:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False
        x, y, yaw = pose
        if x >= self._exit_x - 1.0:
            return False

        preferred_lane_sign, preferred_reason = self._preferred_lane_sign()
        radar_lane_sign, radar_reason = self._radar_clear_lane_sign()
        front_avoid_sign, front_avoid_reason = self._front_obstacle_avoid_lane_sign(
            x, y, yaw)
        goal_y, lane_sign, lane_reason = self._select_explore_goal_y(
            x, y, yaw, preferred_lane_sign, preferred_reason,
            radar_lane_sign, radar_reason,
            front_avoid_sign, front_avoid_reason)
        goal = DoorInfo()
        now = self.get_clock().now().to_msg()
        goal.header.stamp = now
        goal.header.frame_id = 'map'
        goal.door_id = f'explore_waypoint_{x + self._explore_nav_step_m:.1f}'
        goal.door_color = 'explore'
        goal.confidence = 1.0
        goal.door_pose.header.stamp = now
        goal.door_pose.header.frame_id = 'map'
        goal.door_pose.pose.position.x = min(
            x + self._explore_nav_step_m, self._exit_x - 1.0)
        goal.door_pose.pose.position.y = goal_y
        goal.door_pose.pose.orientation.w = 1.0

        if lane_sign != 0.0:
            self._last_lane_sign = lane_sign

        self._explore_nav_goal_id = goal.door_id
        self.get_logger().info(
            f'탐색 waypoint 계획 요청: {goal.door_id} '
            f'({goal.door_pose.pose.position.x:.2f}, '
            f'{goal.door_pose.pose.position.y:.2f}) | {lane_reason}')
        self.target_door_pub.publish(goal)
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.EXPLORING_NAVIGATING)
        return True

    def _defer_far_target_door(self, door: DoorInfo | None) -> bool:
        if door is None or self._target_door_direct_nav_max_dist_m <= 0.0:
            return False
        if door.door_pose.header.frame_id != 'map':
            return False

        pose = self._current_map_pose()
        if pose is None:
            return False

        rx, ry, _ = pose
        dx = door.door_pose.pose.position.x - rx
        dy = door.door_pose.pose.position.y - ry
        dist = math.hypot(dx, dy)
        if dist <= self._target_door_direct_nav_max_dist_m:
            return False

        self.get_logger().info(
            f'목표 문 {door.door_id}까지 {dist:.1f}m. '
            '먼 문 직접 접근 대신 LiDAR+camera 재스캔 waypoint로 먼저 접근합니다.')
        self.target_door = None
        if self._send_explore_waypoint():
            return True
        self._rotate_to_scan()
        return True

    def _run_pre_nav_scan(self, target: DoorInfo) -> bool:
        if self._pre_nav_scan_sec <= 0.0:
            return False
        now = self.get_clock().now()
        target_key = self._pre_nav_scan_key(target)
        if self._pending_nav_scan_target_id != target_key:
            self._pending_nav_scan_target_id = target_key
            self._pre_nav_scan_start_time = now
            self.get_logger().info(
            f'LiDAR+camera segment map 사전 스캔 시작: {target.door_id}')

        elapsed = (now - self._pre_nav_scan_start_time).nanoseconds / 1e9
        if elapsed < self._pre_nav_scan_sec:
            twist = Twist()
            twist.angular.z = self._pre_nav_scan_angular_vel
            self.cmd_vel_pub.publish(twist)
            return True

        self._pre_nav_scan_start_time = None
        self._pending_nav_scan_target_id = None
        self.get_logger().info(
            f'사전 스캔 완료. Nav2가 segment map 기반 경로를 계획합니다: {target.door_id}')
        return False

    def _pre_nav_scan_key(self, target: DoorInfo) -> str:
        return f'{target.door_color}:{target.door_id}'

    def _run_final_scan_before_exit(self) -> bool:
        if self._final_scan_before_exit_sec <= 0.0:
            return False
        now = self.get_clock().now()
        if self._final_scan_start_time is None:
            self._final_scan_start_time = now
            self.get_logger().info('비상구 이동 전 마지막 측면 문 스캔 시작')

        elapsed = (now - self._final_scan_start_time).nanoseconds / 1e9
        if elapsed < self._final_scan_before_exit_sec:
            twist = Twist()
            twist.angular.z = self._final_scan_angular_vel
            self.cmd_vel_pub.publish(twist)
            return True

        self._final_scan_start_time = None
        self._explore_start_time = now
        return False

    def _ready_to_start_navigation(self) -> bool:
        pose = self._current_map_pose()
        if pose is None:
            return True
        _, y, yaw = pose
        heading_error = abs(self._normalize_angle(self._explore_forward_yaw - yaw))
        if abs(y - self._explore_center_y) > self._nav_start_max_abs_y_m:
            self.get_logger().info(
                f'Nav2 목표 대기: 중앙 복귀 중 '
                f'(y={y:.2f}, limit={self._nav_start_max_abs_y_m:.2f})',
                once=True)
            return False
        if heading_error > self._nav_start_max_heading_error:
            self.get_logger().info(
                f'Nav2 목표 대기: 진행 방향 정렬 중 '
                f'(heading_error={math.degrees(heading_error):.0f}deg)',
                once=True)
            return False
        return True

    def _filter_forward_doors(self, doors: list[DoorInfo]) -> list[DoorInfo]:
        candidates = [
            d for d in doors
            if d.door_pose.header.frame_id == 'map'
        ]

        if self._min_target_door_abs_y_m > 0.0:
            side_doors = [
                d for d in candidates
                if abs(d.door_pose.pose.position.y - self._explore_center_y)
                >= self._min_target_door_abs_y_m
            ]
            skipped = len(candidates) - len(side_doors)
            if skipped > 0 and not side_doors:
                self.get_logger().info(
                    f'중앙쪽 파란 문 후보 {skipped}개 무시 '
                    f'(|y| < {self._min_target_door_abs_y_m:.2f})', once=True)
            candidates = side_doors

        if not self._prefer_forward_doors:
            return candidates

        min_x = self._mission_progress_x + self._next_door_min_forward_m
        forward = [
            d for d in candidates
            if self._door_identity_xy(d)[0] > min_x
        ]
        skipped = len(candidates) - len(forward)
        if skipped > 0 and not forward:
            self.get_logger().info(
                f'뒤쪽/중복 파란 문 후보 {skipped}개 무시 '
                f'(다음 목표 x > {min_x:.2f} 대기)', once=True)
        return forward

    def _mark_door_opened(self, door: DoorInfo):
        self._opened_door_ids.add(door.door_id)
        if door.door_pose.header.frame_id == 'map':
            door_x, door_y = self._door_identity_xy(door)
            self._mission_progress_x = max(
                self._mission_progress_x,
                door_x)

        ox, oy = self._door_identity_xy(door)
        for other in self.detected_doors:
            if other.door_id in self._opened_door_ids:
                continue
            if other.door_color != door.door_color:
                continue
            if not self._door_has_map_identity(other):
                continue
            other_x, other_y = self._door_identity_xy(other)
            dx = other_x - ox
            dy = other_y - oy
            if math.hypot(dx, dy) <= self._opened_door_merge_dist_m:
                self._opened_door_ids.add(other.door_id)

    def _on_exiting(self):
        if self._nav_done:
            self._nav_done = False
            self._transition(State.MISSION_COMPLETE)
        elif self._nav_failed:
            self._nav_failed = False
            self.get_logger().warn('비상구 이동 실패. 재시도...')
            self._send_exit_goal()

    def _on_mission_complete(self):
        self.get_logger().info(
            f'미션 완료! 개방한 문: {len(self._opened_door_ids)}개. '
            '로봇이 비상구에 도착했습니다.', once=True)

    def _on_emergency_stop(self):
        self.get_logger().error('비상 정지.', once=True)

    # ── 비상구 이동 ───────────────────────────────────────
    def _past_exit_line(self) -> bool:
        if not self._complete_when_past_exit_x:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False
        x, y, _ = pose
        return x >= (self._exit_x - self._exit_complete_margin_m)

    def _send_exit_goal(self):
        """탐지된 초록 비상구 or 하드코딩 좌표를 navigation_node에 전달"""
        if self._valid_detected_exit():
            # 카메라로 탐지한 비상구 위치 사용
            self.get_logger().info(
                f'탐지된 비상구로 이동: {self._exit_door.door_id} '
                f'({self._exit_door.door_pose.pose.position.x:.1f}, '
                f'{self._exit_door.door_pose.pose.position.y:.1f})')
            self.target_door_pub.publish(self._exit_door)
        else:
            # 비상구 미탐지 → 파라미터 하드코딩 좌표 폴백
            self.get_logger().warn(
                f'비상구 미탐지 → 하드코딩 좌표 '
                f'({self._exit_x:.1f}, {self._exit_y:.1f}) 사용')
            msg = DoorInfo()
            now = self.get_clock().now().to_msg()
            msg.header.stamp              = now
            msg.header.frame_id           = 'map'
            msg.door_id                   = 'emergency_exit'
            msg.door_color                = 'green'
            msg.door_pose.header.stamp    = now
            msg.door_pose.header.frame_id = 'map'
            msg.door_pose.pose.position.x = self._exit_x
            msg.door_pose.pose.position.y = self._exit_y
            msg.door_pose.pose.orientation.z = math.sin(self._exit_yaw / 2.0)
            msg.door_pose.pose.orientation.w = math.cos(self._exit_yaw / 2.0)
            self.target_door_pub.publish(msg)
        self._nav_done   = False
        self._nav_failed = False

    def _valid_detected_exit(self) -> bool:
        if not self._use_detected_exit or self._exit_door is None:
            return False
        pose = self._exit_door.door_pose
        if pose.header.frame_id != 'map':
            return False

        x = pose.pose.position.x
        y = pose.pose.position.y
        min_x = self._exit_x - self._detected_exit_min_x_margin_m
        if x < min_x or abs(y - self._exit_y) > self._detected_exit_max_y_error_m:
            self.get_logger().warn(
                f'비상구 후보 {self._exit_door.door_id} 좌표가 예상 출구 영역 밖입니다 '
                f'({x:.1f}, {y:.1f}); 하드코딩 출구를 사용합니다.')
            self._exit_door = None
            return False
        return True

    # ── 유틸 ─────────────────────────────────────────────
    def _rotate_to_scan(self):
        twist = Twist()
        scan = self._latest_scan
        if scan is None:
            twist.linear.x = float(self._explore_linear_vel)
            twist.angular.z = float(self._explore_angular_vel)
            self.cmd_vel_pub.publish(twist)
            return

        front_min = self._sector_min(
            scan, -self._explore_front_angle, self._explore_front_angle)
        left_min = self._sector_min(
            scan, self._explore_front_angle, self._explore_side_angle)
        right_min = self._sector_min(
            scan, -self._explore_side_angle, -self._explore_front_angle)
        lane_sign, lane_reason = self._preferred_lane_sign()
        if lane_sign != self._last_lane_sign:
            self.get_logger().info(f'회피 차선 선호: {lane_reason}')
            self._last_lane_sign = lane_sign

        turn_sign = lane_sign if lane_sign != 0.0 else (
            1.0 if left_min >= right_min else -1.0)
        if left_min < self._explore_side_clearance_m <= right_min:
            turn_sign = -1.0
        elif right_min < self._explore_side_clearance_m <= left_min:
            turn_sign = 1.0
        elif (left_min < self._explore_side_clearance_m
              and right_min < self._explore_side_clearance_m):
            turn_sign = 1.0 if left_min >= right_min else -1.0

        if front_min < self._explore_obstacle_stop_m:
            twist.linear.x = 0.0
            twist.angular.z = turn_sign * self._explore_avoid_angular_vel
        elif left_min < self._explore_side_clearance_m:
            twist.linear.x = min(float(self._explore_linear_vel), 0.10)
            twist.angular.z = -self._explore_avoid_angular_vel * 0.8
        elif right_min < self._explore_side_clearance_m:
            twist.linear.x = min(float(self._explore_linear_vel), 0.10)
            twist.angular.z = self._explore_avoid_angular_vel * 0.8
        elif front_min < self._explore_obstacle_slow_m:
            twist.linear.x = min(float(self._explore_linear_vel), 0.16)
            twist.angular.z = turn_sign * self._explore_avoid_angular_vel * 0.75
        else:
            twist.linear.x = float(self._explore_linear_vel)
            twist.angular.z = float(self._explore_angular_vel)
            if lane_sign != 0.0:
                preferred_clearance = left_min if lane_sign > 0.0 else right_min
                if preferred_clearance > self._explore_side_clearance_m + 0.35:
                    twist.angular.z = lane_sign * min(
                        self._explore_avoid_angular_vel * 0.22, 0.22)
        self._apply_forward_heading_guard(twist)
        self.cmd_vel_pub.publish(twist)

    def _apply_forward_heading_guard(self, twist: Twist):
        pose = self._current_map_pose()
        if pose is None:
            return

        _, y, yaw = pose
        desired_yaw = self._explore_forward_yaw
        lateral_error = y - self._explore_center_y
        center_active = abs(lateral_error) > self._explore_y_soft_limit_m
        if center_active:
            desired_yaw = math.atan2(
                -lateral_error, max(0.1, self._explore_center_lookahead_m))

        error = self._normalize_angle(desired_yaw - yaw)
        abs_error = abs(error)
        if abs_error < self._explore_heading_soft_limit and not center_active:
            return

        correction = max(
            -self._explore_avoid_angular_vel,
            min(self._explore_avoid_angular_vel,
                self._explore_heading_kp * error))

        wall_zone = abs(lateral_error) >= self._explore_y_hard_limit_m
        if abs_error >= self._explore_heading_hard_limit:
            twist.linear.x = 0.0
            twist.angular.z = correction
            return

        if wall_zone:
            twist.linear.x = min(twist.linear.x, 0.06)
            twist.angular.z = correction
            return

        twist.linear.x = min(twist.linear.x, 0.10 if center_active else 0.08)
        if twist.angular.z * correction < 0.0 or abs(twist.angular.z) < abs(correction):
            twist.angular.z = correction

    def _current_map_pose(self) -> tuple[float, float, float] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                'map', 'base_link', Time(), timeout=Duration(seconds=0.05))
        except TransformException:
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return float(t.x), float(t.y), yaw

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _preferred_lane_sign(self) -> tuple[float, str]:
        """Return +1 for robot/map-left lane and -1 for robot/map-right lane."""
        if not self._explore_color_lane_bias:
            return 0.0, '색상 차선 선호 비활성'

        blue_doors = [
            d for d in self.detected_doors
            if d.door_color == 'blue'
            and d.door_id not in self._failed_door_ids
            and d.door_id not in self._opened_door_ids
        ]
        blue_doors = self._filter_forward_doors(blue_doors)
        if blue_doors:
            blue = self._nearest_forward_blue_door(blue_doors)
            blue_dist = self._door_distance_from_robot(blue)
            if (blue_dist is not None
                    and self._explore_nav_blue_lane_bias_max_dist_m > 0.0
                    and blue_dist > self._explore_nav_blue_lane_bias_max_dist_m):
                return (
                    0.0,
                    f'파란 문 {blue.door_id} {blue_dist:.1f}m 전방, '
                    '근접 전까지 LiDAR 차선 유지',
                )
            side = self._door_side_sign(blue) if blue is not None else 0.0
            if side != 0.0:
                return side, f'가장 가까운 파란 문 {blue.door_id} 쪽 차선 고정'

        red_doors = [d for d in self.detected_doors if d.door_color == 'red']
        if red_doors:
            red = max(red_doors, key=lambda d: d.confidence)
            side = self._door_side_sign(red)
            if side != 0.0:
                return -side, f'빨간 문 {red.door_id} 반대쪽 차선'

        if self.fire_info and self.fire_info.detected:
            fire_pos = self.fire_info.fire_position
            if abs(fire_pos.point.y) > 0.08:
                side = 1.0 if fire_pos.point.y > 0.0 else -1.0
                return -side, '화재/빨간 문 반대쪽 차선'

        return 0.0, 'LiDAR 여유 공간 기준'

    def _door_side_sign(self, door: DoorInfo) -> float:
        y = door.door_pose.pose.position.y
        if abs(y) <= 0.08:
            return 0.0
        return 1.0 if y > 0.0 else -1.0

    def _door_has_map_identity(self, door: DoorInfo) -> bool:
        return (
            door.handle_position.header.frame_id == 'map'
            or door.door_pose.header.frame_id == 'map'
        )

    def _door_identity_xy(self, door: DoorInfo) -> tuple[float, float]:
        """Physical door identity point, preferring handle over approach pose."""
        if door.handle_position.header.frame_id == 'map':
            return (
                float(door.handle_position.point.x),
                float(door.handle_position.point.y),
            )
        return (
            float(door.door_pose.pose.position.x),
            float(door.door_pose.pose.position.y),
        )

    def _nearest_forward_blue_door(self, doors: list[DoorInfo]) -> DoorInfo | None:
        if not doors:
            return None
        pose = self._current_map_pose()
        if pose is None:
            return min(doors, key=lambda d: self._door_identity_xy(d)[0])
        rx, ry, _ = pose
        return min(
            doors,
            key=lambda d: math.hypot(
                self._door_identity_xy(d)[0] - rx,
                self._door_identity_xy(d)[1] - ry,
            ),
        )

    def _door_distance_from_robot(self, door: DoorInfo | None) -> float | None:
        if door is None or door.door_pose.header.frame_id != 'map':
            return None
        pose = self._current_map_pose()
        if pose is None:
            return None
        rx, ry, _ = pose
        dx = door.door_pose.pose.position.x - rx
        dy = door.door_pose.pose.position.y - ry
        return math.hypot(dx, dy)

    def _flip_explore_lane_after_failure(self):
        self._last_lane_sign = -self._last_lane_sign if self._last_lane_sign != 0.0 else -1.0
        side = '좌측' if self._last_lane_sign > 0.0 else '우측'
        self.get_logger().info(f'다음 탐색 waypoint는 {side} 차선을 우선합니다.')

    def _alternate_explore_lane_after_success(self):
        has_color_hint = any(
            d.door_color in ('blue', 'red')
            and d.door_id not in self._failed_door_ids
            and d.door_id not in self._opened_door_ids
            for d in self.detected_doors)
        if has_color_hint:
            return
        self._last_lane_sign = -self._last_lane_sign if self._last_lane_sign != 0.0 else -1.0
        side = '좌측' if self._last_lane_sign > 0.0 else '우측'
        self.get_logger().info(f'색상 단서 없음. 다음 탐색 waypoint는 {side} 차선으로 스캔합니다.')

    def _select_explore_goal_y(self, robot_x: float, robot_y: float, robot_yaw: float,
                               preferred_sign: float, preferred_reason: str,
                               radar_sign: float, radar_reason: str,
                               front_avoid_sign: float,
                               front_avoid_reason: str) -> tuple[float, float, str]:
        lane_abs_y = abs(self._explore_nav_lane_y_m)
        candidates: list[tuple[float, float]] = []

        def add_candidate(sign: float):
            if sign > 0.0:
                value = self._explore_center_y + lane_abs_y
                norm_sign = 1.0
            elif sign < 0.0:
                value = self._explore_center_y - lane_abs_y
                norm_sign = -1.0
            else:
                value = self._explore_center_y
                norm_sign = 0.0
            if abs(value - self._explore_center_y) > self._explore_y_hard_limit_m:
                value = self._explore_center_y + norm_sign * self._explore_y_hard_limit_m
            if not any(abs(value - existing[0]) < 0.05 for existing in candidates):
                candidates.append((value, norm_sign))

        add_candidate(front_avoid_sign)
        add_candidate(preferred_sign)
        add_candidate(radar_sign)
        add_candidate(self._last_lane_sign)
        add_candidate(-1.0)
        add_candidate(1.0)
        add_candidate(0.0)

        blue_lane_active = (
            preferred_sign != 0.0
            and preferred_reason.startswith('가장 가까운 파란 문')
        )
        best: tuple[float, float, float, float] | None = None
        scored_text: list[str] = []
        for goal_y, sign in candidates:
            clearance = self._lane_scan_clearance(robot_x, robot_y, robot_yaw, goal_y)
            score = clearance
            if sign != 0.0 and sign == preferred_sign:
                score += self._explore_nav_color_bias_score
                if blue_lane_active:
                    score += self._explore_nav_blue_door_bias_score
            elif blue_lane_active and sign != 0.0:
                score -= 0.45
            if sign != 0.0 and sign == radar_sign:
                score += self._explore_nav_color_bias_score * (0.25 if blue_lane_active else 0.55)
            if sign != 0.0 and sign == front_avoid_sign:
                if blue_lane_active and sign != preferred_sign:
                    score += self._explore_nav_front_obstacle_bias_score * 0.25
                else:
                    score += self._explore_nav_front_obstacle_bias_score
            if (blue_lane_active and front_avoid_sign != 0.0
                    and sign == preferred_sign
                    and clearance >= self._explore_nav_blue_lane_min_clearance_m):
                score += self._explore_nav_front_obstacle_bias_score * 0.65
            if sign != 0.0 and sign == self._last_lane_sign:
                score += self._explore_nav_keep_lane_bias_score * (2.0 if blue_lane_active else 1.0)
            if sign == 0.0:
                score -= 0.28
            score -= abs(goal_y - robot_y) * 0.06
            if clearance < self._explore_obstacle_stop_m:
                score -= 2.0
            elif clearance < self._explore_obstacle_slow_m:
                score -= 0.55
            if (blue_lane_active and sign == preferred_sign
                    and clearance < self._explore_nav_blue_lane_min_clearance_m):
                score -= 1.75

            label = 'C' if sign == 0.0 else ('L' if sign > 0.0 else 'R')
            scored_text.append(f'{label}:y={goal_y:.2f}/clear={clearance:.2f}/score={score:.2f}')
            if best is None or score > best[0]:
                best = (score, goal_y, sign, clearance)

        if best is None:
            return self._explore_center_y, 0.0, 'LiDAR scan 대기 중'

        _, goal_y, sign, clearance = best
        side = '중앙' if sign == 0.0 else ('좌측' if sign > 0.0 else '우측')
        if front_avoid_sign != 0.0:
            hint = front_avoid_reason
        else:
            hint = preferred_reason if preferred_sign != 0.0 else radar_reason
        reason = (
            f'{side} 차선 선택 | {hint} | '
            f'LiDAR lane scores: {", ".join(scored_text)}')
        if clearance < self._explore_obstacle_slow_m:
            reason += ' | 가까운 장애물 감지, 우회 차선 우선'
        return goal_y, sign, reason

    def _lane_scan_clearance(self, robot_x: float, robot_y: float, robot_yaw: float,
                             goal_y: float) -> float:
        scan = self._latest_scan
        lookahead = max(self._explore_nav_step_m, self._explore_nav_scan_lookahead_m)
        if scan is None:
            return lookahead

        corridor_axis = self._explore_forward_yaw
        axis_x = math.cos(corridor_axis)
        axis_y = math.sin(corridor_axis)
        side_x = -axis_y
        side_y = axis_x
        goal_lateral = (goal_y - self._explore_center_y)
        half_width = max(0.20, self._explore_nav_lane_width_m * 0.5)
        best_forward = lookahead

        scan_angle = scan.angle_min
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        for rng in scan.ranges:
            if (math.isfinite(rng)
                    and max(scan.range_min, self._explore_scan_min_range_m) <= rng <= scan.range_max):
                base_x = float(rng) * math.cos(scan_angle)
                base_y = float(rng) * math.sin(scan_angle)
                map_x = robot_x + cos_yaw * base_x - sin_yaw * base_y
                map_y = robot_y + sin_yaw * base_x + cos_yaw * base_y
                rel_x = map_x - robot_x
                rel_y = map_y - robot_y
                forward = rel_x * axis_x + rel_y * axis_y
                lateral = rel_x * side_x + rel_y * side_y
                if 0.15 <= forward <= lookahead:
                    if abs(lateral - goal_lateral) <= half_width:
                        best_forward = min(best_forward, forward)
            scan_angle += scan.angle_increment

        return best_forward

    def _front_obstacle_avoid_lane_sign(self, robot_x: float, robot_y: float,
                                        robot_yaw: float) -> tuple[float, str]:
        scan = self._latest_scan
        if scan is None:
            return 0.0, 'LiDAR scan 대기 중'

        lookahead = max(self._explore_obstacle_slow_m,
                        min(self._explore_nav_scan_lookahead_m, 3.5))
        half_width = max(0.25, self._explore_nav_front_obstacle_width_m * 0.5)
        corridor_axis = self._explore_forward_yaw
        axis_x = math.cos(corridor_axis)
        axis_y = math.sin(corridor_axis)
        side_x = -axis_y
        side_y = axis_x
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        nearest_forward = float('inf')
        nearest_lateral = 0.0
        scan_angle = scan.angle_min
        for rng in scan.ranges:
            if (math.isfinite(rng)
                    and max(scan.range_min, self._explore_scan_min_range_m) <= rng <= scan.range_max):
                base_x = float(rng) * math.cos(scan_angle)
                base_y = float(rng) * math.sin(scan_angle)
                map_x = robot_x + cos_yaw * base_x - sin_yaw * base_y
                map_y = robot_y + sin_yaw * base_x + cos_yaw * base_y
                rel_x = map_x - robot_x
                rel_y = map_y - robot_y
                forward = rel_x * axis_x + rel_y * axis_y
                lateral = rel_x * side_x + rel_y * side_y
                if 0.20 <= forward <= lookahead and abs(lateral) <= half_width:
                    if forward < nearest_forward:
                        nearest_forward = forward
                        nearest_lateral = lateral
            scan_angle += scan.angle_increment

        if not math.isfinite(nearest_forward):
            return 0.0, '전방 근접 장애물 없음'
        if nearest_forward > self._explore_obstacle_slow_m:
            return 0.0, (
                f'전방 장애물 여유 {nearest_forward:.2f}m '
                f'(threshold={self._explore_obstacle_slow_m:.2f}m)')

        if abs(nearest_lateral) <= 0.08:
            radar_sign, radar_reason = self._radar_clear_lane_sign()
            return radar_sign, f'전방 중앙 장애물 {nearest_forward:.2f}m, {radar_reason}'

        obstacle_side = 1.0 if nearest_lateral > 0.0 else -1.0
        avoid_sign = -obstacle_side
        avoid_side = '우측' if avoid_sign < 0.0 else '좌측'
        obstacle_side_name = '좌측' if obstacle_side > 0.0 else '우측'
        return avoid_sign, (
            f'전방 {obstacle_side_name} 장애물 {nearest_forward:.2f}m '
            f'(lateral={nearest_lateral:.2f}m) → {avoid_side} 차선 우선')

    def _radar_clear_lane_sign(self) -> tuple[float, str]:
        scan = self._latest_scan
        if scan is None:
            return 0.0, 'LiDAR scan 대기 중'

        left_min = self._sector_min(
            scan, self._explore_front_angle, self._explore_side_angle)
        right_min = self._sector_min(
            scan, -self._explore_side_angle, -self._explore_front_angle)

        if math.isinf(left_min) and math.isinf(right_min):
            return 0.0, 'LiDAR 여유 공간 기준'
        if left_min >= right_min + 0.15:
            return 1.0, f'LiDAR 좌측 여유 공간 우선 ({left_min:.2f}m)'
        if right_min > left_min + 0.15:
            return -1.0, f'LiDAR 우측 여유 공간 우선 ({right_min:.2f}m)'
        if self._last_lane_sign != 0.0:
            return self._last_lane_sign, '이전 회피 차선 유지'
        return 1.0, 'LiDAR 좌우 여유 유사, 좌측 차선 우선'

    def _sector_min(self, scan: LaserScan, angle_min: float, angle_max: float) -> float:
        best = float('inf')
        angle = scan.angle_min
        for r in scan.ranges:
            if angle_min <= angle <= angle_max:
                if (math.isfinite(r)
                        and max(scan.range_min, self._explore_scan_min_range_m) <= r <= scan.range_max):
                    best = min(best, float(r))
            angle += scan.angle_increment
        return best

    def _stop_rotation(self):
        self.cmd_vel_pub.publish(Twist())

    def _transition(self, new_state: State):
        self.get_logger().info(f'State: {self.state.name} → {new_state.name}')
        self.state = new_state

        if new_state == State.NAVIGATING:
            self._explore_nav_goal_id = ''
            self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
        elif new_state == State.EXPLORING_NAVIGATING:
            self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
        elif new_state == State.EXITING:
            self._send_exit_goal()
            self._nav_start_time = self.get_clock().now()
        elif new_state == State.EXPLORING:
            # EXPLORING 재진입 시 탐색 타이머 시작
            self._explore_start_time = self.get_clock().now()
            self._nav_start_time = None
            self._final_scan_start_time = None
            self._pre_nav_scan_start_time = None
            self._pending_nav_scan_target_id = None
            self._explore_nav_goal_id = ''
        else:
            self._nav_start_time = None

    def _publish_state(self):
        msg                   = RobotState()
        msg.header.stamp      = self.get_clock().now().to_msg()
        msg.state             = int(self.state)
        msg.state_description = self.state.name
        msg.target_door_id    = (self.target_door.door_id
                                  if self.target_door else self._explore_nav_goal_id)
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
