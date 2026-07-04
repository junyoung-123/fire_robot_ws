import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from enum import IntEnum
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Twist
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
        self.declare_parameter('require_mission_axis', False)
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
        self.declare_parameter('target_door_min_robot_forward_m', 0.35)
        self.declare_parameter('opened_door_merge_dist_m', 1.0)
        self.declare_parameter('min_opened_doors_before_exit', 0)
        self.declare_parameter('opened_physical_door_merge_dist_m', 1.2)
        self.declare_parameter('observed_physical_door_merge_dist_m', 1.2)
        self.declare_parameter('observed_blue_min_observations', 2)
        self.declare_parameter('max_door_approach_failures_before_abandon', 4)
        self.declare_parameter('detected_door_max_age_sec', 8.0)
        self.declare_parameter('observed_blue_same_id_max_jump_m', 1.8)
        self.declare_parameter('observed_blue_stale_abandon_sec', 45.0)
        self.declare_parameter('observed_blue_stale_behind_m', 2.0)
        self.declare_parameter('blue_red_conflict_dist_m', 1.4)
        self.declare_parameter('observed_blue_min_abs_wall_y_m', 0.0)
        self.declare_parameter('observed_blue_max_abs_wall_y_m', 0.0)
        self.declare_parameter('red_blue_conflict_memory_sec', 240.0)
        self.declare_parameter('blue_red_conflict_after_exit_dist_m', 0.0)
        self.declare_parameter('opened_station_blue_suppression_progress_m', 0.0)
        self.declare_parameter('max_door_open_failures_before_abandon', 2)
        self.declare_parameter('complete_when_past_exit_x', False)
        self.declare_parameter('exit_complete_margin_m', 0.50)
        self.declare_parameter('final_scan_before_exit_sec', 0.0)
        self.declare_parameter('final_scan_angular_vel', 0.45)
        self.declare_parameter('door_open_ready_max_dist_m', 0.35)
        self.declare_parameter('door_open_ready_yaw_tolerance_deg', 22.0)
        self.declare_parameter('door_open_settle_sec', 1.0)
        self.declare_parameter('exit_retry_delay_sec', 3.0)
        self.declare_parameter('door_open_align_angular_vel', 0.42)
        self.declare_parameter('door_open_align_kp', 1.2)
        self.declare_parameter('door_open_align_timeout_sec', 10.0)
        self.declare_parameter('pre_nav_scan_sec', 0.0)
        self.declare_parameter('pre_nav_scan_angular_vel', 0.0)
        self.declare_parameter('explore_nav_enabled', True)
        self.declare_parameter('explore_nav_step_m', 1.20)
        self.declare_parameter('explore_nav_min_step_m', 0.30)
        self.declare_parameter('explore_nav_reduce_step_clearance_m', 1.35)
        self.declare_parameter('explore_nav_goal_clearance_margin_m', 0.55)
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
        self.declare_parameter('explore_nav_recenter_abs_y_m', 0.95)
        self.declare_parameter('explore_nav_max_lateral_step_m', 0.45)
        self.declare_parameter('max_explore_nav_failures_before_exit', 8)
        self.declare_parameter('explore_nav_failed_goal_memory_sec', 24.0)
        self.declare_parameter('explore_nav_failed_goal_progress_window_m', 2.4)
        self.declare_parameter('explore_nav_timeout_sec', 45.0)
        self.declare_parameter('min_target_door_abs_y_m', 0.0)
        self.declare_parameter('axis_door_approach_enabled', True)
        self.declare_parameter('axis_door_side_standoff_m', 0.85)
        self.declare_parameter('axis_door_min_abs_lateral_m', 0.45)
        self.declare_parameter('target_door_direct_nav_max_dist_m', 0.0)
        self.declare_parameter('nav_start_max_abs_y_m', 0.95)
        self.declare_parameter('nav_start_max_heading_error_deg', 85.0)
        self.declare_parameter('start_without_fire',  False)
        self.declare_parameter('exit_x',   10.0)
        self.declare_parameter('exit_y',    0.0)
        self.declare_parameter('exit_yaw',  0.0)
        self.declare_parameter('allow_fallback_exit_goal', False)
        self.declare_parameter('use_detected_exit', True)
        self.declare_parameter('detected_exit_min_x_margin_m', 3.0)
        self.declare_parameter('detected_exit_max_y_error_m', 1.5)
        self.declare_parameter('detected_exit_use_expected_region', False)
        self.declare_parameter('detected_exit_max_abs_lateral_m', 0.0)
        self.declare_parameter('detected_exit_min_robot_forward_m', 0.0)
        self.declare_parameter('detected_exit_max_center_y_m', 0.0)
        self.declare_parameter('detected_exit_merge_dist_m', 2.5)
        self.declare_parameter('blue_exit_suppression_margin_m', 0.0)
        self.declare_parameter('exit_pass_through_m', 0.8)
        self.declare_parameter('exit_nav_standoff_m', 0.8)
        self.declare_parameter('exit_cross_linear_vel', 0.22)
        self.declare_parameter('exit_cross_heading_kp', 1.2)
        self.declare_parameter('exit_cross_angular_vel_limit', 0.45)
        self.declare_parameter('exit_cross_timeout_sec', 6.0)

        self._nav_timeout_sec     = self.get_parameter('nav_timeout_sec').value
        self._max_nav_retries     = self.get_parameter('max_nav_retries').value
        self._explore_timeout_sec = self.get_parameter('explore_timeout_sec').value
        self._explore_linear_vel  = self.get_parameter('explore_linear_vel').value
        self._explore_angular_vel = self.get_parameter('explore_angular_vel').value
        self._explore_obstacle_stop_m = float(self.get_parameter('explore_obstacle_stop_m').value)
        self._explore_obstacle_slow_m = float(self.get_parameter('explore_obstacle_slow_m').value)
        self._explore_avoid_angular_vel = float(self.get_parameter('explore_avoid_angular_vel').value)
        self._require_mission_axis = bool(self.get_parameter('require_mission_axis').value)
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
        self._target_door_min_robot_forward_m = float(
            self.get_parameter('target_door_min_robot_forward_m').value)
        self._opened_door_merge_dist_m = float(self.get_parameter('opened_door_merge_dist_m').value)
        self._min_opened_doors_before_exit = int(
            self.get_parameter('min_opened_doors_before_exit').value)
        self._opened_physical_door_merge_dist_m = float(
            self.get_parameter('opened_physical_door_merge_dist_m').value)
        self._observed_physical_door_merge_dist_m = float(
            self.get_parameter('observed_physical_door_merge_dist_m').value)
        self._observed_blue_min_observations = max(1, int(
            self.get_parameter('observed_blue_min_observations').value))
        self._max_door_approach_failures_before_abandon = max(1, int(
            self.get_parameter('max_door_approach_failures_before_abandon').value))
        self._detected_door_max_age_sec = float(
            self.get_parameter('detected_door_max_age_sec').value)
        self._observed_blue_same_id_max_jump_m = float(
            self.get_parameter('observed_blue_same_id_max_jump_m').value)
        self._observed_blue_stale_abandon_sec = float(
            self.get_parameter('observed_blue_stale_abandon_sec').value)
        self._observed_blue_stale_behind_m = float(
            self.get_parameter('observed_blue_stale_behind_m').value)
        self._blue_red_conflict_dist_m = float(
            self.get_parameter('blue_red_conflict_dist_m').value)
        self._observed_blue_min_abs_wall_y_m = float(
            self.get_parameter('observed_blue_min_abs_wall_y_m').value)
        self._observed_blue_max_abs_wall_y_m = float(
            self.get_parameter('observed_blue_max_abs_wall_y_m').value)
        self._red_blue_conflict_memory_sec = float(
            self.get_parameter('red_blue_conflict_memory_sec').value)
        self._blue_red_conflict_after_exit_dist_m = float(
            self.get_parameter('blue_red_conflict_after_exit_dist_m').value)
        self._opened_station_blue_suppression_progress_m = float(
            self.get_parameter('opened_station_blue_suppression_progress_m').value)
        self._max_door_open_failures_before_abandon = max(1, int(
            self.get_parameter('max_door_open_failures_before_abandon').value))
        self._complete_when_past_exit_x = bool(
            self.get_parameter('complete_when_past_exit_x').value)
        self._exit_complete_margin_m = float(
            self.get_parameter('exit_complete_margin_m').value)
        self._final_scan_before_exit_sec = float(
            self.get_parameter('final_scan_before_exit_sec').value)
        self._final_scan_angular_vel = float(
            self.get_parameter('final_scan_angular_vel').value)
        self._door_open_ready_max_dist_m = float(
            self.get_parameter('door_open_ready_max_dist_m').value)
        self._door_open_ready_yaw_tolerance = math.radians(float(
            self.get_parameter('door_open_ready_yaw_tolerance_deg').value))
        self._door_open_settle_sec = float(self.get_parameter('door_open_settle_sec').value)
        self._exit_retry_delay_sec = float(self.get_parameter('exit_retry_delay_sec').value)
        self._door_open_align_angular_vel = float(
            self.get_parameter('door_open_align_angular_vel').value)
        self._door_open_align_kp = float(self.get_parameter('door_open_align_kp').value)
        self._door_open_align_timeout_sec = float(
            self.get_parameter('door_open_align_timeout_sec').value)
        self._pre_nav_scan_sec = float(
            self.get_parameter('pre_nav_scan_sec').value)
        self._pre_nav_scan_angular_vel = float(
            self.get_parameter('pre_nav_scan_angular_vel').value)
        self._explore_nav_enabled = bool(
            self.get_parameter('explore_nav_enabled').value)
        self._explore_nav_step_m = float(
            self.get_parameter('explore_nav_step_m').value)
        self._explore_nav_min_step_m = float(
            self.get_parameter('explore_nav_min_step_m').value)
        self._explore_nav_reduce_step_clearance_m = float(
            self.get_parameter('explore_nav_reduce_step_clearance_m').value)
        self._explore_nav_goal_clearance_margin_m = float(
            self.get_parameter('explore_nav_goal_clearance_margin_m').value)
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
        self._explore_nav_recenter_abs_y_m = float(
            self.get_parameter('explore_nav_recenter_abs_y_m').value)
        self._explore_nav_max_lateral_step_m = float(
            self.get_parameter('explore_nav_max_lateral_step_m').value)
        self._max_explore_nav_failures_before_exit = int(
            self.get_parameter('max_explore_nav_failures_before_exit').value)
        self._explore_nav_failed_goal_memory_sec = float(
            self.get_parameter('explore_nav_failed_goal_memory_sec').value)
        self._explore_nav_failed_goal_progress_window_m = float(
            self.get_parameter('explore_nav_failed_goal_progress_window_m').value)
        self._explore_nav_timeout_sec = float(
            self.get_parameter('explore_nav_timeout_sec').value)
        self._min_target_door_abs_y_m = float(
            self.get_parameter('min_target_door_abs_y_m').value)
        self._axis_door_approach_enabled = bool(
            self.get_parameter('axis_door_approach_enabled').value)
        self._axis_door_side_standoff_m = float(
            self.get_parameter('axis_door_side_standoff_m').value)
        self._axis_door_min_abs_lateral_m = float(
            self.get_parameter('axis_door_min_abs_lateral_m').value)
        self._target_door_direct_nav_max_dist_m = float(
            self.get_parameter('target_door_direct_nav_max_dist_m').value)
        self._nav_start_max_abs_y_m = float(self.get_parameter('nav_start_max_abs_y_m').value)
        self._nav_start_max_heading_error = math.radians(
            float(self.get_parameter('nav_start_max_heading_error_deg').value))
        self._start_without_fire  = self.get_parameter('start_without_fire').value
        self._exit_x   = self.get_parameter('exit_x').value
        self._exit_y   = self.get_parameter('exit_y').value
        self._exit_yaw = self.get_parameter('exit_yaw').value
        self._allow_fallback_exit_goal = bool(
            self.get_parameter('allow_fallback_exit_goal').value)
        self._use_detected_exit = bool(self.get_parameter('use_detected_exit').value)
        self._detected_exit_min_x_margin_m = float(
            self.get_parameter('detected_exit_min_x_margin_m').value)
        self._detected_exit_max_y_error_m = float(
            self.get_parameter('detected_exit_max_y_error_m').value)
        self._detected_exit_use_expected_region = bool(
            self.get_parameter('detected_exit_use_expected_region').value)
        self._detected_exit_max_abs_lateral_m = float(
            self.get_parameter('detected_exit_max_abs_lateral_m').value)
        self._detected_exit_min_robot_forward_m = float(
            self.get_parameter('detected_exit_min_robot_forward_m').value)
        self._detected_exit_max_center_y_m = float(
            self.get_parameter('detected_exit_max_center_y_m').value)
        self._detected_exit_merge_dist_m = float(
            self.get_parameter('detected_exit_merge_dist_m').value)
        self._blue_exit_suppression_margin_m = float(
            self.get_parameter('blue_exit_suppression_margin_m').value)
        self._exit_pass_through_m = float(self.get_parameter('exit_pass_through_m').value)
        self._exit_nav_standoff_m = float(
            self.get_parameter('exit_nav_standoff_m').value)
        self._exit_cross_linear_vel = float(
            self.get_parameter('exit_cross_linear_vel').value)
        self._exit_cross_heading_kp = float(
            self.get_parameter('exit_cross_heading_kp').value)
        self._exit_cross_angular_vel_limit = float(
            self.get_parameter('exit_cross_angular_vel_limit').value)
        self._exit_cross_timeout_sec = float(
            self.get_parameter('exit_cross_timeout_sec').value)

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
        self.axis_sub = self.create_subscription(
            PoseStamped, '/mission_axis', self.mission_axis_callback, 10,
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
        self._opened_blue_door_positions: list[tuple[float, float]] = []
        self._observed_blue_doors: list[dict[str, float]] = []
        self._observed_blue_door_id_to_cluster: dict[str, dict[str, float]] = {}
        self._observed_red_door_positions: list[dict[str, float]] = []
        self._abandoned_door_ids: set[str] = set()
        self._abandoned_blue_door_positions: list[tuple[float, float]] = []
        self._mission_progress_x = self._mission_start_x
        self._mission_axis_origin_x = 0.0
        self._mission_axis_origin_y = 0.0
        self._mission_axis_yaw = self._explore_forward_yaw
        self._mission_axis_received = False
        self._nav_retry_count = 0

        self._door_opening_in_progress = False
        self._final_scan_start_time: Time | None = None
        self._door_open_settle_start_time: Time | None = None
        self._door_open_align_start_time: Time | None = None
        self._last_exit_goal_time: Time | None = None
        self._pre_nav_scan_start_time: Time | None = None
        self._pending_nav_scan_target_id: str | None = None
        self._explore_nav_goal_id: str = ''
        self._explore_nav_failure_count = 0
        self._explore_nav_goal_sign = 0.0
        self._explore_nav_goal_progress = 0.0
        self._recent_failed_explore_goals: list[tuple[float, float, float]] = []
        self._force_final_scan_before_exit = False
        self._active_exit_goal: tuple[float, float, float] | None = None
        self._exit_crossing_start_time: Time | None = None

        self.timer = self.create_timer(0.5, self.fsm_loop)
        exit_desc = (
            f'fallback=({self._exit_x:.1f}, {self._exit_y:.1f})'
            if self._allow_fallback_exit_goal
            else 'observed green exit only')
        self.get_logger().info(
            f'StateMachineNode started | '
            f'nav_timeout={self._nav_timeout_sec}s  '
            f'max_retries={self._max_nav_retries}  '
            f'explore_timeout={self._explore_timeout_sec}s  '
            f'exit={exit_desc}'
        )

    # ── 콜백 ──────────────────────────────────────────────
    def door_callback(self, msg: DoorInfo):
        # 초록 문(비상구) → 별도 저장, detected_doors에 추가하지 않음
        if msg.door_color == 'green':
            self._update_detected_exit(msg)
            return

        if msg.door_color == 'red':
            self._record_observed_physical_red_door(msg)

        # 1차: 동일 door_id라도 map 위치가 가까울 때만 같은 물리 문으로 갱신
        existing = next(
            (d for d in self.detected_doors
             if self._is_same_door_id_observation(d, msg)), None)
        if existing:
            self._merge_door_observation(existing, msg)
            self._record_observed_physical_blue_door(existing)
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
            self._record_observed_physical_blue_door(nearby)
            return
        self._ensure_unique_detected_door_id(msg)
        self.detected_doors.append(msg)
        self._record_observed_physical_blue_door(msg)

    def _update_detected_exit(self, msg: DoorInfo):
        new_xy = self._exit_observation_xy(msg)
        if new_xy is None:
            if self._exit_door is None:
                self._exit_door = copy.deepcopy(msg)
            return

        if not self._exit_candidate_axis_valid(new_xy, msg.door_id):
            return

        if self._exit_door is None:
            self.get_logger().info(
                f'비상구 탐지: {msg.door_id} ({new_xy[0]:.1f}, {new_xy[1]:.1f})')
            self._exit_door = copy.deepcopy(msg)
            return

        old_xy = self._exit_observation_xy(self._exit_door)
        if old_xy is None:
            self._exit_door = copy.deepcopy(msg)
            return

        old_id = self._exit_door.door_id
        if not self._exit_candidate_axis_valid(old_xy, old_id, clear_existing=True):
            self.get_logger().info(
                f'Existing exit candidate {old_id} failed current centerline validation; '
                f'replacing it with {msg.door_id}.')
            self._exit_door = copy.deepcopy(msg)
            return

        old_progress = self._axis_progress_xy(old_xy[0], old_xy[1])
        new_progress = self._axis_progress_xy(new_xy[0], new_xy[1])
        old_center_error = self._exit_center_y_error(old_xy)
        new_center_error = self._exit_center_y_error(new_xy)
        if (new_center_error + 0.15 < old_center_error
                and new_progress >= old_progress - 1.0
                and msg.confidence >= self._exit_door.confidence - 0.15):
            self.get_logger().warn(
                f'Exit candidate updated to more centered observation: '
                f'{old_id} error {old_center_error:.1f}m -> '
                f'{msg.door_id} error {new_center_error:.1f}m.')
            self._exit_door = copy.deepcopy(msg)
            return

        jump = math.hypot(new_xy[0] - old_xy[0], new_xy[1] - old_xy[1])
        if jump <= max(0.1, self._detected_exit_merge_dist_m):
            if (self._detected_exit_max_center_y_m > 0.0
                    and new_center_error > old_center_error + 0.20):
                self.get_logger().debug(
                    f'Exit candidate {msg.door_id} is farther from centerline '
                    f'({new_center_error:.1f}m > {old_center_error:.1f}m); '
                    'keeping the centered estimate.')
                return
            self._merge_door_observation(self._exit_door, msg)
            return

        old_lateral = abs(self._axis_lateral_xy(old_xy[0], old_xy[1]) - self._explore_center_y)
        new_lateral = abs(self._axis_lateral_xy(new_xy[0], new_xy[1]) - self._explore_center_y)
        if (msg.confidence > self._exit_door.confidence + 0.20
                and new_progress >= old_progress - 0.5
                and new_lateral <= old_lateral + 0.8):
            self.get_logger().warn(
                f'비상구 관측이 {jump:.1f}m 이동했습니다. 더 강한 후보 '
                f'{msg.door_id} ({new_xy[0]:.1f}, {new_xy[1]:.1f})로 갱신합니다.')
            self._exit_door = copy.deepcopy(msg)
            return

        self.get_logger().debug(
            f'비상구 후보 {msg.door_id} ({new_xy[0]:.1f}, {new_xy[1]:.1f})는 '
            f'기존 관측과 {jump:.1f}m 떨어져 있어 출구 좌표 갱신에서 제외합니다.')

    def _exit_observation_xy(self, door: DoorInfo) -> tuple[float, float] | None:
        if door.door_pose.header.frame_id == 'map':
            return (float(door.door_pose.pose.position.x), float(door.door_pose.pose.position.y))
        if door.handle_position.header.frame_id == 'map':
            return (float(door.handle_position.point.x), float(door.handle_position.point.y))
        return None

    def _is_same_door_id_observation(
            self, existing: DoorInfo, msg: DoorInfo) -> bool:
        if existing.door_id != msg.door_id:
            return False
        if (existing.door_pose.header.frame_id == 'map'
                and msg.door_pose.header.frame_id == 'map'):
            ex = existing.door_pose.pose.position
            mp = msg.door_pose.pose.position
            return math.hypot(mp.x - ex.x, mp.y - ex.y) <= max(
                0.8, self._observed_candidate_merge_dist())
        return True

    def _ensure_unique_detected_door_id(self, msg: DoorInfo):
        existing_ids = {d.door_id for d in self.detected_doors}
        if msg.door_id not in existing_ids:
            return
        base_id = msg.door_id
        suffix = 1
        while f'{base_id}_{suffix}' in existing_ids:
            suffix += 1
        msg.door_id = f'{base_id}_{suffix}'

    def fire_callback(self, msg: FireInfo):
        self.fire_info = msg

    def scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def mission_axis_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id != 'map':
            return
        yaw = self._pose_yaw(msg.pose)
        self._mission_axis_origin_x = float(msg.pose.position.x)
        self._mission_axis_origin_y = float(msg.pose.position.y)
        self._mission_axis_yaw = yaw
        self._mission_axis_received = True
        pose = self._current_map_pose()
        if pose is not None:
            progress = self._axis_progress_xy(pose[0], pose[1])
            if self.state == State.IDLE:
                self._mission_progress_x = progress
            else:
                self._mission_progress_x = max(self._mission_progress_x, progress)
        self.get_logger().info(
            f'Mission axis updated from /map: origin='
            f'({self._mission_axis_origin_x:.2f}, {self._mission_axis_origin_y:.2f}), '
            f'yaw={math.degrees(self._mission_axis_yaw):.1f}deg',
            once=True)

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
        if self._require_mission_axis and not self._mission_axis_received:
            self.get_logger().info(
                'Mission axis 대기 중: 초기 SLAM/static map 기준축 수신 후 탐색을 시작합니다.',
                once=True)
            return

        if self._start_without_fire:
            self.get_logger().info(
                'start_without_fire enabled. Starting corridor exploration.')
            self._failed_door_ids.clear()
            self._opened_door_ids.clear()
            self._opened_blue_door_positions.clear()
            self._abandoned_door_ids.clear()
            self._abandoned_blue_door_positions.clear()
            self._observed_blue_doors.clear()
            self._observed_blue_door_id_to_cluster.clear()
            self._observed_red_door_positions.clear()
            self._mission_progress_x = self._current_mission_progress()
            self._exit_door = None
            self._active_exit_goal = None
            self._nav_retry_count = 0
            self._transition(State.EXPLORING)
            return

        if self.fire_info and self.fire_info.detected:
            self.get_logger().info(
                f'Fire detected ({self.fire_info.red_door_count} red door(s)). '
                'Starting mission.')
            self._failed_door_ids.clear()
            self._opened_door_ids.clear()
            self._opened_blue_door_positions.clear()
            self._abandoned_door_ids.clear()
            self._abandoned_blue_door_positions.clear()
            self._observed_blue_doors.clear()
            self._observed_blue_door_id_to_cluster.clear()
            self._observed_red_door_positions.clear()
            self._mission_progress_x = self._current_mission_progress()
            self._exit_door = None
            self._active_exit_goal = None
            self._nav_retry_count = 0
            self._transition(State.EXPLORING)

    def _on_exploring(self):
        # 개방 완료 + 실패 블랙리스트 제외한 파란 문 후보
        blue_candidates = [
            d for d in self.detected_doors
            if d.door_color == 'blue'
            and d.door_id not in self._opened_door_ids
            and d.door_id not in self._abandoned_door_ids
            and self._is_door_recent_observation(d)
            and not self._is_door_physically_opened(d)
            and not self._is_door_observation_suppressed(d)
            and not self._is_door_physically_abandoned(d)
            and self._is_blue_door_observation_confirmed(d)
            and not self._is_blue_candidate_near_recent_red(d)
            and self._is_blue_door_at_valid_wall_position(d)
        ]
        raw_safe_doors = [
            d for d in blue_candidates
            if d.door_id not in self._failed_door_ids
        ]

        safe_doors = [
            d for d in raw_safe_doors
            if self._door_has_map_identity(d)
        ]
        if not safe_doors:
            stable_revisit_doors = self._stable_unopened_detected_blue_doors(
                [d for d in blue_candidates if self._door_has_map_identity(d)])
            if stable_revisit_doors:
                safe_doors = stable_revisit_doors
                self.get_logger().warn(
                    f'관찰됐지만 아직 안 열린 파란 문 {len(safe_doors)}개 재확인. '
                    '가장 가까운 후보로 즉시 재접근합니다.')

        if self._continue_pending_target_navigation():
            return

        if (self._opened_exit_ready_count() >= self._min_opened_doors_before_exit
                and not self._observed_unopened_blue_doors()
                and self._past_exit_line()):
            self.get_logger().info('비상구 라인 통과 확인. 미션 완료로 전환합니다.')
            self._transition(State.MISSION_COMPLETE)
            return
        if self._final_scan_start_time is not None:
            if self._run_final_scan_before_exit():
                return
            self._force_final_scan_before_exit = False
            if not safe_doors:
                unopened_observed = self._observed_unopened_blue_doors()
                if unopened_observed:
                    self._hold_exit_for_unopened_blue_doors(unopened_observed)
                    self._rotate_to_scan()
                    return

                opened_ready_count = self._opened_exit_ready_count()
                if opened_ready_count < self._min_opened_doors_before_exit:
                    self._explore_start_time = self.get_clock().now()
                    self._rotate_to_scan()
                    return

                if self._valid_detected_exit() or self._allow_fallback_exit_goal:
                    self.get_logger().info(
                        f'Final exit scan complete. '
                        f'opened={self._resolved_blue_door_count()}, '
                        'no remaining blue door selected; moving to exit.')
                    self._transition(State.EXITING)
                    return

                self.get_logger().warn(
                    'Final exit scan complete, but no valid green exit is '
                    'available yet. Continuing observation-based scan.')
                self._explore_start_time = self.get_clock().now()


        if not safe_doors:
            if ((self._valid_detected_exit() or self._allow_fallback_exit_goal)
                    and self._opened_exit_ready_count() >= self._min_opened_doors_before_exit
                    and not self._observed_unopened_blue_doors()):
                if self._run_final_scan_before_exit():
                    return
                self._force_final_scan_before_exit = False

                unopened_observed = self._observed_unopened_blue_doors()
                if unopened_observed:
                    self._hold_exit_for_unopened_blue_doors(unopened_observed)
                    self._rotate_to_scan()
                    return

                if self._valid_detected_exit() or self._allow_fallback_exit_goal:
                    self.get_logger().info(
                        f'Observed exit scan complete. '
                        f'opened={self._resolved_blue_door_count()}, '
                        'no remaining blue door selected; moving to exit.')
                    self._transition(State.EXITING)
                    return

                self.get_logger().warn(
                    'Observed exit was lost after final scan. Continuing scan.')
                self._explore_start_time = self.get_clock().now()
                self._rotate_to_scan()
                return

            # 탐색 타임아웃 → 모든 파란 문 개방 완료 판단
            if self._explore_start_time is not None:
                force_exit_check = self._force_final_scan_before_exit
                elapsed = (
                    self.get_clock().now() - self._explore_start_time
                ).nanoseconds / 1e9
                if force_exit_check or elapsed >= self._explore_timeout_sec:
                    opened_count = self._resolved_blue_door_count()
                    if self._run_final_scan_before_exit():
                        return
                    self._force_final_scan_before_exit = False

                    unopened_observed = self._observed_unopened_blue_doors()
                    if unopened_observed:
                        self._hold_exit_for_unopened_blue_doors(unopened_observed)
                        self._rotate_to_scan()
                        return

                    opened_ready_count = self._opened_exit_ready_count()
                    if opened_ready_count < self._min_opened_doors_before_exit:
                        self.get_logger().warn(
                            f'{elapsed:.0f}s 탐색 후 새 파란 문 없음. '
                            f'아직 개방한 문이 {opened_ready_count}/'
                            f'{self._min_opened_doors_before_exit}개라 탐색을 계속합니다.')
                        self._explore_start_time = self.get_clock().now()
                        self._rotate_to_scan()
                        return

                    if (not self._valid_detected_exit()
                            and not self._allow_fallback_exit_goal):
                        self.get_logger().warn(
                            f'{elapsed:.0f}s 탐색 후 새 파란 문 없음. '
                            '초록 비상구가 아직 관측되지 않아 출구 좌표를 '
                            '하드코딩하지 않고 계속 스캔합니다.')
                        self._explore_start_time = self.get_clock().now()
                        if self._send_explore_waypoint():
                            return
                        self._rotate_to_scan()
                        return

                    self.get_logger().info(
                        f'{elapsed:.0f}s 탐색 후 새 파란 문 없음. '
                        f'총 {opened_count}개 문 개방 완료. '
                        '비상구로 이동합니다.')
                    self._transition(State.EXITING)
                    return
            if self._force_final_scan_before_exit:
                self._rotate_to_scan()
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

        # 파란 문이 관측되면 탐색 waypoint를 더 보내지 않고 가장 가까운 문을
        # 즉시 Nav2 목표로 고정한다. 목표까지의 장애물 회피는 Nav2 costmap과
        # planner/controller가 담당한다.
        selected_door = self._nearest_forward_blue_door(safe_doors)
        if selected_door is None:
            self._rotate_to_scan()
            return
        self.target_door = self._door_with_axis_aligned_approach(selected_door)

        if self._run_pre_nav_scan(self.target_door):
            return

        self.get_logger().info(
            f'파란 문 발견 즉시 목표 고정: {self.target_door.door_id}  '
            f'개방 완료: {self._resolved_blue_door_count()}개  '
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
            if (self.target_door is not None
                    and self.target_door.door_color == 'blue'):
                prep = self._prepare_door_opening_pose(self.target_door)
                if prep == 'waiting':
                    self._nav_done = True
                    return
                if prep == 'retry':
                    if self._retry_precise_door_approach():
                        return
                    self._handle_nav_failure(reason='door_not_aligned')
                    return
                if prep == 'failed':
                    self._handle_nav_failure(reason='door_not_aligned')
                    return

            if self._settle_before_opening_door():
                self._nav_done = True
                return
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
                self._remember_failed_explore_goal()
                self._flip_explore_lane_after_failure()
                self._nav_done = False
                self._nav_failed = False
                self._transition(State.EXPLORING)
                return

        if self._nav_done:
            self._nav_done = False
            self._explore_nav_failure_count = 0
            self._recent_failed_explore_goals.clear()
            self.get_logger().info('탐색 waypoint 도착. LiDAR+camera 재스캔을 계속합니다.')
            self._alternate_explore_lane_after_success()
            self._transition(State.EXPLORING)
        elif self._nav_failed:
            self._nav_failed = False
            self._remember_failed_explore_goal()
            if self._handle_explore_nav_failure_limit():
                return
            self._flip_explore_lane_after_failure()
            self.get_logger().warn('탐색 waypoint 이동 실패. 반대 차선 우선으로 재스캔합니다.')
            self._transition(State.EXPLORING)

    def _handle_explore_nav_failure_limit(self) -> bool:
        self._explore_nav_failure_count += 1
        if (self._opened_exit_ready_count() >= self._min_opened_doors_before_exit
                and not self._observed_unopened_blue_doors()
                and self._explore_nav_failure_count
                >= self._max_explore_nav_failures_before_exit):
            self.get_logger().warn(
                f'탐색 waypoint 실패 {self._explore_nav_failure_count}회 누적. '
                '비상구 전 최종 파란 문 스캔으로 복귀합니다.')
            self._explore_nav_failure_count = 0
            self._force_final_scan_before_exit = True
            self._transition(State.EXPLORING)
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
                    abandoned = self._record_observed_blue_door_failure(
                        self.target_door, failure_kind='approach')
                    self.target_door = None
                    self._nav_retry_count = 0
                    if abandoned:
                        self.get_logger().error(
                            f'접근 포기: {failed_id}. 제한 횟수까지 문 앞 접근에 실패해 '
                            '해당 물리 문 후보를 미션에서 제외하고 탐색을 계속합니다.')
                    else:
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
            self._handle_door_open_failure(f'service error: {e}')
            return

        if result.success:
            self._mark_door_opened(self.target_door)
            if self.target_door is not None:
                self._clear_observed_blue_door_failures(self.target_door)
            self._nav_retry_count = 0   # 성공 시 재시도 카운터 리셋
            self.get_logger().info(
                f'문 개방 성공: {self.target_door.door_id}  '
                f'누적 개방: {self._resolved_blue_door_count()}개')
            self._transition(State.DOOR_OPENED)
        else:
            self.get_logger().error(f'문 개방 실패: {result.message}')
            self._handle_door_open_failure(result.message)

    def _on_door_opened(self):
        # 즉시 다음 파란 문 탐색으로 전환
        self.target_door = None
        self.get_logger().info(
            f'다음 파란 문 탐색 시작 (개방 완료: {self._resolved_blue_door_count()}개)')
        self._transition(State.EXPLORING)

    def _send_explore_waypoint(self) -> bool:
        if not self._explore_nav_enabled:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False
        x, y, yaw = pose
        progress = self._axis_progress_xy(x, y)
        lateral = self._axis_lateral_xy(x, y)
        explore_limit_progress = self._explore_limit_progress()
        min_forward_step = max(0.25, self._explore_nav_min_step_m)
        if progress >= explore_limit_progress:
            return False
        if explore_limit_progress - progress < min_forward_step:
            self.get_logger().info(
                f'탐색 waypoint 보류: 출구 전방 한계까지 남은 거리가 '
                f'{explore_limit_progress - progress:.2f}m라 최종 스캔/출구 판단을 우선합니다.')
            return False
        lateral_error = lateral - self._explore_center_y
        if abs(lateral_error) > self._explore_nav_recenter_abs_y_m:
            self.get_logger().info(
                f'탐색 waypoint 중앙 복귀: 측면 이탈 lateral={lateral:.2f}, '
                f'먼 문 waypoint보다 중앙 차선 재진입 우선 '
                f'(limit={self._explore_nav_recenter_abs_y_m:.2f})')
            return self._send_recenter_waypoint(
                progress, lateral, explore_limit_progress, min_forward_step)

        preferred_lane_sign, preferred_reason = self._preferred_lane_sign()
        radar_lane_sign, radar_reason = self._radar_clear_lane_sign()
        front_avoid_sign, front_avoid_reason = self._front_obstacle_avoid_lane_sign(
            x, y, yaw)
        goal_lateral, lane_sign, lane_reason = self._select_explore_goal_y(
            x, y, yaw, lateral, preferred_lane_sign, preferred_reason,
            radar_lane_sign, radar_reason,
            front_avoid_sign, front_avoid_reason)
        limited_goal_lateral = self._limit_explore_goal_lateral_step(lateral, goal_lateral)
        if abs(limited_goal_lateral - goal_lateral) > 0.01:
            lane_reason += (
                f' | 좌우 이동량 제한 {goal_lateral - lateral:.2f}m→'
                f'{limited_goal_lateral - lateral:.2f}m')
            goal_lateral = limited_goal_lateral
        selected_clearance = self._lane_scan_clearance(x, y, yaw, goal_lateral)
        step_m = self._explore_nav_step_m
        if (math.isfinite(selected_clearance)
                and selected_clearance < self._explore_nav_reduce_step_clearance_m):
            safe_step = selected_clearance - self._explore_nav_goal_clearance_margin_m
            if safe_step < 0.20:
                self.get_logger().info(
                    f'탐색 waypoint 보류: 선택 차선 전방 여유가 '
                    f'{selected_clearance:.2f}m로 너무 짧아 재스캔합니다.')
                return False
            step_m = min(
                step_m,
                max(self._explore_nav_min_step_m, safe_step))
            lane_reason += (
                f' | 근접 장애물 여유 {selected_clearance:.2f}m, '
                f'waypoint 전진거리 {step_m:.2f}m로 축소')
        goal_progress = min(progress + step_m, explore_limit_progress)
        if goal_progress - progress < min_forward_step:
            self.get_logger().info(
                f'탐색 waypoint 보류: 요청 가능한 전진거리 '
                f'{goal_progress - progress:.2f}m가 너무 짧아 최종 스캔/출구 판단을 우선합니다.')
            return False
        goal_x, goal_y = self._axis_to_map_xy(goal_progress, goal_lateral)

        goal = DoorInfo()
        now = self.get_clock().now().to_msg()
        goal.header.stamp = now
        goal.header.frame_id = 'map'
        goal.door_id = f'explore_waypoint_{goal_progress:.1f}'
        goal.door_color = 'explore'
        goal.confidence = 1.0
        goal.door_pose.header.stamp = now
        goal.door_pose.header.frame_id = 'map'
        goal.door_pose.pose.position.x = goal_x
        goal.door_pose.pose.position.y = goal_y
        goal.door_pose.pose.orientation.w = 1.0

        if lane_sign != 0.0:
            self._last_lane_sign = lane_sign

        self._explore_nav_goal_id = goal.door_id
        self._explore_nav_goal_sign = lane_sign
        self._explore_nav_goal_progress = goal_progress
        self.get_logger().info(
            f'탐색 waypoint 계획 요청: {goal.door_id} '
            f'progress={goal_progress:.2f}, lateral={goal_lateral:.2f}, '
            f'map=({goal_x:.2f}, {goal_y:.2f}) | {lane_reason}')
        self.target_door_pub.publish(goal)
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.EXPLORING_NAVIGATING)
        return True

    def _limit_explore_goal_lateral_step(self, current_y: float, goal_y: float) -> float:
        max_step = self._explore_nav_max_lateral_step_m
        if max_step <= 0.0:
            return goal_y
        delta = goal_y - current_y
        if abs(delta) <= max_step:
            return goal_y
        limited = current_y + math.copysign(max_step, delta)
        hard_limit = max(
            abs(self._explore_y_hard_limit_m),
            abs(self._explore_nav_recenter_abs_y_m))
        return max(
            self._explore_center_y - hard_limit,
            min(self._explore_center_y + hard_limit, limited))

    def _send_recenter_waypoint(
            self, progress: float, lateral: float, explore_limit_progress: float,
            min_forward_step: float) -> bool:
        if explore_limit_progress - progress < min_forward_step:
            return False
        sign = 1.0 if lateral > self._explore_center_y else -1.0
        lateral_step = max(0.25, self._explore_nav_max_lateral_step_m)
        goal_lateral = lateral - sign * lateral_step
        if sign > 0.0:
            goal_lateral = max(self._explore_center_y, goal_lateral)
        else:
            goal_lateral = min(self._explore_center_y, goal_lateral)
        hard_limit = max(
            abs(self._explore_y_hard_limit_m),
            abs(self._explore_nav_recenter_abs_y_m))
        goal_lateral = max(
            self._explore_center_y - hard_limit,
            min(self._explore_center_y + hard_limit, goal_lateral))
        goal_progress = min(progress + min_forward_step, explore_limit_progress)
        goal_x, goal_y = self._axis_to_map_xy(goal_progress, goal_lateral)

        goal = DoorInfo()
        now = self.get_clock().now().to_msg()
        goal.header.stamp = now
        goal.header.frame_id = 'map'
        goal.door_id = f'explore_recenter_{goal_progress:.1f}'
        goal.door_color = 'explore'
        goal.confidence = 1.0
        goal.door_pose.header.stamp = now
        goal.door_pose.header.frame_id = 'map'
        goal.door_pose.pose.position.x = goal_x
        goal.door_pose.pose.position.y = goal_y
        goal.door_pose.pose.orientation.w = 1.0
        self._explore_nav_goal_id = goal.door_id
        self._explore_nav_goal_sign = 0.0
        self._explore_nav_goal_progress = goal_progress
        self.get_logger().info(
            f'중앙 복귀 waypoint 계획 요청: {goal.door_id} '
            f'progress={goal_progress:.2f}, lateral={goal_lateral:.2f}, '
            f'map=({goal_x:.2f}, {goal_y:.2f})')
        self.target_door_pub.publish(goal)
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.EXPLORING_NAVIGATING)
        return True

    def _defer_far_target_door(self, door: DoorInfo | None) -> bool:
        if door is not None and door.door_color == 'blue':
            return False
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

    def _continue_pending_target_navigation(self) -> bool:
        target = self.target_door
        if target is None or target.door_color != 'blue':
            return False

        target_key = self._pre_nav_scan_key(target)
        if (self._pending_nav_scan_target_id != target_key
                and self._pre_nav_scan_start_time is None):
            return False

        if (target.door_id in self._opened_door_ids
                or target.door_id in self._abandoned_door_ids
                or self._is_door_physically_opened(target)
                or self._is_door_physically_abandoned(target)):
            self._pre_nav_scan_start_time = None
            self._pending_nav_scan_target_id = None
            self.target_door = None
            return False

        self._explore_start_time = self.get_clock().now()
        if not self._ready_to_start_navigation():
            self._rotate_to_scan()
            return True
        if self._run_pre_nav_scan(target):
            return True

        self.get_logger().info(
            f'Locked blue target after pre-scan: {target.door_id} '
            f'opened={self._resolved_blue_door_count()} '
            f'retry={self._nav_retry_count}/{self._max_nav_retries}')
        self.target_door_pub.publish(target)
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.NAVIGATING)
        return True

    def _run_pre_nav_scan(self, target: DoorInfo) -> bool:
        if self._pre_nav_scan_sec <= 0.0:
            return False
        now = self.get_clock().now()
        target_key = self._pre_nav_scan_key(target)
        if self._pending_nav_scan_target_id != target_key or self._pre_nav_scan_start_time is None:
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
        if target.door_color == 'blue' and self._door_has_map_identity(target):
            x, y = self._door_identity_xy(target)
            progress = self._axis_progress_xy(x, y)
            lateral = self._axis_lateral_xy(x, y)
            # HSV fallback can re-issue IDs for the same physical door. Use the
            # observed map position so the pre-navigation scan keeps accumulating.
            progress_key = round(progress * 2.0) / 2.0
            lateral_key = round(lateral * 2.0) / 2.0
            return f'{target.door_color}:{progress_key:.1f}:{lateral_key:.1f}'
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
        x, y, yaw = pose
        lateral = self._axis_lateral_xy(x, y)
        heading_error = abs(self._normalize_angle(self._mission_forward_yaw() - yaw))
        if abs(lateral - self._explore_center_y) > self._nav_start_max_abs_y_m:
            self.get_logger().info(
                f'Nav2 목표 대기: 중앙 복귀 중 '
                f'(lateral={lateral:.2f}, limit={self._nav_start_max_abs_y_m:.2f})',
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
                if abs(self._door_lateral(d) - self._explore_center_y)
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
        pose = self._current_map_pose()
        if pose is not None:
            robot_progress = self._axis_progress_xy(pose[0], pose[1])
            min_x = max(
                min_x,
                robot_progress + self._target_door_min_robot_forward_m)
        forward = [
            d for d in candidates
            if self._door_progress(d) > min_x
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
            door_progress = self._axis_progress_xy(door_x, door_y)
            self._mission_progress_x = max(
                self._mission_progress_x,
                door_progress)

        self._record_opened_physical_blue_door(door)
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

    def _resolved_blue_door_count(self) -> int:
        physical_count = len(self._opened_blue_door_positions) + len(
            self._abandoned_blue_door_positions)
        if physical_count:
            return physical_count
        return len([door_id for door_id in self._opened_door_ids if door_id])

    def _opened_exit_ready_count(self) -> int:
        resolved_positions = (
            len(self._opened_blue_door_positions)
            + len(self._abandoned_blue_door_positions)
        )
        if resolved_positions:
            return resolved_positions
        return (
            len([door_id for door_id in self._opened_door_ids if door_id])
            + len([door_id for door_id in self._abandoned_door_ids if door_id])
        )

    def _on_exiting(self):
        if self._exit_crossing_start_time is not None:
            self._continue_exit_crossing()
            return

        if self._nav_done:
            self._nav_done = False
            if self._exit_pose_complete():
                self._transition(State.MISSION_COMPLETE)
            else:
                self._start_exit_crossing()
        elif self._nav_failed:
            if self._active_exit_goal is not None:
                self._nav_failed = False
                self.get_logger().warn(
                    'Exit Nav2 alignment failed; switching to direct exit crossing.')
                self._start_exit_crossing()
                return
            if self._exit_retry_ready():
                self._nav_failed = False
                self.get_logger().warn('비상구 이동 실패. 재시도...')
                self._send_exit_goal()

    def _on_mission_complete(self):
        self.get_logger().info(
            f'미션 완료! 개방한 문: {self._resolved_blue_door_count()}개. '
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
        if self._valid_detected_exit():
            xy = self._exit_identity_xy()
            if xy is not None:
                exit_progress = self._axis_progress_xy(xy[0], xy[1])
            else:
                return False
        elif self._allow_fallback_exit_goal:
            exit_progress = self._axis_progress_xy(self._exit_x, self._exit_y)
        else:
            return False
        robot_progress = self._axis_progress_xy(x, y)
        return robot_progress >= (exit_progress - self._exit_complete_margin_m)

    def _exit_pose_complete(self) -> bool:
        pose = self._current_map_pose()
        if pose is None:
            return False
        x, y, yaw = pose
        goal = self._active_exit_goal
        if goal is None:
            return False
        goal_x, goal_y, goal_yaw = goal
        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)
        yaw_error = abs(self._normalize_angle(goal_yaw - yaw))
        robot_progress = self._axis_progress_xy(x, y)
        goal_progress = self._axis_progress_xy(goal_x, goal_y)
        complete = (
            dist <= max(0.30, self._exit_complete_margin_m)
            and yaw_error <= 0.65
        ) or robot_progress >= goal_progress - self._exit_complete_margin_m
        if not complete:
            self.get_logger().warn(
                f'비상구 완료 조건 미달: dist={dist:.2f}m, '
                f'goal=({goal_x:.2f}, {goal_y:.2f}), '
                f'robot=({x:.2f}, {y:.2f}), '
                f'yaw_error={math.degrees(yaw_error):.0f}deg')
        return complete

    def _start_exit_crossing(self):
        self._exit_crossing_start_time = self.get_clock().now()
        self.get_logger().info(
            '비상구 앞 정렬 완료. 마지막 통과 구간은 짧은 직진으로 수행합니다.')
        self.cmd_vel_pub.publish(Twist())

    def _continue_exit_crossing(self):
        now = self.get_clock().now()
        elapsed = (now - self._exit_crossing_start_time).nanoseconds / 1e9
        if self._exit_pose_complete():
            self.cmd_vel_pub.publish(Twist())
            self._exit_crossing_start_time = None
            self._transition(State.MISSION_COMPLETE)
            return

        if elapsed >= self._exit_cross_timeout_sec:
            self.cmd_vel_pub.publish(Twist())
            self._exit_crossing_start_time = None
            self.get_logger().warn(
                f'비상구 직진 통과 제한시간 {elapsed:.1f}s 초과. 출구 목표를 다시 보냅니다.')
            self._send_exit_goal()
            return

        twist = Twist()
        twist.linear.x = max(0.0, self._exit_cross_linear_vel)
        goal = self._active_exit_goal
        pose = self._current_map_pose()
        if goal is not None and pose is not None:
            yaw_error = self._normalize_angle(goal[2] - pose[2])
            angular_limit = max(0.0, self._exit_cross_angular_vel_limit)
            angular = self._exit_cross_heading_kp * yaw_error
            if angular_limit > 0.0:
                angular = max(-angular_limit, min(angular_limit, angular))
            twist.angular.z = angular
            yaw_abs = abs(yaw_error)
            if yaw_abs > math.radians(10.0):
                scale = max(
                    0.35,
                    1.0 - min(yaw_abs, math.pi / 2.0) / (math.pi / 2.0) * 0.65)
                twist.linear.x *= scale
        self.cmd_vel_pub.publish(twist)

    def _send_exit_goal(self) -> bool:
        """Send an observed exit goal. Fallback coordinates are opt-in only."""
        exit_nav_goal = None
        exit_complete_goal = None
        exit_goal_pair = self._detected_exit_goal_pair()
        if exit_goal_pair is not None:
            exit_nav_goal, exit_complete_goal = exit_goal_pair

        if self._valid_detected_exit():
            if exit_nav_goal is None or exit_complete_goal is None:
                if not self._allow_fallback_exit_goal:
                    self.get_logger().warn(
                        '비상구 관측은 있으나 통과 목표 계산에 실패했습니다. '
                        '하드코딩 좌표를 쓰지 않고 재관측합니다.')
                    self._active_exit_goal = None
                    return False
                exit_nav_goal = (self._exit_x, self._exit_y, self._exit_yaw)
                exit_complete_goal = exit_nav_goal
            self.get_logger().info(
                f'비상구 탐지 확인: {self._exit_door.door_id} '
                f'({self._exit_door.door_pose.pose.position.x:.1f}, '
                f'{self._exit_door.door_pose.pose.position.y:.1f}); '
                f'Nav2 정렬 목표 ({exit_nav_goal[0]:.1f}, {exit_nav_goal[1]:.1f}), '
                f'통과 완료 목표 ({exit_complete_goal[0]:.1f}, {exit_complete_goal[1]:.1f})를 사용합니다.')
        elif self._allow_fallback_exit_goal:
            exit_nav_goal = (self._exit_x, self._exit_y, self._exit_yaw)
            exit_complete_goal = exit_nav_goal
            self.get_logger().warn(
                f'비상구 미탐지 → fallback 통과 좌표 '
                f'({self._exit_x:.1f}, {self._exit_y:.1f}) 사용')
        else:
            self.get_logger().warn(
                '비상구 미탐지. 초기 로봇 원점 기준 관측 좌표만 사용하므로 '
                '출구가 보일 때까지 탐색을 계속합니다.')
            self._active_exit_goal = None
            return False
        self._active_exit_goal = exit_complete_goal

        msg = DoorInfo()
        now = self.get_clock().now().to_msg()
        msg.header.stamp              = now
        msg.header.frame_id           = 'map'
        msg.door_id                   = 'emergency_exit_pass_through'
        msg.door_color                = 'green'
        msg.door_pose.header.stamp    = now
        msg.door_pose.header.frame_id = 'map'
        msg.door_pose.pose.position.x = exit_nav_goal[0]
        msg.door_pose.pose.position.y = exit_nav_goal[1]
        msg.door_pose.pose.orientation.z = math.sin(exit_nav_goal[2] / 2.0)
        msg.door_pose.pose.orientation.w = math.cos(exit_nav_goal[2] / 2.0)
        self.target_door_pub.publish(msg)
        self._last_exit_goal_time = self.get_clock().now()
        self._nav_done   = False
        self._nav_failed = False
        return True

    def _exit_retry_ready(self) -> bool:
        if self._last_exit_goal_time is None:
            return True
        elapsed = (
            self.get_clock().now() - self._last_exit_goal_time
        ).nanoseconds / 1e9
        return elapsed >= self._exit_retry_delay_sec

    def _valid_detected_exit(self) -> bool:
        if not self._use_detected_exit or self._exit_door is None:
            return False
        pose = self._exit_door.door_pose
        if pose.header.frame_id != 'map':
            return False

        x = pose.pose.position.x
        y = pose.pose.position.y
        if not self._exit_candidate_axis_valid(
                (float(x), float(y)),
                self._exit_door.door_id,
                clear_existing=True):
            return False

        if not self._detected_exit_use_expected_region:
            return True
        exit_progress = self._axis_progress_xy(x, y)
        fallback_progress = self._axis_progress_xy(self._exit_x, self._exit_y)
        exit_lateral = self._axis_lateral_xy(x, y)
        fallback_lateral = self._axis_lateral_xy(self._exit_x, self._exit_y)
        min_progress = fallback_progress - self._detected_exit_min_x_margin_m
        if (exit_progress < min_progress
                or abs(exit_lateral - fallback_lateral) > self._detected_exit_max_y_error_m):
            self.get_logger().warn(
                f'비상구 후보 {self._exit_door.door_id} 좌표가 예상 출구 영역 밖입니다 '
                f'({x:.1f}, {y:.1f}); 관측에서 제외합니다.')
            self._exit_door = None
            return False
        return True

    def _exit_candidate_axis_valid(
            self,
            xy: tuple[float, float],
            door_id: str = '',
            clear_existing: bool = False) -> bool:
        x, y = xy
        max_lateral = self._detected_exit_max_abs_lateral_m
        lateral = self._axis_lateral_xy(x, y)
        lateral_error = abs(lateral - self._explore_center_y)
        if max_lateral > 0.0 and lateral_error > max_lateral:
            self.get_logger().info(
                f'비상구 후보 {door_id or "(unknown)"} 제외: '
                f'진행축 횡방향 {lateral_error:.1f}m가 허용값 '
                f'{max_lateral:.1f}m를 초과합니다.',
                throttle_duration_sec=3.0)
            if clear_existing:
                self._exit_door = None
            return False

        max_center_y = self._detected_exit_max_center_y_m
        center_error = self._exit_center_y_error(xy)
        if max_center_y > 0.0 and center_error > max_center_y:
            self.get_logger().info(
                f'Exit candidate {door_id or "(unknown)"} rejected: '
                f'centerline y error {center_error:.1f}m exceeds '
                f'{max_center_y:.1f}m.',
                throttle_duration_sec=3.0)
            if clear_existing:
                self._exit_door = None
            return False

        min_forward = self._detected_exit_min_robot_forward_m
        if min_forward > 0.0:
            pose = self._current_map_pose()
            if pose is not None:
                robot_progress = self._axis_progress_xy(pose[0], pose[1])
                exit_progress = self._axis_progress_xy(x, y)
                if exit_progress < robot_progress + min_forward:
                    self.get_logger().info(
                        f'비상구 후보 {door_id or "(unknown)"} 제외: '
                        f'로봇 진행축보다 충분히 앞에 있지 않습니다 '
                        f'(exit={exit_progress:.1f}, robot={robot_progress:.1f}).',
                        throttle_duration_sec=3.0)
                    if clear_existing:
                        self._exit_door = None
                    return False

        return True

    def _exit_center_y_error(self, xy: tuple[float, float]) -> float:
        _x, y = xy
        center_y = self._mission_axis_origin_y + self._explore_center_y
        return abs(y - center_y)

    def _exit_identity_xy(self) -> tuple[float, float] | None:
        if self._exit_door is None:
            return None
        if self._exit_door.door_pose.header.frame_id == 'map':
            return (
                float(self._exit_door.door_pose.pose.position.x),
                float(self._exit_door.door_pose.pose.position.y),
            )
        if self._exit_door.handle_position.header.frame_id == 'map':
            return (
                float(self._exit_door.handle_position.point.x),
                float(self._exit_door.handle_position.point.y),
            )
        return None

    def _detected_exit_goal_pair(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        if not self._valid_detected_exit():
            return None
        xy = self._exit_identity_xy()
        pose = self._current_map_pose()
        if xy is None or pose is None:
            return None
        exit_x, exit_y = xy
        robot_x, robot_y, _ = pose
        vx = exit_x - robot_x
        vy = exit_y - robot_y
        dist = math.hypot(vx, vy)
        if dist < 0.05:
            yaw = self._exit_yaw
            ux = math.cos(yaw)
            uy = math.sin(yaw)
        else:
            ux = vx / dist
            uy = vy / dist
            yaw = math.atan2(uy, ux)
        pass_m = max(0.0, self._exit_pass_through_m)
        standoff = max(0.0, self._exit_nav_standoff_m)
        nav_goal = (
            exit_x - ux * standoff,
            exit_y - uy * standoff,
            yaw,
        )
        complete_goal = (
            exit_x + ux * pass_m,
            exit_y + uy * pass_m,
            yaw,
        )
        return nav_goal, complete_goal

    def _explore_limit_progress(self) -> float:
        if self._valid_detected_exit():
            xy = self._exit_identity_xy()
            if xy is not None:
                return self._axis_progress_xy(xy[0], xy[1]) - 0.6
        if not self._allow_fallback_exit_goal:
            return float('inf')
        return self._axis_progress_xy(self._exit_x, self._exit_y) - 1.0

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

        x, y, yaw = pose
        desired_yaw = self._mission_forward_yaw()
        lateral_error = self._axis_lateral_xy(x, y) - self._explore_center_y
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
        if center_active and twist.linear.x <= 0.01:
            twist.linear.x = 0.08

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

    def _mission_forward_yaw(self) -> float:
        return self._mission_axis_yaw

    def _axis_unit(self) -> tuple[float, float]:
        return (
            math.cos(self._mission_axis_yaw),
            math.sin(self._mission_axis_yaw),
        )

    def _axis_side_unit(self) -> tuple[float, float]:
        ux, uy = self._axis_unit()
        return -uy, ux

    def _axis_progress_xy(self, x: float, y: float) -> float:
        ux, uy = self._axis_unit()
        return (
            (x - self._mission_axis_origin_x) * ux
            + (y - self._mission_axis_origin_y) * uy)

    def _axis_lateral_xy(self, x: float, y: float) -> float:
        sx, sy = self._axis_side_unit()
        return (
            (x - self._mission_axis_origin_x) * sx
            + (y - self._mission_axis_origin_y) * sy)

    def _axis_to_map_xy(self, progress: float, lateral: float) -> tuple[float, float]:
        ux, uy = self._axis_unit()
        sx, sy = self._axis_side_unit()
        return (
            self._mission_axis_origin_x + progress * ux + lateral * sx,
            self._mission_axis_origin_y + progress * uy + lateral * sy,
        )

    def _current_mission_progress(self) -> float:
        pose = self._current_map_pose()
        if pose is None:
            return self._mission_start_x
        return self._axis_progress_xy(pose[0], pose[1])

    def _door_progress(self, door: DoorInfo) -> float:
        x, y = self._door_identity_xy(door)
        return self._axis_progress_xy(x, y)

    def _door_lateral(self, door: DoorInfo) -> float:
        x, y = self._door_identity_xy(door)
        return self._axis_lateral_xy(x, y)

    def _cluster_progress(self, cluster: dict[str, float]) -> float:
        return self._axis_progress_xy(
            float(cluster.get('x', 0.0)),
            float(cluster.get('y', 0.0)))

    def _preferred_lane_sign(self) -> tuple[float, str]:
        """Return +1 for robot/map-left lane and -1 for robot/map-right lane."""
        if not self._explore_color_lane_bias:
            return 0.0, '색상 차선 선호 비활성'

        blue_doors = [
            d for d in self.detected_doors
            if d.door_color == 'blue'
            and d.door_id not in self._failed_door_ids
            and d.door_id not in self._opened_door_ids
            and d.door_id not in self._abandoned_door_ids
            and self._is_door_recent_observation(d)
            and not self._is_door_physically_opened(d)
            and not self._is_door_observation_suppressed(d)
            and not self._is_door_physically_abandoned(d)
            and self._is_blue_door_observation_confirmed(d)
            and not self._is_blue_candidate_near_recent_red(d)
            and self._is_blue_door_at_valid_wall_position(d)
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

        red_doors = [
            d for d in self.detected_doors
            if d.door_color == 'red'
            and self._is_door_recent_observation(d)
        ]
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
        if door.door_pose.header.frame_id == 'map':
            y = self._door_lateral(door)
        else:
            y = door.door_pose.pose.position.y
        if abs(y - self._explore_center_y) <= 0.08:
            return 0.0
        return 1.0 if y > self._explore_center_y else -1.0

    def _door_has_map_identity(self, door: DoorInfo) -> bool:
        return (
            door.handle_position.header.frame_id == 'map'
            or door.door_pose.header.frame_id == 'map'
        )

    def _door_identity_xy(self, door: DoorInfo) -> tuple[float, float]:
        """Stable map point for duplicate/opened-door identity checks."""
        if door.door_color == 'blue' and door.handle_position.header.frame_id == 'map':
            return (
                float(door.handle_position.point.x),
                float(door.handle_position.point.y),
            )

        if door.door_pose.header.frame_id == 'map':
            return (
                float(door.door_pose.pose.position.x),
                float(door.door_pose.pose.position.y),
            )
        return (
            float(door.handle_position.point.x),
            float(door.handle_position.point.y),
        )

    def _door_handle_xy(self, door: DoorInfo) -> tuple[float, float]:
        """Manipulation/approach point, preferring the estimated handle."""
        if door.handle_position.header.frame_id == 'map':
            return (
                float(door.handle_position.point.x),
                float(door.handle_position.point.y),
            )
        return self._door_identity_xy(door)

    def _record_observed_physical_red_door(self, door: DoorInfo):
        if door.door_color != 'red' or not self._door_has_map_identity(door):
            return
        xy = self._door_identity_xy(door)
        if not self._is_xy_at_configured_wall_lateral(xy):
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        for cluster in self._observed_red_door_positions:
            if math.hypot(
                    xy[0] - float(cluster.get('x', 0.0)),
                    xy[1] - float(cluster.get('y', 0.0))) <= self._blue_red_conflict_dist_m:
                old_count = int(cluster.get('count', 1.0))
                new_count = min(old_count + 1, 999)
                alpha = 1.0 / min(new_count, 5)
                cluster['x'] = (1.0 - alpha) * float(cluster.get('x', xy[0])) + alpha * xy[0]
                cluster['y'] = (1.0 - alpha) * float(cluster.get('y', xy[1])) + alpha * xy[1]
                cluster['count'] = float(new_count)
                cluster['last_seen'] = now_sec
                return
        self._observed_red_door_positions.append({
            'x': xy[0],
            'y': xy[1],
            'count': 1.0,
            'last_seen': now_sec,
        })

    def _record_observed_physical_blue_door(self, door: DoorInfo):
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return
        xy = self._door_identity_xy(door)
        if not self._is_blue_xy_at_valid_wall_position(xy):
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        confidence = float(door.confidence)
        cluster = self._observed_blue_door_id_to_cluster.get(door.door_id)
        if cluster is None:
            cluster = self._find_observed_blue_cluster(
                xy, merge_dist=self._observed_candidate_merge_dist())
        else:
            old_x = float(cluster.get('x', xy[0]))
            old_y = float(cluster.get('y', xy[1]))
            jump = math.hypot(xy[0] - old_x, xy[1] - old_y)
            if (self._observed_blue_same_id_max_jump_m > 0.0
                    and jump > self._observed_blue_same_id_max_jump_m):
                self.get_logger().debug(
                    f'파란 문 후보 {door.door_id} 위치 점프 {jump:.1f}m. '
                    '같은 물리 문 평균 갱신 대신 새 관찰로 분리합니다.')
                cluster = self._find_observed_blue_cluster(
                    xy, merge_dist=self._observed_candidate_merge_dist())
        if cluster is not None:
            old_x = float(cluster.get('x', xy[0]))
            old_y = float(cluster.get('y', xy[1]))
            old_count = int(cluster.get('count', 1.0))
            new_count = min(old_count + 1, 999)
            alpha = 1.0 / min(new_count, 5)
            cluster['x'] = (1.0 - alpha) * old_x + alpha * xy[0]
            cluster['y'] = (1.0 - alpha) * old_y + alpha * xy[1]
            cluster['count'] = float(new_count)
            cluster['last_seen'] = now_sec
            cluster['confidence'] = max(
                float(cluster.get('confidence', 0.0)), confidence)
            self._observed_blue_door_id_to_cluster[door.door_id] = cluster
            return

        cluster = {
            'x': xy[0],
            'y': xy[1],
            'count': 1.0,
            'last_seen': now_sec,
            'confidence': confidence,
            'failures': 0.0,
            'approach_failures': 0.0,
            'open_failures': 0.0,
        }
        self._observed_blue_doors.append(cluster)
        self._observed_blue_door_id_to_cluster[door.door_id] = cluster

    def _observed_candidate_merge_dist(self) -> float:
        merge_dist = self._observed_physical_door_merge_dist_m
        if self._observed_blue_same_id_max_jump_m > 0.0:
            merge_dist = min(merge_dist, self._observed_blue_same_id_max_jump_m)
        return merge_dist

    def _find_observed_blue_cluster(
            self, xy: tuple[float, float],
            merge_dist: float | None = None) -> dict[str, float] | None:
        dist_limit = (self._observed_physical_door_merge_dist_m
                      if merge_dist is None else merge_dist)
        for cluster in self._observed_blue_doors:
            if math.hypot(
                    xy[0] - float(cluster.get('x', 0.0)),
                    xy[1] - float(cluster.get('y', 0.0))) <= dist_limit:
                return cluster
        return None

    def _stable_observed_blue_doors(self) -> list[dict[str, float]]:
        return [
            cluster for cluster in self._observed_blue_doors
            if int(cluster.get('count', 0.0)) >= self._observed_blue_min_observations
            and not self._is_observed_blue_cluster_suppressed(cluster)
        ]

    def _is_blue_position_opened(self, xy: tuple[float, float]) -> bool:
        return any(
            math.hypot(xy[0] - old[0], xy[1] - old[1])
            <= self._opened_physical_door_merge_dist_m
            for old in self._opened_blue_door_positions
        )

    def _is_door_physically_opened(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        return self._is_blue_position_opened(self._door_identity_xy(door))

    def _is_observed_blue_cluster_suppressed(
            self, cluster: dict[str, float]) -> bool:
        if cluster.get('abandoned', 0.0) >= 1.0:
            return True
        if self._maybe_abandon_stale_observed_blue_cluster(cluster):
            return True
        if self._is_blue_xy_near_detected_exit(self._cluster_xy(cluster)):
            return True
        if self._is_blue_xy_opposite_opened_station(self._cluster_xy(cluster)):
            return True
        if self._is_blue_xy_near_observed_red(self._cluster_xy(cluster)):
            return True
        approach_failures = int(
            cluster.get('approach_failures', cluster.get('failures', 0.0)))
        open_failures = int(cluster.get('open_failures', 0.0))
        return (
            approach_failures >= self._max_door_approach_failures_before_abandon
            or open_failures >= self._max_door_open_failures_before_abandon
        )

    def _is_door_observation_suppressed(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        return (
            cluster is not None
            and self._is_observed_blue_cluster_suppressed(cluster)
        )

    def _is_blue_door_observation_confirmed(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        return int(cluster.get('count', 0.0)) >= self._observed_blue_min_observations

    def _is_door_recent_observation(self, door: DoorInfo) -> bool:
        if self._detected_door_max_age_sec <= 0.0:
            return True
        try:
            stamp = Time.from_msg(door.header.stamp)
        except Exception:
            return True
        if stamp.nanoseconds <= 0:
            return True
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        return age <= self._detected_door_max_age_sec

    def _is_blue_candidate_near_recent_red(self, door: DoorInfo) -> bool:
        if self._blue_red_conflict_dist_m <= 0.0:
            return False
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        blue_xy = self._door_identity_xy(door)
        for red in self.detected_doors:
            if red.door_color != 'red':
                continue
            if not self._is_door_recent_observation(red):
                continue
            if not self._door_has_map_identity(red):
                continue
            red_xy = self._door_identity_xy(red)
            if math.hypot(
                    blue_xy[0] - red_xy[0],
                    blue_xy[1] - red_xy[1]) <= self._blue_red_conflict_dist_m:
                return True
        return self._is_blue_xy_near_observed_red(blue_xy, door.door_id)

    def _cluster_xy(self, cluster: dict[str, float]) -> tuple[float, float]:
        return (
            float(cluster.get('x', 0.0)),
            float(cluster.get('y', 0.0)),
        )

    def _is_blue_xy_near_observed_red(
            self, blue_xy: tuple[float, float],
            candidate_id: str = '') -> bool:
        conflict_dist = self._blue_red_conflict_dist_m
        if self._valid_detected_exit() or self._final_scan_start_time is not None:
            conflict_dist = max(
                conflict_dist,
                self._blue_red_conflict_after_exit_dist_m)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        for red in self._observed_red_door_positions:
            if self._red_blue_conflict_memory_sec > 0.0:
                age = now_sec - float(red.get('last_seen', now_sec))
                if age > self._red_blue_conflict_memory_sec:
                    continue
            red_x = float(red.get('x', 0.0))
            red_y = float(red.get('y', 0.0))
            if math.hypot(blue_xy[0] - red_x, blue_xy[1] - red_y) <= conflict_dist:
                label = candidate_id or f'cluster@({blue_xy[0]:.1f}, {blue_xy[1]:.1f})'
                self.get_logger().info(
                    f'파란 문 후보 {label} 제외: 최근 관찰된 빨간 문 위치 '
                    f'({red_x:.1f}, {red_y:.1f})와 겹칩니다.',
                    once=True)
                return True
        return False

    def _is_blue_door_at_valid_wall_position(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        return self._is_blue_xy_at_valid_wall_position(self._door_identity_xy(door))

    def _is_blue_xy_at_valid_wall_position(self, xy: tuple[float, float]) -> bool:
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False
        if self._is_blue_xy_near_detected_exit(xy):
            return False
        if self._is_blue_xy_opposite_opened_station(xy):
            return False
        return True

    def _is_xy_at_configured_wall_lateral(self, xy: tuple[float, float]) -> bool:
        abs_y = abs(self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        if (self._observed_blue_min_abs_wall_y_m > 0.0
                and abs_y < self._observed_blue_min_abs_wall_y_m):
            return False
        if (self._observed_blue_max_abs_wall_y_m > 0.0
                and abs_y > self._observed_blue_max_abs_wall_y_m):
            return False
        return True

    def _is_blue_xy_near_detected_exit(self, xy: tuple[float, float]) -> bool:
        margin = max(0.0, self._blue_exit_suppression_margin_m)
        if margin <= 0.0 or not self._valid_detected_exit():
            return False
        exit_xy = self._exit_identity_xy()
        if exit_xy is None:
            return False
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        if candidate_progress < exit_progress - margin:
            return False
        self.get_logger().info(
            f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
            f'관측된 비상구 진행축 {margin:.1f}m 이내/이후 영역입니다.',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_xy_opposite_opened_station(self, xy: tuple[float, float]) -> bool:
        window = max(0.0, self._opened_station_blue_suppression_progress_m)
        if window <= 0.0:
            return False
        if not self._opened_blue_door_positions:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False

        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        robot_progress = self._axis_progress_xy(pose[0], pose[1])
        if robot_progress - candidate_progress < self._observed_blue_stale_behind_m:
            return False

        for opened_xy in self._opened_blue_door_positions:
            opened_progress = self._axis_progress_xy(opened_xy[0], opened_xy[1])
            if abs(candidate_progress - opened_progress) > window:
                continue
            self.get_logger().info(
                f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
                f'이미 연 문 진행축 {opened_progress:.1f}m 근처의 재관측 후보입니다.',
                throttle_duration_sec=3.0)
            return True
        return False

    def _maybe_abandon_stale_observed_blue_cluster(
            self, cluster: dict[str, float]) -> bool:
        if (self._observed_blue_stale_abandon_sec <= 0.0
                or self._observed_blue_stale_behind_m <= 0.0):
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False
        robot_progress = self._axis_progress_xy(pose[0], pose[1])
        cluster_progress = self._cluster_progress(cluster)
        behind = robot_progress - cluster_progress
        if behind < self._observed_blue_stale_behind_m:
            return False

        now_sec = self.get_clock().now().nanoseconds / 1e9
        age = now_sec - float(cluster.get('last_seen', now_sec))
        if age < self._observed_blue_stale_abandon_sec:
            return False

        count = int(cluster.get('count', 0.0))
        approach_failures = int(
            cluster.get('approach_failures', cluster.get('failures', 0.0)))
        open_failures = int(cluster.get('open_failures', 0.0))
        weak_observation = count < max(self._observed_blue_min_observations + 2, 4)
        previously_failed = approach_failures > 0 or open_failures > 0
        far_behind = behind >= self._observed_blue_stale_behind_m * 2.0
        if not (previously_failed or (weak_observation and far_behind)):
            return False

        self._mark_observed_blue_cluster_abandoned(
            cluster,
            f'최근 {age:.0f}s 동안 재관찰되지 않았고 로봇보다 {behind:.1f}m 뒤쪽입니다')
        return True

    def _observed_unopened_blue_doors(self) -> list[dict[str, float]]:
        return [
            cluster for cluster in self._stable_observed_blue_doors()
            if not self._is_blue_position_opened((
                float(cluster.get('x', 0.0)),
                float(cluster.get('y', 0.0))))
        ]

    def _stable_unopened_detected_blue_doors(
            self, doors: list[DoorInfo]) -> list[DoorInfo]:
        unopened = self._observed_unopened_blue_doors()
        if not unopened:
            return []

        revisit: list[DoorInfo] = []
        merge_dist = max(
            self._observed_physical_door_merge_dist_m,
            self._opened_physical_door_merge_dist_m)
        for door in doors:
            if door.door_color != 'blue' or not self._door_has_map_identity(door):
                continue
            xy = self._door_identity_xy(door)
            if any(
                    math.hypot(
                        xy[0] - float(cluster.get('x', 0.0)),
                        xy[1] - float(cluster.get('y', 0.0))) <= merge_dist
                    for cluster in unopened):
                revisit.append(door)
        return revisit

    def _hold_exit_for_unopened_blue_doors(
            self, unopened: list[dict[str, float]]):
        preview = ', '.join(
            f'({cluster.get("x", 0.0):.1f},{cluster.get("y", 0.0):.1f})/'
            f'{int(cluster.get("count", 0.0))}obs'
            for cluster in unopened[:4])
        if len(unopened) > 4:
            preview += ', ...'
        self.get_logger().warn(
            f'비상구 이동 보류: 관찰됐지만 아직 안 열린 파란 문 '
            f'{len(unopened)}개가 남아 있습니다 [{preview}]. '
            '문 후보 재스캔/재계획을 계속합니다.')
        self._explore_start_time = self.get_clock().now()

    def _record_observed_blue_door_failure(
            self, door: DoorInfo, failure_kind: str) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        cluster = self._find_observed_blue_cluster(
            xy,
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            self._record_observed_physical_blue_door(door)
            cluster = self._find_observed_blue_cluster(xy)
        if cluster is None:
            return False

        if failure_kind == 'open':
            field = 'open_failures'
            limit = self._max_door_open_failures_before_abandon
            label = '문 개방'
        else:
            field = 'approach_failures'
            limit = self._max_door_approach_failures_before_abandon
            label = '문 앞 접근'

        failures = int(cluster.get(field, 0.0)) + 1
        cluster[field] = float(failures)
        if field == 'approach_failures':
            cluster['failures'] = float(failures)
        cx = float(cluster.get('x', xy[0]))
        cy = float(cluster.get('y', xy[1]))
        if failures >= limit:
            cluster['abandoned'] = 1.0
            self._mark_door_abandoned(door)
            self.get_logger().warn(
                f'파란 문 후보 ({cx:.1f}, {cy:.1f}) {label} 실패 '
                f'{failures}/{limit}. 접근/개방 불가 후보로 포기하고 '
                '출구 판단을 막지 않도록 처리합니다.')
            return True

        self.get_logger().warn(
            f'파란 문 후보 ({cx:.1f}, {cy:.1f}) {label} 실패 '
            f'{failures}/{limit}. 아직 포기하지 않고 다음 탐색에서 재시도합니다.')
        return False

    def _clear_observed_blue_door_failures(self, door: DoorInfo):
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return
        cluster = self._find_observed_blue_cluster(self._door_identity_xy(door))
        if cluster is None:
            return
        cluster['approach_failures'] = 0.0
        cluster['open_failures'] = 0.0
        cluster['failures'] = 0.0

    def _mark_door_abandoned(self, door: DoorInfo):
        self._abandoned_door_ids.add(door.door_id)
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return
        xy = self._door_identity_xy(door)
        if any(
                math.hypot(xy[0] - old[0], xy[1] - old[1])
                <= self._opened_physical_door_merge_dist_m
                for old in self._abandoned_blue_door_positions):
            return
        self._abandoned_blue_door_positions.append(xy)

    def _mark_observed_blue_cluster_abandoned(
            self, cluster: dict[str, float], reason: str):
        if cluster.get('abandoned', 0.0) >= 1.0:
            return
        cluster['abandoned'] = 1.0
        xy = (
            float(cluster.get('x', 0.0)),
            float(cluster.get('y', 0.0)),
        )
        if not any(
                math.hypot(xy[0] - old[0], xy[1] - old[1])
                <= self._opened_physical_door_merge_dist_m
                for old in self._abandoned_blue_door_positions):
            self._abandoned_blue_door_positions.append(xy)
        self.get_logger().warn(
            f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 포기: {reason}. '
            '불안정/실패 후보로 보고 다음 관찰을 계속합니다.')

    def _is_door_physically_abandoned(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        return any(
            math.hypot(xy[0] - old[0], xy[1] - old[1])
            <= self._opened_physical_door_merge_dist_m
            for old in self._abandoned_blue_door_positions
        )

    def _handle_door_open_failure(self, message: str):
        if self.target_door is None:
            self.get_logger().error(f'문 개방 실패: {message}. target_door 없음.')
            self._transition(State.EMERGENCY_STOP)
            return
        if self.target_door.door_color != 'blue':
            self.get_logger().error(f'문 개방 실패: {message}')
            self._transition(State.EMERGENCY_STOP)
            return

        abandoned = self._record_observed_blue_door_failure(
            self.target_door, failure_kind='open')
        if abandoned:
            abandoned_id = self.target_door.door_id
            self._failed_door_ids.add(abandoned_id)
            self.target_door = None
            self._nav_retry_count = 0
            self.get_logger().error(
                f'문 개방 포기: {abandoned_id}. 제한 횟수까지 개방에 실패해 '
                '해당 문 후보를 제외하고 다음 파란 문 탐색을 계속합니다.')
            self._transition(State.EXPLORING)
            return

        self.get_logger().warn('문 개방을 같은 위치에서 다시 시도합니다.')

    def _record_opened_physical_blue_door(self, door: DoorInfo):
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return
        xy = self._door_identity_xy(door)
        if any(
                math.hypot(xy[0] - old[0], xy[1] - old[1])
                <= self._opened_physical_door_merge_dist_m
                for old in self._opened_blue_door_positions):
            return
        self._opened_blue_door_positions.append(xy)

    def _door_with_axis_aligned_approach(self, door: DoorInfo) -> DoorInfo:
        adjusted = copy.deepcopy(door)
        if (not self._axis_door_approach_enabled
                or adjusted.door_color != 'blue'
                or not self._door_has_map_identity(adjusted)):
            return adjusted

        handle_x, handle_y = self._door_handle_xy(adjusted)
        handle_progress = self._axis_progress_xy(handle_x, handle_y)
        handle_lateral = self._axis_lateral_xy(handle_x, handle_y)
        lateral_from_center = handle_lateral - self._explore_center_y
        if abs(lateral_from_center) < self._axis_door_min_abs_lateral_m:
            return adjusted

        side = 1.0 if lateral_from_center > 0.0 else -1.0
        standoff = max(0.10, self._axis_door_side_standoff_m)
        approach_lateral = handle_lateral - side * standoff

        max_abs_lateral = max(
            self._axis_door_min_abs_lateral_m,
            self._nav_start_max_abs_y_m,
        )
        rel_lateral = approach_lateral - self._explore_center_y
        if rel_lateral * side < 0.0:
            rel_lateral = 0.0
        if abs(rel_lateral) > max_abs_lateral:
            rel_lateral = side * max_abs_lateral

        approach_lateral = self._explore_center_y + rel_lateral
        goal_x, goal_y = self._axis_to_map_xy(
            handle_progress, approach_lateral)
        adjusted.door_pose.header.frame_id = 'map'
        adjusted.door_pose.pose.position.x = goal_x
        adjusted.door_pose.pose.position.y = goal_y
        adjusted.door_pose.pose.position.z = 0.0
        self._set_pose_yaw(
            adjusted.door_pose.pose,
            self._mission_forward_yaw() + side * math.pi / 2.0,
        )
        self.get_logger().info(
            f'문 접근 목표 보정: {adjusted.door_id} '
            f'handle_axis=({handle_progress:.2f}, {handle_lateral:.2f}) '
            f'goal_axis=({handle_progress:.2f}, {approach_lateral:.2f}) '
            f'yaw={math.degrees(self._mission_forward_yaw() + side * math.pi / 2.0):.0f}deg')
        return adjusted

    def _set_pose_yaw(self, pose, yaw: float):
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)

    def _nearest_forward_blue_door(self, doors: list[DoorInfo]) -> DoorInfo | None:
        if not doors:
            return None
        pose = self._current_map_pose()
        if pose is None:
            return min(doors, key=lambda d: self._door_progress(d))
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

    def _pose_yaw(self, pose) -> float:
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _door_open_pose_error(self, door: DoorInfo):
        if door.door_pose.header.frame_id != 'map':
            return None
        pose = self._current_map_pose()
        if pose is None:
            return None

        rx, ry, robot_yaw = pose
        gx = door.door_pose.pose.position.x
        gy = door.door_pose.pose.position.y
        goal_yaw = self._pose_yaw(door.door_pose.pose)
        dist = math.hypot(gx - rx, gy - ry)
        yaw_error = self._normalize_angle(goal_yaw - robot_yaw)
        return dist, yaw_error

    def _prepare_door_opening_pose(self, door: DoorInfo) -> str:
        error = self._door_open_pose_error(door)
        if error is None:
            return 'ready'
        dist, yaw_error = error
        yaw_abs = abs(yaw_error)

        if dist > self._door_open_ready_max_dist_m:
            self.get_logger().warn(
                f'문 개방 자세 미달: {door.door_id} '
                f'dist={dist:.2f}m/{self._door_open_ready_max_dist_m:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg/'
                f'{math.degrees(self._door_open_ready_yaw_tolerance):.0f}deg')
            self._door_open_align_start_time = None
            return 'retry'

        if yaw_abs <= self._door_open_ready_yaw_tolerance:
            self._door_open_align_start_time = None
            return 'ready'

        now = self.get_clock().now()
        if self._door_open_align_start_time is None:
            self._door_open_align_start_time = now
            self.get_logger().info(
                f'문 앞 위치 도달. 문을 바라보도록 제자리 회전 정렬 시작: '
                f'{door.door_id}, yaw_error={math.degrees(yaw_abs):.0f}deg')

        elapsed = (now - self._door_open_align_start_time).nanoseconds / 1e9
        if elapsed > self._door_open_align_timeout_sec:
            self.get_logger().warn(
                f'문 앞 회전 정렬 timeout: {door.door_id}, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg')
            self._door_open_align_start_time = None
            return 'retry'

        twist = Twist()
        cmd = self._door_open_align_kp * yaw_error
        limit = abs(self._door_open_align_angular_vel)
        twist.angular.z = max(-limit, min(limit, cmd))
        if abs(twist.angular.z) < 0.10:
            twist.angular.z = 0.10 if yaw_error > 0.0 else -0.10
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'문 앞 회전 정렬 중: {door.door_id}, '
            f'yaw_error={math.degrees(yaw_abs):.0f}deg, cmd={twist.angular.z:.2f}')
        return 'waiting'

    def _retry_precise_door_approach(self) -> bool:
        if self.target_door is None:
            return False
        self._nav_retry_count += 1
        if self._nav_retry_count > self._max_nav_retries:
            return False

        self.get_logger().info(
            f'문 앞 정렬 재접근: {self.target_door.door_id} '
            f'({self._nav_retry_count}/{self._max_nav_retries})')
        self._door_open_settle_start_time = None
        self._nav_done = False
        self._nav_failed = False
        self._nav_start_time = self.get_clock().now()
        self.target_door = self._door_with_axis_aligned_approach(self.target_door)
        self.target_door_pub.publish(self.target_door)
        return True

    def _settle_before_opening_door(self) -> bool:
        if self._door_open_settle_sec <= 0.0:
            return False
        now = self.get_clock().now()
        if self._door_open_settle_start_time is None:
            self._door_open_settle_start_time = now
            self.get_logger().info(
                f'문 앞 정렬 완료. {self._door_open_settle_sec:.1f}s 정지 후 개방합니다.')
        self.cmd_vel_pub.publish(Twist())
        elapsed = (now - self._door_open_settle_start_time).nanoseconds / 1e9
        if elapsed < self._door_open_settle_sec:
            return True
        self._door_open_settle_start_time = None
        return False

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

    def _remember_failed_explore_goal(self):
        if self._explore_nav_goal_progress <= 0.0:
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self._recent_failed_explore_goals.append((
            float(self._explore_nav_goal_sign),
            float(self._explore_nav_goal_progress),
            now_sec,
        ))
        self._prune_failed_explore_goals(now_sec)

    def _prune_failed_explore_goals(self, now_sec: float):
        memory = max(0.0, self._explore_nav_failed_goal_memory_sec)
        if memory <= 0.0:
            self._recent_failed_explore_goals.clear()
            return
        self._recent_failed_explore_goals = [
            item for item in self._recent_failed_explore_goals
            if now_sec - item[2] <= memory
        ]

    def _is_explore_lane_recently_failed(self, sign: float, progress: float) -> bool:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self._prune_failed_explore_goals(now_sec)
        window = max(0.4, self._explore_nav_failed_goal_progress_window_m)
        return any(s == sign and abs(p - progress) <= window
                   for s, p, _ in self._recent_failed_explore_goals)

    def _select_explore_goal_y(self, robot_x: float, robot_y: float, robot_yaw: float,
                               current_lateral: float,
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
        current_progress = self._axis_progress_xy(robot_x, robot_y)
        for goal_y, sign in candidates:
            clearance = self._lane_scan_clearance(robot_x, robot_y, robot_yaw, goal_y)
            score = clearance
            lane_blocked = self._is_explore_lane_recently_failed(sign, current_progress)
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
            score -= abs(goal_y - current_lateral) * 0.06
            if clearance < self._explore_obstacle_stop_m:
                score -= 2.0
            elif clearance < self._explore_obstacle_slow_m:
                score -= 0.55
            if (blue_lane_active and sign == preferred_sign
                    and clearance < self._explore_nav_blue_lane_min_clearance_m):
                score -= 1.75
            if lane_blocked:
                score -= 8.0

            label = 'C' if sign == 0.0 else ('L' if sign > 0.0 else 'R')
            blocked_text = '/recent_fail' if lane_blocked else ''
            scored_text.append(f'{label}:y={goal_y:.2f}/clear={clearance:.2f}/score={score:.2f}{blocked_text}')
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

        corridor_axis = self._mission_forward_yaw()
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
        corridor_axis = self._mission_forward_yaw()
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
            self._door_open_settle_start_time = None
            self._door_open_align_start_time = None
        elif new_state == State.EXPLORING_NAVIGATING:
            self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
        elif new_state == State.EXITING:
            if self._send_exit_goal():
                self._nav_start_time = self.get_clock().now()
            else:
                self.state = State.EXPLORING
                self._nav_start_time = None
                if self._explore_start_time is None:
                    self._explore_start_time = self.get_clock().now()
        elif new_state == State.EXPLORING:
            self._exit_crossing_start_time = None
            self.cmd_vel_pub.publish(Twist())
            # 새 파란 문이 없을 때는 waypoint 왕복 중에도 타임아웃이 누적되어야 한다.
            if self._explore_start_time is None:
                self._explore_start_time = self.get_clock().now()
            self._nav_start_time = None
            self._final_scan_start_time = None
            self._pre_nav_scan_start_time = None
            self._pending_nav_scan_target_id = None
            self._door_open_settle_start_time = None
            self._door_open_align_start_time = None
            self._explore_nav_goal_id = ''
        elif new_state == State.MISSION_COMPLETE:
            self._exit_crossing_start_time = None
            self.cmd_vel_pub.publish(Twist())
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
