import copy
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from enum import IntEnum
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from fire_robot_interfaces.msg import DoorInfo, FireInfo, RobotState
from fire_robot_interfaces.srv import OpenDoor
from fire_robot_fsm.fsm_subsystems import DoorApproachSubFsm, DoorTargetSubFsm


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
        self.declare_parameter('allow_near_backtracking_blue_revisit', False)
        self.declare_parameter('mission_start_x', 0.0)
        self.declare_parameter('next_door_min_forward_m', 0.75)
        self.declare_parameter('target_door_min_robot_forward_m', 0.35)
        self.declare_parameter('blue_target_max_robot_ahead_m', 0.0)
        self.declare_parameter('side_wall_door_progress_bias_m', 0.0)
        self.declare_parameter('opened_door_merge_dist_m', 1.0)
        self.declare_parameter('min_opened_doors_before_exit', 0)
        self.declare_parameter('opened_physical_door_merge_dist_m', 1.2)
        self.declare_parameter('observed_physical_door_merge_dist_m', 1.2)
        self.declare_parameter('observed_blue_min_observations', 2)
        self.declare_parameter('observed_blue_target_min_confidence', 0.50)
        self.declare_parameter('observed_blue_low_conf_target_min_observations', 6)
        self.declare_parameter('observed_blue_stable_open_min_confidence', 0.72)
        self.declare_parameter('observed_blue_fresh_evidence_max_progress_gap_m', 0.95)
        self.declare_parameter('observed_blue_fresh_evidence_max_lateral_gap_m', 0.85)
        self.declare_parameter('semantic_door_memory_enabled', True)
        self.declare_parameter('semantic_door_merge_dist_m', 1.10)
        self.declare_parameter('semantic_same_wall_merge_progress_m', 1.80)
        self.declare_parameter('semantic_same_wall_merge_lateral_m', 1.15)
        self.declare_parameter('semantic_blue_min_observations', 2)
        self.declare_parameter('semantic_blue_min_ratio', 0.58)
        self.declare_parameter('semantic_exit_tail_blue_min_observations', 4)
        self.declare_parameter('semantic_exit_min_green_observations', 3)
        self.declare_parameter('semantic_exit_min_green_ratio', 0.50)
        self.declare_parameter('blue_target_immediate_min_confidence', 0.50)
        self.declare_parameter('exit_interrupt_blue_min_confidence', 0.55)
        self.declare_parameter('use_observed_blue_clusters_as_targets', False)
        self.declare_parameter('max_door_approach_failures_before_abandon', 4)
        self.declare_parameter('pre_exit_blue_approach_failures_before_abandon', 1)
        self.declare_parameter('detected_door_max_age_sec', 8.0)
        self.declare_parameter('observed_blue_same_id_max_jump_m', 1.8)
        self.declare_parameter('observed_blue_same_id_max_total_drift_m', 1.5)
        self.declare_parameter('observed_blue_stale_abandon_sec', 45.0)
        self.declare_parameter('observed_blue_stale_behind_m', 2.0)
        self.declare_parameter('observed_blue_target_max_behind_m', 0.80)
        self.declare_parameter('observed_blue_same_side_after_opened_min_progress_m', 0.0)
        self.declare_parameter('blue_same_side_after_opened_min_progress_m', 0.0)
        self.declare_parameter('blue_red_conflict_dist_m', 0.75)
        self.declare_parameter('blue_red_conflict_strict_dist_m', 0.45)
        self.declare_parameter('observed_blue_min_abs_wall_y_m', 0.0)
        self.declare_parameter('observed_blue_max_abs_wall_y_m', 0.0)
        self.declare_parameter('blue_handle_max_wall_overshoot_m', 0.0)
        self.declare_parameter('blue_handle_validation_overshoot_m', 0.0)
        self.declare_parameter('red_blue_conflict_memory_sec', 240.0)
        self.declare_parameter('blue_red_conflict_after_exit_dist_m', 0.0)
        self.declare_parameter('red_observation_min_confidence', 0.0)
        self.declare_parameter('red_observation_min_count_for_blue_suppression', 1)
        self.declare_parameter('confirmed_blue_bypass_red_conflict_count', 0)
        self.declare_parameter('confirmed_blue_bypass_red_conflict_confidence', 1.0)
        self.declare_parameter('opened_station_blue_suppression_progress_m', 0.0)
        self.declare_parameter('opened_station_same_side_blue_suppression_progress_m', 0.0)
        self.declare_parameter('opened_station_opposite_side_blue_suppression_progress_m', 0.0)
        self.declare_parameter('opened_station_same_id_blue_suppression_progress_m', 0.0)
        self.declare_parameter('abandoned_station_same_side_blue_suppression_progress_m', 0.0)
        self.declare_parameter('abandoned_station_opposite_side_blue_suppression_progress_m', 0.0)
        self.declare_parameter('max_door_open_failures_before_abandon', 2)
        self.declare_parameter('door_open_requires_fresh_blue', True)
        self.declare_parameter('door_open_fresh_blue_max_age_sec', 6.0)
        self.declare_parameter('door_open_fresh_blue_max_dist_m', 1.10)
        self.declare_parameter('door_open_fresh_blue_min_confidence', 0.18)
        self.declare_parameter('complete_when_past_exit_x', False)
        self.declare_parameter('exit_complete_margin_m', 0.50)
        self.declare_parameter('final_scan_before_exit_sec', 0.0)
        self.declare_parameter('final_scan_angular_vel', 0.45)
        self.declare_parameter('door_open_ready_max_dist_m', 0.24)
        self.declare_parameter('door_open_straight_ready_max_dist_m', 0.28)
        self.declare_parameter('door_open_ready_yaw_tolerance_deg', 8.0)
        self.declare_parameter('door_open_ready_lateral_tolerance_m', 0.10)
        self.declare_parameter('door_open_fine_control_max_dist_m', 0.95)
        self.declare_parameter('door_open_fine_lateral_tolerance_m', 0.60)
        self.declare_parameter('door_open_fine_position_kp', 0.65)
        self.declare_parameter('door_open_fine_linear_vel', 0.10)
        self.declare_parameter('door_open_fine_min_linear_vel', 0.035)
        self.declare_parameter('door_open_fine_allow_reverse', False)
        self.declare_parameter('door_open_axis_min_side_lateral_m', 0.0)
        self.declare_parameter('door_open_settle_sec', 1.0)
        self.declare_parameter('exit_retry_delay_sec', 3.0)
        self.declare_parameter('door_open_align_angular_vel', 0.42)
        self.declare_parameter('door_open_align_kp', 1.2)
        self.declare_parameter('door_open_align_timeout_sec', 34.0)
        self.declare_parameter('post_open_reorient_enabled', True)
        self.declare_parameter('post_open_reorient_yaw_tolerance_deg', 14.0)
        self.declare_parameter('post_open_reorient_angular_vel', 0.45)
        self.declare_parameter('post_open_reorient_timeout_sec', 8.0)
        self.declare_parameter('post_open_clearance_advance_enabled', True)
        self.declare_parameter('post_open_clearance_advance_m', 0.45)
        self.declare_parameter('post_open_clearance_linear_vel', 0.12)
        self.declare_parameter('post_open_clearance_front_min_m', 0.55)
        self.declare_parameter('post_open_clearance_front_angle_deg', 24.0)
        self.declare_parameter('post_open_clearance_timeout_sec', 8.0)
        self.declare_parameter('post_open_clearance_heading_kp', 0.8)
        self.declare_parameter('post_open_clearance_angular_vel_limit', 0.30)
        self.declare_parameter('post_open_side_retreat_enabled', True)
        self.declare_parameter('post_open_side_retreat_m', 1.60)
        self.declare_parameter('post_open_side_retreat_target_abs_y_m', 0.85)
        self.declare_parameter('post_open_side_retreat_linear_vel', 0.16)
        self.declare_parameter('post_open_side_retreat_timeout_sec', 12.0)
        self.declare_parameter('post_open_side_retreat_yaw_tolerance_deg', 35.0)
        self.declare_parameter('pre_nav_scan_sec', 0.0)
        self.declare_parameter('pre_nav_scan_angular_vel', 0.0)
        self.declare_parameter('explore_observation_wait_sec', 0.0)
        self.declare_parameter('explore_observation_wait_angular_vel', 0.0)
        self.declare_parameter('nav_target_republish_sec', 0.0)
        self.declare_parameter('door_nav_stuck_watch_enabled', True)
        self.declare_parameter('door_nav_stuck_timeout_sec', 18.0)
        self.declare_parameter('door_nav_stuck_min_progress_m', 0.08)
        self.declare_parameter('door_nav_stuck_min_goal_dist_m', 0.90)
        self.declare_parameter('explore_waypoint_scan_sec', 0.0)
        self.declare_parameter('explore_waypoint_scan_angular_vel', 0.45)
        self.declare_parameter('explore_interrupt_min_confidence', 0.35)
        self.declare_parameter('explore_interrupt_max_distance_m', 5.5)
        self.declare_parameter('explore_nav_enabled', True)
        self.declare_parameter('explore_nav_step_m', 1.20)
        self.declare_parameter('explore_nav_min_step_m', 0.30)
        self.declare_parameter('explore_nav_reduce_step_clearance_m', 1.35)
        self.declare_parameter('explore_nav_goal_clearance_margin_m', 0.55)
        self.declare_parameter('explore_nav_clearance_deferral_limit', 5)
        self.declare_parameter('explore_nav_planner_probe_step_m', 0.45)
        self.declare_parameter('explore_nav_lane_y_m', 0.75)
        self.declare_parameter('explore_nav_lane_width_m', 1.35)
        self.declare_parameter('explore_nav_scan_lookahead_m', 2.80)
        self.declare_parameter('explore_nav_color_bias_score', 0.35)
        self.declare_parameter('explore_nav_keep_lane_bias_score', 0.12)
        self.declare_parameter('explore_nav_lane_alternate_success_interval', 2)
        self.declare_parameter('explore_nav_front_obstacle_width_m', 1.35)
        self.declare_parameter('explore_nav_front_obstacle_bias_score', 3.0)
        self.declare_parameter('explore_nav_blue_door_bias_score', 2.2)
        self.declare_parameter('explore_nav_blue_lane_min_clearance_m', 0.80)
        self.declare_parameter('explore_nav_blue_lane_bias_max_dist_m', 3.0)
        self.declare_parameter('explore_nav_front_block_scan_clearance_m', 0.0)
        self.declare_parameter('explore_nav_front_block_scan_after_opened_m', 2.0)
        self.declare_parameter('max_explore_past_last_opened_blue_m', 0.0)
        self.declare_parameter('abandoned_blue_extends_explore_limit', False)
        self.declare_parameter('blue_target_max_beyond_explore_limit_m', 0.75)
        self.declare_parameter('observed_exit_explore_grace_m', 1.25)
        self.declare_parameter('observed_exit_blue_candidate_margin_m', 0.35)
        self.declare_parameter('explore_nav_recenter_abs_y_m', 0.95)
        self.declare_parameter('explore_nav_max_lateral_step_m', 0.45)
        self.declare_parameter('max_explore_nav_failures_before_exit', 8)
        self.declare_parameter('explore_nav_failed_goal_memory_sec', 24.0)
        self.declare_parameter('explore_nav_failed_goal_progress_window_m', 2.4)
        self.declare_parameter('explore_nav_timeout_sec', 45.0)
        self.declare_parameter('explore_nav_stuck_wall_timeout_sec', 10.0)
        self.declare_parameter('explore_nav_stuck_min_progress_m', 0.12)
        self.declare_parameter('min_target_door_abs_y_m', 0.0)
        self.declare_parameter('axis_door_approach_enabled', True)
        self.declare_parameter('axis_door_side_standoff_m', 0.85)
        self.declare_parameter('axis_door_min_side_goal_lateral_m', 0.35)
        self.declare_parameter('axis_door_min_abs_lateral_m', 0.45)
        self.declare_parameter('axis_door_max_handle_pose_progress_delta_m', 1.6)
        self.declare_parameter('axis_door_use_detected_pose_lateral', False)
        self.declare_parameter('locked_target_refine_enabled', False)
        self.declare_parameter('locked_target_refine_min_shift_m', 0.18)
        self.declare_parameter('locked_target_refine_max_progress_jump_m', 1.10)
        self.declare_parameter('locked_target_refine_max_total_progress_drift_m', 0.80)
        self.declare_parameter('locked_target_refine_min_period_sec', 1.0)
        self.declare_parameter('target_door_direct_nav_max_dist_m', 0.0)
        self.declare_parameter('nav_start_max_abs_y_m', 0.95)
        self.declare_parameter('nav_start_max_heading_error_deg', 85.0)
        self.declare_parameter('nav_start_center_recovery_enabled', True)
        self.declare_parameter('nav_start_center_recovery_target_abs_y_m', 0.72)
        self.declare_parameter('nav_start_center_recovery_linear_vel', 0.13)
        self.declare_parameter('nav_start_center_recovery_angular_vel', 0.40)
        self.declare_parameter('nav_start_center_recovery_yaw_tolerance_deg', 32.0)
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
        self.declare_parameter('no_blue_exit_min_robot_forward_m', 3.0)
        self.declare_parameter('detected_exit_max_center_y_m', 0.0)
        self.declare_parameter('detected_exit_merge_dist_m', 2.5)
        self.declare_parameter('detected_exit_min_after_opened_blue_m', 0.0)
        self.declare_parameter('detected_exit_min_robot_past_opened_blue_m', 0.0)
        self.declare_parameter('detected_exit_require_front_wall_confirmation', False)
        self.declare_parameter('detected_exit_front_wall_tolerance_m', 1.25)
        self.declare_parameter('blue_exit_suppression_margin_m', 0.0)
        self.declare_parameter('pre_exit_blue_requires_fresh_detection', False)
        self.declare_parameter('pre_exit_blue_suppression_margin_m', 0.0)
        self.declare_parameter('pre_exit_blue_min_confidence', 0.45)
        self.declare_parameter('pre_exit_blue_fresh_max_age_sec', 4.0)
        self.declare_parameter('exit_pass_through_m', 0.8)
        self.declare_parameter('exit_nav_standoff_m', 0.8)
        self.declare_parameter('exit_visible_nav_max_dist_m', 0.0)
        self.declare_parameter('exit_direct_cross_max_dist_m', 2.6)
        self.declare_parameter('exit_cross_linear_vel', 0.22)
        self.declare_parameter('exit_cross_heading_kp', 1.2)
        self.declare_parameter('exit_cross_angular_vel_limit', 0.45)
        self.declare_parameter('exit_cross_timeout_sec', 6.0)
        self.declare_parameter('exit_cross_rotate_in_place_threshold_deg', 75.0)
        self.declare_parameter('exit_interrupt_blue_before_exit_margin_m', 0.35)
        self.declare_parameter('allow_front_wall_exit_fallback', False)
        self.declare_parameter('front_wall_exit_min_range_m', 0.45)
        self.declare_parameter('front_wall_exit_max_range_m', 2.4)
        self.declare_parameter('front_wall_exit_half_angle_deg', 35.0)
        self.declare_parameter('front_wall_exit_min_coverage', 0.45)
        self.declare_parameter('front_wall_exit_max_spread_m', 1.2)
        self.declare_parameter('front_wall_exit_min_after_opened_blue_m', 0.0)

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
        self._allow_near_backtracking_blue_revisit = bool(
            self.get_parameter('allow_near_backtracking_blue_revisit').value)
        self._mission_start_x = float(self.get_parameter('mission_start_x').value)
        self._next_door_min_forward_m = float(self.get_parameter('next_door_min_forward_m').value)
        self._target_door_min_robot_forward_m = float(
            self.get_parameter('target_door_min_robot_forward_m').value)
        self._blue_target_max_robot_ahead_m = float(
            self.get_parameter('blue_target_max_robot_ahead_m').value)
        self._side_wall_door_progress_bias_m = float(
            self.get_parameter('side_wall_door_progress_bias_m').value)
        self._opened_door_merge_dist_m = float(self.get_parameter('opened_door_merge_dist_m').value)
        self._min_opened_doors_before_exit = int(
            self.get_parameter('min_opened_doors_before_exit').value)
        self._opened_physical_door_merge_dist_m = float(
            self.get_parameter('opened_physical_door_merge_dist_m').value)
        self._observed_physical_door_merge_dist_m = float(
            self.get_parameter('observed_physical_door_merge_dist_m').value)
        self._observed_blue_min_observations = max(1, int(
            self.get_parameter('observed_blue_min_observations').value))
        self._observed_blue_target_min_confidence = float(
            self.get_parameter('observed_blue_target_min_confidence').value)
        self._observed_blue_low_conf_target_min_observations = max(
            self._observed_blue_min_observations,
            int(self.get_parameter(
                'observed_blue_low_conf_target_min_observations').value))
        self._observed_blue_stable_open_min_confidence = float(
            self.get_parameter('observed_blue_stable_open_min_confidence').value)
        self._observed_blue_fresh_evidence_max_progress_gap_m = float(
            self.get_parameter(
                'observed_blue_fresh_evidence_max_progress_gap_m').value)
        self._observed_blue_fresh_evidence_max_lateral_gap_m = float(
            self.get_parameter(
                'observed_blue_fresh_evidence_max_lateral_gap_m').value)
        self._semantic_door_memory_enabled = bool(
            self.get_parameter('semantic_door_memory_enabled').value)
        self._semantic_door_merge_dist_m = float(
            self.get_parameter('semantic_door_merge_dist_m').value)
        self._semantic_same_wall_merge_progress_m = float(
            self.get_parameter('semantic_same_wall_merge_progress_m').value)
        self._semantic_same_wall_merge_lateral_m = float(
            self.get_parameter('semantic_same_wall_merge_lateral_m').value)
        self._semantic_blue_min_observations = max(1, int(
            self.get_parameter('semantic_blue_min_observations').value))
        self._semantic_blue_min_ratio = float(
            self.get_parameter('semantic_blue_min_ratio').value)
        self._semantic_exit_tail_blue_min_observations = max(1, int(
            self.get_parameter('semantic_exit_tail_blue_min_observations').value))
        self._semantic_exit_min_green_observations = max(1, int(
            self.get_parameter('semantic_exit_min_green_observations').value))
        self._semantic_exit_min_green_ratio = float(
            self.get_parameter('semantic_exit_min_green_ratio').value)
        self._blue_target_immediate_min_confidence = float(
            self.get_parameter('blue_target_immediate_min_confidence').value)
        self._exit_interrupt_blue_min_confidence = float(
            self.get_parameter('exit_interrupt_blue_min_confidence').value)
        self._use_observed_blue_clusters_as_targets = bool(
            self.get_parameter('use_observed_blue_clusters_as_targets').value)
        self._max_door_approach_failures_before_abandon = max(1, int(
            self.get_parameter('max_door_approach_failures_before_abandon').value))
        self._pre_exit_blue_approach_failures_before_abandon = max(1, int(
            self.get_parameter(
                'pre_exit_blue_approach_failures_before_abandon').value))
        self._detected_door_max_age_sec = float(
            self.get_parameter('detected_door_max_age_sec').value)
        self._observed_blue_same_id_max_jump_m = float(
            self.get_parameter('observed_blue_same_id_max_jump_m').value)
        self._observed_blue_same_id_max_total_drift_m = float(
            self.get_parameter('observed_blue_same_id_max_total_drift_m').value)
        self._observed_blue_stale_abandon_sec = float(
            self.get_parameter('observed_blue_stale_abandon_sec').value)
        self._observed_blue_stale_behind_m = float(
            self.get_parameter('observed_blue_stale_behind_m').value)
        self._observed_blue_target_max_behind_m = float(
            self.get_parameter('observed_blue_target_max_behind_m').value)
        self._observed_blue_same_side_after_opened_min_progress_m = float(
            self.get_parameter(
                'observed_blue_same_side_after_opened_min_progress_m').value)
        self._blue_same_side_after_opened_min_progress_m = float(
            self.get_parameter(
                'blue_same_side_after_opened_min_progress_m').value)
        self._blue_red_conflict_dist_m = float(
            self.get_parameter('blue_red_conflict_dist_m').value)
        self._blue_red_conflict_strict_dist_m = float(
            self.get_parameter('blue_red_conflict_strict_dist_m').value)
        self._observed_blue_min_abs_wall_y_m = float(
            self.get_parameter('observed_blue_min_abs_wall_y_m').value)
        self._observed_blue_max_abs_wall_y_m = float(
            self.get_parameter('observed_blue_max_abs_wall_y_m').value)
        self._blue_handle_max_wall_overshoot_m = float(
            self.get_parameter('blue_handle_max_wall_overshoot_m').value)
        self._blue_handle_validation_overshoot_m = float(
            self.get_parameter('blue_handle_validation_overshoot_m').value)
        self._red_blue_conflict_memory_sec = float(
            self.get_parameter('red_blue_conflict_memory_sec').value)
        self._blue_red_conflict_after_exit_dist_m = float(
            self.get_parameter('blue_red_conflict_after_exit_dist_m').value)
        self._red_observation_min_confidence = float(
            self.get_parameter('red_observation_min_confidence').value)
        self._red_observation_min_count_for_blue_suppression = max(1, int(
            self.get_parameter(
                'red_observation_min_count_for_blue_suppression').value))
        self._confirmed_blue_bypass_red_conflict_count = max(0, int(
            self.get_parameter(
                'confirmed_blue_bypass_red_conflict_count').value))
        self._confirmed_blue_bypass_red_conflict_confidence = float(
            self.get_parameter(
                'confirmed_blue_bypass_red_conflict_confidence').value)
        self._opened_station_blue_suppression_progress_m = float(
            self.get_parameter('opened_station_blue_suppression_progress_m').value)
        self._opened_station_same_side_blue_suppression_progress_m = float(
            self.get_parameter(
                'opened_station_same_side_blue_suppression_progress_m').value)
        self._opened_station_opposite_side_blue_suppression_progress_m = float(
            self.get_parameter(
                'opened_station_opposite_side_blue_suppression_progress_m').value)
        self._opened_station_same_id_blue_suppression_progress_m = float(
            self.get_parameter(
                'opened_station_same_id_blue_suppression_progress_m').value)
        self._abandoned_station_same_side_blue_suppression_progress_m = float(
            self.get_parameter(
                'abandoned_station_same_side_blue_suppression_progress_m').value)
        self._abandoned_station_opposite_side_blue_suppression_progress_m = float(
            self.get_parameter(
                'abandoned_station_opposite_side_blue_suppression_progress_m').value)
        self._max_door_open_failures_before_abandon = max(1, int(
            self.get_parameter('max_door_open_failures_before_abandon').value))
        self._door_open_requires_fresh_blue = bool(
            self.get_parameter('door_open_requires_fresh_blue').value)
        self._door_open_fresh_blue_max_age_sec = float(
            self.get_parameter('door_open_fresh_blue_max_age_sec').value)
        self._door_open_fresh_blue_max_dist_m = float(
            self.get_parameter('door_open_fresh_blue_max_dist_m').value)
        self._door_open_fresh_blue_min_confidence = float(
            self.get_parameter('door_open_fresh_blue_min_confidence').value)
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
        self._door_open_straight_ready_max_dist_m = float(
            self.get_parameter('door_open_straight_ready_max_dist_m').value)
        self._door_open_ready_yaw_tolerance = math.radians(float(
            self.get_parameter('door_open_ready_yaw_tolerance_deg').value))
        self._door_open_ready_lateral_tolerance_m = float(
            self.get_parameter('door_open_ready_lateral_tolerance_m').value)
        self._door_open_fine_control_max_dist_m = float(
            self.get_parameter('door_open_fine_control_max_dist_m').value)
        self._door_open_fine_lateral_tolerance_m = float(
            self.get_parameter('door_open_fine_lateral_tolerance_m').value)
        self._door_open_fine_position_kp = float(
            self.get_parameter('door_open_fine_position_kp').value)
        self._door_open_fine_linear_vel = float(
            self.get_parameter('door_open_fine_linear_vel').value)
        self._door_open_fine_min_linear_vel = float(
            self.get_parameter('door_open_fine_min_linear_vel').value)
        self._door_open_fine_allow_reverse = bool(
            self.get_parameter('door_open_fine_allow_reverse').value)
        self._door_open_axis_min_side_lateral_m = float(
            self.get_parameter('door_open_axis_min_side_lateral_m').value)
        self._door_open_settle_sec = float(self.get_parameter('door_open_settle_sec').value)
        self._exit_retry_delay_sec = float(self.get_parameter('exit_retry_delay_sec').value)
        self._door_open_align_angular_vel = float(
            self.get_parameter('door_open_align_angular_vel').value)
        self._door_open_align_kp = float(self.get_parameter('door_open_align_kp').value)
        self._door_open_align_timeout_sec = float(
            self.get_parameter('door_open_align_timeout_sec').value)
        self._post_open_reorient_enabled = bool(
            self.get_parameter('post_open_reorient_enabled').value)
        self._post_open_reorient_yaw_tolerance = math.radians(float(
            self.get_parameter('post_open_reorient_yaw_tolerance_deg').value))
        self._post_open_reorient_angular_vel = float(
            self.get_parameter('post_open_reorient_angular_vel').value)
        self._post_open_reorient_timeout_sec = float(
            self.get_parameter('post_open_reorient_timeout_sec').value)
        self._post_open_clearance_advance_enabled = bool(
            self.get_parameter('post_open_clearance_advance_enabled').value)
        self._post_open_clearance_advance_m = float(
            self.get_parameter('post_open_clearance_advance_m').value)
        self._post_open_clearance_linear_vel = float(
            self.get_parameter('post_open_clearance_linear_vel').value)
        self._post_open_clearance_front_min_m = float(
            self.get_parameter('post_open_clearance_front_min_m').value)
        self._post_open_clearance_front_angle = math.radians(float(
            self.get_parameter('post_open_clearance_front_angle_deg').value))
        self._post_open_clearance_timeout_sec = float(
            self.get_parameter('post_open_clearance_timeout_sec').value)
        self._post_open_clearance_heading_kp = float(
            self.get_parameter('post_open_clearance_heading_kp').value)
        self._post_open_clearance_angular_vel_limit = float(
            self.get_parameter('post_open_clearance_angular_vel_limit').value)
        self._post_open_side_retreat_enabled = bool(
            self.get_parameter('post_open_side_retreat_enabled').value)
        self._post_open_side_retreat_m = float(
            self.get_parameter('post_open_side_retreat_m').value)
        self._post_open_side_retreat_target_abs_y_m = float(
            self.get_parameter('post_open_side_retreat_target_abs_y_m').value)
        self._post_open_side_retreat_linear_vel = float(
            self.get_parameter('post_open_side_retreat_linear_vel').value)
        self._post_open_side_retreat_timeout_sec = float(
            self.get_parameter('post_open_side_retreat_timeout_sec').value)
        self._post_open_side_retreat_yaw_tolerance = math.radians(float(
            self.get_parameter('post_open_side_retreat_yaw_tolerance_deg').value))
        self._pre_nav_scan_sec = float(
            self.get_parameter('pre_nav_scan_sec').value)
        self._pre_nav_scan_angular_vel = float(
            self.get_parameter('pre_nav_scan_angular_vel').value)
        self._explore_observation_wait_sec = float(
            self.get_parameter('explore_observation_wait_sec').value)
        self._explore_observation_wait_angular_vel = float(
            self.get_parameter('explore_observation_wait_angular_vel').value)
        self._nav_target_republish_sec = float(
            self.get_parameter('nav_target_republish_sec').value)
        self._door_nav_stuck_watch_enabled = bool(
            self.get_parameter('door_nav_stuck_watch_enabled').value)
        self._door_nav_stuck_timeout_sec = float(
            self.get_parameter('door_nav_stuck_timeout_sec').value)
        self._door_nav_stuck_min_progress_m = float(
            self.get_parameter('door_nav_stuck_min_progress_m').value)
        self._door_nav_stuck_min_goal_dist_m = float(
            self.get_parameter('door_nav_stuck_min_goal_dist_m').value)
        self._explore_waypoint_scan_sec = float(
            self.get_parameter('explore_waypoint_scan_sec').value)
        self._explore_waypoint_scan_angular_vel = float(
            self.get_parameter('explore_waypoint_scan_angular_vel').value)
        self._explore_interrupt_min_confidence = float(
            self.get_parameter('explore_interrupt_min_confidence').value)
        self._explore_interrupt_max_distance_m = float(
            self.get_parameter('explore_interrupt_max_distance_m').value)
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
        self._explore_nav_clearance_deferral_limit = int(
            self.get_parameter('explore_nav_clearance_deferral_limit').value)
        self._explore_nav_planner_probe_step_m = float(
            self.get_parameter('explore_nav_planner_probe_step_m').value)
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
        self._explore_nav_lane_alternate_success_interval = max(1, int(
            self.get_parameter('explore_nav_lane_alternate_success_interval').value))
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
        self._explore_nav_front_block_scan_clearance_m = float(
            self.get_parameter('explore_nav_front_block_scan_clearance_m').value)
        self._explore_nav_front_block_scan_after_opened_m = float(
            self.get_parameter('explore_nav_front_block_scan_after_opened_m').value)
        self._max_explore_past_last_opened_blue_m = float(
            self.get_parameter('max_explore_past_last_opened_blue_m').value)
        self._abandoned_blue_extends_explore_limit = bool(
            self.get_parameter('abandoned_blue_extends_explore_limit').value)
        self._blue_target_max_beyond_explore_limit_m = float(
            self.get_parameter('blue_target_max_beyond_explore_limit_m').value)
        self._observed_exit_explore_grace_m = float(
            self.get_parameter('observed_exit_explore_grace_m').value)
        self._observed_exit_blue_candidate_margin_m = float(
            self.get_parameter('observed_exit_blue_candidate_margin_m').value)
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
        self._explore_nav_stuck_wall_timeout_sec = float(
            self.get_parameter('explore_nav_stuck_wall_timeout_sec').value)
        self._explore_nav_stuck_min_progress_m = float(
            self.get_parameter('explore_nav_stuck_min_progress_m').value)
        self._min_target_door_abs_y_m = float(
            self.get_parameter('min_target_door_abs_y_m').value)
        self._axis_door_approach_enabled = bool(
            self.get_parameter('axis_door_approach_enabled').value)
        self._axis_door_side_standoff_m = float(
            self.get_parameter('axis_door_side_standoff_m').value)
        self._axis_door_min_side_goal_lateral_m = float(
            self.get_parameter('axis_door_min_side_goal_lateral_m').value)
        self._axis_door_min_abs_lateral_m = float(
            self.get_parameter('axis_door_min_abs_lateral_m').value)
        self._axis_door_max_handle_pose_progress_delta_m = float(
            self.get_parameter('axis_door_max_handle_pose_progress_delta_m').value)
        self._axis_door_use_detected_pose_lateral = bool(
            self.get_parameter('axis_door_use_detected_pose_lateral').value)
        self._locked_target_refine_enabled = bool(
            self.get_parameter('locked_target_refine_enabled').value)
        self._locked_target_refine_min_shift_m = float(
            self.get_parameter('locked_target_refine_min_shift_m').value)
        self._locked_target_refine_max_progress_jump_m = float(
            self.get_parameter('locked_target_refine_max_progress_jump_m').value)
        self._locked_target_refine_max_total_progress_drift_m = float(
            self.get_parameter(
                'locked_target_refine_max_total_progress_drift_m').value)
        self._locked_target_refine_min_period_sec = float(
            self.get_parameter('locked_target_refine_min_period_sec').value)
        self._target_door_direct_nav_max_dist_m = float(
            self.get_parameter('target_door_direct_nav_max_dist_m').value)
        self._nav_start_max_abs_y_m = float(self.get_parameter('nav_start_max_abs_y_m').value)
        self._nav_start_max_heading_error = math.radians(
            float(self.get_parameter('nav_start_max_heading_error_deg').value))
        self._nav_start_center_recovery_enabled = bool(
            self.get_parameter('nav_start_center_recovery_enabled').value)
        self._nav_start_center_recovery_target_abs_y_m = float(
            self.get_parameter('nav_start_center_recovery_target_abs_y_m').value)
        self._nav_start_center_recovery_linear_vel = float(
            self.get_parameter('nav_start_center_recovery_linear_vel').value)
        self._nav_start_center_recovery_angular_vel = float(
            self.get_parameter('nav_start_center_recovery_angular_vel').value)
        self._nav_start_center_recovery_yaw_tolerance = math.radians(float(
            self.get_parameter('nav_start_center_recovery_yaw_tolerance_deg').value))
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
        self._no_blue_exit_min_robot_forward_m = float(
            self.get_parameter('no_blue_exit_min_robot_forward_m').value)
        self._detected_exit_max_center_y_m = float(
            self.get_parameter('detected_exit_max_center_y_m').value)
        self._detected_exit_merge_dist_m = float(
            self.get_parameter('detected_exit_merge_dist_m').value)
        self._detected_exit_min_after_opened_blue_m = float(
            self.get_parameter('detected_exit_min_after_opened_blue_m').value)
        self._detected_exit_min_robot_past_opened_blue_m = float(
            self.get_parameter(
                'detected_exit_min_robot_past_opened_blue_m').value)
        self._detected_exit_require_front_wall_confirmation = bool(
            self.get_parameter(
                'detected_exit_require_front_wall_confirmation').value)
        self._detected_exit_front_wall_tolerance_m = float(
            self.get_parameter('detected_exit_front_wall_tolerance_m').value)
        self._blue_exit_suppression_margin_m = float(
            self.get_parameter('blue_exit_suppression_margin_m').value)
        self._pre_exit_blue_requires_fresh_detection = bool(
            self.get_parameter('pre_exit_blue_requires_fresh_detection').value)
        self._pre_exit_blue_suppression_margin_m = float(
            self.get_parameter('pre_exit_blue_suppression_margin_m').value)
        self._pre_exit_blue_min_confidence = float(
            self.get_parameter('pre_exit_blue_min_confidence').value)
        self._pre_exit_blue_fresh_max_age_sec = float(
            self.get_parameter('pre_exit_blue_fresh_max_age_sec').value)
        self._exit_pass_through_m = float(self.get_parameter('exit_pass_through_m').value)
        self._exit_nav_standoff_m = float(
            self.get_parameter('exit_nav_standoff_m').value)
        self._exit_visible_nav_max_dist_m = float(
            self.get_parameter('exit_visible_nav_max_dist_m').value)
        self._exit_direct_cross_max_dist_m = float(
            self.get_parameter('exit_direct_cross_max_dist_m').value)
        self._exit_cross_linear_vel = float(
            self.get_parameter('exit_cross_linear_vel').value)
        self._exit_cross_heading_kp = float(
            self.get_parameter('exit_cross_heading_kp').value)
        self._exit_cross_angular_vel_limit = float(
            self.get_parameter('exit_cross_angular_vel_limit').value)
        self._exit_cross_timeout_sec = float(
            self.get_parameter('exit_cross_timeout_sec').value)
        self._exit_cross_rotate_in_place_threshold = math.radians(float(
            self.get_parameter('exit_cross_rotate_in_place_threshold_deg').value))
        self._exit_interrupt_blue_before_exit_margin_m = float(
            self.get_parameter(
                'exit_interrupt_blue_before_exit_margin_m').value)
        self._allow_front_wall_exit_fallback = bool(
            self.get_parameter('allow_front_wall_exit_fallback').value)
        self._front_wall_exit_min_range_m = float(
            self.get_parameter('front_wall_exit_min_range_m').value)
        self._front_wall_exit_max_range_m = float(
            self.get_parameter('front_wall_exit_max_range_m').value)
        self._front_wall_exit_half_angle = math.radians(float(
            self.get_parameter('front_wall_exit_half_angle_deg').value))
        self._front_wall_exit_min_coverage = float(
            self.get_parameter('front_wall_exit_min_coverage').value)
        self._front_wall_exit_max_spread_m = float(
            self.get_parameter('front_wall_exit_max_spread_m').value)
        self._front_wall_exit_min_after_opened_blue_m = float(
            self.get_parameter('front_wall_exit_min_after_opened_blue_m').value)

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
        axis_qos = QoSProfile(depth=1)
        axis_qos.reliability = ReliabilityPolicy.RELIABLE
        axis_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.axis_sub = self.create_subscription(
            PoseStamped, '/mission_axis', self.mission_axis_callback, axis_qos,
            callback_group=cb_group)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.state_pub       = self.create_publisher(RobotState, '/robot_state', 10)
        self.target_door_pub = self.create_publisher(DoorInfo,   '/target_door', 10)
        self.cmd_vel_pub     = self.create_publisher(Twist,      '/cmd_vel',     10)
        self.nav_cancel_pub = self.create_publisher(Bool, '/cancel_navigation', 10)

        self.open_door_client = self.create_client(
            OpenDoor, '/open_door', callback_group=cb_group)

        self.state              = State.IDLE
        self.detected_doors:    list[DoorInfo] = []
        self.fire_info:         FireInfo | None = None
        self.target_door:       DoorInfo | None = None
        self._exit_door:        DoorInfo | None = None   # 탐지된 초록 비상구
        self._last_valid_exit_xy: tuple[float, float] | None = None
        self._last_valid_exit_from_front_wall = False
        self._last_provisional_exit_xy: tuple[float, float] | None = None
        self._latest_scan:      LaserScan | None = None
        self._last_lane_sign    = 0.0
        self._explore_lane_success_since_flip = 0

        self._nav_done   = False
        self._nav_failed = False
        self._nav_start_time:     Time | None = None
        self._door_nav_wall_start_time: float | None = None
        self._door_nav_last_progress: float | None = None
        self._door_nav_last_progress_wall_time: float | None = None
        self._explore_start_time: Time | None = None

        self._failed_door_ids: set[str] = set()   # 네비게이션 실패 문
        self._opened_door_ids: set[str] = set()   # 개방 완료 문
        self._opened_blue_door_positions: list[tuple[float, float]] = []
        self._observed_blue_doors: list[dict[str, float]] = []
        self._observed_blue_door_id_to_cluster: dict[str, dict[str, float]] = {}
        self._semantic_door_landmarks: list[dict[str, float]] = []
        self._observed_red_door_positions: list[dict[str, float]] = []
        self._detected_blue_origin_by_id: dict[str, tuple[float, float]] = {}
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
        self._last_target_republish_time: Time | None = None
        self._pre_nav_scan_start_time: Time | None = None
        self._pending_nav_scan_target_id: str | None = None
        self._explore_observation_wait_start_time: Time | None = None
        self._explore_observation_wait_completed = False
        self._explore_waypoint_scan_pending = False
        self._explore_waypoint_scan_start_time: Time | None = None
        self._explore_waypoint_scan_direction = 1.0
        self._explore_waypoint_scan_bilateral = False
        self._explore_nav_goal_id: str = ''
        self._explore_nav_failure_count = 0
        self._explore_nav_goal_sign = 0.0
        self._explore_nav_goal_progress = 0.0
        self._explore_nav_goal_xy: tuple[float, float] | None = None
        self._explore_nav_wall_start_time: float | None = None
        self._explore_nav_last_progress: float | None = None
        self._explore_nav_last_goal_dist: float | None = None
        self._explore_nav_last_yaw: float | None = None
        self._explore_nav_last_progress_wall_time: float | None = None
        self._explore_nav_clearance_deferral_count = 0
        self._recent_failed_explore_goals: list[tuple[float, float, float]] = []
        self._force_final_scan_before_exit = False
        self._active_exit_goal: tuple[float, float, float] | None = None
        self._exit_crossing_start_time: Time | None = None
        self._last_locked_target_refine_time: Time | None = None
        self._locked_blue_anchor_key: str | None = None
        self._locked_blue_anchor_xy: tuple[float, float] | None = None
        self._post_open_reorient_pending = False
        self._post_open_reorient_start_time: Time | None = None
        self._post_open_advance_pending = False
        self._post_open_advance_start_time: Time | None = None
        self._post_open_advance_start_progress: float | None = None
        self._post_open_side_retreat_pending = False
        self._post_open_side_retreat_start_time: Time | None = None
        self._post_open_side_retreat_start_lateral: float | None = None
        self._nav_start_pose_recovery_active = False
        self._target_subfsm = DoorTargetSubFsm(self)
        self._approach_subfsm = DoorApproachSubFsm(self)

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
        msg = self._door_with_side_wall_progress_bias(msg)
        self._record_semantic_door_observation(msg)

        # 초록 문(비상구) → 별도 저장, detected_doors에 추가하지 않음
        if msg.door_color == 'green':
            self._update_detected_exit(msg)
            return

        if msg.door_color == 'red':
            self._record_observed_physical_red_door(msg)
        elif msg.door_color == 'blue':
            if self._is_door_opened_for_observation(msg):
                xy = self._door_identity_xy(msg)
                self.get_logger().info(
                    f'Blue candidate {msg.door_id} ignored: physical/opened '
                    f'door station already handled nearby at '
                    f'({xy[0]:.2f}, {xy[1]:.2f}).',
                    throttle_duration_sec=3.0)
                return
            if self._is_door_abandoned_for_observation(msg):
                self.get_logger().info(
                    f'Blue candidate {msg.door_id} ignored: physical/abandoned '
                    'door station already handled nearby.',
                    throttle_duration_sec=3.0)
                return

        # 1차: 동일 door_id라도 map 위치가 가까울 때만 같은 물리 문으로 갱신
        existing = next(
            (d for d in self.detected_doors
             if self._is_same_door_id_observation(d, msg)), None)
        if existing:
            self._merge_door_observation(existing, msg)
            self._record_observed_physical_blue_door(existing)
            self._maybe_refine_locked_blue_target(msg)
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
            self._maybe_refine_locked_blue_target(msg)
            return
        self._ensure_unique_detected_door_id(msg)
        self._remember_detected_blue_origin(msg)
        self.detected_doors.append(msg)
        self._record_observed_physical_blue_door(msg)
        self._maybe_refine_locked_blue_target(msg)

    def _door_with_side_wall_progress_bias(self, door: DoorInfo) -> DoorInfo:
        bias = max(0.0, self._side_wall_door_progress_bias_m)
        if bias <= 0.0:
            return door
        if door.door_color not in ('blue', 'red'):
            return door
        if not self._door_has_map_identity(door):
            return door

        ref_xy: tuple[float, float] | None = None
        if door.handle_position.header.frame_id == 'map':
            ref_xy = (
                float(door.handle_position.point.x),
                float(door.handle_position.point.y),
            )
        elif door.door_pose.header.frame_id == 'map':
            ref_xy = (
                float(door.door_pose.pose.position.x),
                float(door.door_pose.pose.position.y),
            )
        if ref_xy is None:
            return door

        ref_lateral = (
            self._axis_lateral_xy(ref_xy[0], ref_xy[1])
            - self._explore_center_y)
        wall_threshold = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        if abs(ref_lateral) < wall_threshold:
            return door

        dx = bias * math.cos(self._mission_axis_yaw)
        dy = bias * math.sin(self._mission_axis_yaw)
        if (
                door.door_color == 'blue'
                and self._opened_blue_door_positions
                and not self._is_door_id_opened(door.door_id)
                and door.door_pose.header.frame_id == 'map'):
            raw_xy = (
                float(door.door_pose.pose.position.x),
                float(door.door_pose.pose.position.y),
            )
            biased_xy = (raw_xy[0] - dx, raw_xy[1] - dy)
            raw_progress = self._axis_progress_xy(raw_xy[0], raw_xy[1])
            biased_progress = self._axis_progress_xy(biased_xy[0], biased_xy[1])
            raw_lateral = (
                self._axis_lateral_xy(raw_xy[0], raw_xy[1])
                - self._explore_center_y)
            wall_side_threshold = max(
                0.15,
                self._observed_blue_min_abs_wall_y_m * 0.5)
            opened_window = max(
                self._opened_physical_door_merge_dist_m,
                self._opened_station_blue_suppression_progress_m,
                self._opened_station_same_side_blue_suppression_progress_m)
            for opened_xy in self._opened_blue_door_positions:
                opened_progress = self._axis_progress_xy(
                    opened_xy[0], opened_xy[1])
                opened_lateral = (
                    self._axis_lateral_xy(opened_xy[0], opened_xy[1])
                    - self._explore_center_y)
                same_wall_side = (
                    abs(raw_lateral) >= wall_side_threshold
                    and abs(opened_lateral) >= wall_side_threshold
                    and raw_lateral * opened_lateral > 0.0)
                if not same_wall_side:
                    continue
                if (
                        raw_progress > opened_progress + opened_window + 0.35
                        and biased_progress <= opened_progress + opened_window):
                    self.get_logger().info(
                        f'Side-wall progress bias skipped for {door.door_id}: '
                        'bias would collapse a forward blue observation into '
                        'an already-opened station.',
                        throttle_duration_sec=3.0)
                    return door

        adjusted = copy.deepcopy(door)
        if adjusted.door_pose.header.frame_id == 'map':
            adjusted.door_pose.pose.position.x -= dx
            adjusted.door_pose.pose.position.y -= dy
        if adjusted.handle_position.header.frame_id == 'map':
            adjusted.handle_position.point.x -= dx
            adjusted.handle_position.point.y -= dy
        self.get_logger().info(
            f'Side-wall door progress bias applied: {door.door_id}, '
            f'color={door.door_color}, bias={bias:.2f}m.',
            throttle_duration_sec=3.0)
        return adjusted

    def _maybe_refine_locked_blue_target(self, observation: DoorInfo):
        if not self._locked_target_refine_enabled:
            return
        if self.state != State.NAVIGATING:
            return
        target = self.target_door
        if target is None or target.door_color != 'blue':
            return
        if observation.door_color != 'blue':
            return
        if not self._door_has_map_identity(observation):
            return
        if not self._is_blue_door_at_valid_wall_position(observation):
            return
        if not self._is_same_locked_blue_target_observation(observation):
            return

        old_xy = self._door_identity_xy(target)
        new_xy = self._door_identity_xy(observation)
        if (
                self._is_blue_xy_at_or_after_observed_exit(new_xy)
                or self._blue_xy_in_strict_exit_tail_without_persistent_memory(
                    new_xy, observation)):
            self.get_logger().info(
                f'Locked blue target refine skipped: {observation.door_id} '
                'moved into observed exit tail.',
                throttle_duration_sec=2.0)
            return
        target_key = self._canonical_door_id(target.door_id)
        if (self._locked_blue_anchor_xy is None
                or self._locked_blue_anchor_key != target_key):
            self._set_locked_blue_anchor(target)
        anchor_xy = self._locked_blue_anchor_xy
        old_progress = self._axis_progress_xy(old_xy[0], old_xy[1])
        new_progress = self._axis_progress_xy(new_xy[0], new_xy[1])
        progress_jump = abs(new_progress - old_progress)
        max_jump = max(0.0, self._locked_target_refine_max_progress_jump_m)
        if max_jump > 0.0 and progress_jump > max_jump:
            self.get_logger().info(
                f'Locked blue target refine skipped: {observation.door_id} '
                f'progress jump {progress_jump:.2f}m > {max_jump:.2f}m',
                throttle_duration_sec=2.0)
            return

        max_total_jump = max(
            0.0,
            self._locked_target_refine_max_total_progress_drift_m)
        if anchor_xy is not None and max_total_jump > 0.0:
            anchor_progress = self._axis_progress_xy(anchor_xy[0], anchor_xy[1])
            total_progress_jump = abs(new_progress - anchor_progress)
            total_xy_jump = math.hypot(
                new_xy[0] - anchor_xy[0], new_xy[1] - anchor_xy[1])
            if total_progress_jump > max_total_jump or total_xy_jump > max_total_jump:
                self.get_logger().info(
                    f'Locked blue target refine skipped: {observation.door_id} '
                    f'total drift progress={total_progress_jump:.2f}m, '
                    f'xy={total_xy_jump:.2f}m > {max_total_jump:.2f}m '
                    'from locked anchor',
                    throttle_duration_sec=2.0)
                return

        old_lateral = self._axis_lateral_xy(old_xy[0], old_xy[1]) - self._explore_center_y
        new_lateral = self._axis_lateral_xy(new_xy[0], new_xy[1]) - self._explore_center_y
        if old_lateral * new_lateral < 0.0:
            return
        if anchor_xy is not None:
            anchor_lateral = (
                self._axis_lateral_xy(anchor_xy[0], anchor_xy[1])
                - self._explore_center_y)
            if anchor_lateral * new_lateral < 0.0:
                self.get_logger().info(
                    f'Locked blue target refine skipped: {observation.door_id} '
                    'moved to the opposite side of the locked anchor.',
                    throttle_duration_sec=2.0)
                return
            if self._refine_moves_too_far_toward_center(
                    anchor_lateral, new_lateral, observation.door_id):
                return
        elif self._refine_moves_too_far_toward_center(
                old_lateral, new_lateral, observation.door_id):
            return

        refined = self._door_with_axis_aligned_approach(observation)
        # Keep the locked target identity stable; only refine its observed pose.
        refined.door_id = target.door_id
        if refined.door_pose.header.frame_id != 'map' or target.door_pose.header.frame_id != 'map':
            return
        old_pose = target.door_pose.pose.position
        new_pose = refined.door_pose.pose.position
        shift = math.hypot(new_pose.x - old_pose.x, new_pose.y - old_pose.y)
        if shift < max(0.0, self._locked_target_refine_min_shift_m):
            return

        now = self.get_clock().now()
        if self._last_locked_target_refine_time is not None:
            elapsed = (now - self._last_locked_target_refine_time).nanoseconds / 1e9
            if elapsed < max(0.0, self._locked_target_refine_min_period_sec):
                return
        self._last_locked_target_refine_time = now

        self.target_door = refined
        self._nav_done = False
        self._nav_failed = False
        self._nav_start_time = now
        self.get_logger().warn(
            f'Locked blue target refined from newer observation: '
            f'{target.door_id} -> {refined.door_id}, '
            f'goal shift={shift:.2f}m, '
            f'identity=({old_xy[0]:.2f},{old_xy[1]:.2f})'
            f' -> ({new_xy[0]:.2f},{new_xy[1]:.2f})')
        self.target_door_pub.publish(self.target_door)

    def _refine_moves_too_far_toward_center(
            self, reference_lateral: float, new_lateral: float,
            door_id: str) -> bool:
        reference_abs = abs(reference_lateral)
        new_abs = abs(new_lateral)
        wall_reference = max(
            self._axis_door_min_side_goal_lateral_m,
            self._observed_blue_min_abs_wall_y_m,
            0.65,
        )
        max_inward_shift = 0.45
        if reference_abs < wall_reference:
            return False
        if new_abs + max_inward_shift >= reference_abs:
            return False
        self.get_logger().info(
            f'Locked blue target refine skipped: {door_id} moved inward '
            f'{reference_abs:.2f}m -> {new_abs:.2f}m from mission axis.',
            throttle_duration_sec=2.0)
        return True

    def _is_same_locked_blue_target_observation(self, observation: DoorInfo) -> bool:
        target = self.target_door
        if target is None:
            return False
        if observation.door_id in self._door_id_keys(target.door_id):
            return True
        if target.door_id in self._door_id_keys(observation.door_id):
            return True
        if self._canonical_door_id(observation.door_id) == self._canonical_door_id(target.door_id):
            return True
        return False

    def _set_locked_blue_anchor(self, door: DoorInfo | None):
        if door is None or door.door_color != 'blue' or not self._door_has_map_identity(door):
            self._clear_locked_blue_anchor()
            return
        self._locked_blue_anchor_key = self._canonical_door_id(door.door_id)
        self._locked_blue_anchor_xy = self._door_identity_xy(door)
        self._last_locked_target_refine_time = None

    def _clear_locked_blue_anchor(self):
        self._locked_blue_anchor_key = None
        self._locked_blue_anchor_xy = None
        self._last_locked_target_refine_time = None

    def _update_detected_exit(self, msg: DoorInfo):
        new_xy = self._exit_observation_xy(msg)
        if new_xy is None:
            if self._exit_door is None:
                self._exit_door = copy.deepcopy(msg)
            return

        self._remember_provisional_exit_observation(new_xy, msg.door_id)

        if not self._exit_candidate_axis_valid(new_xy, msg.door_id):
            return

        if self._exit_door is None:
            self.get_logger().info(
                f'비상구 탐지: {msg.door_id} ({new_xy[0]:.1f}, {new_xy[1]:.1f})')
            self._exit_door = copy.deepcopy(msg)
            self._last_valid_exit_xy = new_xy
            self._last_valid_exit_from_front_wall = False
            return

        old_xy = self._exit_observation_xy(self._exit_door)
        if old_xy is None:
            self._exit_door = copy.deepcopy(msg)
            self._last_valid_exit_xy = new_xy
            self._last_valid_exit_from_front_wall = False
            return

        old_id = self._exit_door.door_id
        if not self._exit_candidate_axis_valid(old_xy, old_id, clear_existing=True):
            self.get_logger().info(
                f'Existing exit candidate {old_id} failed current centerline validation; '
                f'replacing it with {msg.door_id}.')
            self._exit_door = copy.deepcopy(msg)
            self._last_valid_exit_xy = new_xy
            self._last_valid_exit_from_front_wall = False
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
            self._last_valid_exit_xy = new_xy
            self._last_valid_exit_from_front_wall = False
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
            merged_xy = self._exit_observation_xy(self._exit_door)
            if merged_xy is not None:
                self._last_valid_exit_xy = merged_xy
                self._last_valid_exit_from_front_wall = False
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
            self._last_valid_exit_xy = new_xy
            self._last_valid_exit_from_front_wall = False
            return

        self.get_logger().debug(
            f'비상구 후보 {msg.door_id} ({new_xy[0]:.1f}, {new_xy[1]:.1f})는 '
            f'기존 관측과 {jump:.1f}m 떨어져 있어 출구 좌표 갱신에서 제외합니다.')

    def _remember_provisional_exit_observation(
            self, xy: tuple[float, float], door_id: str = ''):
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return
        exit_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_lateral = self._axis_lateral_xy(xy[0], xy[1])
        lateral_error = abs(exit_lateral - self._explore_center_y)
        max_lateral = max(
            0.0,
            self._detected_exit_max_abs_lateral_m,
            self._detected_exit_max_center_y_m + 0.6)
        if max_lateral > 0.0 and lateral_error > max_lateral:
            return
        if self._opened_blue_door_positions:
            farthest_opened = max(
                self._axis_progress_xy(x, y)
                for x, y in self._opened_blue_door_positions)
            min_after = max(
                0.8,
                self._detected_exit_min_after_opened_blue_m * 0.5)
            if exit_progress < farthest_opened + min_after:
                return

        old_xy = self._last_provisional_exit_xy
        if old_xy is not None:
            old_progress = self._axis_progress_xy(old_xy[0], old_xy[1])
            old_lateral_error = abs(
                self._axis_lateral_xy(old_xy[0], old_xy[1])
                - self._explore_center_y)
            # The final green exit should normally be the farthest centered
            # observation along the initially mapped corridor. Do not let a
            # nearer green false positive pull the provisional exit backward.
            if exit_progress < old_progress - 0.8:
                if lateral_error >= old_lateral_error - 0.25:
                    return
            if (exit_progress > old_progress + 2.0
                    and lateral_error > old_lateral_error + 0.55):
                return
        self._last_provisional_exit_xy = xy
        self.get_logger().info(
            f'보조 비상구 관측 저장: {door_id or "green"} '
            f'({xy[0]:.1f}, {xy[1]:.1f})',
            throttle_duration_sec=3.0)

    def _provisional_exit_xy_for_blue_suppression(self) -> tuple[float, float] | None:
        xy = self._last_provisional_exit_xy
        if xy is None:
            return None
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return None
        if self._opened_blue_door_positions:
            exit_progress = self._axis_progress_xy(xy[0], xy[1])
            farthest_opened = max(
                self._axis_progress_xy(x, y)
                for x, y in self._opened_blue_door_positions)
            min_after = max(
                0.8,
                self._detected_exit_min_after_opened_blue_m * 0.5)
            if exit_progress < farthest_opened + min_after:
                return None

        if (self._detected_exit_require_front_wall_confirmation
                and not self._no_blue_exit_mode()):
            wall_xy = self._front_wall_exit_xy()
            if wall_xy is None:
                return None
            exit_progress = self._axis_progress_xy(xy[0], xy[1])
            wall_progress = self._axis_progress_xy(wall_xy[0], wall_xy[1])
            tolerance = max(0.0, self._detected_exit_front_wall_tolerance_m)
            if abs(exit_progress - wall_progress) > tolerance:
                return None
        return xy

    def _exit_observation_xy(self, door: DoorInfo) -> tuple[float, float] | None:
        candidates: list[tuple[float, float]] = []
        if door.door_pose.header.frame_id == 'map':
            candidates.append((
                float(door.door_pose.pose.position.x),
                float(door.door_pose.pose.position.y)))
        if door.handle_position.header.frame_id == 'map':
            candidates.append((
                float(door.handle_position.point.x),
                float(door.handle_position.point.y)))
        if not candidates:
            return None
        if door.door_color == 'green' and len(candidates) > 1:
            # For the green exit, depth/handle projection can jump far along a
            # side wall. Prefer the farthest centered observation, but ignore
            # candidates whose lateral error is not compatible with the exit
            # centerline.
            max_lateral = max(
                0.0,
                self._detected_exit_max_abs_lateral_m,
                self._detected_exit_max_center_y_m + 0.6)
            centered = [
                xy for xy in candidates
                if max_lateral <= 0.0
                or abs(self._axis_lateral_xy(xy[0], xy[1])
                       - self._explore_center_y) <= max_lateral
            ]
            usable = centered if centered else candidates
            return max(
                usable,
                key=lambda xy: (
                    -abs(self._axis_lateral_xy(xy[0], xy[1])
                         - self._explore_center_y),
                    self._axis_progress_xy(xy[0], xy[1])))
        return candidates[0]

    def _is_same_door_id_observation(
            self, existing: DoorInfo, msg: DoorInfo) -> bool:
        if existing.door_id != msg.door_id:
            return False
        if (existing.door_pose.header.frame_id == 'map'
                and msg.door_pose.header.frame_id == 'map'):
            if existing.door_color == 'blue' and msg.door_color == 'blue':
                origin = self._detected_blue_origin_by_id.get(existing.door_id)
                if origin is None:
                    origin = self._door_identity_xy(existing)
                    self._detected_blue_origin_by_id[existing.door_id] = origin
                msg_xy = self._door_identity_xy(msg)
                max_total = max(0.0, self._observed_blue_same_id_max_total_drift_m)
                if (max_total > 0.0
                        and math.hypot(msg_xy[0] - origin[0],
                                       msg_xy[1] - origin[1]) > max_total):
                    return False
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

    def _remember_detected_blue_origin(self, door: DoorInfo):
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return
        self._detected_blue_origin_by_id.setdefault(
            door.door_id,
            self._door_identity_xy(door))

    def _canonical_door_id(self, door_id: str) -> str:
        if not door_id:
            return ''
        base_id, sep, suffix = door_id.rpartition('_')
        # Numeric suffixes like "..._1" are assigned locally by
        # _ensure_unique_detected_door_id. Do not collapse detector IDs such as
        # "door_blue_123456".
        if sep and suffix.isdigit() and base_id.count('_') >= 2:
            return base_id
        return door_id

    def _door_id_keys(self, door_id: str) -> set[str]:
        if not door_id:
            return set()
        return {door_id, self._canonical_door_id(door_id)}

    def _add_door_id_keys(self, target: set[str], door_id: str):
        target.update(key for key in self._door_id_keys(door_id) if key)

    def _is_door_id_in_set(self, target: set[str], door_id: str) -> bool:
        return bool(self._door_id_keys(door_id) & target)

    def _is_door_id_opened(self, door_id: str) -> bool:
        return self._is_door_id_in_set(self._opened_door_ids, door_id)

    def _is_door_id_abandoned(self, door_id: str) -> bool:
        return self._is_door_id_in_set(self._abandoned_door_ids, door_id)

    def _is_door_id_failed(self, door_id: str) -> bool:
        return self._is_door_id_in_set(self._failed_door_ids, door_id)

    def _is_door_opened_for_observation(self, door: DoorInfo) -> bool:
        if door.door_color == 'blue' and self._door_has_map_identity(door):
            return (
                self._is_door_physically_opened(door)
                or self._is_blue_same_id_opened_in_exit_tail(door)
                or self._is_blue_same_id_near_opened_station(door)
            )
        if self._is_door_id_opened(door.door_id):
            return True
        return False

    def _is_blue_same_id_opened_in_exit_tail(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        if not self._is_door_id_opened(door.door_id):
            return False
        if not self._opened_blue_door_positions:
            return False
        xy = self._door_identity_xy(door)
        if not (
                self._is_blue_xy_in_pre_exit_zone(xy)
                or self._is_blue_xy_at_or_after_observed_exit(xy)
                or self._is_blue_xy_near_detected_exit(xy)):
            return False

        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        candidate_lateral = (
            self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        candidate_on_wall = abs(candidate_lateral) >= wall_side_threshold
        max_progress_gap = max(
            self._opened_physical_door_merge_dist_m,
            self._observed_candidate_merge_dist(),
            self._opened_station_same_id_blue_suppression_progress_m)

        for opened_xy in self._opened_blue_door_positions:
            opened_progress = self._axis_progress_xy(
                opened_xy[0], opened_xy[1])
            opened_lateral = (
                self._axis_lateral_xy(opened_xy[0], opened_xy[1])
                - self._explore_center_y)
            opened_on_wall = abs(opened_lateral) >= wall_side_threshold
            if (
                    candidate_on_wall
                    and opened_on_wall
                    and candidate_lateral * opened_lateral < 0.0):
                continue
            if abs(candidate_progress - opened_progress) <= max_progress_gap:
                self.get_logger().info(
                    f'Blue candidate {door.door_id} ignored: same canonical '
                    f'ID near opened exit-tail station '
                    f'{opened_progress:.1f}m (candidate '
                    f'{candidate_progress:.1f}m).',
                    throttle_duration_sec=3.0)
                return True
        return False

    def _is_blue_same_id_near_opened_station(self, door: DoorInfo) -> bool:
        window = max(
            0.0, self._opened_station_same_id_blue_suppression_progress_m)
        if window <= 0.0:
            return False
        if not self._is_door_id_opened(door.door_id):
            return False
        if not self._opened_blue_door_positions:
            return True
        xy = self._door_identity_xy(door)
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        candidate_lateral = (
            self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        candidate_on_wall = abs(candidate_lateral) >= wall_side_threshold
        for opened_xy in self._opened_blue_door_positions:
            opened_progress = self._axis_progress_xy(
                opened_xy[0], opened_xy[1])
            opened_lateral = (
                self._axis_lateral_xy(opened_xy[0], opened_xy[1])
                - self._explore_center_y)
            opened_on_wall = abs(opened_lateral) >= wall_side_threshold
            if (
                    candidate_on_wall
                    and opened_on_wall
                    and candidate_lateral * opened_lateral < 0.0):
                continue
            if abs(candidate_progress - opened_progress) <= window:
                self.get_logger().info(
                    f'Blue candidate {door.door_id} ignored: same detector '
                    f'ID family near opened station '
                    f'{opened_progress:.1f}m (candidate '
                    f'{candidate_progress:.1f}m).',
                    throttle_duration_sec=3.0)
                return True
        return False

    def _is_door_abandoned_for_observation(self, door: DoorInfo) -> bool:
        if door.door_color == 'blue' and self._door_has_map_identity(door):
            return self._is_door_physically_abandoned(door)
        if self._is_door_id_abandoned(door.door_id):
            return True
        return False

    def _is_door_failed_for_observation(self, door: DoorInfo) -> bool:
        if door.door_color == 'blue' and self._door_has_map_identity(door):
            return self._is_door_physically_abandoned(door)
        if self._is_door_id_failed(door.door_id):
            return True
        return False

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
            if self.state == State.IDLE or not self._opened_blue_door_positions:
                self._mission_progress_x = progress
            else:
                opened_progresses = [
                    self._axis_progress_xy(x, y)
                    for x, y in self._opened_blue_door_positions
                ]
                if opened_progresses:
                    self._mission_progress_x = max(
                        self._mission_start_x,
                        max(opened_progresses))
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

    def _record_semantic_door_observation(self, door: DoorInfo):
        if not self._semantic_door_memory_enabled:
            return
        if not self._door_has_map_identity(door):
            return
        color = str(door.door_color).lower()
        if color not in ('blue', 'red', 'green'):
            return
        if color == 'green':
            xy = self._exit_observation_xy(door)
            if xy is None:
                return
        else:
            xy = self._door_identity_xy(door)
        landmark = self._find_semantic_door_landmark(
            xy, merge_dist=self._semantic_door_merge_dist_m)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if landmark is None:
            landmark = {
                'x': xy[0],
                'y': xy[1],
                'count': 0.0,
                'blue_count': 0.0,
                'red_count': 0.0,
                'green_count': 0.0,
                'confidence': 0.0,
                'last_seen': now_sec,
                'opened': 0.0,
                'abandoned': 0.0,
            }
            self._semantic_door_landmarks.append(landmark)

        count = int(landmark.get('count', 0.0)) + 1
        alpha = 1.0 / min(count, 8)
        landmark['x'] = (1.0 - alpha) * float(landmark.get('x', xy[0])) + alpha * xy[0]
        landmark['y'] = (1.0 - alpha) * float(landmark.get('y', xy[1])) + alpha * xy[1]
        landmark['count'] = float(count)
        landmark[f'{color}_count'] = float(landmark.get(f'{color}_count', 0.0)) + 1.0
        landmark['confidence'] = max(float(landmark.get('confidence', 0.0)), float(door.confidence))
        landmark['last_seen'] = now_sec

    def _find_semantic_door_landmark(
            self,
            xy: tuple[float, float],
            merge_dist: float | None = None) -> dict[str, float] | None:
        if not self._semantic_door_memory_enabled:
            return None
        dist_limit = max(
            0.20,
            self._semantic_door_merge_dist_m if merge_dist is None else merge_dist)
        query_progress = self._axis_progress_xy(xy[0], xy[1])
        query_lateral = self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y
        wall_threshold = max(0.35, self._observed_blue_min_abs_wall_y_m * 0.5)
        query_side = 0.0
        if abs(query_lateral) >= wall_threshold:
            query_side = 1.0 if query_lateral > 0.0 else -1.0
        same_wall_progress = max(0.0, self._semantic_same_wall_merge_progress_m)
        same_wall_lateral = max(0.0, self._semantic_same_wall_merge_lateral_m)

        best: tuple[float, dict[str, float]] | None = None
        for landmark in self._semantic_door_landmarks:
            lx = float(landmark.get('x', 0.0))
            ly = float(landmark.get('y', 0.0))
            dist = math.hypot(xy[0] - lx, xy[1] - ly)
            progress_gap = abs(query_progress - self._axis_progress_xy(lx, ly))
            landmark_lateral = self._axis_lateral_xy(lx, ly) - self._explore_center_y
            landmark_side = 0.0
            if abs(landmark_lateral) >= wall_threshold:
                landmark_side = 1.0 if landmark_lateral > 0.0 else -1.0
            lateral_gap = abs(query_lateral - landmark_lateral)
            same_wall_match = (
                query_side != 0.0
                and query_side == landmark_side
                and progress_gap <= same_wall_progress
                and lateral_gap <= same_wall_lateral)
            if (dist <= dist_limit or same_wall_match) and (
                    best is None or dist < best[0]):
                best = (dist, landmark)
        return best[1] if best is not None else None

    def _semantic_landmark_color_votes(
            self, landmark: dict[str, float]) -> tuple[int, int, int]:
        return (
            int(landmark.get('blue_count', 0.0)),
            int(landmark.get('red_count', 0.0)),
            int(landmark.get('green_count', 0.0)),
        )

    def _semantic_blue_landmark_confirmed(
            self,
            landmark: dict[str, float],
            xy: tuple[float, float] | None = None) -> bool:
        blue_count, red_count, green_count = self._semantic_landmark_color_votes(landmark)
        total_color_votes = max(1, blue_count + red_count + green_count)
        min_blue_count = max(
            self._semantic_blue_min_observations,
            self._observed_blue_min_observations)
        if xy is not None and (
                self._is_blue_xy_in_pre_exit_zone(xy)
                or self._is_blue_xy_in_exit_search_tail(xy)):
            min_blue_count = max(
                min_blue_count,
                self._semantic_exit_tail_blue_min_observations)
        if blue_count < min_blue_count:
            return False
        if blue_count / float(total_color_votes) < self._semantic_blue_min_ratio:
            return False
        # Green/red votes at the same map landmark mean this is likely an exit,
        # handle, or fire-door color bleed rather than a real blue door.
        non_blue_count = red_count + green_count
        if non_blue_count > 0:
            if blue_count <= non_blue_count:
                return False
            if blue_count < max(min_blue_count + 2, non_blue_count + 2):
                return False
        return True

    def _semantic_green_exit_landmark_confirmed(
            self,
            landmark: dict[str, float],
            xy: tuple[float, float]) -> bool:
        blue_count, red_count, green_count = self._semantic_landmark_color_votes(landmark)
        total_color_votes = max(1, blue_count + red_count + green_count)
        if green_count < self._semantic_exit_min_green_observations:
            return False
        if green_count / float(total_color_votes) < self._semantic_exit_min_green_ratio:
            return False
        if green_count <= blue_count:
            return False
        if red_count > green_count:
            return False
        return self._exit_candidate_axis_valid(
            xy,
            'semantic_green_exit',
            clear_existing=False,
            require_front_wall_confirmation=False)

    def _semantic_green_exit_xy(self) -> tuple[float, float] | None:
        if not self._semantic_door_memory_enabled:
            return None
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return None
        best: tuple[float, float, int, tuple[float, float]] | None = None
        for landmark in self._semantic_door_landmarks:
            xy = (
                float(landmark.get('x', 0.0)),
                float(landmark.get('y', 0.0)),
            )
            if not self._semantic_green_exit_landmark_confirmed(landmark, xy):
                continue
            progress = self._axis_progress_xy(xy[0], xy[1])
            center_error = self._exit_center_y_error(xy)
            green_count = int(landmark.get('green_count', 0.0))
            score = (progress, -center_error, green_count)
            if best is None or score > (best[0], best[1], best[2]):
                best = (progress, -center_error, green_count, xy)
        if best is None:
            return None
        xy = best[3]
        self._last_valid_exit_xy = xy
        self._last_valid_exit_from_front_wall = False
        return xy

    def _semantic_blue_door_confirmed(self, door: DoorInfo) -> bool:
        if not self._semantic_door_memory_enabled:
            return True
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        return self._semantic_blue_xy_confirmed(xy)

    def _semantic_blue_xy_confirmed(self, xy: tuple[float, float]) -> bool:
        if not self._semantic_door_memory_enabled:
            return False
        landmark = self._find_semantic_door_landmark(xy)
        if landmark is None:
            return False
        return self._semantic_blue_landmark_confirmed(landmark, xy)

    def _semantic_blue_xy_opened(self, xy: tuple[float, float]) -> bool:
        if not self._semantic_door_memory_enabled:
            return False
        landmark = self._find_semantic_door_landmark(
            xy,
            merge_dist=self._semantic_door_merge_dist_m)
        return landmark is not None and landmark.get('opened', 0.0) >= 1.0

    def _semantic_blue_xy_abandoned(self, xy: tuple[float, float]) -> bool:
        if not self._semantic_door_memory_enabled:
            return False
        landmark = self._find_semantic_door_landmark(
            xy,
            merge_dist=self._semantic_door_merge_dist_m)
        return landmark is not None and landmark.get('abandoned', 0.0) >= 1.0

    def _semantic_blue_xy_suppressed(self, xy: tuple[float, float]) -> bool:
        if not self._semantic_door_memory_enabled:
            return False
        if self._semantic_blue_xy_opened(xy) or self._semantic_blue_xy_abandoned(xy):
            return True
        landmark = self._find_semantic_door_landmark(xy)
        if landmark is None:
            return False
        return not self._semantic_blue_landmark_confirmed(landmark, xy)

    def _mark_semantic_door_opened(self, door: DoorInfo):
        if not self._semantic_door_memory_enabled:
            return
        if not self._door_has_map_identity(door):
            return
        landmark = self._find_semantic_door_landmark(
            self._door_identity_xy(door),
            merge_dist=self._semantic_door_merge_dist_m)
        if landmark is not None:
            landmark['opened'] = 1.0

    def _mark_semantic_door_abandoned(self, door: DoorInfo):
        if not self._semantic_door_memory_enabled:
            return
        if not self._door_has_map_identity(door):
            return
        landmark = self._find_semantic_door_landmark(
            self._door_identity_xy(door),
            merge_dist=self._semantic_door_merge_dist_m)
        if landmark is not None:
            landmark['abandoned'] = 1.0

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
            self._semantic_door_landmarks.clear()
            self._observed_red_door_positions.clear()
            self._detected_blue_origin_by_id.clear()
            self._clear_locked_blue_anchor()
            self._post_open_reorient_pending = False
            self._post_open_reorient_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._mission_progress_x = self._current_mission_progress()
            self._exit_door = None
            self._last_valid_exit_xy = None
            self._last_valid_exit_from_front_wall = False
            self._last_provisional_exit_xy = None
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
            self._semantic_door_landmarks.clear()
            self._observed_red_door_positions.clear()
            self._mission_progress_x = self._current_mission_progress()
            self._post_open_reorient_pending = False
            self._post_open_reorient_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._exit_door = None
            self._last_valid_exit_xy = None
            self._last_valid_exit_from_front_wall = False
            self._last_provisional_exit_xy = None
            self._active_exit_goal = None
            self._nav_retry_count = 0
            self._transition(State.EXPLORING)

    def _on_exploring(self):
        if self._run_post_open_side_retreat():
            return
        if self._run_post_open_reorient():
            return
        if self._run_post_open_clearance_advance():
            return
        if self._run_explore_waypoint_scan():
            return

        # 개방 완료 + 실패 블랙리스트 제외한 파란 문 후보
        blue_candidates, raw_safe_doors, safe_doors = (
            self._target_subfsm.explore_blue_candidates())
        if safe_doors:
            self._explore_observation_wait_start_time = None
        elif self._run_explore_observation_wait():
            return
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
                and not self._exit_blocking_unopened_blue_doors()
                and self._past_exit_line()):
            self.get_logger().info('비상구 라인 통과 확인. 미션 완료로 전환합니다.')
            self._transition(State.MISSION_COMPLETE)
            return
        if self._final_scan_start_time is not None:
            if self._run_final_scan_before_exit():
                return
            self._force_final_scan_before_exit = False
            if not safe_doors:
                unopened_observed = self._exit_blocking_unopened_blue_doors()
                if unopened_observed:
                    if self._target_exit_blocking_unopened_blue(
                            unopened_observed, 'final exit scan'):
                        return
                    self._hold_exit_for_unopened_blue_doors(unopened_observed)
                    self._rotate_to_scan()
                    return

                opened_ready_count = self._opened_exit_ready_count()
                if opened_ready_count < self._min_opened_doors_before_exit:
                    self._explore_start_time = self.get_clock().now()
                    self._rotate_to_scan()
                    return

                self._maybe_set_front_wall_exit('final exit scan')
                if (self._valid_detected_exit()
                        or self._last_valid_exit_xy is not None
                        or self._provisional_exit_xy_for_blue_suppression() is not None
                        or self._semantic_green_exit_xy() is not None
                        or self._allow_fallback_exit_goal):
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
            if self._opened_exit_ready_count() >= self._min_opened_doors_before_exit:
                self._maybe_set_front_wall_exit('no blue candidate scan')

            if ((self._valid_detected_exit()
                    or self._last_valid_exit_xy is not None
                    or self._provisional_exit_xy_for_blue_suppression() is not None
                    or self._semantic_green_exit_xy() is not None
                        or self._allow_fallback_exit_goal)
                    and self._opened_exit_ready_count() >= self._min_opened_doors_before_exit
                    and not self._exit_blocking_unopened_blue_doors()
                    and self._exit_scan_allowed_by_exploration()):
                if self._run_final_scan_before_exit():
                    return
                self._force_final_scan_before_exit = False

                unopened_observed = self._exit_blocking_unopened_blue_doors()
                if unopened_observed:
                    if self._target_exit_blocking_unopened_blue(
                            unopened_observed, 'final exit scan'):
                        return
                    self._hold_exit_for_unopened_blue_doors(unopened_observed)
                    self._rotate_to_scan()
                    return

                self._maybe_set_front_wall_exit('observed exit scan')
                if (self._valid_detected_exit()
                        or self._last_valid_exit_xy is not None
                        or self._provisional_exit_xy_for_blue_suppression() is not None
                        or self._semantic_green_exit_xy() is not None
                        or self._allow_fallback_exit_goal):
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
                    if not self._exit_scan_allowed_by_exploration():
                        self._force_final_scan_before_exit = False
                        self._explore_start_time = self.get_clock().now()
                        self.get_logger().info(
                            'Exit scan deferred: continuing observation-based '
                            'forward exploration before leaving.')
                        if self._send_explore_waypoint():
                            return
                        self._rotate_to_scan()
                        return

                    opened_count = self._resolved_blue_door_count()
                    if self._run_final_scan_before_exit():
                        return
                    self._force_final_scan_before_exit = False

                    unopened_observed = self._exit_blocking_unopened_blue_doors()
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

                    self._maybe_set_front_wall_exit('explore timeout')
                    if (not self._valid_detected_exit()
                            and self._last_valid_exit_xy is None
                            and self._provisional_exit_xy_for_blue_suppression() is None
                            and self._semantic_green_exit_xy() is None
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
                if not self._exit_scan_allowed_by_exploration():
                    self._force_final_scan_before_exit = False
                    if self._send_explore_waypoint():
                        return
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
            if not self._nav_start_pose_recovery_active:
                self._rotate_to_scan()
            return

        # 파란 문이 관측되면 탐색 waypoint를 더 보내지 않고 가장 가까운 문을
        pending_retry_doors = [
            d for d in safe_doors
            if self._observed_blue_retry_pending(d)
        ]
        if pending_retry_doors:
            safe_doors = pending_retry_doors
            self.get_logger().warn(
                f'접근/개방 실패 예산이 남은 파란 문 {len(safe_doors)}개를 '
                '새 후보보다 우선 재시도합니다.',
                throttle_duration_sec=3.0)
        else:
            forward_safe_doors = self._filter_forward_doors(safe_doors)
            if forward_safe_doors:
                safe_doors = forward_safe_doors
            else:
                stable_revisit = [
                    d for d in safe_doors
                    if d.door_id.startswith('observed_blue_')
                ]
                if stable_revisit:
                    safe_doors = stable_revisit
                    self.get_logger().warn(
                        '앞쪽 후보는 없지만 아직 안 열린 안정 파란 문 관측이 남아 있어 '
                        '뒤쪽/측면 후보를 재방문합니다.')
                else:
                    self.get_logger().info(
                        '현재 위치보다 앞쪽의 새 파란 문 후보가 없어 waypoint 탐색을 계속합니다.',
                        throttle_duration_sec=2.0)
                    if self._send_explore_waypoint():
                        return
                    self._rotate_to_scan()
                    return
        # 즉시 Nav2 목표로 고정한다. 목표까지의 장애물 회피는 Nav2 costmap과
        # planner/controller가 담당한다.
        selected_door = self._nearest_forward_blue_door(safe_doors)
        if selected_door is None:
            self._rotate_to_scan()
            return
        self.target_door = self._door_with_axis_aligned_approach(selected_door)
        self._set_locked_blue_anchor(self.target_door)

        if self._run_pre_nav_scan(self.target_door):
            return
        invalid_reason = self._blue_target_invalid_after_pre_nav_scan(
            self.target_door)
        if invalid_reason:
            rejected = self.target_door
            rejected_id = rejected.door_id if rejected is not None else '(unknown)'
            if rejected is not None:
                abandoned = self._record_observed_blue_door_failure(
                    rejected, failure_kind='approach')
                if abandoned:
                    self._add_door_id_keys(self._failed_door_ids, rejected_id)
            self._clear_locked_blue_anchor()
            self.target_door = None
            self._nav_retry_count = 0
            self._explore_start_time = self.get_clock().now()
            self.get_logger().warn(
                f'사전 스캔 후 파란 문 목표 {rejected_id} 제외: '
                f'{invalid_reason}. 새 관측 후보를 다시 선택합니다.')
            return
        if self._defer_far_target_door(self.target_door):
            return

        self.get_logger().info(
            f'파란 문 발견 즉시 목표 고정: {self.target_door.door_id}  '
            f'개방 완료: {self._resolved_blue_door_count()}개  '
            f'재시도: {self._nav_retry_count}/{self._max_nav_retries}')
        self._nav_done   = False
        self._nav_failed = False
        self._transition(State.NAVIGATING)
        self.target_door_pub.publish(self.target_door)

    def _recover_blue_nav_start_if_needed(self) -> bool:
        if (self.target_door is None
                or self.target_door.door_color != 'blue'
                or self._door_open_align_start_time is not None
                or self._door_open_settle_start_time is not None):
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False

        x, y, _yaw = pose
        lateral = self._axis_lateral_xy(x, y)
        lateral_error = lateral - self._explore_center_y
        target_abs = max(0.05, self._nav_start_center_recovery_target_abs_y_m)
        needs_recovery = (
            abs(lateral_error) > self._nav_start_max_abs_y_m
            or (
                self._nav_start_pose_recovery_active
                and abs(lateral_error) > target_abs))
        if needs_recovery:
            self._nav_done = False
            self._nav_failed = False
            self._request_navigation_cancel_for_manual_control(
                'blue door start center recovery')
            if self._run_nav_start_center_recovery(pose, lateral_error):
                return True
            return False

        if self._nav_start_pose_recovery_active:
            self._nav_start_pose_recovery_active = False
            self._nav_done = False
            self._nav_failed = False
            self._nav_start_time = self.get_clock().now()
            self.get_logger().warn(
                f'Blue door Nav2 start pose recovered near center: '
                f'lateral={lateral:.2f}m, target={target_abs:.2f}m. '
                f'Resending {self.target_door.door_id}.')
            self.target_door_pub.publish(self.target_door)
            return True
        return False

    def _on_navigating(self):
        if (self.target_door is not None
                and self.target_door.door_color == 'blue'
                and not self._nav_done
                and self._door_open_align_start_time is None
                and self._door_open_settle_start_time is None):
            invalid_reason = self._blue_target_exit_tail_invalid_reason(
                self.target_door)
            if invalid_reason:
                target_id = self.target_door.door_id
                self._record_observed_blue_door_failure(
                    self.target_door, failure_kind='approach')
                self.cmd_vel_pub.publish(Twist())
                self._clear_locked_blue_anchor()
                self.target_door = None
                self._nav_retry_count = 0
                self._door_opening_in_progress = False
                self.get_logger().warn(
                    f'주행 중 파란 문 목표 {target_id} 제외: '
                    f'{invalid_reason}. 새 관측 후보를 다시 선택합니다.')
                self._transition(State.EXPLORING)
                return

        if self._recover_blue_nav_start_if_needed():
            return

        if self._nav_start_time is not None:
            elapsed = (
                self.get_clock().now() - self._nav_start_time
            ).nanoseconds / 1e9
            if elapsed > self._nav_timeout_sec:
                self.get_logger().warn(f'Navigation timeout ({elapsed:.1f}s).')
                self._handle_nav_failure(reason='timeout')
                return

        if not self._nav_done and not self._nav_failed:
            if self._door_nav_goal_close_enough_for_fine_alignment():
                self._request_navigation_cancel_for_manual_control(
                    'door fine alignment')
                self.cmd_vel_pub.publish(Twist())
                self._nav_done = True
            elif self._door_nav_wall_stuck():
                self._handle_nav_failure(reason='door_nav_stuck')
                return
            else:
                self._republish_active_navigation_target()

        if self._nav_done:
            self._nav_done = False
            settling_after_ready_alignment = (
                self.target_door is not None
                and self.target_door.door_color == 'blue'
                and self._door_open_settle_start_time is not None)

            if settling_after_ready_alignment:
                if self._settle_before_opening_door():
                    self._nav_done = True
                    return
            else:
                if (self.target_door is not None
                        and self.target_door.door_color == 'blue'
                        and self._door_open_align_start_time is None
                        and not self._door_nav_goal_within_fine_alignment_window()
                        and not self._approach_subfsm.can_continue_with_fine_alignment(
                            'nav_success')):
                    dist = self._door_distance_from_robot(self.target_door)
                    error = self._door_open_pose_error(self.target_door)
                    lateral = error[3] if error is not None else float('nan')
                    self.get_logger().warn(
                        f'Nav2 success ignored because robot is not near the door: '
                        f'{self.target_door.door_id}, '
                        f'dist={dist if dist is not None else -1.0:.2f}m, '
                        f'lateral={lateral:.2f}m. Treating it as a navigation retry.')
                    self._handle_nav_failure(reason='nav_success_far_from_door')
                    return
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
            if (self.target_door is not None
                    and self.target_door.door_color == 'blue'
                    and self._blue_target_needs_fresh_open_confirmation(self.target_door)
                    and not self._blue_target_has_safe_open_memory(self.target_door)
                    and not self._observed_blue_target_has_stable_open_evidence(self.target_door)
                    and not self._door_target_has_fresh_blue_evidence(self.target_door)):
                target_id = self.target_door.door_id
                abandoned = self._record_observed_blue_door_failure(
                    self.target_door, failure_kind='approach')
                if abandoned:
                    self.get_logger().warn(
                        f'문 열기 직전 파란문 재확인 실패 누적 한도 도달: '
                        f'{target_id}. 해당 후보를 포기합니다.')
                self.cmd_vel_pub.publish(Twist())
                self._clear_locked_blue_anchor()
                self.target_door = None
                self._nav_retry_count = 0
                self._door_opening_in_progress = False
                self.get_logger().warn(
                    f'문 열기 직전 파란문 재확인 실패: {target_id}. '
                    '가짜/소실 후보로 보고 다음 관찰을 계속합니다.')
                self._transition(State.EXPLORING)
                return
            if (self.target_door is not None
                    and self.target_door.door_color == 'blue'
                    and self._door_target_has_recent_non_blue_conflict(self.target_door)
                    and not self._observed_blue_target_has_stable_open_evidence(self.target_door)):
                target_id = self.target_door.door_id
                self._record_observed_blue_door_failure(
                    self.target_door, failure_kind='approach')
                self.cmd_vel_pub.publish(Twist())
                self._clear_locked_blue_anchor()
                self.target_door = None
                self._nav_retry_count = 0
                self._door_opening_in_progress = False
                self.get_logger().warn(
                    f'문 열기 직전 non-blue 충돌 관측: {target_id}. '
                    '빨간/출구 오탐 가능성이 있어 후보를 버리고 재스캔합니다.')
                self._transition(State.EXPLORING)
                return
            self._door_opening_in_progress = False
            self._transition(State.OPENING_DOOR)
        elif self._nav_failed:
            self._nav_failed = False
            if self._approach_subfsm.can_continue_with_fine_alignment(
                    'nav2_failure'):
                self.cmd_vel_pub.publish(Twist())
                self._nav_done = True
                return
            self._handle_nav_failure(reason='nav2_failure')

    def _republish_active_navigation_target(self):
        if self.target_door is None:
            return
        period = max(0.0, self._nav_target_republish_sec)
        if period <= 0.0:
            return
        now = self.get_clock().now()
        if self._last_target_republish_time is not None:
            elapsed = (now - self._last_target_republish_time).nanoseconds / 1e9
            if elapsed < period:
                return
        self._last_target_republish_time = now
        self.target_door_pub.publish(self.target_door)
        self.get_logger().info(
            f'Republishing active navigation target: {self.target_door.door_id}',
            throttle_duration_sec=3.0)

    def _request_navigation_cancel_for_manual_control(self, reason: str):
        msg = Bool()
        msg.data = True
        self.nav_cancel_pub.publish(msg)
        self.get_logger().info(
            f'Nav2 goal cancellation requested before {reason}.',
            throttle_duration_sec=1.0)

    def _door_fine_handoff_limits(self) -> tuple[float, float]:
        handoff_dist = min(
            self._door_open_fine_control_max_dist_m,
            max(self._door_open_ready_max_dist_m + 0.18, 0.38))
        handoff_lateral = min(
            self._door_open_fine_lateral_tolerance_m,
            max(self._door_open_ready_lateral_tolerance_m + 0.08, 0.22))
        return handoff_dist, handoff_lateral

    def _door_nav_goal_close_enough_for_fine_alignment(self) -> bool:
        if self.target_door is None or self.target_door.door_color != 'blue':
            return False
        error = self._door_open_pose_error(self.target_door)
        if error is None:
            return False
        dist, yaw_error, _forward_error, lateral_error = error
        yaw_abs = abs(yaw_error)
        yaw_ready = yaw_abs <= max(
            self._door_open_ready_yaw_tolerance * 3.0,
            math.radians(45.0))
        very_close = dist <= max(self._door_open_ready_max_dist_m + 0.04, 0.28)
        handoff_dist, handoff_lateral = self._door_fine_handoff_limits()
        if (dist <= handoff_dist
                and abs(lateral_error) <= handoff_lateral
                and (yaw_ready or very_close)):
            self.get_logger().info(
                f'Nav2 goal is close enough for fine door alignment: '
                f'{self.target_door.door_id}, dist={dist:.2f}m/'
                f'{handoff_dist:.2f}m, lateral={lateral_error:.2f}m/'
                f'{handoff_lateral:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg.')
            return True
        if dist <= handoff_dist and abs(lateral_error) <= handoff_lateral:
            self.get_logger().info(
                f'Nav2 door goal position is near, but yaw is not ready yet: '
                f'{self.target_door.door_id}, dist={dist:.2f}m/'
                f'{handoff_dist:.2f}m, lateral={lateral_error:.2f}m/'
                f'{handoff_lateral:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg.')
        return False

    def _door_nav_goal_within_fine_alignment_window(self) -> bool:
        if self.target_door is None or self.target_door.door_color != 'blue':
            return False
        error = self._door_open_pose_error(self.target_door)
        if error is None:
            return False
        dist, _yaw_error, _forward_error, lateral_error = error
        handoff_dist, handoff_lateral = self._door_fine_handoff_limits()
        return (
            dist <= handoff_dist
            and abs(lateral_error) <= handoff_lateral)

    def _reset_door_nav_stuck_watch(self):
        now_wall = time.monotonic()
        self._door_nav_wall_start_time = now_wall
        self._door_nav_last_progress_wall_time = now_wall
        pose = self._current_map_pose()
        self._door_nav_last_progress = (
            self._axis_progress_xy(pose[0], pose[1]) if pose is not None else None)

    def _clear_door_nav_stuck_watch(self):
        self._door_nav_wall_start_time = None
        self._door_nav_last_progress = None
        self._door_nav_last_progress_wall_time = None

    def _door_nav_wall_stuck(self) -> bool:
        if not self._door_nav_stuck_watch_enabled:
            return False
        timeout = max(0.0, self._door_nav_stuck_timeout_sec)
        if timeout <= 0.0 or self.target_door is None:
            return False
        if self.target_door.door_color != 'blue':
            return False

        dist = self._door_distance_from_robot(self.target_door)
        min_goal_dist = max(0.0, self._door_nav_stuck_min_goal_dist_m)
        if dist is not None and dist <= min_goal_dist:
            self._reset_door_nav_stuck_watch()
            return False

        now_wall = time.monotonic()
        if self._door_nav_wall_start_time is None:
            self._reset_door_nav_stuck_watch()
            return False
        start_wall = self._door_nav_wall_start_time

        pose = self._current_map_pose()
        if pose is None:
            last_wall = self._door_nav_last_progress_wall_time
            if last_wall is None:
                self._door_nav_last_progress_wall_time = now_wall
                return False
            last_wall = max(last_wall, start_wall)
            if last_wall != self._door_nav_last_progress_wall_time:
                self._door_nav_last_progress_wall_time = last_wall
            if now_wall - last_wall > timeout:
                self.get_logger().warn(
                    f'Door navigation stuck: no map pose for {now_wall - last_wall:.1f}s.')
                return True
            return False

        progress = self._axis_progress_xy(pose[0], pose[1])
        min_progress = max(0.01, self._door_nav_stuck_min_progress_m)
        last_progress = self._door_nav_last_progress
        if last_progress is None or progress > last_progress + min_progress:
            self._door_nav_last_progress = progress
            self._door_nav_last_progress_wall_time = now_wall
            return False

        last_wall = self._door_nav_last_progress_wall_time
        if last_wall is None:
            self._door_nav_last_progress_wall_time = now_wall
            return False
        last_wall = max(last_wall, start_wall)
        if last_wall != self._door_nav_last_progress_wall_time:
            self._door_nav_last_progress_wall_time = last_wall

        if now_wall - last_wall > timeout:
            self.get_logger().warn(
                f'Door navigation stuck: target={self.target_door.door_id}, '
                f'progress={progress:.2f}, no_forward_progress={now_wall - last_wall:.1f}s, '
                f'dist={dist if dist is not None else -1.0:.2f}m.')
            return True
        return False

    def _on_exploring_navigating(self):
        if self._interrupt_explore_waypoint_for_blue_door():
            return

        if self._nav_done:
            self._nav_done = False
            self._explore_nav_failure_count = 0
            self._recent_failed_explore_goals.clear()
            self.get_logger().info('탐색 waypoint 도착. LiDAR+camera 재스캔을 계속합니다.')
            self._alternate_explore_lane_after_success()
            self._start_explore_waypoint_scan()
            self._transition(State.EXPLORING)
            return
        if self._nav_failed:
            self._nav_failed = False
            self._remember_failed_explore_goal()
            if self._handle_explore_nav_failure_limit():
                return
            self._flip_explore_lane_after_failure()
            self.get_logger().warn('탐색 waypoint 이동 실패. 반대 차선 우선으로 재스캔합니다.')
            self._transition(State.EXPLORING)
            return

        if self._explore_waypoint_wall_stuck():
            if self._handle_explore_nav_failure_limit():
                return
            self._remember_failed_explore_goal()
            self._flip_explore_lane_after_failure()
            self._nav_done = False
            self._nav_failed = False
            self.get_logger().warn(
                'Explore waypoint stuck with no forward progress; '
                'replanning to an alternate observed lane.')
            self._transition(State.EXPLORING)
            return

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

    def _reset_explore_nav_stuck_watch(self):
        now_wall = time.monotonic()
        self._explore_nav_wall_start_time = now_wall
        self._explore_nav_last_progress_wall_time = now_wall
        pose = self._current_map_pose()
        self._explore_nav_last_progress = (
            self._axis_progress_xy(pose[0], pose[1]) if pose is not None else None)
        self._explore_nav_last_yaw = pose[2] if pose is not None else None
        if pose is not None and self._explore_nav_goal_xy is not None:
            gx, gy = self._explore_nav_goal_xy
            self._explore_nav_last_goal_dist = math.hypot(gx - pose[0], gy - pose[1])
        else:
            self._explore_nav_last_goal_dist = None

    def _clear_explore_nav_stuck_watch(self):
        self._explore_nav_wall_start_time = None
        self._explore_nav_last_progress = None
        self._explore_nav_last_goal_dist = None
        self._explore_nav_last_yaw = None
        self._explore_nav_last_progress_wall_time = None

    def _explore_waypoint_wall_stuck(self) -> bool:
        timeout = max(0.0, self._explore_nav_stuck_wall_timeout_sec)
        if timeout <= 0.0:
            return False

        now_wall = time.monotonic()
        if self._explore_nav_wall_start_time is None:
            self._reset_explore_nav_stuck_watch()
            return False

        pose = self._current_map_pose()
        if pose is None:
            last_wall = self._explore_nav_last_progress_wall_time
            if last_wall is None:
                self._explore_nav_last_progress_wall_time = now_wall
                return False
            if now_wall - last_wall > timeout:
                self.get_logger().warn(
                    f'Explore waypoint stuck: no map pose for {now_wall - last_wall:.1f}s.')
                return True
            return False

        progress = self._axis_progress_xy(pose[0], pose[1])
        min_progress = max(0.01, self._explore_nav_stuck_min_progress_m)
        last_progress = self._explore_nav_last_progress
        goal_dist = None
        if self._explore_nav_goal_xy is not None:
            gx, gy = self._explore_nav_goal_xy
            goal_dist = math.hypot(gx - pose[0], gy - pose[1])
            near_goal = max(0.65, self._explore_obstacle_stop_m + 0.10)
            if goal_dist <= near_goal:
                self._explore_nav_last_progress = progress
                self._explore_nav_last_goal_dist = goal_dist
                self._explore_nav_last_yaw = pose[2]
                self._explore_nav_last_progress_wall_time = now_wall
                return False
        last_goal_dist = self._explore_nav_last_goal_dist
        goal_dist_improved = (
            goal_dist is not None
            and (last_goal_dist is None or goal_dist < last_goal_dist - min_progress))
        if (last_progress is None
                or progress > last_progress + min_progress
                or goal_dist_improved):
            self._explore_nav_last_progress = progress
            self._explore_nav_last_goal_dist = goal_dist
            self._explore_nav_last_yaw = pose[2]
            self._explore_nav_last_progress_wall_time = now_wall
            return False

        last_wall = self._explore_nav_last_progress_wall_time
        if last_wall is None:
            self._explore_nav_last_progress_wall_time = now_wall
            return False

        if now_wall - last_wall > timeout:
            goal_progress = self._explore_nav_goal_progress
            goal_dist_text = (
                f', goal_dist={goal_dist:.2f}m'
                if goal_dist is not None else '')
            self.get_logger().warn(
                f'Explore waypoint stuck: progress={progress:.2f}, '
                f'goal={goal_progress:.2f}{goal_dist_text}, '
                f'no_forward_progress={now_wall - last_wall:.1f}s.')
            return True
        return False

    def _is_blue_door_ready_to_interrupt_explore(self, door: DoorInfo) -> bool:
        min_conf = max(0.0, self._explore_interrupt_min_confidence)
        max_dist = max(0.0, self._explore_interrupt_max_distance_m)
        confidence = float(door.confidence)
        dist = self._door_distance_from_robot(door)

        xy = self._door_identity_xy(door)
        if self._is_blue_xy_too_far_ahead_for_target(xy, door.door_id):
            return False
        if (
                self._is_blue_xy_at_or_after_observed_exit(xy)
                or self._is_blue_xy_near_detected_exit(xy)):
            return False
        if self._blue_target_in_strict_exit_tail_without_persistent_memory(door):
            return False

        if self._is_pre_exit_blue_recheck_candidate(door):
            relaxed_max_dist = max(max_dist, self._explore_interrupt_max_distance_m)
            if relaxed_max_dist <= 0.0 or dist is None or dist <= relaxed_max_dist:
                self.get_logger().info(
                    f'Pre-exit blue candidate {door.door_id} kept for recheck '
                    f'before exit (conf={confidence:.2f}, '
                    f'dist={dist if dist is not None else -1.0:.1f}m).',
                    throttle_duration_sec=3.0)
                return True

        if not door.door_id.startswith('observed_blue_'):
            exit_sensitive = (
                self._is_blue_xy_in_pre_exit_zone(xy)
                or self._is_blue_xy_near_detected_exit(xy))
            direct_min_conf = max(
                min_conf,
                self._blue_target_immediate_min_confidence)
            if confidence < direct_min_conf:
                self.get_logger().info(
                    f'Blue candidate {door.door_id} is waiting for stronger '
                    f'direct evidence before interrupting exploration '
                    f'(conf={confidence:.2f}/{direct_min_conf:.2f}).',
                    throttle_duration_sec=3.0)
                return False

        if min_conf <= 0.0 or confidence >= min_conf:
            if max_dist <= 0.0 or dist is None or dist <= max_dist:
                return True
        if (max_dist > 0.0
                and dist is not None
                and dist <= max_dist
                and door.door_id.startswith('observed_blue_')):
            return True

        self.get_logger().info(
            f'Blue candidate {door.door_id} observed while moving to an '
            f'explore waypoint, but not interrupting yet '
            f'(conf={confidence:.2f}/{min_conf:.2f}, '
            f'dist={dist if dist is not None else -1.0:.1f}m/'
            f'{max_dist:.1f}m).',
            throttle_duration_sec=3.0)
        return False

    def _interrupt_explore_waypoint_for_blue_door(
            self, exit_context: bool = False) -> bool:
        """Switch from a generic explore waypoint to a confirmed blue door immediately."""
        blue_candidates = [
            d for d in self.detected_doors
            if d.door_color == 'blue'
            and not self._is_door_opened_for_observation(d)
            and not self._is_door_abandoned_for_observation(d)
            and not self._is_door_failed_for_observation(d)
            and self._is_door_recent_observation(d)
            and not self._is_door_physically_opened(d)
            and not self._is_door_observation_suppressed(d)
            and not self._is_door_physically_abandoned(d)
            and not self._is_blue_candidate_blocked_by_red(d)
            and self._is_blue_door_at_valid_wall_position(d)
            and self._door_has_map_identity(d)
        ]
        if self._use_observed_blue_clusters_as_targets:
            blue_candidates = self._merge_blue_target_lists(
                blue_candidates, self._stable_unopened_observed_blue_targets())
        blue_candidates = [
            d for d in blue_candidates
            if self._is_blue_door_ready_to_interrupt_explore(d)
        ]
        if exit_context:
            blue_candidates = [
                d for d in blue_candidates
                if self._is_blue_door_before_current_exit(d)
            ]
            min_exit_conf = max(0.0, self._exit_interrupt_blue_min_confidence)
            if min_exit_conf > 0.0:
                kept = []
                for candidate in blue_candidates:
                    confidence = float(candidate.confidence)
                    if (
                            confidence >= min_exit_conf
                            or self._door_has_persistent_observed_blue_exit_memory(
                                candidate)):
                        kept.append(candidate)
                    else:
                        self.get_logger().info(
                            f'Blue candidate {candidate.door_id} ignored during exit: '
                            f'confidence {confidence:.2f} < {min_exit_conf:.2f}.',
                            throttle_duration_sec=3.0)
                blue_candidates = kept
        blue_candidates = self._filter_forward_doors(blue_candidates)
        if not blue_candidates:
            return False

        candidate_text: list[str] = []
        for candidate in blue_candidates[:6]:
            xy = self._door_identity_xy(candidate)
            progress = self._axis_progress_xy(xy[0], xy[1])
            lateral = self._axis_lateral_xy(xy[0], xy[1])
            dist = self._door_distance_from_robot(candidate)
            dist_text = f'{dist:.1f}m' if dist is not None else 'n/a'
            candidate_text.append(
                f'{candidate.door_id}:p={progress:.2f},lat={lateral:.2f},'
                f'd={dist_text},conf={float(candidate.confidence):.2f}')
        if candidate_text:
            self.get_logger().info(
                '탐색 중 파란 문 후보 평가: ' + '; '.join(candidate_text))

        selected = self._nearest_forward_blue_door(blue_candidates)
        if selected is None:
            return False

        self.target_door = self._door_with_axis_aligned_approach(selected)
        self._set_locked_blue_anchor(self.target_door)
        self.get_logger().info(
            f'탐색 waypoint 이동 중 파란 문 확인: {self.target_door.door_id}. '
            '현재 waypoint를 취소하고 문 접근으로 즉시 전환합니다.')
        self._nav_done = False
        self._nav_failed = False
        self._explore_start_time = self.get_clock().now()
        self._transition(State.NAVIGATING)
        self.target_door_pub.publish(self.target_door)
        return True
    def _handle_explore_nav_failure_limit(self) -> bool:
        self._explore_nav_failure_count += 1
        if (self._opened_exit_ready_count() >= self._min_opened_doors_before_exit
                and not self._exit_blocking_unopened_blue_doors()
                and (self._exit_door is not None or self._last_valid_exit_xy is not None)
                and self._explore_nav_failure_count
                >= self._max_explore_nav_failures_before_exit):
            self.get_logger().warn(
                f'탐색 waypoint 실패 {self._explore_nav_failure_count}회 누적. '
                '비상구 전 최종 파란 문 스캔으로 복귀합니다.')
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
                self._add_door_id_keys(self._failed_door_ids, self.target_door.door_id)
                if self.target_door.door_color == 'blue':
                    failed_door = self.target_door
                    failed_id = failed_door.door_id
                    abandoned = self._record_observed_blue_door_failure(
                        failed_door, failure_kind='approach')
                    self._clear_locked_blue_anchor()
                    self.target_door = None
                    self._nav_retry_count = 0
                    if abandoned:
                        self.get_logger().error(
                            f'접근 포기: {failed_id}. 제한 횟수까지 문 앞 접근에 실패해 '
                            '해당 물리 문 후보를 미션에서 제외하고 탐색을 계속합니다.')
                    else:
                        self.get_logger().warn(
                            f'접근 재스캔: {failed_id}. 이번 Nav2 접근 묶음은 실패했지만 '
                            '물리 문 후보는 아직 포기하지 않고, 새 관측으로 다시 시도합니다.')
                    self._transition(State.EXPLORING)
                    return
                self._clear_locked_blue_anchor()
                self.target_door = None
            self.get_logger().error('Max retries 초과. Emergency stop.')
            self._transition(State.EMERGENCY_STOP)
            return

        if self.target_door is not None and self.target_door.door_color == 'blue':
            retry_target = copy.deepcopy(self.target_door)
            retry_target.confidence = -max(1.0e-3, abs(float(retry_target.confidence)))
            self.get_logger().info(
                f'같은 파란 문 재접근 유지: {retry_target.door_id}. '
                '열기 전까지 다른 후보로 전환하지 않고 Nav2 목표를 다시 보냅니다.')
            self._nav_done = False
            self._nav_failed = False
            self._transition(State.NAVIGATING)
            self.target_door_pub.publish(retry_target)
            return

        self.get_logger().info('재탐색/자세 보정 후 같은 문 재시도 허용')
        self._clear_locked_blue_anchor()
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
        self._clear_locked_blue_anchor()
        self.target_door = None
        self._post_open_side_retreat_pending = self._post_open_side_retreat_enabled
        self._post_open_side_retreat_start_time = None
        self._post_open_side_retreat_start_lateral = None
        self._post_open_reorient_pending = self._post_open_reorient_enabled
        self._post_open_reorient_start_time = None
        self._post_open_advance_pending = self._post_open_clearance_advance_enabled
        self._post_open_advance_start_time = None
        self._post_open_advance_start_progress = None
        self.get_logger().info(
            f'다음 파란 문 탐색 시작 (개방 완료: {self._resolved_blue_door_count()}개)')
        self._transition(State.EXPLORING)

    def _run_post_open_side_retreat(self) -> bool:
        if not self._post_open_side_retreat_pending:
            return False
        if (not self._post_open_side_retreat_enabled
                or self._post_open_side_retreat_m <= 0.0):
            self._finish_post_open_side_retreat('disabled')
            return False

        pose = self._current_map_pose()
        if pose is None:
            self.cmd_vel_pub.publish(Twist())
            return True

        now = self.get_clock().now()
        lateral = self._axis_lateral_xy(pose[0], pose[1]) - self._explore_center_y
        abs_lateral = abs(lateral)
        target_abs = max(0.05, self._post_open_side_retreat_target_abs_y_m)
        if abs_lateral <= target_abs:
            self._finish_post_open_side_retreat(
                f'lateral={abs_lateral:.2f}m <= target={target_abs:.2f}m')
            return False

        if self._post_open_side_retreat_start_time is None:
            self._post_open_side_retreat_start_time = now
            self._post_open_side_retreat_start_lateral = lateral
            self.get_logger().info(
                f'Post-open side retreat started: lateral={lateral:.2f}m, '
                f'target_abs={target_abs:.2f}m.')

        start_lateral = (
            self._post_open_side_retreat_start_lateral
            if self._post_open_side_retreat_start_lateral is not None
            else lateral)
        moved_toward_center = max(0.0, abs(start_lateral) - abs_lateral)
        if moved_toward_center >= self._post_open_side_retreat_m:
            self._finish_post_open_side_retreat(
                f'moved_toward_center={moved_toward_center:.2f}m')
            return False

        elapsed = (
            now - self._post_open_side_retreat_start_time).nanoseconds / 1e9
        if elapsed > max(0.0, self._post_open_side_retreat_timeout_sec):
            self._finish_post_open_side_retreat(
                f'timeout after {elapsed:.1f}s, lateral={abs_lateral:.2f}m')
            return False

        side = 1.0 if lateral > 0.0 else -1.0
        center_facing_yaw = self._mission_forward_yaw() - side * math.pi / 2.0
        yaw_error = self._normalize_angle(center_facing_yaw - pose[2])
        twist = Twist()
        if abs(yaw_error) > self._post_open_side_retreat_yaw_tolerance:
            angular = math.copysign(
                abs(self._post_open_reorient_angular_vel), yaw_error)
            twist.angular.z = angular
        else:
            # Reverse is blocked by cmd_vel_safety; turn inward and drive forward.
            twist.linear.x = abs(self._post_open_side_retreat_linear_vel)
            angular_limit = abs(self._post_open_clearance_angular_vel_limit)
            twist.angular.z = max(
                -angular_limit,
                min(angular_limit, self._post_open_clearance_heading_kp * yaw_error))
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'Post-open side retreat: lateral={lateral:.2f}m, '
            f'moved={moved_toward_center:.2f}/{self._post_open_side_retreat_m:.2f}m, '
            f'yaw_error={math.degrees(yaw_error):.0f}deg, '
            f'cmd=({twist.linear.x:.2f}, {twist.angular.z:.2f})',
            throttle_duration_sec=1.0)
        return True

    def _finish_post_open_side_retreat(self, reason: str):
        self._post_open_side_retreat_pending = False
        self._post_open_side_retreat_start_time = None
        self._post_open_side_retreat_start_lateral = None
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info(f'Post-open side retreat complete: {reason}')

    def _run_post_open_reorient(self) -> bool:
        if not self._post_open_reorient_pending:
            return False
        if not self._post_open_reorient_enabled:
            self._post_open_reorient_pending = False
            self._post_open_reorient_start_time = None
            return False

        pose = self._current_map_pose()
        if pose is None:
            self.cmd_vel_pub.publish(Twist())
            return True

        now = self.get_clock().now()
        if self._post_open_reorient_start_time is None:
            self._post_open_reorient_start_time = now
            self.get_logger().info(
                '문 개방 후 복도 진행 방향으로 제자리 회전 정렬을 시작합니다.')

        yaw_error = self._normalize_angle(self._mission_forward_yaw() - pose[2])
        if abs(yaw_error) <= self._post_open_reorient_yaw_tolerance:
            self._post_open_reorient_pending = False
            self._post_open_reorient_start_time = None
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().info(
                f'문 개방 후 진행 방향 정렬 완료: yaw_error='
                f'{math.degrees(yaw_error):.0f}deg')
            return False

        elapsed = (now - self._post_open_reorient_start_time).nanoseconds / 1e9
        if elapsed > max(0.0, self._post_open_reorient_timeout_sec):
            self._post_open_reorient_pending = False
            self._post_open_reorient_start_time = None
            self.cmd_vel_pub.publish(Twist())
            self.get_logger().warn(
                f'문 개방 후 진행 방향 정렬 timeout: yaw_error='
                f'{math.degrees(yaw_error):.0f}deg. Nav2 재계획으로 이어갑니다.')
            return False

        twist = Twist()
        angular = math.copysign(
            abs(self._post_open_reorient_angular_vel),
            yaw_error)
        if abs(yaw_error) < 0.45:
            angular = math.copysign(
                max(0.14, abs(self._post_open_reorient_angular_vel) * 0.45),
                yaw_error)
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'문 개방 후 진행 방향 정렬 중: yaw_error='
            f'{math.degrees(yaw_error):.0f}deg, cmd={twist.angular.z:.2f}',
            throttle_duration_sec=1.5)
        return True

    def _run_post_open_clearance_advance(self) -> bool:
        if not self._post_open_advance_pending:
            return False
        if (not self._post_open_clearance_advance_enabled
                or self._post_open_clearance_advance_m <= 0.0):
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            return False

        pose = self._current_map_pose()
        if pose is None:
            self.cmd_vel_pub.publish(Twist())
            return True

        now = self.get_clock().now()
        progress = self._axis_progress_xy(pose[0], pose[1])
        if self._post_open_advance_start_time is None:
            self._post_open_advance_start_time = now
            self._post_open_advance_start_progress = progress
            self.get_logger().info(
                'Post-open corridor clearance advance started.')

        start_progress = (
            self._post_open_advance_start_progress
            if self._post_open_advance_start_progress is not None
            else progress)
        advanced = progress - start_progress
        if advanced >= self._post_open_clearance_advance_m:
            self._finish_post_open_clearance_advance(
                f'advanced={advanced:.2f}m')
            return False

        elapsed = (
            now - self._post_open_advance_start_time).nanoseconds / 1e9
        if elapsed > max(0.0, self._post_open_clearance_timeout_sec):
            self._finish_post_open_clearance_advance(
                f'timeout after {elapsed:.1f}s, advanced={advanced:.2f}m')
            return False

        front_min = (
            self._sector_min(
                self._latest_scan,
                -self._post_open_clearance_front_angle,
                self._post_open_clearance_front_angle)
            if self._latest_scan is not None else float('inf'))
        if front_min < self._post_open_clearance_front_min_m:
            self._finish_post_open_clearance_advance(
                f'front obstacle at {front_min:.2f}m')
            return False

        yaw_error = self._normalize_angle(self._mission_forward_yaw() - pose[2])
        yaw_abs = abs(yaw_error)
        twist = Twist()
        angular_limit = abs(self._post_open_clearance_angular_vel_limit)
        twist.angular.z = max(
            -angular_limit,
            min(angular_limit, self._post_open_clearance_heading_kp * yaw_error))
        if yaw_abs < math.radians(35.0):
            speed = abs(self._post_open_clearance_linear_vel)
            if yaw_abs > math.radians(18.0):
                speed *= 0.45
            twist.linear.x = speed
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(
            f'Post-open corridor clearance advance: '
            f'{advanced:.2f}/{self._post_open_clearance_advance_m:.2f}m, '
            f'front={front_min:.2f}m, '
            f'yaw_error={math.degrees(yaw_error):.0f}deg',
            throttle_duration_sec=1.0)
        return True

    def _finish_post_open_clearance_advance(self, reason: str):
        self._post_open_advance_pending = False
        self._post_open_advance_start_time = None
        self._post_open_advance_start_progress = None
        self._post_open_side_retreat_pending = False
        self._post_open_side_retreat_start_time = None
        self._post_open_side_retreat_start_lateral = None
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info(
            f'Post-open corridor clearance advance complete: {reason}')

    def _start_explore_waypoint_scan(self):
        if self._explore_waypoint_scan_sec <= 0.0:
            return
        self._explore_waypoint_scan_pending = True
        self._explore_waypoint_scan_start_time = None
        self._explore_waypoint_scan_bilateral = (
            self._should_use_bilateral_explore_waypoint_scan())
        self._explore_waypoint_scan_direction *= -1.0
        if self._explore_waypoint_scan_direction == 0.0:
            self._explore_waypoint_scan_direction = 1.0

    def _should_use_bilateral_explore_waypoint_scan(self) -> bool:
        opened_count = self._opened_exit_ready_count()
        if opened_count <= 0:
            return False
        if self._exit_xy_for_blue_candidate_filter() is not None:
            return True
        if self._min_opened_doors_before_exit > 0:
            return opened_count >= max(1, self._min_opened_doors_before_exit - 1)
        return opened_count >= 2

    def _run_explore_waypoint_scan(self) -> bool:
        if not self._explore_waypoint_scan_pending:
            return False
        if self._explore_waypoint_scan_sec <= 0.0:
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._explore_waypoint_scan_bilateral = False
            return False

        now = self.get_clock().now()
        if self._explore_waypoint_scan_start_time is None:
            self._explore_waypoint_scan_start_time = now
            scan_mode = 'bilateral' if self._explore_waypoint_scan_bilateral else 'single-side'
            self.get_logger().info(
                f'Explore waypoint reached; holding position for {scan_mode} camera/LiDAR side scan.')

        elapsed = (now - self._explore_waypoint_scan_start_time).nanoseconds / 1e9
        phase_sec = max(0.0, self._explore_waypoint_scan_sec)
        total_scan_sec = phase_sec * (2.0 if self._explore_waypoint_scan_bilateral else 1.0)
        if elapsed < total_scan_sec:
            twist = Twist()
            angular = abs(self._explore_waypoint_scan_angular_vel)
            direction = self._explore_waypoint_scan_direction
            if self._explore_waypoint_scan_bilateral and elapsed >= phase_sec:
                direction *= -1.0
            twist.angular.z = direction * angular
            self.cmd_vel_pub.publish(twist)
            return True

        self._explore_waypoint_scan_pending = False
        self._explore_waypoint_scan_start_time = None
        self._explore_waypoint_scan_bilateral = False
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info('Explore waypoint side scan complete.')
        return False

    def _send_explore_waypoint(self) -> bool:
        if not self._explore_nav_enabled:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False
        x, y, yaw = pose
        if not self._ready_to_start_navigation():
            return self._nav_start_pose_recovery_active
        raw_progress = self._axis_progress_xy(x, y)
        progress = raw_progress
        if self._opened_blue_door_positions:
            progress_floor = self._farthest_resolved_blue_progress()
            if progress_floor is None:
                progress_floor = self._mission_progress_x
            if raw_progress < progress_floor - 1.50:
                self.get_logger().warn(
                    f'탐색 pose 진행축 점프 감지: current={raw_progress:.2f}, '
                    f'confirmed={progress_floor:.2f}. 뒤쪽 waypoint 생성을 막기 위해 '
                    '열린 파란문 기준 진행축으로 탐색을 계속합니다.',
                    throttle_duration_sec=3.0)
            progress = max(raw_progress, progress_floor)
        lateral = self._axis_lateral_xy(x, y)
        explore_limit_progress = self._explore_limit_progress()
        distance_to_limit = explore_limit_progress - progress
        # Keep explore waypoints farther than navigation's reached-distance
        # fallback, otherwise Nav2 can accept and immediately complete the
        # same near waypoint without moving the robot.
        min_forward_step = max(0.40, self._explore_nav_min_step_m)
        if progress >= explore_limit_progress:
            return False
        if distance_to_limit < min_forward_step:
            if self._opened_exit_ready_count() >= self._min_opened_doors_before_exit:
                self._force_final_scan_before_exit = True
            self.get_logger().info(
                f'탐색 waypoint 보류: 출구 전방 한계까지 남은 거리가 '
                f'{explore_limit_progress - progress:.2f}m라 최종 스캔/출구 판단을 우선합니다.')
            return False
        lateral_error = lateral - self._explore_center_y
        if abs(lateral_error) > self._explore_nav_recenter_abs_y_m:
            if self._pending_exit_scan_after_opened_blue(progress):
                if self._send_exit_pending_scan_waypoint(
                        x, y, yaw, progress, lateral,
                        explore_limit_progress, min_forward_step):
                    return True
                self.get_logger().info(
                    'Exit-pending lane scan could not be planned from the '
                    'current side lane; falling back to normal recenter logic.')
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
        center_clearance = self._lane_scan_clearance(
            x, y, yaw, self._explore_center_y)
        target_lane_clearance = self._lane_scan_clearance(x, y, yaw, goal_lateral)
        full_lateral_change = (
            front_avoid_sign != 0.0
            and center_clearance < self._explore_obstacle_slow_m
            and target_lane_clearance >= max(
                self._explore_obstacle_stop_m,
                self._explore_nav_blue_lane_min_clearance_m)
        )
        large_lateral_change = (
            self._explore_nav_max_lateral_step_m > 0.0
            and abs(goal_lateral - lateral) > self._explore_nav_max_lateral_step_m)
        if full_lateral_change:
            limited_goal_lateral = goal_lateral
            if large_lateral_change:
                lane_reason += ' | direct front obstacle: full lane change allowed'
        else:
            limited_goal_lateral = self._limit_explore_goal_lateral_step(lateral, goal_lateral)
        if abs(limited_goal_lateral - goal_lateral) > 0.01:
            lane_reason += (
                f' | 좌우 이동량 제한 {goal_lateral - lateral:.2f}m→'
                f'{limited_goal_lateral - lateral:.2f}m')
            goal_lateral = limited_goal_lateral
        selected_clearance = self._lane_scan_clearance(x, y, yaw, goal_lateral)
        front_min = self._sector_min(
            self._latest_scan,
            -self._explore_front_angle,
            self._explore_front_angle) if self._latest_scan is not None else float('inf')
        if self._should_hold_explore_waypoint_for_side_scan(progress, front_min):
            self.get_logger().warn(
                f'탐색 waypoint 보류: 전방 여유 {front_min:.2f}m가 짧고 '
                '이미 열린 문 이후 충분히 진행했습니다. 벽/장애물에 붙지 않도록 '
                'LiDAR+camera 측면 스캔을 우선합니다.')
            return False
        step_m = self._explore_nav_step_m
        if (math.isfinite(selected_clearance)
                and selected_clearance < self._explore_nav_reduce_step_clearance_m):
            safe_step = selected_clearance - self._explore_nav_goal_clearance_margin_m
            if safe_step < min_forward_step:
                near_exit_scan_limit = max(
                    0.75,
                    self._explore_nav_step_m,
                    self._explore_nav_goal_clearance_margin_m + min_forward_step)
                if (self._opened_exit_ready_count() >= self._min_opened_doors_before_exit
                        and distance_to_limit <= near_exit_scan_limit):
                    self._force_final_scan_before_exit = True
                    self.get_logger().info(
                        f'탐색 waypoint 보류: 선택 차선 전방 여유가 '
                        f'{selected_clearance:.2f}m로 짧고 '
                        f'남은 탐색 거리 {distance_to_limit:.2f}m. '
                        '최종 측면 문 스캔/출구 판단을 우선합니다.')
                    return False
                self._explore_nav_clearance_deferral_count += 1
                deferral_limit = max(1, self._explore_nav_clearance_deferral_limit)
                if self._explore_nav_clearance_deferral_count < deferral_limit:
                    self.get_logger().info(
                        f'탐색 waypoint 보류: 선택 차선 전방 여유가 '
                        f'{selected_clearance:.2f}m로 짧아 재스캔합니다 '
                        f'({self._explore_nav_clearance_deferral_count}/{deferral_limit}).')
                    return False
                if selected_clearance < self._explore_obstacle_stop_m:
                    self._explore_nav_clearance_deferral_count = 0
                    if lane_sign != 0.0:
                        self._last_lane_sign = -lane_sign
                    self.get_logger().warn(
                        f'탐색 waypoint 취소: 선택 차선 전방 여유가 '
                        f'{selected_clearance:.2f}m로 stop threshold '
                        f'{self._explore_obstacle_stop_m:.2f}m보다 짧습니다. '
                        '안전한 차선을 다시 스캔합니다.')
                    return False
                step_m = min(
                    self._explore_nav_step_m,
                    max(min_forward_step, self._explore_nav_planner_probe_step_m))
                self._explore_nav_clearance_deferral_count = 0
                lane_reason += (
                    f' | 반복된 짧은 LiDAR 여유 {selected_clearance:.2f}m; '
                    f'Nav2 검증용 probe waypoint {step_m:.2f}m')
            else:
                self._explore_nav_clearance_deferral_count = 0
                step_m = min(
                    step_m,
                    max(self._explore_nav_min_step_m, safe_step))
                lane_reason += (
                    f' | 근접 장애물 여유 {selected_clearance:.2f}m, '
                    f'waypoint 전진거리 {step_m:.2f}m로 축소')
        else:
            self._explore_nav_clearance_deferral_count = 0
        if full_lateral_change and large_lateral_change:
            step_m = min(
                step_m,
                max(min_forward_step, self._explore_nav_planner_probe_step_m))
            lane_reason += (
                f' | 큰 차선 변경은 probe waypoint {step_m:.2f}m로 먼저 검증')
        goal_progress = min(progress + step_m, explore_limit_progress)
        if goal_progress - progress < min_forward_step:
            if distance_to_limit < max(0.45, min_forward_step):
                self._force_final_scan_before_exit = True
            self.get_logger().info(
                f'탐색 waypoint 보류: 요청 가능한 전진거리 '
                f'{goal_progress - progress:.2f}m가 너무 짧아 최종 스캔/출구 판단을 우선합니다.')
            return False
        goal_x, goal_y = self._axis_to_map_xy(goal_progress, goal_lateral)

        goal = DoorInfo()
        now = self.get_clock().now().to_msg()
        goal.header.stamp = now
        goal.header.frame_id = 'map'
        lateral_tag = int(round((goal_lateral - self._explore_center_y) * 100.0))
        goal.door_id = f'explore_waypoint_{goal_progress:.1f}_lat{lateral_tag:+d}'
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
        self._explore_nav_goal_xy = (goal_x, goal_y)
        self._explore_nav_clearance_deferral_count = 0
        self.get_logger().info(
            f'탐색 waypoint 계획 요청: {goal.door_id} '
            f'progress={goal_progress:.2f}, lateral={goal_lateral:.2f}, '
            f'map=({goal_x:.2f}, {goal_y:.2f}) | {lane_reason}')
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.EXPLORING_NAVIGATING)
        self.target_door_pub.publish(goal)
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

    def _should_hold_explore_waypoint_for_side_scan(
            self, progress: float, front_min: float) -> bool:
        threshold = max(0.0, self._explore_nav_front_block_scan_clearance_m)
        if threshold <= 0.0:
            return False
        if not math.isfinite(front_min) or front_min > threshold:
            return False

        farthest_opened = self._farthest_resolved_blue_progress()
        if farthest_opened is None:
            return False
        after_opened = max(0.0, self._explore_nav_front_block_scan_after_opened_m)
        if progress < farthest_opened + after_opened:
            return False
        return True

    def _farthest_resolved_blue_progress(self) -> float | None:
        positions = (
            list(self._opened_blue_door_positions)
            + list(self._abandoned_blue_door_positions)
        )
        if not positions:
            return None
        return max(self._axis_progress_xy(x, y) for x, y in positions)

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
        lateral_tag = int(round((goal_lateral - self._explore_center_y) * 100.0))
        goal.door_id = f'explore_recenter_{goal_progress:.1f}_lat{lateral_tag:+d}'
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
        self._explore_nav_goal_xy = (goal_x, goal_y)
        self._explore_nav_clearance_deferral_count = 0
        self.get_logger().info(
            f'중앙 복귀 waypoint 계획 요청: {goal.door_id} '
            f'progress={goal_progress:.2f}, lateral={goal_lateral:.2f}, '
            f'map=({goal_x:.2f}, {goal_y:.2f})')
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.EXPLORING_NAVIGATING)
        self.target_door_pub.publish(goal)
        return True

    def _pending_exit_scan_after_opened_blue(self, progress: float) -> bool:
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False
        if self._exit_blocking_unopened_blue_doors():
            return False
        if self._exit_door is None and self._last_valid_exit_xy is None:
            return False
        if not self._opened_blue_door_positions:
            return False
        min_after = max(
            0.0,
            self._detected_exit_min_after_opened_blue_m,
            self._front_wall_exit_min_after_opened_blue_m,
            self._detected_exit_min_robot_past_opened_blue_m)
        if min_after <= 0.0:
            return False
        farthest_opened = max(
            self._axis_progress_xy(x, y)
            for x, y in self._opened_blue_door_positions)
        return progress < farthest_opened + min_after

    def _send_exit_pending_scan_waypoint(
            self, x: float, y: float, yaw: float, progress: float,
            lateral: float, explore_limit_progress: float,
            min_forward_step: float) -> bool:
        if not self._opened_blue_door_positions:
            return False

        goal_lateral_hint = self._limit_explore_goal_lateral_step(
            lateral, self._explore_center_y)
        current_clearance = self._lane_scan_clearance(
            x, y, yaw, goal_lateral_hint)
        if (math.isfinite(current_clearance)
                and current_clearance < self._explore_obstacle_stop_m):
            self.get_logger().info(
                f'Exit-pending centerline scan blocked: selected lane clearance '
                f'{current_clearance:.2f}m < {self._explore_obstacle_stop_m:.2f}m.')
            return False

        min_after = max(
            0.0,
            self._detected_exit_min_after_opened_blue_m,
            self._front_wall_exit_min_after_opened_blue_m,
            self._detected_exit_min_robot_past_opened_blue_m)
        farthest_opened = max(
            self._axis_progress_xy(px, py)
            for px, py in self._opened_blue_door_positions)
        required_progress = farthest_opened + min_after + 0.20

        step_m = self._explore_nav_step_m
        if math.isfinite(current_clearance):
            safe_step = current_clearance - self._explore_nav_goal_clearance_margin_m
            if safe_step < min_forward_step:
                return False
            step_m = min(step_m, max(min_forward_step, safe_step))

        goal_progress = min(
            progress + step_m,
            required_progress,
            explore_limit_progress)
        if goal_progress - progress < min_forward_step:
            self._force_final_scan_before_exit = True
            return False

        hard_limit = max(
            abs(self._explore_y_hard_limit_m),
            abs(self._explore_nav_recenter_abs_y_m))
        goal_lateral = max(
            self._explore_center_y - hard_limit,
            min(self._explore_center_y + hard_limit, goal_lateral_hint))
        goal_x, goal_y = self._axis_to_map_xy(goal_progress, goal_lateral)

        goal = DoorInfo()
        now = self.get_clock().now().to_msg()
        goal.header.stamp = now
        goal.header.frame_id = 'map'
        lateral_tag = int(round((goal_lateral - self._explore_center_y) * 100.0))
        goal.door_id = f'explore_exit_center_scan_{goal_progress:.1f}_lat{lateral_tag:+d}'
        goal.door_color = 'explore'
        goal.confidence = 1.0
        goal.door_pose.header.stamp = now
        goal.door_pose.header.frame_id = 'map'
        goal.door_pose.pose.position.x = goal_x
        goal.door_pose.pose.position.y = goal_y
        goal.door_pose.pose.orientation.w = 1.0

        if abs(goal_lateral - self._explore_center_y) < 0.10:
            lane_sign = 0.0
        else:
            lane_sign = 1.0 if goal_lateral > self._explore_center_y else -1.0
        self._last_lane_sign = lane_sign
        self._explore_nav_goal_id = goal.door_id
        self._explore_nav_goal_sign = lane_sign
        self._explore_nav_goal_progress = goal_progress
        self._explore_nav_goal_xy = (goal_x, goal_y)
        self._explore_nav_clearance_deferral_count = 0
        self.get_logger().info(
            f'Exit-pending centerline scan waypoint requested: {goal.door_id} '
            f'progress={goal_progress:.2f}, lateral={goal_lateral:.2f}, '
            f'map=({goal_x:.2f}, {goal_y:.2f}), '
            f'clearance={current_clearance:.2f}m, '
            f'required_progress={required_progress:.2f}.')
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.EXPLORING_NAVIGATING)
        self.target_door_pub.publish(goal)
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
        self._clear_locked_blue_anchor()
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

        if (self._is_door_opened_for_observation(target)
                or self._is_door_abandoned_for_observation(target)
                or self._is_door_physically_opened(target)
                or self._is_door_physically_abandoned(target)):
            self._pre_nav_scan_start_time = None
            self._pending_nav_scan_target_id = None
            self._clear_locked_blue_anchor()
            self.target_door = None
            return False

        self._explore_start_time = self.get_clock().now()
        if not self._ready_to_start_navigation():
            if not self._nav_start_pose_recovery_active:
                self._rotate_to_scan()
            return True
        if self._run_pre_nav_scan(target):
            return True
        invalid_reason = self._blue_target_conflict_invalid_after_pre_nav_scan(target)
        if invalid_reason:
            target_id = target.door_id
            self._pre_nav_scan_start_time = None
            self._pending_nav_scan_target_id = None
            self._clear_locked_blue_anchor()
            self.target_door = None
            self._nav_retry_count = 0
            self.get_logger().warn(
                f'보류 중이던 파란 문 목표 {target_id} 제외: '
                f'{invalid_reason}. 새 관측 후보를 다시 선택합니다.')
            return False
        if self._defer_far_target_door(target):
            return True

        self.get_logger().info(
            f'Locked blue target after pre-scan: {target.door_id} '
            f'opened={self._resolved_blue_door_count()} '
            f'retry={self._nav_retry_count}/{self._max_nav_retries}')
        self._nav_done = False
        self._nav_failed = False
        self._transition(State.NAVIGATING)
        self.target_door_pub.publish(target)
        return True

    def _run_pre_nav_scan(self, target: DoorInfo) -> bool:
        if self._pre_nav_scan_sec <= 0.0:
            return False
        now = self.get_clock().now()
        target_key = self._pre_nav_scan_key(target)
        if self._pre_nav_scan_start_time is None:
            self._pending_nav_scan_target_id = target_key
            self._pre_nav_scan_start_time = now
            self.get_logger().info(
            f'LiDAR+camera segment map 사전 스캔 시작: {target.door_id}')
        elif self._pending_nav_scan_target_id != target_key:
            # The same physical door can be re-issued with a slightly different
            # projected map key while the robot is waiting. Keep the locked
            # scan window moving instead of restarting it every timer tick.
            self.get_logger().debug(
                f'Continuing locked pre-nav scan for {target.door_id}: '
                f'{self._pending_nav_scan_target_id} -> {target_key}',
                throttle_duration_sec=2.0)

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

    def _blue_target_invalid_after_pre_nav_scan(
            self, target: DoorInfo | None) -> str:
        if target is None:
            return 'target_door 없음'
        if target.door_color != 'blue':
            return ''
        if not self._door_has_map_identity(target):
            return 'map 기준 관측 좌표 없음'
        if self._is_door_opened_for_observation(target):
            return '이미 열린 물리 문과 같은 위치'
        if self._is_door_physically_abandoned(target):
            return '이미 접근 실패로 제외된 물리 문'
        if not self._is_blue_door_at_valid_wall_position(target):
            return '벽면 문 위치 조건 불일치'
        if self._semantic_non_blue_conflict_blocks_blue_target(target):
            return '관측 색상 메모리에서 비파란 문 우세'
        if self._blue_target_in_strict_exit_tail_without_persistent_memory(target):
            return '관측된 비상구 직전 tail 후보지만 반복 관측 메모리 부족'
        if self._is_door_observation_suppressed(target):
            if self._pre_exit_live_side_wall_blue_overrides_memory_conflict(target):
                self.get_logger().info(
                    f'Observation suppression ignored for {target.door_id}: '
                    'pre-exit side-wall blue evidence must be rechecked before exit.',
                    throttle_duration_sec=3.0)
            else:
                return '출구/열린 문/실패 후보 억제 조건과 충돌'
        if self._is_blue_candidate_blocked_by_red(target):
            return '사전 스캔 중 빨간문 근접 충돌 확인'
        return ''

    def _blue_target_conflict_invalid_after_pre_nav_scan(
            self, target: DoorInfo | None) -> str:
        if target is None:
            return 'target_door 없음'
        if target.door_color != 'blue':
            return ''
        if not self._door_has_map_identity(target):
            return 'map 기준 관측 좌표 없음'
        if self._is_door_opened_for_observation(target):
            return '이미 열린 물리 문과 같은 위치'
        if self._is_door_physically_abandoned(target):
            return '이미 접근 실패로 제외된 물리 문'
        if self._semantic_non_blue_conflict_blocks_blue_target(target):
            return '관측 색상 메모리에서 비파란 문 우세'
        if self._blue_target_in_strict_exit_tail_without_persistent_memory(target):
            return '관측된 비상구 직전 tail 후보지만 반복 관측 메모리 부족'
        if self._is_door_observation_suppressed(target):
            if self._pre_exit_live_side_wall_blue_overrides_memory_conflict(target):
                self.get_logger().info(
                    f'Observation suppression ignored for {target.door_id}: '
                    'pre-exit side-wall blue evidence must be rechecked before exit.',
                    throttle_duration_sec=3.0)
            else:
                return '출구/열린 문/실패 후보 억제 조건과 충돌'
        if self._is_blue_candidate_blocked_by_red(target):
            return '사전 스캔 중 빨간문 근접 충돌 확인'
        return ''

    def _blue_target_exit_tail_invalid_reason(
            self, target: DoorInfo | None) -> str:
        if target is None:
            return ''
        if target.door_color != 'blue' or not self._door_has_map_identity(target):
            return ''
        xy = self._door_identity_xy(target)
        if self._is_blue_xy_at_or_after_observed_exit(xy):
            return '관측된 비상구 진행축 이후/tail 후보'
        if self._blue_target_in_strict_exit_tail_without_persistent_memory(target):
            return '관측된 비상구 직전 tail 후보지만 반복 관측 메모리 부족'
        return ''

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

    def _run_explore_observation_wait(self) -> bool:
        wait_sec = max(0.0, self._explore_observation_wait_sec)
        if wait_sec <= 0.0 or self._explore_observation_wait_completed:
            return False

        now = self.get_clock().now()
        if self._explore_observation_wait_start_time is None:
            self._explore_observation_wait_start_time = now
            self.get_logger().info(
                f'파란 문 후보 없음. {wait_sec:.1f}s 동안 정지 스캔 후 '
                '관측 기반 waypoint를 계획합니다.')

        elapsed = (
            now - self._explore_observation_wait_start_time
        ).nanoseconds / 1e9
        if elapsed >= wait_sec:
            self._explore_observation_wait_start_time = None
            self._explore_observation_wait_completed = True
            return False

        twist = Twist()
        twist.angular.z = float(self._explore_observation_wait_angular_vel)
        self.cmd_vel_pub.publish(twist)
        return True

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

    def _exit_scan_allowed_by_exploration(self) -> bool:
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False
        if self._exit_blocking_unopened_blue_doors():
            return False

        pose = self._current_map_pose()
        if pose is None:
            return False

        if self._no_blue_exit_mode() and self._maybe_set_front_wall_exit(
                'no-blue forward exploration'):
            self.get_logger().info(
                'Exit scan allowed: no-blue mode has an observed '
                'front-wall exit candidate.',
                throttle_duration_sec=3.0)
            return True
        if self._no_blue_exit_mode() and self._maybe_set_no_blue_axis_exit(
                'no-blue forward exploration'):
            self.get_logger().info(
                'Exit scan allowed: no-blue mode reached the observed '
                'forward scan limit without blue-door evidence.',
                throttle_duration_sec=3.0)
            return True

        if self._valid_detected_exit():
            xy = self._exit_identity_xy()
            if xy is not None:
                distance_to_exit = math.hypot(xy[0] - pose[0], xy[1] - pose[1])
                if self._no_blue_exit_mode():
                    self.get_logger().info(
                        'Exit scan allowed: no-blue validation mode has an '
                        'observed exit candidate.',
                        throttle_duration_sec=3.0)
                    return True
                max_visible_nav_dist = max(0.0, self._exit_visible_nav_max_dist_m)
                if max_visible_nav_dist > 0.0 and distance_to_exit <= max_visible_nav_dist:
                    return True

        failure_limit = max(1, self._max_explore_nav_failures_before_exit)
        if self._explore_nav_failure_count >= failure_limit:
            self.get_logger().warn(
                f'Exit scan allowed after repeated explore navigation '
                f'failures ({self._explore_nav_failure_count}/{failure_limit}); '
                'continuing with a final observation scan from the current pose.')
            return True

        progress = self._axis_progress_xy(pose[0], pose[1])
        progress_floor = self._farthest_resolved_blue_progress()
        if progress_floor is not None and progress < progress_floor:
            progress = progress_floor
        exit_xy = self._exit_xy_for_blue_suppression()
        if exit_xy is None:
            exit_xy = self._semantic_green_exit_xy()
        if exit_xy is not None:
            exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
            stop_before_exit = max(0.45, self._exit_nav_standoff_m)
            if progress >= exit_progress - stop_before_exit:
                self.get_logger().info(
                    f'Exit scan allowed near observed exit: progress={progress:.2f}, '
                    f'exit_progress={exit_progress:.2f}, '
                    f'stop_before={stop_before_exit:.2f}.',
                    throttle_duration_sec=3.0)
                return True

        limit = self._explore_limit_progress()
        if not math.isfinite(limit):
            return False

        tolerance = max(
            0.45,
            min(
                0.95,
                max(
                    self._explore_nav_step_m,
                    self._explore_nav_min_step_m,
                    self._explore_nav_goal_clearance_margin_m + 0.30)))
        if progress + tolerance < limit:
            self.get_logger().info(
                f'Exit scan deferred: progress={progress:.2f}, '
                f'explore_limit={limit:.2f}. Continuing blue-door search.',
                throttle_duration_sec=3.0)
            return False
        return True

    def _ready_to_start_navigation(self) -> bool:
        pose = self._current_map_pose()
        if pose is None:
            self._nav_start_pose_recovery_active = False
            return True
        x, y, yaw = pose
        lateral = self._axis_lateral_xy(x, y)
        lateral_error = lateral - self._explore_center_y
        heading_error = abs(self._normalize_angle(self._mission_forward_yaw() - yaw))
        if abs(lateral_error) > self._nav_start_max_abs_y_m:
            if self._run_nav_start_center_recovery(pose, lateral_error):
                return False
            self.get_logger().info(
                f'Nav2 목표 대기: 중앙 복귀 중 '
                f'(lateral={lateral:.2f}, limit={self._nav_start_max_abs_y_m:.2f})',
                once=True)
            return False
        self._nav_start_pose_recovery_active = False
        if heading_error > self._nav_start_max_heading_error:
            self.get_logger().info(
                f'Nav2 목표 대기: 진행 방향 정렬 중 '
                f'(heading_error={math.degrees(heading_error):.0f}deg)',
                once=True)
            return False
        return True

    def _run_nav_start_center_recovery(
            self, pose: tuple[float, float, float], lateral_error: float) -> bool:
        if not self._nav_start_center_recovery_enabled:
            self._nav_start_pose_recovery_active = False
            return False
        target_abs = max(0.05, self._nav_start_center_recovery_target_abs_y_m)
        if abs(lateral_error) <= target_abs:
            self._nav_start_pose_recovery_active = False
            return False
        self._nav_start_pose_recovery_active = True
        _, _, yaw = pose
        side = 1.0 if lateral_error > 0.0 else -1.0
        center_facing_yaw = self._mission_forward_yaw() - side * math.pi / 2.0
        yaw_error = self._normalize_angle(center_facing_yaw - yaw)
        twist = Twist()
        yaw_tolerance = max(0.01, self._nav_start_center_recovery_yaw_tolerance)
        if abs(yaw_error) > yaw_tolerance:
            twist.angular.z = math.copysign(
                abs(self._nav_start_center_recovery_angular_vel), yaw_error)
        else:
            twist.linear.x = abs(self._nav_start_center_recovery_linear_vel)
            angular_limit = abs(self._nav_start_center_recovery_angular_vel)
            twist.angular.z = max(-angular_limit, min(angular_limit, 0.8 * yaw_error))
        self.cmd_vel_pub.publish(twist)
        self.get_logger().warn(
            f'Nav2 시작 위치가 벽/측면에 가까워 직접 중앙 복귀 중: '
            f'lateral_error={lateral_error:.2f}m, target_abs={target_abs:.2f}m, '
            f'yaw_error={math.degrees(yaw_error):.0f}deg, '
            f'cmd=({twist.linear.x:.2f}, {twist.angular.z:.2f})',
            throttle_duration_sec=1.0)
        return True

    def _filter_forward_doors(self, doors: list[DoorInfo]) -> list[DoorInfo]:
        candidates = [
            d for d in doors
            if d.door_pose.header.frame_id == 'map'
        ]

        if self._min_target_door_abs_y_m > 0.0:
            side_doors = [
                d for d in candidates
                if abs(self._door_reference_lateral(d) - self._explore_center_y)
                >= self._min_target_door_abs_y_m
            ]
            skipped = len(candidates) - len(side_doors)
            if skipped > 0 and not side_doors:
                self.get_logger().info(
                    f'중앙쪽 파란 문 후보 {skipped}개 무시 '
                    f'(|y| < {self._min_target_door_abs_y_m:.2f})', once=True)
            candidates = side_doors

        limited_candidates = [
            d for d in candidates
            if not self._is_blue_xy_beyond_explore_limit(
                self._door_identity_xy(d), d.door_id)
        ]
        candidates = limited_candidates

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
        if self._allow_near_backtracking_blue_revisit and pose is not None:
            robot_progress = self._axis_progress_xy(pose[0], pose[1])
            behind_tolerance = max(0.0, self._observed_blue_stale_behind_m)
            near_revisit = [
                d for d in candidates
                if -behind_tolerance <= (
                    self._door_progress(d) - robot_progress
                ) <= self._target_door_min_robot_forward_m
            ]
            if near_revisit:
                combined = self._merge_blue_target_lists(forward, near_revisit)
                self.get_logger().warn(
                    f'Including {len(near_revisit)} unopened blue candidate(s) '
                    'near the robot before continuing toward farther targets.',
                    throttle_duration_sec=3.0)
                return combined

            # A stable observed_blue_* cluster means the robot has mapped an
            # unopened physical blue door. Keep it selectable even if the robot
            # has already driven past it, so the exit cannot win before a missed
            # side door is revisited.
            revisit = [
                d for d in candidates
                if d.door_id.startswith('observed_blue_')
                and self._door_progress(d) <= min_x
            ]
            if revisit:
                combined = self._merge_blue_target_lists(forward, revisit)
                self.get_logger().warn(
                    f'Including {len(revisit)} stable observed blue revisit '
                    'candidate(s) before allowing exit progression.',
                    throttle_duration_sec=3.0)
                return combined
        skipped = len(candidates) - len(forward)
        if skipped > 0 and not forward:
            self.get_logger().info(
                f'뒤쪽/중복 파란 문 후보 {skipped}개 무시 '
                f'(다음 목표 x > {min_x:.2f} 대기)', once=True)
        return forward

    def _mark_door_opened(self, door: DoorInfo):
        self._add_door_id_keys(self._opened_door_ids, door.door_id)
        if door.door_pose.header.frame_id == 'map':
            door_x, door_y = self._door_identity_xy(door)
            opened_progresses = [
                self._axis_progress_xy(x, y)
                for x, y in self._opened_blue_door_positions
            ]
            opened_progresses.append(self._axis_progress_xy(door_x, door_y))
            self._mission_progress_x = max(
                self._mission_start_x,
                max(opened_progresses))

        self._record_opened_physical_blue_door(door)
        self._mark_semantic_door_opened(door)
        ox, oy = self._door_identity_xy(door)
        for other in self.detected_doors:
            if self._is_door_opened_for_observation(other):
                continue
            if other.door_color != door.door_color:
                continue
            if not self._door_has_map_identity(other):
                continue
            other_x, other_y = self._door_identity_xy(other)
            dx = other_x - ox
            dy = other_y - oy
            if math.hypot(dx, dy) <= self._opened_door_merge_dist_m:
                self._add_door_id_keys(self._opened_door_ids, other.door_id)
        self._prune_opened_blue_observations()

    def _prune_opened_blue_observations(self):
        before = len(self.detected_doors)
        self.detected_doors = [
            d for d in self.detected_doors
            if not (
                d.door_color == 'blue'
                and self._door_has_map_identity(d)
                and self._is_door_opened_for_observation(d)
            )
        ]
        removed = before - len(self.detected_doors)
        if removed > 0:
            self.get_logger().info(
                f'열린 파란 문 재관측 후보 {removed}개 정리. '
                '다음 목표 선택에서 제외합니다.')

    def _resolved_blue_door_count(self) -> int:
        # User-facing "opened" logs must not include abandoned/stale candidates.
        return self._opened_blue_door_count()

    def _opened_blue_door_count(self) -> int:
        physical_count = len(self._opened_blue_door_positions)
        if physical_count:
            return physical_count
        return len([door_id for door_id in self._opened_door_ids if door_id])

    def _no_blue_exit_mode(self) -> bool:
        return (
            self._min_opened_doors_before_exit <= 0
            and self._opened_blue_door_count() <= 0
            and not self._opened_blue_door_positions
            and not self._abandoned_blue_door_positions
        )

    def _opened_exit_ready_count(self) -> int:
        opened_positions = len(self._opened_blue_door_positions)
        if opened_positions:
            return opened_positions + len(self._abandoned_blue_door_positions)
        opened_ids = len([door_id for door_id in self._opened_door_ids if door_id])
        if opened_ids:
            return opened_ids + len([door_id for door_id in self._abandoned_door_ids if door_id])
        return 0

    def _on_exiting(self):
        if self._interrupt_exit_for_blue_door():
            return

        if self._exit_crossing_start_time is not None:
            self._continue_exit_crossing()
            return

        if (self._active_exit_goal is not None
                and (self._nav_failed or self._nav_start_pose_recovery_active)
                and self._recover_exit_nav_start_if_needed()):
            return

        if self._nav_done:
            self._nav_done = False
            if self._exit_pose_complete():
                self._transition(State.MISSION_COMPLETE)
            else:
                self._start_exit_crossing()
        elif self._nav_failed:
            if self._active_exit_goal is not None:
                if not self._ready_for_direct_exit_crossing():
                    if self._exit_retry_ready():
                        self._nav_failed = False
                        self.get_logger().warn(
                            'Exit Nav2 alignment failed while still far from the '
                            'pass-through goal; retrying Nav2 alignment.')
                        self._send_exit_goal()
                    return

                self._nav_failed = False
                self.get_logger().warn(
                    'Exit Nav2 alignment failed near the exit; switching to '
                    'goal-corrected direct crossing.')
                self._start_exit_crossing()
                return
            if self._exit_retry_ready():
                self._nav_failed = False
                self.get_logger().warn('비상구 이동 실패. 재시도...')
                self._send_exit_goal()

    def _recover_exit_nav_start_if_needed(self) -> bool:
        pose = self._current_map_pose()
        if pose is None:
            self._nav_start_pose_recovery_active = False
            return False
        x, y, _yaw = pose
        lateral = self._axis_lateral_xy(x, y)
        lateral_error = lateral - self._explore_center_y
        if abs(lateral_error) > self._nav_start_max_abs_y_m:
            self._nav_failed = False
            return self._run_nav_start_center_recovery(pose, lateral_error)
        if not self._nav_start_pose_recovery_active:
            return False
        self._nav_start_pose_recovery_active = False
        self._nav_failed = False
        self.get_logger().warn(
            'Exit Nav2 start pose recovered near the corridor center; '
            'resending the observed exit goal.')
        self._send_exit_goal()
        return True

    def _interrupt_exit_for_blue_door(self) -> bool:
        if not self._interrupt_explore_waypoint_for_blue_door(exit_context=True):
            return False
        self._exit_crossing_start_time = None
        self._active_exit_goal = None
        self._last_exit_goal_time = None
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().warn(
            '출구 이동 중 아직 열지 않은 파란 문이 새로 관측되어 '
            '비상구 목표를 취소하고 문 개방 순서로 복귀합니다.')
        return True

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
            '비상구 앞 정렬 완료. 마지막 통과 구간은 목표점 방향으로 보정 주행합니다.')
        self.cmd_vel_pub.publish(Twist())

    def _ready_for_direct_exit_crossing(self) -> bool:
        dist = self._active_exit_goal_distance()
        if dist is None:
            return False
        max_dist = max(
            0.5,
            self._exit_direct_cross_max_dist_m,
            self._exit_nav_standoff_m + self._exit_pass_through_m + 0.3)
        if self._last_valid_exit_from_front_wall and self._no_blue_exit_mode():
            max_dist = max(
                max_dist,
                self._exit_nav_standoff_m + self._exit_pass_through_m + 1.2)
        if dist > max_dist:
            self.get_logger().warn(
                f'비상구 직접 통과 보류: 목표까지 {dist:.2f}m 남음 '
                f'(limit={max_dist:.2f}m). Nav2 정렬을 재시도합니다.',
                throttle_duration_sec=2.0)
            return False
        return True

    def _active_exit_goal_distance(self) -> float | None:
        goal = self._active_exit_goal
        pose = self._current_map_pose()
        if goal is None or pose is None:
            return None
        return math.hypot(goal[0] - pose[0], goal[1] - pose[1])

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
            dx = goal[0] - pose[0]
            dy = goal[1] - pose[1]
            desired_yaw = goal[2]
            if math.hypot(dx, dy) > 0.10:
                desired_yaw = math.atan2(dy, dx)
            yaw_error = self._normalize_angle(desired_yaw - pose[2])
            angular_limit = max(0.0, self._exit_cross_angular_vel_limit)
            angular = self._exit_cross_heading_kp * yaw_error
            if angular_limit > 0.0:
                angular = max(-angular_limit, min(angular_limit, angular))
            twist.angular.z = angular
            yaw_abs = abs(yaw_error)
            if yaw_abs > math.radians(10.0):
                if yaw_abs >= self._exit_cross_rotate_in_place_threshold:
                    twist.linear.x = 0.0
                else:
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
        elif self._last_valid_exit_xy is not None:
            exit_nav_goal, exit_complete_goal = self._exit_goal_pair_from_xy(
                self._last_valid_exit_xy)
            self.get_logger().warn(
                f'Using last observed green exit goal '
                f'({self._last_valid_exit_xy[0]:.1f}, '
                f'{self._last_valid_exit_xy[1]:.1f}) after current exit '
                'candidate became temporarily invalid.')
        elif self._provisional_exit_xy_for_blue_suppression() is not None:
            provisional_xy = self._provisional_exit_xy_for_blue_suppression()
            exit_nav_goal, exit_complete_goal = self._exit_goal_pair_from_xy(
                provisional_xy)
            self.get_logger().warn(
                f'Using provisional observed green exit goal '
                f'({provisional_xy[0]:.1f}, {provisional_xy[1]:.1f}) after '
                'final scan found no remaining blue doors.')
        elif self._semantic_green_exit_xy() is not None:
            semantic_xy = self._semantic_green_exit_xy()
            exit_nav_goal, exit_complete_goal = self._exit_goal_pair_from_xy(
                semantic_xy)
            self.get_logger().warn(
                f'Using semantic green exit goal '
                f'({semantic_xy[0]:.1f}, {semantic_xy[1]:.1f}) from accumulated '
                'map observations after no remaining blue doors were selected.')
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

        if self.state == State.EXITING and self._active_exit_goal is not None:
            locked_exit_goal = self._active_exit_goal
            locked_yaw = locked_exit_goal[2]
            backoff = max(0.0, self._exit_pass_through_m) + max(
                0.0,
                self._exit_nav_standoff_m)
            exit_complete_goal = locked_exit_goal
            exit_nav_goal = (
                locked_exit_goal[0] - math.cos(locked_yaw) * backoff,
                locked_exit_goal[1] - math.sin(locked_yaw) * backoff,
                locked_yaw,
            )
            self.get_logger().info(
                f'EXITING 상태에서는 기존 관측 기반 출구 목표를 유지합니다: '
                f'Nav2 정렬 목표 ({exit_nav_goal[0]:.1f}, {exit_nav_goal[1]:.1f}), '
                f'통과 완료 목표 ({exit_complete_goal[0]:.1f}, {exit_complete_goal[1]:.1f}).',
                throttle_duration_sec=2.0)
        else:
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
        xy = self._exit_observation_xy(self._exit_door)
        if xy is None:
            return False

        x, y = xy
        if not self._exit_candidate_axis_valid(
                (float(x), float(y)),
                self._exit_door.door_id,
                clear_existing=True):
            return False

        if not self._detected_exit_use_expected_region:
            self._last_valid_exit_xy = (float(x), float(y))
            self._last_valid_exit_from_front_wall = (
                self._exit_door.door_id in (
                    'front_wall_exit_observed',
                    'no_blue_axis_exit_observed'))
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
            self._last_valid_exit_from_front_wall = False
            return False
        self._last_valid_exit_xy = (float(x), float(y))
        self._last_valid_exit_from_front_wall = (
            self._exit_door.door_id in (
                'front_wall_exit_observed',
                'no_blue_axis_exit_observed'))
        return True

    def _exit_candidate_axis_valid(
            self,
            xy: tuple[float, float],
            door_id: str = '',
            clear_existing: bool = False,
            require_front_wall_confirmation: bool = True) -> bool:
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
                self._last_valid_exit_xy = None
                self._last_valid_exit_from_front_wall = False
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
                self._last_valid_exit_xy = None
                self._last_valid_exit_from_front_wall = False
            return False

        if (require_front_wall_confirmation
                and self._detected_exit_require_front_wall_confirmation
                and door_id != 'front_wall_exit_observed'
                and not self._no_blue_exit_mode()):
            wall_xy = self._front_wall_exit_xy()
            if wall_xy is None:
                self.get_logger().info(
                    f'Exit candidate {door_id or "(unknown)"} rejected: '
                    'front-wall confirmation is not available yet.',
                    throttle_duration_sec=3.0)
                if clear_existing:
                    self._exit_door = None
                    self._last_valid_exit_xy = None
                    self._last_valid_exit_from_front_wall = False
                return False
            exit_progress = self._axis_progress_xy(x, y)
            wall_progress = self._axis_progress_xy(wall_xy[0], wall_xy[1])
            tolerance = max(0.0, self._detected_exit_front_wall_tolerance_m)
            if abs(exit_progress - wall_progress) > tolerance:
                self.get_logger().info(
                    f'Exit candidate {door_id or "(unknown)"} rejected: '
                    f'green progress {exit_progress:.1f}m does not match '
                    f'front-wall progress {wall_progress:.1f}m '
                    f'(tol={tolerance:.1f}m).',
                    throttle_duration_sec=3.0)
                if clear_existing:
                    self._exit_door = None
                    self._last_valid_exit_xy = None
                    self._last_valid_exit_from_front_wall = False
                return False

        min_forward = self._detected_exit_min_robot_forward_m
        if self._no_blue_exit_mode():
            min_forward = max(min_forward, self._no_blue_exit_min_robot_forward_m)
        if min_forward > 0.0:
            pose = self._current_map_pose()
            if pose is not None:
                robot_progress = self._axis_progress_xy(pose[0], pose[1])
                exit_progress = self._axis_progress_xy(x, y)
                if self._no_blue_exit_mode():
                    if robot_progress < min_forward:
                        self.get_logger().info(
                            f'비상구 후보 {door_id or "(unknown)"} 제외: '
                            f'파란 문 없음 판단을 위해 로봇 진행축 '
                            f'{robot_progress:.1f}m가 최소 스캔 거리 '
                            f'{min_forward:.1f}m에 도달하지 않았습니다.',
                            throttle_duration_sec=3.0)
                        if clear_existing:
                            self._exit_door = None
                            self._last_valid_exit_xy = None
                            self._last_valid_exit_from_front_wall = False
                        return False
                elif exit_progress < robot_progress + min_forward:
                    self.get_logger().info(
                        f'비상구 후보 {door_id or "(unknown)"} 제외: '
                        f'로봇 진행축보다 충분히 앞에 있지 않습니다 '
                        f'(exit={exit_progress:.1f}, robot={robot_progress:.1f}).',
                        throttle_duration_sec=3.0)
                    if clear_existing:
                        keep_locked_exit_memory = (
                            door_id == 'front_wall_exit_observed'
                            and self.state == State.EXITING
                            and self._active_exit_goal is not None)
                        self._exit_door = None
                        if not keep_locked_exit_memory:
                            self._last_valid_exit_xy = None
                            self._last_valid_exit_from_front_wall = False
                    return False

        min_after_blue = max(0.0, self._detected_exit_min_after_opened_blue_m)
        if min_after_blue > 0.0 and self._opened_blue_door_positions:
            exit_progress = self._axis_progress_xy(x, y)
            farthest_opened = max(
                self._axis_progress_xy(blue_x, blue_y)
                for blue_x, blue_y in self._opened_blue_door_positions)
            pose = self._current_map_pose()
            min_robot_past = max(
                0.0, self._detected_exit_min_robot_past_opened_blue_m)
            if exit_progress < farthest_opened + min_after_blue:
                self.get_logger().info(
                    f'Exit candidate {door_id or "(unknown)"} rejected: '
                    f'progress {exit_progress:.1f}m is only '
                    f'{exit_progress - farthest_opened:.1f}m after the last '
                    f'opened blue door; need {min_after_blue:.1f}m before '
                    'treating green as the final exit.',
                    throttle_duration_sec=3.0)
                if clear_existing:
                    self._exit_door = None
                    self._last_valid_exit_xy = None
                    self._last_valid_exit_from_front_wall = False
                return False
            if pose is not None and min_robot_past > 0.0:
                robot_progress = self._axis_progress_xy(pose[0], pose[1])
                if robot_progress < farthest_opened + min_robot_past:
                    self.get_logger().info(
                        f'Exit candidate {door_id or "(unknown)"} rejected: '
                        f'robot progress {robot_progress:.1f}m has not scanned '
                        f'past the last opened blue door at {farthest_opened:.1f}m '
                        f'by {min_robot_past:.1f}m yet.',
                        throttle_duration_sec=3.0)
                    if clear_existing:
                        self._exit_door = None
                        self._last_valid_exit_xy = None
                        self._last_valid_exit_from_front_wall = False
                    return False

        return True

    def _maybe_set_front_wall_exit(self, reason: str) -> bool:
        """Promote an observed front wall/end opening to an exit candidate.

        This is a late fallback for the exit phase only. It must not suppress
        unopened blue side-wall doors because a nearby obstacle can look like a
        front wall before the real exit is visible.
        """
        if self._valid_detected_exit():
            return True
        if not self._allow_front_wall_exit_fallback:
            return False
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False

        wall_xy = self._front_wall_exit_xy()
        if wall_xy is None:
            return False
        if self._no_blue_exit_mode():
            pose = self._current_map_pose()
            if pose is not None:
                robot_progress = self._axis_progress_xy(pose[0], pose[1])
                min_forward = max(0.0, self._no_blue_exit_min_robot_forward_m)
                if robot_progress < min_forward:
                    return False
        elif self._last_provisional_exit_xy is None:
            # In normal blue-door missions, do not treat an arbitrary LiDAR
            # end wall/obstacle as the exit before a green exit observation has
            # ever appeared. Otherwise a mid-corridor obstacle can suppress
            # still-unopened blue doors as "past the exit".
            return False
        if not self._front_wall_exit_is_far_enough_after_opened_blue(wall_xy):
            return False

        yaw = self._mission_forward_yaw()
        now = self.get_clock().now().to_msg()
        msg = DoorInfo()
        msg.header.stamp = now
        msg.header.frame_id = 'map'
        msg.door_id = 'front_wall_exit_observed'
        msg.door_color = 'green'
        msg.confidence = 0.35
        msg.is_open = True
        msg.door_pose.header.stamp = now
        msg.door_pose.header.frame_id = 'map'
        msg.door_pose.pose.position.x = wall_xy[0]
        msg.door_pose.pose.position.y = wall_xy[1]
        msg.door_pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.door_pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.handle_position.header.stamp = now
        msg.handle_position.header.frame_id = 'map'
        msg.handle_position.point.x = wall_xy[0]
        msg.handle_position.point.y = wall_xy[1]
        msg.handle_position.point.z = 0.9
        self._exit_door = msg
        self._last_valid_exit_xy = wall_xy
        self._last_valid_exit_from_front_wall = True
        self.get_logger().warn(
            f'초록 비상구 색상 후보가 없어 전방 LiDAR 끝벽 관측을 '
            f'비상구 후보로 사용합니다 ({reason}): '
            f'({wall_xy[0]:.2f}, {wall_xy[1]:.2f})')
        return True

    def _maybe_set_no_blue_axis_exit(self, reason: str) -> bool:
        """Create a no-blue exit candidate from the observed mission axis.

        This is only for scenes where repeated exploration produced no blue-door
        evidence and the green panel itself was not detected. It does not use the
        configured fallback exit coordinates.
        """
        if self._valid_detected_exit():
            return True
        if not self._allow_front_wall_exit_fallback:
            return False
        if not self._no_blue_exit_mode():
            return False
        if self._exit_blocking_unopened_blue_doors():
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False

        robot_progress = self._axis_progress_xy(pose[0], pose[1])
        min_forward = max(0.0, self._no_blue_exit_min_robot_forward_m)
        if robot_progress < min_forward:
            return False
        if self._stable_unopened_observed_blue_targets():
            return False

        ahead = max(
            1.20,
            self._explore_nav_min_step_m + 0.80,
            self._exit_nav_standoff_m + 0.45)
        exit_progress = robot_progress + ahead
        exit_xy = self._axis_to_map_xy(exit_progress, self._explore_center_y)
        if not self._exit_candidate_axis_valid(
                exit_xy,
                'no_blue_axis_exit_observed',
                clear_existing=False,
                require_front_wall_confirmation=False):
            return False

        yaw = self._mission_forward_yaw()
        now = self.get_clock().now().to_msg()
        msg = DoorInfo()
        msg.header.stamp = now
        msg.header.frame_id = 'map'
        msg.door_id = 'no_blue_axis_exit_observed'
        msg.door_color = 'green'
        msg.confidence = 0.30
        msg.is_open = True
        msg.door_pose.header.stamp = now
        msg.door_pose.header.frame_id = 'map'
        msg.door_pose.pose.position.x = exit_xy[0]
        msg.door_pose.pose.position.y = exit_xy[1]
        msg.door_pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.door_pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.handle_position.header.stamp = now
        msg.handle_position.header.frame_id = 'map'
        msg.handle_position.point.x = exit_xy[0]
        msg.handle_position.point.y = exit_xy[1]
        msg.handle_position.point.z = 0.9
        self._exit_door = msg
        self._last_valid_exit_xy = exit_xy
        self._last_valid_exit_from_front_wall = True
        self.get_logger().warn(
            f'파란 문 관측/개방 기록 없이 진행축 {robot_progress:.1f}m까지 '
            f'스캔했고 초록 비상구 후보가 검출되지 않아, 현재 관측 진행축 '
            f'전방을 no-blue 출구 후보로 사용합니다 ({reason}): '
            f'({exit_xy[0]:.2f}, {exit_xy[1]:.2f})')
        return True

    def _front_wall_exit_xy(self) -> tuple[float, float] | None:
        scan = self._latest_scan
        pose = self._current_map_pose()
        if scan is None or pose is None:
            return None

        robot_x, robot_y, robot_yaw = pose
        axis_yaw = self._mission_forward_yaw()
        axis_x = math.cos(axis_yaw)
        axis_y = math.sin(axis_yaw)
        side_x = -axis_y
        side_y = axis_x
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        min_range = max(scan.range_min, self._front_wall_exit_min_range_m)
        max_range = min(scan.range_max, self._front_wall_exit_max_range_m)
        if max_range <= min_range:
            return None

        forward_hits: list[float] = []
        total_front_rays = 0
        angle = scan.angle_min
        for rng in scan.ranges:
            if -self._front_wall_exit_half_angle <= angle <= self._front_wall_exit_half_angle:
                total_front_rays += 1
                if math.isfinite(rng) and min_range <= rng <= max_range:
                    base_x = float(rng) * math.cos(angle)
                    base_y = float(rng) * math.sin(angle)
                    map_x = robot_x + cos_yaw * base_x - sin_yaw * base_y
                    map_y = robot_y + sin_yaw * base_x + cos_yaw * base_y
                    rel_x = map_x - robot_x
                    rel_y = map_y - robot_y
                    forward = rel_x * axis_x + rel_y * axis_y
                    lateral = rel_x * side_x + rel_y * side_y
                    if forward >= min_range and abs(lateral) <= max(0.55, self._explore_nav_lane_width_m):
                        forward_hits.append(forward)
            angle += scan.angle_increment

        if total_front_rays <= 0 or len(forward_hits) < 5:
            return None
        coverage = len(forward_hits) / float(total_front_rays)
        if coverage < self._front_wall_exit_min_coverage:
            return None

        forward_hits.sort()
        median = forward_hits[len(forward_hits) // 2]
        p10 = forward_hits[max(0, int(len(forward_hits) * 0.10))]
        p90 = forward_hits[min(len(forward_hits) - 1, int(len(forward_hits) * 0.90))]
        if p90 - p10 > self._front_wall_exit_max_spread_m:
            return None

        robot_progress = self._axis_progress_xy(robot_x, robot_y)
        wall_progress = robot_progress + median
        return self._axis_to_map_xy(wall_progress, self._explore_center_y)

    def _front_wall_exit_is_far_enough_after_opened_blue(
            self, wall_xy: tuple[float, float]) -> bool:
        min_after = max(0.0, self._front_wall_exit_min_after_opened_blue_m)
        if min_after <= 0.0 or not self._opened_blue_door_positions:
            return True

        farthest_opened_progress = max(
            self._axis_progress_xy(x, y)
            for x, y in self._opened_blue_door_positions)
        wall_progress = self._axis_progress_xy(wall_xy[0], wall_xy[1])
        pose = self._current_map_pose()
        robot_progress = (
            self._axis_progress_xy(pose[0], pose[1])
            if pose is not None
            else wall_progress)

        if wall_progress < farthest_opened_progress + min_after:
            self.get_logger().debug(
                f'전방 끝벽 후보 제외: 마지막 개방 파란문 진행축 '
                f'{farthest_opened_progress:.1f}m 이후 {min_after:.1f}m '
                f'이상 떨어지지 않았습니다 (wall={wall_progress:.1f}m).')
            return False
        robot_past_required = max(
            min_after * 0.5,
            self._max_explore_past_last_opened_blue_m)
        if robot_progress < farthest_opened_progress + robot_past_required:
            self.get_logger().debug(
                f'전방 끝벽 후보 제외: 로봇이 마지막 개방 파란문 이후 '
                f'충분히 전진하지 않았습니다 '
                f'(robot={robot_progress:.1f}m, opened={farthest_opened_progress:.1f}, '
                f'need={robot_past_required:.1f}m).')
            return False
        return True

    def _front_wall_exit_xy_for_blue_suppression(self) -> tuple[float, float] | None:
        if not self._allow_front_wall_exit_fallback:
            return None
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return None
        wall_xy = self._front_wall_exit_xy()
        if wall_xy is None:
            return None
        if not self._front_wall_exit_is_far_enough_after_opened_blue(wall_xy):
            return None
        return wall_xy

    def _exit_center_y_error(self, xy: tuple[float, float]) -> float:
        return abs(self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)

    def _exit_identity_xy(self) -> tuple[float, float] | None:
        if self._exit_door is None:
            return None
        return self._exit_observation_xy(self._exit_door)

    def _detected_exit_goal_pair(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        if not self._valid_detected_exit():
            return None
        xy = self._exit_identity_xy()
        if xy is None:
            return None
        return self._exit_goal_pair_from_xy(xy)

    def _exit_goal_pair_from_xy(
            self,
            xy: tuple[float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        exit_x, exit_y = xy
        exit_progress = self._axis_progress_xy(exit_x, exit_y)
        exit_lateral = self._axis_lateral_xy(exit_x, exit_y)
        lateral_error = exit_lateral - self._explore_center_y
        if (self._detected_exit_max_center_y_m > 0.0
                and abs(lateral_error) <= self._detected_exit_max_center_y_m):
            # The green panel is on the end wall; projection noise can shift its
            # apparent y. Approach along the mission axis, biased toward center.
            exit_lateral = self._explore_center_y + lateral_error * 0.35
            exit_x, exit_y = self._axis_to_map_xy(exit_progress, exit_lateral)

        yaw = self._mission_forward_yaw()
        ux = math.cos(yaw)
        uy = math.sin(yaw)
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

    def _opened_blue_explore_limit_progress(self) -> float:
        opened_margin = max(0.0, self._max_explore_past_last_opened_blue_m)
        opened_positions = list(self._opened_blue_door_positions)
        if self._abandoned_blue_extends_explore_limit:
            opened_positions += list(self._abandoned_blue_door_positions)
        if opened_margin > 0.0 and opened_positions:
            farthest_opened = max(
                self._axis_progress_xy(x, y)
                for x, y in opened_positions)
            return farthest_opened + opened_margin
        return float('inf')

    def _explore_limit_progress(self) -> float:
        opened_limit = self._opened_blue_explore_limit_progress()
        if self._valid_detected_exit():
            xy = self._exit_identity_xy()
            if xy is not None:
                return self._axis_progress_xy(xy[0], xy[1]) - 0.6
        observed_exit_xy = self._exit_xy_for_blue_suppression()
        if observed_exit_xy is not None:
            grace = max(0.0, self._observed_exit_explore_grace_m)
            observed_limit = (
                self._axis_progress_xy(observed_exit_xy[0], observed_exit_xy[1])
                + grace)
            if math.isfinite(opened_limit):
                return min(observed_limit, opened_limit)
            return observed_limit
        if self._opened_blue_door_count() <= 0:
            return float('inf')
        if not self._allow_fallback_exit_goal:
            # Do not reject farther blue doors just because the robot has
            # already opened one. The corridor end must be observed first; until
            # then exploration remains open-ended and fully observation-based.
            return float('inf')
        fallback_limit = self._axis_progress_xy(self._exit_x, self._exit_y) - 1.0
        if math.isfinite(opened_limit):
            return min(fallback_limit, opened_limit)
        return fallback_limit

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
        if self._no_blue_exit_mode():
            return 0.0, 'No-blue mode: LiDAR obstacle clearance only'
        if not self._explore_color_lane_bias:
            return 0.0, '색상 차선 선호 비활성'

        blue_doors = [
            d for d in self.detected_doors
            if d.door_color == 'blue'
            and not self._is_door_failed_for_observation(d)
            and not self._is_door_opened_for_observation(d)
            and not self._is_door_abandoned_for_observation(d)
            and self._is_door_recent_observation(d)
            and not self._is_door_physically_opened(d)
            and not self._is_door_observation_suppressed(d)
            and not self._is_door_physically_abandoned(d)
            and self._is_blue_door_target_ready(d)
            and not self._is_blue_candidate_blocked_by_red(d)
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
        y = self._door_reference_lateral(door)
        if abs(y - self._explore_center_y) <= 0.08:
            return 0.0
        return 1.0 if y > self._explore_center_y else -1.0

    def _door_reference_lateral(self, door: DoorInfo) -> float:
        if door.door_color == 'blue' and self._door_has_map_identity(door):
            xy = self._door_identity_xy(door)
            return self._axis_lateral_xy(xy[0], xy[1])
        if door.door_pose.header.frame_id == 'map':
            return self._door_lateral(door)
        return float(door.door_pose.pose.position.y) + self._explore_center_y

    def _door_has_map_identity(self, door: DoorInfo) -> bool:
        return (
            door.handle_position.header.frame_id == 'map'
            or door.door_pose.header.frame_id == 'map'
        )

    def _door_identity_xy(self, door: DoorInfo) -> tuple[float, float]:
        """Stable map point for duplicate/opened-door identity checks."""
        if door.door_color == 'blue':
            pose_xy: tuple[float, float] | None = None
            handle_xy: tuple[float, float] | None = None
            if door.door_pose.header.frame_id == 'map':
                pose_xy = (
                    float(door.door_pose.pose.position.x),
                    float(door.door_pose.pose.position.y),
                )
            if door.handle_position.header.frame_id == 'map':
                handle_xy = (
                    float(door.handle_position.point.x),
                    float(door.handle_position.point.y),
                )

            if pose_xy is not None and handle_xy is not None:
                pose_lateral = abs(
                    self._axis_lateral_xy(pose_xy[0], pose_xy[1])
                    - self._explore_center_y)
                handle_lateral = abs(
                    self._axis_lateral_xy(handle_xy[0], handle_xy[1])
                    - self._explore_center_y)
                pose_on_wall = self._is_xy_at_configured_wall_lateral(pose_xy)
                handle_on_wall = self._is_xy_at_configured_wall_lateral(handle_xy)
                inward_shift = max(
                    0.35,
                    self._axis_door_side_standoff_m * 0.4)
                if (
                        pose_on_wall
                        and (not handle_on_wall
                             or pose_lateral > handle_lateral + inward_shift)):
                    return pose_xy

            if (
                    handle_xy is not None
                    and self._is_xy_at_configured_wall_lateral(handle_xy)
                    and self._door_handle_consistent_with_pose(door, handle_xy)):
                return handle_xy
            if pose_xy is not None:
                return pose_xy

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
            handle_xy = (
                float(door.handle_position.point.x),
                float(door.handle_position.point.y),
            )
            if (
                    self._is_xy_at_configured_wall_lateral(handle_xy)
                    and self._door_handle_consistent_with_pose(door, handle_xy)):
                return handle_xy
        if door.door_pose.header.frame_id == 'map':
            return (
                float(door.door_pose.pose.position.x),
                float(door.door_pose.pose.position.y),
            )
        return self._door_identity_xy(door)

    def _blue_evidence_points(self, door: DoorInfo) -> list[tuple[float, float]]:
        points = [self._door_handle_xy(door), self._door_identity_xy(door)]
        raw = self._raw_blue_handle_xy_for_evidence(door)
        if raw is not None and all(
                math.hypot(raw[0] - xy[0], raw[1] - xy[1]) > 0.05
                for xy in points):
            points.append(raw)
        return points

    def _raw_blue_handle_xy_for_evidence(
            self, door: DoorInfo) -> tuple[float, float] | None:
        if door.door_color != 'blue':
            return None
        if door.handle_position.header.frame_id != 'map':
            return None
        xy = (
            float(door.handle_position.point.x),
            float(door.handle_position.point.y),
        )
        lateral_abs = abs(
            self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        min_wall = max(0.0, self._observed_blue_min_abs_wall_y_m)
        max_wall = max(0.0, self._observed_blue_max_abs_wall_y_m)
        overshoot = max(
            0.0,
            self._blue_handle_max_wall_overshoot_m,
            self._blue_handle_validation_overshoot_m)
        if min_wall > 0.0 and lateral_abs < min_wall:
            return None
        if max_wall > 0.0 and lateral_abs > max_wall + overshoot:
            return None
        return xy

    def _axis_side_door_front_parking_pose(
            self, handle_progress: float, handle_lateral: float,
            min_side_lateral: float = 0.0) -> tuple[float, float, float, float] | None:
        lateral_from_center = handle_lateral - self._explore_center_y
        if abs(lateral_from_center) < self._axis_door_min_abs_lateral_m:
            return None

        side = 1.0 if lateral_from_center > 0.0 else -1.0
        standoff = max(0.10, self._axis_door_side_standoff_m)
        wall_abs_lateral = abs(lateral_from_center)
        min_wall = max(0.0, self._observed_blue_min_abs_wall_y_m)
        max_wall = max(0.0, self._observed_blue_max_abs_wall_y_m)
        if max_wall > 0.0:
            wall_abs_lateral = max(
                wall_abs_lateral,
                max_wall - max(0.0, self._blue_handle_max_wall_overshoot_m))
            wall_abs_lateral = min(wall_abs_lateral, max_wall)
        elif min_wall > 0.0:
            wall_abs_lateral = max(wall_abs_lateral, min_wall + standoff)

        rel_lateral_abs = max(0.0, wall_abs_lateral - standoff)
        min_side_lane = max(
            0.0,
            self._axis_door_min_side_goal_lateral_m,
            min_side_lateral)
        if rel_lateral_abs < min_side_lane:
            rel_lateral_abs = min_side_lane

        max_abs_lateral = max(
            self._axis_door_min_abs_lateral_m,
            self._nav_start_max_abs_y_m,
            min_side_lane)
        if rel_lateral_abs > max_abs_lateral:
            rel_lateral_abs = max_abs_lateral

        parking_lateral = self._explore_center_y + side * rel_lateral_abs
        wall_lateral = self._explore_center_y + side * wall_abs_lateral
        parking_yaw = self._mission_forward_yaw() + side * math.pi / 2.0
        return handle_progress, parking_lateral, parking_yaw, wall_lateral

    def _door_handle_consistent_with_pose(
            self, door: DoorInfo, handle_xy: tuple[float, float]) -> bool:
        """Reject side-camera handle estimates that project onto the wrong wall."""
        if door.door_pose.header.frame_id != 'map':
            return True

        pose_xy = (
            float(door.door_pose.pose.position.x),
            float(door.door_pose.pose.position.y),
        )
        handle_progress = self._axis_progress_xy(handle_xy[0], handle_xy[1])
        pose_progress = self._axis_progress_xy(pose_xy[0], pose_xy[1])
        handle_lateral = (
            self._axis_lateral_xy(handle_xy[0], handle_xy[1])
            - self._explore_center_y)
        pose_lateral = (
            self._axis_lateral_xy(pose_xy[0], pose_xy[1])
            - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)

        if (
                abs(handle_lateral) >= wall_side_threshold
                and abs(pose_lateral) >= wall_side_threshold
                and handle_lateral * pose_lateral < 0.0):
            self.get_logger().info(
                f'파란 문 {door.door_id} handle 제외: door_pose와 반대쪽 벽으로 투영됨 '
                f'handle_axis=({handle_progress:.2f}, {handle_lateral:.2f}), '
                f'pose_axis=({pose_progress:.2f}, {pose_lateral:.2f})',
                throttle_duration_sec=3.0)
            return False

        max_delta = max(0.0, self._axis_door_max_handle_pose_progress_delta_m)
        if max_delta > 0.0 and abs(handle_progress - pose_progress) > max_delta:
            self.get_logger().info(
                f'파란 문 {door.door_id} handle 제외: door_pose와 진행축 차이가 큼 '
                f'handle_axis=({handle_progress:.2f}, {handle_lateral:.2f}), '
                f'pose_axis=({pose_progress:.2f}, {pose_lateral:.2f}), '
                f'max_delta={max_delta:.2f}m',
                throttle_duration_sec=3.0)
            return False

        return True

    def _is_direct_blue_pose_lateral_consistent(self, door: DoorInfo) -> bool:
        """Reject direct blue boxes whose wall point and drive-up pose disagree."""
        if door.door_color != 'blue' or door.door_id.startswith('observed_blue_'):
            return True
        if (
                door.door_pose.header.frame_id != 'map'
                or door.handle_position.header.frame_id != 'map'):
            return True

        pose_xy = (
            float(door.door_pose.pose.position.x),
            float(door.door_pose.pose.position.y),
        )
        handle_xy = (
            float(door.handle_position.point.x),
            float(door.handle_position.point.y),
        )
        pose_lateral = (
            self._axis_lateral_xy(pose_xy[0], pose_xy[1])
            - self._explore_center_y)
        handle_lateral = (
            self._axis_lateral_xy(handle_xy[0], handle_xy[1])
            - self._explore_center_y)
        wall_side_threshold = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        min_pose_lateral = max(
            0.25,
            min(
                0.32,
                max(0.25, self._nav_start_max_abs_y_m * 0.40),
                max(0.25, self._axis_door_min_side_goal_lateral_m * 0.60),
            ))

        if abs(handle_lateral) < wall_side_threshold:
            return True
        if abs(pose_lateral) >= min_pose_lateral:
            return True

        self.get_logger().info(
            f'파란 문 후보 {door.door_id} 제외: 손잡이는 벽 쪽이지만 '
            f'접근 pose가 중앙선에 가까움 '
            f'(pose_lateral={pose_lateral:.2f}, '
            f'handle_lateral={handle_lateral:.2f}, '
            f'min_pose_lateral={min_pose_lateral:.2f}).',
            throttle_duration_sec=3.0)
        return False

    def _direct_blue_candidate_matches_observed_wall_memory(
            self,
            door: DoorInfo,
            reference_xy: tuple[float, float] | None = None) -> bool:
        """Allow a live blue box with a wall-side handle to confirm map memory.

        Side camera projection can put the detected door pose near the corridor
        center while the handle/wall evidence is stable. Map memory may use that
        wall evidence for targeting, but only after repeated blue observations.
        """
        if door.door_color != 'blue' or door.door_id.startswith('observed_blue_'):
            return False
        if not self._door_has_map_identity(door):
            return False
        raw_xy = self._raw_blue_handle_xy_for_evidence(door)
        if raw_xy is None:
            return False

        merge_dist = max(
            0.45,
            self._observed_candidate_merge_dist(),
            min(
                max(0.20, self._door_open_fresh_blue_max_dist_m),
                self._observed_physical_door_merge_dist_m + 0.35))
        search_xy = reference_xy if reference_xy is not None else raw_xy
        cluster = self._find_observed_blue_cluster(
            search_xy,
            merge_dist=merge_dist)
        if cluster is None:
            cluster = self._find_observed_blue_cluster(
                raw_xy,
                merge_dist=merge_dist)
        if cluster is None:
            return False

        if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
            return False
        cluster_xy = self._cluster_xy(cluster)
        if self._is_blue_position_opened(cluster_xy) or self._is_blue_position_abandoned(cluster_xy):
            return False
        if int(cluster.get('count', 0.0)) < self._observed_blue_min_observations:
            return False

        confidence = max(
            float(cluster.get('confidence', 0.0)),
            float(door.confidence))
        if confidence < max(
                0.0,
                min(
                    self._observed_blue_target_min_confidence,
                    self._blue_target_immediate_min_confidence)):
            return False

        if reference_xy is not None:
            if math.hypot(reference_xy[0] - cluster_xy[0],
                          reference_xy[1] - cluster_xy[1]) > merge_dist:
                return False
            if math.hypot(reference_xy[0] - raw_xy[0],
                          reference_xy[1] - raw_xy[1]) > merge_dist:
                return False
        if math.hypot(raw_xy[0] - cluster_xy[0],
                      raw_xy[1] - cluster_xy[1]) > merge_dist:
            return False

        self.get_logger().info(
            f'Blue candidate {door.door_id} accepted as live wall evidence for '
            f'observed map memory at ({cluster_xy[0]:.1f}, {cluster_xy[1]:.1f}) '
            f'despite center-biased pose projection.',
            throttle_duration_sec=3.0)
        return True

    def _record_observed_physical_red_door(self, door: DoorInfo):
        if door.door_color != 'red' or not self._door_has_map_identity(door):
            return
        if float(door.confidence) < self._red_observation_min_confidence:
            return
        xy = self._door_handle_xy(door)
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
                cluster['confidence'] = max(
                    float(cluster.get('confidence', 0.0)),
                    float(door.confidence))
                cluster['last_seen'] = now_sec
                return
        self._observed_red_door_positions.append({
            'x': xy[0],
            'y': xy[1],
            'count': 1.0,
            'confidence': float(door.confidence),
            'last_seen': now_sec,
        })

    def _is_blue_xy_recordable_wall_observation(
            self, xy: tuple[float, float]) -> bool:
        lateral = abs(
            self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        min_lateral = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        if lateral < min_lateral:
            return False
        max_wall = max(0.0, self._observed_blue_max_abs_wall_y_m)
        if (
                max_wall > 0.0
                and lateral > max_wall + self._blue_handle_validation_overshoot_m):
            return False
        if self._is_blue_position_opened(xy):
            return False
        if self._is_blue_position_abandoned(xy):
            return False
        return True

    def _record_observed_physical_blue_door(self, door: DoorInfo):
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return
        identity_xy = self._door_identity_xy(door)
        raw_xy = self._raw_blue_handle_xy_for_evidence(door)
        xy = identity_xy
        valid_target_xy = self._is_blue_xy_at_valid_wall_position(xy)
        handle_in_wall_band = self._is_blue_handle_in_wall_band(door)
        direct_pose_consistent = self._is_direct_blue_pose_lateral_consistent(door)
        identity_recordable = self._is_blue_xy_recordable_wall_observation(identity_xy)
        if not valid_target_xy:
            raw_recordable = (
                raw_xy is not None
                and self._is_blue_xy_recordable_wall_observation(raw_xy))
            if raw_recordable and (not direct_pose_consistent or not identity_recordable):
                xy = raw_xy
                valid_target_xy = self._is_blue_xy_at_valid_wall_position(xy)
                handle_in_wall_band = True
                self.get_logger().info(
                    f'Blue candidate {door.door_id} recorded from wall-side '
                    'handle memory because the direct pose is center-biased.',
                    throttle_duration_sec=3.0)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return
        if not valid_target_xy and not self._is_blue_xy_recordable_wall_observation(xy):
            return
        if not handle_in_wall_band and not valid_target_xy:
            return
        if not handle_in_wall_band:
            self.get_logger().info(
                f'Blue candidate {door.door_id} kept in wall observation memory '
                'using its door pose; handle projection is outside the wall band.',
                throttle_duration_sec=3.0)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        confidence = float(door.confidence)
        cluster = self._observed_blue_door_id_to_cluster.get(door.door_id)
        if cluster is None:
            cluster = self._find_observed_blue_cluster(
                xy, merge_dist=self._observed_candidate_merge_dist())
        else:
            old_x = float(cluster.get('x', xy[0]))
            old_y = float(cluster.get('y', xy[1]))
            origin_x = float(cluster.get('origin_x', old_x))
            origin_y = float(cluster.get('origin_y', old_y))
            cluster['origin_x'] = origin_x
            cluster['origin_y'] = origin_y
            jump = math.hypot(xy[0] - old_x, xy[1] - old_y)
            total_drift = math.hypot(xy[0] - origin_x, xy[1] - origin_y)
            max_total = max(0.0, self._observed_blue_same_id_max_total_drift_m)
            if max_total > 0.0 and total_drift > max_total:
                self.get_logger().debug(
                    f'Blue candidate {door.door_id} total drift '
                    f'{total_drift:.1f}m from cluster origin. '
                    'Treating it as a separate physical observation.')
                cluster = self._find_observed_blue_cluster(
                    xy, merge_dist=self._observed_candidate_merge_dist())
            if (self._observed_blue_same_id_max_jump_m > 0.0
                    and jump > self._observed_blue_same_id_max_jump_m
                    and (max_total <= 0.0 or total_drift <= max_total)):
                self.get_logger().debug(
                    f'파란 문 후보 {door.door_id} 위치 점프 {jump:.1f}m. '
                    '같은 물리 문 평균 갱신 대신 새 관찰로 분리합니다.')
                cluster = self._find_observed_blue_cluster(
                    xy, merge_dist=self._observed_candidate_merge_dist())
        if cluster is not None:
            self._add_observed_blue_cluster_id(cluster, door.door_id)
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
            if not handle_in_wall_band:
                cluster['bad_handle_count'] = float(
                    int(cluster.get('bad_handle_count', 0.0)) + 1)
            self._observed_blue_door_id_to_cluster[door.door_id] = cluster
            return

        cluster = {
            'x': xy[0],
            'y': xy[1],
            'count': 1.0,
            'last_seen': now_sec,
            'confidence': confidence,
            'bad_handle_count': 0.0 if handle_in_wall_band else 1.0,
            'failures': 0.0,
            'approach_failures': 0.0,
            'open_failures': 0.0,
            'origin_x': xy[0],
            'origin_y': xy[1],
            'ids': set(self._door_id_keys(door.door_id)),
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

    def _add_observed_blue_cluster_id(self, cluster: dict[str, float], door_id: str):
        ids = cluster.get('ids')
        if isinstance(ids, set):
            id_set = ids
        elif isinstance(ids, (list, tuple)):
            id_set = set(str(item) for item in ids)
        elif isinstance(ids, str) and ids:
            id_set = {ids}
        else:
            id_set = set()
        id_set.update(self._door_id_keys(door_id))
        cluster['ids'] = id_set

    def _observed_blue_cluster_has_id_in_set(
            self, cluster: dict[str, float], target: set[str]) -> bool:
        ids = cluster.get('ids')
        if isinstance(ids, set):
            id_iter = ids
        elif isinstance(ids, (list, tuple)):
            id_iter = ids
        elif isinstance(ids, str) and ids:
            id_iter = (ids,)
        else:
            return False
        return any(self._is_door_id_in_set(target, str(door_id)) for door_id in id_iter)

    def _stable_observed_blue_doors(self) -> list[dict[str, float]]:
        return [
            cluster for cluster in self._observed_blue_doors
            if int(cluster.get('count', 0.0)) >= self._observed_blue_min_observations
            and not self._is_observed_blue_cluster_suppressed(cluster)
        ]

    def _is_blue_xy_too_far_ahead_for_target(
            self, xy: tuple[float, float], label: str = 'blue candidate') -> bool:
        max_ahead = max(0.0, self._blue_target_max_robot_ahead_m)
        if max_ahead <= 0.0:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False
        robot_progress = self._axis_progress_xy(pose[0], pose[1])
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        ahead = candidate_progress - robot_progress
        if ahead <= max_ahead:
            return False
        self.get_logger().info(
            f'{label} waiting for closer confirmation: '
            f'ahead={ahead:.1f}m > {max_ahead:.1f}m.',
            throttle_duration_sec=3.0)
        return True

    def _blue_xy_behind_distance_from_robot(
            self, xy: tuple[float, float]) -> float | None:
        pose = self._current_map_pose()
        if pose is None:
            return None
        robot_progress = self._axis_progress_xy(pose[0], pose[1])
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        return robot_progress - candidate_progress

    def _is_blue_xy_too_far_behind_for_target(self, xy: tuple[float, float]) -> bool:
        max_behind = self._observed_blue_target_max_behind_m
        if max_behind < 0.0:
            return False
        behind = self._blue_xy_behind_distance_from_robot(xy)
        if behind is None or behind <= max_behind:
            return False
        self.get_logger().info(
            f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
            f'현재 진행 위치보다 {behind:.1f}m 뒤쪽입니다 '
            f'(limit={max_behind:.1f}m).',
            throttle_duration_sec=3.0)
        return True

    def _is_observed_blue_cluster_too_far_behind_for_target(
            self, cluster: dict[str, float]) -> bool:
        max_behind = self._observed_blue_target_max_behind_m
        if max_behind < 0.0:
            return False
        xy = self._cluster_xy(cluster)
        behind = self._blue_xy_behind_distance_from_robot(xy)
        if behind is None or behind <= max_behind:
            return False
        if self._observed_blue_cluster_is_stable_unopened_target_memory(cluster):
            self.get_logger().warn(
                f'관측 지도에 남은 미개방 파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) '
                f'재접근 허용: 현재 위치보다 {behind:.1f}m 뒤쪽이지만 '
                '충분히 반복 관측된 후보입니다.',
                throttle_duration_sec=3.0)
            return False
        self.get_logger().info(
            f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
            f'현재 진행 위치보다 {behind:.1f}m 뒤쪽입니다 '
            f'(limit={max_behind:.1f}m).',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_xy_beyond_explore_limit(
            self,
            xy: tuple[float, float],
            label: str = 'blue candidate') -> bool:
        limit = self._explore_limit_progress()
        if not math.isfinite(limit):
            return False
        margin = max(0.0, self._blue_target_max_beyond_explore_limit_m)
        progress = self._axis_progress_xy(xy[0], xy[1])
        if progress <= limit + margin:
            return False
        self.get_logger().info(
            f'{label} rejected: progress={progress:.1f}m is beyond the '
            f'observed exploration limit {limit:.1f}m '
            f'(margin={margin:.1f}m).',
            throttle_duration_sec=3.0)
        return True

    def _is_exit_blocking_blue_xy_beyond_scan_window(
            self, xy: tuple[float, float]) -> bool:
        limit = self._explore_limit_progress()
        if not math.isfinite(limit):
            return False
        margin = max(
            0.0,
            self._blue_target_max_beyond_explore_limit_m,
            self._observed_exit_explore_grace_m + 1.25)
        progress = self._axis_progress_xy(xy[0], xy[1])
        if progress <= limit + margin:
            return False
        self.get_logger().info(
            f'exit-blocking observed blue candidate rejected: '
            f'progress={progress:.1f}m is beyond the observed exploration '
            f'limit {limit:.1f}m (exit-block margin={margin:.1f}m).',
            throttle_duration_sec=3.0)
        return True

    def _observed_blue_cluster_is_target_ready(
            self, cluster: dict[str, float]) -> bool:
        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        min_conf = max(0.0, self._observed_blue_target_min_confidence)
        if confidence >= min_conf:
            return True

        min_count = max(
            self._observed_blue_min_observations,
            self._observed_blue_low_conf_target_min_observations)
        low_conf_min = max(0.0, self._explore_interrupt_min_confidence)
        if count >= min_count and confidence >= low_conf_min:
            self.get_logger().info(
                f'Observed blue cluster accepted as low-confidence target; '
                f'opening will still require live blue evidence: '
                f'count={count}, conf={confidence:.2f}/{min_conf:.2f}.',
                throttle_duration_sec=3.0)
            return True

        self.get_logger().info(
            f'Observed blue cluster waiting for stronger evidence before target: '
            f'count={count}/{min_count}, conf={confidence:.2f}/{min_conf:.2f}.',
            throttle_duration_sec=3.0)
        return False


    def _observed_blue_cluster_is_stable_unopened_target_memory(
            self, cluster: dict[str, float]) -> bool:
        if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
            return False
        xy = self._cluster_xy(cluster)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        min_count = max(
            self._observed_blue_min_observations,
            self._observed_blue_low_conf_target_min_observations)
        min_conf = max(0.0, self._observed_blue_target_min_confidence)
        if count < min_count or confidence < min_conf:
            return False
        if self._semantic_blue_xy_suppressed(xy):
            return False
        if self._is_blue_xy_near_detected_exit(xy):
            return False
        if self._is_blue_xy_at_or_after_observed_exit(xy):
            return False
        if self._is_observed_blue_cluster_weak_in_pre_exit_zone(cluster):
            return False
        if self._is_blue_xy_opposite_opened_station(xy):
            return False
        if self._is_blue_xy_near_abandoned_station(xy):
            return False
        if self._is_blue_xy_near_observed_red(xy):
            return False
        if not self._is_blue_xy_at_valid_wall_position(xy):
            return False
        landmark = self._find_semantic_door_landmark(xy)
        if landmark is not None:
            blue_count, red_count, green_count = self._semantic_landmark_color_votes(landmark)
            if blue_count <= red_count + green_count:
                return False
        return True

    def _blue_xy_same_side_short_gap_after_opened(
            self,
            xy: tuple[float, float],
            label: str,
            min_progress: float | None = None) -> bool:
        # Layout-specific same-side progress gaps are intentionally disabled.
        # Opened/failed door suppression is handled by physical map-distance
        # merging so new worlds do not depend on corridor spacing assumptions.
        return False

    def _is_blue_door_same_side_short_gap_after_opened(
            self, door: DoorInfo) -> bool:
        return False

    def _observed_blue_cluster_same_side_too_close_to_opened(
            self, cluster: dict[str, float]) -> bool:
        return False

    def _stable_unopened_observed_blue_targets(self) -> list[DoorInfo]:
        targets: list[DoorInfo] = []
        for cluster in self._stable_observed_blue_doors():
            xy = self._cluster_xy(cluster)
            if self._is_observed_blue_cluster_suppressed(cluster):
                continue
            if not self._observed_blue_cluster_is_target_ready(cluster):
                continue
            if self._is_blue_xy_beyond_explore_limit(xy, 'observed blue candidate'):
                continue
            if self._is_observed_blue_cluster_too_far_behind_for_target(cluster):
                continue
            if self._is_blue_xy_too_far_ahead_for_target(
                    xy, 'observed blue candidate'):
                continue
            if (
                    self._is_blue_xy_at_or_after_observed_exit(xy)
                    or self._is_blue_xy_near_detected_exit(xy)):
                continue
            if self._is_observed_blue_cluster_weak_in_pre_exit_zone(cluster):
                continue
            if not self._is_blue_xy_at_valid_wall_position(xy):
                continue
            if not self._observed_blue_cluster_has_fresh_direct_evidence(cluster):
                self.get_logger().info(
                    f'Observed blue cluster at ({xy[0]:.1f}, {xy[1]:.1f}) '
                    'selected from map memory without a current camera hit; '
                    'opening will still require live blue confirmation.',
                    throttle_duration_sec=3.0)
            targets.append(self._observed_blue_cluster_to_door(cluster))
        return targets

    def _observed_blue_cluster_has_fresh_direct_evidence(
            self, cluster: dict[str, float]) -> bool:
        xy = self._cluster_xy(cluster)
        max_age = max(0.0, self._door_open_fresh_blue_max_age_sec)
        max_dist = max(
            0.45,
            self._door_open_fresh_blue_max_dist_m,
            self._observed_candidate_merge_dist())
        min_conf = max(
            0.0,
            min(
                self._observed_blue_target_min_confidence,
                self._door_open_fresh_blue_min_confidence))
        now = self.get_clock().now()
        for candidate in self.detected_doors:
            if candidate.door_color != 'blue':
                continue
            if candidate.door_id.startswith('observed_blue_'):
                continue
            if not self._door_has_map_identity(candidate):
                continue
            if float(candidate.confidence) < min_conf:
                continue
            if (not self._is_direct_blue_pose_lateral_consistent(candidate)
                    and not self._direct_blue_candidate_matches_observed_wall_memory(
                        candidate, xy)):
                continue
            if max_age > 0.0:
                try:
                    stamp = Time.from_msg(candidate.header.stamp)
                except Exception:
                    stamp = None
                if stamp is not None and stamp.nanoseconds > 0:
                    age = (now - stamp).nanoseconds / 1e9
                    if age > max_age:
                        continue
            for candidate_xy in self._blue_evidence_points(candidate):
                if math.hypot(xy[0] - candidate_xy[0], xy[1] - candidate_xy[1]) <= max_dist:
                    return True
        return False

    def _observed_blue_cluster_to_door(
            self, cluster: dict[str, float]) -> DoorInfo:
        x, y = self._cluster_xy(cluster)
        progress = self._axis_progress_xy(x, y)
        lateral = self._axis_lateral_xy(x, y)
        progress_key = f'{progress:.1f}'.replace('-', 'm').replace('.', 'p')
        lateral_key = f'{lateral:.1f}'.replace('-', 'm').replace('.', 'p')

        msg = DoorInfo()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = 'map'
        msg.door_id = f'observed_blue_{progress_key}_{lateral_key}'
        msg.door_color = 'blue'
        msg.confidence = float(cluster.get('confidence', 1.0))
        msg.is_open = False
        msg.distance_from_fire = 0.0
        msg.door_pose.header.stamp = now
        msg.door_pose.header.frame_id = 'map'
        msg.door_pose.pose.position.x = x
        msg.door_pose.pose.position.y = y
        msg.door_pose.pose.orientation.w = 1.0
        msg.handle_position.header.stamp = now
        msg.handle_position.header.frame_id = 'map'
        msg.handle_position.point.x = x
        msg.handle_position.point.y = y
        msg.handle_position.point.z = 0.9
        return msg

    def _merge_blue_target_lists(
            self,
            primary: list[DoorInfo],
            secondary: list[DoorInfo]) -> list[DoorInfo]:
        merged: list[DoorInfo] = []
        merge_dist = max(
            self._observed_candidate_merge_dist(),
            self._opened_physical_door_merge_dist_m)

        def add_if_new(door: DoorInfo):
            if not self._door_has_map_identity(door):
                return
            xy = self._door_identity_xy(door)
            for existing in merged:
                exy = self._door_identity_xy(existing)
                if math.hypot(xy[0] - exy[0], xy[1] - exy[1]) <= merge_dist:
                    return
            merged.append(door)

        for door in primary:
            add_if_new(door)
        for door in secondary:
            add_if_new(door)
        return merged

    def _is_blue_position_opened(self, xy: tuple[float, float]) -> bool:
        if self._semantic_blue_xy_opened(xy):
            return True
        return any(
            self._blue_xy_matches_physical_station(
                xy, old, self._opened_physical_door_merge_dist_m)
            for old in self._opened_blue_door_positions
        )

    def _is_blue_position_abandoned(self, xy: tuple[float, float]) -> bool:
        if self._semantic_blue_xy_abandoned(xy):
            return True
        return any(
            self._blue_xy_matches_physical_station(
                xy, old, self._opened_physical_door_merge_dist_m)
            for old in self._abandoned_blue_door_positions
        )

    def _blue_xy_matches_physical_station(
            self,
            xy: tuple[float, float],
            station_xy: tuple[float, float],
            merge_dist: float) -> bool:
        euclidean_dist = math.hypot(xy[0] - station_xy[0], xy[1] - station_xy[1])
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        station_progress = self._axis_progress_xy(station_xy[0], station_xy[1])
        progress_gap = abs(candidate_progress - station_progress)
        candidate_side = self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y
        station_side = (
            self._axis_lateral_xy(station_xy[0], station_xy[1])
            - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        candidate_on_wall = abs(candidate_side) >= wall_side_threshold
        station_on_wall = abs(station_side) >= wall_side_threshold
        lateral_gap = abs(candidate_side - station_side)

        if candidate_on_wall and station_on_wall:
            if candidate_side * station_side <= 0.0:
                return euclidean_dist <= min(merge_dist, 0.35)
            if progress_gap > max(merge_dist, self._observed_candidate_merge_dist()):
                return False
            return lateral_gap <= max(
                1.25,
                self._axis_door_side_standoff_m + 0.55)

        if (
                candidate_on_wall != station_on_wall
                and euclidean_dist > min(merge_dist, 0.55)):
            return False

        if progress_gap > max(merge_dist, self._observed_candidate_merge_dist()):
            return False
        return euclidean_dist <= min(merge_dist, 0.85)

    def _is_door_physically_opened(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if (
                cluster is not None
                and self._observed_blue_cluster_retry_pending(cluster)):
            return False
        return self._is_blue_position_opened(self._door_identity_xy(door))

    def _observed_blue_cluster_retry_pending(
            self, cluster: dict[str, float]) -> bool:
        if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
            return False
        xy = self._cluster_xy(cluster)
        if self._is_blue_position_abandoned(xy):
            return False
        if self._maybe_abandon_stale_observed_blue_cluster(cluster):
            return False
        approach_failures = int(
            cluster.get('approach_failures', cluster.get('failures', 0.0)))
        open_failures = int(cluster.get('open_failures', 0.0))
        if approach_failures <= 0 and open_failures <= 0:
            return False
        if approach_failures >= self._max_door_approach_failures_before_abandon:
            return False
        if open_failures >= self._max_door_open_failures_before_abandon:
            return False
        if self._semantic_blue_xy_suppressed(xy):
            return False
        return True

    def _is_observed_blue_cluster_suppressed(
            self, cluster: dict[str, float]) -> bool:
        if cluster.get('opened', 0.0) >= 1.0:
            return True
        if cluster.get('abandoned', 0.0) >= 1.0:
            return True
        if self._observed_blue_cluster_retry_pending(cluster):
            return False
        xy = self._cluster_xy(cluster)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return True
        if (self._observed_blue_cluster_has_id_in_set(cluster, self._opened_door_ids)
                and self._is_blue_position_opened(xy)):
            return True
        if (self._observed_blue_cluster_has_id_in_set(cluster, self._abandoned_door_ids)
                and self._is_blue_position_abandoned(xy)):
            return True
        if self._maybe_abandon_stale_observed_blue_cluster(cluster):
            return True
        recheckable_wall_blue = (
            self._observed_blue_cluster_has_recheckable_wall_memory(cluster))
        exit_sensitive_zone = (
            self._is_blue_xy_in_pre_exit_zone(xy)
            or self._is_blue_xy_in_exit_search_tail(xy)
            or self._is_blue_xy_near_detected_exit(xy))
        if self._semantic_blue_xy_suppressed(xy):
            if recheckable_wall_blue and not exit_sensitive_zone:
                self.get_logger().info(
                    f'Observed blue cluster ({xy[0]:.1f}, {xy[1]:.1f}) '
                    'will be rechecked despite mixed semantic color memory.',
                    throttle_duration_sec=3.0)
            else:
                return True
        if self._is_blue_xy_near_detected_exit(xy):
            exit_xy = self._exit_xy_for_blue_candidate_filter()
            if (
                    recheckable_wall_blue
                    and exit_xy is not None
                    and self._blue_xy_reliable_side_wall_candidate_near_exit(
                        xy, exit_xy)):
                self.get_logger().info(
                    f'Observed blue cluster ({xy[0]:.1f}, {xy[1]:.1f}) '
                    'will be rechecked despite tentative exit proximity.',
                    throttle_duration_sec=3.0)
            else:
                return True
        if self._is_observed_blue_cluster_weak_in_pre_exit_zone(cluster):
            return True
        if self._is_blue_xy_opposite_opened_station(xy):
            return True
        if self._is_blue_xy_near_abandoned_station(xy):
            return True
        if self._is_blue_xy_near_observed_red(xy):
            if recheckable_wall_blue and not exit_sensitive_zone:
                self.get_logger().info(
                    f'Observed blue cluster ({xy[0]:.1f}, {xy[1]:.1f}) '
                    'will be rechecked despite nearby red memory.',
                    throttle_duration_sec=3.0)
            else:
                return True
        approach_failures = int(
            cluster.get('approach_failures', cluster.get('failures', 0.0)))
        open_failures = int(cluster.get('open_failures', 0.0))
        return (
            approach_failures >= self._max_door_approach_failures_before_abandon
            or open_failures >= self._max_door_open_failures_before_abandon
        )

    def _observed_blue_cluster_has_recheckable_wall_memory(
            self, cluster: dict[str, float]) -> bool:
        if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
            return False
        xy = self._cluster_xy(cluster)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False
        wall_lateral = abs(
            self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        min_lateral = max(0.35, self._observed_blue_min_abs_wall_y_m * 0.5)
        max_lateral = max(0.0, self._observed_blue_max_abs_wall_y_m)
        if wall_lateral < min_lateral:
            return False
        if (
                max_lateral > 0.0
                and wall_lateral > max_lateral + self._blue_handle_validation_overshoot_m):
            return False
        if self._is_blue_xy_opposite_opened_station(xy):
            return False
        if self._is_blue_xy_near_abandoned_station(xy):
            return False
        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        min_count = max(2, self._observed_blue_min_observations)
        min_conf = max(
            0.30,
            min(
                max(0.0, self._observed_blue_target_min_confidence),
                max(0.0, self._door_open_fresh_blue_min_confidence)))
        return count >= min_count and confidence >= min_conf

    def _is_door_observation_suppressed(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        if self._is_blue_door_weak_in_pre_exit_zone(door):
            return True
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        if self._is_weak_observed_blue_cluster_revisit_candidate(cluster):
            return False
        return self._is_observed_blue_cluster_suppressed(cluster)

    def _is_weak_observed_blue_cluster_revisit_candidate(
            self, cluster: dict[str, float]) -> bool:
        count = int(cluster.get('count', 0.0))
        if count <= 0 or count >= self._observed_blue_min_observations:
            return False
        confidence = float(cluster.get('confidence', 0.0))
        min_confidence = min(
            max(0.0, self._blue_target_immediate_min_confidence),
            max(0.0, self._explore_interrupt_min_confidence))
        if confidence < min_confidence:
            return False
        if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
            return False
        xy = self._cluster_xy(cluster)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        if self._maybe_abandon_stale_observed_blue_cluster(cluster):
            return False
        if self._is_blue_xy_too_far_behind_for_target(xy):
            return False
        if self._is_blue_xy_beyond_explore_limit(xy, 'weak observed blue candidate'):
            return False
        if not self._is_blue_xy_at_valid_wall_position(xy):
            return False
        if self._is_blue_xy_near_detected_exit(xy):
            return False
        if self._is_observed_blue_cluster_weak_in_pre_exit_zone(cluster):
            return False
        if self._is_blue_xy_opposite_opened_station(xy):
            return False
        if self._is_blue_xy_near_abandoned_station(xy):
            return False
        if self._is_blue_xy_near_observed_red(xy):
            return False

        landmark = self._find_semantic_door_landmark(xy)
        if landmark is not None:
            blue_count, red_count, green_count = self._semantic_landmark_color_votes(landmark)
            if blue_count <= red_count + green_count:
                return False
        return True

    def _is_blue_door_observation_confirmed(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        return int(cluster.get('count', 0.0)) >= self._observed_blue_min_observations

    def _is_blue_door_target_ready(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        if (
                self._is_blue_xy_at_or_after_observed_exit(xy)
                or self._is_blue_xy_near_detected_exit(xy)):
            return False
        if self._blue_target_in_strict_exit_tail_without_persistent_memory(door):
            return False
        if door.door_id.startswith('observed_blue_'):
            return True

        if self._semantic_blue_xy_confirmed(xy):
            return True
        if self._is_blue_door_observation_confirmed(door):
            return True

        cluster = self._find_observed_blue_cluster(
            xy,
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is not None and not self._is_observed_blue_cluster_suppressed(cluster):
            count = int(cluster.get('count', 0.0))
            if count >= self._observed_blue_min_observations:
                return True

        exit_sensitive = (
            self._is_blue_xy_in_pre_exit_zone(xy)
            or self._is_blue_xy_near_detected_exit(xy))
        if not exit_sensitive:
            threshold = max(
                0.0,
                max(
                    self._blue_target_immediate_min_confidence,
                    self._explore_interrupt_min_confidence))
            if threshold <= 0.0 or abs(float(door.confidence)) >= threshold:
                return True

        self.get_logger().info(
            f'Blue candidate {door.door_id} waiting for repeated observed '
            'map evidence before becoming a navigation target.',
            throttle_duration_sec=3.0)
        return False

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

    def _is_blue_candidate_blocked_by_red(self, door: DoorInfo) -> bool:
        if not self._is_blue_candidate_near_recent_red(door):
            return False
        self.get_logger().info(
            f'Blue candidate {door.door_id} rejected near recent red observation '
            f'(conf={float(door.confidence):.2f}).',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_xy_in_exit_search_tail(self, xy: tuple[float, float]) -> bool:
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False
        if not self._opened_blue_door_positions:
            return False
        min_after = max(
            0.0,
            self._detected_exit_min_after_opened_blue_m,
            self._front_wall_exit_min_after_opened_blue_m,
            self._detected_exit_min_robot_past_opened_blue_m)
        if min_after <= 0.0:
            return False
        farthest_opened = max(
            self._axis_progress_xy(x, y)
            for x, y in self._opened_blue_door_positions)
        return self._axis_progress_xy(xy[0], xy[1]) >= farthest_opened + min_after

    def _is_blue_candidate_near_recent_red(self, door: DoorInfo) -> bool:
        if self._blue_red_conflict_dist_m <= 0.0:
            return False
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        blue_xy = self._door_identity_xy(door)
        evidence_points = self._unique_xy_points(self._blue_evidence_points(door))
        if not evidence_points:
            evidence_points = [blue_xy]
        blue_confidence = float(door.confidence)
        strong_blue, strong_confidence, strong_count = (
            self._blue_target_observation_strength(door))
        cluster_for_red_yield = self._find_observed_blue_cluster(
            blue_xy, merge_dist=self._observed_candidate_merge_dist())
        recheckable_wall_blue = (
            cluster_for_red_yield is not None
            and self._observed_blue_cluster_has_recheckable_wall_memory(
                cluster_for_red_yield))
        if cluster_for_red_yield is not None:
            strong_count = max(
                strong_count,
                int(cluster_for_red_yield.get('count', 0.0)))
            strong_confidence = max(
                strong_confidence,
                float(cluster_for_red_yield.get('confidence', 0.0)))
        in_pre_exit_zone = self._is_blue_xy_in_pre_exit_zone(blue_xy)
        exit_seen = (
            self._exit_door is not None
            or self._last_valid_exit_xy is not None
            or self._final_scan_start_time is not None
            or self.state == State.EXITING)
        exit_sensitive_blue = (
            in_pre_exit_zone
            or self._is_blue_xy_in_exit_search_tail(blue_xy)
            or self._is_blue_xy_near_detected_exit(blue_xy))
        normal_red_bypass_zone = (
            not exit_seen
            and not in_pre_exit_zone
            and not self._is_blue_xy_in_exit_search_tail(blue_xy))
        side_wall_memory_bypass = strong_blue and any(
            self._blue_xy_has_unopened_openable_side_wall_memory(point)
            for point in evidence_points)
        if recheckable_wall_blue:
            side_wall_memory_bypass = True
        red_bypass_ready = (
            (normal_red_bypass_zone or side_wall_memory_bypass or recheckable_wall_blue)
            and (
                any(
                    self._blue_observation_confident_enough_to_bypass_red(
                        point, blue_confidence)
                    for point in evidence_points)
                or strong_blue
                or recheckable_wall_blue))
        strict_dist = max(0.0, self._blue_red_conflict_strict_dist_m)
        for red in self.detected_doors:
            if red.door_color != 'red':
                continue
            if not self._is_door_recent_observation(red):
                continue
            if not self._door_has_map_identity(red):
                continue
            if float(red.confidence) < self._red_observation_min_confidence:
                continue
            red_xy = self._door_handle_xy(red)
            conflict_checks = [
                (point, *self._red_xy_conflicts_with_blue_xy(
                    point, red_xy, self._blue_red_conflict_dist_m))
                for point in evidence_points
            ]
            conflicts = [check for check in conflict_checks if check[1]]
            if not conflicts:
                continue
            stable_non_conflicting_wall_point = any(
                not point_conflicts
                and self._blue_xy_has_unopened_openable_side_wall_memory(point)
                for point, point_conflicts, _ in conflict_checks)
            if side_wall_memory_bypass and stable_non_conflicting_wall_point:
                self.get_logger().info(
                    f'Blue candidate {door.door_id} accepted despite nearby red '
                    'observation: wall-side blue evidence point remains distinct.',
                    throttle_duration_sec=3.0)
                continue
            distance = min(check[2] for check in conflicts)
            if recheckable_wall_blue and not exit_sensitive_blue:
                self.get_logger().info(
                    f'Blue candidate {door.door_id} accepted despite nearby red '
                    'observation: repeated wall-side blue map memory will be '
                    'rechecked before opening.',
                    throttle_duration_sec=3.0)
                continue
            if (
                    red_bypass_ready
                    and distance > strict_dist
                    and float(red.confidence) <= max(
                        blue_confidence, strong_confidence) + 0.12):
                self.get_logger().info(
                    f'Blue candidate {door.door_id} accepted despite nearby red '
                    f'observation: repeated blue evidence wins '
                    f'(blue_count={strong_count}, blue_conf='
                    f'{max(blue_confidence, strong_confidence):.2f}, '
                    f'red_conf={float(red.confidence):.2f}, '
                    f'dist={distance:.2f}m).',
                    throttle_duration_sec=3.0)
                continue
            return True
        if side_wall_memory_bypass or strong_blue:
            for point in evidence_points:
                if not self._is_blue_xy_near_observed_red(
                        point, door.door_id, blue_confidence):
                    return False
            return True
        return self._is_blue_xy_near_observed_red(
            blue_xy, door.door_id, blue_confidence)

    def _blue_observation_confident_enough_to_bypass_red(
            self, blue_xy: tuple[float, float],
            confidence: float = 0.0) -> bool:
        count_limit = self._confirmed_blue_bypass_red_conflict_count
        confidence_limit = self._confirmed_blue_bypass_red_conflict_confidence
        if count_limit <= 0:
            return False
        cluster = self._find_observed_blue_cluster(
            blue_xy, merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        count = int(cluster.get('count', 0.0))
        best_confidence = max(
            float(cluster.get('confidence', 0.0)),
            float(confidence))
        return count >= count_limit and best_confidence >= confidence_limit

    def _red_xy_conflicts_with_blue_xy(
            self,
            blue_xy: tuple[float, float],
            red_xy: tuple[float, float],
            conflict_dist: float) -> tuple[bool, float]:
        if conflict_dist <= 0.0:
            return False, math.inf
        distance = math.hypot(blue_xy[0] - red_xy[0], blue_xy[1] - red_xy[1])

        blue_side = (
            self._axis_lateral_xy(blue_xy[0], blue_xy[1])
            - self._explore_center_y)
        red_side = (
            self._axis_lateral_xy(red_xy[0], red_xy[1])
            - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        strict_dist = max(0.0, self._blue_red_conflict_strict_dist_m)
        blue_on_wall = abs(blue_side) >= wall_side_threshold
        red_on_wall = abs(red_side) >= wall_side_threshold

        lateral_gap = abs(blue_side - red_side)

        if blue_on_wall and red_on_wall:
            if blue_side * red_side < 0.0:
                return False, distance
            blue_progress = self._axis_progress_xy(blue_xy[0], blue_xy[1])
            red_progress = self._axis_progress_xy(red_xy[0], red_xy[1])
            forward_separation = blue_progress - red_progress
            if forward_separation >= max(conflict_dist, 2.35):
                return False, distance
            if distance <= strict_dist:
                return True, distance
            same_wall_progress_window = max(conflict_dist, 1.35)
            same_wall_lateral_window = max(
                0.35,
                min(0.55, wall_side_threshold + 0.25))
            progress_gap = abs(blue_progress - red_progress)
            if lateral_gap > same_wall_lateral_window:
                blue_is_more_wall_side = (
                    abs(blue_side) >= abs(red_side) + 0.35)
                if blue_is_more_wall_side:
                    return False, distance
                return True, distance
            if distance <= conflict_dist:
                return True, distance
            if (
                    progress_gap <= same_wall_progress_window
                    and distance <= max(conflict_dist, strict_dist + 0.20)):
                return True, distance
            return False, distance

        if distance > conflict_dist:
            return False, distance
        if blue_on_wall and not red_on_wall and distance > strict_dist:
            return False, distance
        if red_on_wall and not blue_on_wall and distance > strict_dist:
            return False, distance
        return True, distance

    def _blue_red_on_same_wall_side(
            self,
            blue_xy: tuple[float, float],
            red_xy: tuple[float, float]) -> bool:
        blue_side = (
            self._axis_lateral_xy(blue_xy[0], blue_xy[1])
            - self._explore_center_y)
        red_side = (
            self._axis_lateral_xy(red_xy[0], red_xy[1])
            - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        return (
            abs(blue_side) >= wall_side_threshold
            and abs(red_side) >= wall_side_threshold
            and blue_side * red_side > 0.0)

    def _cluster_xy(self, cluster: dict[str, float]) -> tuple[float, float]:
        return (
            float(cluster.get('x', 0.0)),
            float(cluster.get('y', 0.0)),
        )

    def _is_blue_xy_near_observed_red(
            self, blue_xy: tuple[float, float],
        candidate_id: str = '',
        confidence: float = 0.0) -> bool:
        in_pre_exit_zone = self._is_blue_xy_in_pre_exit_zone(blue_xy)
        exit_seen = (
            self._exit_door is not None
            or self._last_valid_exit_xy is not None
            or self._final_scan_start_time is not None
            or self.state == State.EXITING)
        cluster_for_red_yield = self._find_observed_blue_cluster(
            blue_xy, merge_dist=self._observed_candidate_merge_dist())
        memory_yield_allowed = bool(candidate_id) or cluster_for_red_yield is not None
        strong_memory_blue = False
        memory_blue_confidence = float(confidence)
        memory_blue_count = 0
        if cluster_for_red_yield is not None:
            memory_blue_count = int(cluster_for_red_yield.get('count', 0.0))
            memory_blue_confidence = max(
                memory_blue_confidence,
                float(cluster_for_red_yield.get('confidence', 0.0)))
            strong_memory_blue = (
                memory_blue_count >= max(
                    self._observed_blue_min_observations,
                    self._confirmed_blue_bypass_red_conflict_count)
                and memory_blue_confidence >= max(
                    0.45,
                    self._confirmed_blue_bypass_red_conflict_confidence))
        allow_red_bypass = (
            not exit_seen
            and not in_pre_exit_zone
            and not self._is_blue_xy_in_exit_search_tail(blue_xy))
        red_bypass_ready = (
            allow_red_bypass
            and (
                self._blue_observation_confident_enough_to_bypass_red(
                    blue_xy, confidence)
                or strong_memory_blue))
        conflict_dist = self._blue_red_conflict_dist_m
        exit_sensitive_blue = (
            in_pre_exit_zone
            or self._is_blue_xy_near_detected_exit(blue_xy)
            or self._is_blue_xy_in_exit_search_tail(blue_xy))
        if (
                exit_sensitive_blue
                and (
                    self._valid_detected_exit()
                    or self._final_scan_start_time is not None
                    or self.state == State.EXITING)):
            conflict_dist = max(
                conflict_dist,
                self._blue_red_conflict_after_exit_dist_m)
        now_sec = self.get_clock().now().nanoseconds / 1e9
        blue_lateral = self._axis_lateral_xy(blue_xy[0], blue_xy[1])
        blue_side = blue_lateral - self._explore_center_y
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        for red in self._observed_red_door_positions:
            if self._red_blue_conflict_memory_sec > 0.0:
                age = now_sec - float(red.get('last_seen', now_sec))
                if age > self._red_blue_conflict_memory_sec:
                    continue
            if int(red.get('count', 0.0)) < self._red_observation_min_count_for_blue_suppression:
                continue
            if float(red.get('confidence', 0.0)) < self._red_observation_min_confidence:
                continue
            red_x = float(red.get('x', 0.0))
            red_y = float(red.get('y', 0.0))
            conflicts, distance = self._red_xy_conflicts_with_blue_xy(
                blue_xy, (red_x, red_y), conflict_dist)
            strict_dist = max(0.0, self._blue_red_conflict_strict_dist_m)
            if not conflicts:
                continue
            strict_same_wall_red = self._same_wall_red_conflict_is_strict(
                blue_xy, (red_x, red_y), distance, conflict_dist)
            if (
                    not strict_same_wall_red
                    and
                    memory_yield_allowed
                    and self._red_memory_yields_to_blue_target(
                        red,
                        strong_memory_blue,
                        memory_blue_confidence,
                        memory_blue_count)):
                continue
            if (
                    not strict_same_wall_red
                    and
                    cluster_for_red_yield is not None
                    and self._observed_blue_cluster_has_recheckable_wall_memory(
                        cluster_for_red_yield)
                    and not exit_sensitive_blue):
                self.get_logger().info(
                    f'파란 문 후보 {candidate_id or f"cluster@({blue_xy[0]:.1f}, {blue_xy[1]:.1f})"} '
                    '유지: 반복 관측된 벽면 파란문 메모리를 빨간문 근접 메모리보다 '
                    '우선 재확인합니다.',
                    throttle_duration_sec=3.0)
                continue
            if (
                    not strict_same_wall_red
                    and
                    red_bypass_ready
                    and distance > strict_dist
                    and not self._blue_red_on_same_wall_side(
                        blue_xy, (red_x, red_y))):
                continue
            label = candidate_id or f'cluster@({blue_xy[0]:.1f}, {blue_xy[1]:.1f})'
            self.get_logger().info(
                f'파란 문 후보 {label} 제외: 최근 관찰된 빨간 문 위치 '
                f'({red_x:.1f}, {red_y:.1f})와 겹칩니다.',
                once=True)
            return True
        return False

    def _blue_xy_conflicts_recent_red_memory(
            self,
            blue_xy: tuple[float, float],
            conflict_dist: float | None = None) -> bool:
        limit = (
            max(0.0, self._blue_red_conflict_dist_m)
            if conflict_dist is None
            else max(0.0, conflict_dist))
        if limit <= 0.0:
            return False
        now_sec = self.get_clock().now().nanoseconds / 1e9
        for red in self._observed_red_door_positions:
            if self._red_blue_conflict_memory_sec > 0.0:
                age = now_sec - float(red.get('last_seen', now_sec))
                if age > self._red_blue_conflict_memory_sec:
                    continue
            if int(red.get('count', 0.0)) < self._red_observation_min_count_for_blue_suppression:
                continue
            if float(red.get('confidence', 0.0)) < self._red_observation_min_confidence:
                continue
            red_xy = (
                float(red.get('x', 0.0)),
                float(red.get('y', 0.0)))
            conflicts, _ = self._red_xy_conflicts_with_blue_xy(
                blue_xy, red_xy, limit)
            if conflicts:
                return True

        for red in self.detected_doors:
            if red.door_color != 'red':
                continue
            if not self._is_door_recent_observation(red):
                continue
            if not self._door_has_map_identity(red):
                continue
            if float(red.confidence) < self._red_observation_min_confidence:
                continue
            conflicts, _ = self._red_xy_conflicts_with_blue_xy(
                blue_xy, self._door_handle_xy(red), limit)
            if conflicts:
                return True
        return False

    def _same_wall_red_conflict_is_strict(
            self,
            blue_xy: tuple[float, float],
            red_xy: tuple[float, float],
            distance: float,
            conflict_dist: float) -> bool:
        if not self._blue_red_on_same_wall_side(blue_xy, red_xy):
            return False
        strict_dist = max(0.0, self._blue_red_conflict_strict_dist_m)
        blue_progress = self._axis_progress_xy(blue_xy[0], blue_xy[1])
        red_progress = self._axis_progress_xy(red_xy[0], red_xy[1])
        progress_gap = abs(blue_progress - red_progress)
        return (
            distance <= max(conflict_dist, strict_dist + 0.20)
            and progress_gap <= max(conflict_dist, 1.35))

    def _blue_door_has_openable_wall_lateral(self, door: DoorInfo) -> bool:
        min_lateral = max(0.0, self._door_open_axis_min_side_lateral_m)
        if min_lateral <= 0.0:
            return True
        points = [self._door_identity_xy(door), self._door_handle_xy(door)]
        best_lateral = max(
            abs(self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
            for xy in self._unique_xy_points(points))
        if best_lateral >= min_lateral:
            return True
        self.get_logger().info(
            f'Blue candidate {door.door_id} rejected near exit: openable '
            f'wall lateral {best_lateral:.2f}m < {min_lateral:.2f}m.',
            throttle_duration_sec=3.0)
        return False

    def _is_blue_door_at_valid_wall_position(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        identity_xy = self._door_identity_xy(door)
        points = [identity_xy]
        handle_xy = None
        if door.handle_position.header.frame_id == 'map':
            handle_xy = (
                float(door.handle_position.point.x),
                float(door.handle_position.point.y),
            )
            points.append(handle_xy)
        raw_xy = self._raw_blue_handle_xy_for_evidence(door)
        if raw_xy is not None:
            points.append(raw_xy)
        wall_position_ok = any(
            self._is_blue_xy_at_valid_wall_position(xy)
            for xy in self._unique_xy_points(points))
        if (
                wall_position_ok
                and self._is_blue_xy_in_pre_exit_zone(identity_xy)
                and not self._blue_door_has_openable_wall_lateral(door)):
            return False
        return (
            wall_position_ok
            and self._is_blue_handle_in_wall_band(door)
            and self._is_direct_blue_pose_lateral_consistent(door)
        )

    def _unique_xy_points(
            self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        unique: list[tuple[float, float]] = []
        for xy in points:
            if not all(math.isfinite(v) for v in xy):
                continue
            if all(math.hypot(xy[0] - old[0], xy[1] - old[1]) > 0.05
                   for old in unique):
                unique.append(xy)
        return unique

    def _semantic_non_blue_conflict_blocks_blue_target(
            self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        strong_blue, blue_confidence, blue_count = (
            self._blue_target_observation_strength(door))
        if self._semantic_conflict_yields_to_live_side_wall_blue_target(
                door, strong_blue, blue_confidence, blue_count):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if (
                cluster is not None
                and self._observed_blue_cluster_has_recheckable_wall_memory(cluster)):
            self.get_logger().info(
                f'Semantic non-blue conflict ignored for {door.door_id}: '
                'repeated wall-side blue map memory will be rechecked at the '
                'door before opening.',
                throttle_duration_sec=3.0)
            return False
        for xy in self._unique_xy_points(self._blue_evidence_points(door)):
            if self._semantic_non_blue_conflict_blocks_blue_xy(
                    xy, strong_blue, blue_confidence, blue_count):
                return True
        return False

    def _pre_exit_live_side_wall_blue_overrides_memory_conflict(
            self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if exit_xy is None:
            return False
        if not self._blue_xy_reliable_side_wall_candidate_near_exit(xy, exit_xy):
            return False

        strong_blue, blue_confidence, blue_count = (
            self._blue_target_observation_strength(door))
        cluster = self._find_observed_blue_cluster(
            xy,
            merge_dist=self._observed_candidate_merge_dist())
        cluster_count = 0
        cluster_confidence = 0.0
        if cluster is not None:
            if cluster.get('abandoned', 0.0) >= 1.0:
                return False
            cluster_count = int(cluster.get('count', 0.0))
            cluster_confidence = float(cluster.get('confidence', 0.0))

        evidence_count = max(blue_count, cluster_count)
        evidence_confidence = max(
            blue_confidence,
            cluster_confidence,
            abs(float(door.confidence)))
        min_count = max(2, self._observed_blue_min_observations)
        min_confidence = max(
            0.34,
            max(0.0, self._pre_exit_blue_min_confidence),
            max(0.0, self._door_open_fresh_blue_min_confidence))
        if (
                not strong_blue
                and evidence_count < min_count
                and evidence_confidence < min_confidence):
            return False
        if not self._door_target_has_fresh_blue_evidence(door):
            self.get_logger().info(
                f'Pre-exit side-wall blue target {door.door_id} cannot '
                'override non-blue memory: no current live blue evidence at '
                'the same mapped station.',
                throttle_duration_sec=3.0)
            return False

        self.get_logger().info(
            f'Pre-exit live side-wall blue target {door.door_id} overrides '
            f'non-blue memory conflict: count={evidence_count}, '
            f'conf={evidence_confidence:.2f}.',
            throttle_duration_sec=3.0)
        return True

    def _semantic_conflict_yields_to_live_side_wall_blue_target(
            self,
            door: DoorInfo,
            strong_blue: bool,
            blue_confidence: float,
            blue_count: int) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        if self._pre_exit_live_side_wall_blue_overrides_memory_conflict(door):
            return True
        xy = self._door_identity_xy(door)
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if exit_xy is None:
            return False
        if not self._blue_xy_reliable_side_wall_candidate_near_exit(xy, exit_xy):
            return False

        cluster = self._find_observed_blue_cluster(
            xy,
            merge_dist=self._observed_candidate_merge_dist())
        cluster_count = 0
        cluster_confidence = 0.0
        if cluster is not None:
            if cluster.get('abandoned', 0.0) >= 1.0:
                return False
            cluster_count = int(cluster.get('count', 0.0))
            cluster_confidence = float(cluster.get('confidence', 0.0))

        evidence_count = max(blue_count, cluster_count)
        evidence_confidence = max(
            blue_confidence,
            cluster_confidence,
            abs(float(door.confidence)))
        min_confidence = max(
            0.22,
            min(
                max(0.0, self._pre_exit_blue_min_confidence),
                max(0.0, self._door_open_fresh_blue_min_confidence)))
        if (
                not strong_blue
                and evidence_count < self._observed_blue_min_observations
                and evidence_confidence < min_confidence):
            return False

        self.get_logger().info(
            f'Semantic non-blue conflict ignored for {door.door_id}: '
            f'live side-wall blue evidence near the observed exit is reliable '
            f'(count={evidence_count}, conf={evidence_confidence:.2f}).',
            throttle_duration_sec=3.0)
        return True

    def _semantic_green_conflict_yields_to_side_wall_blue(
            self,
            xy: tuple[float, float],
            green_count: int,
            red_count: int,
            blue_count: int,
            blue_confidence: float,
            strong_blue: bool = False) -> bool:
        if green_count <= 0:
            return False
        if red_count >= self._red_observation_min_count_for_blue_suppression:
            return False
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if exit_xy is None:
            return False
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        tail_gap = exit_progress - candidate_progress
        strict_tail_gap = max(
            1.15,
            self._blue_exit_suppression_margin_m + 0.35)
        if not self._no_blue_exit_mode() and self._opened_blue_door_count() > 0:
            strict_tail_gap = max(
                strict_tail_gap,
                self._pre_exit_blue_suppression_margin_m)
        if 0.0 <= tail_gap < strict_tail_gap:
            cluster = self._find_observed_blue_cluster(
                xy,
                merge_dist=self._observed_candidate_merge_dist())
            if (
                    cluster is None
                    or not self._observed_blue_cluster_has_persistent_exit_memory(
                        cluster)):
                return False
        if not self._blue_xy_reliable_side_wall_candidate_near_exit(xy, exit_xy):
            return False
        min_blue_count = max(
            self._semantic_blue_min_observations,
            self._observed_blue_min_observations)
        min_confidence = max(
            0.34,
            max(0.0, self._pre_exit_blue_min_confidence),
            max(0.0, self._door_open_fresh_blue_min_confidence))
        return (
            strong_blue
            or blue_count >= min_blue_count
            or blue_confidence >= min_confidence)

    def _semantic_non_blue_conflict_blocks_blue_xy(
            self,
            xy: tuple[float, float],
            strong_blue: bool = False,
            blue_confidence: float = 0.0,
            blue_count: int = 0) -> bool:
        landmark = self._find_semantic_door_landmark(xy)
        if landmark is None:
            return False
        landmark_blue, red_count, green_count = (
            self._semantic_landmark_color_votes(landmark))
        blue_count = max(blue_count, landmark_blue)
        non_blue_count = red_count + green_count
        if non_blue_count <= 0:
            return False

        total_votes = max(1, blue_count + non_blue_count)
        blue_ratio = blue_count / float(total_votes)
        min_blue_count = max(
            self._semantic_blue_min_observations,
            self._observed_blue_min_observations)
        if strong_blue and blue_count >= max(min_blue_count + 2, non_blue_count + 2):
            if blue_ratio >= self._semantic_blue_min_ratio:
                return False
        if self._semantic_green_conflict_yields_to_side_wall_blue(
                xy, green_count, red_count, blue_count,
                blue_confidence, strong_blue):
            self.get_logger().info(
                f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 유지: '
                f'출구 근처 초록 관측과 충돌했지만 옆벽 파란문 관측이 '
                f'충분합니다. blue={blue_count}, green={green_count}, '
                f'conf={blue_confidence:.2f}',
                throttle_duration_sec=3.0)
            return False
        if red_count >= self._red_observation_min_count_for_blue_suppression:
            return True
        if green_count >= self._semantic_exit_min_green_observations:
            return True
        if non_blue_count >= max(2, blue_count):
            return True
        if non_blue_count > 0 and blue_confidence < self._red_observation_min_confidence:
            return True
        return False

    def _is_blue_xy_at_valid_wall_position(self, xy: tuple[float, float]) -> bool:
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False
        if self._is_blue_xy_at_or_after_observed_exit(xy):
            return False
        if self._is_blue_xy_near_detected_exit(xy):
            return False
        if self._is_blue_xy_opposite_opened_station(xy):
            return False
        if self._is_blue_xy_near_abandoned_station(xy):
            return False
        return True

    def _is_blue_handle_in_wall_band(self, door: DoorInfo) -> bool:
        if door.handle_position.header.frame_id != 'map':
            return True
        max_abs_y = max(0.0, self._observed_blue_max_abs_wall_y_m)
        if max_abs_y <= 0.0:
            return True
        overshoot = max(
            0.0,
            self._blue_handle_max_wall_overshoot_m,
            self._blue_handle_validation_overshoot_m)
        handle_xy = (
            float(door.handle_position.point.x),
            float(door.handle_position.point.y),
        )
        if not self._door_handle_consistent_with_pose(door, handle_xy):
            return False
        handle_abs_y = abs(
            self._axis_lateral_xy(handle_xy[0], handle_xy[1])
            - self._explore_center_y)
        if handle_abs_y <= max_abs_y + overshoot:
            return True
        self.get_logger().info(
            f'Blue candidate {door.door_id} rejected: handle lateral '
            f'{handle_abs_y:.2f}m is outside wall band '
            f'{max_abs_y:.2f}+{overshoot:.2f}m.',
            throttle_duration_sec=3.0)
        return False

    def _is_xy_at_configured_wall_lateral(self, xy: tuple[float, float]) -> bool:
        abs_y = abs(self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        if (self._observed_blue_min_abs_wall_y_m > 0.0
                and abs_y < self._observed_blue_min_abs_wall_y_m):
            return False
        if (self._observed_blue_max_abs_wall_y_m > 0.0
                and abs_y > self._observed_blue_max_abs_wall_y_m):
            return False
        return True

    def _exit_xy_for_blue_suppression(self) -> tuple[float, float] | None:
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return None
        exit_xy = self._exit_identity_xy()
        if exit_xy is not None:
            door_id = self._exit_door.door_id if self._exit_door is not None else ''
            if self._exit_candidate_axis_valid(
                    exit_xy, door_id, clear_existing=False):
                self._last_valid_exit_xy = exit_xy
                self._last_valid_exit_from_front_wall = (
                    door_id == 'front_wall_exit_observed')
                return exit_xy
        provisional_xy = self._provisional_exit_xy_for_blue_suppression()
        if provisional_xy is not None:
            return provisional_xy
        front_wall_xy = self._front_wall_exit_xy_for_blue_suppression()
        if front_wall_xy is not None:
            return front_wall_xy
        return self._last_valid_exit_xy

    def _exit_xy_for_blue_candidate_filter(self) -> tuple[float, float] | None:
        """Best observed green-exit estimate used only to reject blue false positives.

        Exit navigation still waits for the normal opened-door policy. This
        helper lets an already observed green exit suppress blue candidates in
        its tail area before the robot has opened every blue door.
        """
        xy = self._exit_xy_for_blue_suppression()
        if xy is not None:
            return xy

        candidates: list[tuple[float, float]] = []
        exit_xy = self._exit_identity_xy()
        if exit_xy is not None:
            candidates.append(exit_xy)
        if self._last_valid_exit_xy is not None:
            candidates.append(self._last_valid_exit_xy)
        if self._last_provisional_exit_xy is not None:
            candidates.append(self._last_provisional_exit_xy)

        for candidate in candidates:
            if self._exit_candidate_axis_valid(
                    candidate,
                    'blue_candidate_filter_exit',
                    clear_existing=False,
                    require_front_wall_confirmation=False):
                return candidate
        return None

    def _is_blue_door_before_current_exit(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        if self._is_blue_xy_at_or_after_observed_exit(self._door_identity_xy(door)):
            return False
        exit_xy = self._exit_xy_for_blue_suppression()
        if exit_xy is None:
            return True
        door_xy = self._door_identity_xy(door)
        door_progress = self._axis_progress_xy(door_xy[0], door_xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        margin = max(0.0, self._exit_interrupt_blue_before_exit_margin_m)
        if door_progress < exit_progress - margin:
            return True
        self.get_logger().info(
            f'Blue candidate {door.door_id} ignored during exit: '
            f'candidate_progress={door_progress:.1f}m is at/after '
            f'exit_progress={exit_progress:.1f}m (margin={margin:.1f}m).',
            throttle_duration_sec=3.0)
        return False

    def _blue_exit_candidate_suppression_active(self) -> bool:
        # Once the exit is observed and the mission has enough opened blue
        # doors to be exit-ready, reject blue candidates at/after that observed
        # exit even during normal exploration. This keeps real side doors
        # before the exit, while preventing exit-color tail noise from becoming
        # another blue-door target.
        if self._final_scan_start_time is not None or self.state == State.EXITING:
            return True
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False
        return self._exit_xy_for_blue_candidate_filter() is not None

    def _is_blue_xy_at_or_after_observed_exit(
            self, xy: tuple[float, float]) -> bool:
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if exit_xy is None:
            return False
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        margin = max(0.0, self._observed_exit_blue_candidate_margin_m)
        if candidate_progress < exit_progress - margin:
            return False

        candidate_lateral = self._axis_lateral_xy(xy[0], xy[1])
        exit_lateral = self._axis_lateral_xy(exit_xy[0], exit_xy[1])
        wall_lateral = abs(candidate_lateral - self._explore_center_y)
        lateral_gap = abs(candidate_lateral - exit_lateral)
        side_wall_threshold = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m)
        side_wall_projection_grace = max(
            0.75,
            self._observed_exit_blue_candidate_margin_m + 0.75)
        if (
                self._blue_xy_reliable_side_wall_candidate_near_exit(xy, exit_xy)
                and candidate_progress <= exit_progress + side_wall_projection_grace):
            return False

        self.get_logger().info(
            f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) rejected: '
            f'candidate_progress={candidate_progress:.1f}m is at/after '
            f'observed green exit progress={exit_progress:.1f}m '
            f'(margin={margin:.1f}m).',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_xy_near_detected_exit(self, xy: tuple[float, float]) -> bool:
        margin = max(0.0, self._blue_exit_suppression_margin_m)
        if margin <= 0.0:
            return False
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if exit_xy is None:
            return False
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        candidate_lateral = self._axis_lateral_xy(xy[0], xy[1])
        exit_lateral = self._axis_lateral_xy(exit_xy[0], exit_xy[1])
        wall_lateral = abs(candidate_lateral - self._explore_center_y)
        lateral_gap = abs(candidate_lateral - exit_lateral)
        side_wall_threshold = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m)
        if candidate_progress < exit_progress - margin:
            return False
        side_wall_projection_grace = max(
            0.75,
            self._observed_exit_blue_candidate_margin_m + 0.75)
        if (
                self._blue_xy_reliable_side_wall_candidate_near_exit(xy, exit_xy)
                and candidate_progress <= exit_progress + side_wall_projection_grace):
            return False
        if candidate_progress >= exit_progress + 0.20:
            self.get_logger().info(
                f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
                f'관측된 비상구 진행축 이후의 재관측 후보입니다.',
                throttle_duration_sec=3.0)
            return True
        self.get_logger().info(
            f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
            f'관측된 비상구 진행축 {margin:.1f}m 이내/이후 영역입니다.',
            throttle_duration_sec=3.0)
        return True

    def _observed_blue_cluster_has_persistent_exit_memory(
            self, cluster: dict[str, float]) -> bool:
        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        persistent_count = max(
            self._observed_blue_min_observations + 6,
            self._confirmed_blue_bypass_red_conflict_count + 4)
        persistent_conf = max(
            0.50,
            max(0.0, self._pre_exit_blue_min_confidence),
            max(0.0, self._observed_blue_stable_open_min_confidence - 0.02))
        return count >= persistent_count and confidence >= persistent_conf

    def _observed_blue_cluster_has_pre_exit_target_memory(
            self, cluster: dict[str, float]) -> bool:
        if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
            return False
        xy = self._cluster_xy(cluster)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        if self._semantic_blue_xy_suppressed(xy):
            return False
        if self._is_blue_xy_near_observed_red(xy):
            return False
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False

        wall_lateral = abs(
            self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        min_lateral = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        max_lateral = max(0.0, self._observed_blue_max_abs_wall_y_m)
        if wall_lateral < min_lateral:
            return False
        if (
                max_lateral > 0.0
                and wall_lateral > max_lateral + self._blue_handle_validation_overshoot_m):
            return False

        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        min_count = max(
            self._observed_blue_min_observations,
            self._observed_blue_low_conf_target_min_observations)
        min_conf = max(0.0, self._observed_blue_target_min_confidence)
        return count >= min_count and confidence >= min_conf

    def _blue_xy_has_unopened_openable_side_wall_memory(
            self, xy: tuple[float, float]) -> bool:
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        if self._is_blue_xy_near_abandoned_station(xy):
            return False
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False

        wall_lateral = abs(self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
        min_lateral = max(
            0.0,
            self._door_open_axis_min_side_lateral_m,
            self._observed_blue_min_abs_wall_y_m)
        max_lateral = max(0.0, self._observed_blue_max_abs_wall_y_m)
        if min_lateral > 0.0 and wall_lateral < min_lateral:
            return False
        if max_lateral > 0.0 and wall_lateral > max_lateral:
            return False

        cluster = self._find_observed_blue_cluster(
            xy,
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        if bool(cluster.get('opened', False)) or bool(cluster.get('abandoned', False)):
            return False

        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        min_count = max(1, self._observed_blue_min_observations)
        min_conf = max(0.0, self._observed_blue_target_min_confidence)
        return count >= min_count or confidence >= min_conf

    def _blue_xy_reliable_side_wall_candidate_near_exit(
            self,
            xy: tuple[float, float],
            exit_xy: tuple[float, float]) -> bool:
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        tail_gap = exit_progress - candidate_progress
        hard_tail_gap = max(
            1.15,
            self._blue_exit_suppression_margin_m + 0.35)
        in_hard_exit_tail = 0.0 <= tail_gap < hard_tail_gap
        cluster = self._find_observed_blue_cluster(
            xy,
            merge_dist=self._observed_candidate_merge_dist())
        if in_hard_exit_tail:
            if (
                    cluster is not None
                    and self._observed_blue_cluster_has_persistent_exit_memory(cluster)):
                return True
            if (
                    cluster is not None
                    and self._observed_blue_cluster_has_pre_exit_target_memory(cluster)):
                self.get_logger().info(
                    f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) kept in '
                    'strict exit tail: repeated unopened side-wall blue memory '
                    'must be opened or explicitly failed before exiting.',
                    throttle_duration_sec=3.0)
                return True
            return False

        candidate_lateral = self._axis_lateral_xy(xy[0], xy[1])
        exit_lateral = self._axis_lateral_xy(exit_xy[0], exit_xy[1])
        wall_lateral = abs(candidate_lateral - self._explore_center_y)
        lateral_gap = abs(candidate_lateral - exit_lateral)
        side_wall_threshold = max(
            0.35,
            self._observed_blue_min_abs_wall_y_m)
        if wall_lateral < side_wall_threshold or lateral_gap <= 0.55:
            return False
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False
        red_conflict_dist = max(
            self._blue_red_conflict_dist_m,
            self._blue_red_conflict_after_exit_dist_m)
        if self._blue_xy_conflicts_recent_red_memory(xy, red_conflict_dist):
            self.get_logger().info(
                f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) suppressed near '
                'observed exit: side-wall blue memory overlaps recent red memory.',
                throttle_duration_sec=3.0)
            return False
        if self._blue_xy_has_unopened_openable_side_wall_memory(xy):
            self.get_logger().info(
                f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) kept near observed exit: '
                'unopened side-wall blue memory is stronger than the early exit estimate.',
                throttle_duration_sec=3.0)
            return True

        if (
                cluster is not None
                and self._observed_blue_cluster_has_persistent_exit_memory(cluster)):
            return True

        min_conf = max(
            0.16,
            min(
                max(0.0, self._pre_exit_blue_min_confidence),
                max(0.0, self._door_open_fresh_blue_min_confidence)))
        max_age = max(0.0, self._pre_exit_blue_fresh_max_age_sec)
        if (
                cluster is not None
                and self._observed_blue_cluster_has_fresh_live_detection(
                    cluster, max_age, min_conf)):
            return True

        merge_dist = max(
            0.45,
            self._observed_candidate_merge_dist(),
            self._door_open_fresh_blue_max_dist_m)
        now = self.get_clock().now()
        for door in self.detected_doors:
            if door.door_color != 'blue':
                continue
            if door.door_id.startswith('observed_blue_'):
                continue
            if not self._door_has_map_identity(door):
                continue
            if float(door.confidence) < min_conf:
                continue
            if (not self._is_direct_blue_pose_lateral_consistent(door)
                    and not self._direct_blue_candidate_matches_observed_wall_memory(
                        door, xy)):
                continue
            if max_age > 0.0:
                try:
                    stamp = Time.from_msg(door.header.stamp)
                except Exception:
                    stamp = None
                if stamp is not None and stamp.nanoseconds > 0:
                    age = (now - stamp).nanoseconds / 1e9
                    if age > max_age:
                        continue
            for point in self._blue_evidence_points(door):
                if not self._is_xy_at_configured_wall_lateral(point):
                    continue
                if math.hypot(xy[0] - point[0], xy[1] - point[1]) <= merge_dist:
                    return True
        return False

    def _blue_xy_in_strict_exit_tail_without_persistent_memory(
            self,
            xy: tuple[float, float],
            door: DoorInfo | None = None,
            cluster: dict[str, float] | None = None) -> bool:
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if exit_xy is None:
            return False
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        tail_gap = exit_progress - candidate_progress
        strict_tail_gap = max(
            1.15,
            self._blue_exit_suppression_margin_m + 0.35)
        if not self._no_blue_exit_mode() and self._opened_blue_door_count() > 0:
            strict_tail_gap = max(
                strict_tail_gap,
                self._pre_exit_blue_suppression_margin_m)
        if tail_gap < 0.0 or tail_gap >= strict_tail_gap:
            return False
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False
        if self._blue_xy_reliable_side_wall_candidate_near_exit(xy, exit_xy):
            self.get_logger().info(
                f'Pre-exit blue candidate '
                f'{door.door_id if door is not None else f"({xy[0]:.1f},{xy[1]:.1f})"} '
                'kept: live side-wall blue evidence separates it from exit tail.',
                throttle_duration_sec=3.0)
            return False
        if cluster is None:
            cluster = self._find_observed_blue_cluster(
                xy,
                merge_dist=self._observed_candidate_merge_dist())
        if (
                cluster is not None
                and self._observed_blue_cluster_has_persistent_exit_memory(cluster)):
            return False
        door_id = (
            door.door_id if door is not None
            else f'({xy[0]:.1f},{xy[1]:.1f})')
        confidence = abs(float(door.confidence)) if door is not None else 0.0
        self.get_logger().info(
            f'Pre-exit blue candidate {door_id} rejected: '
            f'only {tail_gap:.2f}m before observed green exit and no '
            f'persistent blue memory (conf={confidence:.2f}).',
            throttle_duration_sec=3.0)
        return True

    def _blue_target_in_strict_exit_tail_without_persistent_memory(
            self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        return self._blue_xy_in_strict_exit_tail_without_persistent_memory(
            self._door_identity_xy(door), door)

    def _observed_blue_retry_pending(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        xy = self._cluster_xy(cluster)
        if (
                self._is_blue_xy_at_or_after_observed_exit(xy)
                or self._is_blue_xy_near_detected_exit(xy)):
            return False
        return self._observed_blue_cluster_retry_pending(cluster)

    def _door_has_persistent_observed_blue_exit_memory(
            self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not door.door_id.startswith('observed_blue_'):
            return False
        if not self._door_has_map_identity(door):
            return False
        cluster = self._find_observed_blue_cluster(
            self._door_identity_xy(door),
            merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False
        return self._observed_blue_cluster_has_persistent_exit_memory(cluster)

    def _is_pre_exit_blue_recheck_candidate(self, door: DoorInfo) -> bool:
        if not self._pre_exit_blue_requires_fresh_detection:
            return False
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        if not self._is_blue_xy_in_pre_exit_zone(xy):
            return False
        confidence = float(door.confidence)
        min_conf = max(
            0.40,
            max(0.0, self._pre_exit_blue_min_confidence),
            max(0.0, self._door_open_fresh_blue_min_confidence),
            max(0.0, self._exit_interrupt_blue_min_confidence))
        if confidence < min_conf:
            self.get_logger().info(
                f'Pre-exit blue candidate {door.door_id} rejected: direct '
                f'confidence {confidence:.2f} < {min_conf:.2f}.',
                throttle_duration_sec=3.0)
            return False
        if not self._is_xy_at_configured_wall_lateral(xy):
            return False
        if not self._blue_door_has_openable_wall_lateral(door):
            return False
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        if self._is_blue_xy_near_detected_exit(xy):
            return False
        if self._is_blue_xy_at_or_after_observed_exit(xy):
            return False
        exit_xy = self._exit_xy_for_blue_suppression()
        if exit_xy is not None:
            cluster = self._find_observed_blue_cluster(
                xy,
                merge_dist=self._observed_candidate_merge_dist())
            if self._blue_xy_in_strict_exit_tail_without_persistent_memory(
                    xy, door, cluster):
                return False
        if self._is_blue_xy_opposite_opened_station(xy):
            return False
        if self._is_blue_xy_near_abandoned_station(xy):
            return False
        if self._is_blue_xy_near_observed_red(xy, candidate_id=door.door_id):
            return False
        landmark = self._find_semantic_door_landmark(xy)
        if landmark is not None:
            blue_count, red_count, green_count = self._semantic_landmark_color_votes(landmark)
            if red_count + green_count > blue_count:
                if not self._semantic_green_conflict_yields_to_side_wall_blue(
                        xy, green_count, red_count, blue_count,
                        confidence, False):
                    return False
        return True

    def _is_observed_blue_cluster_weak_in_pre_exit_zone(
            self, cluster: dict[str, float]) -> bool:
        if not self._pre_exit_blue_requires_fresh_detection:
            return False
        xy = self._cluster_xy(cluster)
        if not self._is_blue_xy_in_pre_exit_zone(xy):
            return False
        # Near an already observed exit, stable observed_blue clusters are prone
        # to being formed by color bleed or partial wall/exit detections. Keep
        # them only when the current live detector still sees a strong blue door,
        # or when repeated blue votes are already strong enough to treat this as
        # a mapped side-wall door that must be retried before exiting.
        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        strong_count = max(
            self._observed_blue_min_observations + 3,
            self._confirmed_blue_bypass_red_conflict_count)
        strong_conf = max(
            0.45,
            self._confirmed_blue_bypass_red_conflict_confidence - 0.05)
        if count >= strong_count and confidence >= strong_conf:
            return False
        if self._observed_blue_cluster_has_pre_exit_target_memory(cluster):
            self.get_logger().info(
                f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) kept near '
                'observed exit: repeated unopened side-wall blue memory must '
                'be rechecked before exiting.',
                throttle_duration_sec=3.0)
            return False
        if self._observed_blue_cluster_has_persistent_exit_memory(cluster):
            return False
        min_conf = max(0.0, self._pre_exit_blue_min_confidence)
        max_age = max(0.0, self._pre_exit_blue_fresh_max_age_sec)
        if self._observed_blue_cluster_has_fresh_live_detection(
                cluster, max_age, min_conf):
            return False
        self.get_logger().info(
            f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) suppressed near '
            f'observed exit: no fresh live blue detection '
            f'(count={int(cluster.get("count", 0.0))}, '
            f'conf={float(cluster.get("confidence", 0.0)):.2f}).',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_door_weak_in_pre_exit_zone(self, door: DoorInfo) -> bool:
        if not self._pre_exit_blue_requires_fresh_detection:
            return False
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        if not self._is_blue_xy_in_pre_exit_zone(xy):
            return False
        min_conf = max(0.0, self._pre_exit_blue_min_confidence)
        confidence = float(door.confidence)
        if confidence >= min_conf:
            return False
        progress = self._axis_progress_xy(xy[0], xy[1])
        lateral = self._axis_lateral_xy(xy[0], xy[1])
        self.get_logger().info(
            f'Blue candidate {door.door_id} suppressed near observed exit: '
            f'conf={confidence:.2f} < {min_conf:.2f}, '
            f'p={progress:.2f}, lat={lateral:.2f}.',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_xy_in_pre_exit_zone(self, xy: tuple[float, float]) -> bool:
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False
        window = max(0.0, self._pre_exit_blue_suppression_margin_m)
        if window <= 0.0:
            return False
        exit_xy = self._exit_xy_for_blue_suppression()
        if exit_xy is None:
            return False
        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
        return exit_progress - window <= candidate_progress <= exit_progress + 0.35

    def _observed_blue_cluster_has_fresh_live_detection(
            self,
            cluster: dict[str, float],
            max_age: float,
            min_confidence: float) -> bool:
        xy = self._cluster_xy(cluster)
        merge_dist = max(0.45, self._observed_candidate_merge_dist())
        now = self.get_clock().now()
        for door in self.detected_doors:
            if door.door_color != 'blue':
                continue
            if door.door_id.startswith('observed_blue_'):
                continue
            if not self._door_has_map_identity(door):
                continue
            if float(door.confidence) < min_confidence:
                continue
            if max_age > 0.0:
                try:
                    stamp = Time.from_msg(door.header.stamp)
                except Exception:
                    stamp = None
                if stamp is not None and stamp.nanoseconds > 0:
                    age = (now - stamp).nanoseconds / 1e9
                    if age > max_age:
                        continue
            door_xy = self._door_identity_xy(door)
            if math.hypot(xy[0] - door_xy[0], xy[1] - door_xy[1]) <= merge_dist:
                return True
        return False

    def _blue_target_has_safe_open_memory(self, door: DoorInfo) -> bool:
        if self._door_open_requires_fresh_blue:
            # A stored direct detection is useful for navigation, but opening
            # the door must be based on the current live camera observation at
            # the aligned door station. This prevents distant side-camera
            # projections from being opened before the robot reaches the real
            # door face.
            return False
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        if door.door_id.startswith('observed_blue_'):
            return False
        xy = self._door_identity_xy(door)
        if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
            return False
        if self._semantic_non_blue_conflict_blocks_blue_target(door):
            return False

        in_pre_exit_zone = self._is_blue_xy_in_pre_exit_zone(xy)
        near_detected_exit = self._is_blue_xy_near_detected_exit(xy)
        confidence = abs(float(door.confidence))
        min_conf = max(
            0.20,
            min(
                self._blue_target_immediate_min_confidence,
                max(
                    self._explore_interrupt_min_confidence,
                    self._door_open_fresh_blue_min_confidence)))
        if confidence < min_conf:
            return False

        if in_pre_exit_zone or near_detected_exit:
            if not self._blue_target_open_pose_aligned_for_safe_memory(door):
                return False
            self.get_logger().info(
                f'Blue target {door.door_id} uses aligned direct blue memory '
                f'near tentative exit for opening: conf={confidence:.2f}.',
                throttle_duration_sec=3.0)
            return True

        if confidence >= min_conf:
            self.get_logger().info(
                f'Blue target {door.door_id} uses direct blue detection memory '
                f'for opening: conf={confidence:.2f}.',
                throttle_duration_sec=3.0)
            return True
        return False

    def _blue_target_needs_fresh_open_confirmation(self, door: DoorInfo) -> bool:
        if not self._door_open_requires_fresh_blue:
            return False
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        xy = self._door_identity_xy(door)
        in_pre_exit_zone = self._is_blue_xy_in_pre_exit_zone(xy)
        if door.door_id.startswith('observed_blue_'):
            # Observed targets come from fused map memory. They are useful for
            # selecting where to drive, but opening must be based on a live
            # blue frame at that same mapped station.
            return True

        # Direct camera detections still need a live final blue confirmation.
        # A stale direct detection can be projected to the wall band but several
        # meters ahead of the physical door while the robot is turning.
        return True

    def _observed_blue_target_has_stable_open_evidence(self, door: DoorInfo) -> bool:
        # Map memory is allowed to pick a target, but it must not by itself
        # count as door-open evidence. The opening transition requires a live
        # blue observation at the mapped station plus the front-alignment check.
        return False

    def _door_target_has_fresh_blue_evidence(self, door: DoorInfo) -> bool:
        if not self._door_open_requires_fresh_blue:
            return True
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return True
        target_xy_for_checks = self._door_identity_xy(door)
        if (
                not door.door_id.startswith('observed_blue_')
                and not self._is_direct_blue_pose_lateral_consistent(door)
                and not self._direct_blue_candidate_matches_observed_wall_memory(
                    door, target_xy_for_checks)):
            return False
        if self._blue_target_in_strict_exit_tail_without_persistent_memory(door):
            return False
        target_points = self._blue_evidence_points(door)
        target_in_pre_exit = self._is_blue_xy_in_pre_exit_zone(target_xy_for_checks)
        if target_in_pre_exit:
            exit_xy = self._exit_xy_for_blue_candidate_filter()
            if exit_xy is not None:
                target_progress = self._axis_progress_xy(
                    target_xy_for_checks[0], target_xy_for_checks[1])
                exit_progress = self._axis_progress_xy(exit_xy[0], exit_xy[1])
                min_gap = max(1.15, self._blue_exit_suppression_margin_m + 0.35)
                exit_tail_gap = exit_progress - target_progress
                if exit_tail_gap < min_gap:
                    # Being near a tentative green exit is not enough to discard a
                    # locked blue target. First require live blue evidence at the
                    # same mapped station; if that evidence is missing, the normal
                    # final confirmation below will reject it. This prevents a
                    # true last blue door from being skipped when the exit is also
                    # visible in the same scene.
                    self.get_logger().info(
                        f'Fresh blue evidence near tentative exit tail for '
                        f'{door.door_id}: checking live blue before rejection '
                        f'(gap={exit_tail_gap:.2f}m, need {min_gap:.2f}m).',
                        throttle_duration_sec=3.0)
        max_age = max(0.0, self._door_open_fresh_blue_max_age_sec)
        max_dist = max(0.20, self._door_open_fresh_blue_max_dist_m)
        if door.door_id.startswith('observed_blue_'):
            # observed_blue_* targets come from a fused map cluster. As the
            # robot gets closer, the live camera projection may shift along the
            # same wall station before the cluster ID is merged. Keep the
            # final "is this still blue?" check within the physical door merge
            # radius instead of failing a real door because of that projection
            # drift.
            max_dist = max(
                max_dist,
                self._observed_physical_door_merge_dist_m + 0.65)
        min_conf = max(0.0, self._door_open_fresh_blue_min_confidence)
        if target_in_pre_exit:
            if door.door_id.startswith('observed_blue_'):
                min_conf = max(min_conf, 0.34)
            else:
                min_conf = max(min_conf, 0.42)
        elif not door.door_id.startswith('observed_blue_'):
            min_conf = max(
                min_conf,
                self._blue_target_immediate_min_confidence)
        now = self.get_clock().now()
        for candidate in self.detected_doors:
            if candidate.door_color != 'blue':
                continue
            if candidate.door_id.startswith('observed_blue_'):
                continue
            if not self._door_has_map_identity(candidate):
                continue
            if (not self._is_direct_blue_pose_lateral_consistent(candidate)
                    and not self._direct_blue_candidate_matches_observed_wall_memory(
                        candidate, target_xy_for_checks)):
                continue
            if float(candidate.confidence) < min_conf:
                continue
            if max_age > 0.0:
                try:
                    stamp = Time.from_msg(candidate.header.stamp)
                except Exception:
                    stamp = None
                if stamp is not None and stamp.nanoseconds > 0:
                    age = (now - stamp).nanoseconds / 1e9
                    if age > max_age:
                        continue
            candidate_points = self._blue_evidence_points(candidate)
            if self._target_subfsm.live_blue_evidence_matches_target(
                    door, candidate, target_points, candidate_points, max_dist):
                return True
        target_xy = target_points[0]
        tx_progress = self._axis_progress_xy(target_xy[0], target_xy[1])
        tx_lateral = self._axis_lateral_xy(target_xy[0], target_xy[1])
        self.get_logger().warn(
            f'Fresh blue evidence missing near target {door.door_id}: '
            f'p={tx_progress:.2f}, lat={tx_lateral:.2f}, '
            f'radius={max_dist:.2f}m, max_age={max_age:.1f}s, '
            f'min_conf={min_conf:.2f}.')
        return False


    def _door_target_has_recent_non_blue_conflict(self, door: DoorInfo) -> bool:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False
        target_points = [self._door_handle_xy(door), self._door_identity_xy(door)]
        max_age = max(0.0, self._door_open_fresh_blue_max_age_sec)
        max_dist = max(0.45, self._blue_red_conflict_strict_dist_m + 0.20)
        now = self.get_clock().now()
        strong_blue, blue_confidence, blue_count = (
            self._blue_target_observation_strength(door))
        fresh_blue_confirmed = self._door_target_has_fresh_blue_evidence(door)
        target_xy = self._door_identity_xy(door)
        exit_xy = self._exit_xy_for_blue_candidate_filter()
        if (
                fresh_blue_confirmed
                and self._blue_target_open_pose_aligned_for_safe_memory(door)):
            self.get_logger().info(
                f'Non-blue conflict ignored for {door.door_id}: '
                'door is already aligned at a live blue opening pose.',
                throttle_duration_sec=3.0)
            return False
        if (
                fresh_blue_confirmed
                and exit_xy is not None
                and self._blue_xy_reliable_side_wall_candidate_near_exit(
                    target_xy, exit_xy)):
            self.get_logger().info(
                f'Non-blue conflict ignored for {door.door_id}: live side-wall '
                'blue evidence is reliable near the observed exit.',
                throttle_duration_sec=3.0)
            return False
        if self._semantic_non_blue_conflict_blocks_blue_target(door):
            return True
        fresh_blue_yields_conflict = (
            fresh_blue_confirmed
            and blue_confidence >= max(
                0.40,
                self._door_open_fresh_blue_min_confidence))
        if fresh_blue_yields_conflict:
            blue_count = max(blue_count, self._observed_blue_min_observations)

        for candidate in self.detected_doors:
            if candidate.door_color not in ('red', 'green'):
                continue
            if not self._door_has_map_identity(candidate):
                continue
            min_conf = self._red_observation_min_confidence
            if candidate.door_color == 'green':
                min_conf = max(min_conf, 0.35)
            if float(candidate.confidence) < min_conf:
                continue
            if max_age > 0.0:
                try:
                    stamp = Time.from_msg(candidate.header.stamp)
                except Exception:
                    stamp = None
                if stamp is not None and stamp.nanoseconds > 0:
                    age = (now - stamp).nanoseconds / 1e9
                    if age > max_age:
                        continue
            candidate_points = [
                self._door_handle_xy(candidate),
                self._door_identity_xy(candidate),
            ]
            for tp in target_points:
                for cp in candidate_points:
                    if math.hypot(tp[0] - cp[0], tp[1] - cp[1]) <= max_dist:
                        if self._non_blue_conflict_yields_to_blue_target(
                                door, candidate,
                                strong_blue or fresh_blue_yields_conflict,
                                blue_confidence,
                                blue_count):
                            continue
                        if (
                                fresh_blue_yields_conflict
                                and float(candidate.confidence) <= blue_confidence + 0.25):
                            self.get_logger().info(
                                f'Non-blue conflict ignored for {door.door_id}: '
                                f'fresh aligned blue evidence conf={blue_confidence:.2f}, '
                                f'{candidate.door_color}_conf={float(candidate.confidence):.2f}.',
                                throttle_duration_sec=3.0)
                            continue
                        return True

        now_sec = now.nanoseconds / 1e9
        memory_sec = max(0.0, self._red_blue_conflict_memory_sec)
        for red in self._observed_red_door_positions:
            if memory_sec > 0.0 and now_sec - float(red.get('last_seen', now_sec)) > memory_sec:
                continue
            if int(red.get('count', 0.0)) < self._red_observation_min_count_for_blue_suppression:
                continue
            if float(red.get('confidence', 0.0)) < self._red_observation_min_confidence:
                continue
            red_xy = (float(red.get('x', 0.0)), float(red.get('y', 0.0)))
            if any(math.hypot(tp[0] - red_xy[0], tp[1] - red_xy[1]) <= max_dist
                   for tp in target_points):
                strict_same_wall_red = any(
                    self._same_wall_red_conflict_is_strict(
                        tp,
                        red_xy,
                        math.hypot(tp[0] - red_xy[0], tp[1] - red_xy[1]),
                        max_dist)
                    for tp in target_points)
                if self._red_memory_yields_to_blue_target(
                        red,
                        strong_blue or fresh_blue_yields_conflict,
                        blue_confidence,
                        blue_count) and not strict_same_wall_red:
                    continue
                if (
                        not strict_same_wall_red
                        and
                        fresh_blue_yields_conflict
                        and float(red.get('confidence', 0.0)) <= blue_confidence + 0.25):
                    self.get_logger().info(
                        f'Red memory conflict ignored for {door.door_id}: '
                        f'fresh aligned blue evidence conf={blue_confidence:.2f}, '
                        f'red_conf={float(red.get("confidence", 0.0)):.2f}.',
                        throttle_duration_sec=3.0)
                    continue
                return True
        return False

    def _blue_target_observation_strength(
            self, door: DoorInfo) -> tuple[bool, float, int]:
        if door.door_color != 'blue' or not self._door_has_map_identity(door):
            return False, 0.0, 0
        target_points = self._unique_xy_points(self._blue_evidence_points(door))
        xy = self._door_identity_xy(door)
        confidence = max(0.0, float(door.confidence))
        count = 0
        for target_point in target_points:
            cluster = self._find_observed_blue_cluster(
                target_point, merge_dist=self._observed_candidate_merge_dist())
            if cluster is not None:
                count = max(count, int(cluster.get('count', 0.0)))
                confidence = max(confidence, float(cluster.get('confidence', 0.0)))

        max_age = max(0.0, self._door_open_fresh_blue_max_age_sec)
        max_dist = max(0.20, self._door_open_fresh_blue_max_dist_m)
        min_conf = max(0.0, self._door_open_fresh_blue_min_confidence)
        now = self.get_clock().now()
        for candidate in self.detected_doors:
            if candidate.door_color != 'blue':
                continue
            if candidate.door_id.startswith('observed_blue_'):
                continue
            if not self._door_has_map_identity(candidate):
                continue
            candidate_confidence = float(candidate.confidence)
            if candidate_confidence < min_conf:
                continue
            if max_age > 0.0:
                try:
                    stamp = Time.from_msg(candidate.header.stamp)
                except Exception:
                    stamp = None
                if stamp is not None and stamp.nanoseconds > 0:
                    age = (now - stamp).nanoseconds / 1e9
                    if age > max_age:
                        continue
            candidate_points = self._unique_xy_points(
                self._blue_evidence_points(candidate))
            if any(
                    math.hypot(tp[0] - cp[0], tp[1] - cp[1]) <= max_dist
                    for tp in target_points
                    for cp in candidate_points):
                count = max(count, self._observed_blue_min_observations)
                confidence = max(confidence, candidate_confidence)

        strong_count = max(
            self._observed_blue_min_observations,
            self._confirmed_blue_bypass_red_conflict_count)
        strong_confidence = max(
            0.45, self._confirmed_blue_bypass_red_conflict_confidence)
        strong = count >= strong_count and confidence >= strong_confidence
        return strong, confidence, count

    def _non_blue_conflict_yields_to_blue_target(
            self,
            blue_door: DoorInfo,
            non_blue: DoorInfo,
            strong_blue: bool,
            blue_confidence: float,
            blue_count: int) -> bool:
        if not strong_blue:
            return False
        non_blue_confidence = float(non_blue.confidence)
        if non_blue.door_color == 'green':
            blue_xy = self._door_identity_xy(blue_door)
            green_xy = self._door_identity_xy(non_blue)
            blue_lateral = abs(
                self._axis_lateral_xy(blue_xy[0], blue_xy[1])
                - self._explore_center_y)
            green_lateral = abs(
                self._axis_lateral_xy(green_xy[0], green_xy[1])
                - self._explore_center_y)
            wall_threshold = max(
                0.15, self._observed_blue_min_abs_wall_y_m * 0.5)
            if blue_lateral >= wall_threshold and green_lateral < wall_threshold:
                self.get_logger().info(
                    f'Green conflict ignored for {blue_door.door_id}: '
                    f'strong blue evidence count={blue_count}, '
                    f'conf={blue_confidence:.2f}.',
                    throttle_duration_sec=3.0)
                return True
        if non_blue.door_color == 'red':
            blue_xy = self._door_identity_xy(blue_door)
            red_xy = self._door_identity_xy(non_blue)
            distance = math.hypot(blue_xy[0] - red_xy[0], blue_xy[1] - red_xy[1])
            if self._same_wall_red_conflict_is_strict(
                    blue_xy, red_xy, distance, self._blue_red_conflict_dist_m):
                return False
            if non_blue_confidence <= blue_confidence + 0.10:
                self.get_logger().info(
                    f'Red conflict ignored for {blue_door.door_id}: '
                    f'strong blue evidence count={blue_count}, '
                    f'blue_conf={blue_confidence:.2f}, '
                    f'red_conf={non_blue_confidence:.2f}.',
                    throttle_duration_sec=3.0)
                return True
        return False

    def _red_memory_yields_to_blue_target(
            self,
            red: dict[str, float],
            strong_blue: bool,
            blue_confidence: float,
            blue_count: int) -> bool:
        if not strong_blue:
            return False
        red_count = int(red.get('count', 0.0))
        red_confidence = float(red.get('confidence', 0.0))
        if red_count > max(blue_count + 2, blue_count * 1.5):
            return False
        if red_confidence > blue_confidence + 0.12:
            return False
        self.get_logger().info(
            f'Red memory conflict ignored: strong blue evidence '
            f'count={blue_count}, blue_conf={blue_confidence:.2f}, '
            f'red_count={red_count}, red_conf={red_confidence:.2f}.',
            throttle_duration_sec=3.0)
        return True

    def _is_blue_xy_opposite_opened_station(self, xy: tuple[float, float]) -> bool:
        base_window = max(0.0, self._opened_station_blue_suppression_progress_m)
        same_side_window = max(
            0.0,
            self._opened_station_same_side_blue_suppression_progress_m)
        opposite_side_window = max(
            0.0,
            self._opened_station_opposite_side_blue_suppression_progress_m)
        if base_window <= 0.0 and same_side_window <= 0.0 and opposite_side_window <= 0.0:
            return False
        if not self._opened_blue_door_positions:
            return False
        pose = self._current_map_pose()
        if pose is None:
            return False

        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        candidate_lateral = self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)

        for opened_xy in self._opened_blue_door_positions:
            opened_progress = self._axis_progress_xy(opened_xy[0], opened_xy[1])
            opened_lateral = (
                self._axis_lateral_xy(opened_xy[0], opened_xy[1])
                - self._explore_center_y)
            candidate_on_wall = abs(candidate_lateral) >= wall_side_threshold
            opened_on_wall = abs(opened_lateral) >= wall_side_threshold
            opposite_walls = (
                candidate_on_wall
                and opened_on_wall
                and candidate_lateral * opened_lateral < 0.0)
            same_wall = (
                candidate_on_wall
                and opened_on_wall
                and candidate_lateral * opened_lateral > 0.0)
            progress_window = base_window
            if same_wall:
                progress_window = max(progress_window, same_side_window)
            elif opposite_walls:
                progress_window = max(0.0, opposite_side_window)
                if progress_window <= 0.0:
                    continue
            if abs(candidate_progress - opened_progress) > progress_window:
                continue
            self.get_logger().info(
                f'파란 문 후보 ({xy[0]:.1f}, {xy[1]:.1f}) 제외: '
                f'이미 연 문 진행축 {opened_progress:.1f}m 근처의 재관측 후보입니다.',
                throttle_duration_sec=3.0)
            return True
        return False

    def _is_blue_xy_near_abandoned_station(self, xy: tuple[float, float]) -> bool:
        if not self._abandoned_blue_door_positions:
            return False
        same_window = max(
            0.0,
            self._abandoned_station_same_side_blue_suppression_progress_m)
        opposite_window = max(
            0.0,
            self._abandoned_station_opposite_side_blue_suppression_progress_m)
        if same_window <= 0.0 and opposite_window <= 0.0:
            return False

        candidate_progress = self._axis_progress_xy(xy[0], xy[1])
        candidate_lateral = (
            self._axis_lateral_xy(xy[0], xy[1])
            - self._explore_center_y)
        wall_side_threshold = max(
            0.15,
            self._observed_blue_min_abs_wall_y_m * 0.5)
        candidate_on_wall = abs(candidate_lateral) >= wall_side_threshold

        for abandoned_xy in self._abandoned_blue_door_positions:
            abandoned_progress = self._axis_progress_xy(
                abandoned_xy[0], abandoned_xy[1])
            abandoned_lateral = (
                self._axis_lateral_xy(abandoned_xy[0], abandoned_xy[1])
                - self._explore_center_y)
            abandoned_on_wall = abs(abandoned_lateral) >= wall_side_threshold
            same_wall = (
                candidate_on_wall
                and abandoned_on_wall
                and candidate_lateral * abandoned_lateral > 0.0)
            opposite_wall = (
                candidate_on_wall
                and abandoned_on_wall
                and candidate_lateral * abandoned_lateral < 0.0)
            if same_wall:
                progress_window = same_window
            elif opposite_wall:
                progress_window = opposite_window
            else:
                progress_window = min(same_window, opposite_window)
            if progress_window <= 0.0:
                continue
            if abs(candidate_progress - abandoned_progress) > progress_window:
                continue
            self.get_logger().info(
                f'Blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) rejected: '
                f'near an abandoned same physical door station '
                f'(station_progress={abandoned_progress:.1f}m, '
                f'window={progress_window:.1f}m).',
                throttle_duration_sec=3.0)
            return True
        return False

    def _maybe_abandon_stale_observed_blue_cluster(
            self, cluster: dict[str, float]) -> bool:
        if (self._observed_blue_stale_abandon_sec <= 0.0
                or self._observed_blue_stale_behind_m <= 0.0):
            return False
        count = int(cluster.get('count', 0.0))
        confidence = float(cluster.get('confidence', 0.0))
        if (count >= self._observed_blue_min_observations
                and confidence >= max(0.24, self._door_open_fresh_blue_min_confidence)
                and cluster.get('opened', 0.0) < 1.0
                and cluster.get('abandoned', 0.0) < 1.0):
            # A stable map-memory blue door should be retried or explicitly
            # abandoned by the approach/open failure budget, not silently dropped
            # just because exploration has moved past it.
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
            and not self._is_observed_blue_cluster_too_far_behind_for_target(cluster)
            and not self._is_blue_xy_beyond_explore_limit(
                self._cluster_xy(cluster), 'observed unopened blue candidate')
            and not self._is_blue_xy_at_or_after_observed_exit(
                self._cluster_xy(cluster))
            and not self._is_blue_xy_near_detected_exit(self._cluster_xy(cluster))

        ]

    def _exit_blocking_unopened_blue_doors(self) -> list[dict[str, float]]:
        blockers: list[dict[str, float]] = []
        for cluster in self._observed_blue_doors:
            if int(cluster.get('count', 0.0)) < self._observed_blue_min_observations:
                continue
            if cluster.get('opened', 0.0) >= 1.0 or cluster.get('abandoned', 0.0) >= 1.0:
                continue
            xy = self._cluster_xy(cluster)
            if self._is_observed_blue_cluster_suppressed(cluster):
                continue
            if not self._observed_blue_cluster_is_stable_unopened_target_memory(cluster):
                continue
            if self._is_exit_blocking_blue_xy_beyond_scan_window(xy):
                continue
            if self._is_blue_position_opened(xy) or self._is_blue_position_abandoned(xy):
                continue
            if (
                    self._is_blue_xy_at_or_after_observed_exit(xy)
                    or self._is_blue_xy_near_detected_exit(xy)):
                continue
            if self._is_observed_blue_cluster_weak_in_pre_exit_zone(cluster):
                continue
            lateral = abs(
                self._axis_lateral_xy(xy[0], xy[1]) - self._explore_center_y)
            wall_side_threshold = max(
                0.35,
                self._observed_blue_min_abs_wall_y_m * 0.5)
            wall_side_candidate = lateral >= wall_side_threshold
            max_wall = max(0.0, self._observed_blue_max_abs_wall_y_m)
            if (
                    max_wall > 0.0
                    and lateral > max_wall + self._blue_handle_validation_overshoot_m):
                continue
            if not wall_side_candidate:
                continue

            # This helper only decides whether exiting must be delayed. A stable
            # wall-side blue cluster near the observed exit should still block
            # exit, because it may be the last door. Target selection keeps its
            # stricter exit/green suppression elsewhere.
            if self._is_blue_xy_opposite_opened_station(xy):
                continue
            if self._is_blue_xy_near_abandoned_station(xy):
                continue
            if self._is_blue_xy_near_observed_red(xy):
                continue
            blockers.append(cluster)
        return blockers

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

    def _target_exit_blocking_unopened_blue(
            self, unopened: list[dict[str, float]], reason: str) -> bool:
        targets = [self._observed_blue_cluster_to_door(cluster)
                   for cluster in unopened]
        if not targets:
            return False

        pose = self._current_map_pose()
        robot_progress = None
        if pose is not None:
            robot_progress = self._axis_progress_xy(pose[0], pose[1])
            behind_tolerance = max(
                0.0,
                self._observed_blue_stale_behind_m,
                self._observed_blue_target_max_behind_m)
            near_or_ahead = [
                door for door in targets
                if self._door_progress(door) >= robot_progress - behind_tolerance
            ]
            if near_or_ahead:
                targets = near_or_ahead

        def score(door: DoorInfo) -> tuple[float, float, float]:
            dist = self._door_distance_from_robot(door)
            progress = self._door_progress(door)
            behind_penalty = 0.0
            if robot_progress is not None and progress < robot_progress:
                behind_penalty = robot_progress - progress
            return (
                behind_penalty,
                dist if dist is not None else float('inf'),
                progress,
            )

        selected = min(targets, key=score)
        self.target_door = self._door_with_axis_aligned_approach(selected)
        self._set_locked_blue_anchor(self.target_door)
        xy = self._door_identity_xy(self.target_door)
        self.get_logger().warn(
            f'비상구 전 미개방 파란 문 우선 접근({reason}): '
            f'{self.target_door.door_id} at ({xy[0]:.1f}, {xy[1]:.1f}), '
            f'conf={float(self.target_door.confidence):.2f}.')
        self._nav_done = False
        self._nav_failed = False
        self._explore_start_time = self.get_clock().now()
        self._transition(State.NAVIGATING)
        self.target_door_pub.publish(self.target_door)
        return True

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
            cluster = self._find_observed_blue_cluster(
                xy,
                merge_dist=self._observed_candidate_merge_dist())
        if cluster is None:
            return False

        if failure_kind == 'open':
            field = 'open_failures'
            limit = self._max_door_open_failures_before_abandon
            label = '문 개방'
        else:
            field = 'approach_failures'
            limit = self._max_door_approach_failures_before_abandon
            if self._is_pre_exit_blue_failure_candidate(xy):
                limit = min(
                    limit,
                    self._pre_exit_blue_approach_failures_before_abandon)
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

    def _is_pre_exit_blue_failure_candidate(self, xy: tuple[float, float]) -> bool:
        if self._opened_exit_ready_count() < self._min_opened_doors_before_exit:
            return False
        if not self._opened_blue_door_positions:
            return False
        if (
                self._exit_xy_for_blue_suppression() is None
                and self._exit_xy_for_blue_candidate_filter() is None
                and self._last_valid_exit_xy is None
                and self._last_provisional_exit_xy is None
                and self._semantic_green_exit_xy() is None):
            return False

        progress = self._axis_progress_xy(xy[0], xy[1])
        farthest_opened = self._farthest_resolved_blue_progress()
        if farthest_opened is None:
            return False
        min_after_opened = max(
            0.35,
            self._door_open_fresh_blue_max_dist_m * 0.5)
        if progress < farthest_opened + min_after_opened:
            return False

        self.get_logger().warn(
            f'Pre-exit blue candidate ({xy[0]:.1f}, {xy[1]:.1f}) '
            'approach failure will use the stricter exit-tail abandon limit.',
            throttle_duration_sec=3.0)
        return True

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
        self._add_door_id_keys(self._abandoned_door_ids, door.door_id)
        self._mark_semantic_door_abandoned(door)
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
            self._add_door_id_keys(self._failed_door_ids, abandoned_id)
            self._clear_locked_blue_anchor()
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

        merge_dist = max(
            self._observed_candidate_merge_dist(),
            self._opened_physical_door_merge_dist_m)
        evidence_points: list[tuple[float, float]] = []
        for point in self._blue_evidence_points(door):
            if all(math.hypot(point[0] - old[0], point[1] - old[1]) > 0.05
                   for old in evidence_points):
                evidence_points.append(point)

        for point in evidence_points:
            cluster = self._find_observed_blue_cluster(point, merge_dist=merge_dist)
            if cluster is None:
                continue
            self._add_observed_blue_cluster_id(cluster, door.door_id)
            cluster['opened'] = 1.0
            cluster['approach_failures'] = 0.0
            cluster['open_failures'] = 0.0
            cluster['failures'] = 0.0

        station_xy = self._door_identity_xy(door)
        if not any(
                self._blue_xy_matches_physical_station(
                    station_xy, old, self._opened_physical_door_merge_dist_m)
                for old in self._opened_blue_door_positions):
            self._opened_blue_door_positions.append(station_xy)

        opened_match_points = [station_xy] + evidence_points

        for cluster in self._observed_blue_doors:
            cluster_xy = self._cluster_xy(cluster)
            if any(
                    math.hypot(cluster_xy[0] - opened_xy[0],
                               cluster_xy[1] - opened_xy[1]) <= merge_dist
                    or self._blue_xy_matches_physical_station(
                        cluster_xy, opened_xy, self._opened_physical_door_merge_dist_m)
                    for opened_xy in opened_match_points):
                cluster['opened'] = 1.0
                cluster['approach_failures'] = 0.0
                cluster['open_failures'] = 0.0
                cluster['failures'] = 0.0

    def _door_with_axis_aligned_approach(self, door: DoorInfo) -> DoorInfo:
        adjusted = copy.deepcopy(door)
        if (not self._axis_door_approach_enabled
                or adjusted.door_color != 'blue'
                or not self._door_has_map_identity(adjusted)):
            return adjusted

        handle_x, handle_y = self._door_handle_xy(adjusted)
        handle_progress = self._axis_progress_xy(handle_x, handle_y)
        handle_lateral = self._axis_lateral_xy(handle_x, handle_y)
        approach_progress = handle_progress
        progress_note = ''
        if adjusted.door_pose.header.frame_id == 'map':
            pose_x = float(adjusted.door_pose.pose.position.x)
            pose_y = float(adjusted.door_pose.pose.position.y)
            pose_progress = self._axis_progress_xy(pose_x, pose_y)
            pose_lateral = self._axis_lateral_xy(pose_x, pose_y)
            handle_side = handle_lateral - self._explore_center_y
            pose_side = pose_lateral - self._explore_center_y
            max_delta = max(0.0, self._axis_door_max_handle_pose_progress_delta_m)
            same_side = handle_side * pose_side >= 0.0
            if max_delta > 0.0 and abs(handle_progress - pose_progress) > max_delta:
                if same_side or abs(pose_side) < 0.35:
                    approach_progress = pose_progress
                    progress_note = (
                        f', pose_axis=({pose_progress:.2f}, {pose_lateral:.2f})')
        parking_pose = self._axis_side_door_front_parking_pose(
            approach_progress, handle_lateral)
        if parking_pose is None:
            return adjusted
        approach_progress, approach_lateral, approach_yaw, wall_lateral = parking_pose
        goal_x, goal_y = self._axis_to_map_xy(
            approach_progress, approach_lateral)
        adjusted.door_pose.header.frame_id = 'map'
        adjusted.door_pose.pose.position.x = goal_x
        adjusted.door_pose.pose.position.y = goal_y
        adjusted.door_pose.pose.position.z = 0.0
        self._set_pose_yaw(adjusted.door_pose.pose, approach_yaw)
        self.get_logger().info(
            f'문 접근 목표 보정: {adjusted.door_id} '
            f'handle_axis=({handle_progress:.2f}, {handle_lateral:.2f}) '
            f'goal_axis=({approach_progress:.2f}, {approach_lateral:.2f})'
            f' wall_axis=({approach_progress:.2f}, {wall_lateral:.2f})'
            f'{progress_note} '
            f'yaw={math.degrees(approach_yaw):.0f}deg')
        return adjusted

    def _set_pose_yaw(self, pose, yaw: float):
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw / 2.0)
        pose.orientation.w = math.cos(yaw / 2.0)

    def _nearest_forward_blue_door(self, doors: list[DoorInfo]) -> DoorInfo | None:
        if not doors:
            return None
        fresh_detected = [
            d for d in doors
            if not d.door_id.startswith('observed_blue_')
            and self._is_door_recent_observation(d)
        ]
        if fresh_detected:
            doors = fresh_detected
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

    def _door_open_target_pose_xy_yaw(
            self, door: DoorInfo) -> tuple[float, float, float] | None:
        if door.door_pose.header.frame_id != 'map':
            return None

        if (
                self._axis_door_approach_enabled
                and door.door_color == 'blue'
                and self._door_has_map_identity(door)):
            handle_x, handle_y = self._door_handle_xy(door)
            handle_progress = self._axis_progress_xy(handle_x, handle_y)
            handle_lateral = self._axis_lateral_xy(handle_x, handle_y)
            parking_pose = self._axis_side_door_front_parking_pose(
                handle_progress, handle_lateral,
                min_side_lateral=self._door_open_axis_min_side_lateral_m)
            if parking_pose is not None:
                open_progress, open_lateral, open_yaw, _ = parking_pose
                goal_x, goal_y = self._axis_to_map_xy(
                    open_progress, open_lateral)
                return goal_x, goal_y, open_yaw

        return (
            float(door.door_pose.pose.position.x),
            float(door.door_pose.pose.position.y),
            self._pose_yaw(door.door_pose.pose))

    def _pose_yaw(self, pose) -> float:
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _door_open_pose_error(self, door: DoorInfo):
        target_pose = self._door_open_target_pose_xy_yaw(door)
        if target_pose is None:
            return None
        pose = self._current_map_pose()
        if pose is None:
            return None

        rx, ry, robot_yaw = pose
        gx, gy, goal_yaw = target_pose
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)
        yaw_error = self._normalize_angle(goal_yaw - robot_yaw)
        forward_error = math.cos(robot_yaw) * dx + math.sin(robot_yaw) * dy
        lateral_error = -math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dy
        return dist, yaw_error, forward_error, lateral_error

    def _blue_target_open_pose_aligned_for_safe_memory(self, door: DoorInfo) -> bool:
        error = self._door_open_pose_error(door)
        if error is None:
            return False
        dist, yaw_error, _forward_error, lateral_error = error
        return (
            dist <= max(self._door_open_ready_max_dist_m + 0.06, 0.30)
            and abs(lateral_error) <= max(
                self._door_open_ready_lateral_tolerance_m + 0.08, 0.20)
            and abs(yaw_error) <= max(
                self._door_open_ready_yaw_tolerance + math.radians(4.0),
                math.radians(12.0))
        )

    def _prepare_door_opening_pose(self, door: DoorInfo) -> str:
        error = self._door_open_pose_error(door)
        if error is None:
            return 'ready'
        dist, yaw_error, forward_error, lateral_error = error
        yaw_abs = abs(yaw_error)
        lateral_abs = abs(lateral_error)
        ready_dist_tolerance = self._door_open_ready_max_dist_m + 0.015
        ready_lateral_tolerance = self._door_open_ready_lateral_tolerance_m + 0.015
        ready_yaw_tolerance = self._door_open_ready_yaw_tolerance + math.radians(1.0)
        position_ready = (
            dist <= ready_dist_tolerance
            and lateral_abs <= ready_lateral_tolerance)

        if position_ready and yaw_abs <= ready_yaw_tolerance:
            self._door_open_align_start_time = None
            return 'ready'

        close_ready_lateral_tolerance = max(
            ready_lateral_tolerance,
            self._door_open_ready_lateral_tolerance_m + 0.06,
            0.18)
        if (
                dist <= ready_dist_tolerance
                and lateral_abs <= close_ready_lateral_tolerance
                and yaw_abs <= ready_yaw_tolerance):
            self.get_logger().info(
                f'Fine door approach accepted within close-ready tolerance: '
                f'{door.door_id}, dist={dist:.2f}m, '
                f'lateral={lateral_abs:.2f}m/'
                f'{close_ready_lateral_tolerance:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg.')
            self._door_open_align_start_time = None
            return 'ready'

        straight_ready_dist_tolerance = max(
            ready_dist_tolerance,
            self._door_open_straight_ready_max_dist_m)
        straight_ready_lateral_tolerance = min(
            close_ready_lateral_tolerance,
            max(0.06, self._door_open_ready_lateral_tolerance_m * 0.5))
        if (
                dist <= straight_ready_dist_tolerance
                and lateral_abs <= straight_ready_lateral_tolerance
                and yaw_abs <= ready_yaw_tolerance):
            self.get_logger().info(
                f'Fine door approach accepted from straight close pose: '
                f'{door.door_id}, dist={dist:.2f}m/'
                f'{straight_ready_dist_tolerance:.2f}m, '
                f'lateral={lateral_abs:.2f}m/'
                f'{straight_ready_lateral_tolerance:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg.')
            self._door_open_align_start_time = None
            return 'ready'

        if (dist <= self._door_open_fine_control_max_dist_m
                and lateral_abs <= self._door_open_fine_lateral_tolerance_m):
            now = self.get_clock().now()
            if self._door_open_align_start_time is None:
                self._door_open_align_start_time = now
                self.get_logger().info(
                    f'Fine door approach started: {door.door_id}, '
                    f'dist={dist:.2f}m, lateral={lateral_error:.2f}m, '
                    f'forward={forward_error:.2f}m, '
                    f'yaw_error={math.degrees(yaw_abs):.0f}deg')

            elapsed = (now - self._door_open_align_start_time).nanoseconds / 1e9
            if elapsed > self._door_open_align_timeout_sec:
                self.get_logger().warn(
                    f'Fine door approach timeout: {door.door_id}, '
                    f'dist={dist:.2f}m, lateral={lateral_error:.2f}m, '
                    f'forward={forward_error:.2f}m, '
                    f'yaw_error={math.degrees(yaw_abs):.0f}deg')
                self._door_open_align_start_time = None
                return 'retry'

            if (lateral_abs > ready_lateral_tolerance
                    and yaw_abs <= ready_yaw_tolerance):
                if (forward_error < -0.05
                        and not self._door_open_fine_allow_reverse):
                    self.get_logger().warn(
                        f'Fine door approach cannot correct lateral error '
                        f'without reversing: {door.door_id}, '
                        f'lateral={lateral_abs:.2f}m/'
                        f'{self._door_open_ready_lateral_tolerance_m:.2f}m, '
                        f'forward={forward_error:.2f}m, '
                        f'dist={dist:.2f}m')
                    self._door_open_align_start_time = None
                    return 'retry'

            if (dist > ready_dist_tolerance
                    and yaw_abs <= ready_yaw_tolerance
                    and forward_error <= 0.0
                    and not self._door_open_fine_allow_reverse):
                self.get_logger().warn(
                    f'Fine door approach cannot correct by forward motion only: '
                    f'{door.door_id}, forward={forward_error:.2f}m, '
                    f'dist={dist:.2f}m')
                self._door_open_align_start_time = None
                return 'retry'

            if (dist > ready_dist_tolerance
                    and forward_error < -0.05
                    and not self._door_open_fine_allow_reverse):
                self.get_logger().warn(
                    f'Fine door approach is past the parking pose: '
                    f'{door.door_id}, forward={forward_error:.2f}m, '
                    f'dist={dist:.2f}m. Retrying with Nav2 instead of '
                    'backing up in fine control.')
                self._door_open_align_start_time = None
                return 'retry'

            twist = Twist()
            if dist > ready_dist_tolerance:
                linear = self._door_open_fine_position_kp * forward_error
                if not self._door_open_fine_allow_reverse:
                    linear = max(0.0, linear)
                limit = abs(self._door_open_fine_linear_vel)
                twist.linear.x = max(-limit, min(limit, linear))
                if (abs(twist.linear.x) < self._door_open_fine_min_linear_vel
                        and abs(forward_error) > ready_dist_tolerance):
                    if forward_error > 0.0 or self._door_open_fine_allow_reverse:
                        sign = 1.0 if forward_error >= 0.0 else -1.0
                        twist.linear.x = sign * self._door_open_fine_min_linear_vel

            if (yaw_abs > ready_yaw_tolerance
                    and not (
                        lateral_abs > ready_lateral_tolerance
                        and yaw_abs <= math.radians(28.0))):
                angular = self._door_open_align_kp * yaw_error
                limit = abs(self._door_open_align_angular_vel)
                twist.angular.z = max(-limit, min(limit, angular))
                if abs(twist.angular.z) < 0.08:
                    twist.angular.z = 0.08 if yaw_error > 0.0 else -0.08
                if (forward_error > ready_dist_tolerance
                        and lateral_abs <= self._door_open_fine_lateral_tolerance_m
                        and yaw_abs <= math.radians(35.0)):
                    slow_forward = min(
                        abs(self._door_open_fine_linear_vel),
                        max(self._door_open_fine_min_linear_vel, 0.035))
                    twist.linear.x = max(twist.linear.x, slow_forward)
                else:
                    twist.linear.x = 0.0
            elif lateral_abs > ready_lateral_tolerance:
                limit = abs(self._door_open_align_angular_vel)
                direction = 1.0
                if self._door_open_fine_allow_reverse and forward_error < -0.02:
                    direction = -1.0
                angular = (
                    self._door_open_align_kp * 0.55 * yaw_error
                    + self._door_open_align_kp * lateral_error)
                if direction < 0.0:
                    angular = (
                        self._door_open_align_kp * 0.55 * yaw_error
                        - self._door_open_align_kp * lateral_error)
                twist.angular.z = max(-limit, min(limit, angular))
                if abs(twist.angular.z) < 0.08:
                    twist.angular.z = 0.08 if angular >= 0.0 else -0.08
                slow_forward = min(
                    abs(self._door_open_fine_linear_vel),
                    max(self._door_open_fine_min_linear_vel, 0.035))
                twist.linear.x = direction * slow_forward

            self.cmd_vel_pub.publish(twist)
            self.get_logger().info(
                f'Fine door approach: {door.door_id}, '
                f'dist={dist:.2f}m/{self._door_open_ready_max_dist_m:.2f}m, '
                f'lateral={lateral_abs:.2f}m/'
                f'{self._door_open_ready_lateral_tolerance_m:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg/'
                f'{math.degrees(self._door_open_ready_yaw_tolerance):.0f}deg, '
                f'cmd=({twist.linear.x:.2f}, {twist.angular.z:.2f})')
            return 'waiting'

        if lateral_abs > ready_lateral_tolerance:
            self.get_logger().warn(
                f'Door opening pose lateral error too large: {door.door_id} '
                f'lateral={lateral_abs:.2f}m/'
                f'{self._door_open_ready_lateral_tolerance_m:.2f}m, '
                f'dist={dist:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg')
            self._door_open_align_start_time = None
            return 'retry'

        if dist > ready_dist_tolerance:
            self.get_logger().warn(
                f'문 개방 자세 미달: {door.door_id} '
                f'dist={dist:.2f}m/{self._door_open_ready_max_dist_m:.2f}m, '
                f'yaw_error={math.degrees(yaw_abs):.0f}deg/'
                f'{math.degrees(self._door_open_ready_yaw_tolerance):.0f}deg')
            self._door_open_align_start_time = None
            return 'retry'

        if yaw_abs <= ready_yaw_tolerance:
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
        self._reset_door_nav_stuck_watch()
        retry_target = copy.deepcopy(self.target_door)
        retry_target.confidence = -max(1.0e-3, abs(float(retry_target.confidence)))
        self.target_door_pub.publish(retry_target)
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
        self._explore_lane_success_since_flip = 0
        self._last_lane_sign = -self._last_lane_sign if self._last_lane_sign != 0.0 else -1.0
        side = '좌측' if self._last_lane_sign > 0.0 else '우측'
        self.get_logger().info(f'다음 탐색 waypoint는 {side} 차선을 우선합니다.')

    def _alternate_explore_lane_after_success(self):
        has_color_hint = any(
            d.door_color in ('blue', 'red')
            and not self._is_door_failed_for_observation(d)
            and not self._is_door_opened_for_observation(d)
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
        red_avoid_sign, red_avoid_reason = self._red_avoidance_lane_sign()
        lane_clearances = {
            sign: self._lane_scan_clearance(robot_x, robot_y, robot_yaw, goal_y)
            for goal_y, sign in candidates
        }
        left_clearance = lane_clearances.get(1.0, 0.0)
        right_clearance = lane_clearances.get(-1.0, 0.0)
        side_obstacle_avoid_sign = 0.0
        side_block_threshold = max(
            self._explore_obstacle_stop_m,
            min(self._explore_obstacle_slow_m, 1.20))
        side_open_threshold = max(
            self._explore_nav_blue_lane_min_clearance_m,
            side_block_threshold + 0.25)
        if (right_clearance < side_block_threshold
                and left_clearance >= max(side_open_threshold, right_clearance + 0.35)):
            side_obstacle_avoid_sign = 1.0
        elif (left_clearance < side_block_threshold
                and right_clearance >= max(side_open_threshold, left_clearance + 0.35)):
            side_obstacle_avoid_sign = -1.0
        center_clearance = lane_clearances.get(0.0, self._lane_scan_clearance(
            robot_x, robot_y, robot_yaw, self._explore_center_y))
        center_recently_failed = (
            self._is_explore_lane_recently_failed(0.0, self._axis_progress_xy(robot_x, robot_y))
            or self._is_explore_lane_recently_failed(
                0.0,
                self._axis_progress_xy(robot_x, robot_y) + max(
                    self._explore_nav_min_step_m,
                    min(self._explore_nav_step_m, self._explore_nav_scan_lookahead_m)))
        )
        centerline_preferred_without_target = (
            not blue_lane_active
            and front_avoid_sign == 0.0
            and side_obstacle_avoid_sign == 0.0
        )
        if (centerline_preferred_without_target
                and not center_recently_failed
                and center_clearance >= self._explore_obstacle_slow_m):
            scored_text = []
            for goal_y, sign in candidates:
                clearance = lane_clearances.get(sign, self._lane_scan_clearance(
                    robot_x, robot_y, robot_yaw, goal_y))
                label = 'C' if sign == 0.0 else ('L' if sign > 0.0 else 'R')
                scored_text.append(f'{label}:y={goal_y:.2f}/clear={clearance:.2f}')
            if red_avoid_sign != 0.0:
                reason = (
                    '중앙 차선 선택 | 파란 문 목표/전방 장애물 없음, '
                    f'{red_avoid_reason}; 중앙선 유지로 벽면 빨간문과 거리 확보 | '
                    f'LiDAR lane clearances: {", ".join(scored_text)}')
            else:
                reason = (
                    '중앙 차선 선택 | 파란 문/전방 장애물/빨간 문 회피 이유 없음, '
                    '초기 진행축 유지 | '
                    f'LiDAR lane clearances: {", ".join(scored_text)}')
            return self._explore_center_y, 0.0, reason
        best: tuple[float, float, float, float] | None = None
        scored_text: list[str] = []
        current_progress = self._axis_progress_xy(robot_x, robot_y)
        expected_goal_progress = current_progress + max(
            self._explore_nav_min_step_m,
            min(self._explore_nav_step_m, self._explore_nav_scan_lookahead_m))
        for goal_y, sign in candidates:
            clearance = lane_clearances.get(sign, self._lane_scan_clearance(
                robot_x, robot_y, robot_yaw, goal_y))
            score = clearance
            lane_blocked = (
                self._is_explore_lane_recently_failed(sign, current_progress)
                or self._is_explore_lane_recently_failed(sign, expected_goal_progress)
            )
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
            if red_avoid_sign != 0.0:
                if sign == red_avoid_sign:
                    score += self._explore_nav_front_obstacle_bias_score * 0.90
                elif sign == -red_avoid_sign:
                    score -= self._explore_nav_front_obstacle_bias_score * 1.20
                elif sign == 0.0:
                    score -= 0.15
            if side_obstacle_avoid_sign != 0.0:
                if sign == side_obstacle_avoid_sign:
                    score += 0.85
                elif sign == -side_obstacle_avoid_sign:
                    score -= 1.10
                elif sign == 0.0:
                    score -= 0.70
            if (blue_lane_active and front_avoid_sign != 0.0
                    and sign == preferred_sign
                    and clearance >= self._explore_nav_blue_lane_min_clearance_m):
                score += self._explore_nav_front_obstacle_bias_score * 0.65
            if sign != 0.0 and sign == self._last_lane_sign:
                score += self._explore_nav_keep_lane_bias_score * (2.0 if blue_lane_active else 1.0)
            if sign == 0.0 and (blue_lane_active or front_avoid_sign != 0.0):
                score -= 0.28
            score -= abs(goal_y - current_lateral) * 0.06
            if clearance < self._explore_obstacle_stop_m:
                score -= 10.0
            elif clearance < self._explore_obstacle_slow_m:
                score -= 0.55
            if (blue_lane_active and sign == preferred_sign
                    and clearance < self._explore_nav_blue_lane_min_clearance_m):
                score -= 1.75
            if lane_blocked:
                score -= 14.0

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
        elif red_avoid_sign != 0.0:
            hint = red_avoid_reason
        else:
            hint = preferred_reason if preferred_sign != 0.0 else radar_reason
        reason = (
            f'{side} 차선 선택 | {hint} | '
            f'LiDAR lane scores: {", ".join(scored_text)}')
        if clearance < self._explore_obstacle_slow_m:
            reason += ' | 가까운 장애물 감지, 우회 차선 우선'
        return goal_y, sign, reason

    def _red_avoidance_lane_sign(self) -> tuple[float, str]:
        if self._no_blue_exit_mode():
            return 0.0, 'No-blue mode: keep center unless LiDAR obstacle requires avoidance'

        pose = self._current_map_pose()
        red_candidates: list[tuple[float, float, str]] = []

        red_doors = [
            d for d in self.detected_doors
            if d.door_color == 'red'
            and self._is_door_recent_observation(d)
            and self._door_has_map_identity(d)
        ]
        if pose is None:
            for red in red_doors:
                red_side = self._door_side_sign(red)
                if red_side != 0.0:
                    red_candidates.append((-float(red.confidence), red_side, red.door_id))
        else:
            rx, ry, _ = pose
            robot_progress = self._axis_progress_xy(rx, ry)
            for red in red_doors:
                xy = self._door_identity_xy(red)
                red_side = self._door_side_sign(red)
                if red_side == 0.0:
                    continue
                dist = math.hypot(xy[0] - rx, xy[1] - ry)
                red_candidates.append((dist, red_side, red.door_id))

            now_sec = self.get_clock().now().nanoseconds / 1e9
            memory_sec = max(0.0, self._red_blue_conflict_memory_sec)
            for idx, red in enumerate(self._observed_red_door_positions):
                last_seen = float(red.get('last_seen', 0.0))
                if memory_sec > 0.0 and now_sec - last_seen > memory_sec:
                    continue
                red_x = float(red.get('x', 0.0))
                red_y = float(red.get('y', 0.0))
                red_progress = self._axis_progress_xy(red_x, red_y)
                red_lateral = self._axis_lateral_xy(red_x, red_y)
                progress_delta = red_progress - robot_progress
                if progress_delta < -2.2 or progress_delta > 3.8:
                    continue
                if abs(red_lateral) < max(0.35, self._observed_blue_min_abs_wall_y_m * 0.5):
                    continue
                red_side = 1.0 if red_lateral > 0.0 else -1.0
                score_dist = abs(progress_delta) + max(0.0, 0.45 - abs(red_lateral))
                red_candidates.append((score_dist, red_side, f'observed_red_{idx}'))

        if not red_candidates:
            return 0.0, 'No red door lane memory'

        _, red_side, red_label = min(red_candidates, key=lambda item: item[0])
        if red_side == 0.0:
            return 0.0, f'Red door {red_label} centered'
        safe_sign = -red_side
        safe_side = 'left' if safe_sign > 0.0 else 'right'
        red_side_name = 'left' if red_side > 0.0 else 'right'
        return safe_sign, (
            f'Red door memory {red_label} on {red_side_name}; '
            f'prefer {safe_side} lane')

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
        current_lateral = self._axis_lateral_xy(robot_x, robot_y)
        goal_lateral = goal_y - current_lateral
        half_width = max(0.20, self._explore_nav_lane_width_m * 0.5)
        min_forward = max(0.35, self._explore_scan_min_range_m)
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
                if min_forward <= forward <= lookahead:
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
        return 0.0, 'LiDAR 좌우 여유 유사, 중앙 차선 유지'

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
        old_state = self.state
        nav2_motion_states = {
            State.NAVIGATING,
            State.EXPLORING_NAVIGATING,
            State.EXITING,
        }
        self.get_logger().info(f'State: {old_state.name} → {new_state.name}')
        self.state = new_state

        if new_state == State.NAVIGATING:
            self._explore_observation_wait_start_time = None
            self._explore_observation_wait_completed = True
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._clear_explore_nav_stuck_watch()
            self._reset_door_nav_stuck_watch()
            self._explore_nav_goal_id = ''
            self._explore_nav_goal_xy = None
            self._explore_nav_clearance_deferral_count = 0
            self._last_target_republish_time = None
            if old_state not in nav2_motion_states:
                self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
            self._door_open_settle_start_time = None
            self._door_open_align_start_time = None
        elif new_state == State.EXPLORING_NAVIGATING:
            self._explore_observation_wait_start_time = None
            self._explore_observation_wait_completed = True
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._clear_door_nav_stuck_watch()
            self._explore_nav_clearance_deferral_count = 0
            self._last_target_republish_time = None
            if old_state not in nav2_motion_states:
                self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
            self._reset_explore_nav_stuck_watch()
        elif new_state == State.EXITING:
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._clear_explore_nav_stuck_watch()
            self._clear_door_nav_stuck_watch()
            self._explore_nav_goal_xy = None
            if self._send_exit_goal():
                self._nav_start_time = self.get_clock().now()
            else:
                self.state = State.EXPLORING
                self._nav_start_time = None
                if self._explore_start_time is None:
                    self._explore_start_time = self.get_clock().now()
        elif new_state == State.EXPLORING:
            self._explore_observation_wait_start_time = None
            self._explore_observation_wait_completed = False
            self._clear_explore_nav_stuck_watch()
            self._clear_door_nav_stuck_watch()
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
            self._explore_nav_goal_xy = None
            self._explore_nav_clearance_deferral_count = 0
            self._last_target_republish_time = None
        elif new_state == State.MISSION_COMPLETE:
            self._explore_observation_wait_start_time = None
            self._explore_observation_wait_completed = True
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._clear_explore_nav_stuck_watch()
            self._clear_door_nav_stuck_watch()
            self._explore_nav_goal_xy = None
            self._exit_crossing_start_time = None
            self._last_target_republish_time = None
            self.cmd_vel_pub.publish(Twist())
        else:
            self._explore_observation_wait_start_time = None
            self._explore_observation_wait_completed = True
            self._explore_waypoint_scan_pending = False
            self._explore_waypoint_scan_start_time = None
            self._post_open_advance_pending = False
            self._post_open_advance_start_time = None
            self._post_open_advance_start_progress = None
            self._clear_explore_nav_stuck_watch()
            self._clear_door_nav_stuck_watch()
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
