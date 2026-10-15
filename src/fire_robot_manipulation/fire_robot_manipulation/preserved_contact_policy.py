"""
manipulation_node.py

MoveIt2(moveit_py) 기반 PIPER 로봇팔 문 개방 노드.

동작 시퀀스:
  1. Pre-grasp  : 손잡이 앞 10cm 위치로 팔 이동
  2. Grasp      : 손잡이 위치로 직선 이동 + 그리퍼 닫기
  3. Press      : 레버를 아래로 눌러 래치 해제
  4. Base push  : 레버를 놓고 팔을 회수한 뒤 모바일 베이스가 전진
  5. Release    : 손잡이를 놓고 팔 복귀 후 차체 후진

의존 패키지: moveit_py (MoveIt2 Python bindings)
실제 하드웨어 연동 시 MoveItPy 초기화 블록의 주석을 해제하세요.
"""

import time
import math
import re
import subprocess
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from geometry_msgs.msg import (
    PointStamped, Pose, PoseStamped, Point, Quaternion, Twist)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener

from fire_robot_interfaces.msg import DoorInfo
from fire_robot_interfaces.srv import OpenDoor
from .piper_actual_kinematics import (
    PIPER_HOME, PIPER_JOINT_NAMES, PIPER_JOINT_LIMITS, PIPER_STOW, PIPER_BODY_OUTLINE,
    PiperActualKinematics)
from .contact_feedback import (
    fresh_sample, observed_clearance, next_press_depth, stable_lever_motion, bounded_press_step,
    bounded_contact_target, SimMotionDeadline, measured_wrist_rotation,
    swept_contact_reference, projected_contact_increment, coordinated_contact_endpoint,
    contact_posture_command, released_grip_clearance, robust_heading_rate, measured_press_target,
    predicted_contact_rotation, robust_contact_speed, chassis_supplied_opening_speed,
    active_heading_rate, observed_contact_curvature, observed_probe_motion,
    SceneObservationPending, ObservedSceneGate, ObservedContactProgress, measured_contact_endpoint,
    measured_tangential_relaxation, bounded_wrist_yaw_lead,
    conservative_contact_curvature, contact_steady_heading)

try:
    from moveit.planning import MoveItPy
    _HAS_MOVEIT = True
except ImportError:
    _HAS_MOVEIT = False

try:
    from piper_msgs.msg import PosCmd
    _HAS_PIPER_MSGS = True
except ImportError:
    PosCmd = None
    _HAS_PIPER_MSGS = False


# 문 개방 파라미터
PRE_GRASP_OFFSET = 0.12   # 손잡이 앞 (m)
PULL_DISTANCE    = 0.35   # 당기기 거리 (m)
LEVER_PRESS_DISTANCE = 0.07  # 레버 손잡이를 아래로 누르는 거리 (m)
GRIPPER_OPEN     = 0.08   # 그리퍼 열림 폭 (m)
GRIPPER_CLOSE    = 0.01   # 그리퍼 닫힘 폭 (m)
GRIPPER_LEVER    = 0.045  # 52 mm lever: position error supplies grip force (m)


class ManipulationNode(Node):
    """PIPER 로봇팔 제어 및 문 개방 서비스 노드"""

    def __init__(self):
        super().__init__('manipulation_node')

        self.declare_parameter('planning_group', 'piper_arm')
        self.declare_parameter('gripper_group', 'piper_gripper')
        self.declare_parameter('sim_mode', True)
        self.declare_parameter('sim_control_enabled', False)
        self.declare_parameter('sim_command_transport', 'ign')
        self.declare_parameter('sim_step_sec', 1.0)
        self.declare_parameter('sim_arm_target_timeout_sec', 12.0)
        self.declare_parameter('sim_arm_joint_tolerance_rad', 0.075)
        self.declare_parameter(
            'sim_contact_joint_error_tolerance_rad', 0.06)
        self.declare_parameter('sim_arm_clearance_retract_m', 0.22)
        self.declare_parameter('sim_pre_stow_backoff_linear_x', 0.0)
        self.declare_parameter('sim_pre_stow_backoff_sec', 0.0)
        self.declare_parameter('sim_return_home', True)
        self.declare_parameter('door_open_motion', 'push')
        self.declare_parameter('press_handle_before_push', True)
        self.declare_parameter(
            'lever_press_distance_m', LEVER_PRESS_DISTANCE)
        self.declare_parameter('sim_gripper_lever_width_m', GRIPPER_LEVER)
        self.declare_parameter('sim_grasp_search_step_m', 0.02)
        self.declare_parameter('allow_sim_fallback', False)
        self.declare_parameter('sim_physical_door_opening', False)
        self.declare_parameter('sim_door_physics_required', False)
        self.declare_parameter('sim_door_contact_only', False)
        self.declare_parameter('sim_door_contact_min_angle_rad', 2.05)
        self.declare_parameter('sim_door_open_angle_rad', 2.09439510239)
        self.declare_parameter('sim_door_command_steps', 5)
        self.declare_parameter('sim_door_max_match_distance_m', 0.95)
        self.declare_parameter('sim_door_axis_match_max_progress_m', 0.95)
        self.declare_parameter('sim_door_axis_match_max_lateral_m', 0.75)
        self.declare_parameter('sim_door_panel_half_width_m', 0.45)
        self.declare_parameter(
            'sim_door_topic_prefix', '/fire_robot/door/blue')
        self.declare_parameter('sim_door_command_topic', '')
        self.declare_parameter('sim_door_command_sign', 1.0)
        self.declare_parameter('sim_open_away_from_centerline', False)
        self.declare_parameter('sim_door_world_file', '')
        self.declare_parameter('sim_arm_motion_enabled', False)
        self.declare_parameter('sim_arm_topic_prefix', '/fire_robot/sim_arm')
        self.declare_parameter(
            'sim_arm_trajectory_topic',
            '/piper_arm_controller/joint_trajectory')
        self.declare_parameter(
            'sim_gripper_trajectory_topic',
            '/piper_gripper_controller/joint_trajectory')
        self.declare_parameter('piper_mount_x_m', 0.163)
        self.declare_parameter('piper_mount_y_m', 0.15)
        self.declare_parameter('contact_reference_frame', 'odom')
        self.declare_parameter('piper_mount_z_m', 0.509)
        self.declare_parameter('piper_tool_contact_offset_m', 0.10)
        self.declare_parameter('handle_surface_inset_m', 0.025)
        self.declare_parameter('sim_verify_door', False)
        self.declare_parameter('sim_door_feedback_topic', '/door_joint_states')
        self.declare_parameter('sim_door_feedback_joint_name', 'door_hinge')
        self.declare_parameter('sim_lever_feedback_joint_name', 'lever_joint')
        self.declare_parameter('sim_lever_press_min_angle_rad', 0.08)
        self.declare_parameter('sim_lever_press_timeout_sec', 8.0)
        self.declare_parameter('sim_latch_clear_min_angle_rad', 0.10)
        self.declare_parameter('sim_latch_clear_timeout_sec', 6.0)
        self.declare_parameter('sim_gripper_release_timeout_sec', 3.0)
        self.declare_parameter('sim_gripper_release_half_width_m', 0.030)
        self.declare_parameter('sim_handle_refine_timeout_sec', 2.5)
        self.declare_parameter('sim_handle_refine_min_samples', 3)
        self.declare_parameter('sim_handle_refine_max_shift_m', 0.04)
        self.declare_parameter('sim_door_tolerance_rad', 0.08)
        self.declare_parameter('sim_feedback_timeout_sec', 4.0)
        self.declare_parameter('sim_push_follow_enabled', False)
        self.declare_parameter('sim_push_follow_cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('sim_push_follow_linear_x', 0.13)
        self.declare_parameter('sim_push_follow_angular_z', -0.13)
        self.declare_parameter('sim_push_follow_radius_m', 0.60)
        self.declare_parameter('sim_push_follow_stage_sec', 1.7)
        self.declare_parameter('post_open_backoff_enabled', False)
        self.declare_parameter('post_open_backoff_linear_x', -0.05)
        self.declare_parameter('post_open_backoff_sec', 1.2)
        self.declare_parameter('sim_release_handle_before_base_push', False)
        self.declare_parameter('hold_handle_during_base_open', False)
        self.declare_parameter('control_backend', 'moveit')
        self.declare_parameter('manipulation_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 1.0)
        self.declare_parameter('piper_pos_cmd_topic', '/pos_cmd')
        self.declare_parameter('piper_enable_topic', '/enable_flag')
        self.declare_parameter('piper_command_settle_sec', 1.2)
        self.declare_parameter('piper_roll', 0.0)
        self.declare_parameter('piper_pitch', 0.0)
        self.declare_parameter('piper_yaw', 0.0)
        self.declare_parameter('piper_mode1', 0)
        self.declare_parameter('piper_mode2', 0)
        self.declare_parameter('piper_return_home', False)
        self.declare_parameter('piper_home_x', 0.25)
        self.declare_parameter('piper_home_y', 0.0)
        self.declare_parameter('piper_home_z', 0.30)
        self.declare_parameter('require_detected_handle', False)
        self.declare_parameter('require_yolo_handle', False)
        self.declare_parameter('feedback_contact_enabled', False)
        self.declare_parameter('feedback_max_age_sec', 1.0)
        self.declare_parameter('feedback_clearance_min_m', 0.05)
        self.declare_parameter('feedback_clearance_max_m', 0.15)
        self.declare_parameter('feedback_approach_step_m', 0.02)
        self.declare_parameter('feedback_tool_tolerance_m', 0.015)
        self.declare_parameter('feedback_press_step_m', 0.01)
        self.declare_parameter('feedback_press_max_travel_m', 0.11)
        self.declare_parameter('feedback_press_sign', -1.0)
        self.declare_parameter('feedback_press_roll_lead_rad', 0.035)
        self.declare_parameter('feedback_tracking_limit_m', 0.04)
        self.declare_parameter('feedback_joint_bias_limit_rad', 0.12)
        self.declare_parameter('feedback_handle_min_confidence', 0.45)
        self.declare_parameter('feedback_follow_max_age_sec', 1.0)
        self.declare_parameter('feedback_follow_max_shift_m', 0.15)
        self.declare_parameter('feedback_follow_step_m', 0.008)
        self.declare_parameter('feedback_lever_hold_tolerance_rad', 0.01)
        self.declare_parameter('feedback_encoder_contact_follow', False)
        self.declare_parameter('feedback_opening_speed_mps', 0.012)
        self.declare_parameter('feedback_opening_priority_weight', 0.2)

        self._planning_group = self.get_parameter('planning_group').value
        self._gripper_group  = self.get_parameter('gripper_group').value
        self._sim_mode       = self.get_parameter('sim_mode').value
        self._sim_control_enabled = bool(
            self.get_parameter('sim_control_enabled').value)
        self._sim_command_transport = str(
            self.get_parameter('sim_command_transport').value).strip().lower()
        if self._sim_command_transport not in ('ign', 'ros', 'trajectory'):
            self.get_logger().warn(
                f"Unknown sim_command_transport "
                f"'{self._sim_command_transport}', falling back to 'ign'.")
            self._sim_command_transport = 'ign'
        self._sim_step_sec = max(
            0.0, float(self.get_parameter('sim_step_sec').value))
        self._sim_arm_target_timeout_sec = max(
            self._sim_step_sec + 3.0,
            float(self.get_parameter('sim_arm_target_timeout_sec').value))
        self._sim_arm_joint_tolerance = max(
            0.02, float(self.get_parameter(
                'sim_arm_joint_tolerance_rad').value))
        self._sim_contact_joint_error_tolerance = max(
            0.045, float(self.get_parameter(
                'sim_contact_joint_error_tolerance_rad').value))
        self._sim_arm_clearance_retract = max(
            PRE_GRASP_OFFSET, float(self.get_parameter(
                'sim_arm_clearance_retract_m').value))
        self._sim_pre_stow_backoff_linear_x = min(
            0.0, float(self.get_parameter(
                'sim_pre_stow_backoff_linear_x').value))
        self._sim_pre_stow_backoff_sec = max(
            0.0, float(self.get_parameter(
                'sim_pre_stow_backoff_sec').value))
        self._sim_return_home = bool(
            self.get_parameter('sim_return_home').value)
        self._door_open_motion = str(
            self.get_parameter('door_open_motion').value).strip().lower()
        if self._door_open_motion not in ('pull', 'push'):
            self.get_logger().warn(
                f"Unknown door_open_motion '{self._door_open_motion}', "
                "falling back to 'push'.")
            self._door_open_motion = 'push'
        self._press_handle_before_push = bool(
            self.get_parameter('press_handle_before_push').value)
        self._lever_press_distance = max(
            0.0, float(self.get_parameter('lever_press_distance_m').value))
        self._sim_gripper_lever_width = min(
            GRIPPER_OPEN,
            max(0.0, float(
                self.get_parameter('sim_gripper_lever_width_m').value)))
        self._sim_grasp_search_step = max(
            0.0, float(self.get_parameter(
                'sim_grasp_search_step_m').value))
        self._allow_sim_fallback = self.get_parameter('allow_sim_fallback').value
        self._sim_physical_door_opening = bool(
            self.get_parameter('sim_physical_door_opening').value)
        self._sim_door_physics_required = bool(
            self.get_parameter('sim_door_physics_required').value)
        self._sim_door_contact_only = bool(
            self.get_parameter('sim_door_contact_only').value)
        self._sim_door_contact_min_angle = max(
            0.0,
            float(self.get_parameter('sim_door_contact_min_angle_rad').value))
        self._sim_door_open_angle = float(
            self.get_parameter('sim_door_open_angle_rad').value)
        self._sim_door_command_steps = max(
            1, int(self.get_parameter('sim_door_command_steps').value))
        self._sim_door_max_match_distance = max(
            0.0, float(self.get_parameter('sim_door_max_match_distance_m').value))
        self._sim_door_axis_match_max_progress = max(
            0.0,
            float(self.get_parameter('sim_door_axis_match_max_progress_m').value))
        self._sim_door_axis_match_max_lateral = max(
            0.0,
            float(self.get_parameter('sim_door_axis_match_max_lateral_m').value))
        self._sim_door_panel_half_width = max(
            0.0, float(self.get_parameter('sim_door_panel_half_width_m').value))
        self._sim_door_topic_prefix = str(
            self.get_parameter('sim_door_topic_prefix').value).rstrip('/')
        self._sim_door_command_topic = str(
            self.get_parameter('sim_door_command_topic').value).strip()
        self._sim_door_command_sign = float(
            self.get_parameter('sim_door_command_sign').value)
        self._sim_open_away_from_centerline = bool(
            self.get_parameter('sim_open_away_from_centerline').value)
        self._sim_door_world_file = str(
            self.get_parameter('sim_door_world_file').value)
        self._sim_arm_motion_enabled = bool(
            self.get_parameter('sim_arm_motion_enabled').value
        ) or self._sim_control_enabled
        self._sim_arm_topic_prefix = str(
            self.get_parameter('sim_arm_topic_prefix').value).rstrip('/')
        self._sim_arm_trajectory_topic = str(
            self.get_parameter('sim_arm_trajectory_topic').value)
        self._sim_gripper_trajectory_topic = str(
            self.get_parameter('sim_gripper_trajectory_topic').value)
        self._contact_reference_frame = str(self.get_parameter('contact_reference_frame').value)
        self._piper_kinematics = PiperActualKinematics(
            mount_xyz=(
                float(self.get_parameter('piper_mount_x_m').value),
                float(self.get_parameter('piper_mount_y_m').value),
                float(self.get_parameter('piper_mount_z_m').value)),
            tool_contact_offset_m=float(
                self.get_parameter('piper_tool_contact_offset_m').value))
        self._handle_surface_inset = max(
            0.0, float(self.get_parameter('handle_surface_inset_m').value))
        self._sim_verify_door_feedback = bool(
            self.get_parameter('sim_verify_door').value)
        self._sim_door_feedback_topic = str(
            self.get_parameter('sim_door_feedback_topic').value)
        self._sim_door_feedback_joint_name = str(
            self.get_parameter('sim_door_feedback_joint_name').value)
        self._sim_lever_feedback_joint_name = str(
            self.get_parameter('sim_lever_feedback_joint_name').value)
        self._sim_lever_press_min_angle = max(
            0.0, float(
                self.get_parameter('sim_lever_press_min_angle_rad').value))
        self._sim_lever_press_timeout = max(
            0.1, float(
                self.get_parameter('sim_lever_press_timeout_sec').value))
        self._sim_latch_clear_min_angle = max(
            0.02, float(self.get_parameter(
                'sim_latch_clear_min_angle_rad').value))
        self._sim_latch_clear_timeout = max(
            0.5, float(self.get_parameter(
                'sim_latch_clear_timeout_sec').value))
        self._sim_gripper_release_timeout = max(
            0.5, float(self.get_parameter(
                'sim_gripper_release_timeout_sec').value))
        self._sim_gripper_release_half_width = self._clamp(
            float(self.get_parameter(
                'sim_gripper_release_half_width_m').value),
            0.015, 0.034)
        self._sim_handle_refine_timeout = max(
            0.5, float(self.get_parameter(
                'sim_handle_refine_timeout_sec').value))
        self._sim_handle_refine_min_samples = max(
            1, int(self.get_parameter(
                'sim_handle_refine_min_samples').value))
        self._sim_handle_refine_max_shift = max(
            0.0, float(self.get_parameter(
                'sim_handle_refine_max_shift_m').value))
        self._sim_door_tolerance = max(
            0.0, float(self.get_parameter('sim_door_tolerance_rad').value))
        self._sim_feedback_timeout = max(
            0.0, float(self.get_parameter('sim_feedback_timeout_sec').value))
        self._sim_push_follow_enabled = bool(
            self.get_parameter('sim_push_follow_enabled').value)
        self._sim_push_follow_cmd_vel_topic = str(
            self.get_parameter('sim_push_follow_cmd_vel_topic').value)
        self._sim_push_follow_linear_x = float(
            self.get_parameter('sim_push_follow_linear_x').value)
        self._sim_push_follow_angular_z = float(
            self.get_parameter('sim_push_follow_angular_z').value)
        self._sim_push_follow_radius = max(
            0.35, float(self.get_parameter(
                'sim_push_follow_radius_m').value))
        self._sim_push_follow_stage_sec = max(
            0.1, float(self.get_parameter('sim_push_follow_stage_sec').value))
        self._post_open_backoff_enabled = bool(
            self.get_parameter('post_open_backoff_enabled').value)
        self._post_open_backoff_linear_x = min(
            0.0,
            max(-0.15, float(
                self.get_parameter('post_open_backoff_linear_x').value)))
        self._post_open_backoff_sec = max(
            0.0, float(self.get_parameter('post_open_backoff_sec').value))
        self._sim_release_handle_before_base_push = bool(
            self.get_parameter('sim_release_handle_before_base_push').value)
        self._hold_handle_during_base_open = bool(
            self.get_parameter('hold_handle_during_base_open').value)
        self._control_backend = str(
            self.get_parameter('control_backend').value).strip().lower()
        self._manipulation_frame = self.get_parameter('manipulation_frame').value
        self._tf_timeout_sec = self.get_parameter('tf_timeout_sec').value
        self._piper_pos_cmd_topic = self.get_parameter('piper_pos_cmd_topic').value
        self._piper_enable_topic = self.get_parameter('piper_enable_topic').value
        self._piper_settle_sec = self.get_parameter(
            'piper_command_settle_sec').value
        self._piper_roll = self.get_parameter('piper_roll').value
        self._piper_pitch = self.get_parameter('piper_pitch').value
        self._piper_yaw = self.get_parameter('piper_yaw').value
        self._piper_mode1 = self.get_parameter('piper_mode1').value
        self._piper_mode2 = self.get_parameter('piper_mode2').value
        self._piper_return_home = self.get_parameter('piper_return_home').value
        self._piper_home_x = self.get_parameter('piper_home_x').value
        self._piper_home_y = self.get_parameter('piper_home_y').value
        self._piper_home_z = self.get_parameter('piper_home_z').value
        self._require_detected_handle = bool(
            self.get_parameter('require_detected_handle').value)
        self._require_yolo_handle = bool(
            self.get_parameter('require_yolo_handle').value)
        self._feedback_contact = bool(
            self.get_parameter('feedback_contact_enabled').value)
        self._feedback_encoder_contact_follow = bool(
            self.get_parameter('feedback_encoder_contact_follow').value)
        self._feedback_config = {
            name: float(self.get_parameter('feedback_' + name).value)
            for name in (
                'max_age_sec', 'clearance_min_m', 'clearance_max_m',
                'approach_step_m', 'tool_tolerance_m', 'press_step_m',
                'press_max_travel_m', 'press_sign', 'press_roll_lead_rad', 'tracking_limit_m',
                'handle_min_confidence', 'joint_bias_limit_rad',
                'follow_max_age_sec', 'follow_max_shift_m', 'follow_step_m',
                'lever_hold_tolerance_rad', 'opening_speed_mps', 'opening_priority_weight')}
        if not math.isfinite(self._feedback_config['opening_speed_mps']) or not (
                0. < self._feedback_config['opening_speed_mps'] <= .06):
            raise ValueError('Opening probe speed exceeds bounded contact velocity')
        if (not math.isfinite(self._feedback_config['opening_priority_weight'])
                or self._feedback_config['opening_priority_weight'] <= 0.):
            raise ValueError('Opening priority weight must be positive and finite')
        if self._feedback_contact and not (
                self._sim_mode and self._sim_door_contact_only
                and self._sim_verify_door_feedback
                and self._require_yolo_handle):
            raise ValueError(
                'Feedback contact currently requires the instrumented Gazebo '
                'fixture, lever feedback and live YOLO. No real-sensor adapter '
                'or full-world contact model has been validated yet.')
        self._feedback_arm_stamp = None
        self._feedback_lever_stamp = None
        self._feedback_door_stamp = None
        self._feedback_lever_samples = []
        self._feedback_joint_msg_stamp = None
        self._feedback_lever_msg_stamp = None
        self._feedback_door_msg_stamp = None
        self._feedback_handle_times = {}
        self._feedback_handle_points = []
        self._feedback_contact_observation = None
        self._feedback_contact_image_stamp = None
        self._feedback_follow_reference = None
        self._feedback_contact_orientation_reference = None
        self._feedback_released_rotation = None
        self._feedback_full_handle_follow = False
        self._feedback_follow_mode = 'vision'
        self._feedback_failure = ''
        self._feedback_press_verified = False
        self._feedback_joint_bias = np.zeros(6)

        if self._control_backend not in ('moveit', 'piper_sdk'):
            self.get_logger().warn(
                f"Unknown control_backend '{self._control_backend}', "
                "falling back to 'moveit'.")
            self._control_backend = 'moveit'

        cb_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # 문 개방 서비스 서버
        self.open_door_srv = self.create_service(
            OpenDoor, '/open_door', self.open_door_callback,
            callback_group=cb_group)

        # 조작 완료 신호 발행 (옵션 모니터링용)
        self.manip_done_pub = self.create_publisher(Bool, '/manipulation_done', 10)
        self.manip_phase_pub = self.create_publisher(
            String, '/manipulation_phase', 10)
        self._feedback_pub = self.create_publisher(
            String, '/manipulation_feedback', 20)
        self._sim_cmd_vel_pub = self.create_publisher(
            Twist, self._sim_push_follow_cmd_vel_topic, 10)
        self._sim_arm_trajectory_pub = self.create_publisher(
            JointTrajectory, self._sim_arm_trajectory_topic, 10)
        self._sim_gripper_trajectory_pub = self.create_publisher(
            JointTrajectory, self._sim_gripper_trajectory_topic, 10)
        self._sim_ros_float_pubs: dict[str, object] = {}
        self._sim_arm_joint_positions: dict[str, float] = {}
        self._sim_last_arm_target = PIPER_STOW.copy()
        self._sim_door_position: float | None = None
        self._sim_door_initial_position: float | None = None
        self._sim_lever_position: float | None = None
        self._sim_lever_initial_position: float | None = None
        self._sim_handle_prepressed = False
        self._sim_push_hold_targets: dict[str, float] | None = None
        self._detected_handle_seq = 0
        self._detected_handle_samples: list[
            tuple[int, PointStamped, float, str]] = []
        self.create_subscription(
            DoorInfo, '/detected_door', self._detected_door_callback, 20)
        self._sim_door_state_subscription = None
        if self._sim_mode and self._sim_verify_door_feedback:
            self._sim_door_state_subscription = self.create_subscription(
                JointState,
                self._sim_door_feedback_topic,
                self._sim_door_state_callback,
                10)
        if self._sim_mode and self._sim_arm_motion_enabled:
            self.create_subscription(
                JointState, '/joint_states', self._sim_arm_state_callback, 20)
        self._sim_home_attempts_remaining = 0
        self._sim_initial_home_timer = None
        if (
                self._sim_mode
                and self._sim_arm_motion_enabled
                and self._sim_command_transport in ('ign', 'ros')):
            self._sim_home_attempts_remaining = 3
            self._sim_initial_home_timer = self.create_timer(
                0.75, self._initialize_sim_arm_home)

        # MoveIt2 초기화
        self._arm    = None
        self._gripper = None
        self._moveit  = None
        self._moveit_ready = False
        self._piper_ready = False
        self._piper_pos_pub = None
        self._piper_enable_pub = None
        self._sim_hinged_doors = self._load_sim_hinged_doors()
        self._sim_opened_door_topics: set[str] = set()
        self._last_sim_open_was_idempotent = False
        self._last_sim_open_topic = ''
        self._last_sim_open_xy: tuple[float, float] | None = None

        if not self._sim_mode and self._control_backend == 'piper_sdk':
            if not _HAS_PIPER_MSGS:
                self.get_logger().error(
                    'piper_msgs is not installed; cannot use piper_sdk '
                    'control_backend.')
            else:
                self._piper_pos_pub = self.create_publisher(
                    PosCmd, self._piper_pos_cmd_topic, 10)
                self._piper_enable_pub = self.create_publisher(
                    Bool, self._piper_enable_topic, 10)
                self._piper_ready = True
                self.get_logger().info(
                    'PIPER SDK command backend ready: '
                    f'{self._piper_pos_cmd_topic}, '
                    f'{self._piper_enable_topic}')

        if not self._sim_mode and self._control_backend == 'moveit':
            if not _HAS_MOVEIT:
                self._handle_moveit_init_failure('moveit_py is not installed')
            else:
                try:
                    self._moveit  = MoveItPy(node_name='manipulation_node')
                    self._arm     = self._moveit.get_planning_component(
                        self._planning_group)
                    self._gripper = self._moveit.get_planning_component(
                        self._gripper_group)
                    self._moveit_ready = True
                    self.get_logger().info('MoveItPy initialized.')
                except Exception as e:
                    self._handle_moveit_init_failure(str(e))

        mode_str = 'SIMULATION' if self._sim_mode else 'REAL ROBOT'
        self.get_logger().info(
            f'ManipulationNode started [{mode_str}, '
            f'backend={self._control_backend}]')
        if self._sim_mode and self._sim_arm_motion_enabled:
            self.get_logger().info(
                'Gazebo arm simulation enabled '
                f'(transport={self._sim_command_transport}, '
                f'prefix={self._sim_arm_topic_prefix}).')
        if self._sim_mode and self._sim_verify_door_feedback:
            self.get_logger().info(
                'Gazebo door feedback verification enabled '
                f'({self._sim_door_feedback_topic}/'
                f'{self._sim_door_feedback_joint_name}).')

    def _initialize_sim_arm_home(self):
        """Keep PIPER folded so the front camera can observe the lever."""
        if self._sim_home_attempts_remaining <= 0:
            if self._sim_initial_home_timer is not None:
                self._sim_initial_home_timer.cancel()
            return
        self._publish_sim_arm_targets(dict(zip(
            PIPER_JOINT_NAMES, PIPER_STOW.tolist())))
        self._command_sim_gripper(GRIPPER_OPEN)
        self._sim_home_attempts_remaining -= 1
        if self._sim_home_attempts_remaining <= 0:
            self._sim_initial_home_timer.cancel()
            self.get_logger().info(
                'Gazebo PIPER startup camera-clear stow initialized.')

    # ── 서비스 핸들러 ─────────────────────────────────────
    def open_door_callback(self,
                           request: OpenDoor.Request,
                           response: OpenDoor.Response):
        handle_method = str(
            getattr(request, 'handle_detection_method', '') or 'estimated')
        handle_detected = bool(getattr(request, 'handle_detected', False))
        handle_confidence = float(getattr(request, 'handle_confidence', 0.0))
        self.get_logger().info(
            f'Open door request: id={request.door_id}, '
            f'handle_method={handle_method}, '
            f'handle_detected={handle_detected}, '
            f'handle_conf={handle_confidence:.2f}')
        self._last_sim_open_was_idempotent = False
        self._last_sim_open_topic = ''
        self._last_sim_open_xy = None
        self._sim_push_hold_targets = None
        self._feedback_failure = ''
        self._feedback_press_verified = False
        self._feedback_joint_bias = np.zeros(6)

        if self._require_detected_handle and not handle_detected:
            response.success = False
            response.message = 'Detected handle is required before opening'
            self.get_logger().error(response.message)
            done_msg = Bool()
            done_msg.data = False
            self.manip_done_pub.publish(done_msg)
            return response
        if ((self._require_yolo_handle and not handle_method.startswith('yolo')
             and not (self._feedback_contact and handle_method.startswith('track:primary_yolo')))
                or (self._feedback_contact
                    and not handle_method.endswith(':registered_depth'))):
            response.success = False
            response.message = 'YOLO handle with required depth source is needed before opening'
            self.get_logger().error(response.message)
            done_msg = Bool()
            done_msg.data = False
            self.manip_done_pub.publish(done_msg)
            return response

        success = self._execute_door_open_sequence(
            request.handle_position, request.door_id)
        if not success and self._feedback_contact:
            self._feedback_stop(self._feedback_failure or 'Sequence failed')

        response.success = success
        if success and self._last_sim_open_was_idempotent:
            response.message = 'All hinged blue doors already open'
        elif success and self._last_sim_open_topic and self._last_sim_open_xy is not None:
            sx, sy = self._last_sim_open_xy
            response.message = (
                'Door opened successfully '
                f'sim_topic={self._last_sim_open_topic} '
                f'sim_xy={sx:.3f},{sy:.3f}')
        else:
            response.message = ('Door opened successfully'
                                if success else 'Failed to open door: '
                                + self._feedback_failure)

        done_msg = Bool()
        done_msg.data = success
        self.manip_done_pub.publish(done_msg)
        return response

    # ── 문 개방 시퀀스 ────────────────────────────────────
    def _execute_door_open_sequence(self,
                                    handle_position: PointStamped,
                                    door_id: str = '') -> bool:
        self._feedback_contact_orientation_reference = None
        self._feedback_released_rotation = None
        self._feedback_full_handle_follow = False
        held_open = getattr(self, '_hold_handle_during_base_open', False)
        if held_open and (not self._sim_mode or not self._feedback_contact
                          or self._sim_release_handle_before_base_push):
            return self._feedback_stop('Handle-held opening requires instrumented feedback and no early release')
        self._set_manip_phase('LOCALIZE_HANDLE', door_id)
        if not self._sim_mode:
            if self._control_backend == 'moveit' and not self._moveit_ready:
                self.get_logger().error(
                    'MoveItPy is not ready in real robot mode; '
                    'refusing to report simulated success.')
                return False
            if self._control_backend == 'piper_sdk' and not self._piper_ready:
                self.get_logger().error(
                    'PIPER SDK command backend is not ready; '
                    'cannot open the door safely.')
                return False

        door_frame_handle_position = handle_position
        arm_handle_position = self._transform_handle_to_manipulation_frame(
            handle_position)
        if arm_handle_position is None:
            self._set_manip_phase('FAILED_LOCALIZE_HANDLE', door_id)
            return False
        if self._sim_mode and self._sim_door_contact_only:
            if self._feedback_contact and not self._feedback_fresh():
                return self._feedback_stop('Missing initial joint/lever feedback')
            self._sim_door_initial_position = self._sim_door_position
            self._sim_lever_initial_position = self._sim_lever_position
            self._sim_handle_prepressed = False

        self._set_manip_phase('PRE_GRASP', door_id)
        self.get_logger().info('Step 1/4: Moving to pre-grasp position')
        if not self._move_to_pre_grasp(arm_handle_position):
            self.get_logger().error('Pre-grasp failed')
            self._set_manip_phase('FAILED_PRE_GRASP', door_id)
            return False

        if (self._sim_mode and self._sim_door_contact_only
                and not self._feedback_contact):
            refined = self._refine_handle_after_pre_grasp(
                arm_handle_position)
            if refined is not None:
                arm_handle_position = refined
                door_frame_handle_position = refined
            # Contact acceptance must come from lever motion caused by the
            # grasp itself, not from a small disturbance during pre-grasp.
            self._sim_lever_initial_position = self._sim_lever_position

        self._set_manip_phase('GRASP_HANDLE', door_id)
        self.get_logger().info('Step 2/4: Grasping handle')
        if not self._grasp_handle(arm_handle_position):
            self.get_logger().error('Grasp failed')
            self._set_manip_phase('FAILED_GRASP_HANDLE', door_id)
            return False

        do_lever_press = (
            self._door_open_motion == 'push'
            and self._press_handle_before_push
            and (self._feedback_contact or self._lever_press_distance > 0.0))
        if do_lever_press:
            self._set_manip_phase('PRESS_HANDLE', door_id)
            self.get_logger().info('Step 3/5: Pressing lever handle')
            if not self._press_handle(arm_handle_position):
                self.get_logger().error('Lever press failed')
                self._set_manip_phase('FAILED_PRESS_HANDLE', door_id)
                return False

            if held_open:
                self._set_manip_phase('LATCH_CLEAR_PUSH', door_id)
                if not self._run_sim_latch_clear_push():
                    return False
                self._set_manip_phase('LATCH_RELEASED', door_id)
                self._feedback_event('handle_retained_after_latch')
            elif self._sim_mode and self._sim_release_handle_before_base_push:
                if self._sim_door_contact_only:
                    self._set_manip_phase('LATCH_CLEAR_PUSH', door_id)
                    self.get_logger().info(
                        'Keeping the lever depressed while the chassis '
                        'pushes the panel clear of the latch')
                    if not self._run_sim_latch_clear_push():
                        self.get_logger().error(
                            'Door did not leave the latch while the physical '
                            'lever contact was held')
                        self._set_manip_phase(
                            'FAILED_LATCH_CLEAR_PUSH', door_id)
                        return False
                self._set_manip_phase('LATCH_RELEASED', door_id)
                self.get_logger().info(
                    'Lever press verified; opening the gripper before the '
                    'chassis push')
                self._command_sim_gripper(GRIPPER_OPEN)
                if not self._wait_for_sim_gripper_release():
                    self.get_logger().error(
                        'The physical gripper did not clear the lever')
                    self._set_manip_phase(
                        'FAILED_RELEASE_HANDLE', door_id)
                    return False
                self._set_manip_phase('RETRACT_FROM_HANDLE', door_id)
                if self._sim_pre_stow_backoff_sec > 0.0:
                    self.get_logger().info(
                        'Retracting the released gripper with a short '
                        'low-speed chassis backoff before folding')
                    self._run_sim_base_motion(
                        self._sim_pre_stow_backoff_linear_x,
                        0.0,
                        self._sim_pre_stow_backoff_sec)
                else:
                    self.get_logger().info(
                        'Retracting along the handle approach axis before '
                        'folding')
                    if not self._command_sim_arm_pose(
                            'clearance_retract', arm_handle_position):
                        return False
                    if not self._wait_for_sim_arm_target(
                            tolerance_rad=0.060):
                        return False
                self._set_manip_phase('ARM_CLEAR_FOR_BASE_PUSH', door_id)
                self.get_logger().info(
                    'Folding the arm into its chassis-clear stow pose; the chassis will '
                    'provide all remaining opening force')
                self._sim_push_hold_targets = None
                if not self._move_sim_arm_to_stow():
                    return False

        if held_open:
            open_phase = 'HANDLE_HELD_BASE_OPEN'
        elif self._door_open_motion == 'pull':
            open_phase = 'PULL_OPEN'
        elif (
                self._sim_mode
                and self._sim_door_contact_only
                and self._sim_push_follow_enabled):
            open_phase = 'BASE_PUSH_OPEN'
        else:
            open_phase = 'PUSH_OPEN'
        self._set_manip_phase(open_phase, door_id)
        self.get_logger().info(
            f"Step {'4/5' if do_lever_press else '3/4'}: "
            f'{self._door_open_motion.title()}ing door open')
        if not self._open_door_motion(
                arm_handle_position, door_id, door_frame_handle_position):
            self.get_logger().error(
                f'{self._door_open_motion.title()} failed')
            self._set_manip_phase(f'FAILED_{open_phase}', door_id)
            return False

        if held_open:
            self._set_manip_phase('RETURN_LEVER', door_id)
            if not self._feedback_restore_lever_before_release():
                return False
            self._set_manip_phase('RELEASE_HANDLE', door_id)
            grip_span = (abs(self._sim_arm_joint_positions['gripper_left_joint'])
                         + abs(self._sim_arm_joint_positions['gripper_right_joint']))
            self._command_sim_gripper(GRIPPER_OPEN)
            if not self._wait_for_sim_gripper_release():
                return self._feedback_stop('Handle did not release after full opening')
            self._feedback_event('handle_released_after_full_open',
                                 door_delta_rad=abs(self._sim_door_position-self._sim_door_initial_position))
            self._set_manip_phase('RETRACT_FROM_HANDLE', door_id)
            if not self._feedback_retract_released_handle(grip_span):
                return False
            if not self._move_sim_arm_to_stow():
                return False
        elif self._sim_mode and self._sim_release_handle_before_base_push:
            self._set_manip_phase('RELEASE_HANDLE', door_id)
            self.get_logger().info(
                'Releasing the lever before arm recovery and base backoff')
            self._command_sim_gripper(GRIPPER_OPEN)

        self._set_manip_phase('RETURN_HOME', door_id)
        self.get_logger().info(
            f"Step {'5/5' if do_lever_press else '4/4'}: "
            'Returning to home position')
        home_ok = self._move_to_home()
        if self._feedback_contact and not home_ok:
            self._set_manip_phase('FAILED_RETURN_HOME', door_id)
            return False
        if self._post_open_backoff_enabled:
            self._set_manip_phase('POST_OPEN_BACKOFF', door_id)
            self.get_logger().info(
                'Backing away from opened door with limited micro reverse')
            self._post_open_backoff()
        self._set_manip_phase('COMPLETE', door_id)
        return True

    # ── 동작 단계 구현 ────────────────────────────────────
    def _handle_moveit_init_failure(self, reason: str):
        if self._allow_sim_fallback:
            self.get_logger().warn(
                f'MoveItPy init failed ({reason}). Falling back to sim mode.')
            self._sim_mode = True
            return

        self.get_logger().error(
            f'MoveItPy init failed ({reason}). Door opening requests will '
            'fail until MoveIt is available.')

    def _set_manip_phase(self, phase: str, door_id: str = ''):
        msg = String()
        msg.data = f'{phase}:{door_id}' if door_id else phase
        self.manip_phase_pub.publish(msg)
        self.get_logger().info(f'Manipulation phase: {msg.data}')

    def _transform_handle_to_manipulation_frame(
            self, handle_position: PointStamped):
        source_frame = handle_position.header.frame_id.strip()
        if not source_frame:
            msg = ('handle_position has no frame_id; cannot command the '
                   'arm safely.')
            if self._sim_mode and not self._feedback_contact:
                self.get_logger().warn(
                    msg + ' Using the raw sim coordinate as a fallback.')
                return handle_position
            self.get_logger().error(msg)
            return None

        if source_frame == self._manipulation_frame:
            return handle_position

        try:
            transform = self._tf_buffer.lookup_transform(
                self._manipulation_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self._tf_timeout_sec))
            transformed = do_transform_point(handle_position, transform)
        except TransformException as e:
            if self._sim_mode and not self._feedback_contact:
                self.get_logger().warn(
                    f'Cannot transform sim handle_position from '
                    f'{source_frame} to {self._manipulation_frame}: {e}. '
                    'Using the raw coordinate for arm visualization.')
                return handle_position
            self.get_logger().error(
                f'Cannot transform handle_position from {source_frame} to '
                f'{self._manipulation_frame}: {e}')
            return None
        except Exception as e:
            if self._sim_mode and not self._feedback_contact:
                self.get_logger().warn(
                    f'Failed to transform sim handle_position: {e}. '
                    'Using the raw coordinate for arm visualization.')
                return handle_position
            self.get_logger().error(
                f'Failed to transform handle_position: {e}')
            return None

        self.get_logger().info(
            f'Transformed handle_position {source_frame} -> '
            f'{self._manipulation_frame}: '
            f'({transformed.point.x:.3f}, '
            f'{transformed.point.y:.3f}, '
            f'{transformed.point.z:.3f})')
        return transformed

    def _move_to_pre_grasp(self, handle_pos: PointStamped) -> bool:
        if self._feedback_contact:
            refined = self._refine_handle_after_pre_grasp(handle_pos)
            if refined is None:
                return self._feedback_stop('No consistent live YOLO handle')
            try:
                tool_protrusion = max(0.0, self._piper_kinematics.tool_front_extent_m
                                      - self._piper_kinematics.tool_contact_offset_m)
                safety_floor = max(
                    self._feedback_config['clearance_min_m'],
                    tool_protrusion + self._feedback_config['tool_tolerance_m']
                    + self._feedback_config['approach_step_m'])
                center, clearance, spread = observed_clearance(
                    self._feedback_handle_points,
                    safety_floor,
                    self._feedback_config['clearance_max_m'])
            except ValueError as exc:
                return self._feedback_stop(str(exc))
            handle_pos.point = refined.point
            try:
                self._feedback_handle_anchor = self._tf_buffer.transform(
                    refined, self._contact_reference_frame, timeout=Duration(seconds=.2))
            except Exception:
                return self._feedback_stop('Cannot anchor observed handle in odom')
            goal = center.copy()
            goal[0] -= clearance
            self._feedback_preferred_contact_posture = goal.copy()
            self._feedback_event('approach_clearance', clearance_m=clearance,
                                 tool_geometry_safety_floor_m=safety_floor,
                                 observation_spread_m=spread,
                                 handle_xyz=center.tolist())
            self._command_sim_gripper(GRIPPER_OPEN)
            start = self._feedback_tool()
            if start is None:
                return self._feedback_stop('Missing feedback on pre-grasp path')
            return self._feedback_joint_pregrasp(goal, center[0])
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Pre-grasp at '
                f'({handle_pos.point.x - PRE_GRASP_OFFSET:.3f}, '
                f'{handle_pos.point.y:.3f}, '
                f'{handle_pos.point.z:.3f})')
            self._command_sim_gripper(GRIPPER_OPEN)
            if not self._command_sim_arm_pose('pre_grasp', handle_pos):
                return False
            return self._wait_for_sim_arm_target()

        target = self._make_pose(
            x=handle_pos.point.x - PRE_GRASP_OFFSET,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_OPEN)
        return self._plan_and_execute_cartesian(target)

    def _grasp_handle(self, handle_pos: PointStamped) -> bool:
        if self._feedback_contact:
            return self._feedback_grasp(handle_pos)
        if self._sim_mode:
            self.get_logger().info('  [SIM] Gripper closing')
            close_width = (
                self._sim_gripper_lever_width
                if self._sim_door_contact_only
                else GRIPPER_CLOSE)
            original_x = float(handle_pos.point.x)
            original_y = float(handle_pos.point.y)
            original_z = float(handle_pos.point.z)
            corrections = ((0.0, 0.0, 0.0),)
            if self._sim_door_contact_only:
                step = self._sim_grasp_search_step
                corrections = (
                    (0.0, 0.0, 0.0),
                    (step, 0.0, 0.0),
                    (2.0 * step, 0.0, 0.0),
                    (-step, 0.0, 0.0),
                    (step, step, 0.0),
                    (step, -step, 0.0),
                    (0.0, step, 0.0),
                    (0.0, -step, 0.0),
                    (2.0 * step, step, 0.0),
                    (2.0 * step, -step, 0.0),
                    (0.0, 0.0, step),
                    (0.0, 0.0, -step),
                )

            for attempt, (
                    depth_correction,
                    lateral_correction,
                    height_correction) in enumerate(
                    corrections, start=1):
                if attempt > 1:
                    self._command_sim_gripper(GRIPPER_OPEN)
                    if not self._wait_for_sim_duration(0.30):
                        return False
                handle_pos.point.x = original_x + depth_correction
                handle_pos.point.y = original_y + lateral_correction
                handle_pos.point.z = original_z + height_correction
                if not self._command_sim_arm_pose('grasp', handle_pos):
                    return False
                if not self._wait_for_sim_arm_target(
                        # Open fingers should reach the observed lever centre
                        # accurately before closure. If they touch early, the
                        # lever-motion fallback below still accepts contact
                        # using the separate contact tolerance.
                        tolerance_rad=(
                            0.035 if self._sim_door_contact_only else 0.045),
                        contact_lever_delta_rad=(
                            0.025 if self._sim_door_contact_only else None)):
                    continue
                self._command_sim_gripper(close_width)
                if not self._wait_for_sim_duration(
                        1.10 if self._sim_door_contact_only
                        else max(0.55, self._sim_step_sec)):
                    return False

                left = self._sim_arm_joint_positions.get(
                    'gripper_left_joint')
                right = self._sim_arm_joint_positions.get(
                    'gripper_right_joint')
                measured_width = (
                    left - right
                    if left is not None and right is not None else None)
                if measured_width is not None:
                    self.get_logger().info(
                        '  [PIPER SIM] Gripper closure: '
                        f'attempt={attempt}, '
                        f'depth_correction={depth_correction:+.3f}m, '
                        f'lateral_correction={lateral_correction:+.3f}m, '
                        f'height_correction={height_correction:+.3f}m, '
                        f'command={close_width:.3f}m, '
                        f'measured={measured_width:.3f}m')

                if not self._sim_door_contact_only:
                    return True
                if (measured_width is not None
                        and measured_width >= close_width + 0.008):
                    self.get_logger().info(
                        '  [PIPER SIM] Gripper contact candidate found; '
                        'verifying it with a short physical lever probe')
                else:
                    self.get_logger().info(
                        '  [PIPER SIM] No reliable finger residual; '
                        'checking direct lever motion before rejecting it')
                if self._probe_sim_lever_contact(handle_pos):
                    return True
                self.get_logger().warn(
                    '  [PIPER SIM] Contact did not rotate the lever; '
                    'rejecting panel or gripper-body contact')
                self.get_logger().warn(
                    '  [PIPER SIM] Empty grasp detected; retrying around '
                    'the observed handle depth')

            handle_pos.point.x = original_x
            handle_pos.point.y = original_y
            handle_pos.point.z = original_z
            self.get_logger().error(
                '  [PIPER SIM] No physical handle contact after local search')
            return False

        # 손잡이 위치로 직선 이동
        target = self._make_pose(
            x=handle_pos.point.x + self._handle_surface_inset,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )

        if self._using_piper_sdk():
            if not self._send_piper_pose(target, GRIPPER_OPEN):
                return False
            return self._send_piper_pose(target, GRIPPER_CLOSE)

        # 그리퍼 열기
        self._set_gripper(GRIPPER_OPEN)
        if not self._plan_and_execute_cartesian(target):
            return False

        # 그리퍼 닫기
        return self._set_gripper(GRIPPER_CLOSE)

    def _probe_sim_lever_contact(self, handle_pos: PointStamped) -> bool:
        """Confirm a grasp by moving the physical lever, not by finger stall."""
        initial = self._sim_lever_initial_position
        if initial is None:
            initial = self._sim_lever_position
        if initial is None:
            return False

        start_z = float(handle_pos.point.z)
        probe_depth = min(0.10, self._lever_press_distance)
        best_delta = 0.0
        for step in range(1, 6):
            probe = PointStamped()
            probe.header = handle_pos.header
            probe.point.x = float(handle_pos.point.x)
            probe.point.y = float(handle_pos.point.y)
            probe.point.z = start_z - probe_depth * step / 5.0
            targets = self._sim_arm_targets_for_stage('press', probe)
            if targets is None:
                return False
            if not self._command_sim_arm_targets(
                    f'grasp_contact_probe_{step}_of_5', targets):
                return False
            if not self._wait_for_sim_duration(0.55):
                return False

            current = self._sim_lever_position
            lever_delta = (
                abs(current - initial) if current is not None else 0.0)
            best_delta = max(best_delta, lever_delta)
            if lever_delta >= self._sim_lever_press_min_angle:
                handle_pos.point.z = float(probe.point.z)
                self._sim_handle_prepressed = True
                self.get_logger().info(
                    '  [PIPER SIM] Physical handle grasp and press verified: '
                    f'lever_delta={lever_delta:.3f}rad, '
                    f'contact_z={handle_pos.point.z:.3f}m')
                return True

        self.get_logger().info(
            '  [PIPER SIM] Lever probe below threshold: '
            f'best_delta={best_delta:.3f}rad, '
            f'required={self._sim_lever_press_min_angle:.3f}rad')
        return False

    def _pressed_handle_position(
            self, handle_pos: PointStamped) -> PointStamped:
        pressed = PointStamped()
        pressed.header = handle_pos.header
        pressed.point.x = handle_pos.point.x
        pressed.point.y = handle_pos.point.y
        pressed.point.z = handle_pos.point.z - self._lever_press_distance
        return pressed

    def _press_handle(self, handle_pos: PointStamped) -> bool:
        if self._feedback_contact:
            return self._feedback_press(handle_pos)
        pressed = self._pressed_handle_position(handle_pos)
        if self._sim_mode:
            if self._sim_door_contact_only and self._sim_handle_prepressed:
                self.get_logger().info(
                    '  [PIPER SIM] Lever press already verified during '
                    'closed-loop contact acquisition')
                return True
            self.get_logger().info(
                f'  [SIM] Pressing lever down '
                f'{self._lever_press_distance:.3f}m before push')
            # Contact can prevent exact convergence while the controller keeps
            # applying force. Advance through Cartesian waypoints so the real
            # PIPER chain does not cut diagonally away from the grasped lever.
            if self._sim_door_contact_only and self._sim_verify_door_feedback:
                initial_z = float(handle_pos.point.z)
                for step in range(1, 7):
                    waypoint = PointStamped()
                    waypoint.header = handle_pos.header
                    waypoint.point.x = float(handle_pos.point.x)
                    waypoint.point.y = float(handle_pos.point.y)
                    waypoint.point.z = (
                        initial_z
                        - self._lever_press_distance * step / 6.0)
                    targets = self._sim_arm_targets_for_stage(
                        'press', waypoint)
                    if targets is None:
                        return False
                    self._sim_push_hold_targets = targets
                    if not self._command_sim_arm_targets(
                            f'press_{step}_of_6', targets):
                        return False
                    if not self._wait_for_sim_duration(0.32):
                        return False
                    initial = self._sim_lever_initial_position
                    current = self._sim_lever_position
                    if (initial is not None and current is not None
                            and abs(current - initial)
                            >= self._sim_lever_press_min_angle):
                        self.get_logger().info(
                            '  [PIPER SIM] Incremental physical lever press '
                            f'verified at waypoint {step}/6: '
                            f'delta={abs(current - initial):.3f}rad')
                        return True
                return self._wait_for_sim_lever_contact_motion(
                    self._sim_lever_press_min_angle)
            self._sim_push_hold_targets = self._sim_arm_targets_for_stage(
                'press', pressed)
            if not self._command_sim_arm_pose('press', pressed):
                return False
            return self._wait_for_sim_duration(self._sim_step_sec)

        target = self._make_pose(
            x=pressed.point.x + self._handle_surface_inset,
            y=pressed.point.y,
            z=pressed.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_CLOSE)
        return self._plan_and_execute_cartesian(target)

    def _open_door_motion(self, handle_pos: PointStamped,
                          door_id: str = '',
                          door_frame_handle_pos: PointStamped | None = None
                          ) -> bool:
        if self._door_open_motion == 'push':
            return self._push_open_door(
                handle_pos, door_id, door_frame_handle_pos)
        return self._pull_open_door(
            handle_pos, door_id, door_frame_handle_pos)

    def _pull_open_door(self, handle_pos: PointStamped,
                        door_id: str = '',
                        door_frame_handle_pos: PointStamped | None = None
                        ) -> bool:
        """손잡이를 잡은 상태에서 문을 로봇 후방으로 당겨 연다."""
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Pulling handle and opening hinged Gazebo door '
                f'{self._sim_door_open_angle:.2f}rad')
            if (
                    self._sim_door_contact_only
                    and self._sim_door_initial_position is None):
                self._sim_door_initial_position = self._sim_door_position
            self._command_sim_arm_pose('pull', handle_pos)
            door_match_handle = door_frame_handle_pos or handle_pos
            if not self._open_sim_gazebo_door(door_match_handle, door_id):
                return False
            return self._wait_for_sim_duration(self._sim_step_sec)

        # The manipulation frame uses x forward and z up. Pulling moves the
        # handle back along -x toward the robot.
        target = self._make_pose(
            x=(handle_pos.point.x + self._handle_surface_inset
               - PULL_DISTANCE),
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_CLOSE)
        return self._plan_and_execute_cartesian(target)

    def _push_open_door(self, handle_pos: PointStamped,
                        door_id: str = '',
                        door_frame_handle_pos: PointStamped | None = None
                        ) -> bool:
        """손잡이를 잡은 상태에서 문을 로봇 전방으로 밀어 연다."""
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Holding the pressed handle while the base opens '
                f'the hinged Gazebo door '
                f'{self._sim_door_open_angle:.2f}rad')
            if (
                    self._sim_door_contact_only
                    and self._sim_door_initial_position is None):
                self._sim_door_initial_position = self._sim_door_position
            if not (
                    self._sim_door_contact_only
                    and self._sim_push_follow_enabled):
                self._command_sim_arm_pose('push', handle_pos)
            door_match_handle = door_frame_handle_pos or handle_pos
            if not self._open_sim_gazebo_door(door_match_handle, door_id):
                return False
            return self._wait_for_sim_duration(self._sim_step_sec)

        # The manipulation frame uses x forward and z up. A push-open door
        # should move the handle farther along +x, away from the robot body.
        target = self._make_pose(
            x=(handle_pos.point.x + self._handle_surface_inset
               + PULL_DISTANCE),
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_CLOSE)
        return self._plan_and_execute_cartesian(target)

    def _move_to_home(self):
        if self._feedback_contact:
            if not self._feedback_fresh():
                return self._feedback_stop('Lost feedback before final arm stow')
            if not self._command_sim_arm_pose('stow', None):
                return self._feedback_stop('Final stow command failed')
            if not self._wait_for_sim_arm_target(tolerance_rad=0.070):
                return self._feedback_stop('Final arm stow was not confirmed')
            self._feedback_event('home_verified')
            return True
        if self._sim_mode:
            self._command_sim_gripper(GRIPPER_OPEN)
            if self._sim_return_home:
                if self._sim_door_contact_only:
                    self.get_logger().info(
                        '  [SIM] Keeping the physical PIPER in its '
                        'navigation stow pose')
                    # The arm was already folded before the chassis push.
                    # Replaying the full recovery sequence here would raise
                    # the elbow into the doorway for no physical benefit.
                    if self._command_sim_arm_pose('stow', None):
                        self._wait_for_sim_arm_target(
                            tolerance_rad=0.070)
                    return
                self.get_logger().info('  [SIM] Returning to home position')
                if not self._command_sim_arm_pose('home', None):
                    return
                self._wait_for_sim_arm_target()
            else:
                self._wait_for_sim_duration(self._sim_step_sec)
            return

        if self._using_piper_sdk():
            if not self._piper_return_home:
                self.get_logger().info(
                    '  [PIPER SDK] Skipping home pose; '
                    'set piper_return_home:=true after measuring a safe pose.')
                return
            target = self._make_pose(
                x=self._piper_home_x,
                y=self._piper_home_y,
                z=self._piper_home_z,
            )
            self._send_piper_pose(target, GRIPPER_OPEN)
            return

        if self._arm is None:
            return
        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(configuration_name='home')
        plan_result = self._arm.plan()
        if plan_result:
            self._moveit.execute(plan_result.trajectory,
                                  controllers=[])

    # ── 실제 PIPER SDK 토픽 제어 유틸 ─────────────────────
    def _using_piper_sdk(self) -> bool:
        return (not self._sim_mode and
                self._control_backend == 'piper_sdk')

    # ── Gazebo 물리 문 개방 시뮬레이션 유틸 ───────────────
    def _open_sim_gazebo_door(self, handle_pos: PointStamped,
                              door_id: str = '') -> bool:
        if not self._sim_physical_door_opening:
            self._last_sim_open_was_idempotent = False
            self._last_sim_open_topic = ''
            self._last_sim_open_xy = None
            return True

        if self._sim_door_contact_only:
            self._last_sim_open_was_idempotent = False
            self._last_sim_open_topic = ''
            self._last_sim_open_xy = None
            self.get_logger().info(
                '  [SIM] Contact-only door verification enabled; '
                'not publishing any Gazebo door hinge command.')
            if not self._sim_verify_door_feedback:
                self.get_logger().error(
                    'Door feedback is required for contact-only verification.')
                return False
            if (
                    self._door_open_motion == 'push'
                    and self._sim_push_follow_enabled):
                return self._run_sim_push_contact_follow(
                    self._sim_door_contact_min_angle)
            return self._wait_for_sim_door_contact_motion(
                self._sim_door_contact_min_angle)

        if self._sim_door_command_topic:
            topic = self._sim_door_command_topic
            door_x = float(handle_pos.point.x)
            door_y = float(handle_pos.point.y)
            open_sign = self._sim_door_command_sign
            self.get_logger().info(
                f'  [SIM] Using explicit Gazebo door command topic={topic}')
        else:
            selected = self._select_sim_door_topic(handle_pos, door_id)
            if selected is None:
                if (self._sim_hinged_doors
                    and len(self._sim_opened_door_topics) >= len(self._sim_hinged_doors)):
                    self.get_logger().info(
                        '  [SIM] All hinged blue doors are already open; '
                        'treating request as idempotently complete.')
                    self._last_sim_open_was_idempotent = True
                    self._last_sim_open_topic = ''
                    self._last_sim_open_xy = None
                    return True
                msg = (
                    'No Gazebo hinged blue-door command topic matched this '
                    'open request.')
                if self._sim_door_physics_required:
                    self.get_logger().error(msg)
                    return False
                self.get_logger().warn(
                    msg + ' Falling back to logical sim open.')
                self._last_sim_open_was_idempotent = False
                self._last_sim_open_topic = ''
                self._last_sim_open_xy = None
                return True

            topic, door_x, door_y, open_sign = selected

        self._last_sim_open_was_idempotent = False
        self._last_sim_open_topic = topic
        self._last_sim_open_xy = (float(door_x), float(door_y))
        angle = self._sim_door_open_angle * open_sign
        self.get_logger().info(
            f'  [SIM] Gazebo door topic={topic}, target_angle={angle:.3f}')

        if self._sim_verify_door_feedback:
            self._sim_door_position = None

        for step in range(1, self._sim_door_command_steps + 1):
            target = angle * step / self._sim_door_command_steps
            if not self._publish_sim_command(topic, target):
                if self._sim_door_physics_required:
                    return False
                return True
            if not self._wait_for_sim_duration(0.12):
                return False

        success = True
        if self._sim_verify_door_feedback:
            success = self._wait_for_sim_door_target(angle)
        if success:
            self._sim_opened_door_topics.add(topic)
        return success

    def _select_sim_door_topic(self, handle_pos: PointStamped,
                               door_id: str = ''):
        if not self._sim_hinged_doors:
            self._sim_hinged_doors = self._load_sim_hinged_doors()
        if not self._sim_hinged_doors:
            return None

        if handle_pos.header.frame_id and handle_pos.header.frame_id != 'map':
            self.get_logger().warn(
                f'  [SIM] Cannot match Gazebo hinged door from '
                f'{handle_pos.header.frame_id} coordinates; expected map.')
            return None

        candidates: list[
            tuple[float, str, float, float, float, float, float, str, float, float]
        ] = []
        hx = float(handle_pos.point.x)
        hy = float(handle_pos.point.y)

        observed_station = None
        station_match = re.match(
            r"observed_blue_([mp]?\d+(?:p\d+)?)_([mp]?\d+(?:p\d+)?)",
            door_id)
        if station_match:
            observed_station = (
                self._decode_axis_hint_token(station_match.group(1)),
                self._decode_axis_hint_token(station_match.group(2)))

        match_points: list[tuple[float, float, str]] = [(hx, hy, 'map_handle')]
        if observed_station is not None:
            match_points.append((
                observed_station[0], observed_station[1], 'observed_station'))
        for axis_x, axis_y in self._axis_hints_from_door_id(door_id):
            match_points.append((axis_x, axis_y, 'mission_axis_hint'))

        for door in self._sim_hinged_doors:
            door_x = door['x']
            door_y = door['y']
            open_sign = door['open_sign']
            topic = door['topic']
            if topic in self._sim_opened_door_topics:
                continue

            best_match = None
            for match_x, match_y, source in match_points:
                progress_gap = max(
                    0.0, abs(match_x - door_x) - self._sim_door_panel_half_width)
                lateral_gap = abs(match_y - door_y)
                score = progress_gap ** 2 + 0.35 * lateral_gap ** 2
                item = (
                    score, progress_gap, lateral_gap, source, match_x, match_y)
                if best_match is None or item[0] < best_match[0]:
                    best_match = item
            if best_match is None:
                continue
            score, progress_gap, lateral_gap, source, match_x, match_y = best_match
            weighted_distance = math.sqrt(score)
            candidates.append((
                score, topic, door_y, open_sign, weighted_distance,
                progress_gap, lateral_gap, source, match_x, match_y))

        if not candidates:
            if self._sim_opened_door_topics:
                self.get_logger().warn(
                    f'  [SIM] No unopened Gazebo hinged blue door remains '
                    f'for {door_id or "(unknown)"}; opened='
                    f'{len(self._sim_opened_door_topics)}/'
                    f'{len(self._sim_hinged_doors)}.')
            return None

        candidates.sort(key=lambda item: item[0])
        (
            best_score, best_topic, best_y, best_open_sign, best_distance,
            best_progress_gap, best_lateral_gap, best_source,
            best_match_x, best_match_y) = candidates[0]
        second_distance = (
            float(candidates[1][4]) if len(candidates) > 1 else float('inf'))
        unique_axis_match = second_distance >= best_distance + 0.35
        if (
                self._sim_door_max_match_distance > 0.0
                and best_distance > self._sim_door_max_match_distance):
            same_wall_side = (
                abs(best_match_y) < 0.20
                or abs(best_y) < 0.20
                or best_match_y * best_y > 0.0)
            axis_consistent = (
                same_wall_side
                and self._sim_door_axis_match_max_progress > 0.0
                and self._sim_door_axis_match_max_lateral > 0.0
                and best_progress_gap <= self._sim_door_axis_match_max_progress
                and best_lateral_gap <= self._sim_door_axis_match_max_lateral
                and unique_axis_match)
            if axis_consistent:
                self.get_logger().warn(
                    f'  [SIM] Accepting Gazebo door match for '
                    f'{door_id or "(unknown)"} by same-wall axis consistency: '
                    f'nearest={best_topic}, distance={best_distance:.2f}m, '
                    f'progress_gap={best_progress_gap:.2f}m/'
                    f'{self._sim_door_axis_match_max_progress:.2f}m, '
                    f'lateral_gap={best_lateral_gap:.2f}m/'
                    f'{self._sim_door_axis_match_max_lateral:.2f}m.')
            else:
                self.get_logger().warn(
                    f'  [SIM] Rejected Gazebo door match for '
                    f'{door_id or "(unknown)"}: nearest={best_topic}, '
                    f'distance={best_distance:.2f}m > '
                    f'{self._sim_door_max_match_distance:.2f}m '
                    f'(score={best_score:.3f}, '
                    f'progress_gap={best_progress_gap:.2f}m, '
                    f'lateral_gap={best_lateral_gap:.2f}m, '
                    f'source={best_source}, '
                    f'match=({best_match_x:.2f},{best_match_y:.2f})).')
                return None
        self.get_logger().info(
            f'  [SIM] Matched observed door {door_id or "(unknown)"} to '
            f'{best_topic} (distance={best_distance:.2f}m, '
            f'score={best_score:.3f}, source={best_source}, '
            f'match=({best_match_x:.2f},{best_match_y:.2f}))')
        best_x = 0.0
        for door in self._sim_hinged_doors:
            if door['topic'] == best_topic:
                best_x = float(door['x'])
                break
        return best_topic, best_x, best_y, best_open_sign

    @staticmethod
    def _decode_axis_hint_token(value: str) -> float:
        sign = -1.0 if value.startswith("m") else 1.0
        digits = value[1:] if value[:1] in ("m", "p") else value
        return sign * float(digits.replace("p", "."))

    def _axis_hints_from_door_id(self, door_id: str) -> list[tuple[float, float]]:
        hints: list[tuple[float, float]] = []
        for match in re.finditer(
                r"__axis_([mp]\d+(?:p\d+)?)_([mp]\d+(?:p\d+)?)",
                door_id or ''):
            try:
                hints.append((
                    self._decode_axis_hint_token(match.group(1)),
                    self._decode_axis_hint_token(match.group(2))))
            except ValueError:
                continue
        return hints

    def _load_sim_hinged_doors(self) -> list[dict[str, float | str]]:
        world_file = self._sim_door_world_file.strip()
        if not world_file:
            return []
        path = Path(world_file)
        if not path.exists():
            self.get_logger().warn(
                f'  [SIM] Hinged door world file not found: {world_file}')
            return []
        try:
            text = path.read_text(encoding='utf-8')
        except Exception as e:
            self.get_logger().warn(
                f'  [SIM] Failed to read hinged door world file '
                f'{world_file}: {e}')
            return []

        prefix = re.escape(self._sim_door_topic_prefix)
        pattern = re.compile(
            rf'<topic>(?P<topic>{prefix}/x(?P<x>-?\d+)/'
            rf'y(?P<ysign>[pn])(?P<yabs>\d+)'
            rf'(?:/s(?P<open_sign>[pn]))?/cmd)</topic>')
        doors: list[dict[str, float | str]] = []
        for match in pattern.finditer(text):
            door_y = int(match.group('yabs')) / 100.0
            if match.group('ysign') == 'n':
                door_y *= -1.0
            sign_token = match.group('open_sign')
            if sign_token is None:
                open_sign = 1.0 if door_y >= 0.0 else -1.0
            else:
                open_sign = 1.0 if sign_token == 'p' else -1.0
            if self._sim_open_away_from_centerline:
                open_sign = 1.0 if door_y >= 0.0 else -1.0
            doors.append({
                'topic': match.group('topic'),
                'x': int(match.group('x')) / 100.0,
                'y': door_y,
                'open_sign': open_sign,
            })

        if doors:
            self.get_logger().info(
                f'  [SIM] Loaded {len(doors)} hinged blue doors from '
                f'{path.name}')
        return doors

    def _publish_ign_double(self, topic: str, value: float,
                            wait: bool = True) -> bool:
        cmd = [
            'ign', 'topic',
            '-t', topic,
            '-m', 'ignition.msgs.Double',
            '-p', f'data: {float(value):.6f}',
        ]
        if not wait:
            try:
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                self.get_logger().warn(
                    f'  [SIM] Failed to publish Ignition topic {topic}: {e}')
                return False

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0)
        except Exception as e:
            self.get_logger().warn(
                f'  [SIM] Failed to publish Ignition topic {topic}: {e}')
            return False
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            self.get_logger().warn(
                f'  [SIM] Ignition topic publish failed for {topic}: {detail}')
            return False
        return True

    def _publish_sim_command(self, topic: str, value: float,
                             wait: bool = True) -> bool:
        if self._sim_command_transport == 'ros':
            self._publish_float(self._sim_ros_publisher(topic), value)
            return True
        return self._publish_ign_double(topic, value, wait=wait)

    def _sim_ros_publisher(self, topic: str):
        publisher = self._sim_ros_float_pubs.get(topic)
        if publisher is None:
            publisher = self.create_publisher(Float64, topic, 10)
            self._sim_ros_float_pubs[topic] = publisher
        return publisher

    @staticmethod
    def _publish_float(publisher, value: float):
        if publisher is None:
            return
        msg = Float64()
        msg.data = float(value)
        publisher.publish(msg)

    def _sim_door_state_callback(self, msg: JointState):
        for joint_name, attr_name in (
                (self._sim_door_feedback_joint_name, '_sim_door_position'),
                (self._sim_lever_feedback_joint_name, '_sim_lever_position')):
            try:
                index = msg.name.index(joint_name)
                position = float(msg.position[index])
                if not math.isfinite(position):
                    continue
                setattr(self, attr_name, position)
                stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
                if (attr_name == '_sim_door_position'
                        and stamp != self._feedback_door_msg_stamp):
                    self._feedback_door_msg_stamp = stamp
                    self._feedback_door_stamp = time.monotonic()
                if (attr_name == '_sim_lever_position'
                        and stamp != self._feedback_lever_msg_stamp):
                    self._feedback_lever_msg_stamp = stamp
                    self._feedback_lever_stamp = time.monotonic()
                    self._feedback_lever_samples.append(
                        (self._feedback_lever_stamp, position))
                    del self._feedback_lever_samples[:-1000]
            except (ValueError, IndexError):
                continue

    def _feedback_handle_is_confident(self, msg):
        return (math.isfinite(msg.handle_confidence) and
                msg.handle_confidence >= self._feedback_config['handle_min_confidence'])

    def _detected_door_callback(self, msg: DoorInfo):
        if not msg.handle_detected or not msg.handle_position.header.frame_id:
            return
        if self._feedback_contact and (
                msg.door_color != 'blue'
                or not str(msg.handle_detection_method).startswith(
                    ('yolo:primary', 'track:primary_yolo'))
                or not str(msg.handle_detection_method).endswith(':registered_depth')
                or not self._feedback_handle_is_confident(msg)):
            return
        point = PointStamped()
        point.header = msg.handle_position.header
        point.point.x = float(msg.handle_position.point.x)
        point.point.y = float(msg.handle_position.point.y)
        point.point.z = float(msg.handle_position.point.z)
        if not all(math.isfinite(v) for v in (
                point.point.x, point.point.y, point.point.z)):
            return
        if self._feedback_contact:
            stamp = (point.header.stamp.sec, point.header.stamp.nanosec)
            image_time = stamp[0] + stamp[1] * 1.e-9
            if (stamp != (0, 0) and stamp != self._feedback_contact_image_stamp
                    and fresh_sample(image_time, self._sim_time_sec(),
                                     self._feedback_config['follow_max_age_sec'])):
                try:
                    # Transform at exposure time, not at the later callback time.
                    anchored = self._tf_buffer.transform(
                        point, self._contact_reference_frame, timeout=Duration(seconds=.1))
                except TransformException:
                    anchored = None
                if anchored is not None:
                    self._feedback_contact_image_stamp = stamp
                    self._feedback_contact_observation = (
                        time.monotonic(), anchored, str(msg.handle_detection_method))
        self._detected_handle_seq += 1
        self._detected_handle_samples.append((
            self._detected_handle_seq,
            point,
            float(msg.handle_confidence),
            str(msg.handle_detection_method)))
        self._feedback_handle_times[self._detected_handle_seq] = time.monotonic()
        del self._detected_handle_samples[:-30]
        self._feedback_handle_times = {
            key: value for key, value in self._feedback_handle_times.items()
            if key > self._detected_handle_seq - 30}

    def _fresh_refinement_samples(self, start_seq):
        samples = [sample for sample in self._detected_handle_samples
                   if sample[0] > start_seq]
        if not self._feedback_contact:
            return samples[-7:]
        if not samples or not fresh_sample(
                self._feedback_handle_times.get(samples[-1][0]), time.monotonic(), 3.):
            return []
        # Scene age uses the exposure clock. Receipt age independently detects
        # a dead stream; slow simulation does not make distinct images identical.
        unique = {}
        now = self._sim_time_sec()
        for sample in samples:
            stamp = sample[1].header.stamp
            key = (stamp.sec, stamp.nanosec)
            exposure = stamp.sec + stamp.nanosec * 1.e-9
            if key != (0, 0) and fresh_sample(exposure, now, 3.):
                unique[key] = sample
        return sorted(unique.values(), key=lambda sample: sample[0])[-7:]

    def _refine_handle_after_pre_grasp(
            self, previous: PointStamped) -> PointStamped | None:
        """Refresh a handle moved by contact using only live observations."""
        start_seq = self._detected_handle_seq
        start_sim = self._sim_time_sec()
        wall_deadline = time.monotonic() + max(
            90.0, 3.0 * self._sim_handle_refine_timeout)
        while (
                rclpy.ok()
                and time.monotonic() < wall_deadline
                and self._sim_time_sec() - start_sim
                < self._sim_handle_refine_timeout):
            current = self._fresh_refinement_samples(start_seq)
            if len(current) >= self._sim_handle_refine_min_samples:
                break
            time.sleep(0.02)
        else:
            return None

        usable = []
        for sample in current[-7:]:
            transformed = self._transform_handle_to_manipulation_frame(
                sample[1])
            if transformed is not None:
                usable.append((
                    sample[0], transformed, sample[2], sample[3]))
        if len(usable) < self._sim_handle_refine_min_samples:
            return None

        coordinates = np.asarray([
            (sample[1].point.x, sample[1].point.y, sample[1].point.z)
            for sample in usable], dtype=float)
        if not np.all(np.isfinite(coordinates)):
            return None
        median = np.median(coordinates, axis=0)
        shift = float(np.linalg.norm(median - np.asarray((
            previous.point.x, previous.point.y, previous.point.z))))
        if shift > self._sim_handle_refine_max_shift:
            self.get_logger().warn(
                'Ignoring implausible post-pre-grasp handle jump: '
                f'{shift:.3f}m > '
                f'{self._sim_handle_refine_max_shift:.3f}m')
            return None
        self._feedback_handle_points = coordinates.tolist()

        refined = PointStamped()
        refined.header = usable[-1][1].header
        refined.point.x = float(median[0])
        refined.point.y = float(median[1])
        refined.point.z = float(median[2])
        transformed = self._transform_handle_to_manipulation_frame(refined)
        if transformed is None:
            return None
        self.get_logger().info(
            'Refined handle from post-pre-grasp observations: '
            f'samples={len(usable)}, shift={shift:.3f}m, '
            f'xyz=({transformed.point.x:.3f}, '
            f'{transformed.point.y:.3f}, {transformed.point.z:.3f})')
        return transformed

    def _sim_arm_state_callback(self, msg: JointState):
        finite = {}
        for name, position in zip(msg.name, msg.position):
            if name in PIPER_JOINT_NAMES or name in (
                    'gripper_left_joint', 'gripper_right_joint'):
                if math.isfinite(position):
                    finite[name] = float(position)
        self._sim_arm_joint_positions.update(finite)
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if (all(name in finite for name in PIPER_JOINT_NAMES)
                and stamp != self._feedback_joint_msg_stamp):
            self._feedback_arm_stamp = time.monotonic()
            self._feedback_joint_msg_stamp = stamp

    def _feedback_event(self, event, **values):
        payload = {'event': event, 'sim_time_sec': self._sim_time_sec(),
                   'feedback_source': 'gazebo_joint_states', **values}
        msg = String()
        msg.data = json.dumps(payload, allow_nan=False)
        self._feedback_pub.publish(msg)
        self.get_logger().info('CONTACT_FEEDBACK ' + msg.data)

    def _feedback_fresh(self):
        now = time.monotonic()
        age = self._feedback_config['max_age_sec']
        return (fresh_sample(self._feedback_arm_stamp, now, age)
                and fresh_sample(self._feedback_lever_stamp, now, age)
                and fresh_sample(self._feedback_door_stamp, now, age))

    def _feedback_tool(self):
        if not self._feedback_fresh():
            return None
        joints = [self._sim_arm_joint_positions.get(name, float('nan'))
                  for name in PIPER_JOINT_NAMES]
        if not np.all(np.isfinite(joints)):
            return None
        point, _ = self._piper_kinematics.forward(joints)
        return point

    def _feedback_stop(self, reason):
        self._feedback_failure = reason
        self._publish_sim_cmd_vel(0.0, 0.0)
        # Remove the outstanding press target. Hold only fresh measured joints;
        # stale feedback must never generate another arm target.
        if fresh_sample(self._feedback_arm_stamp, time.monotonic(),
                        self._feedback_config['max_age_sec']):
            targets = {name: self._sim_arm_joint_positions[name]
                       for name in PIPER_JOINT_NAMES}
            self._command_sim_arm_targets('feedback_stop_hold', targets)
        self._feedback_event('stopped', reason=reason)
        return False

    def _feedback_command_point(self, goal, stage):
        if self._feedback_tool() is None:
            return self._feedback_stop('Joint or lever feedback is stale')
        seed = [self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES]
        try:
            rotation = self._feedback_contact_rotation()
        except (TransformException, ValueError) as exc:
            return self._feedback_stop('Wrist following rejected: ' + str(exc))
        # Released-hand servos require 2 mm convergence. Their IK residual must
        # be smaller, otherwise a settled joint command can never finish.
        released_step = stage in (
            'unload_finger_release', 'released_wrist_unload', 'released_wrist_retreat',
            'held_lever_return')
        solution = self._piper_kinematics.solve(
            goal, seed=seed, position_tolerance_m=0.001 if released_step else 0.003,
            orientation_tolerance_rad=0.015 if stage == 'held_lever_return' else 0.10,
            target_rotation=rotation,
            prefer_seed_solution=getattr(self, '_feedback_full_handle_follow', False))
        if solution is None:
            return self._feedback_stop('Observed target outside PIPER IK workspace')
        self._feedback_ideal_arm_target = solution.positions.copy()
        commanded = solution.positions + self._feedback_joint_bias
        if (np.any(commanded < PIPER_JOINT_LIMITS[:, 0])
                or np.any(commanded > PIPER_JOINT_LIMITS[:, 1])):
            return self._feedback_stop('Feedback correction would exceed PIPER joint limits')
        targets = dict(zip(PIPER_JOINT_NAMES, commanded.tolist()))
        if self._feedback_tool() is None:
            return self._feedback_stop('Joint or lever feedback expired during IK')
        success = self._command_sim_arm_targets(stage, targets)
        if success:
            self._feedback_last_cartesian_target = np.asarray(goal).copy()
            if stage in ('feedback_press', 'feedback_press_hold', 'feedback_lever_hold',
                         'feedback_latch_press', 'encoder_contact_follow'):
                actual, actual_rotation = self._piper_kinematics.forward([
                    self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
                self._feedback_event('contact_command', stage=stage,
                    requested_tool_xyz=np.asarray(goal).tolist(), actual_tool_xyz=actual.tolist(),
                    requested_rotation=(rotation.tolist() if rotation is not None else None),
                    actual_rotation=actual_rotation.tolist(),
                    lever_position_rad=self._sim_lever_position,
                    door_position_rad=self._sim_door_position)
        return success

    def _feedback_loaded_settle_goal(self, goal, actual):
        return goal

    def _feedback_move_tool(self, goal, stage):
        if not self._feedback_command_point(goal, stage):
            return False
        deadline = SimMotionDeadline(self._sim_time_sec(), time.monotonic(), 8.0,
                                     wall_limit=90.0)
        stable_since = None
        next_sample = self._sim_time_sec()
        previous_sample = None
        loaded_servo_corrections = 0
        while rclpy.ok():
            failure = deadline.failure(self._sim_time_sec(), time.monotonic())
            if failure:
                return self._feedback_stop('Approach step: ' + failure)
            actual = self._feedback_tool()
            if actual is None:
                return self._feedback_stop('Lost feedback while approaching handle')
            error = float(np.linalg.norm(actual - goal))
            if (stage == 'encoder_contact_follow'
                    and getattr(self, '_feedback_full_handle_follow', False)):
                lever = self._feedback_config['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                if lever < self._sim_lever_press_min_angle:
                    return self._feedback_stop('Lever contact lost while settling held arm')
                if error > self._feedback_config['tracking_limit_m']:
                    return self._feedback_stop('Held arm exceeded tracking limit while settling')
            joint_error = max(abs(self._sim_arm_joint_positions[name] - target)
                              for name, target in zip(
                                  PIPER_JOINT_NAMES, self._sim_last_arm_target))
            _, rotation = self._piper_kinematics.forward([
                self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
            try:
                desired = self._feedback_contact_rotation()
            except (TransformException, ValueError) as exc:
                return self._feedback_stop('Lost wrist reference during motion: ' + str(exc))
            if desired is None:
                desired = self._piper_kinematics.approach_rotation
            orientation_error = float(np.linalg.norm(self._piper_kinematics._rotation_error(rotation, desired)))
            if stage == 'encoder_contact_follow' and getattr(self, '_feedback_full_handle_follow', False):
                updated = self._feedback_loaded_settle_goal(goal, actual)
                if updated is None:
                    return False
                if not np.array_equal(updated, goal):
                    goal = updated
                    error = float(np.linalg.norm(actual-goal))
                    stable_since = None
                    previous_sample = None
            if self._sim_time_sec() >= next_sample:
                self._feedback_event('approach_sample', stage=stage,
                                     actual_tool_xyz=actual.tolist(),
                                     requested_xyz=np.asarray(goal).tolist(),
                                     error_m=error, orientation_error_rad=orientation_error,
                                     joint_tracking_error_rad=float(joint_error))
                # Bounded integral action cancels measured static servo bias
                # in both position and orientation, without changing the IK goal.
                if (previous_sample is not None
                        and np.linalg.norm(actual-previous_sample) < .002
                        and (error > self._feedback_config['tool_tolerance_m']
                             or orientation_error > .10)):
                    if stage == 'encoder_contact_follow' and getattr(self, '_feedback_full_handle_follow', False):
                        if orientation_error > .10:
                            return self._feedback_stop('Loaded wrist orientation cannot settle without further force')
                        try:
                            door0, yaw0, _ = self._feedback_contact_orientation_reference
                            heading = self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
                            normal = np.array([math.cos(heading), math.sin(heading), 0.])
                            relaxed = measured_tangential_relaxation(goal, actual, normal,
                                self._feedback_config['tracking_limit_m'],
                                self._feedback_config['follow_step_m'],
                                self._feedback_config['tool_tolerance_m'])
                            point = PointStamped(); point.header.frame_id = self._manipulation_frame
                            point.point.x, point.point.y, point.point.z = map(float, relaxed)
                            fixed = self._tf_buffer.transform(point, self._contact_reference_frame,
                                                             timeout=Duration(seconds=.1)).point
                        except (ValueError, TransformException) as exc:
                            return self._feedback_stop(str(exc))
                        old_goal = np.asarray(goal).copy()
                        if (np.linalg.norm(relaxed-old_goal) < .0005
                                and loaded_servo_corrections < 3):
                            # Tangential relaxation cannot remove a settled
                            # normal/vertical servo residual. Keep the SAME
                            # setpoint and use the bounded hold-servo correction.
                            if not self._feedback_correct_hold_servo():
                                return False
                            loaded_servo_corrections += 1
                            previous_sample = actual.copy()
                            next_sample = self._sim_time_sec()+.5
                            continue
                        goal = relaxed
                        if not self._feedback_command_point(goal, stage):
                            return False
                        _, previous_contact, previous_angle = self._feedback_sweep_state
                        self._feedback_sweep_state = (np.array([fixed.x, fixed.y, fixed.z]),
                                                       previous_contact, previous_angle)
                        self._feedback_press_origin[:2] = goal[:2]
                        self._feedback_event('stopped_tangential_contact_relaxation',
                            previous_target_xyz=old_goal.tolist(), requested_xyz=goal.tolist(),
                            actual_tool_xyz=actual.tolist(), normal_shift_m=float((goal-old_goal)@normal),
                            joint_bias_rad=self._feedback_joint_bias.tolist())
                        previous_sample = actual.copy()
                        next_sample = self._sim_time_sec()+.5
                        continue
                    measured = np.asarray([self._sim_arm_joint_positions[name]
                                           for name in PIPER_JOINT_NAMES])
                    candidate = self._feedback_joint_bias + .5 * (
                        self._feedback_ideal_arm_target-measured)
                    if np.max(np.abs(candidate)) > self._feedback_config['joint_bias_limit_rad']:
                        return self._feedback_stop('Joint bias correction limit exceeded')
                    self._feedback_joint_bias = candidate
                    if not self._feedback_command_point(goal, stage):
                        return False
                    self._feedback_event('servo_correction',
                                         joint_bias_rad=candidate.tolist(),
                                         measured_error_m=error,
                                         orientation_error_rad=orientation_error)
                previous_sample = actual.copy()
                next_sample = self._sim_time_sec() + .5
            if (stage == 'observed_pre_grasp'
                    and abs(self._sim_lever_position - self._sim_lever_initial_position) > .03):
                return self._feedback_stop('Unexpected lever contact during pre-grasp')
            if (error <= self._feedback_config['tool_tolerance_m']
                    and orientation_error <= .10):
                stable_since = stable_since or self._sim_time_sec()
                if self._sim_time_sec() - stable_since >= 0.35:
                    self._feedback_event('approach_reached', stage=stage,
                                         actual_tool_xyz=actual.tolist(),
                                         requested_xyz=np.asarray(goal).tolist(),
                                         error_m=error)
                    return True
            else:
                stable_since = None
            time.sleep(0.025)
        return self._feedback_stop('Approach failed to converge; no blind advance')

    def _feedback_joint_pregrasp(self, goal, observed_panel_x):
        start=np.array([self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
        try:
            path=self._piper_kinematics.joint_approach_path(goal,start,observed_panel_x)
        except ValueError as exc:
            return self._feedback_stop(str(exc)+'; measured joints='+str(start.tolist()))
        start=path[0].copy()
        total=float(np.max(np.abs(path[-1]-start)))/.25
        begin=self._sim_time_sec()
        deadline=SimMotionDeadline(begin,time.monotonic(),total+10.,wall_limit=120.)
        self._feedback_event('joint_pregrasp_plan', samples=len(path), duration_sec=total,
                             observed_goal_xyz=np.asarray(goal).tolist(),
                             panel_plane_x=observed_panel_x, collision_planner_used=False)
        while rclpy.ok() and self._sim_time_sec()-begin < total:
            failure=deadline.failure(self._sim_time_sec(),time.monotonic())
            if failure or self._feedback_tool() is None:
                return self._feedback_stop('Joint pre-grasp: '+(failure or 'stale encoders'))
            fraction=min(1.,(self._sim_time_sec()-begin)/max(total,.001))
            commanded=start+fraction*(path[-1]-start)
            measured=np.array([self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
            if np.max(np.abs(commanded-measured)) > .20:
                return self._feedback_stop('Joint pre-grasp tracking exceeded: inspect contact')
            self._publish_sim_arm_targets(dict(zip(PIPER_JOINT_NAMES,commanded.tolist())))
            self._sim_last_arm_target=commanded.copy()
            time.sleep(.025)
        return self._feedback_move_tool(goal,'observed_pre_grasp')

    def _feedback_grasp(self, handle_pos):
        # The gripper occludes a forward camera at close range. Freeze the
        # measured, stationary-handle anchor in odom (not an old base frame),
        # then use fresh encoders and lever feedback for contact acquisition.
        self._feedback_event('encoder_contact_handoff',
                             reason='close-range gripper occlusion',
                             anchor_source='repeated YOLO/RGB-D or image tracking')
        deadline = SimMotionDeadline(self._sim_time_sec(), time.monotonic(), 20.0,
                                     wall_limit=180.0)
        while rclpy.ok():
            failure = deadline.failure(self._sim_time_sec(), time.monotonic())
            if failure:
                return self._feedback_stop('Contact approach: ' + failure)
            anchor = PointStamped()
            anchor.header.frame_id = self._contact_reference_frame
            anchor.point = self._feedback_handle_anchor.point
            refined = self._transform_handle_to_manipulation_frame(anchor)
            if refined is None:
                return self._feedback_stop('Lost TF for observed stationary handle')
            handle_pos.point = refined.point
            goal = np.asarray((refined.point.x, refined.point.y, refined.point.z))
            actual = self._feedback_tool()
            if actual is None:
                return self._feedback_stop('No fresh tool feedback during grasp')
            delta = goal - actual
            distance = float(np.linalg.norm(delta))
            # Closing at the looser loaded-servo tolerance contradicts the
            # observed pre-contact alignment and lever-height checks.
            if distance <= min(self._feedback_config['tool_tolerance_m'], .005):
                if not self._feedback_close_gripper():
                    return False
                self._feedback_event('grasp_candidate', distance_m=distance,
                                     actual_tool_xyz=actual.tolist(),
                                     handle_xyz=goal.tolist())
                # Finger closure alone is not proof of grasp. PRESS_HANDLE
                # must cause measured lever motion before base push is allowed.
                return self._feedback_fresh()
            step = min(distance, self._feedback_config['approach_step_m'])
            if not self._feedback_move_tool(
                    actual + delta * step / distance, 'feedback_approach'):
                return False
        return self._feedback_stop('Approach travel timed out')

    def _feedback_close_gripper(self):
        # Close under the existing URDF force cap, rather than assuming the
        # handle diameter. Encoders only nominate contact; lever motion must
        # independently confirm the grasp during PRESS_HANDLE.
        self._command_sim_gripper(0.)
        start = self._sim_time_sec()
        deadline = SimMotionDeadline(start, time.monotonic(), 3.,
                                     stall_limit=2., wall_limit=25.)
        previous = None
        stable_since = None
        next_sample = start
        while rclpy.ok():
            now = self._sim_time_sec()
            if deadline.failure(now, time.monotonic()):
                return self._feedback_stop('Gripper closure did not establish opposed contact')
            if not self._feedback_fresh():
                return self._feedback_stop('Lost feedback during gripper closure')
            left = self._sim_arm_joint_positions.get('gripper_left_joint')
            right = self._sim_arm_joint_positions.get('gripper_right_joint')
            if left is None or right is None or not all(map(math.isfinite, (left, right))):
                return self._feedback_stop('Missing finger encoder feedback')
            if now >= next_sample:
                aperture = np.asarray((left, -right))
                # 2 mm exceeds the measured sub-mm free-space gravity bias.
                opposed = np.all(aperture > .002) and np.all(aperture <= .036)
                stable = previous is not None and np.max(np.abs(aperture-previous)) < .0005
                stable_since = (stable_since if stable_since is not None else now) if opposed and stable else None
                self._feedback_event('gripper_closure_sample', left_m=left, right_m=right,
                                     opposed_contact_candidate=bool(opposed))
                if stable_since is not None and now-stable_since >= .2:
                    self._feedback_event('gripper_contact_candidate', left_m=left, right_m=right,
                                         force_sensor_used=False)
                    return True
                previous = aperture
                next_sample = now + .05
            time.sleep(.02)
        return self._feedback_stop('Gripper closure interrupted')

    def _feedback_press_increment(self, goal, actual, step):
        return bounded_press_step(goal, actual, step,
                                  self._feedback_config['tracking_limit_m'])

    def _feedback_press(self, handle_pos):
        actual = self._feedback_tool()
        initial = self._sim_lever_initial_position
        if actual is None or initial is None or not math.isfinite(initial):
            return self._feedback_stop('Missing lever baseline or fresh arm feedback')
        if not self._feedback_anchor_contact_wrist():
            return False
        start = actual.copy()
        goal = start.copy()
        depth = 0.0
        began = time.monotonic()
        deadline = began + 30.0
        sim_start = self._sim_time_sec()
        cfg = self._feedback_config
        while (rclpy.ok() and time.monotonic() < deadline
               and self._sim_time_sec() - sim_start < self._sim_lever_press_timeout):
            actual = self._feedback_tool()
            if actual is None:
                return self._feedback_stop('Lost arm/lever feedback while pressing')
            lever = self._sim_lever_position
            recent = [(t, v) for t, v in self._feedback_lever_samples if t >= began]
            stable = stable_lever_motion(
                recent, began, initial, cfg['press_sign'],
                self._sim_lever_press_min_angle)
            if stable:
                self._feedback_press_verified = True
                # A loaded position servo needs its existing position error to
                # maintain force. Targeting measured FK here unloads the lever.
                # Keep the last bounded target; do not add another press step.
                if depth == 0.:
                    goal = getattr(self, '_feedback_last_cartesian_target', goal).copy()
                if np.linalg.norm(goal-actual) > cfg['tracking_limit_m']:
                    return self._feedback_stop('Press hold exceeds tracking limit')
                self._feedback_press_origin = goal.copy()
                self._feedback_press_origin[2] += depth
                self._feedback_press_depth = depth
                if not self._feedback_command_point(goal, 'feedback_press_hold'):
                    return False
                self._sim_push_hold_targets = dict(zip(
                    PIPER_JOINT_NAMES, self._sim_last_arm_target.tolist()))
                self._feedback_event(
                    'press_verified', commanded_depth_m=depth,
                    measured_tool_down_m=float(start[2] - actual[2]),
                    lever_delta_rad=cfg['press_sign'] * (lever - initial),
                    actual_tool_xyz=actual.tolist(), retained_target_xyz=goal.tolist())
                return True
            try:
                next_depth, pending = next_press_depth(
                    depth, cfg['press_step_m'], cfg['press_max_travel_m'],
                    initial, lever, cfg['press_sign'], self._sim_lever_press_min_angle)
            except ValueError as exc:
                return self._feedback_stop(str(exc))
            if pending:
                time.sleep(0.025)
                continue
            if float(np.linalg.norm(goal - actual)) > cfg['tracking_limit_m']:
                return self._feedback_stop('Press tracking error: possible jam/contact loss')
            try:
                increment = self._feedback_press_increment(goal, actual, next_depth-depth)
            except ValueError as exc:
                return self._feedback_stop(str(exc))
            if increment == 0.:
                time.sleep(.025)
                continue
            depth += increment
            try:
                goal = measured_press_target(start,actual,depth,cfg['press_max_travel_m'])
            except ValueError as exc:
                return self._feedback_stop(str(exc))
            if not self._feedback_command_point(goal, 'feedback_press'):
                return False
            self._feedback_event('press_step', commanded_depth_m=depth,
                                 lever_delta_rad=cfg['press_sign'] * (lever - initial),
                                 actual_tool_xyz=actual.tolist())
            if not self._wait_for_sim_duration(0.35, wall_stall_timeout_sec=2.0):
                return self._feedback_stop('Clock stalled during press')
        return self._feedback_stop('Lever feedback did not confirm press before timeout')

    def _feedback_live_contact_observation(self):
        sample = self._feedback_contact_observation
        if sample is None or not fresh_sample(
                sample[0], time.monotonic(),
                self._feedback_config['follow_max_age_sec']):
            return None
        point = sample[1]
        stamp = point.header.stamp.sec + point.header.stamp.nanosec * 1.e-9
        if not fresh_sample(stamp, self._sim_time_sec(),
                            self._feedback_config['follow_max_age_sec']):
            return None
        return sample

    def _feedback_begin_contact_follow(self):
        self._feedback_follow_mode = 'vision'
        self._publish_sim_cmd_vel(0.0, 0.0)
        deadline = time.monotonic() + 4.0
        while rclpy.ok() and time.monotonic() < deadline:
            sample = self._feedback_live_contact_observation()
            actual = self._feedback_tool()
            if actual is None:
                return self._feedback_stop('Lost feedback before visual contact following')
            if sample is not None:
                point = PointStamped()
                point.header.frame_id = self._manipulation_frame
                point.point.x, point.point.y, point.point.z = map(float, actual)
                try:
                    tool = self._tf_buffer.transform(point, self._contact_reference_frame, timeout=Duration(seconds=.1))
                except TransformException:
                    return self._feedback_stop('Cannot anchor measured contact tool in odom')
                observed = sample[1].point
                reference = np.asarray((observed.x, observed.y, observed.z))
                contact = np.asarray((tool.point.x, tool.point.y, tool.point.z))
                if np.linalg.norm(reference-contact) > self._feedback_config['tracking_limit_m']:
                    self._feedback_event(
                        'contact_visual_rejected', observation_source=sample[2],
                        observed_handle_odom=reference.tolist(),
                        measured_tool_odom=contact.tolist(),
                        separation_m=float(np.linalg.norm(reference-contact)))
                    if getattr(self, '_feedback_encoder_contact_follow', False):
                        # Reject the shifted image target, not the independently
                        # measured contact. This fallback still verifies freshness
                        # and lever depression before any new motion.
                        return self._feedback_begin_encoder_follow()
                    return self._feedback_stop('Live handle does not agree with the measured gripper contact')
                self._feedback_follow_reference = (reference, contact)
                self._feedback_follow_last_command = -math.inf
                self._feedback_event('contact_follow_started',
                                     observation_source=sample[2],
                                     observed_handle_odom=reference.tolist(),
                                     measured_tool_odom=contact.tolist())
                return True
            time.sleep(.025)
        if self._feedback_encoder_contact_follow:
            return self._feedback_begin_encoder_follow()
        return self._feedback_stop('No fresh RGB-D handle for moving-door contact following')

    def _feedback_begin_encoder_follow(self):
        actual = self._feedback_tool()
        if (not self._feedback_press_verified or actual is None
                or self._feedback_config['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                < self._sim_lever_press_min_angle):
            return self._feedback_stop('Occluded following requires fresh verified lever contact')
        self._feedback_follow_mode = 'encoder_lever_contact'
        try:
            yaw = self._feedback_base_yaw()
        except (TransformException, ValueError) as exc:
            return self._feedback_stop('Cannot anchor contact wrist: ' + str(exc))
        if not self._feedback_anchor_contact_wrist():
            return False
        point = PointStamped()
        point.header.frame_id = self._manipulation_frame
        point.point.x, point.point.y, point.point.z = map(float, actual)
        try:
            fixed = self._tf_buffer.transform(point, self._contact_reference_frame, timeout=Duration(seconds=.1))
        except TransformException as exc:
            return self._feedback_stop('Cannot anchor contact point: ' + str(exc))
        origin = np.asarray((fixed.point.x, fixed.point.y, fixed.point.z))
        self._feedback_sweep_state = (origin, origin.copy(), 0.)
        self._feedback_sweep_initial_door = self._sim_door_position
        self._feedback_sweep_normal = np.asarray((math.cos(yaw), math.sin(yaw), 0.))
        self._feedback_encoder_follow_origin = actual.copy()
        self._feedback_follow_last_command = -math.inf
        self._feedback_event('contact_follow_started',
                             observation_source='encoder_lever_contact',
                             reason='gripper occludes RGB-D; contact is still measured',
                             actual_tool_xyz=actual.tolist(),
                             force_sensor_used=False)
        return True

    def _feedback_base_heading_sample(self, rate_window=False):
        transform = self._tf_buffer.lookup_transform(
            self._contact_reference_frame, self._manipulation_frame, Time(), timeout=Duration(seconds=.1))
        stamp = transform.header.stamp.sec + transform.header.stamp.nanosec * 1.e-9
        now = self._sim_time_sec()
        maximum = self._feedback_config['max_age_sec']
        available = fresh_sample(stamp, now, maximum)
        # ROS clock and TF callbacks can arrive in either order. Rate samples
        # are buffered only; active_heading_rate discards stamps beyond the
        # completed drive interval before any rate is used by the controller.
        if rate_window and 0. < stamp-now <= maximum:
            available = True
        if not available:
            raise ValueError(f'Base orientation TF is stale or future: age={now-stamp:.6f}s')
        q = transform.transform.rotation
        if not all(math.isfinite(v) for v in (q.x, q.y, q.z, q.w)):
            raise ValueError('Base orientation is not finite')
        return stamp, math.atan2(2. * (q.w*q.z + q.x*q.y),
                                1. - 2. * (q.y*q.y + q.z*q.z))

    def _feedback_base_yaw(self):
        return self._feedback_base_heading_sample()[1]

    def _feedback_base_pose_sample(self):
        now=self._sim_time_sec()
        transform=self._tf_buffer.lookup_transform(self._contact_reference_frame,
            self._manipulation_frame,Time(),timeout=Duration(seconds=.1))
        stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
        if stamp>now:
            transform=self._tf_buffer.lookup_transform(self._contact_reference_frame,
                self._manipulation_frame,Time(nanoseconds=int(now*1.e9)),
                timeout=Duration(seconds=.1))
            stamp=transform.header.stamp.sec+transform.header.stamp.nanosec*1.e-9
        if not fresh_sample(stamp,now,.15):
            raise ValueError('Chassis progress needs fresh nonfuture wheel localization')
        p,q=transform.transform.translation,transform.transform.rotation
        pose=np.array([p.x,p.y,math.atan2(2.*(q.w*q.z+q.x*q.y),
                                        1.-2.*(q.y*q.y+q.z*q.z))])
        if not np.all(np.isfinite(pose)):
            raise ValueError('Chassis progress pose is not finite')
        return stamp,pose

    def _feedback_anchor_contact_wrist(self):
        if getattr(self, '_feedback_contact_orientation_reference', None) is not None:
            return True
        try:
            yaw = self._feedback_base_yaw()
        except (TransformException, ValueError) as exc:
            return self._feedback_stop('Cannot anchor contact wrist: ' + str(exc))
        if not self._feedback_fresh():
            return self._feedback_stop('Wrist anchor requires fresh lever feedback')
        self._feedback_contact_orientation_reference = (
            self._sim_door_position, yaw,
            self._piper_kinematics.approach_rotation.copy())
        self._feedback_wrist_lever_zero = self._sim_lever_initial_position
        self._feedback_peak_lever_press = 0.
        return True

    def _feedback_contact_rotation(self):
        released = getattr(self, '_feedback_released_rotation', None)
        if released is not None:
            return released.copy()
        reference = getattr(self, '_feedback_contact_orientation_reference', None)
        if reference is None:
            return None
        initial_door, initial_yaw, rotation = reference
        base_delta = self._feedback_base_yaw() - initial_yaw
        door_delta = self._sim_door_position - initial_door
        lever_zero = getattr(self, '_feedback_wrist_lever_zero', None)
        lever_delta = 0. if lever_zero is None else self._sim_lever_position - lever_zero
        cfg = getattr(self, '_feedback_config', {})
        lead = cfg.get('press_roll_lead_rad', 0.)
        if not math.isfinite(lead) or not 0. <= lead <= .1:
            raise ValueError('Wrist press lead exceeds bounded contact probe')
        if lever_zero is not None:
            # A rigidly grasped lever cannot rotate if the wrist only holds
            # its current angle. A small lead applies a bounded rotary press,
            # while measured lever motion still decides when pressing ends.
            sign = cfg.get('press_sign', -1.)
            peak = max(getattr(self, '_feedback_peak_lever_press', 0.), sign * lever_delta)
            self._feedback_peak_lever_press = peak
            # Do not unwind the wrist when the spring relaxes during IK or a
            # base probe. Keep the already measured pressed posture loaded.
            lever_delta = sign * (peak + lead)
        return measured_wrist_rotation(rotation, door_delta, base_delta, lever_delta)

    def _feedback_restore_lever_before_release(self):
        return self._feedback_stop('Observed neutral lever return is not configured')

    def _feedback_retract_released_handle(self, grip_span):
        self._publish_sim_cmd_vel(0.,0.)
        actual = self._feedback_tool()
        if actual is None:
            return self._feedback_stop('Missing tool feedback before released-hand retraction')
        _, rotation = self._piper_kinematics.forward([
            self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
        # Once released, follow the measured wrist pose, not the now freely
        # springing lever. Avoid a large folded-IK jump toward the chassis.
        self._feedback_released_rotation = rotation.copy()
        try:
            clearance = released_grip_clearance(
                self._piper_kinematics.tool_front_extent_m,
                self._piper_kinematics.tool_contact_offset_m, grip_span,
                self._feedback_config['tool_tolerance_m'])
        except ValueError as exc:
            return self._feedback_stop(str(exc))
        count = max(1,int(math.ceil(clearance/self._feedback_config['approach_step_m'])))
        for index in range(1,count+1):
            goal = actual-clearance*index/count*rotation[:,2]
            if not self._feedback_move_tool(goal,'released_handle_retract'):
                return False
        final = self._feedback_tool()
        if final is None:
            return self._feedback_stop('Lost feedback while verifying released-hand clearance')
        measured = float((actual-final)@rotation[:,2])
        if measured < clearance-self._feedback_config['tool_tolerance_m']:
            return self._feedback_stop('Released fingertips have not cleared the measured grip')
        self._feedback_event('released_handle_clearance_verified',
                             measured_grip_span_m=grip_span, requested_clearance_m=clearance,
                             measured_clearance_m=measured, actual_tool_xyz=final.tolist())
        return True

    def _feedback_follow_encoder_contact(self):
        actual = self._feedback_tool()
        if (actual is None or not self._feedback_press_verified
                or self._feedback_config['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                < self._sim_lever_press_min_angle):
            return self._feedback_stop('Occluded following lost measured lever contact')
        if (not getattr(self, '_feedback_full_handle_follow', False)
                and abs(actual[1]-self._feedback_encoder_follow_origin[1]) > self._feedback_config['follow_max_shift_m']):
            return self._feedback_stop('Occluded contact exceeded bounded lateral travel')
        now = self._sim_time_sec()
        if now-self._feedback_follow_last_command < .1:
            return True
        goal = self._feedback_press_origin.copy()
        measured = PointStamped()
        measured.header.frame_id = self._manipulation_frame
        measured.point.x, measured.point.y, measured.point.z = map(float, actual)
        try:
            fixed = self._tf_buffer.transform(measured, self._contact_reference_frame, timeout=Duration(seconds=.1))
            current = np.asarray((fixed.point.x, fixed.point.y, fixed.point.z))
            reference, previous, previous_angle = self._feedback_sweep_state
            angle = self._sim_door_position - self._feedback_sweep_initial_door
            if getattr(self, '_feedback_full_handle_follow', False):
                door0, yaw0, _ = self._feedback_contact_orientation_reference
                midpoint = yaw0 + self._feedback_sweep_initial_door - door0 + (previous_angle+angle)/2.
                tangent = np.asarray((math.cos(midpoint), math.sin(midpoint), 0.))
                # A measured pose is not a new command: rebasing on every FK
                # sample integrates load/servo error and can close the door.
                # Only an active chassis probe advances the world-frame target.
                projected_contact_increment(reference, previous, current, tangent,
                                            self._feedback_config['follow_max_shift_m'])
                target = reference.copy()
            else:
                target = swept_contact_reference(
                    reference, previous, current, self._feedback_sweep_normal,
                    previous_angle, angle, self._feedback_config['follow_max_shift_m'])
            point = PointStamped()
            point.header.frame_id = self._contact_reference_frame
            point.point.x, point.point.y, point.point.z = map(float, target)
            local = self._tf_buffer.transform(point, self._manipulation_frame,
                                              timeout=Duration(seconds=.1))
            goal[1] = local.point.y
            if getattr(self, '_feedback_full_handle_follow', False):
                goal[0] = local.point.x
        except (TransformException, ValueError) as exc:
            return self._feedback_stop('Contact sweep rejected: ' + str(exc))
        goal[2] -= self._feedback_press_depth
        held_height = getattr(self, '_feedback_held_height', None)
        if getattr(self, '_feedback_full_handle_follow', False) and held_height is not None:
            goal[2] = held_height
        if np.linalg.norm(goal-actual) > self._feedback_config['tracking_limit_m']:
            return self._feedback_stop('Occluded following exceeded arm tracking limit')
        if not self._feedback_command_point(goal, 'encoder_contact_follow'):
            return False
        self._feedback_sweep_state = (target, current, angle)
        self._feedback_press_origin[:2] = goal[:2]
        self._feedback_follow_last_command = now
        self._feedback_event('contact_follow_sample',
                             observation_source='encoder_lever_contact',
                             actual_tool_xyz=actual.tolist(), requested_tool_xyz=goal.tolist(),
                             lever_delta_rad=self._feedback_config['press_sign'] * (
                                 self._sim_lever_position-self._sim_lever_initial_position),
                             force_sensor_used=False)
        return True

    def _feedback_follow_contact(self):
        if self._feedback_follow_mode == 'encoder_lever_contact':
            return self._feedback_follow_encoder_contact()
        sample = self._feedback_live_contact_observation()
        actual = self._feedback_tool()
        if sample is None or actual is None:
            return self._feedback_stop('Lost live RGB-D or encoder feedback during contact following')
        if (abs(self._sim_door_position-self._sim_door_initial_position) > .03
                and self._feedback_config['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                < self._sim_lever_press_min_angle):
            return self._feedback_stop('Lever contact lost during visual following; reobservation required')
        now = self._sim_time_sec()
        if now - self._feedback_follow_last_command < .1:
            return True
        measured = PointStamped()
        measured.header.frame_id = self._manipulation_frame
        measured.point.x, measured.point.y, measured.point.z = map(float, actual)
        try:
            tool = self._tf_buffer.transform(measured, self._contact_reference_frame, timeout=Duration(seconds=.1))
            observed = sample[1].point
            reference, contact = self._feedback_follow_reference
            # Follow x/y scene displacement; lever feedback owns vertical press.
            observation = np.asarray((observed.x, observed.y, reference[2]))
            tool_position = np.asarray((tool.point.x, tool.point.y, tool.point.z))
            target = bounded_contact_target(
                observation, reference, contact, tool_position,
                self._feedback_config['follow_max_shift_m'],
                self._feedback_config['follow_step_m'])
            point = PointStamped()
            point.header.frame_id = self._contact_reference_frame
            point.point.x, point.point.y, point.point.z = map(float, target)
            local = self._tf_buffer.transform(point, self._manipulation_frame,
                                              timeout=Duration(seconds=.1))
        except (TransformException, ValueError) as exc:
            return self._feedback_stop('Contact follow rejected: ' + str(exc))
        goal = np.asarray((local.point.x, local.point.y,
                           self._feedback_press_origin[2] - self._feedback_press_depth))
        if np.linalg.norm(goal-actual) > self._feedback_config['tracking_limit_m']:
            return self._feedback_stop('Contact following exceeded measured tracking limit')
        if not self._feedback_command_point(goal, 'observed_contact_follow'):
            return False
        # Advance the measured reference, not a fixed-radius door trajectory.
        # Preserve the unclipped target so small servo steps cannot lose motion.
        self._feedback_follow_reference = (
            observation, contact + observation-reference)
        self._feedback_press_origin[:2] = goal[:2]
        self._feedback_follow_last_command = now
        self._feedback_event('contact_follow_sample', observation_source=sample[2],
                             observed_handle_odom=[observed.x, observed.y, observed.z],
                             requested_tool_xyz=goal.tolist(), actual_tool_xyz=actual.tolist())
        return True

    def _feedback_confirmed_hold_reference(self, previous_goal, goal):
        return True

    def _feedback_retain_lever_press(self):
        actual = self._feedback_tool()
        cfg = self._feedback_config
        lever_delta = cfg['press_sign'] * (
            self._sim_lever_position - self._sim_lever_initial_position)
        if actual is None or not self._feedback_fresh():
            return self._feedback_stop('Lost feedback while retaining lever press')
        if not math.isfinite(lever_delta) or lever_delta < self._sim_lever_press_min_angle:
            return self._feedback_stop('Lever contact lost before hold correction')
        if self._feedback_lever_hold_delta - lever_delta <= cfg['lever_hold_tolerance_rad']:
            return True
        # Stop the base before correcting relaxation; never press after loss.
        self._publish_sim_cmd_vel(0.0, 0.0)
        try:
            goal = measured_press_target(self._feedback_press_origin,actual,
                self._feedback_press_depth,cfg['press_max_travel_m'])
        except ValueError as exc:
            return self._feedback_stop(str(exc))
        if np.linalg.norm(goal - actual) > cfg['tracking_limit_m']:
            return self._feedback_stop('Lever hold tracking error: possible jam')
        # A loaded servo may sit below its old command. Lowering that command
        # alone can still request upward motion, so probe below the measured tip.
        goal[2] = min(goal[2], actual[2])
        try:
            increment = self._feedback_press_increment(goal, actual, min(.005, cfg['press_step_m']))
        except ValueError as exc:
            return self._feedback_stop(str(exc))
        if increment == 0.:
            return True
        depth = self._feedback_press_origin[2] - goal[2] + increment
        if depth > cfg['press_max_travel_m']:
            return self._feedback_stop('Lever hold correction reached press travel limit')
        goal[2] = self._feedback_press_origin[2] - depth
        previous_goal = self._feedback_last_cartesian_target.copy()
        if not self._feedback_command_point(goal, 'feedback_lever_hold'):
            return False
        # IK runs while physics continues. Compare the response with the
        # state at command dispatch, not a state from before planning.
        lever_before = self._sim_lever_position
        door_before = abs(self._sim_door_position - self._sim_door_initial_position)
        self._feedback_press_depth = depth
        if (getattr(self, '_feedback_full_handle_follow', False)
                and getattr(self, '_feedback_held_height', None) is not None):
            # Following must retain the accepted press command, not restore
            # the older held height and then accumulate another correction.
            self._feedback_held_height = float(goal[2])
        self._feedback_event('lever_hold_correction', commanded_depth_m=depth,
                             actual_tool_xyz=actual.tolist(), requested_tool_xyz=goal.tolist(),
                             lever_delta_rad=lever_delta,
                             retained_lever_delta_rad=self._feedback_lever_hold_delta)
        if not self._feedback_confirm_latch_press_response(
                lever_before, door_before, require_lever_motion=True):
            return False
        return self._feedback_confirmed_hold_reference(previous_goal, goal)

    def _feedback_confirm_latch_press_response(
            self, lever_before, door_before, require_lever_motion=False):
        # A lever can stay depressed against the strike after the fingers slip.
        # A fresh angle alone is insufficient: a correction must cause motion.
        self._publish_sim_cmd_vel(0.0, 0.0)
        start = self._sim_time_sec()
        deadline = SimMotionDeadline(start, time.monotonic(), 4.,
                                     stall_limit=2., wall_limit=20.)
        last_movement = start
        previous = self._feedback_tool()
        next_sample = start
        servo_corrections = 0
        while rclpy.ok():
            if not self._feedback_fresh():
                return self._feedback_stop('Lost feedback while confirming latch press response')
            now = self._sim_time_sec()
            if deadline.failure(now, time.monotonic()):
                break
            actual = self._feedback_tool()
            if actual is None:
                return self._feedback_stop('Lost tool feedback during latch correction')
            lever_motion = self._feedback_config['press_sign'] * (
                self._sim_lever_position - lever_before)
            door_motion = abs(self._sim_door_position - self._sim_door_initial_position) - door_before
            if lever_motion >= .005 or (not require_lever_motion and door_motion >= .003):
                self._feedback_event('latch_press_response',
                                     lever_motion_rad=lever_motion,
                                     door_motion_rad=door_motion)
                return True
            if now >= next_sample:
                moving = previous is not None and np.linalg.norm(actual-previous) >= .0005
                if moving:
                    last_movement = now
                elif (require_lever_motion
                      and (getattr(self, '_feedback_full_handle_follow', False)
                           or (getattr(self, '_phase', None) == 'LATCH_CLEAR_PUSH'
                               and getattr(self, '_feedback_follow_mode', None) == 'encoder_lever_contact'))
                      and servo_corrections < 3 and now-start >= .5
                      and now-last_movement >= .4
                      and np.linalg.norm(actual-self._feedback_last_cartesian_target) > .0025):
                    # Correct static servo error on the SAME bounded setpoint,
                    # not a deeper press. Lever response still decides success.
                    if not self._feedback_correct_hold_servo():
                        return False
                    servo_corrections += 1
                    last_movement = now
                previous = actual.copy()
                next_sample = now + .1
                self._feedback_event('latch_response_sample',
                                     lever_motion_rad=lever_motion,
                                     door_motion_rad=door_motion,
                                     actual_tool_xyz=actual.tolist())
            # Do not call a moving servo unresponsive, but a settled arm with
            # no lever/door response is not permission for another press step.
            if now - start >= 1. and now - last_movement >= .5:
                break
            time.sleep(.025)
        return self._feedback_stop(
            'No measured response to latch press correction; possible contact loss or jam, reobservation required')

    def _feedback_correct_hold_servo(self):
        self._publish_sim_cmd_vel(0.,0.)
        actual = self._feedback_tool()
        cfg = self._feedback_config
        if actual is None or cfg['press_sign']*(
                self._sim_lever_position-self._sim_lever_initial_position) < self._sim_lever_press_min_angle:
            return self._feedback_stop('Lost lever contact before hold servo correction')
        goal = self._feedback_last_cartesian_target.copy()
        if np.linalg.norm(goal-actual) > cfg['tracking_limit_m']:
            return self._feedback_stop('Hold servo tracking limit exceeded')
        measured = np.array([self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
        candidate = self._feedback_joint_bias + .5*(self._feedback_ideal_arm_target-measured)
        if np.max(np.abs(candidate)) > cfg['joint_bias_limit_rad']:
            return self._feedback_stop('Hold servo bias limit exceeded')
        self._feedback_joint_bias = candidate
        if not self._feedback_command_point(goal, 'feedback_lever_hold'):
            return False
        self._feedback_event('hold_servo_correction', requested_tool_xyz=goal.tolist(),
                             actual_tool_xyz=actual.tolist(), joint_bias_rad=candidate.tolist())
        return True

    def _feedback_settle_held_target(self):
        if not getattr(self, '_feedback_full_handle_follow', False):
            return True
        actual = self._feedback_tool()
        if actual is None:
            return self._feedback_stop('Held arm feedback lost before base motion')
        goal = self._feedback_last_cartesian_target
        error = float(np.linalg.norm(goal-actual))
        if error > self._feedback_config['tracking_limit_m']:
            return self._feedback_stop('Held arm exceeded tracking limit before base motion')
        _, rotation = self._piper_kinematics.forward([
            self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES])
        try:
            desired = self._feedback_contact_rotation()
        except (TransformException, ValueError) as exc:
            return self._feedback_stop('Held wrist feedback rejected: ' + str(exc))
        if desired is None:
            return self._feedback_stop('Held wrist reference missing before base motion')
        orientation_error = float(np.linalg.norm(
            self._piper_kinematics._rotation_error(rotation, desired)))
        if (error <= self._feedback_config['tool_tolerance_m']
                and orientation_error <= .10):
            return True
        # Do not let base motion outrun the loaded arm servo. The existing
        # bounded bias correction runs stopped, with the original joint limits.
        self._publish_sim_cmd_vel(0., 0.)
        self._feedback_event('held_arm_settling', error_m=error,
                             orientation_error_rad=orientation_error)
        return self._feedback_move_tool(goal, 'encoder_contact_follow')

    def _feedback_wait_contact_chassis_rest(self):
        return True

    def _feedback_probe_pose_sample(self):
        return self._feedback_base_pose_sample()

    def _feedback_probe_pulse_scale(self,progress,duration,pulse_duration):
        return 1.-progress

    def _feedback_commit_probe_points(self,point,measured):
        return (self._tf_buffer.transform(point,self._contact_reference_frame,
                    timeout=Duration(seconds=.1)).point,
                self._tf_buffer.transform(measured,self._contact_reference_frame,
                    timeout=Duration(seconds=.1)).point)

    def _feedback_qualify_probe_reference(self,stamp,goal):
        return None

    def _feedback_contact_ik_reobservation(self,goal,rotation,seed,solution):
        return False

    def _feedback_probe_contact_motion(self, angular_z=0., speed=.02, opening_speed=None):
        # IK can take longer than a short probe. Plan while stopped, then
        # measure only the commanded motion interval, with live safety checks.
        self._publish_sim_cmd_vel(0., 0.)
        if not self._feedback_wait_contact_chassis_rest():
            return None
        if not self._feedback_retain_lever_press() or not self._feedback_follow_contact():
            return None
        if not self._feedback_settle_held_target():
            return None
        scene=self._feedback_scene_command_safe(speed,angular_z)
        if scene == 'replan':
            return 'scene_wait'
        if not scene:
            return None
        cfg = self._feedback_config
        duration = min(.35, cfg.get('follow_step_m', .008) / max(abs(speed),.001))
        coordinated = None
        driven_opening_speed = opening_speed
        if getattr(self, '_feedback_full_handle_follow', False):
            try:
                door0, yaw0, _ = self._feedback_contact_orientation_reference
                alpha = self._sim_door_position-door0-self._feedback_base_yaw()+yaw0
                normal = np.array([math.cos(alpha), math.sin(alpha), 0.])
                origin = self._feedback_last_cartesian_target.copy()
                # Only the chassis supplies normal opening travel. The arm
                # can retract to absorb excess drive and reach its preferred
                # posture, but must not advance beyond the chassis supply.
                driven_opening_speed=chassis_supplied_opening_speed(
                    origin,normal,speed,angular_z,opening_speed)
                # Check the actual local step instead of overestimating it by
                # summing base and tip speeds that mostly cancel one another.
                for _ in range(8):
                    try:
                        endpoint, inverse_yaw = coordinated_contact_endpoint(
                            origin, normal, speed, angular_z, duration,
                            .75*cfg['follow_step_m'], driven_opening_speed)
                        break
                    except ValueError as exc:
                        if 'exceeds bounded contact step' not in str(exc):
                            raise
                        duration *= .5
                else:
                    raise ValueError('No bounded coordinated contact duration')
                rotation = inverse_yaw @ self._feedback_contact_rotation()
                seed = self._feedback_ideal_arm_target.copy()
                solution = self._piper_kinematics.solve(
                    endpoint, seed=seed, position_tolerance_m=.001,
                    orientation_tolerance_rad=.03, target_rotation=rotation,
                    prefer_seed_solution=True)
                if solution is None:
                    raise ValueError('Coordinated contact endpoint outside arm workspace')
                end_joints = solution.positions + self._feedback_joint_bias
                if (np.any(end_joints < PIPER_JOINT_LIMITS[:, 0])
                        or np.any(end_joints > PIPER_JOINT_LIMITS[:, 1])):
                    raise ValueError('Coordinated contact would exceed PIPER limits')
                coordinated = (origin, endpoint, seed, solution.positions)
                world_angle = self._sim_door_position-door0+yaw0
                world_normal = np.array([math.cos(world_angle), math.sin(world_angle), 0.])
                initial_pose_stamp,initial_base_pose=self._feedback_probe_pose_sample()
                last_target_stamp = initial_pose_stamp
                self._feedback_qualified_probe_reference = self._feedback_qualify_probe_reference(
                    initial_pose_stamp,self._feedback_last_cartesian_target)
            except (ValueError, TransformException) as exc:
                self._feedback_stop(str(exc))
                return None
        start = self._sim_time_sec()
        deadline = SimMotionDeadline(start, time.monotonic(), duration + 1.,
                                     stall_limit=2., wall_limit=10.)
        interrupted = False
        sensor_interrupted = False
        scene_interrupted = False
        fraction = 0.
        chassis_fraction = 0.
        coordinated_pulses = 0
        commanded_sim_sec = 0.
        minimum_pulse_scale = 1.
        active_door_samples, active_base_samples = [], []
        active_sensor = getattr(self, '_door_input_source', None)
        try:
            # Sensor and IK work can consume a short probe's nominal duration.
            # Coordinated motion ends at measured travel, with the same deadline.
            while rclpy.ok() and (
                    (coordinated is None and self._sim_time_sec() - start < duration)
                    or (coordinated is not None and chassis_fraction < 1.)):
                if coordinated is not None:
                    # Never leave a short chassis command active during scan
                    # processing or IK. Motion is bounded between these checks.
                    self._publish_sim_cmd_vel(0., 0.)
                generation=getattr(self,'_feedback_measurement_pause_generation',0)
                if not self._feedback_fresh() or self._feedback_tool() is None:
                    self._feedback_stop('Latch probe lost live joint feedback')
                    return None
                if getattr(self,'_feedback_measurement_pause_generation',0) != generation:
                    # The base stopped during the wait. Never advance the arm
                    # interpolation as though that paused motion had occurred.
                    sensor_interrupted=True
                    break
                failure = deadline.failure(self._sim_time_sec(), time.monotonic())
                if failure:
                    self._feedback_stop('Latch probe: ' + failure)
                    return None
                lever_delta = cfg['press_sign'] * (
                    self._sim_lever_position - self._sim_lever_initial_position)
                if lever_delta < self._sim_lever_press_min_angle:
                    self._feedback_stop('Lever contact lost during chassis probe')
                    return None
                scene=self._feedback_scene_command_safe(speed,angular_z,active=True)
                if scene=='replan':
                    scene_interrupted=True
                    break
                if not scene:
                    return None
                if self._feedback_lever_hold_delta - lever_delta > cfg['lever_hold_tolerance_rad']:
                    interrupted = True
                    break
                if coordinated is not None:
                    stamp = self._feedback_door_msg_stamp
                    sensor = getattr(self, '_door_input_source', None)
                    if sensor != active_sensor:
                        active_door_samples.clear()
                        active_sensor = sensor
                    if stamp is not None:
                        measured = stamp[0]+stamp[1]*1.e-9
                        if not active_door_samples or measured != active_door_samples[-1][0]:
                            active_door_samples.append((measured, self._sim_door_position))
                    try:
                        measured, heading = self._feedback_base_heading_sample(rate_window=True)
                    except (ValueError, TransformException) as exc:
                        self._feedback_stop('Active base orientation: '+str(exc))
                        return None
                    if not active_base_samples or measured != active_base_samples[-1][0]:
                        if active_base_samples:
                            previous = active_base_samples[-1][1]
                            heading = previous+math.atan2(math.sin(heading-previous),math.cos(heading-previous))
                        active_base_samples.append((measured, heading))
                    origin, endpoint, seed, end = coordinated
                    tip_speed = abs(speed)+np.linalg.norm(origin[:2])*abs(angular_z)
                    base_pose = None
                    try:
                        pose_stamp,base_pose=self._feedback_probe_pose_sample()
                        if getattr(self,'_feedback_measurement_pause_generation',0)!=generation:
                            sensor_interrupted=True
                            break
                        fraction,motion_residual,chassis_fraction=observed_probe_motion(
                            initial_base_pose,base_pose,origin,normal,speed,angular_z,
                            duration,cfg['follow_step_m'])
                    except (ValueError,TransformException) as exc:
                        self._feedback_event('measured_probe_rejected',
                            initial_pose=initial_base_pose.tolist(),
                            current_pose=base_pose.tolist() if base_pose is not None else None,
                            linear_x=speed, angular_z=angular_z, planned_duration_sec=duration,
                            elapsed_sim_sec=self._sim_time_sec()-start, reason=str(exc))
                        self._feedback_stop('Measured chassis progress: '+str(exc))
                        return None
                    try:
                        goal, measured_advance = measured_contact_endpoint(initial_base_pose,base_pose,
                            origin,normal,driven_opening_speed*duration*fraction,cfg['follow_step_m'])
                        reference = self._feedback_qualify_probe_reference(pose_stamp,goal)
                        # Compensate the accepted turning command's small
                        # remaining rotation without inventing base translation.
                        requested_rotation,wrist_lead=bounded_wrist_yaw_lead(
                            self._feedback_contact_rotation(),angular_z,
                            max(0.,duration-(self._sim_time_sec()-start)),
                            .25*cfg['follow_step_m'],self._piper_kinematics.tool_front_extent_m)
                        orientation_tolerance=(min(.03,max(.003,.5*abs(wrist_lead)))
                                               if abs(wrist_lead)>.003 else .03)
                        local_seed=self._feedback_ideal_arm_target.copy()
                        live = self._piper_kinematics.solve(goal,
                            seed=local_seed, max_iterations=25,
                            position_tolerance_m=.001, orientation_tolerance_rad=orientation_tolerance,
                            target_rotation=requested_rotation, prefer_seed_solution=True,
                            maximum_seed_step_rad=.14)
                        if live is None:
                            if self._feedback_contact_ik_reobservation(goal,requested_rotation,local_seed,None):
                                sensor_interrupted=True
                                break
                            raise ValueError('Measured contact compensation is outside PIPER workspace')
                        ideal = live.positions
                        if np.any(ideal<local_seed-.14) or np.any(ideal>local_seed+.14):
                            if self._feedback_contact_ik_reobservation(goal,requested_rotation,local_seed,live):
                                sensor_interrupted=True
                                break
                            raise ValueError('Measured contact compensation exceeds bounded joint step')
                        self._feedback_contact_ik_pending_since=None
                    except (ValueError,TransformException) as exc:
                        self._feedback_stop(str(exc))
                        return None
                    if np.linalg.norm(goal-self._feedback_tool()) > cfg['tracking_limit_m']:
                        self._feedback_stop('Coordinated arm/base probe tracking exceeded')
                        return None
                    joints = ideal + self._feedback_joint_bias
                    if (np.any(joints<PIPER_JOINT_LIMITS[:,0])
                            or np.any(joints>PIPER_JOINT_LIMITS[:,1])):
                        self._feedback_stop('Measured compensation would exceed PIPER joint limits')
                        return None
                    self._publish_sim_arm_targets(dict(zip(PIPER_JOINT_NAMES, joints.tolist())))
                    self._sim_last_arm_target = joints.copy()
                    self._feedback_ideal_arm_target = ideal.copy()
                    self._feedback_last_cartesian_target = goal.copy()
                    last_target_stamp = pose_stamp
                    self._feedback_qualified_probe_reference = reference
                    self._feedback_press_origin[:2] = goal[:2]
                    # Leave headroom for sensor/solver latency and wheel
                    # deceleration. The original 8 mm hard limit still applies.
                    if (chassis_fraction >= 1.
                            or (1.-chassis_fraction)*duration*tip_speed <= .05*cfg['follow_step_m']
                            or np.linalg.norm(goal-origin)>=.75*cfg['follow_step_m']):
                        break
                if coordinated is None:
                    self._publish_sim_cmd_vel(speed, angular_z)
                    time.sleep(.025)
                else:
                    # Time limits this pulse only; it never supplies arm travel.
                    # Compensation and completion still use measured SE2 motion.
                    pulse_start = self._sim_time_sec()
                    # Taper both components together, preserving the checked
                    # turning radius while leaving headroom for wheel stopping.
                    pulse_scale = 1.-chassis_fraction
                    minimum_pulse_scale = min(minimum_pulse_scale, pulse_scale)
                    pulse_duration = min(.05, duration,
                        .25*cfg['follow_step_m']/max(tip_speed*pulse_scale, .001))
                    try:
                        pulse_scale=self._feedback_probe_pulse_scale(
                            chassis_fraction,duration,pulse_duration)
                    except ValueError as exc:
                        self._feedback_stop('Measured contact pulse timing: '+str(exc))
                        return None
                    if getattr(self,'_feedback_measurement_pause_generation',0)!=generation:
                        sensor_interrupted=True
                        break
                    self._publish_sim_cmd_vel(speed*pulse_scale, angular_z*pulse_scale)
                    try:
                        while rclpy.ok() and self._sim_time_sec()-pulse_start < pulse_duration:
                            failure = deadline.failure(self._sim_time_sec(), time.monotonic())
                            if failure:
                                self._feedback_stop('Latch probe pulse: '+failure)
                                return None
                            time.sleep(.005)
                    finally:
                        self._publish_sim_cmd_vel(0., 0.)
                        coordinated_pulses += 1
                        commanded_sim_sec += self._sim_time_sec()-pulse_start
        finally:
            self._publish_sim_cmd_vel(0., 0.)
        if coordinated is not None:
            try:
                active_door_rate = active_heading_rate(active_door_samples, start, self._sim_time_sec())
                active_base_rate = active_heading_rate(active_base_samples, start, self._sim_time_sec())
            except ValueError as exc:
                self._feedback_stop('Active heading rate rejected: '+str(exc))
                return None
            for name, rate in (('door',active_door_rate),('base',active_base_rate)):
                if rate is not None:
                    history = getattr(self, '_feedback_active_'+name+'_rate_history', [])
                    history = (history+[rate])[-3:]
                    setattr(self, '_feedback_active_'+name+'_rate_history', history)
                    setattr(self, '_feedback_active_'+name+'_yaw_rate', float(np.median(history)))
            # Commit the bounded target actually dispatched before a pause,
            # not the unexecuted endpoint and not the loaded encoder residual.
            point = PointStamped()
            point.header.frame_id = self._manipulation_frame
            # The local target belongs to the pose used to calculate it,
            # not a newer pose received after the chassis command stopped.
            point.header.stamp = Time(seconds=last_target_stamp).to_msg()
            point.point.x, point.point.y, point.point.z = map(
                float, self._feedback_last_cartesian_target)
            measured = PointStamped()
            measured.header.frame_id = self._manipulation_frame
            actual = self._feedback_tool()
            if actual is None:
                self._feedback_stop('Lost encoders while committing contact probe')
                return None
            measured.point.x, measured.point.y, measured.point.z = map(float, actual)
            try:
                target,current=self._feedback_commit_probe_points(point,measured)
            except (TransformException,ValueError) as exc:
                self._feedback_stop('Cannot commit observed contact probe: ' + str(exc))
                return None
            self._feedback_sweep_state = (
                np.array([target.x, target.y, target.z]),
                np.array([current.x, current.y, current.z]),
                getattr(self,'_feedback_commit_panel_angle',self._sim_door_position)
                -self._feedback_sweep_initial_door)
        self._feedback_event('contact_probe_motion', started_sim_sec=start,
                             driven_sim_sec=self._sim_time_sec()-start,
                             planned_duration_sec=duration,
                             interrupted_for_press=interrupted,
                             interrupted_for_sensor_wait=sensor_interrupted,
                             interrupted_for_scene_recheck=scene_interrupted,
                             linear_x=speed, angular_z=angular_z,
                             opening_speed_mps=driven_opening_speed,
                             measured_arm_fraction=fraction if coordinated is not None else None,
                             measured_chassis_fraction=chassis_fraction if coordinated is not None else None,
                             arm_progress_basis='wheel_localization' if coordinated is not None else None,
                             termination_basis=('measured_chassis_progress' if coordinated is not None
                                                else 'elapsed_command_duration'),
                             coordinated_pulses=coordinated_pulses,
                             commanded_sim_sec=commanded_sim_sec if coordinated is not None else None,
                             minimum_pulse_scale=minimum_pulse_scale if coordinated is not None else None,
                             completion_travel_tolerance_m=.05*cfg['follow_step_m'],
                             active_door_yaw_rate=getattr(self,'_feedback_active_door_yaw_rate',None),
                             active_base_yaw_rate=getattr(self,'_feedback_active_base_yaw_rate',None))
        return ('scene_wait' if scene_interrupted else
                'sensor_wait' if sensor_interrupted else ('relaxed' if interrupted else 'complete'))

    def _feedback_opening_complete(self, delta, min_delta):
        return delta >= min_delta

    def _feedback_panel_clearance_margin(self, actual):
        return .03

    def _feedback_panel_barrier_geometry(self, actual, normal):
        return None, None, self._feedback_panel_clearance_margin(actual)

    def _feedback_scene_constraints(self):
        return None

    def _feedback_scene_command_safe(self, speed, angular, active=False):
        return True

    def _feedback_handle_held_open(self, min_delta):
        """The hand is the load path; the chassis moves without touching the panel."""
        if not self._feedback_press_verified or not self._feedback_fresh():
            return self._feedback_stop('Handle-held opening requires verified live lever contact')
        fingers = self._sim_arm_joint_positions
        if not (0.002 < fingers.get('gripper_left_joint',0.) <= .036
                and .002 < -fingers.get('gripper_right_joint',0.) <= .036):
            self._publish_sim_cmd_vel(0.,0.)
            return self._feedback_stop('Opposed finger contact lost before posture selection')
        # Latch-clear probes used a different following law. Anchor the held
        # opening once at its measured starting pose before advancing it.
        if not self._feedback_begin_encoder_follow():
            return False
        for name in ('door', 'base'):
            setattr(self, '_feedback_active_'+name+'_rate_history', [])
            attribute = '_feedback_active_'+name+'_yaw_rate'
            if hasattr(self, attribute):
                delattr(self, attribute)
        self._feedback_full_handle_follow = True
        preferred = getattr(self, '_feedback_preferred_contact_posture',
                            self._feedback_encoder_follow_origin).copy()
        actual = self._feedback_tool()
        if actual is None:
            return self._feedback_stop('No measured grip posture for base following')
        preferred[2] = actual[2]
        # Search from measured reach, not a previous near-body goal that can
        # truncate every future-orientation candidate before IK is evaluated.
        observed_posture = actual.copy()
        minimum_reach = (float(PIPER_BODY_OUTLINE[:,0].max())
                         + self._piper_kinematics.tool_front_extent_m
                         + self._feedback_config['tool_tolerance_m'])
        try:
            preferred, condition = self._piper_kinematics.preferred_contact_posture(
                observed_posture, self._feedback_contact_rotation(),
                [self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES],
                minimum_reach)
        except (ValueError, TransformException) as exc:
            return self._feedback_stop(str(exc))
        self._feedback_event('preferred_contact_posture', preferred_tool_xyz=preferred.tolist(),
                             translation_condition_ratio=condition,
                             basis='observed_grip_height_and_calibrated_robot_workspace')
        deadline = SimMotionDeadline(self._sim_time_sec(), time.monotonic(), 360.,
                                     stall_limit=3., wall_limit=3000.)
        last_progress = self._sim_time_sec()
        heading_samples = [(last_progress, self._sim_door_position)]
        base_heading_samples = [(last_progress,self._feedback_base_yaw())]
        observed_yaw_rate = 0.
        observed_base_yaw_rate = 0.
        observed_advance_speed = .012
        contact_samples = []
        contact_geometry_samples = list(getattr(self, '_observed_latch_geometry', []))
        geometry_fit = (observed_contact_curvature(contact_geometry_samples)
                        if len(contact_geometry_samples)>=12 else None)
        geometry_fit_angle = None
        prepare_stance = bool(getattr(self, '_feedback_prepare_contact_stance', False))
        stance_ready = not prepare_stance
        measured_stance_adopted = False
        if prepare_stance and geometry_fit is None:
            return self._feedback_stop('No independently observed latch trajectory for safe opening stance')
        posture_heading = None
        posture_curvature = None
        posture_margin = None
        progress_angle = abs(self._sim_door_position-self._sim_door_initial_position)
        reversal_peak_lower = 0.
        progress_watch = ObservedContactProgress(last_progress, progress_angle)
        reversal_gate = ObservedSceneGate()
        try:
            while rclpy.ok():
                failure = deadline.failure(self._sim_time_sec(), time.monotonic())
                if failure:
                    return self._feedback_stop('Handle-held opening: ' + failure)
                actual = self._feedback_tool()
                if actual is None:
                    return self._feedback_stop('Handle-held opening lost live encoders')
                fingers = self._sim_arm_joint_positions
                left, right = fingers.get('gripper_left_joint', 0.), fingers.get('gripper_right_joint', 0.)
                if not (0.002 < left <= .036 and .002 < -right <= .036):
                    return self._feedback_stop('Opposed finger contact lost during handle-held opening')
                lever = self._feedback_config['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                if lever < self._sim_lever_press_min_angle:
                    return self._feedback_stop('Lever released before full opening')
                delta = abs(self._sim_door_position-self._sim_door_initial_position)
                progress_delta = (self._feedback_progress_angle() if
                                  hasattr(self, '_feedback_progress_angle') else delta)
                if progress_delta is None:
                    self._publish_sim_cmd_vel(0., 0.)
                    time.sleep(.025)
                    continue
                try:
                    if not hasattr(self, '_feedback_progress_angle'):
                        if progress_angle-delta > .08:
                            return self._feedback_stop('Observed door reversed during held opening')
                    else:
                        stamp = self._feedback_door_msg_stamp
                        if stamp is None:
                            return self._feedback_stop('No stamped panel observation for reversal check')
                        measured_time = stamp[0]+stamp[1]*1.e-9
                        if not fresh_sample(measured_time, self._sim_time_sec(), .25):
                            self._publish_sim_cmd_vel(0., 0.)
                            time.sleep(.025)
                            continue
                        interval = (self._feedback_progress_interval() if
                                    hasattr(self, '_feedback_progress_interval') else
                                    (progress_delta, progress_delta))
                        if interval is None:
                            self._publish_sim_cmd_vel(0., 0.)
                            time.sleep(.025)
                            continue
                        # Acquiring the interval can advance /clock. Use one
                        # current snapshot for qualification and its gate.
                        # An expired sample requests stopped reobservation,
                        # never a motion certificate or a reversal diagnosis.
                        checked_now=self._sim_time_sec()
                        if not fresh_sample(measured_time,checked_now,.25):
                            self._publish_sim_cmd_vel(0.,0.)
                            time.sleep(.025)
                            continue
                        lower, upper = interval
                        reversal_gate.observe(measured_time,
                            checked_now, time.monotonic(), reversal_peak_lower-upper <= .08)
                        reversal_peak_lower = max(reversal_peak_lower, lower)
                except SceneObservationPending:
                    self._publish_sim_cmd_vel(0., 0.)
                    time.sleep(.025)
                    continue
                except ValueError as exc:
                    return self._feedback_stop('Observed door reversal recheck: '+str(exc))
                if progress_delta > progress_angle + .003:
                    last_progress, progress_angle = self._sim_time_sec(), progress_delta
                if delta >= min_delta and self._feedback_opening_complete(delta,min_delta):
                    self._feedback_event('handle_held_open_verified', door_delta_rad=delta,
                                         lever_delta_rad=lever, actual_tool_xyz=actual.tolist())
                    return True
                door0, yaw0, _ = self._feedback_contact_orientation_reference
                yaw = self._feedback_base_yaw()
                normal_error = math.atan2(math.sin(self._sim_door_position-door0-yaw+yaw0),
                                          math.cos(self._sim_door_position-door0-yaw+yaw0))
                normal = np.array([math.cos(normal_error),math.sin(normal_error),0.])
                try:
                    measured = PointStamped()
                    measured.header.frame_id = self._manipulation_frame
                    measured.point.x, measured.point.y, measured.point.z = map(float, actual)
                    fixed = self._tf_buffer.transform(measured, self._contact_reference_frame,
                                                     timeout=Duration(seconds=.1))
                    contact = np.array([fixed.point.x,fixed.point.y,fixed.point.z])
                    sample_now = self._sim_time_sec()
                    heading = yaw0+self._sim_door_position-door0
                    axis = np.array([math.cos(heading),math.sin(heading),0.])
                    if not contact_samples or sample_now-contact_samples[-1][0] >= .1:
                        contact_samples.append((sample_now,*contact.tolist()))
                        contact_samples=[sample for sample in contact_samples if sample_now-sample[0] <= .75]
                        advance=robust_contact_speed(contact_samples,axis)
                        if advance is not None:
                            observed_advance_speed=.5*observed_advance_speed+.5*max(0.,advance)
                    stamp = self._feedback_door_msg_stamp
                    joint_stamp = self._feedback_joint_msg_stamp
                    if stamp is not None and joint_stamp is not None:
                        measured_stamp = stamp[0]+stamp[1]*1.e-9
                        measured_panel_angle=self._sim_door_position
                        raw_panel=getattr(self,'_raw_contact_panel',None)
                        if prepare_stance and raw_panel is not None:
                            measured_stamp,measured_panel_angle=raw_panel
                        joint_time = joint_stamp[0]+joint_stamp[1]*1.e-9
                        if (0. <= sample_now-measured_stamp <= .1
                                and abs(measured_stamp-joint_time) <= .1
                                and (not contact_geometry_samples
                                     or measured_stamp > contact_geometry_samples[-1][0])):
                            contact_geometry_samples.append((measured_stamp,
                                measured_panel_angle, contact[0], contact[1]))
                    if (len(contact_geometry_samples) >= 12
                            and (geometry_fit_angle is None
                                 or abs(self._sim_door_position-geometry_fit_angle) >= .1)):
                        fit = observed_contact_curvature(contact_geometry_samples)
                        if fit is not None:
                            geometry_fit = fit
                            geometry_fit_angle = self._sim_door_position
                            self._feedback_event('observed_contact_geometry', **fit,
                                basis='sensor_panel_headings_and_encoder_tcp_in_robot_localization_frame')
                except (ValueError, TransformException) as exc:
                    return self._feedback_stop('Measured contact velocity: '+str(exc))
                # The measured average includes deliberately stopped IK time.
                # Do not feed it back as the active request and cause self-decay.
                # Measurements still gate freshness, progress and safety.
                advance_speed = self._feedback_config['opening_speed_mps']
                now = self._sim_time_sec()
                if now-heading_samples[-1][0] >= .1:
                    heading_samples.append((now,self._sim_door_position))
                    previous_yaw=base_heading_samples[-1][1]
                    unwrapped_yaw=previous_yaw+math.atan2(math.sin(yaw-previous_yaw),math.cos(yaw-previous_yaw))
                    base_heading_samples.append((now,unwrapped_yaw))
                    heading_samples=[sample for sample in heading_samples if now-sample[0] <= .75]
                    base_heading_samples=[sample for sample in base_heading_samples if now-sample[0] <= .75]
                    try:
                        rate=robust_heading_rate(heading_samples)
                        base_rate=robust_heading_rate(base_heading_samples)
                    except ValueError as exc:
                        return self._feedback_stop(str(exc))
                    if rate is not None:
                        observed_yaw_rate=.5*observed_yaw_rate+.5*rate
                    if base_rate is not None:
                        observed_base_yaw_rate=.5*observed_base_yaw_rate+.5*base_rate
                measured_curvature = None if geometry_fit is None else geometry_fit['curvature_per_m']
                self._feedback_observed_contact_curvature=measured_curvature
                radius=None if geometry_fit is None else geometry_fit.get('radius_m')
                radius_uncertainty=None if geometry_fit is None else geometry_fit.get('radius_uncertainty_m')
                self._feedback_contact_curvature_uncertainty=(None if radius is None or
                    radius_uncertainty is None else abs(measured_curvature)*radius_uncertainty/radius)
                planning_curvature=(conservative_contact_curvature(geometry_fit)
                    if radius is not None and radius_uncertainty is not None else measured_curvature)
                try:
                    current_margin = self._feedback_panel_clearance_margin(actual)
                except (ValueError, TransformException) as exc:
                    return self._feedback_stop('Held clearance observation: '+str(exc))
                if not measured_stance_adopted and (posture_heading is None or (not prepare_stance and abs(normal_error-posture_heading) > .025)
                        or (measured_curvature is not None and posture_curvature is None)
                        or (prepare_stance and planning_curvature is not None and posture_curvature is not None
                            and (not math.isclose(planning_curvature,posture_curvature,rel_tol=.01)
                                 or abs(measured_curvature*preferred[0])>=1.))
                        or (posture_margin is not None and current_margin > posture_margin+.005)):
                    try:
                        rotation = self._feedback_contact_rotation()
                        future_rotation = predicted_contact_rotation(rotation,
                            getattr(self,'_feedback_active_door_yaw_rate',observed_yaw_rate),
                            getattr(self,'_feedback_active_base_yaw_rate',observed_base_yaw_rate))
                        preferred, condition = self._piper_kinematics.preferred_contact_posture(
                            observed_posture, rotation,
                            [self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES],
                            minimum_reach, future_rotation=future_rotation,
                            # This is a soft future stance, never an arm goal.
                            # Live IK and joint/scene barriers guard every
                            # reachable intermediate measured posture.
                            observed_curvature=planning_curvature,
                            minimum_panel_clearance=current_margin,
                            steady_orientation_only=prepare_stance and measured_curvature is not None,
                            minimum_displacement=prepare_stance)
                    except TransformException as exc:
                        return self._feedback_stop('Held posture update: ' + str(exc))
                    except ValueError as exc:
                        if 'No reachable contact posture' not in str(exc):
                            return self._feedback_stop('Held posture update: ' + str(exc))
                        if measured_curvature is not None and abs(measured_curvature*preferred[0])>=1.:
                            return self._feedback_stop('Observed geometry invalidated the previous contact stance')
                        # Keep the last verified goal; live encoder barriers below
                        # may recover the chassis without a normal arm push.
                        self._feedback_event('held_workspace_recovery_required', reason=str(exc),
                            actual_tool_xyz=actual.tolist(), preferred_tool_xyz=preferred.tolist())
                    posture_heading = normal_error
                    posture_curvature = planning_curvature
                    posture_margin = current_margin
                    self._feedback_event('held_workspace_posture_updated',
                        preferred_tool_xyz=preferred.tolist(),
                        translation_condition_ratio=condition,
                        relative_panel_heading_rad=normal_error,
                        predicted_relative_rate_rad_sec=observed_yaw_rate-observed_base_yaw_rate,
                        observed_contact_curvature_per_m=measured_curvature,
                        planned_contact_curvature_per_m=planning_curvature,
                        planned_panel_clearance_margin_m=current_margin,
                        joint_margin_rad=.20)
                if prepare_stance and measured_curvature is not None:
                    stance_error=float(np.linalg.norm(actual[:2]-preferred[:2]))
                    reach_error=abs(float(actual[0]-preferred[0]))
                    stance_tolerance=self._feedback_config['tool_tolerance_m']
                    near_selected=(stance_error<=2.*stance_tolerance
                                   and reach_error<=stance_tolerance)
                    # The selected point is a soft preparation goal. An
                    # attained, independently safe steady stance may keep
                    # more reach, but cannot shorten the selected clearance.
                    reach_preserved=actual[0]>=preferred[0]-stance_tolerance
                    if not stance_ready and not measured_stance_adopted and reach_preserved:
                        try:
                            anchor,plane,independent_margin=self._feedback_panel_barrier_geometry(actual,normal)
                            uncertainty=getattr(self,'_feedback_contact_curvature_uncertainty',None)
                            curvatures=[measured_curvature]
                            if uncertainty is not None:
                                curvatures.extend([measured_curvature-2.*uncertainty,
                                                   measured_curvature+2.*uncertainty])
                            checked=[self._piper_kinematics.measured_steady_contact_posture(
                                [self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES],
                                self._feedback_contact_rotation(),curve,
                                max(current_margin,independent_margin),
                                panel_anchor=anchor,panel_normal=plane) for curve in curvatures]
                            attained=checked[0][0]
                            attained_condition=min(item[1] for item in checked)
                        except (ValueError,SceneObservationPending):
                            pass
                        else:
                            condition=min(condition,attained_condition)
                            measured_stance_adopted=True
                            selected=preferred.copy()
                            preferred=attained.copy()
                            stance_error=reach_error=0.
                            near_selected=True
                            self._contact_lookahead_cache=None
                            self._contact_lookahead_arc_pose=None
                            self._feedback_event('measured_steady_stance_adopted',
                                actual_tool_xyz=attained.tolist(),translation_condition_ratio=condition,
                                selected_tool_xyz=selected.tolist(),
                                observed_contact_curvature_per_m=measured_curvature,
                                panel_clearance_margin_m=max(current_margin,independent_margin),
                                curvature_uncertainty_per_m=uncertainty,
                                basis='own_encoders_and_independent_plane_validate_preserved_reach_and_steady_IK')
                    desired_heading=contact_steady_heading(measured_curvature,preferred[0])
                    heading_error=abs(normal_error-desired_heading)
                    if stance_ready and (stance_error>3.*stance_tolerance
                                         or reach_error>2.*stance_tolerance or heading_error>.20):
                        stance_ready=False
                    if not stance_ready:
                        stance_ready=bool(near_selected and measured_stance_adopted and heading_error<=.12)
                    # A differential-drive base cannot generally reach a
                    # lateral steady stance while keeping the held door fixed.
                    # Let chassis-supplied opening and posture tracking converge
                    # together, under the same arm/panel/scene barriers below.
                    self._feedback_event('observed_contact_stance',ready=stance_ready,
                        tool_error_m=stance_error,heading_error_rad=heading_error,
                        requested_opening_speed_mps=advance_speed,
                        basis='observed_latch_path_and_actual_robot_workspace',
                        strategy='joint_constrained_simultaneous_stance_and_opening')
                try:
                    progress_heading = (contact_steady_heading(measured_curvature,preferred[0])
                        if prepare_stance and measured_curvature is not None else None)
                    opened, corrected = progress_watch.observe(
                        self._sim_time_sec(), progress_delta, actual, preferred,
                        actual_heading=normal_error if progress_heading is not None else None,
                        preferred_heading=progress_heading)
                    if corrected and not opened:
                        self._feedback_event('measured_posture_correction_progress',
                            actual_tool_xyz=actual.tolist(), preferred_tool_xyz=preferred.tolist(),
                            posture_error_m=progress_watch.best_error,
                            heading_error_rad=progress_watch.best_heading_error,
                            seconds_without_opening=self._sim_time_sec()-progress_watch.last_opening)
                    panel_anchor, panel_normal, clearance_margin = self._feedback_panel_barrier_geometry(actual, normal)
                    active_door_rate = getattr(self,'_feedback_active_door_yaw_rate',observed_yaw_rate)
                    modeled_panel_drift = active_door_rate if measured_curvature is None else 0.
                    arm_rows, arm_bounds, measured_condition = self._piper_kinematics.contact_velocity_barriers(
                        [self._sim_arm_joint_positions[name] for name in PIPER_JOINT_NAMES],
                        normal, modeled_panel_drift, contact_curvature=measured_curvature)
                    speed, angular, posture, clearance = self._feedback_contact_velocity(
                        actual, normal, preferred, PIPER_BODY_OUTLINE,
                        margin=clearance_margin, door_yaw_rate=modeled_panel_drift,
                        panel_anchor=panel_anchor, panel_normal=panel_normal,
                        observed_curvature=measured_curvature,
                        # Preparing the measured opening stance must not trade
                        # continuing arm extension for a little more opening.
                        # Heading remains soft until the stance is attained.
                        opening_speed=advance_speed, limit_posture_drift=prepare_stance,
                        opening_priority_weight=self._feedback_config.get('opening_priority_weight', .2),
                        # Convert angular-rate error to measured arc-speed error;
                        # a fixed weight of 20 implied a fictitious 4.47m lever.
                        heading_priority_weight=1./measured_curvature**2 if prepare_stance and measured_curvature is not None else 0.,
                        heading_tolerance_rad=.15 if prepare_stance and stance_ready else None,
                        measured_heading_reference=prepare_stance and not stance_ready,
                        allow_posture_recovery=self._sim_time_sec()-last_progress>1.,
                        arm_constraints=(arm_rows, arm_bounds),
                        scene_constraints=self._feedback_scene_constraints())
                except SceneObservationPending:
                    self._publish_sim_cmd_vel(0.,0.)
                    time.sleep(.025)
                    continue
                except (ValueError, TransformException) as exc:
                    return self._feedback_stop(str(exc))
                probe=self._feedback_probe_contact_motion(angular, speed, advance_speed)
                if probe is None:
                    return False
                if probe == 'scene_wait':
                    continue
                self._feedback_event('handle_held_base_sample', door_delta_rad=delta,
                                     lever_delta_rad=lever, angular_z=angular,
                                     linear_x=speed, preferred_tool_xyz=posture.tolist(),
                                     relative_panel_heading_rad=normal_error,
                                     observed_door_yaw_rate=active_door_rate,
                                     average_door_yaw_rate=observed_yaw_rate,
                                     observed_base_yaw_rate=observed_base_yaw_rate,
                                     observed_contact_advance_mps=observed_advance_speed,
                                     measured_body_plane_clearance_m=clearance,
                                     clearance_margin_m=clearance_margin,
                                     measured_arm_condition_ratio=measured_condition,
                                     actual_tool_xyz=actual.tolist(), force_sensor_used=False)
                if speed < 0.:
                    self._feedback_event('bounded_chassis_posture_recovery', linear_x=speed,
                                         angular_z=angular, actual_tool_xyz=actual.tolist())
        finally:
            self._publish_sim_cmd_vel(0., 0.)
        return self._feedback_stop('Handle-held opening interrupted')

    def _feedback_contact_velocity(self, *args, **kwargs):
        return contact_posture_command(*args, **kwargs)

    def _feedback_latch_motion(self):
        return .02, 0., None

    def _feedback_latch_clear_push(self):
        # Lever motion proves contact, not latch release. Briefly probe door
        # motion, stopping the chassis before any additional press correction.
        initial = self._sim_door_initial_position
        if initial is None or not self._feedback_fresh():
            return self._feedback_stop('Missing feedback before latch-clear push')
        cfg = self._feedback_config
        deadline = SimMotionDeadline(self._sim_time_sec(), time.monotonic(), 30.,
                                     stall_limit=12., wall_limit=360.)
        previous = abs(self._sim_door_position - initial)
        self._feedback_lever_hold_delta = cfg['press_sign'] * (
            self._sim_lever_position - self._sim_lever_initial_position)
        try:
            # An already lost contact must never trigger a new arm target.
            if (previous > .03 and cfg['press_sign'] * (
                    self._sim_lever_position-self._sim_lever_initial_position)
                    < self._sim_lever_press_min_angle):
                return self._feedback_stop(
                    'Lever contact lost as door rotated; reobservation required')
            if not self._feedback_begin_contact_follow():
                return False
            while rclpy.ok():
                failure = deadline.failure(self._sim_time_sec(), time.monotonic())
                if failure:
                    return self._feedback_stop('Latch release feedback: '+failure)
                if not self._feedback_fresh():
                    return self._feedback_stop('Lost fresh latch release observation')
                actual = self._feedback_tool()
                if actual is None:
                    return self._feedback_stop('Lost feedback during latch-clear push')
                lever_delta = cfg['press_sign'] * (
                    self._sim_lever_position - self._sim_lever_initial_position)
                current = abs(self._sim_door_position - initial)
                if (current > .03
                        and lever_delta < self._sim_lever_press_min_angle):
                    return self._feedback_stop(
                        'Lever contact lost as door rotated; reobservation required')
                # A partial pulse can already release the latch. Recheck the
                # fresh measured state before driving again, not only after a
                # pulse finishes without a sensor or scene pause.
                if current >= self._sim_latch_clear_min_angle:
                    self._publish_sim_cmd_vel(0., 0.)
                    self._feedback_event('latch_released', door_delta_rad=current)
                    return True
                if lever_delta >= self._sim_lever_press_min_angle:
                    try:
                        speed, angular, advance = self._feedback_latch_motion()
                    except SceneObservationPending:
                        self._publish_sim_cmd_vel(0., 0.)
                        time.sleep(.025)
                        continue
                    except (ValueError, TransformException) as exc:
                        return self._feedback_stop('Observed latch motion: '+str(exc))
                    probe = self._feedback_probe_contact_motion(angular, speed, advance)
                    if probe is None:
                        return False
                    if probe in ('relaxed','sensor_wait','scene_wait'):
                        time.sleep(.025)
                        continue
                current = abs(self._sim_door_position - initial)
                self._feedback_event('latch_probe', door_delta_rad=current,
                                     lever_delta_rad=lever_delta,
                                     commanded_depth_m=self._feedback_press_depth)
                if current >= self._sim_latch_clear_min_angle:
                    self._feedback_event('latch_released', door_delta_rad=current)
                    return True
                if current - previous >= .003:
                    previous = current
                    continue
                # A stationary door is never permission to keep driving.
                self._publish_sim_cmd_vel(0.0, 0.0)
                actual = self._feedback_tool()
                if actual is None:
                    return self._feedback_stop('Latch correction lost measured tool')
                try:
                    goal = measured_press_target(self._feedback_press_origin,actual,
                        self._feedback_press_depth,cfg['press_max_travel_m'])
                except ValueError as exc:
                    return self._feedback_stop(str(exc))
                if np.linalg.norm(goal-actual) > cfg['tracking_limit_m']:
                    return self._feedback_stop('Latch press tracking error: possible jam')
                try:
                    increment = self._feedback_press_increment(goal, actual, min(.005, cfg['press_step_m']))
                except ValueError as exc:
                    return self._feedback_stop(str(exc))
                if increment == 0.:
                    time.sleep(.025)
                    continue
                depth = self._feedback_press_depth + increment
                if depth > cfg['press_max_travel_m']:
                    return self._feedback_stop('Latch not released within press travel limit')
                goal[2] = self._feedback_press_origin[2] - depth
                if not self._feedback_command_point(goal, 'feedback_latch_press'):
                    return False
                self._feedback_press_origin[:2] = goal[:2]
                lever_before = self._sim_lever_position
                door_before = abs(self._sim_door_position - initial)
                self._feedback_press_depth = depth
                self._feedback_event('latch_press_correction', commanded_depth_m=depth)
                if not self._feedback_confirm_latch_press_response(lever_before, door_before):
                    return False
                self._feedback_lever_hold_delta = max(
                    self._feedback_lever_hold_delta,
                    cfg['press_sign'] * (
                        self._sim_lever_position-self._sim_lever_initial_position))
                previous = current
        finally:
            self._publish_sim_cmd_vel(0.0, 0.0)
        return self._feedback_stop('Latch release feedback timed out')

    def _sim_time_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _wait_for_sim_duration(
            self, duration_sec: float,
            wall_stall_timeout_sec: float = 90.0) -> bool:
        if duration_sec <= 0.0:
            return True
        start_sim = self._sim_time_sec()
        last_sim = start_sim
        last_progress_wall = time.monotonic()
        while rclpy.ok() and self._sim_time_sec() - start_sim < duration_sec:
            now_sim = self._sim_time_sec()
            if now_sim > last_sim + 1.0e-6:
                last_sim = now_sim
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > wall_stall_timeout_sec:
                self.get_logger().error(
                    '  [SIM] Simulation clock stalled while waiting for motion.')
                return False
            time.sleep(0.02)
        return rclpy.ok()

    def _wait_for_sim_arm_target(
            self, timeout_sim_sec: float | None = None,
            tolerance_rad: float | None = None,
            contact_lever_delta_rad: float | None = None) -> bool:
        tolerance = (
            self._sim_arm_joint_tolerance
            if tolerance_rad is None else max(0.0, float(tolerance_rad)))
        targets = dict(zip(
            PIPER_JOINT_NAMES, self._sim_last_arm_target.tolist()))
        timeout = max(
            self._sim_arm_target_timeout_sec,
            float(timeout_sim_sec or 0.0))
        start_sim = self._sim_time_sec()
        last_sim = start_sim
        next_republish_sim = start_sim
        last_progress_wall = time.monotonic()
        max_error = float('inf')
        while rclpy.ok() and self._sim_time_sec() - start_sim < timeout:
            now_sim = self._sim_time_sec()
            if now_sim >= next_republish_sim:
                # Position controllers hold their latest setpoint. Repeating
                # the same target also covers ROS/Gazebo discovery at startup
                # without altering the requested PIPER trajectory.
                self._publish_sim_arm_targets(targets)
                next_republish_sim = now_sim + 0.25
            if all(
                    name in self._sim_arm_joint_positions
                    for name in PIPER_JOINT_NAMES):
                max_error = max(
                    abs(self._sim_arm_joint_positions[name] - target)
                    for name, target in targets.items())
                if max_error <= tolerance:
                    self.get_logger().info(
                        '  [PIPER SIM] Joint target reached; '
                        f'max_error={max_error:.3f}rad')
                    return True
            if contact_lever_delta_rad is not None:
                initial_lever = self._sim_lever_initial_position
                current_lever = self._sim_lever_position
                if initial_lever is not None and current_lever is not None:
                    lever_delta = abs(current_lever - initial_lever)
                    if (lever_delta >= contact_lever_delta_rad
                            and max_error
                            <= self._sim_contact_joint_error_tolerance):
                        self.get_logger().info(
                            '  [PIPER SIM] Gripper contact reached; '
                            f'lever_delta={lever_delta:.3f}rad, '
                            f'joint_error={max_error:.3f}rad')
                        return True
            if now_sim > last_sim + 1.0e-6:
                last_sim = now_sim
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 90.0:
                break
            time.sleep(0.02)
        self.get_logger().error(
            '  [PIPER SIM] Joint trajectory did not converge; '
            f'max_error={max_error:.3f}rad, tolerance={tolerance:.3f}rad')
        if all(
                name in self._sim_arm_joint_positions
                for name in PIPER_JOINT_NAMES):
            residuals = ', '.join(
                f'{name}={self._sim_arm_joint_positions[name] - target:+.3f}'
                for name, target in targets.items())
            self.get_logger().error(
                f'  [PIPER SIM] Joint residuals (actual-target): {residuals}')
        return False

    def _wait_for_sim_door_target(self, target: float) -> bool:
        start_sim = self._sim_time_sec()
        last_sim = start_sim
        last_progress_wall = time.monotonic()
        while self._sim_time_sec() - start_sim < self._sim_feedback_timeout:
            position = self._sim_door_position
            if (
                    position is not None
                    and abs(position - target) <= self._sim_door_tolerance):
                self.get_logger().info(
                    f'  [SIM] Door feedback verified: '
                    f'{position:.3f} rad')
                return True
            now_sim = self._sim_time_sec()
            if now_sim > last_sim + 1.0e-6:
                last_sim = now_sim
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 90.0:
                break
            time.sleep(0.05)

        self.get_logger().error(
            '  [SIM] Door hinge did not reach target; '
            f'last={self._sim_door_position}, target={target:.3f}, '
                f'tolerance={self._sim_door_tolerance:.3f}')
        return False

    def _wait_for_sim_door_contact_motion(self, min_delta: float) -> bool:
        initial = self._sim_door_initial_position
        if initial is None:
            initial = self._sim_door_position
        if initial is None:
            initial = 0.0

        start_sim = self._sim_time_sec()
        last_sim = start_sim
        last_progress_wall = time.monotonic()
        best_delta = 0.0
        while self._sim_time_sec() - start_sim < self._sim_feedback_timeout:
            position = self._sim_door_position
            if position is not None:
                best_delta = max(best_delta, abs(position - initial))
                if best_delta >= min_delta:
                    self.get_logger().info(
                        f'  [SIM] Contact-only door motion verified: '
                        f'initial={initial:.3f} rad, current={position:.3f} rad, '
                        f'delta={best_delta:.3f} rad')
                    return True
            now_sim = self._sim_time_sec()
            if now_sim > last_sim + 1.0e-6:
                last_sim = now_sim
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 90.0:
                break
            time.sleep(0.05)

        self.get_logger().error(
            '  [SIM] Contact-only door motion did not reach threshold; '
            f'initial={initial:.3f}, last={self._sim_door_position}, '
            f'best_delta={best_delta:.3f}, required={min_delta:.3f} rad')
        return False

    def _wait_for_sim_lever_contact_motion(self, min_delta: float) -> bool:
        initial = self._sim_lever_initial_position
        if initial is None:
            initial = self._sim_lever_position
        if initial is None:
            initial = 0.0

        start_sim = self._sim_time_sec()
        last_sim = start_sim
        last_progress_wall = time.monotonic()
        best_delta = 0.0
        targets = self._sim_push_hold_targets
        while self._sim_time_sec() - start_sim < self._sim_lever_press_timeout:
            if targets is not None:
                self._publish_sim_arm_targets(targets)
            position = self._sim_lever_position
            if position is not None:
                best_delta = max(best_delta, abs(position - initial))
                if best_delta >= min_delta:
                    self.get_logger().info(
                        '  [SIM] Physical lever press verified: '
                        f'initial={initial:.3f} rad, current={position:.3f} rad, '
                        f'delta={best_delta:.3f} rad')
                    return True
            now_sim = self._sim_time_sec()
            if now_sim > last_sim + 1.0e-6:
                last_sim = now_sim
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 90.0:
                break
            time.sleep(0.02)

        self.get_logger().error(
            '  [SIM] Lever did not move under gripper contact; '
            f'initial={initial:.3f}, last={self._sim_lever_position}, '
            f'best_delta={best_delta:.3f}, required={min_delta:.3f} rad')
        return False

    def _run_sim_push_contact_follow(self, min_delta: float) -> bool:
        if getattr(self, '_hold_handle_during_base_open', False):
            return self._feedback_handle_held_open(min_delta)
        initial = self._sim_door_initial_position
        if initial is None:
            initial = self._sim_door_position
        if initial is None:
            initial = 0.0

        self.get_logger().info(
            '  [SIM] Holding the pressed lever pose while the mobile base '
            'pushes the door open.')
        start_sim = self._sim_time_sec()
        deadline_sim = start_sim + self._sim_feedback_timeout
        last_sim = start_sim
        last_progress_wall = time.monotonic()
        best_delta = 0.0
        contact_start_position = self._sim_door_position
        contact_reacquired = False
        hold_targets = self._sim_push_hold_targets
        if hold_targets is None and not self._sim_release_handle_before_base_push:
            hold_targets = dict(zip(
                PIPER_JOINT_NAMES, self._sim_last_arm_target.tolist()))
        control_mode = ''
        try:
            while self._sim_time_sec() < deadline_sim:
                if self._feedback_contact and not self._feedback_fresh():
                    return self._feedback_stop('Lost joint/door feedback during base push')
                if hold_targets is not None:
                    self._publish_sim_arm_targets(hold_targets)
                position = self._sim_door_position
                progress = best_delta
                if position is not None:
                    best_delta = max(best_delta, abs(position - initial))
                    progress = abs(position - initial)
                    if best_delta >= min_delta:
                        self.get_logger().info(
                            '  [SIM] Contact follow verified: '
                            f'initial={initial:.3f} rad, '
                                f'current={position:.3f} rad, '
                                f'delta={best_delta:.3f} rad')
                        return True

                # Retraction before stowing leaves a deliberate gap to the
                # panel. Reacquire it in a straight line and use measured
                # hinge motion, rather than elapsed time or a world pose, as
                # the contact signal. Otherwise an already-cracked door makes
                # the chassis start its arc while it is still in free space.
                contact_delta = 0.0
                if position is not None and contact_start_position is not None:
                    contact_delta = abs(position - contact_start_position)
                    contact_reacquired = contact_reacquired or contact_delta >= 0.015

                # Once contact is measured, the chassis centre follows the
                # circular sweep of the edge-hinged panel. Delaying the turn
                # after contact drives the rectangular J100 into the hinge
                # post as the door approaches 90 degrees.
                if not contact_reacquired:
                    mode = 'straight_reacquire_contact'
                    linear_x = self._sim_push_follow_linear_x
                    angular_z = 0.0
                else:
                    mode = 'hinge_follow_arc'
                    linear_x = self._sim_push_follow_linear_x * (
                        0.82 if progress < 1.65 else 0.65)
                    angular_z = math.copysign(
                        linear_x / self._sim_push_follow_radius,
                        self._sim_push_follow_angular_z)
                if mode != control_mode:
                    control_mode = mode
                    self.get_logger().info(
                        '  [SIM] Chassis contact controller: '
                        f'mode={mode}, door_delta={progress:.3f}rad, '
                        f'contact_delta={contact_delta:.3f}rad, '
                        f'v={linear_x:.3f}m/s, w={angular_z:.3f}rad/s')
                self._publish_sim_cmd_vel(linear_x, angular_z)
                now_sim = self._sim_time_sec()
                if now_sim > last_sim + 1.0e-6:
                    last_sim = now_sim
                    last_progress_wall = time.monotonic()
                elif time.monotonic() - last_progress_wall > 90.0:
                    return False
                time.sleep(0.05)
        finally:
            self._publish_sim_cmd_vel(0.0, 0.0)

        self.get_logger().error(
            '  [SIM] Contact-follow door motion did not reach threshold; '
            f'initial={initial:.3f}, last={self._sim_door_position}, '
            f'best_delta={best_delta:.3f}, required={min_delta:.3f} rad')
        return False

    def _command_sim_arm_pose(self, stage: str,
                              handle_pos: PointStamped | None):
        if not self._sim_arm_motion_enabled:
            return False

        targets = self._sim_arm_targets_for_stage(stage, handle_pos)
        if targets is None:
            return False

        return self._command_sim_arm_targets(stage, targets)

    def _command_sim_arm_targets(
            self, stage: str, targets: dict[str, float]) -> bool:
        if not self._sim_arm_motion_enabled:
            return False

        pretty = ', '.join(
            f'{name}={value:.2f}' for name, value in targets.items())
        self.get_logger().info(
            f'  [SIM] Arm target stage={stage}: {pretty}')
        # The convergence loop republishes this target while it waits. Keep
        # home and other non-IK stages synchronized too, otherwise an older
        # contact pose can overwrite the new command on the next cycle.
        self._sim_last_arm_target = np.asarray(
            [targets[name] for name in PIPER_JOINT_NAMES], dtype=float)
        if self._sim_command_transport == 'trajectory':
            self._publish_piper_arm_trajectory(
                targets, max(0.25, self._sim_step_sec))
            return True
        for joint_name, target in targets.items():
            self._publish_sim_command(
                f'{self._sim_arm_topic_prefix}/{joint_name}_cmd',
                target,
                wait=False)
        return True

    def _move_sim_arm_to_stow(self) -> bool:
        """Fold the physical PIPER chain without sweeping through the frame."""
        targets = {
            name: self._sim_arm_joint_positions.get(
                name, float(self._sim_last_arm_target[index]))
            for index, name in enumerate(PIPER_JOINT_NAMES)
        }
        # These are robot joint-space recovery waypoints, not door/world
        # coordinates. First swing the released gripper away from the door,
        # then raise the elbow before folding the wrist over the chassis.
        waypoints = (
            # A few degrees of residual yaw can remain while the open
            # gripper clears the lever. The following elbow-lift waypoint
            # removes that contact, so do not reject this safe intermediate
            # pose using the tighter final-stow tolerance.
            ('stow_swing_clear', {'joint1': 0.0}, 0.075),
            # The first shoulder step lifts the open jaws clear. Tuck the
            # elbow at that height before folding the wrist; a deeper shoulder
            # step alone intersects the J100's upper structure.
            ('stow_shoulder_clear', {'joint2': 1.40}, 0.060),
            ('stow_tuck_elbow', {'joint3': -1.0}, 0.060),
            ('stow_fold_wrist', {
                'joint4': 0.0, 'joint5': 1.0, 'joint6': 0.0}, 0.060),
            ('stow_lower_shoulder', {'joint2': 1.0}, 0.060),
            ('stow', dict(zip(
                PIPER_JOINT_NAMES, PIPER_STOW.tolist())), 0.060),
        )
        if self._feedback_contact and self._hold_handle_during_base_open:
            # In the supplied chain, reducing joint2 raises and retracts the
            # hand. The legacy 1.40-rad step lowers it toward the open panel.
            waypoints = (
                ('stow_lift_shoulder', {'joint2': 0.0}, 0.060),
                ('stow_swing_clear', {'joint1': 0.0}, 0.060),
                ('stow_tuck_elbow', {'joint3': -1.0}, 0.060),
                ('stow_fold_wrist', {
                    'joint4': 0.0, 'joint5': 1.0, 'joint6': 0.0}, 0.060),
                ('stow', dict(zip(PIPER_JOINT_NAMES, PIPER_STOW.tolist())), 0.060),
            )
        for stage, updates, tolerance in waypoints:
            targets.update(updates)
            if not self._command_sim_arm_targets(stage, dict(targets)):
                return False
            if not self._wait_for_sim_arm_target(tolerance_rad=tolerance):
                self.get_logger().error(
                    f'  [PIPER SIM] Recovery target not reached at {stage}')
                return False
        return True

    def _publish_sim_arm_targets(self, targets: dict[str, float]):
        if not self._sim_arm_motion_enabled:
            return
        if self._sim_command_transport == 'trajectory':
            self._publish_piper_arm_trajectory(targets, 0.25)
            return
        for joint_name, target in targets.items():
            self._publish_sim_command(
                f'{self._sim_arm_topic_prefix}/{joint_name}_cmd',
                target,
                wait=False)

    def _publish_sim_cmd_vel(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._sim_cmd_vel_pub.publish(msg)

    def _run_sim_base_motion(
            self, linear_x: float, angular_z: float,
            duration_sec: float):
        start_sim = self._sim_time_sec()
        try:
            while self._sim_time_sec() - start_sim < duration_sec:
                self._publish_sim_cmd_vel(linear_x, angular_z)
                time.sleep(0.05)
        finally:
            self._publish_sim_cmd_vel(0.0, 0.0)

    def _run_sim_latch_clear_push(self) -> bool:
        """Crack the door open while the PIPER still depresses the lever."""
        if self._feedback_contact:
            return self._feedback_latch_clear_push()
        initial = self._sim_door_initial_position
        if initial is None:
            initial = self._sim_door_position
        if initial is None:
            return False

        start_sim = self._sim_time_sec()
        best_delta = 0.0
        try:
            while (
                    rclpy.ok()
                    and self._sim_time_sec() - start_sim
                    < self._sim_latch_clear_timeout):
                linear_x = min(
                    0.055, max(0.03, self._sim_push_follow_linear_x))
                current = self._sim_door_position
                if current is not None:
                    best_delta = max(best_delta, abs(current - initial))
                    if best_delta >= self._sim_latch_clear_min_angle:
                        self.get_logger().info(
                            '  [SIM] Latch clearance verified from physical '
                            f'door motion: delta={best_delta:.3f}rad')
                        return True
                angular_z = 0.0
                if best_delta >= 0.08:
                    angular_z = math.copysign(
                        linear_x / self._sim_push_follow_radius,
                        self._sim_push_follow_angular_z)
                self._publish_sim_cmd_vel(linear_x, angular_z)
                time.sleep(0.05)
        finally:
            self._publish_sim_cmd_vel(0.0, 0.0)
        self.get_logger().error(
            '  [SIM] Latch-clear push below threshold: '
            f'best_delta={best_delta:.3f}rad, '
            f'required={self._sim_latch_clear_min_angle:.3f}rad')
        return False

    def _post_open_backoff(self):
        if not self._sim_mode:
            self.get_logger().warn(
                'Post-open backoff is disabled outside simulation until '
                'mobile base safety is verified on hardware.')
            return
        if self._post_open_backoff_sec <= 0.0:
            return
        self._run_sim_base_motion(
            self._post_open_backoff_linear_x,
            0.0,
            self._post_open_backoff_sec)

    def _sim_arm_targets_for_stage(
            self,
            stage: str,
            handle_pos: PointStamped | None) -> dict[str, float] | None:
        if stage == 'home':
            return dict(zip(PIPER_JOINT_NAMES, PIPER_HOME.tolist()))
        if stage == 'stow':
            return dict(zip(PIPER_JOINT_NAMES, PIPER_STOW.tolist()))
        if handle_pos is None:
            return None

        px = float(handle_pos.point.x)
        py = float(handle_pos.point.y)
        pz = float(handle_pos.point.z)
        if stage == 'pre_grasp':
            px -= PRE_GRASP_OFFSET
        elif stage == 'clearance_retract':
            px -= self._sim_arm_clearance_retract
        elif stage == 'pull':
            px += self._handle_surface_inset - PULL_DISTANCE
        elif stage == 'push':
            px += self._handle_surface_inset + min(0.10, PULL_DISTANCE)
        elif stage in ('grasp', 'press'):
            px += self._handle_surface_inset
        else:
            return None

        return self._sim_arm_ik_targets(px, py, pz)

    def _sim_arm_ik_targets(
            self, px: float, py: float,
            pz: float) -> dict[str, float] | None:
        """Solve on the supplied PIPER chain without changing link lengths."""
        seed = [
            self._sim_arm_joint_positions.get(name, value)
            for name, value in zip(PIPER_JOINT_NAMES, self._sim_last_arm_target)]
        result = self._piper_kinematics.solve((px, py, pz), seed=seed)
        if result is None:
            self.get_logger().error(
                '  [PIPER SIM] Observed handle target is outside the real '
                f'PIPER workspace: ({px:.3f}, {py:.3f}, {pz:.3f})')
            return None
        self._sim_last_arm_target = result.positions.copy()
        self.get_logger().info(
            '  [PIPER SIM] Actual-chain IK '
            f'error={result.position_error_m * 1000.0:.1f}mm, '
            f'orientation_error={result.orientation_error_rad:.3f}rad, '
            f'iterations={result.iterations}')
        return dict(zip(PIPER_JOINT_NAMES, result.positions.tolist()))

    def _publish_piper_arm_trajectory(
            self, targets: dict[str, float], duration_sec: float):
        trajectory = JointTrajectory()
        trajectory.joint_names = list(PIPER_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = [float(targets[name]) for name in PIPER_JOINT_NAMES]
        point.time_from_start = Duration(
            seconds=max(0.05, float(duration_sec))).to_msg()
        trajectory.points = [point]
        self._sim_arm_trajectory_pub.publish(trajectory)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))

    def _command_sim_gripper(self, width: float):
        if not self._sim_arm_motion_enabled:
            return
        left = max(0.0, min(0.035, float(width) / 2.0))
        right = -left
        if self._sim_command_transport == 'trajectory':
            trajectory = JointTrajectory()
            trajectory.joint_names = [
                'gripper_left_joint', 'gripper_right_joint']
            point = JointTrajectoryPoint()
            point.positions = [left, right]
            point.time_from_start = Duration(seconds=0.45).to_msg()
            trajectory.points = [point]
            self._sim_gripper_trajectory_pub.publish(trajectory)
            return
        self._publish_sim_command(
            f'{self._sim_arm_topic_prefix}/gripper_left_joint_cmd',
            left,
            wait=False)
        self._publish_sim_command(
            f'{self._sim_arm_topic_prefix}/gripper_right_joint_cmd',
            right,
            wait=False)

    def _wait_for_sim_gripper_release(self) -> bool:
        """Require measured finger clearance before moving away from a lever."""
        left = right = None
        start_sim = self._sim_time_sec()
        last_sim = start_sim
        last_progress_wall = time.monotonic()
        next_sample = start_sim
        while (
                rclpy.ok()
                and self._sim_time_sec() - start_sim
                < self._sim_gripper_release_timeout):
            left = self._sim_arm_joint_positions.get('gripper_left_joint')
            right = self._sim_arm_joint_positions.get('gripper_right_joint')
            if self._feedback_contact:
                if not self._feedback_fresh():
                    return self._feedback_stop('Lost feedback during gripper release')
                if self._sim_time_sec() >= next_sample:
                    self._feedback_event('gripper_release_sample',
                                         left_m=left, right_m=right,
                                         required_half_width_m=self._sim_gripper_release_half_width)
                    next_sample = self._sim_time_sec() + .1
            if (left is not None and right is not None
                    and left >= self._sim_gripper_release_half_width
                    and right <= -self._sim_gripper_release_half_width):
                self.get_logger().info(
                    '  [PIPER SIM] Measured gripper release: '
                    f'left={left:.3f}m, right={right:.3f}m')
                return True
            now_sim = self._sim_time_sec()
            if now_sim > last_sim + 1.0e-6:
                last_sim = now_sim
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > 90.0:
                return False
            time.sleep(0.02)
        if self._feedback_contact:
            return self._feedback_stop(
                f'Gripper release aperture not reached: left={left}, right={right}, '
                f'required=+/-{self._sim_gripper_release_half_width}')
        return False

    def _send_piper_pose(self, target_pose: Pose,
                         gripper_width: float) -> bool:
        if self._piper_pos_pub is None or self._piper_enable_pub is None:
            return False

        enable_msg = Bool()
        enable_msg.data = True
        self._piper_enable_pub.publish(enable_msg)

        cmd = PosCmd()
        cmd.x = float(target_pose.position.x)
        cmd.y = float(target_pose.position.y)
        cmd.z = float(target_pose.position.z)
        cmd.roll = float(self._piper_roll)
        cmd.pitch = float(self._piper_pitch)
        cmd.yaw = float(self._piper_yaw)
        cmd.gripper = float(gripper_width)
        cmd.mode1 = int(self._piper_mode1)
        cmd.mode2 = int(self._piper_mode2)

        self._piper_pos_pub.publish(cmd)
        self.get_logger().info(
            f'  [PIPER SDK] pos_cmd frame={self._manipulation_frame} '
            f'x={cmd.x:.3f}, y={cmd.y:.3f}, z={cmd.z:.3f}, '
            f'gripper={cmd.gripper:.3f}')
        time.sleep(float(self._piper_settle_sec))
        return True

    # ── MoveIt2 유틸 ─────────────────────────────────────
    def _plan_and_execute_cartesian(self, target_pose: Pose) -> bool:
        if self._arm is None:
            return False
        # MoveIt2 set_goal_state는 PoseStamped를 요구함
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self._manipulation_frame
        pose_stamped.header.stamp    = self.get_clock().now().to_msg()
        pose_stamped.pose            = target_pose
        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(pose_stamped_msg=pose_stamped,
                                  pose_link='arm_link6')
        plan_result = self._arm.plan()
        if not plan_result:
            self.get_logger().warn('MoveIt2 planning failed')
            return False
        self._moveit.execute(plan_result.trajectory, controllers=[])
        return True

    def _set_gripper(self, width: float) -> bool:
        if self._gripper is None:
            return False
        self._gripper.set_start_state_to_current_state()
        self._gripper.set_goal_state(
            configuration_name='open' if width > 0.05 else 'close')
        plan_result = self._gripper.plan()
        if not plan_result:
            return False
        self._moveit.execute(plan_result.trajectory, controllers=[])
        return True

    @staticmethod
    def _make_pose(x: float, y: float, z: float) -> Pose:
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = ManipulationNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
