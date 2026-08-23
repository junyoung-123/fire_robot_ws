"""
manipulation_node.py

MoveIt2(moveit_py) 기반 PIPER 로봇팔 문 개방 노드.

동작 시퀀스:
  1. Pre-grasp  : 손잡이 앞 10cm 위치로 팔 이동
  2. Grasp      : 손잡이 위치로 직선 이동 + 그리퍼 닫기
  3. Push/Pull  : 손잡이를 밀거나 당겨 힌지 문 개방 (기본 Push)
  4. Home       : 기본 자세 복귀

의존 패키지: moveit_py (MoveIt2 Python bindings)
실제 하드웨어 연동 시 MoveItPy 초기화 블록의 주석을 해제하세요.
"""

import time
import math
import re
import subprocess
from pathlib import Path

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
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener

from fire_robot_interfaces.msg import DoorInfo
from fire_robot_interfaces.srv import OpenDoor

try:
    from moveit.planning import MoveItPy
    from moveit.core.robot_state import RobotState
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


class ManipulationNode(Node):
    """PIPER 로봇팔 제어 및 문 개방 서비스 노드"""

    def __init__(self):
        super().__init__('manipulation_node')

        self.declare_parameter('planning_group', 'piper_arm')
        self.declare_parameter('gripper_group', 'piper_gripper')
        self.declare_parameter('velocity_scaling', 0.3)
        self.declare_parameter('sim_mode', True)
        self.declare_parameter('sim_control_enabled', False)
        self.declare_parameter('sim_command_transport', 'ign')
        self.declare_parameter('sim_step_sec', 1.0)
        self.declare_parameter('sim_return_home', True)
        self.declare_parameter('door_open_motion', 'push')
        self.declare_parameter('press_handle_before_push', True)
        self.declare_parameter(
            'lever_press_distance_m', LEVER_PRESS_DISTANCE)
        self.declare_parameter('allow_sim_fallback', False)
        self.declare_parameter('sim_physical_door_opening', False)
        self.declare_parameter('sim_door_physics_required', False)
        self.declare_parameter('sim_door_contact_only', False)
        self.declare_parameter('sim_door_contact_min_angle_rad', 0.35)
        self.declare_parameter('sim_door_open_angle_rad', 2.09439510239)
        self.declare_parameter('sim_door_command_steps', 5)
        self.declare_parameter('sim_door_max_match_distance_m', 1.6)
        self.declare_parameter('sim_door_axis_match_max_progress_m', 3.1)
        self.declare_parameter('sim_door_axis_match_max_lateral_m', 1.45)
        self.declare_parameter(
            'sim_door_topic_prefix', '/fire_robot/door/blue')
        self.declare_parameter('sim_door_command_topic', '')
        self.declare_parameter('sim_door_command_sign', 1.0)
        self.declare_parameter('sim_door_world_file', '')
        self.declare_parameter('sim_arm_motion_enabled', False)
        self.declare_parameter('sim_arm_topic_prefix', '/fire_robot/sim_arm')
        self.declare_parameter('sim_verify_door', False)
        self.declare_parameter('sim_door_feedback_topic', '/door_joint_states')
        self.declare_parameter('sim_door_feedback_joint_name', 'door_hinge')
        self.declare_parameter('sim_door_tolerance_rad', 0.08)
        self.declare_parameter('sim_feedback_timeout_sec', 4.0)
        self.declare_parameter('sim_push_follow_enabled', False)
        self.declare_parameter('sim_push_follow_cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('sim_push_follow_linear_x', 0.13)
        self.declare_parameter('sim_push_follow_angular_z', -0.13)
        self.declare_parameter('sim_push_follow_stage_sec', 1.7)
        self.declare_parameter('post_open_backoff_enabled', False)
        self.declare_parameter('post_open_backoff_linear_x', -0.05)
        self.declare_parameter('post_open_backoff_sec', 1.2)
        self.declare_parameter('post_open_backoff_detach_first', True)
        self.declare_parameter(
            'sim_handle_detach_topic', '/fire_robot/sim_arm/handle_detach')
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

        self._planning_group = self.get_parameter('planning_group').value
        self._gripper_group  = self.get_parameter('gripper_group').value
        self._vel_scale      = self.get_parameter('velocity_scaling').value
        self._sim_mode       = self.get_parameter('sim_mode').value
        self._sim_control_enabled = bool(
            self.get_parameter('sim_control_enabled').value)
        self._sim_command_transport = str(
            self.get_parameter('sim_command_transport').value).strip().lower()
        if self._sim_command_transport not in ('ign', 'ros'):
            self.get_logger().warn(
                f"Unknown sim_command_transport "
                f"'{self._sim_command_transport}', falling back to 'ign'.")
            self._sim_command_transport = 'ign'
        self._sim_step_sec = max(
            0.0, float(self.get_parameter('sim_step_sec').value))
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
        self._sim_door_topic_prefix = str(
            self.get_parameter('sim_door_topic_prefix').value).rstrip('/')
        self._sim_door_command_topic = str(
            self.get_parameter('sim_door_command_topic').value).strip()
        self._sim_door_command_sign = float(
            self.get_parameter('sim_door_command_sign').value)
        self._sim_door_world_file = str(
            self.get_parameter('sim_door_world_file').value)
        self._sim_arm_motion_enabled = bool(
            self.get_parameter('sim_arm_motion_enabled').value
        ) or self._sim_control_enabled
        self._sim_arm_topic_prefix = str(
            self.get_parameter('sim_arm_topic_prefix').value).rstrip('/')
        self._sim_verify_door_feedback = bool(
            self.get_parameter('sim_verify_door').value)
        self._sim_door_feedback_topic = str(
            self.get_parameter('sim_door_feedback_topic').value)
        self._sim_door_feedback_joint_name = str(
            self.get_parameter('sim_door_feedback_joint_name').value)
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
        self._sim_push_follow_stage_sec = max(
            0.1, float(self.get_parameter('sim_push_follow_stage_sec').value))
        self._post_open_backoff_enabled = bool(
            self.get_parameter('post_open_backoff_enabled').value)
        self._post_open_backoff_linear_x = min(
            0.0,
            max(-0.05, float(
                self.get_parameter('post_open_backoff_linear_x').value)))
        self._post_open_backoff_sec = max(
            0.0, float(self.get_parameter('post_open_backoff_sec').value))
        self._post_open_backoff_detach_first = bool(
            self.get_parameter('post_open_backoff_detach_first').value)
        self._sim_handle_detach_topic = str(
            self.get_parameter('sim_handle_detach_topic').value)
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
        self._sim_cmd_vel_pub = self.create_publisher(
            Twist, self._sim_push_follow_cmd_vel_topic, 10)
        self._sim_ros_float_pubs: dict[str, object] = {}
        self._sim_door_position: float | None = None
        self._sim_door_initial_position: float | None = None
        if self._sim_mode and self._sim_verify_door_feedback:
            self.create_subscription(
                JointState,
                self._sim_door_feedback_topic,
                self._sim_door_state_callback,
                10)

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

        if self._require_detected_handle and not handle_detected:
            response.success = False
            response.message = 'Detected handle is required before opening'
            self.get_logger().error(response.message)
            done_msg = Bool()
            done_msg.data = False
            self.manip_done_pub.publish(done_msg)
            return response
        if self._require_yolo_handle and not handle_method.startswith('yolo'):
            response.success = False
            response.message = 'YOLO handle detection is required before opening'
            self.get_logger().error(response.message)
            done_msg = Bool()
            done_msg.data = False
            self.manip_done_pub.publish(done_msg)
            return response

        success = self._execute_door_open_sequence(
            request.handle_position, request.door_id)

        response.success = success
        response.message = ('Door opened successfully'
                            if success else 'Failed to open door')

        done_msg = Bool()
        done_msg.data = success
        self.manip_done_pub.publish(done_msg)
        return response

    # ── 문 개방 시퀀스 ────────────────────────────────────
    def _execute_door_open_sequence(self,
                                    handle_position: PointStamped,
                                    door_id: str = '') -> bool:
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
            self._sim_door_initial_position = self._sim_door_position

        self._set_manip_phase('PRE_GRASP', door_id)
        self.get_logger().info('Step 1/4: Moving to pre-grasp position')
        if not self._move_to_pre_grasp(arm_handle_position):
            self.get_logger().error('Pre-grasp failed')
            self._set_manip_phase('FAILED_PRE_GRASP', door_id)
            return False

        self._set_manip_phase('GRASP_HANDLE', door_id)
        self.get_logger().info('Step 2/4: Grasping handle')
        if not self._grasp_handle(arm_handle_position):
            self.get_logger().error('Grasp failed')
            self._set_manip_phase('FAILED_GRASP_HANDLE', door_id)
            return False

        do_lever_press = (
            self._door_open_motion == 'push'
            and self._press_handle_before_push
            and self._lever_press_distance > 0.0)
        if do_lever_press:
            self._set_manip_phase('PRESS_HANDLE', door_id)
            self.get_logger().info('Step 3/5: Pressing lever handle')
            if not self._press_handle(arm_handle_position):
                self.get_logger().error('Lever press failed')
                self._set_manip_phase('FAILED_PRESS_HANDLE', door_id)
                return False

        open_phase = (
            'PULL_OPEN'
            if self._door_open_motion == 'pull'
            else 'PUSH_OPEN')
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

        self._set_manip_phase('RETURN_HOME', door_id)
        self.get_logger().info(
            f"Step {'5/5' if do_lever_press else '4/4'}: "
            'Returning to home position')
        self._move_to_home()
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
            if self._sim_mode:
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
            if self._sim_mode:
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
            if self._sim_mode:
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
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Pre-grasp at '
                f'({handle_pos.point.x - PRE_GRASP_OFFSET:.3f}, '
                f'{handle_pos.point.y:.3f}, '
                f'{handle_pos.point.z:.3f})')
            self._command_sim_gripper(GRIPPER_OPEN)
            self._command_sim_arm_pose('pre_grasp', handle_pos)
            time.sleep(self._sim_step_sec)
            return True

        target = self._make_pose(
            x=handle_pos.point.x - PRE_GRASP_OFFSET,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_OPEN)
        return self._plan_and_execute_cartesian(target)

    def _grasp_handle(self, handle_pos: PointStamped) -> bool:
        if self._sim_mode:
            self.get_logger().info('  [SIM] Gripper closing')
            self._command_sim_arm_pose('grasp', handle_pos)
            close_width = 0.0 if self._sim_door_contact_only else GRIPPER_CLOSE
            self._command_sim_gripper(close_width)
            time.sleep(self._sim_step_sec)
            return True

        # 손잡이 위치로 직선 이동
        target = self._make_pose(
            x=handle_pos.point.x,
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

    def _pressed_handle_position(
            self, handle_pos: PointStamped) -> PointStamped:
        pressed = PointStamped()
        pressed.header = handle_pos.header
        pressed.point.x = handle_pos.point.x
        pressed.point.y = handle_pos.point.y
        pressed.point.z = handle_pos.point.z - self._lever_press_distance
        return pressed

    def _press_handle(self, handle_pos: PointStamped) -> bool:
        pressed = self._pressed_handle_position(handle_pos)
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Pressing lever down '
                f'{self._lever_press_distance:.3f}m before push')
            self._command_sim_arm_pose('press', pressed)
            time.sleep(self._sim_step_sec)
            return True

        target = self._make_pose(
            x=pressed.point.x,
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
            time.sleep(self._sim_step_sec)
            return True

        # The manipulation frame uses x forward and z up. Pulling moves the
        # handle back along -x toward the robot.
        target = self._make_pose(
            x=handle_pos.point.x - PULL_DISTANCE,
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
                f'  [SIM] Pushing handle and opening hinged Gazebo door '
                f'{self._sim_door_open_angle:.2f}rad')
            if (
                    self._sim_door_contact_only
                    and self._sim_door_initial_position is None):
                self._sim_door_initial_position = self._sim_door_position
            self._command_sim_arm_pose('push', handle_pos)
            door_match_handle = door_frame_handle_pos or handle_pos
            if not self._open_sim_gazebo_door(door_match_handle, door_id):
                return False
            time.sleep(self._sim_step_sec)
            return True

        # The manipulation frame uses x forward and z up. A push-open door
        # should move the handle farther along +x, away from the robot body.
        target = self._make_pose(
            x=handle_pos.point.x + PULL_DISTANCE,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_CLOSE)
        return self._plan_and_execute_cartesian(target)

    def _move_to_home(self):
        if self._sim_mode:
            self.get_logger().info('  [SIM] Returning to home position')
            self._command_sim_gripper(GRIPPER_OPEN)
            if self._sim_return_home:
                self._command_sim_arm_pose('home', None)
            time.sleep(self._sim_step_sec)
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
            return True

        if self._sim_door_contact_only:
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
            open_sign = self._sim_door_command_sign
            self.get_logger().info(
                f'  [SIM] Using explicit Gazebo door command topic={topic}')
        else:
            selected = self._select_sim_door_topic(handle_pos, door_id)
            if selected is None:
                msg = (
                    'No Gazebo hinged blue-door command topic matched this '
                    'open request.')
                if self._sim_door_physics_required:
                    self.get_logger().error(msg)
                    return False
                self.get_logger().warn(
                    msg + ' Falling back to logical sim open.')
                return True

            topic, _door_y, open_sign = selected

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
            time.sleep(0.12)

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

        candidates: list[tuple[float, str, float, float, float, float, float]] = []
        hx = float(handle_pos.point.x)
        hy = float(handle_pos.point.y)

        for door in self._sim_hinged_doors:
            door_x = door['x']
            door_y = door['y']
            open_sign = door['open_sign']
            topic = door['topic']
            if topic in self._sim_opened_door_topics:
                continue
            score = (hx - door_x) ** 2 + 0.35 * (hy - door_y) ** 2
            weighted_distance = math.sqrt(score)
            progress_gap = abs(hx - door_x)
            lateral_gap = abs(hy - door_y)
            candidates.append((
                score, topic, door_y, open_sign, weighted_distance,
                progress_gap, lateral_gap))

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
            best_progress_gap, best_lateral_gap) = candidates[0]
        if (
                self._sim_door_max_match_distance > 0.0
                and best_distance > self._sim_door_max_match_distance):
            same_wall_side = (
                abs(hy) < 0.20
                or abs(best_y) < 0.20
                or hy * best_y > 0.0)
            axis_consistent = (
                same_wall_side
                and self._sim_door_axis_match_max_progress > 0.0
                and self._sim_door_axis_match_max_lateral > 0.0
                and best_progress_gap <= self._sim_door_axis_match_max_progress
                and best_lateral_gap <= self._sim_door_axis_match_max_lateral)
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
                    f'lateral_gap={best_lateral_gap:.2f}m).')
                return None
        self.get_logger().info(
            f'  [SIM] Matched observed door {door_id or "(unknown)"} to '
            f'{best_topic} (distance={best_distance:.2f}m, '
            f'score={best_score:.3f})')
        return best_topic, best_y, best_open_sign

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

    def _publish_ign_empty(self, topic: str) -> bool:
        cmd = [
            'ign', 'topic',
            '-t', topic,
            '-m', 'ignition.msgs.Empty',
            '-p', '',
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0)
        except Exception as e:
            self.get_logger().warn(
                f'  [SIM] Failed to publish Ignition empty topic {topic}: {e}')
            return False
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            self.get_logger().warn(
                f'  [SIM] Ignition empty publish failed for {topic}: {detail}')
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
        try:
            index = msg.name.index(self._sim_door_feedback_joint_name)
            self._sim_door_position = float(msg.position[index])
        except (ValueError, IndexError):
            return

    def _wait_for_sim_door_target(self, target: float) -> bool:
        deadline = time.monotonic() + self._sim_feedback_timeout
        while time.monotonic() < deadline:
            position = self._sim_door_position
            if (
                    position is not None
                    and abs(position - target) <= self._sim_door_tolerance):
                self.get_logger().info(
                    f'  [SIM] Door feedback verified: '
                    f'{position:.3f} rad')
                return True
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

        deadline = time.monotonic() + self._sim_feedback_timeout
        best_delta = 0.0
        while time.monotonic() < deadline:
            position = self._sim_door_position
            if position is not None:
                best_delta = max(best_delta, abs(position - initial))
                if best_delta >= min_delta:
                    self.get_logger().info(
                        f'  [SIM] Contact-only door motion verified: '
                        f'initial={initial:.3f} rad, current={position:.3f} rad, '
                        f'delta={best_delta:.3f} rad')
                    return True
            time.sleep(0.05)

        self.get_logger().error(
            '  [SIM] Contact-only door motion did not reach threshold; '
            f'initial={initial:.3f}, last={self._sim_door_position}, '
            f'best_delta={best_delta:.3f}, required={min_delta:.3f} rad')
        return False

    def _run_sim_push_contact_follow(self, min_delta: float) -> bool:
        initial = self._sim_door_initial_position
        if initial is None:
            initial = self._sim_door_position
        if initial is None:
            initial = 0.0

        self.get_logger().info(
            '  [SIM] Running coordinated arm/base push-follow during '
            'contact-only verification.')
        deadline = time.monotonic() + self._sim_feedback_timeout
        best_delta = 0.0
        stages = [
            (
                {'joint1': 0.45, 'joint2': 1.20, 'joint3': -0.16,
                 'joint4': 0.0, 'joint5': -0.06, 'joint6': 0.0},
                self._sim_push_follow_linear_x,
                self._sim_push_follow_angular_z,
            ),
            (
                {'joint1': 0.62, 'joint2': 1.20, 'joint3': -0.18,
                 'joint4': 0.0, 'joint5': -0.08, 'joint6': 0.0},
                self._sim_push_follow_linear_x,
                self._sim_push_follow_angular_z * 1.05,
            ),
            (
                {'joint1': 0.82, 'joint2': 1.16, 'joint3': -0.16,
                 'joint4': 0.0, 'joint5': -0.08, 'joint6': 0.0},
                self._sim_push_follow_linear_x * 0.95,
                self._sim_push_follow_angular_z * 1.10,
            ),
            (
                {'joint1': 1.02, 'joint2': 1.08, 'joint3': -0.08,
                 'joint4': 0.0, 'joint5': -0.06, 'joint6': 0.0},
                self._sim_push_follow_linear_x * 0.85,
                self._sim_push_follow_angular_z * 1.15,
            ),
            (
                {'joint1': 1.18, 'joint2': 0.98, 'joint3': -0.02,
                 'joint4': 0.0, 'joint5': -0.04, 'joint6': 0.0},
                self._sim_push_follow_linear_x * 0.70,
                self._sim_push_follow_angular_z * 1.20,
            ),
            (
                {'joint1': 1.34, 'joint2': 0.88, 'joint3': 0.04,
                 'joint4': 0.0, 'joint5': -0.02, 'joint6': 0.0},
                self._sim_push_follow_linear_x * 0.62,
                self._sim_push_follow_angular_z * 1.35,
            ),
            (
                {'joint1': 1.46, 'joint2': 0.78, 'joint3': 0.10,
                 'joint4': 0.0, 'joint5': 0.00, 'joint6': 0.0},
                self._sim_push_follow_linear_x * 0.52,
                self._sim_push_follow_angular_z * 1.55,
            ),
            (
                {'joint1': 1.55, 'joint2': 0.70, 'joint3': 0.15,
                 'joint4': 0.0, 'joint5': 0.02, 'joint6': 0.0},
                self._sim_push_follow_linear_x * 0.42,
                self._sim_push_follow_angular_z * 1.70,
            ),
        ]

        try:
            for arm_targets, linear_x, angular_z in stages:
                stage_end = min(
                    deadline,
                    time.monotonic() + self._sim_push_follow_stage_sec)
                while time.monotonic() < stage_end:
                    self._publish_sim_arm_targets(arm_targets)
                    self._publish_sim_cmd_vel(linear_x, angular_z)
                    position = self._sim_door_position
                    if position is not None:
                        best_delta = max(best_delta, abs(position - initial))
                        if best_delta >= min_delta:
                            self.get_logger().info(
                                '  [SIM] Contact follow verified: '
                                f'initial={initial:.3f} rad, '
                                f'current={position:.3f} rad, '
                                f'delta={best_delta:.3f} rad')
                            return True
                    time.sleep(0.05)

            while time.monotonic() < deadline:
                position = self._sim_door_position
                if position is not None:
                    best_delta = max(best_delta, abs(position - initial))
                    if best_delta >= min_delta:
                        self.get_logger().info(
                            '  [SIM] Contact follow verified: '
                            f'initial={initial:.3f} rad, '
                            f'current={position:.3f} rad, '
                            f'delta={best_delta:.3f} rad')
                        return True
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
            return

        targets = self._sim_arm_targets_for_stage(stage, handle_pos)
        if targets is None:
            return

        pretty = ', '.join(
            f'{name}={value:.2f}' for name, value in targets.items())
        self.get_logger().info(
            f'  [SIM] Arm target stage={stage}: {pretty}')
        for joint_name, target in targets.items():
            self._publish_sim_command(
                f'{self._sim_arm_topic_prefix}/{joint_name}_cmd',
                target,
                wait=False)

    def _publish_sim_arm_targets(self, targets: dict[str, float]):
        if not self._sim_arm_motion_enabled:
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

    def _post_open_backoff(self):
        if not self._sim_mode:
            self.get_logger().warn(
                'Post-open backoff is disabled outside simulation until '
                'mobile base safety is verified on hardware.')
            return
        if self._post_open_backoff_sec <= 0.0:
            return
        if self._post_open_backoff_detach_first:
            self._publish_ign_empty(self._sim_handle_detach_topic)

        deadline = time.monotonic() + self._post_open_backoff_sec
        while time.monotonic() < deadline:
            self._publish_sim_cmd_vel(self._post_open_backoff_linear_x, 0.0)
            time.sleep(0.05)
        self._publish_sim_cmd_vel(0.0, 0.0)

    def _sim_arm_targets_for_stage(
            self,
            stage: str,
            handle_pos: PointStamped | None) -> dict[str, float] | None:
        if stage == 'home' or handle_pos is None:
            return {
                'joint1': 0.0,
                'joint2': 0.0,
                'joint3': 0.0,
                'joint4': 0.0,
                'joint5': 0.0,
                'joint6': 0.0,
            }

        px = float(handle_pos.point.x)
        py = float(handle_pos.point.y)
        pz = float(handle_pos.point.z)
        # If TF was unavailable during a standalone service smoke test, map
        # coordinates can leak into this visual-only arm controller. Clamp them
        # to a plausible relative reach so the sim arm still moves sensibly.
        px = self._clamp(px, 0.18, 0.75)
        py = self._clamp(py, -0.65, 0.65)
        pz = self._clamp(pz, 0.55, 1.15)

        yaw = self._clamp(math.atan2(py, max(0.12, px)), -1.35, 1.35)
        radial = math.hypot(px, py)
        extension = self._clamp((radial - 0.22) / 0.42, 0.0, 1.0)
        height = self._clamp((pz - 0.82) / 0.35, -1.0, 1.0)

        stage_extension = {
            'pre_grasp': max(0.0, extension - 0.20),
            'grasp': extension,
            'press': extension,
            'pull': max(0.0, extension - 0.45),
            'push': min(1.0, extension + 0.55),
        }.get(stage)
        if stage_extension is None:
            return None

        # Lightweight FK-calibrated targets for the simplified PIPER xacro.
        # Positive joint2 extends the actual collision geometry forward; the
        # previous visual-only sign looked plausible but left the gripper above
        # the base, which cannot validate contact dynamics.
        joint2 = self._clamp(
            0.25 + 0.66 * stage_extension - 0.08 * height,
            0.20, 1.15)
        joint3 = self._clamp(
            0.26 - 0.42 * stage_extension + 0.08 * height,
            -0.30, 0.45)
        joint5 = self._clamp(
            0.02 - 0.08 * height,
            -0.25, 0.25)

        if stage == 'push':
            yaw = self._clamp(yaw * 1.08, -1.45, 1.45)
            joint2 = self._clamp(joint2 + 0.25, 0.20, 1.20)
            joint5 = self._clamp(joint5 - 0.04, -0.25, 0.25)
        elif stage == 'press':
            joint2 = self._clamp(joint2 + 0.05, 0.20, 1.20)
            joint3 = self._clamp(joint3 + 0.06, -0.30, 0.45)
            joint5 = self._clamp(joint5 + 0.10, -0.25, 0.25)
        elif stage == 'pull':
            joint2 = self._clamp(joint2 - 0.05, 0.20, 1.15)
            joint3 = self._clamp(joint3 + 0.05, -0.30, 0.45)

        return {
            'joint1': yaw,
            'joint2': joint2,
            'joint3': joint3,
            'joint4': 0.0,
            'joint5': joint5,
            'joint6': 0.0,
        }

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))

    def _command_sim_gripper(self, width: float):
        if not self._sim_arm_motion_enabled:
            return
        left = max(0.0, min(0.04, float(width) / 2.0))
        right = left
        self._publish_sim_command(
            f'{self._sim_arm_topic_prefix}/gripper_left_joint_cmd',
            left,
            wait=False)
        self._publish_sim_command(
            f'{self._sim_arm_topic_prefix}/gripper_right_joint_cmd',
            right,
            wait=False)

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
