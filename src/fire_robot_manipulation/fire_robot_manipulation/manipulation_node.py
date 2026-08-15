"""
manipulation_node.py

MoveIt2(moveit_py) 기반 PIPER 로봇팔 문 개방 노드.

동작 시퀀스:
  1. Pre-grasp  : 손잡이 앞 10cm 위치로 팔 이동
  2. Grasp      : 손잡이 위치로 직선 이동 + 그리퍼 닫기
  3. Pull/Push  : 문 열기 궤적 (현재 Pull 기본)
  4. Home       : 기본 자세 복귀

의존 패키지: moveit_py (MoveIt2 Python bindings)
실제 하드웨어 연동 시 MoveItPy 초기화 블록의 주석을 해제하세요.
"""

import time
import math

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Bool, Float64
from sensor_msgs.msg import JointState
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
        self.declare_parameter('sim_step_sec', 1.5)
        self.declare_parameter('sim_return_home', True)
        self.declare_parameter('sim_verify_door', False)
        self.declare_parameter('sim_door_target_rad', 1.15)
        self.declare_parameter('sim_door_tolerance_rad', 0.08)
        self.declare_parameter('sim_feedback_timeout_sec', 4.0)
        self.declare_parameter('allow_sim_fallback', False)
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

        self._planning_group = self.get_parameter('planning_group').value
        self._gripper_group  = self.get_parameter('gripper_group').value
        self._vel_scale      = self.get_parameter('velocity_scaling').value
        self._sim_mode       = self.get_parameter('sim_mode').value
        self._sim_control_enabled = self.get_parameter('sim_control_enabled').value
        self._sim_step_sec = float(self.get_parameter('sim_step_sec').value)
        self._sim_return_home = self.get_parameter('sim_return_home').value
        self._sim_verify_door = self.get_parameter('sim_verify_door').value
        self._sim_door_target = float(
            self.get_parameter('sim_door_target_rad').value)
        self._sim_door_tolerance = float(
            self.get_parameter('sim_door_tolerance_rad').value)
        self._sim_feedback_timeout = float(
            self.get_parameter('sim_feedback_timeout_sec').value)
        self._allow_sim_fallback = self.get_parameter('allow_sim_fallback').value
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
        self._sim_joint_pubs = []
        self._sim_door_pub = None
        self._sim_door_position = None
        if self._sim_mode and self._sim_control_enabled:
            topics = [f'/sim/joint{i}_position_cmd' for i in range(1, 7)]
            topics += ['/sim/gripper_left_position_cmd', '/sim/gripper_right_position_cmd']
            self._sim_joint_pubs = [self.create_publisher(Float64, topic, 10) for topic in topics]
            self._sim_door_pub = self.create_publisher(Float64, '/sim/door_hinge_position_cmd', 10)
            self.create_subscription(
                JointState, '/door_joint_states',
                self._sim_door_state_callback, 10)

        # MoveIt2 초기화
        self._arm    = None
        self._gripper = None
        self._moveit  = None
        self._moveit_ready = False
        self._piper_ready = False
        self._piper_pos_pub = None
        self._piper_enable_pub = None

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
        if self._sim_mode and self._sim_control_enabled:
            self.get_logger().info('Gazebo joint-control simulation enabled.')

    # ── 서비스 핸들러 ─────────────────────────────────────
    def open_door_callback(self,
                           request: OpenDoor.Request,
                           response: OpenDoor.Response):
        self.get_logger().info(
            f'Open door request: id={request.door_id}')

        success = self._execute_door_open_sequence(request.handle_position)

        response.success = success
        response.message = ('Door opened successfully'
                            if success else 'Failed to open door')

        done_msg = Bool()
        done_msg.data = success
        self.manip_done_pub.publish(done_msg)
        return response

    # ── 문 개방 시퀀스 ────────────────────────────────────
    def _execute_door_open_sequence(self,
                                    handle_position: PointStamped) -> bool:
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

        handle_position = self._transform_handle_to_manipulation_frame(
            handle_position)
        if handle_position is None:
            return False

        self.get_logger().info('Step 1/4: Moving to pre-grasp position')
        if not self._move_to_pre_grasp(handle_position):
            self.get_logger().error('Pre-grasp failed')
            return False

        self.get_logger().info('Step 2/4: Grasping handle')
        if not self._grasp_handle(handle_position):
            self.get_logger().error('Grasp failed')
            return False

        self.get_logger().info('Step 3/4: Pulling door open')
        if not self._pull_open_door(handle_position):
            self.get_logger().error('Pull failed')
            return False

        self.get_logger().info('Step 4/4: Returning to home position')
        self._move_to_home()
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

    def _transform_handle_to_manipulation_frame(
            self, handle_position: PointStamped):
        if self._sim_mode:
            return handle_position

        source_frame = handle_position.header.frame_id.strip()
        if not source_frame:
            self.get_logger().error(
                'handle_position has no frame_id; cannot command the real arm '
                'safely.')
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
            self.get_logger().error(
                f'Cannot transform handle_position from {source_frame} to '
                f'{self._manipulation_frame}: {e}')
            return None
        except Exception as e:
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
            if self._sim_control_enabled:
                self._publish_sim_pose([0.0, 0.72, -0.55, 0.0, -0.18, 0.0], 0.035)
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
            if self._sim_control_enabled:
                self._publish_sim_pose([0.0, 0.88, -0.72, 0.0, -0.16, 0.0], 0.006)
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

    def _pull_open_door(self, handle_pos: PointStamped) -> bool:
        """손잡이를 잡은 상태에서 로봇 후방으로 직선 당기기"""
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Pulling door {PULL_DISTANCE:.2f}m')
            if self._sim_control_enabled:
                self._publish_sim_pose([0.48, 0.70, -0.58, 0.0, -0.12, 0.0], 0.006)
                self._sim_door_position = None
                self._publish_float(self._sim_door_pub, self._sim_door_target)
            time.sleep(self._sim_step_sec)
            if self._sim_control_enabled and self._sim_verify_door:
                return self._wait_for_sim_door_target()
            return True

        # The manipulation frame uses x forward and z up, so pulling back
        # changes x.
        target = self._make_pose(
            x=handle_pos.point.x - PULL_DISTANCE,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if self._using_piper_sdk():
            return self._send_piper_pose(target, GRIPPER_CLOSE)
        return self._plan_and_execute_cartesian(target)

    def _move_to_home(self):
        if self._sim_mode:
            self.get_logger().info('  [SIM] Returning to home position')
            if self._sim_control_enabled and self._sim_return_home:
                self._publish_sim_pose([0.0] * 6, 0.035)
            time.sleep(self._sim_step_sec)
            return

    @staticmethod
    def _publish_float(publisher, value: float):
        if publisher is not None:
            msg = Float64()
            msg.data = float(value)
            publisher.publish(msg)

    def _publish_sim_pose(self, arm_positions, gripper_position):
        if len(self._sim_joint_pubs) != 8:
            return
        for publisher, value in zip(self._sim_joint_pubs[:6], arm_positions):
            self._publish_float(publisher, value)
        self._publish_float(self._sim_joint_pubs[6], gripper_position)
        self._publish_float(self._sim_joint_pubs[7], gripper_position)

    def _sim_door_state_callback(self, msg: JointState):
        try:
            index = msg.name.index('door_hinge')
            self._sim_door_position = float(msg.position[index])
        except (ValueError, IndexError):
            return

    def _wait_for_sim_door_target(self) -> bool:
        deadline = time.monotonic() + self._sim_feedback_timeout
        while time.monotonic() < deadline:
            position = self._sim_door_position
            if (position is not None and
                    abs(position - self._sim_door_target) <=
                    self._sim_door_tolerance):
                self.get_logger().info(
                    f'  [SIM] Door feedback verified: {position:.3f} rad')
                return True
            time.sleep(0.05)
        self.get_logger().error(
            '  [SIM] Door hinge did not reach target; '
            f'last={self._sim_door_position}, '
            f'target={self._sim_door_target:.3f}')
        return False

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
