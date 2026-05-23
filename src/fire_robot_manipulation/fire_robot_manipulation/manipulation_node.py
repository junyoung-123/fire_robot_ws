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
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Bool
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
        self.declare_parameter('allow_sim_fallback', False)
        self.declare_parameter('manipulation_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 1.0)

        self._planning_group = self.get_parameter('planning_group').value
        self._gripper_group  = self.get_parameter('gripper_group').value
        self._vel_scale      = self.get_parameter('velocity_scaling').value
        self._sim_mode       = self.get_parameter('sim_mode').value
        self._allow_sim_fallback = self.get_parameter('allow_sim_fallback').value
        self._manipulation_frame = self.get_parameter('manipulation_frame').value
        self._tf_timeout_sec = self.get_parameter('tf_timeout_sec').value

        cb_group = ReentrantCallbackGroup()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # 문 개방 서비스 서버
        self.open_door_srv = self.create_service(
            OpenDoor, '/open_door', self.open_door_callback,
            callback_group=cb_group)

        # 조작 완료 신호 발행 (옵션 모니터링용)
        self.manip_done_pub = self.create_publisher(Bool, '/manipulation_done', 10)

        # MoveIt2 초기화
        self._arm    = None
        self._gripper = None
        self._moveit  = None
        self._moveit_ready = False

        if not self._sim_mode:
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
        self.get_logger().info(f'ManipulationNode started [{mode_str}]')

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
        if not self._sim_mode and not self._moveit_ready:
            self.get_logger().error(
                'MoveItPy is not ready in real robot mode; '
                'refusing to report simulated success.')
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
            time.sleep(1.0)
            return True

        target = self._make_pose(
            x=handle_pos.point.x - PRE_GRASP_OFFSET,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        return self._plan_and_execute_cartesian(target)

    def _grasp_handle(self, handle_pos: PointStamped) -> bool:
        if self._sim_mode:
            self.get_logger().info('  [SIM] Gripper closing')
            time.sleep(0.5)
            return True

        # 그리퍼 열기
        self._set_gripper(GRIPPER_OPEN)

        # 손잡이 위치로 직선 이동
        target = self._make_pose(
            x=handle_pos.point.x,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        if not self._plan_and_execute_cartesian(target):
            return False

        # 그리퍼 닫기
        return self._set_gripper(GRIPPER_CLOSE)

    def _pull_open_door(self, handle_pos: PointStamped) -> bool:
        """손잡이를 잡은 상태에서 로봇 후방으로 직선 당기기"""
        if self._sim_mode:
            self.get_logger().info(
                f'  [SIM] Pulling door {PULL_DISTANCE:.2f}m')
            time.sleep(1.5)
            return True

        # base_link uses x forward and z up, so pulling back changes x.
        target = self._make_pose(
            x=handle_pos.point.x - PULL_DISTANCE,
            y=handle_pos.point.y,
            z=handle_pos.point.z,
        )
        return self._plan_and_execute_cartesian(target)

    def _move_to_home(self):
        if self._sim_mode:
            self.get_logger().info('  [SIM] Returning to home position')
            time.sleep(1.0)
            return

        if self._arm is None:
            return
        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(configuration_name='home')
        plan_result = self._arm.plan()
        if plan_result:
            self._moveit.execute(plan_result.trajectory,
                                  controllers=[])

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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
