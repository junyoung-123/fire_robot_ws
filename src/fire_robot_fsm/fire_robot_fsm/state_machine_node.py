import math

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from enum import IntEnum
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

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
        self.declare_parameter('exit_x',   10.0)
        self.declare_parameter('exit_y',    0.0)
        self.declare_parameter('exit_yaw',  0.0)

        self._nav_timeout_sec     = self.get_parameter('nav_timeout_sec').value
        self._max_nav_retries     = self.get_parameter('max_nav_retries').value
        self._explore_timeout_sec = self.get_parameter('explore_timeout_sec').value
        self._exit_x   = self.get_parameter('exit_x').value
        self._exit_y   = self.get_parameter('exit_y').value
        self._exit_yaw = self.get_parameter('exit_yaw').value

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

        self._nav_done   = False
        self._nav_failed = False
        self._nav_start_time:     Time | None = None
        self._explore_start_time: Time | None = None

        self._failed_door_ids: set[str] = set()   # 네비게이션 실패 문
        self._opened_door_ids: set[str] = set()   # 개방 완료 문
        self._nav_retry_count = 0

        self._door_opening_in_progress = False

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
            self.detected_doors.remove(existing)
            self.detected_doors.append(msg)
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
            self.detected_doors.remove(nearby)
        self.detected_doors.append(msg)

    def fire_callback(self, msg: FireInfo):
        self.fire_info = msg

    def nav_done_callback(self, msg: Bool):
        if self.state not in (State.NAVIGATING, State.EXITING):
            return
        if msg.data:
            self.get_logger().info('Navigation succeeded.')
            self._nav_done   = True
            self._nav_failed = False
        else:
            self.get_logger().warn('Navigation failed.')
            self._nav_failed = True
            self._nav_done   = False

    # ── FSM 루프 ──────────────────────────────────────────
    def fsm_loop(self):
        if   self.state == State.IDLE:             self._on_idle()
        elif self.state == State.EXPLORING:        self._on_exploring()
        elif self.state == State.NAVIGATING:       self._on_navigating()
        elif self.state == State.OPENING_DOOR:     self._on_opening_door()
        elif self.state == State.DOOR_OPENED:      self._on_door_opened()
        elif self.state == State.EXITING:          self._on_exiting()
        elif self.state == State.MISSION_COMPLETE: self._on_mission_complete()
        elif self.state == State.EMERGENCY_STOP:   self._on_emergency_stop()

        self._publish_state()

    # ── 상태별 동작 ───────────────────────────────────────
    def _on_idle(self):
        if self.fire_info and self.fire_info.detected:
            self.get_logger().info(
                f'Fire detected ({self.fire_info.red_door_count} red door(s)). '
                'Starting mission.')
            self._failed_door_ids.clear()
            self._opened_door_ids.clear()
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

        if not safe_doors:
            # 탐색 타임아웃 → 모든 파란 문 개방 완료 판단
            if self._explore_start_time is not None:
                elapsed = (
                    self.get_clock().now() - self._explore_start_time
                ).nanoseconds / 1e9
                if elapsed >= self._explore_timeout_sec:
                    self.get_logger().info(
                        f'{elapsed:.0f}s 탐색 후 새 파란 문 없음. '
                        f'총 {len(self._opened_door_ids)}개 문 개방 완료. '
                        '비상구로 이동합니다.')
                    self._transition(State.EXITING)
                    return

            # 타임아웃 전 → 제자리 회전 탐색
            self._rotate_to_scan()
            return

        # 새 파란 문 발견 → 탐색 타이머 리셋
        self._explore_start_time = self.get_clock().now()

        # 화재 위치에서 가장 먼 파란 문 선택 (화재 반대편 출구 우선)
        # fire_position은 base_link 프레임 — door_pose는 map 프레임이므로
        # 프레임이 일치할 때만 거리 비교 사용, 아니면 confidence 기반 선택
        fire_pos = (self.fire_info.fire_position
                    if self.fire_info and self.fire_info.detected else None)
        fire_in_map = (fire_pos is not None
                       and fire_pos.header.frame_id == 'map')
        if fire_in_map:
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

    def _handle_nav_failure(self, reason: str = ''):
        door_id = self.target_door.door_id if self.target_door else '(unknown)'
        self._nav_retry_count += 1
        self.get_logger().warn(
            f'Nav failure [{reason}] → {door_id}. '
            f'재시도 {self._nav_retry_count}/{self._max_nav_retries}')
        if self.target_door is not None:
            self._failed_door_ids.add(self.target_door.door_id)
            self.target_door = None
        if self._nav_retry_count >= self._max_nav_retries:
            self.get_logger().error('Max retries 초과. Emergency stop.')
            self._transition(State.EMERGENCY_STOP)
        else:
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
            self._opened_door_ids.add(self.target_door.door_id)
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
        self.get_logger().info(
            f'다음 파란 문 탐색 시작 (개방 완료: {len(self._opened_door_ids)}개)')
        self._transition(State.EXPLORING)

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
    def _send_exit_goal(self):
        """탐지된 초록 비상구 or 하드코딩 좌표를 navigation_node에 전달"""
        if self._exit_door is not None:
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

    # ── 유틸 ─────────────────────────────────────────────
    def _rotate_to_scan(self):
        twist = Twist()
        twist.angular.z = 0.3
        self.cmd_vel_pub.publish(twist)

    def _stop_rotation(self):
        self.cmd_vel_pub.publish(Twist())

    def _transition(self, new_state: State):
        self.get_logger().info(f'State: {self.state.name} → {new_state.name}')
        self.state = new_state

        if new_state == State.NAVIGATING:
            self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
        elif new_state == State.EXITING:
            self._send_exit_goal()
            self._nav_start_time = self.get_clock().now()
        elif new_state == State.EXPLORING:
            # EXPLORING 재진입 시 탐색 타이머 시작
            self._explore_start_time = self.get_clock().now()
            self._nav_start_time = None
        else:
            self._nav_start_time = None

    def _publish_state(self):
        msg                   = RobotState()
        msg.header.stamp      = self.get_clock().now().to_msg()
        msg.state             = int(self.state)
        msg.state_description = self.state.name
        msg.target_door_id    = (self.target_door.door_id
                                  if self.target_door else '')
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
