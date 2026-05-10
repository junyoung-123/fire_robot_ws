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
    IDLE           = 0
    EXPLORING      = 1
    NAVIGATING     = 2
    OPENING_DOOR   = 3
    DOOR_OPENED    = 4
    EMERGENCY_STOP = 5


class StateMachineNode(Node):
    """전체 시스템을 조율하는 유한 상태 머신(FSM) 노드

    네비게이션 실패 처리 정책:
      - 실패 시 해당 문을 블랙리스트에 추가하고 다른 파란 문으로 재탐색
      - 재시도 횟수(max_nav_retries) 초과 → EMERGENCY_STOP
      - 타임아웃(nav_timeout_sec) 초과 → 동일 처리
    """

    def __init__(self):
        super().__init__('state_machine_node')

        # ── 파라미터 ──────────────────────────────────────
        self.declare_parameter('nav_timeout_sec',  60.0)
        self.declare_parameter('max_nav_retries',  3)

        self._nav_timeout_sec = self.get_parameter('nav_timeout_sec').value
        self._max_nav_retries = self.get_parameter('max_nav_retries').value

        cb_group = ReentrantCallbackGroup()

        # ── Subscribers ───────────────────────────────────
        self.door_sub = self.create_subscription(
            DoorInfo, '/detected_door', self.door_callback, 10,
            callback_group=cb_group)
        self.fire_sub = self.create_subscription(
            FireInfo, '/fire_info', self.fire_callback, 10,
            callback_group=cb_group)
        self.nav_done_sub = self.create_subscription(
            Bool, '/navigation_done', self.nav_done_callback, 10,
            callback_group=cb_group)

        # ── Publishers ────────────────────────────────────
        self.state_pub       = self.create_publisher(RobotState, '/robot_state', 10)
        self.target_door_pub = self.create_publisher(DoorInfo,   '/target_door', 10)
        self.cmd_vel_pub     = self.create_publisher(Twist,      '/cmd_vel',     10)

        # ── Service clients ───────────────────────────────
        self.open_door_client = self.create_client(
            OpenDoor, '/open_door', callback_group=cb_group)

        # ── 상태 변수 ──────────────────────────────────────
        self.state             = State.IDLE
        self.detected_doors:   list[DoorInfo] = []
        self.fire_info:        FireInfo | None = None
        self.target_door:      DoorInfo | None = None

        self._nav_done   = False
        self._nav_failed = False
        self._nav_start_time: Time | None = None

        # 이번 미션에서 네비게이션 실패한 문 ID 집합
        self._failed_door_ids: set[str] = set()
        # 이번 미션 누적 실패 횟수
        self._nav_retry_count = 0

        self._door_opening_in_progress = False

        self.timer = self.create_timer(0.5, self.fsm_loop)
        self.get_logger().info(
            f'StateMachineNode started | '
            f'timeout={self._nav_timeout_sec}s  max_retries={self._max_nav_retries}'
        )

    # ── 콜백 ──────────────────────────────────────────────
    def door_callback(self, msg: DoorInfo):
        existing = next(
            (d for d in self.detected_doors if d.door_id == msg.door_id), None)
        if existing:
            self.detected_doors.remove(existing)
        self.detected_doors.append(msg)

    def fire_callback(self, msg: FireInfo):
        self.fire_info = msg

    def nav_done_callback(self, msg: Bool):
        if self.state != State.NAVIGATING:
            return
        if msg.data:
            self.get_logger().info('Navigation succeeded.')
            self._nav_done   = True
            self._nav_failed = False
        else:
            self.get_logger().warn('Navigation failed (nav_done=False).')
            self._nav_failed = True
            self._nav_done   = False

    # ── FSM 루프 ──────────────────────────────────────────
    def fsm_loop(self):
        if   self.state == State.IDLE:           self._on_idle()
        elif self.state == State.EXPLORING:      self._on_exploring()
        elif self.state == State.NAVIGATING:     self._on_navigating()
        elif self.state == State.OPENING_DOOR:   self._on_opening_door()
        elif self.state == State.DOOR_OPENED:    self._on_door_opened()
        elif self.state == State.EMERGENCY_STOP: self._on_emergency_stop()

        self._publish_state()

    # ── 상태별 동작 ───────────────────────────────────────
    def _on_idle(self):
        if self.fire_info and self.fire_info.detected:
            self.get_logger().info(
                f'Red door(s) detected ({self.fire_info.red_door_count}). '
                'Starting exploration.'
            )
            # IDLE → EXPLORING 시 이전 실패 기록 초기화
            self._failed_door_ids.clear()
            self._nav_retry_count = 0
            self._transition(State.EXPLORING)

    def _on_exploring(self):
        # 블랙리스트를 제외한 파란 문만 후보
        safe_doors = [
            d for d in self.detected_doors
            if d.door_color == 'blue' and d.door_id not in self._failed_door_ids
        ]
        if not safe_doors:
            if self._failed_door_ids:
                self.get_logger().error(
                    f'No safe doors remaining after '
                    f'{len(self._failed_door_ids)} failure(s). Emergency stop.')
                self._transition(State.EMERGENCY_STOP)
                return
            # 파란 문이 아직 보이지 않으면 제자리 회전으로 주변 스캔
            self._rotate_to_scan()
            return

        if self.fire_info and self.fire_info.detected:
            fx = self.fire_info.fire_position.point.x
            fy = self.fire_info.fire_position.point.y
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
            f'Target door: {self.target_door.door_id} '
            f'(retry {self._nav_retry_count}/{self._max_nav_retries})'
        )
        self.target_door_pub.publish(self.target_door)
        self._nav_done   = False
        self._nav_failed = False
        self._transition(State.NAVIGATING)

    def _on_navigating(self):
        # ── 타임아웃 체크 ─────────────────────────────────
        if self._nav_start_time is not None:
            elapsed = (
                self.get_clock().now() - self._nav_start_time
            ).nanoseconds / 1e9
            if elapsed > self._nav_timeout_sec:
                self.get_logger().warn(
                    f'Navigation timeout ({elapsed:.1f}s > '
                    f'{self._nav_timeout_sec}s).'
                )
                self._handle_nav_failure(reason='timeout')
                return

        # ── 완료 신호 처리 ────────────────────────────────
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
            f'Nav failure [{reason}] → door {door_id}. '
            f'Retry {self._nav_retry_count}/{self._max_nav_retries}.'
        )

        # 실패한 문 블랙리스트 등록
        if self.target_door is not None:
            self._failed_door_ids.add(self.target_door.door_id)
            self.target_door = None

        if self._nav_retry_count >= self._max_nav_retries:
            self.get_logger().error(
                f'Max retries ({self._max_nav_retries}) reached. '
                'Activating emergency stop.')
            self._transition(State.EMERGENCY_STOP)
        else:
            # 다른 파란 문으로 재탐색
            self._transition(State.EXPLORING)

    def _on_opening_door(self):
        if self.target_door is None or self._door_opening_in_progress:
            return
        if not self.open_door_client.service_is_ready():
            self.get_logger().warn('OpenDoor service not ready, waiting...')
            return

        self._door_opening_in_progress = True
        req           = OpenDoor.Request()
        req.door_id   = self.target_door.door_id
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
            self.get_logger().info('Door opened! Evacuation route secured.')
            self._transition(State.DOOR_OPENED)
        else:
            self.get_logger().error(f'Door open failed: {result.message}')
            self._transition(State.EMERGENCY_STOP)

    def _on_door_opened(self):
        self.get_logger().info(
            'Mission complete. Evacuation route secured.', once=True)

    def _on_emergency_stop(self):
        self.get_logger().error('Emergency stop activated.', once=True)

    def _rotate_to_scan(self):
        """파란 문을 찾을 때까지 제자리 회전으로 주변 스캔."""
        twist = Twist()
        twist.angular.z = 0.3  # rad/s
        self.cmd_vel_pub.publish(twist)

    def _stop_rotation(self):
        self.cmd_vel_pub.publish(Twist())

    # ── 유틸 ─────────────────────────────────────────────
    def _transition(self, new_state: State):
        self.get_logger().info(f'State: {self.state.name} → {new_state.name}')
        self.state = new_state

        # NAVIGATING 진입 시 회전 정지 + 타이머 시작
        if new_state == State.NAVIGATING:
            self._stop_rotation()
            self._nav_start_time = self.get_clock().now()
        else:
            self._nav_start_time = None

    def _publish_state(self):
        msg                  = RobotState()
        msg.header.stamp     = self.get_clock().now().to_msg()
        msg.state            = int(self.state)
        msg.state_description = self.state.name
        msg.target_door_id   = (self.target_door.door_id
                                 if self.target_door else '')
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
