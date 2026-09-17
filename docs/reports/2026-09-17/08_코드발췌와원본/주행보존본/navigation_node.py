from collections import deque
from copy import deepcopy
import math
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from action_msgs.msg import GoalStatus

from fire_robot_interfaces.msg import DoorInfo


class NavigationNode(Node):
    """Nav2 기반 자율주행 노드.

    FSM으로부터 /target_door 를 수신해 Nav2로 목표 전송,
    완료(성공/실패) 시 /navigation_done 발행.
    """

    def __init__(self):
        super().__init__('navigation_node')

        self.declare_parameter('navigation_frame', 'map')
        self.declare_parameter('max_goal_step_m', 0.0)
        self.declare_parameter('adjust_goal_to_free_space', True)
        self.declare_parameter('goal_adjust_search_radius_m', 1.2)
        self.declare_parameter('goal_adjust_max_cost', 70)
        self.declare_parameter('goal_adjust_max_abs_y_m', 0.0)
        self.declare_parameter('goal_adjust_min_progress_m', 0.30)
        self.declare_parameter('goal_adjust_require_reachable', True)
        self.declare_parameter('goal_adjust_connectivity_max_cost', 95)
        self.declare_parameter('goal_adjust_start_search_radius_m', 0.80)
        self.declare_parameter('blue_goal_adjust_lateral_bias', 0.0)
        self.declare_parameter('blue_goal_adjust_min_abs_y_m', 0.0)
        self.declare_parameter('blue_goal_adjust_min_abs_y_slack_m', 0.18)
        self.declare_parameter('blue_goal_adjust_max_abs_y_overshoot_m', 0.0)
        self.declare_parameter('blue_goal_adjust_max_frame_lateral_m', 0.0)
        self.declare_parameter('fallback_goal_reached_dist_m', 0.85)
        self.declare_parameter('precise_goal_reached_dist_m', 0.18)
        self.declare_parameter('precise_goal_reached_yaw_tolerance_rad', 0.35)
        self.declare_parameter('explore_goal_reached_dist_m', 0.55)
        self.declare_parameter('explore_navigation_timeout_sec', 75.0)
        self.declare_parameter('green_navigation_timeout_sec', 0.0)
        self.declare_parameter('lane_biased_intermediate_goals', False)
        self.declare_parameter('lane_goal_min_abs_y_m', 0.75)
        self.declare_parameter('nav2_server_initial_wait_sec', 0.2)
        self.declare_parameter('nav2_server_retry_period_sec', 1.0)
        self.declare_parameter('nav2_server_max_wait_sec', 120.0)
        self.declare_parameter('retarget_preempt_enabled', True)
        self.declare_parameter('retarget_cancel_settle_sec', 0.35)
        self.declare_parameter('duplicate_goal_tolerance_m', 0.30)
        self.declare_parameter('duplicate_goal_replan_after_sec', 6.0)
        self.declare_parameter('early_success_cancel_enabled', False)
        self._navigation_frame = self.get_parameter('navigation_frame').value
        self._max_goal_step_m = float(self.get_parameter('max_goal_step_m').value)
        self._adjust_goal_to_free_space = bool(
            self.get_parameter('adjust_goal_to_free_space').value)
        self._goal_adjust_search_radius_m = float(
            self.get_parameter('goal_adjust_search_radius_m').value)
        self._goal_adjust_max_cost = int(
            self.get_parameter('goal_adjust_max_cost').value)
        self._goal_adjust_max_abs_y_m = float(
            self.get_parameter('goal_adjust_max_abs_y_m').value)
        self._goal_adjust_min_progress_m = float(
            self.get_parameter('goal_adjust_min_progress_m').value)
        self._goal_adjust_require_reachable = bool(
            self.get_parameter('goal_adjust_require_reachable').value)
        self._goal_adjust_connectivity_max_cost = int(
            self.get_parameter('goal_adjust_connectivity_max_cost').value)
        self._goal_adjust_start_search_radius_m = float(
            self.get_parameter('goal_adjust_start_search_radius_m').value)
        self._blue_goal_adjust_lateral_bias = max(
            0.0, float(self.get_parameter('blue_goal_adjust_lateral_bias').value))
        self._blue_goal_adjust_min_abs_y_m = max(
            0.0, float(self.get_parameter('blue_goal_adjust_min_abs_y_m').value))
        self._blue_goal_adjust_min_abs_y_slack_m = max(
            0.0, float(self.get_parameter('blue_goal_adjust_min_abs_y_slack_m').value))
        self._blue_goal_adjust_max_abs_y_overshoot_m = max(
            0.0, float(self.get_parameter('blue_goal_adjust_max_abs_y_overshoot_m').value))
        self._blue_goal_adjust_max_frame_lateral_m = max(
            0.0, float(self.get_parameter('blue_goal_adjust_max_frame_lateral_m').value))
        self._fallback_goal_reached_dist_m = float(
            self.get_parameter('fallback_goal_reached_dist_m').value)
        self._precise_goal_reached_dist_m = float(
            self.get_parameter('precise_goal_reached_dist_m').value)
        self._precise_goal_reached_yaw_tolerance_rad = max(0.0, float(
            self.get_parameter('precise_goal_reached_yaw_tolerance_rad').value))
        self._explore_goal_reached_dist_m = float(
            self.get_parameter('explore_goal_reached_dist_m').value)
        self._explore_navigation_timeout_sec = float(
            self.get_parameter('explore_navigation_timeout_sec').value)
        self._green_navigation_timeout_sec = float(
            self.get_parameter('green_navigation_timeout_sec').value)
        self._lane_biased_intermediate_goals = bool(
            self.get_parameter('lane_biased_intermediate_goals').value)
        self._lane_goal_min_abs_y_m = float(
            self.get_parameter('lane_goal_min_abs_y_m').value)
        self._nav2_server_initial_wait_sec = float(
            self.get_parameter('nav2_server_initial_wait_sec').value)
        self._nav2_server_retry_period_sec = float(
            self.get_parameter('nav2_server_retry_period_sec').value)
        self._nav2_server_max_wait_sec = float(
            self.get_parameter('nav2_server_max_wait_sec').value)
        self._retarget_preempt_enabled = bool(
            self.get_parameter('retarget_preempt_enabled').value)
        self._retarget_cancel_settle_sec = max(
            0.0, float(self.get_parameter('retarget_cancel_settle_sec').value))
        self._duplicate_goal_tolerance_m = max(
            0.0, float(self.get_parameter('duplicate_goal_tolerance_m').value))
        self._duplicate_goal_replan_after_sec = max(
            0.0, float(self.get_parameter('duplicate_goal_replan_after_sec').value))
        self._early_success_cancel_enabled = bool(
            self.get_parameter('early_success_cancel_enabled').value)

        cb_group = ReentrantCallbackGroup()

        self.nav2_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=cb_group)

        self.goal_sub = self.create_subscription(
            DoorInfo, '/target_door', self.target_door_callback, 10,
            callback_group=cb_group)
        self.cancel_sub = self.create_subscription(
            Bool, '/cancel_navigation', self._cancel_navigation_callback, 10,
            callback_group=cb_group)
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback,
            costmap_qos, callback_group=cb_group)
        axis_qos = QoSProfile(depth=1)
        axis_qos.reliability = ReliabilityPolicy.RELIABLE
        axis_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.axis_sub = self.create_subscription(
            PoseStamped, '/mission_axis', self._mission_axis_callback, axis_qos,
            callback_group=cb_group)

        self.nav_done_pub = self.create_publisher(Bool, '/navigation_done', 10)

        self._current_goal_handle = None
        self._navigating = False
        self._final_goal_pose: PoseStamped | None = None
        self._active_goal_pose: PoseStamped | None = None
        self._active_goal_is_intermediate = False
        self._latest_costmap: OccupancyGrid | None = None
        self._current_goal_color = ''
        self._current_goal_door_id = ''
        self._current_goal_request_pose: PoseStamped | None = None
        self._pending_nav2_pose: PoseStamped | None = None
        self._pending_nav2_color = ''
        self._pending_nav2_start_time = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._goal_start_time = None
        self._canceled_goal_handle_ids: set[int] = set()
        self._goal_sequence = 0
        self._pending_retarget_pose: PoseStamped | None = None
        self._pending_retarget_color = ''
        self._pending_retarget_door_id = ''
        self._pending_retarget_ready_wall_time: float | None = None
        self._mission_axis_origin_x = 0.0
        self._mission_axis_origin_y = 0.0
        self._mission_axis_yaw = 0.0
        self._mission_axis_received = False

        self.create_timer(0.5, self._goal_watchdog_callback)
        self.create_timer(
            max(0.2, self._nav2_server_retry_period_sec),
            self._retry_pending_nav2_goal)
        self.create_timer(0.1, self._retry_retarget_after_cancel)
        self.get_logger().info('NavigationNode started')

    def costmap_callback(self, msg: OccupancyGrid):
        self._latest_costmap = msg

    def _mission_axis_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id != self._navigation_frame:
            return
        self._mission_axis_origin_x = float(msg.pose.position.x)
        self._mission_axis_origin_y = float(msg.pose.position.y)
        self._mission_axis_yaw = self._yaw_from_quaternion(msg.pose.orientation)
        self._mission_axis_received = True
        self.get_logger().info(
            f'Mission axis received: origin='
            f'({self._mission_axis_origin_x:.2f}, {self._mission_axis_origin_y:.2f}), '
            f'yaw={math.degrees(self._mission_axis_yaw):.1f}deg',
            once=True)

    # ── 목표 수신 ─────────────────────────────────────────
    def target_door_callback(self, msg: DoorInfo):
        self.get_logger().info(
            f'Navigation goal received: door {msg.door_id} '
            f'(color={msg.door_color}, '
            f'frame={msg.door_pose.header.frame_id}, '
            f'x={msg.door_pose.pose.position.x:.2f}, '
            f'y={msg.door_pose.pose.position.y:.2f})'
        )
        if self._is_duplicate_active_goal(msg):
            self.get_logger().info(
                f'Ignoring duplicate navigation target for active door: {msg.door_id}',
                throttle_duration_sec=3.0)
            return
        if (self._navigating
                or self._current_goal_handle is not None
                or self._pending_retarget_pose is not None):
            if self._retarget_preempt_enabled:
                self.get_logger().warn(
                    f'Already navigating. Preempting current goal without an '
                    f'explicit stop before retargeting to: {msg.door_id}')
                self._preempt_current_goal(msg)
            else:
                self.get_logger().warn(
                    f'Already navigating. Cancelling current goal before retargeting to: '
                    f'{msg.door_id}')
                self._schedule_retarget_goal(msg)
            return

        self._current_goal_color = msg.door_color
        self._current_goal_door_id = msg.door_id
        self._current_goal_request_pose = deepcopy(msg.door_pose)
        self._start_new_navigation(msg.door_pose)

    def _cancel_navigation_callback(self, msg: Bool):
        if not msg.data:
            return
        active = (
            self._navigating
            or self._current_goal_handle is not None
            or self._pending_nav2_pose is not None
            or self._pending_retarget_pose is not None
        )
        if not active:
            return
        self._pending_retarget_pose = None
        self._pending_retarget_color = ''
        self._pending_retarget_door_id = ''
        self._pending_retarget_ready_wall_time = None
        self.get_logger().info(
            'Navigation cancellation requested for manual control handoff.',
            throttle_duration_sec=1.0)
        self._cancel_current_goal()

    def _is_duplicate_active_goal(self, msg: DoorInfo) -> bool:
        active = (
            self._navigating
            or self._current_goal_handle is not None
            or self._pending_retarget_pose is not None
            or self._pending_nav2_pose is not None
        )
        if not active:
            return False

        if float(msg.confidence) < 0.0:
            return False

        door_id = msg.door_id or ''
        known_ids: set[str] = set()
        if door_id:
            known_ids = {
                self._current_goal_door_id,
                self._pending_retarget_door_id,
            }
            if door_id not in known_ids:
                return False

        reference_pose = (
            self._pending_retarget_pose
            or self._current_goal_request_pose
            or self._final_goal_pose
            or self._pending_nav2_pose
            or self._active_goal_pose
        )
        if reference_pose is None:
            return bool(door_id and door_id in known_ids)
        if msg.door_pose.header.frame_id != reference_pose.header.frame_id:
            return False

        dx = msg.door_pose.pose.position.x - reference_pose.pose.position.x
        dy = msg.door_pose.pose.position.y - reference_pose.pose.position.y
        return math.hypot(dx, dy) <= self._duplicate_goal_tolerance_m

    def _start_new_navigation(self, pose: PoseStamped):
        self._goal_sequence += 1
        self._navigate_to_pose(pose)

    def _schedule_retarget_goal(self, msg: DoorInfo):
        self._pending_retarget_pose = deepcopy(msg.door_pose)
        self._pending_retarget_color = msg.door_color
        self._pending_retarget_door_id = msg.door_id
        self._pending_retarget_ready_wall_time = (
            time.monotonic() + self._retarget_cancel_settle_sec)
        self._cancel_current_goal()

    def _preempt_current_goal(self, msg: DoorInfo):
        self._pending_retarget_pose = None
        self._pending_retarget_color = ''
        self._pending_retarget_door_id = ''
        self._pending_retarget_ready_wall_time = None
        self._pending_nav2_pose = None
        self._pending_nav2_color = ''
        self._pending_nav2_start_time = None
        self._current_goal_handle = None
        self._current_goal_color = msg.door_color
        self._current_goal_door_id = msg.door_id
        self._current_goal_request_pose = deepcopy(msg.door_pose)
        self._goal_start_time = None
        self._start_new_navigation(msg.door_pose)

    def _retry_retarget_after_cancel(self):
        if self._pending_retarget_pose is None:
            return
        ready_wall_time = self._pending_retarget_ready_wall_time
        if (ready_wall_time is not None
                and time.monotonic() < ready_wall_time):
            return

        pose = self._pending_retarget_pose
        color = self._pending_retarget_color
        door_id = self._pending_retarget_door_id
        self._pending_retarget_pose = None
        self._pending_retarget_color = ''
        self._pending_retarget_door_id = ''
        self._pending_retarget_ready_wall_time = None
        self._current_goal_color = color
        self._current_goal_door_id = door_id
        self._current_goal_request_pose = deepcopy(pose)
        self.get_logger().info(
            f'Sending retargeted navigation goal after cancel settle: {door_id}')
        self._start_new_navigation(pose)

    # ── Nav2 목표 전송 ────────────────────────────────────
    def _navigate_to_pose(self, pose: PoseStamped):
        if not self.nav2_client.wait_for_server(
                timeout_sec=max(0.0, self._nav2_server_initial_wait_sec)):
            self._queue_until_nav2_ready(pose)
            return

        pose = self._fixed_map_goal(pose)
        if pose is None:
            self._publish_done(success=False)
            return

        adjusted_pose = self._adjust_pose_to_free_space(pose, progress_to=pose)
        if adjusted_pose is None:
            self._publish_done(success=False)
            return
        pose = adjusted_pose
        self._final_goal_pose = pose
        self._navigating = True
        self._pending_nav2_pose = None
        self._pending_nav2_color = ''
        self._pending_nav2_start_time = None
        self._send_goal_step(pose)

    def _queue_until_nav2_ready(self, pose: PoseStamped):
        if self._pending_nav2_start_time is None:
            self._pending_nav2_start_time = self.get_clock().now()
        self._pending_nav2_pose = deepcopy(pose)
        self._pending_nav2_color = self._current_goal_color
        self._navigating = True
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._goal_start_time = None
        self.get_logger().warn(
            'Nav2 action server not available yet; holding navigation goal '
            'instead of reporting failure.',
            throttle_duration_sec=2.0)

    def _retry_pending_nav2_goal(self):
        if self._pending_nav2_pose is None or self._pending_nav2_start_time is None:
            return

        elapsed = (
            self.get_clock().now() - self._pending_nav2_start_time
        ).nanoseconds / 1e9
        if (self._nav2_server_max_wait_sec > 0.0
                and elapsed >= self._nav2_server_max_wait_sec):
            self.get_logger().error(
                f'Nav2 action server still unavailable after {elapsed:.1f}s. '
                'Reporting navigation failure for the held goal.')
            self._pending_nav2_pose = None
            self._pending_nav2_color = ''
            self._pending_nav2_start_time = None
            self._navigating = False
            self._current_goal_color = ''
            self._current_goal_door_id = ''
            self._current_goal_request_pose = None
            self._publish_done(success=False)
            return

        if not self.nav2_client.wait_for_server(timeout_sec=0.05):
            return

        pose = self._pending_nav2_pose
        color = self._pending_nav2_color
        self._pending_nav2_pose = None
        self._pending_nav2_color = ''
        self._pending_nav2_start_time = None
        self._current_goal_color = color
        self.get_logger().info(
            f'Nav2 action server is ready after {elapsed:.1f}s; sending held goal.')
        self._navigate_to_pose(pose)

    def _adjust_pose_to_free_space(self, pose: PoseStamped,
                                   progress_to: PoseStamped | None = None) -> PoseStamped | None:
        grid = self._latest_costmap
        if not self._adjust_goal_to_free_space or grid is None:
            return pose
        if pose.header.frame_id != grid.header.frame_id:
            return pose

        x = pose.pose.position.x
        y = pose.pose.position.y
        cost = self._costmap_value(grid, x, y)
        reachable = None
        goal_cell = self._costmap_indices(grid, x, y)
        if self._goal_adjust_require_reachable:
            reachable = self._reachable_costmap_cells(grid)
            if reachable is None:
                self.get_logger().warn(
                    'Could not evaluate costmap connectivity for the navigation goal; '
                    'falling back to local free-space adjustment only.',
                    throttle_duration_sec=2.0)

        if cost == -1 and not self._goal_adjust_require_reachable:
            return pose
        if cost is not None and 0 <= cost <= self._goal_adjust_max_cost:
            if (reachable is None
                    or self._is_costmap_cell_reachable(grid, reachable, goal_cell)):
                return pose

        best = self._nearest_free_costmap_point(
            grid, x, y, progress_to, reachable)
        if best is None:
            self.get_logger().warn(
                f'Navigation goal is not reachable/free (cost={cost}) and no nearby '
                'connected free cell was found. '
                'Rejecting this goal instead of waiting for a Nav2 timeout.')
            return None

        adjusted = deepcopy(pose)
        adjusted.pose.position.x = best[0]
        adjusted.pose.position.y = best[1]
        self.get_logger().warn(
            f'Adjusted unreachable/occupied navigation goal from '
            f'({x:.2f}, {y:.2f}, cost={cost}) to connected free cell '
            f'({best[0]:.2f}, {best[1]:.2f}, cost={best[2]})')
        return adjusted

    def _costmap_value(self, grid: OccupancyGrid, x: float, y: float) -> int | None:
        mx, my = self._costmap_indices(grid, x, y)
        if mx is None or my is None:
            return None
        return int(grid.data[my * grid.info.width + mx])

    def _costmap_indices(self, grid: OccupancyGrid, x: float, y: float):
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        res = grid.info.resolution
        mx = int((x - ox) / res)
        my = int((y - oy) / res)
        if not (0 <= mx < grid.info.width and 0 <= my < grid.info.height):
            return None, None
        return mx, my

    def _nearest_free_costmap_point(self, grid: OccupancyGrid, x: float, y: float,
                                    progress_to: PoseStamped | None = None,
                                    reachable: bytearray | None = None):
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        res = grid.info.resolution
        cx, cy = self._costmap_indices(grid, x, y)
        if cx is None or cy is None:
            return None

        progress_origin = None
        progress_unit = None
        if progress_to is not None:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._navigation_frame, 'base_link', Time(),
                    timeout=Duration(seconds=0.2))
                rx = transform.transform.translation.x
                ry = transform.transform.translation.y
                fx = progress_to.pose.position.x
                fy = progress_to.pose.position.y
                vx = fx - rx
                vy = fy - ry
                vlen = math.hypot(vx, vy)
                if vlen > 0.01:
                    progress_origin = (rx, ry)
                    progress_unit = (vx / vlen, vy / vlen)
            except TransformException:
                progress_origin = None
                progress_unit = None

        radius_cells = max(1, int(self._goal_adjust_search_radius_m / res))
        best = None
        target_lateral = self._axis_lateral_xy(x, y)
        blue_goal = self._current_goal_color == 'blue' and abs(target_lateral) > 0.15
        explore_goal = self._current_goal_color == 'explore' and abs(target_lateral) > 0.15
        blue_side = 1.0 if target_lateral >= 0.0 else -1.0
        explore_side = 1.0 if target_lateral >= 0.0 else -1.0
        blue_min_abs_lateral = 0.0
        blue_max_abs_lateral = 0.0
        blue_goal_yaw = None
        if blue_goal:
            blue_min_abs_lateral = max(
                self._blue_goal_adjust_min_abs_y_m,
                abs(target_lateral) - self._blue_goal_adjust_min_abs_y_slack_m,
            )
            if self._blue_goal_adjust_max_abs_y_overshoot_m > 0.0:
                blue_max_abs_lateral = (
                    max(abs(target_lateral), blue_min_abs_lateral)
                    + self._blue_goal_adjust_max_abs_y_overshoot_m)
            if progress_to is not None:
                blue_goal_yaw = self._yaw_from_quaternion(
                    progress_to.pose.orientation)

        def search_candidates(relaxed_blue: bool):
            best_candidate = None
            relaxed_min_abs_lateral = blue_min_abs_lateral
            relaxed_max_abs_lateral = blue_max_abs_lateral
            relaxed_frame_lateral_limit = self._blue_goal_adjust_max_frame_lateral_m
            if relaxed_blue:
                relaxed_min_abs_lateral = min(
                    blue_min_abs_lateral,
                    max(0.20, abs(target_lateral) - 0.95))
                relaxed_max_abs_lateral = 0.0
                relaxed_frame_lateral_limit = max(
                    self._blue_goal_adjust_max_frame_lateral_m, 0.60)

            for yy in range(cy - radius_cells, cy + radius_cells + 1):
                for xx in range(cx - radius_cells, cx + radius_cells + 1):
                    if not (0 <= xx < grid.info.width and 0 <= yy < grid.info.height):
                        continue
                    cost = int(grid.data[yy * grid.info.width + xx])
                    if cost < 0 or cost > self._goal_adjust_max_cost:
                        continue
                    if (reachable is not None
                            and reachable[yy * grid.info.width + xx] == 0):
                        continue
                    wx = ox + (xx + 0.5) * res
                    wy = oy + (yy + 0.5) * res
                    dist = math.hypot(wx - x, wy - y)
                    if dist > self._goal_adjust_search_radius_m:
                        continue
                    candidate_lateral = self._axis_lateral_xy(wx, wy)
                    if (self._goal_adjust_max_abs_y_m > 0.0
                            and abs(candidate_lateral) > self._goal_adjust_max_abs_y_m):
                        continue
                    if blue_goal:
                        if candidate_lateral * blue_side <= 0.0:
                            continue
                        if (relaxed_min_abs_lateral > 0.0
                                and abs(candidate_lateral) < relaxed_min_abs_lateral):
                            continue
                        if (relaxed_max_abs_lateral > 0.0
                                and abs(candidate_lateral) > relaxed_max_abs_lateral):
                            continue

                    progress = 0.0
                    if progress_origin is not None and progress_unit is not None:
                        progress = (
                            (wx - progress_origin[0]) * progress_unit[0]
                            + (wy - progress_origin[1]) * progress_unit[1])
                        if progress < self._goal_adjust_min_progress_m:
                            continue

                    score = dist + cost * 0.002 - progress * 0.03
                    frame_lateral_error = 0.0
                    if blue_goal_yaw is not None:
                        frame_lateral_error = abs(
                            -math.sin(blue_goal_yaw) * (wx - x)
                            + math.cos(blue_goal_yaw) * (wy - y))
                        if (relaxed_frame_lateral_limit > 0.0
                                and frame_lateral_error
                                > relaxed_frame_lateral_limit):
                            continue
                        score += frame_lateral_error * 3.0
                    if (blue_goal
                            and self._blue_goal_adjust_lateral_bias > 0.0):
                        side = blue_side
                        target_abs_lateral = max(
                            abs(target_lateral), self._blue_goal_adjust_min_abs_y_m)
                        if candidate_lateral * side < 0.0:
                            score += self._blue_goal_adjust_lateral_bias * 2.0
                        if abs(candidate_lateral) < target_abs_lateral:
                            score += (
                                self._blue_goal_adjust_lateral_bias
                                * (target_abs_lateral - abs(candidate_lateral)))
                        else:
                            score -= (
                                self._blue_goal_adjust_lateral_bias
                                * min(abs(candidate_lateral) - target_abs_lateral, 0.35) * 0.25)
                    if explore_goal:
                        target_abs_lateral = abs(target_lateral)
                        if candidate_lateral * explore_side <= 0.0:
                            score += 2.5
                        score += abs(abs(candidate_lateral) - target_abs_lateral) * 1.2
                        if abs(candidate_lateral) < max(0.20, target_abs_lateral - 0.25):
                            score += 1.5
                    if best_candidate is None or score < best_candidate[3]:
                        best_candidate = (wx, wy, cost, score)
            return best_candidate

        best = search_candidates(relaxed_blue=False)
        if best is None and blue_goal:
            best = search_candidates(relaxed_blue=True)
            if best is not None:
                self.get_logger().warn(
                    'Relaxed blue-door staging constraints to keep Nav2 goal in '
                    'connected free space before the FSM performs fine alignment.')
        return best

    def _is_costmap_cell_reachable(self, grid: OccupancyGrid,
                                   reachable: bytearray,
                                   cell) -> bool:
        if cell is None:
            return False
        mx, my = cell
        if mx is None or my is None:
            return False
        if not (0 <= mx < grid.info.width and 0 <= my < grid.info.height):
            return False
        return reachable[my * grid.info.width + mx] != 0

    def _reachable_costmap_cells(self, grid: OccupancyGrid) -> bytearray | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._navigation_frame, 'base_link', Time(),
                timeout=Duration(seconds=0.2))
        except TransformException:
            return None

        rx = transform.transform.translation.x
        ry = transform.transform.translation.y
        start = self._costmap_indices(grid, rx, ry)
        if start[0] is None or start[1] is None:
            return None

        max_cost = max(
            self._goal_adjust_max_cost,
            min(99, self._goal_adjust_connectivity_max_cost))
        start = self._nearest_traversable_start_cell(grid, start, max_cost)
        if start is None:
            return None

        width = grid.info.width
        height = grid.info.height
        data = grid.data
        visited = bytearray(width * height)
        sx, sy = start
        start_index = sy * width + sx
        visited[start_index] = 1
        queue = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for nx, ny in (
                    (cx + 1, cy),
                    (cx - 1, cy),
                    (cx, cy + 1),
                    (cx, cy - 1),
                    (cx + 1, cy + 1),
                    (cx + 1, cy - 1),
                    (cx - 1, cy + 1),
                    (cx - 1, cy - 1)):
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                index = ny * width + nx
                if visited[index]:
                    continue
                cost = int(data[index])
                if cost < 0 or cost > max_cost:
                    continue
                visited[index] = 1
                queue.append((nx, ny))

        return visited

    def _nearest_traversable_start_cell(self, grid: OccupancyGrid, start,
                                        max_cost: int):
        sx, sy = start
        width = grid.info.width
        height = grid.info.height
        data = grid.data
        start_cost = int(data[sy * width + sx])
        if 0 <= start_cost <= max_cost:
            return start

        res = max(1.0e-6, float(grid.info.resolution))
        radius_cells = max(
            1, int(math.ceil(self._goal_adjust_start_search_radius_m / res)))
        best = None
        for yy in range(sy - radius_cells, sy + radius_cells + 1):
            for xx in range(sx - radius_cells, sx + radius_cells + 1):
                if not (0 <= xx < width and 0 <= yy < height):
                    continue
                cost = int(data[yy * width + xx])
                if cost < 0 or cost > max_cost:
                    continue
                dist_cells = math.hypot(xx - sx, yy - sy)
                if dist_cells > radius_cells:
                    continue
                score = dist_cells + cost * 0.01
                if best is None or score < best[2]:
                    best = (xx, yy, score)
        if best is None:
            return None
        return best[0], best[1]

    def _send_goal_step(self, final_pose: PoseStamped):
        goal_pose = self._limited_goal_step(final_pose)
        goal_pose = self._adjust_pose_to_free_space(
            goal_pose,
            progress_to=final_pose if self._active_goal_is_intermediate else None)
        if goal_pose is None:
            self.get_logger().warn('Navigation step rejected because no free costmap cell exists near the goal.')
            self._navigating = False
            self._final_goal_pose = None
            self._active_goal_pose = None
            self._active_goal_is_intermediate = False
            self._current_goal_color = ''
            self._current_goal_door_id = ''
            self._current_goal_request_pose = None
            self._goal_start_time = None
            self._publish_done(success=False)
            return

        goal = NavigateToPose.Goal()
        send_pose = deepcopy(goal_pose)
        send_pose.header.stamp.sec = 0
        send_pose.header.stamp.nanosec = 0
        self._active_goal_pose = send_pose
        self._goal_start_time = self.get_clock().now()

        goal.pose = send_pose

        send_future = self.nav2_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        goal_sequence = self._goal_sequence
        send_future.add_done_callback(
            lambda future, seq=goal_sequence: self._goal_response_callback(
                future, seq))

    def _limited_goal_step(self, final_pose: PoseStamped) -> PoseStamped:
        if self._current_goal_color in ('blue', 'green', 'explore'):
            self._active_goal_is_intermediate = False
            return final_pose
        if self._max_goal_step_m <= 0.0:
            self._active_goal_is_intermediate = False
            return final_pose

        try:
            transform = self._tf_buffer.lookup_transform(
                self._navigation_frame, 'base_link', Time(),
                timeout=Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(
                f'Failed to get robot pose for stepped navigation goal: {e}')
            self._active_goal_is_intermediate = False
            return final_pose

        rx = transform.transform.translation.x
        ry = transform.transform.translation.y
        fx = final_pose.pose.position.x
        fy = final_pose.pose.position.y
        dx = fx - rx
        dy = fy - ry
        dist = math.hypot(dx, dy)
        if dist <= self._max_goal_step_m:
            self._active_goal_is_intermediate = False
            return final_pose

        ratio = self._max_goal_step_m / dist
        step_pose = deepcopy(final_pose)
        step_pose.pose.position.x = rx + dx * ratio
        step_pose.pose.position.y = ry + dy * ratio
        final_lateral = self._axis_lateral_xy(fx, fy)
        if (self._current_goal_color != 'explore'
                and self._lane_biased_intermediate_goals
                and abs(final_lateral) >= self._lane_goal_min_abs_y_m):
            lane_side = 1.0 if final_lateral > 0.0 else -1.0
            lane_abs_lateral = min(
                abs(final_lateral),
                max(self._lane_goal_min_abs_y_m, abs(final_lateral) * 0.78))
            step_progress = self._axis_progress_xy(
                step_pose.pose.position.x, step_pose.pose.position.y)
            step_lateral = self._axis_lateral_xy(
                step_pose.pose.position.x, step_pose.pose.position.y)
            if abs(step_lateral) < lane_abs_lateral:
                step_pose.pose.position.x, step_pose.pose.position.y = (
                    self._axis_to_map_xy(step_progress, lane_side * lane_abs_lateral))
        if self._goal_adjust_max_abs_y_m > 0.0:
            limit = self._goal_adjust_max_abs_y_m
            step_progress = self._axis_progress_xy(
                step_pose.pose.position.x, step_pose.pose.position.y)
            step_lateral = self._axis_lateral_xy(
                step_pose.pose.position.x, step_pose.pose.position.y)
            step_lateral = max(-limit, min(limit, step_lateral))
            step_pose.pose.position.x, step_pose.pose.position.y = (
                self._axis_to_map_xy(step_progress, step_lateral))
        self._active_goal_is_intermediate = True
        self.get_logger().info(
            f'Long goal split: final=({fx:.2f}, {fy:.2f}), '
            f'intermediate=({step_pose.pose.position.x:.2f}, '
            f'{step_pose.pose.position.y:.2f}), remaining={dist:.2f}m')
        return step_pose

    def _fixed_map_goal(self, pose: PoseStamped):
        source_frame = pose.header.frame_id.strip()
        if source_frame == self._navigation_frame:
            return pose
        if not source_frame:
            self.get_logger().error('Navigation goal has no frame_id.')
            return None

        try:
            transform = self._tf_buffer.lookup_transform(
                self._navigation_frame, source_frame, Time(),
                timeout=Duration(seconds=0.5))
            fixed_pose = do_transform_pose_stamped(pose, transform)
        except TransformException as e:
            self.get_logger().error(
                f'Failed to transform navigation goal from {source_frame} '
                f'to {self._navigation_frame}: {e}')
            return None

        self.get_logger().info(
            f'Fixed navigation goal in {self._navigation_frame}: '
            f'x={fixed_pose.pose.position.x:.2f}, '
            f'y={fixed_pose.pose.position.y:.2f}')
        return fixed_pose

    def _goal_response_callback(self, future, goal_sequence: int):
        goal_handle = future.result()
        if goal_sequence != self._goal_sequence:
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            self.get_logger().debug(
                'Ignoring Nav2 goal response from a canceled navigation generation.')
            return
        if not goal_handle.accepted:
            self.get_logger().warn('Navigation goal rejected by Nav2')
            self._navigating = False
            self._final_goal_pose = None
            self._active_goal_pose = None
            self._active_goal_is_intermediate = False
            self._goal_start_time = None
            self._publish_done(success=False)
            return

        self._current_goal_handle = goal_handle
        self.get_logger().info('Navigation goal accepted by Nav2')
        goal_handle.get_result_async().add_done_callback(
            lambda future, handle=goal_handle, seq=goal_sequence:
            self._result_callback(future, handle, seq))

    def _handle_reached_after_nav2_failure(self, status: int) -> bool:
        active_dist = self._distance_to_robot(self._active_goal_pose)
        final_dist = self._distance_to_robot(self._final_goal_pose)

        fallback_dist = self._goal_reached_fallback_dist()
        if (final_dist is not None and final_dist <= fallback_dist
                and self._precise_goal_yaw_ready(self._final_goal_pose)):
            self.get_logger().warn(
                f'Nav2 failed with status {status}, but robot is {final_dist:.2f}m '
                f'from final goal (fallback={fallback_dist:.2f}m). '
                'Treating navigation as succeeded.')
            self._navigating = False
            self._final_goal_pose = None
            self._active_goal_pose = None
            self._active_goal_is_intermediate = False
            self._goal_start_time = None
            self._publish_done(success=True)
            return True

        if (self._active_goal_is_intermediate and self._final_goal_pose is not None
                and active_dist is not None
                and active_dist <= self._fallback_goal_reached_dist_m):
            self.get_logger().warn(
                f'Nav2 failed with status {status}, but robot is {active_dist:.2f}m '
                'from intermediate goal. Continuing to final goal.')
            self._send_goal_step(self._final_goal_pose)
            return True

        return False

    def _goal_reached_fallback_dist(self) -> float:
        if self._current_goal_color in ('blue', 'green'):
            return min(self._fallback_goal_reached_dist_m, self._precise_goal_reached_dist_m)
        if self._current_goal_color == 'explore':
            return self._explore_goal_reached_dist_m
        return self._fallback_goal_reached_dist_m

    def _distance_to_robot(self, pose: PoseStamped | None) -> float | None:
        if pose is None:
            return None
        if pose.header.frame_id != self._navigation_frame:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._navigation_frame, 'base_link', Time(),
                timeout=Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().debug(f'Failed to get robot pose for failure fallback: {e}')
            return None

        rx = transform.transform.translation.x
        ry = transform.transform.translation.y
        px = pose.pose.position.x
        py = pose.pose.position.y
        return math.hypot(px - rx, py - ry)

    def _precise_goal_yaw_ready(self, pose: PoseStamped | None) -> bool:
        if self._current_goal_color == 'blue':
            return True
        if self._current_goal_color != 'green':
            return True
        yaw_error = self._goal_yaw_error_to_robot(pose)
        if yaw_error is None:
            return True
        if yaw_error <= self._precise_goal_reached_yaw_tolerance_rad:
            return True
        self.get_logger().debug(
            f'Precise goal is close but yaw is not ready: '
            f'error={math.degrees(yaw_error):.1f}deg, '
            f'tolerance={math.degrees(self._precise_goal_reached_yaw_tolerance_rad):.1f}deg',
            throttle_duration_sec=1.0)
        return False

    def _goal_yaw_error_to_robot(self, pose: PoseStamped | None) -> float | None:
        if pose is None:
            return None
        if pose.header.frame_id != self._navigation_frame:
            return None
        try:
            transform = self._tf_buffer.lookup_transform(
                self._navigation_frame, 'base_link', Time(),
                timeout=Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().debug(f'Failed to get robot yaw for precise fallback: {e}')
            return None

        robot_yaw = self._yaw_from_quaternion(transform.transform.rotation)
        goal_yaw = self._yaw_from_quaternion(pose.pose.orientation)
        return abs(self._normalize_angle(goal_yaw - robot_yaw))

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _axis_progress_xy(self, x: float, y: float) -> float:
        dx = x - self._mission_axis_origin_x
        dy = y - self._mission_axis_origin_y
        yaw = self._mission_axis_yaw
        return dx * math.cos(yaw) + dy * math.sin(yaw)

    def _axis_lateral_xy(self, x: float, y: float) -> float:
        dx = x - self._mission_axis_origin_x
        dy = y - self._mission_axis_origin_y
        yaw = self._mission_axis_yaw
        return -dx * math.sin(yaw) + dy * math.cos(yaw)

    def _axis_to_map_xy(self, progress: float, lateral: float) -> tuple[float, float]:
        yaw = self._mission_axis_yaw
        return (
            self._mission_axis_origin_x
            + progress * math.cos(yaw)
            - lateral * math.sin(yaw),
            self._mission_axis_origin_y
            + progress * math.sin(yaw)
            + lateral * math.cos(yaw),
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _result_callback(self, future, goal_handle, goal_sequence: int):
        if goal_sequence != self._goal_sequence:
            self.get_logger().debug(
                'Ignoring Nav2 result from a canceled navigation generation.')
            return
        goal_handle_id = id(goal_handle)
        if goal_handle_id in self._canceled_goal_handle_ids:
            self._canceled_goal_handle_ids.discard(goal_handle_id)
            self.get_logger().debug('Ignoring result from a goal canceled for retargeting.')
            return
        if goal_handle is not self._current_goal_handle:
            self.get_logger().debug('Ignoring stale Nav2 result from a preempted goal.')
            return
        if not self._navigating and self._final_goal_pose is None:
            return
        self._current_goal_handle = None
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            if self._active_goal_is_intermediate and self._final_goal_pose is not None:
                self.get_logger().info('Intermediate navigation goal reached. Continuing to final goal.')
                self._send_goal_step(self._final_goal_pose)
                return

            self._navigating = False
            self._final_goal_pose = None
            self._active_goal_pose = None
            self._current_goal_color = ''
            self._current_goal_door_id = ''
            self._current_goal_request_pose = None
            self._goal_start_time = None
            self.get_logger().info('Navigation succeeded.')
            self._publish_done(success=True)
        else:
            if self._handle_reached_after_nav2_failure(status):
                return

            self._navigating = False
            self._final_goal_pose = None
            self._active_goal_pose = None
            self._current_goal_color = ''
            self._current_goal_door_id = ''
            self._current_goal_request_pose = None
            self._goal_start_time = None
            self.get_logger().warn(f'Navigation failed with status: {status}')
            self._publish_done(success=False)

    def _feedback_callback(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        if dist is not None:
            self.get_logger().debug(f'Distance remaining: {dist:.2f} m')
        self._complete_if_close_enough()
        self._goal_watchdog_callback()

    def _complete_if_close_enough(self):
        if not self._early_success_cancel_enabled:
            return
        if not self._navigating:
            return
        if self._active_goal_is_intermediate:
            return
        if self._current_goal_color not in ('blue', 'green', 'explore'):
            return
        final_dist = self._distance_to_robot(self._final_goal_pose)
        if final_dist is None:
            return
        fallback_dist = self._goal_reached_fallback_dist()
        if final_dist > fallback_dist:
            return
        if not self._precise_goal_yaw_ready(self._final_goal_pose):
            return

        self.get_logger().warn(
            f'Robot is {final_dist:.2f}m from {self._current_goal_color} goal '
            f'(threshold={fallback_dist:.2f}m). Treating navigation as succeeded.')
        if self._current_goal_handle is not None:
            self._canceled_goal_handle_ids.add(id(self._current_goal_handle))
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._goal_sequence += 1
        self._navigating = False
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._current_goal_color = ''
        self._current_goal_door_id = ''
        self._current_goal_request_pose = None
        self._goal_start_time = None
        self._publish_done(success=True)

    def _goal_watchdog_callback(self):
        if not self._navigating:
            return
        if self._goal_start_time is None:
            return

        self._complete_if_close_enough()
        if not self._navigating:
            return

        if self._current_goal_color == 'explore':
            timeout_sec = self._explore_navigation_timeout_sec
            success_threshold = max(
                self._explore_goal_reached_dist_m,
                min(self._fallback_goal_reached_dist_m, 0.90))
            label = 'Explore'
        elif self._current_goal_color == 'green':
            timeout_sec = self._green_navigation_timeout_sec
            success_threshold = self._goal_reached_fallback_dist()
            label = 'Green'
        else:
            return

        if timeout_sec <= 0.0:
            return

        elapsed = (
            self.get_clock().now() - self._goal_start_time
        ).nanoseconds / 1e9
        if elapsed <= timeout_sec:
            return

        final_dist = self._distance_to_robot(self._final_goal_pose)
        success = final_dist is not None and final_dist <= success_threshold
        self.get_logger().warn(
            f'{label} navigation watchdog timeout ({elapsed:.1f}s). '
            f'final_dist={final_dist if final_dist is not None else -1.0:.2f}, '
            f'threshold={success_threshold:.2f}, success={success}.')
        if self._current_goal_handle is not None:
            self._canceled_goal_handle_ids.add(id(self._current_goal_handle))
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._goal_sequence += 1
        self._navigating = False
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._current_goal_color = ''
        self._current_goal_door_id = ''
        self._current_goal_request_pose = None
        self._goal_start_time = None
        self._publish_done(success=success)

    # ── 취소 ──────────────────────────────────────────────
    def _cancel_current_goal(self):
        self._goal_sequence += 1
        if self._current_goal_handle is not None:
            self._canceled_goal_handle_ids.add(id(self._current_goal_handle))
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._pending_nav2_pose = None
        self._pending_nav2_color = ''
        self._pending_nav2_start_time = None
        self._navigating = False
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._current_goal_color = ''
        self._current_goal_door_id = ''
        self._current_goal_request_pose = None
        self._goal_start_time = None

    # ── 완료 신호 발행 ────────────────────────────────────
    def _publish_done(self, success: bool):
        msg = Bool()
        msg.data = success
        self.nav_done_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
