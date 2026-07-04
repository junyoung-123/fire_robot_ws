from copy import deepcopy
import math

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
        self.declare_parameter('fallback_goal_reached_dist_m', 0.85)
        self.declare_parameter('precise_goal_reached_dist_m', 0.30)
        self.declare_parameter('explore_goal_reached_dist_m', 0.55)
        self.declare_parameter('explore_navigation_timeout_sec', 75.0)
        self.declare_parameter('lane_biased_intermediate_goals', False)
        self.declare_parameter('lane_goal_min_abs_y_m', 0.75)
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
        self._fallback_goal_reached_dist_m = float(
            self.get_parameter('fallback_goal_reached_dist_m').value)
        self._precise_goal_reached_dist_m = float(
            self.get_parameter('precise_goal_reached_dist_m').value)
        self._explore_goal_reached_dist_m = float(
            self.get_parameter('explore_goal_reached_dist_m').value)
        self._explore_navigation_timeout_sec = float(
            self.get_parameter('explore_navigation_timeout_sec').value)
        self._lane_biased_intermediate_goals = bool(
            self.get_parameter('lane_biased_intermediate_goals').value)
        self._lane_goal_min_abs_y_m = float(
            self.get_parameter('lane_goal_min_abs_y_m').value)

        cb_group = ReentrantCallbackGroup()

        self.nav2_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=cb_group)

        self.goal_sub = self.create_subscription(
            DoorInfo, '/target_door', self.target_door_callback, 10,
            callback_group=cb_group)
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback,
            costmap_qos, callback_group=cb_group)

        self.nav_done_pub = self.create_publisher(Bool, '/navigation_done', 10)

        self._current_goal_handle = None
        self._navigating = False
        self._final_goal_pose: PoseStamped | None = None
        self._active_goal_pose: PoseStamped | None = None
        self._active_goal_is_intermediate = False
        self._latest_costmap: OccupancyGrid | None = None
        self._current_goal_color = ''
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._goal_start_time = None

        self.create_timer(0.5, self._goal_watchdog_callback)
        self.get_logger().info('NavigationNode started')

    def costmap_callback(self, msg: OccupancyGrid):
        self._latest_costmap = msg

    # ── 목표 수신 ─────────────────────────────────────────
    def target_door_callback(self, msg: DoorInfo):
        if self._navigating:
            self.get_logger().warn(
                f'Already navigating. Cancelling current goal to go to: {msg.door_id}')
            self._cancel_current_goal()

        self.get_logger().info(
            f'Navigation goal received: door {msg.door_id} '
            f'(color={msg.door_color}, '
            f'frame={msg.door_pose.header.frame_id}, '
            f'x={msg.door_pose.pose.position.x:.2f}, '
            f'y={msg.door_pose.pose.position.y:.2f})'
        )
        self._current_goal_color = msg.door_color
        self._navigate_to_pose(msg.door_pose)

    # ── Nav2 목표 전송 ────────────────────────────────────
    def _navigate_to_pose(self, pose: PoseStamped):
        if not self.nav2_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('Nav2 action server not available')
            self._publish_done(success=False)
            return

        pose = self._fixed_map_goal(pose)
        if pose is None:
            self._publish_done(success=False)
            return

        pose = self._adjust_pose_to_free_space(pose, progress_to=pose)
        self._final_goal_pose = pose
        self._navigating = True
        self._send_goal_step(pose)

    def _adjust_pose_to_free_space(self, pose: PoseStamped,
                                   progress_to: PoseStamped | None = None) -> PoseStamped:
        grid = self._latest_costmap
        if not self._adjust_goal_to_free_space or grid is None:
            return pose
        if pose.header.frame_id != grid.header.frame_id:
            return pose

        x = pose.pose.position.x
        y = pose.pose.position.y
        cost = self._costmap_value(grid, x, y)
        if cost == -1:
            return pose
        if cost is not None and 0 <= cost <= self._goal_adjust_max_cost:
            return pose

        best = self._nearest_free_costmap_point(grid, x, y, progress_to)
        if best is None:
            self.get_logger().warn(
                f'Navigation goal is not free (cost={cost}) and no nearby free cell was found.')
            return pose

        adjusted = deepcopy(pose)
        adjusted.pose.position.x = best[0]
        adjusted.pose.position.y = best[1]
        self.get_logger().warn(
            f'Adjusted occupied navigation goal from ({x:.2f}, {y:.2f}, cost={cost}) '
            f'to free cell ({best[0]:.2f}, {best[1]:.2f}, cost={best[2]})')
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
                                    progress_to: PoseStamped | None = None):
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
        for yy in range(cy - radius_cells, cy + radius_cells + 1):
            for xx in range(cx - radius_cells, cx + radius_cells + 1):
                if not (0 <= xx < grid.info.width and 0 <= yy < grid.info.height):
                    continue
                cost = int(grid.data[yy * grid.info.width + xx])
                if cost < 0 or cost > self._goal_adjust_max_cost:
                    continue
                wx = ox + (xx + 0.5) * res
                wy = oy + (yy + 0.5) * res
                dist = math.hypot(wx - x, wy - y)
                if dist > self._goal_adjust_search_radius_m:
                    continue
                if (self._goal_adjust_max_abs_y_m > 0.0
                        and abs(wy) > self._goal_adjust_max_abs_y_m):
                    continue

                progress = 0.0
                if progress_origin is not None and progress_unit is not None:
                    progress = (
                        (wx - progress_origin[0]) * progress_unit[0]
                        + (wy - progress_origin[1]) * progress_unit[1])
                    if progress < self._goal_adjust_min_progress_m:
                        continue

                score = dist + cost * 0.002 - progress * 0.03
                if best is None or score < best[3]:
                    best = (wx, wy, cost, score)
        return best

    def _send_goal_step(self, final_pose: PoseStamped):
        goal_pose = self._limited_goal_step(final_pose)
        goal_pose = self._adjust_pose_to_free_space(
            goal_pose,
            progress_to=final_pose if self._active_goal_is_intermediate else None)

        goal = NavigateToPose.Goal()
        self._active_goal_pose = goal_pose
        self._goal_start_time = self.get_clock().now()

        goal.pose = goal_pose
        goal.pose.header.stamp.sec = 0
        goal.pose.header.stamp.nanosec = 0

        send_future = self.nav2_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        send_future.add_done_callback(self._goal_response_callback)

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
        if (self._current_goal_color != 'explore'
                and self._lane_biased_intermediate_goals
                and abs(fy) >= self._lane_goal_min_abs_y_m):
            lane_side = 1.0 if fy > 0.0 else -1.0
            lane_abs_y = min(abs(fy), max(self._lane_goal_min_abs_y_m, abs(fy) * 0.78))
            if abs(step_pose.pose.position.y) < lane_abs_y:
                step_pose.pose.position.y = lane_side * lane_abs_y
        if self._goal_adjust_max_abs_y_m > 0.0:
            limit = self._goal_adjust_max_abs_y_m
            step_pose.pose.position.y = max(
                -limit, min(limit, step_pose.pose.position.y))
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

    def _goal_response_callback(self, future):
        goal_handle = future.result()
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
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _handle_reached_after_nav2_failure(self, status: int) -> bool:
        active_dist = self._distance_to_robot(self._active_goal_pose)
        final_dist = self._distance_to_robot(self._final_goal_pose)

        fallback_dist = self._goal_reached_fallback_dist()
        if final_dist is not None and final_dist <= fallback_dist:
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

    def _result_callback(self, future):
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

        self.get_logger().warn(
            f'Robot is {final_dist:.2f}m from {self._current_goal_color} goal '
            f'(threshold={fallback_dist:.2f}m). Treating navigation as succeeded.')
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._navigating = False
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._current_goal_color = ''
        self._goal_start_time = None
        self._publish_done(success=True)

    def _goal_watchdog_callback(self):
        if not self._navigating:
            return
        if self._current_goal_color != 'explore':
            return
        if self._goal_start_time is None:
            return
        if self._explore_navigation_timeout_sec <= 0.0:
            return

        elapsed = (
            self.get_clock().now() - self._goal_start_time
        ).nanoseconds / 1e9
        if elapsed <= self._explore_navigation_timeout_sec:
            return

        final_dist = self._distance_to_robot(self._final_goal_pose)
        success_threshold = max(
            self._explore_goal_reached_dist_m,
            min(self._fallback_goal_reached_dist_m, 0.90))
        success = final_dist is not None and final_dist <= success_threshold
        self.get_logger().warn(
            f'Explore navigation watchdog timeout ({elapsed:.1f}s). '
            f'final_dist={final_dist if final_dist is not None else -1.0:.2f}, '
            f'threshold={success_threshold:.2f}, success={success}.')
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._navigating = False
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._current_goal_color = ''
        self._goal_start_time = None
        self._publish_done(success=success)

    # ── 취소 ──────────────────────────────────────────────
    def _cancel_current_goal(self):
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._navigating = False
        self._final_goal_pose = None
        self._active_goal_pose = None
        self._active_goal_is_intermediate = False
        self._current_goal_color = ''
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
