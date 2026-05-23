import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener
from nav2_msgs.action import NavigateToPose
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
        self._navigation_frame = self.get_parameter('navigation_frame').value

        cb_group = ReentrantCallbackGroup()

        self.nav2_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=cb_group)

        self.goal_sub = self.create_subscription(
            DoorInfo, '/target_door', self.target_door_callback, 10,
            callback_group=cb_group)

        self.nav_done_pub = self.create_publisher(Bool, '/navigation_done', 10)

        self._current_goal_handle = None
        self._navigating = False
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self.get_logger().info('NavigationNode started')

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
        self._navigate_to_pose(msg.door_pose)

    # ── Nav2 목표 전송 ────────────────────────────────────
    def _navigate_to_pose(self, pose: PoseStamped):
        if not self.nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available')
            self._publish_done(success=False)
            return

        pose = self._fixed_map_goal(pose)
        if pose is None:
            self._publish_done(success=False)
            return

        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.pose.header.stamp.sec = 0
        goal.pose.header.stamp.nanosec = 0
        self._navigating = True

        send_future = self.nav2_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        send_future.add_done_callback(self._goal_response_callback)

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
            self._publish_done(success=False)
            return

        self._current_goal_handle = goal_handle
        self.get_logger().info('Navigation goal accepted by Nav2')
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self._navigating = False
        self._current_goal_handle = None
        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded.')
            self._publish_done(success=True)
        else:
            self.get_logger().warn(f'Navigation failed with status: {status}')
            self._publish_done(success=False)

    def _feedback_callback(self, feedback_msg):
        dist = feedback_msg.feedback.distance_remaining
        if dist is not None:
            self.get_logger().debug(f'Distance remaining: {dist:.2f} m')

    # ── 취소 ──────────────────────────────────────────────
    def _cancel_current_goal(self):
        if self._current_goal_handle is not None:
            self._current_goal_handle.cancel_goal_async()
            self._current_goal_handle = None
        self._navigating = False

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
