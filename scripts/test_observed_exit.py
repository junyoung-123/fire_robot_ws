import math
import unittest
from unittest.mock import Mock
from types import SimpleNamespace as NS
from rclpy.time import Time

from fire_robot_fsm.state_machine_node import StateMachineNode, State


class ObservedExitTests(unittest.TestCase):
    def node(self, observed=True):
        node = object.__new__(StateMachineNode)
        node._exit_use_observed_lateral = observed
        node._explore_center_y = 0.0
        node._detected_exit_max_center_y_m = .85
        node._exit_candidate_relaxed_after_opened_scan = lambda *a, **kw: False
        node._axis_progress_xy = lambda x, y: x
        node._axis_lateral_xy = lambda x, y: y
        node._axis_to_map_xy = lambda p, l: (p, l)
        node._mission_forward_yaw = lambda: 0.0
        node._exit_pass_through_m = .8
        node._exit_nav_standoff_m = .6
        return node

    def test_observed_exit_not_biased_toward_initial_center(self):
        node = self.node()
        for lateral in (-1.7, -.7, .7, 1.7):
            nav, complete = node._exit_goal_pair_from_xy((20., lateral))
            self.assertEqual(nav[1], lateral)
            self.assertEqual(complete[1], lateral)
            self.assertEqual(node._exit_reference_lateral(complete), lateral)

    def test_legacy_center_bias_unchanged(self):
        node = self.node(False)
        nav, complete = node._exit_goal_pair_from_xy((20., .7))
        self.assertAlmostEqual(nav[1], .7*.35)
        self.assertAlmostEqual(complete[1], .7*.35)
        self.assertEqual(node._exit_reference_lateral(complete), 0.)

    def test_direct_crossing_preserves_observed_lateral(self):
        node = self.node()
        node._current_map_pose = lambda: (19., .9, 0.)
        node._active_exit_goal = (20., 1.1, 0.)
        node._prepare_observed_exit_direct_crossing_goal()
        self.assertEqual(node._active_exit_goal, (20., 1.1, 0.))

    def test_rotated_axis_keeps_observed_goal(self):
        node = self.node()
        angle = .2
        c, s = math.cos(angle), math.sin(angle)
        node._axis_progress_xy = lambda x, y: x*c+y*s
        node._axis_lateral_xy = lambda x, y: -x*s+y*c
        node._mission_forward_yaw = lambda: angle
        nav, complete = node._exit_goal_pair_from_xy((19., 4.))
        expected = node._axis_lateral_xy(19., 4.)
        self.assertAlmostEqual(node._exit_reference_lateral(nav), expected)
        self.assertAlmostEqual(node._exit_reference_lateral(complete), expected)

    def test_nav2_failure_replans_without_manual_crossing(self):
        node = self.node()
        node._nav_done = False
        node._nav_failed = True
        node._exit_retry_ready = lambda: True
        node._send_exit_goal = Mock()
        node._start_exit_crossing = Mock()
        node._continue_nav2_exit_crossing()
        node._send_exit_goal.assert_called_once()
        node._start_exit_crossing.assert_not_called()

    def test_nav2_done_requires_pose_confirmation(self):
        for complete in (False, True):
            node = self.node()
            node._nav_done = True
            node._exit_pose_complete = lambda: complete
            node._send_exit_goal = Mock()
            node._transition = Mock()
            node._continue_nav2_exit_crossing()
            if complete:
                node._transition.assert_called_once_with(State.MISSION_COMPLETE)
            else:
                node._send_exit_goal.assert_called_once()
                node._transition.assert_not_called()
                self.assertTrue(node._exit_nav2_approach_complete)

    def test_color_goal_alone_cannot_complete_observed_exit(self):
        node = self.node()
        node._exit_observed_aperture = True
        node._current_map_pose = lambda: (30., 0., 0.)
        node._active_exit_goal = (20., 0., 0.)
        self.assertFalse(node._exit_pose_complete())

    def test_green_hint_does_not_require_initial_centerline(self):
        node = self.node()
        node._exit_observed_aperture = True
        node._use_detected_exit = True
        node._exit_door = object()
        node._exit_observation_xy = lambda door: (20., 3.)
        node._exit_candidate_axis_valid = Mock(side_effect=AssertionError('prior centerline'))
        self.assertTrue(node._valid_detected_exit())

    def test_observed_plane_requires_body_to_clear_inside_aperture(self):
        node = self.node()
        node._exit_observed_aperture = True
        node._observed_exit_plane = (20., 1., 0., 1.2)
        for pose, expected in (((19.9, 1., 0.), False),
                               ((20.2, 1., 0.), False),
                               ((20.5, 1., 0.), True),
                               ((20.5, 1.5, 0.), False)):
            node._current_map_pose = lambda p=pose: p
            self.assertEqual(node._exit_pose_complete(), expected)

    def test_unconfirmed_approach_is_reobserved_not_crossed(self):
        node = self.node()
        node._exit_observed_aperture = True
        node._nav_done = True
        node._exit_pose_complete = lambda: False
        node._send_exit_goal = Mock()
        node._transition = Mock()
        node._continue_nav2_exit_crossing()
        self.assertFalse(node._exit_nav2_approach_complete)
        node._transition.assert_not_called()
        node._send_exit_goal.assert_called_once()

    def guarded_node(self):
        node = self.node()
        node._exit_observed_aperture = True
        node._active_exit_goal = (15., 5., 0.)
        node._latest_scan = object()
        node._latest_scan_received_sec = 10.
        node.get_clock = lambda: NS(now=lambda: Time(seconds=10))
        node.get_logger = lambda: Mock()
        node._publish_post_open_twist = Mock()
        node._request_navigation_cancel_for_manual_control = Mock()
        node._send_exit_goal = Mock()
        node._transition = Mock()
        node._nav_done, node._nav_failed = False, True
        node._exit_retry_ready = lambda: True
        scene = NS(pose=(10., 5., 0.), odom_pose=(2., 3., 0.),
                   safe=Mock(return_value=True))
        node._post_open_footprint_guard = NS(snapshot=Mock(return_value=scene),
            reset_recovery=Mock(), recovery_command=Mock(return_value=(.03, .08, 'TEST')))
        return node, scene

    def test_exit_recovery_requires_repeated_failure_without_wheel_progress(self):
        node, scene = self.guarded_node()
        node._continue_nav2_exit_crossing()
        node._send_exit_goal.assert_called_once()
        self.assertIsNone(getattr(node, '_exit_local_recovery', None))
        node._nav_failed = True
        node._continue_nav2_exit_crossing()
        self.assertIsNotNone(node._exit_local_recovery)
        self.assertEqual(node._exit_local_recovery[2], (7., 3.))
        node._request_navigation_cancel_for_manual_control.assert_called_once()
        self.assertEqual(node._send_exit_goal.call_count, 1)

    def test_repeated_hint_arrival_without_aperture_starts_reobservation(self):
        node, scene = self.guarded_node()
        node._exit_pose_complete = lambda: False
        node._nav_done, node._nav_failed = True, False
        node._continue_nav2_exit_crossing()
        node._send_exit_goal.assert_called_once()
        self.assertFalse(node._exit_nav2_approach_complete)
        node._nav_done = True
        node._continue_nav2_exit_crossing()
        self.assertIsNotNone(node._exit_local_recovery)
        self.assertEqual(node._send_exit_goal.call_count, 1)
        node._transition.assert_not_called()
        node._post_open_footprint_guard.recovery_command.assert_not_called()

    def test_valid_exit_plane_keeps_normal_crossing_goal_dispatch(self):
        node, scene = self.guarded_node()
        node._exit_pose_complete = lambda: False
        node._observed_exit_plane = (14., 5., 0., 1.5)
        node._nav_done = True
        node._continue_nav2_exit_crossing()
        node._send_exit_goal.assert_called_once()
        self.assertTrue(node._exit_nav2_approach_complete)
        self.assertIsNone(getattr(node, '_exit_local_recovery', None))

    def test_missing_geometry_does_not_turn_hint_arrival_into_success(self):
        node, scene = self.guarded_node()
        node._exit_pose_complete = lambda: False
        node._post_open_footprint_guard.snapshot.return_value = None
        node._nav_done = True
        node._continue_nav2_exit_crossing()
        node._transition.assert_not_called()
        node._request_navigation_cancel_for_manual_control.assert_not_called()
        self.assertFalse(node._exit_nav2_approach_complete)

    def test_exit_retry_progress_resets_recovery_attempts(self):
        node, scene = self.guarded_node()
        node._exit_failed_odom = (0., 3., 0.)
        node._exit_local_recovery_attempts = 3
        self.assertFalse(node._start_observed_exit_recovery())
        self.assertEqual(node._exit_local_recovery_attempts, 0)

    def test_exit_recovery_never_moves_without_fresh_geometry(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        node._post_open_footprint_guard.snapshot.return_value = None
        node._run_observed_exit_recovery()
        node._post_open_footprint_guard.recovery_command.assert_not_called()
        node._send_exit_goal.assert_not_called()
        command = node._publish_post_open_twist.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0., 0.))

    def test_slam_jump_cannot_complete_exit_escape(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        scene.pose = (50., 40., 2.)
        node._run_observed_exit_recovery()
        node._send_exit_goal.assert_not_called()
        node._post_open_footprint_guard.recovery_command.assert_called_once()

    def test_measured_escape_replans_without_declaring_exit_success(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        scene.odom_pose = (2.25, 3., 0.)
        node._run_observed_exit_recovery()
        node._send_exit_goal.assert_called_once()
        node._transition.assert_not_called()
        self.assertIsNone(node._exit_local_recovery)

    def test_exit_escape_handoff_requires_clear_forward_and_both_turns(self):
        for values in ((False, True, True), (True, False, True), (True, True, False)):
            node, scene = self.guarded_node()
            node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
            scene.odom_pose = (2.25, 3., 0.)
            scene.safe.side_effect = values
            node._run_observed_exit_recovery()
            node._send_exit_goal.assert_not_called()

    def test_exit_escape_deadline_is_failure_not_handoff(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (-51., (2., 3., 0.), (7., 3.))
        node._run_observed_exit_recovery()
        node._transition.assert_called_once_with(State.EMERGENCY_STOP)
        node._send_exit_goal.assert_not_called()

    def test_exit_escape_rejects_odometry_jump(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        scene.odom_pose = (5., 3., 0.)
        node._run_observed_exit_recovery()
        node._transition.assert_called_once_with(State.EMERGENCY_STOP)
        node._send_exit_goal.assert_not_called()

    def test_exit_escape_retry_limit_stops(self):
        node, scene = self.guarded_node()
        node._exit_failed_odom = scene.odom_pose
        node._exit_local_recovery_attempts = 3
        self.assertTrue(node._start_observed_exit_recovery())
        node._transition.assert_called_once_with(State.EMERGENCY_STOP)

    def test_exploration_stall_uses_own_pose_not_exit_or_world_geometry(self):
        node, scene = self.guarded_node()
        node._exit_observed_aperture = False
        node._active_exit_goal = None
        node._explore_nav_goal_xy = (15., 5.)
        self.assertFalse(node._start_observed_exit_recovery(exploring=True))
        self.assertTrue(node._start_observed_exit_recovery(exploring=True))
        self.assertEqual(node._observed_recovery_scope, 'explore')
        self.assertEqual(node._exit_local_recovery[2], (7., 3.))

    def test_exploration_handoff_resends_waypoint_not_exit(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        node._observed_recovery_scope = 'explore'
        node._interrupt_explore_waypoint_for_blue_door = Mock(return_value=False)
        node._resend_active_explore_goal = Mock(return_value=True)
        scene.odom_pose = (2.25, 3., 0.)
        node._run_observed_exit_recovery()
        node._resend_active_explore_goal.assert_called_once()
        node._send_exit_goal.assert_not_called()
        node._transition.assert_not_called()

    def test_cancelled_stuck_goal_can_start_recovery_before_failure_reset(self):
        node, scene = self.guarded_node()
        node._run_explore_nav_backoff = lambda: False
        node._interrupt_explore_waypoint_for_blue_door = lambda: False
        node._nav_done = node._nav_failed = False
        node._explore_waypoint_wall_stuck = lambda: True
        node._start_observed_exit_recovery = Mock(return_value=True)
        node._on_exploring_navigating()
        node._start_observed_exit_recovery.assert_called_once_with(exploring=True)
        node._transition.assert_not_called()

    def test_new_blue_goal_does_not_discard_active_checked_escape(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        node._interrupt_explore_waypoint_for_blue_door = Mock(return_value=True)
        node._run_observed_exit_recovery = Mock()
        node._on_exploring_navigating()
        node._run_observed_exit_recovery.assert_called_once()
        node._interrupt_explore_waypoint_for_blue_door.assert_not_called()

    def test_saved_blue_candidate_is_selected_at_safe_escape_handoff(self):
        node, scene = self.guarded_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        node._observed_recovery_scope = 'explore'
        node._interrupt_explore_waypoint_for_blue_door = Mock(return_value=True)
        node._resend_active_explore_goal = Mock()
        scene.odom_pose = (2.25, 3., 0.)
        node._run_observed_exit_recovery()
        node._interrupt_explore_waypoint_for_blue_door.assert_called_once()
        node._resend_active_explore_goal.assert_not_called()
        self.assertIsNone(node._exit_local_recovery)

    def blue_node(self):
        node, scene = self.guarded_node()
        node._active_exit_goal = None
        node._exit_observed_aperture = False
        node.target_door = NS(door_id='observed-A', door_color='blue',
            door_pose=NS(header=NS(frame_id='map'), pose=NS(position=NS(x=15., y=5.))))
        node.target_door_pub = NS(publish=Mock())
        node._reset_door_nav_stuck_watch = Mock()
        return node, scene

    def test_locked_blue_recovery_uses_observed_navigation_goal(self):
        node, scene = self.blue_node()
        self.assertFalse(node._start_observed_exit_recovery(blue=True))
        self.assertTrue(node._start_observed_exit_recovery(blue=True))
        self.assertEqual(node._observed_recovery_scope, 'blue')
        self.assertEqual(node._observed_recovery_blue_id, 'observed-A')
        self.assertEqual(node._exit_local_recovery[2], (7., 3.))

    def test_blue_handoff_republishes_same_target_not_exit_or_new_candidate(self):
        node, scene = self.blue_node()
        node._start_observed_exit_recovery(blue=True)
        node._start_observed_exit_recovery(blue=True)
        node.get_clock = lambda: NS(now=lambda: Time(seconds=15))
        scene.odom_pose = (2.25, 3., 0.)
        original = node.target_door
        node._run_observed_exit_recovery()
        node.target_door_pub.publish.assert_called_once_with(original)
        node._reset_door_nav_stuck_watch.assert_called_once()
        node._send_exit_goal.assert_not_called()
        self.assertIsNone(node._exit_local_recovery)

    def test_blue_escape_owns_navigation_tick_until_handoff(self):
        node, scene = self.blue_node()
        node._exit_local_recovery = (5., (2., 3., 0.), (7., 3.))
        node._run_observed_exit_recovery = Mock()
        node._on_navigating()
        node._run_observed_exit_recovery.assert_called_once()
        node.target_door_pub.publish.assert_not_called()

    def test_changed_blue_identity_stops_before_issuing_escape_motion(self):
        node, scene = self.blue_node()
        node._start_observed_exit_recovery(blue=True)
        node._start_observed_exit_recovery(blue=True)
        node.target_door.door_id = 'observed-B'
        node._run_observed_exit_recovery()
        node._transition.assert_called_once_with(State.EMERGENCY_STOP)
        node._post_open_footprint_guard.recovery_command.assert_not_called()
        node.target_door_pub.publish.assert_not_called()

    def test_blue_recovery_rejects_missing_stale_or_unmapped_observation(self):
        for invalid in ('missing', 'frame', 'nonfinite', 'stale'):
            node, scene = self.blue_node()
            if invalid == 'missing': node.target_door = None
            if invalid == 'frame': node.target_door.door_pose.header.frame_id = 'base_link'
            if invalid == 'nonfinite': node.target_door.door_pose.pose.position.x = math.nan
            if invalid == 'stale': node._post_open_footprint_guard.snapshot.return_value = None
            self.assertFalse(node._start_observed_exit_recovery(blue=True), invalid)
            node._request_navigation_cancel_for_manual_control.assert_not_called()

    def test_failure_history_not_reused_between_exit_and_exploration(self):
        node, scene = self.guarded_node()
        node._exit_failed_odom = scene.odom_pose
        node._exit_local_recovery_attempts = 3
        node._observed_recovery_failure_scope = 'explore'
        self.assertFalse(node._start_observed_exit_recovery())
        self.assertEqual(node._exit_local_recovery_attempts, 0)


if __name__ == '__main__':
    unittest.main()
