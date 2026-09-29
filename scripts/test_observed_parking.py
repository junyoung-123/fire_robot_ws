import math
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from rclpy.time import Time

from fire_robot_fsm.state_machine_node import StateMachineNode, State


class ObservedParkingTests(unittest.TestCase):
    def node(self, bounded=False):
        node = object.__new__(StateMachineNode)
        node._explore_center_y = 0.0
        node._axis_door_min_abs_lateral_m = .45
        node._axis_door_side_standoff_m = 1.1
        node._observed_blue_min_abs_wall_y_m = 0.0
        node._observed_blue_max_abs_wall_y_m = 0.0
        node._axis_door_lane_bounds_enabled = bounded
        node._axis_door_min_side_goal_lateral_m = .75
        node._nav_start_max_abs_y_m = .95
        node._mission_forward_yaw = lambda: 0.0
        return node

    def test_observed_wall_range_preserved_for_both_sides(self):
        for lateral in (-3.4, -1.5, 1.5, 2.8, 3.4):
            node = self.node()
            progress, parking, yaw, wall = node._axis_side_door_front_parking_pose(6.0, lateral)
            self.assertEqual(progress, 6.0)
            self.assertAlmostEqual(abs(wall-parking), 1.1)
            self.assertEqual(wall, lateral)
            self.assertAlmostEqual(yaw, math.copysign(math.pi/2, lateral))

    def test_observation_translation_moves_goal_with_wall(self):
        node = self.node()
        first = node._axis_side_door_front_parking_pose(6.0, 1.9)
        second = node._axis_side_door_front_parking_pose(6.0, 2.8)
        self.assertAlmostEqual(second[1]-first[1], .9)

    def test_legacy_bounded_profile_is_unchanged(self):
        node = self.node(bounded=True)
        self.assertEqual(node._axis_side_door_front_parking_pose(6.0, 2.8)[1], .95)

    def test_missing_normal_does_not_destroy_navigation_position_error(self):
        node = self.node()
        node._door_alignment_observed_normal = True
        node._door_open_target_pose_xy_yaw = lambda _: (1., .2, .7)
        node._current_map_pose = lambda: (1., 0., 0.)
        node._observed_door_wall_normal = lambda _: None
        error = node._door_open_pose_error(None)
        self.assertAlmostEqual(error[0], .2)
        self.assertAlmostEqual(error[3], .2)

    def test_missing_normal_stops_and_waits_before_retry(self):
        node = self.node()
        node._door_alignment_observed_normal = True
        node._observed_door_wall_normal = lambda _: None
        node.cmd_vel_pub = Mock()
        node.get_logger = Mock()
        clock = SimpleNamespace(now=lambda: Time(seconds=10))
        node.get_clock = lambda: clock
        self.assertEqual(node._prepare_door_opening_pose(None), 'waiting')
        node.cmd_vel_pub.publish.assert_called_once()
        clock.now = lambda: Time(seconds=16)
        self.assertEqual(node._prepare_door_opening_pose(None), 'retry')

    def test_final_observation_refresh_applies_to_detector_ids_in_both_directions(self):
        for shift in (-.5, .5):
            node = self.node()
            node._door_alignment_observed_normal = True
            node.target_door = SimpleNamespace(door_id='door_blue_detector', door_color='blue', x=4., y=2.)
            node._door_has_map_identity = lambda _: True
            node._blue_target_open_pose_aligned_for_safe_memory = lambda _: True
            node._door_final_close_refresh_count = 0
            node._door_handle_xy = lambda d: (d.x, d.y)
            node._axis_progress_xy = lambda x, y: x
            node._axis_lateral_xy = lambda x, y: y
            node._door_open_fresh_blue_max_age_sec = 3.
            node._locked_target_close_station_max_forward_correction_m = .8
            node._door_open_fresh_blue_min_confidence = .6
            node._observed_blue_fresh_evidence_max_lateral_gap_m = .5
            node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=10))
            node._stable_observed_blue_doors = lambda: [dict(x=4.+shift, y=2., count=4., confidence=.9, last_seen=9.)]
            node._cluster_target_xy = lambda c: (c['x'], c['y'])
            node._is_observed_blue_cluster_suppressed = lambda _: False
            node._is_blue_xy_recordable_wall_observation = lambda _: True
            node._is_blue_position_opened = lambda _: False
            node._is_blue_position_abandoned = lambda _: False
            node._is_blue_xy_near_observed_red = lambda *args: False
            node._observed_blue_cluster_has_fresh_direct_evidence = lambda _: True
            node._observed_blue_cluster_to_door = lambda c: SimpleNamespace(x=c['x'], y=c['y'], door_id='memory')
            node._door_with_axis_aligned_approach = lambda d: d
            node._set_locked_blue_anchor = Mock()
            node._reset_door_nav_stuck_watch = Mock()
            node.get_logger = Mock()
            node.target_door_pub = Mock()
            self.assertTrue(node._refresh_observed_blue_target_before_opening())
            self.assertEqual(node.target_door.door_id, 'door_blue_detector')
            self.assertEqual(node.target_door.x, 4.+shift)
            self.assertEqual(node._door_final_close_refresh_count, 1)

    def test_close_observed_pose_does_not_bypass_configured_yaw_tolerance(self):
        class ContinueAlignment(Exception):
            pass

        for observed, yaw_degrees, expected in (
                (True, 9., 'control'), (True, 3., 'ready'), (False, 9., 'ready')):
            node = self.node()
            node._door_alignment_observed_normal = observed
            node._observed_door_wall_normal = lambda _: 0.
            node._fine_alignment_path_clear = lambda _: True
            node._door_open_pose_error = lambda _: (.2, math.radians(yaw_degrees), .2, 0.)
            node._door_open_ready_max_dist_m = .18
            node._door_open_ready_lateral_tolerance_m = .15
            node._door_open_ready_yaw_tolerance = math.radians(4.)
            node._door_open_straight_ready_max_dist_m = .24
            node._door_open_fine_control_max_dist_m = .6
            node._door_open_fine_lateral_tolerance_m = .4
            node.get_clock = Mock(side_effect=ContinueAlignment)
            node.get_logger = Mock()
            door = SimpleNamespace(door_id='observed_blue_test')
            if expected == 'control':
                with self.assertRaises(ContinueAlignment):
                    node._prepare_door_opening_pose(door)
            else:
                self.assertEqual(node._prepare_door_opening_pose(door), 'ready')

    def test_observed_handle_is_not_replaced_by_panel_estimate(self):
        class ContinueRefinement(Exception):
            pass

        def door(trusted):
            return SimpleNamespace(door_id='observed_blue_test', door_color='blue',
                handle_detected=trusted, handle_confidence=.8,
                handle_detection_method='yolo:primary:item' if trusted else 'estimated',
                handle_position=SimpleNamespace(header=SimpleNamespace(frame_id='map'),
                                                point=SimpleNamespace(x=5., y=-2.)))

        for observed, new_trusted, blocked in (
                (True, False, True), (True, True, False), (False, False, False)):
            node = self.node()
            node._door_alignment_observed_normal = observed
            node._locked_target_refine_enabled = True
            node.state = State.NAVIGATING
            node._local_obstacle_escape_start_time = None
            node._nav_start_pose_recovery_active = False
            node._door_retarget_backoff_until = None
            node._door_open_settle_start_time = None
            node._door_open_align_start_time = None
            node.target_door = door(True)
            node._door_has_map_identity = lambda _: True
            node._observation_matches_locked_observed_blue_cluster = Mock(
                side_effect=ContinueRefinement)
            if blocked:
                node._maybe_refine_locked_blue_target(door(new_trusted))
                node._observation_matches_locked_observed_blue_cluster.assert_not_called()
                self.assertTrue(node._has_trusted_observed_handle(node.target_door))
                self.assertFalse(node._refresh_observed_blue_target_before_opening())
            else:
                with self.assertRaises(ContinueRefinement):
                    node._maybe_refine_locked_blue_target(door(new_trusted))

    def test_close_handle_refinement_needs_fresh_spatial_agreement(self):
        node = self.node()
        node._door_handle_xy = lambda d: (d.x, d.y)
        door = SimpleNamespace(door_id='one', x=5., y=2.)
        check = lambda t: node._consistent_close_handle_refinement(door, Time(seconds=t))
        self.assertFalse(check(1.))
        self.assertFalse(check(1.05))
        door.x = 5.4
        self.assertFalse(check(1.3))
        door.x = 5.0
        self.assertFalse(check(1.6))
        self.assertFalse(check(1.9))
        self.assertTrue(check(2.2))
        self.assertFalse(check(5.))
        door.door_id = 'other'
        self.assertFalse(check(5.3))

    def test_completed_source_memory_is_excluded_without_expanding_merge_radius(self):
        node = self.node()
        node._door_alignment_observed_normal = True
        original = dict(x=9., y=1.2)
        neighbor = dict(x=9.5, y=2.)
        node._observed_blue_door_id_to_cluster = {'selected': original, 'neighbor': neighbor}
        node.get_logger = Mock()
        door = SimpleNamespace(door_id='selected', door_color='blue')
        node._complete_selected_observed_cluster(door)
        self.assertTrue(node._is_observed_blue_cluster_suppressed(original))
        self.assertNotIn('opened', neighbor)

    def test_refinement_rebinding_does_not_lose_original_memory(self):
        node = self.node()
        node._door_alignment_observed_normal = True
        node._locked_blue_anchor_key = None
        node._locked_blue_anchor_xy = None
        node._door_has_map_identity = lambda _: True
        node._door_identity_xy = lambda d: (d.x, d.y)
        node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=10))
        node.get_logger = Mock()
        original, refined, neighbor = {}, {}, {}
        node._observed_blue_door_id_to_cluster = {'selected': original, 'neighbor': neighbor}
        door = SimpleNamespace(door_id='selected', door_color='blue', x=9., y=1.2)
        node._set_locked_blue_anchor(door)
        node._observed_blue_door_id_to_cluster['selected'] = refined
        door.x, door.y = 10.7, 1.9
        node._set_locked_blue_anchor(door)
        node._complete_selected_observed_cluster(door)
        self.assertEqual(original['opened'], 1.)
        self.assertEqual(refined['opened'], 1.)
        self.assertNotIn('opened', neighbor)
        node._clear_locked_blue_anchor()
        self.assertIsNone(node._locked_blue_source_cluster)
        node._door_alignment_observed_normal = False
        door.door_id = 'neighbor'
        node._complete_selected_observed_cluster(door)
        self.assertNotIn('opened', neighbor)


if __name__ == '__main__':
    unittest.main()
