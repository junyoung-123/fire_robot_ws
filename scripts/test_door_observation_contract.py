#!/usr/bin/env python3
"""Focused sensor/goal contract regressions; source ROS/install before running."""
import copy
import math
import unittest
from types import SimpleNamespace

from fire_robot_interfaces.msg import DoorInfo
from fire_robot_fsm.state_machine_node import StateMachineNode


def door(body=(6.0, 2.0), goal=(5.0, -1.0), handle=(6.3, 2.0)):
    msg = DoorInfo()
    msg.door_id = 'door_blue_test'
    msg.door_color = 'blue'
    msg.confidence = 0.9
    msg.door_pose.header.frame_id = 'map'
    msg.door_pose.pose.position.x, msg.door_pose.pose.position.y = goal
    msg.observed_door_position.header.frame_id = 'map'
    msg.observed_door_position.point.x, msg.observed_door_position.point.y = body
    msg.handle_position.header.frame_id = 'map'
    msg.handle_position.point.x, msg.handle_position.point.y = handle
    msg.handle_detected = True
    msg.handle_detection_method = 'yolo_primary'
    msg.handle_confidence = 0.8
    return msg


class DoorContractTests(unittest.TestCase):
    def setUp(self):
        self.fsm = object.__new__(StateMachineNode)
        self.fsm._axis_progress_xy = lambda x, y: x
        self.fsm._axis_lateral_xy = lambda x, y: y
        self.fsm._explore_center_y = 0.0
        self.fsm._observed_blue_min_abs_wall_y_m = 1.0
        self.fsm._axis_door_max_handle_pose_progress_delta_m = 1.0
        self.fsm.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)

    def test_identity_is_invariant_to_approach_goal_and_yaw(self):
        msg = door()
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            msg.door_pose.pose.position.x = 6.0 - 1.5 * math.cos(yaw)
            msg.door_pose.pose.position.y = 2.0 - 1.5 * math.sin(yaw)
            self.assertEqual(self.fsm._door_identity_xy(msg), (6.0, 2.0))

    def test_cross_corridor_goal_does_not_reject_same_wall_measurements(self):
        msg = door()
        self.assertFalse(self.fsm._blue_pose_handle_opposite_wall(msg))
        self.assertTrue(self.fsm._door_handle_consistent_with_pose(msg, (6.3, 2.0)))

    def test_genuinely_inconsistent_handle_still_rejected(self):
        msg = door(handle=(6.3, -2.0))
        self.assertTrue(self.fsm._blue_pose_handle_opposite_wall(msg))
        self.assertFalse(self.fsm._door_handle_consistent_with_pose(msg, (6.3, -2.0)))
        self.assertFalse(self.fsm._door_handle_consistent_with_pose(msg, (9.0, 2.0)))

    def test_merge_does_not_inherit_yolo_label_for_inferred_point(self):
        current = door()
        update = door(body=(6.2, 2.0), handle=(6.2, 2.0))
        update.handle_detected = False
        update.handle_detection_method = 'estimated'
        update.handle_confidence = 0.0
        self.fsm._merge_door_observation(current, update)
        self.assertAlmostEqual(current.observed_door_position.point.x, 6.05)
        self.assertFalse(current.handle_detected)
        self.assertEqual(current.handle_detection_method, 'estimated')
        self.assertEqual(current.handle_confidence, 0.0)
        self.assertAlmostEqual(current.handle_position.point.x, 6.2)

    def test_fresh_yolo_coordinates_are_not_blended_with_projection(self):
        current = door(handle=(7.5, 2.0))
        update = door(handle=(6.3, 2.0))
        self.fsm._merge_door_observation(current, update)
        self.assertAlmostEqual(current.handle_position.point.x, 6.3)
        update.handle_position.point.x = 100.0
        self.assertAlmostEqual(current.handle_position.point.x, 6.3)

    def test_observed_memory_id_keeps_trusted_handle_during_parking(self):
        msg = door(goal=(6.0, 2.0))
        msg.door_id = 'observed_blue_6p0_2p0'
        f = self.fsm
        f._axis_door_approach_enabled = True
        f._door_open_fresh_blue_max_dist_m = 1.0
        f._observed_candidate_merge_dist = lambda: 1.0
        f._find_observed_blue_cluster = lambda *a, **k: None
        f._is_blue_xy_recordable_wall_observation = lambda xy: True
        f._axis_side_door_front_parking_pose = lambda p, l: (p, 0.8, math.pi / 2, 2.0)
        f._axis_to_map_xy = lambda p, l: (p, l)
        f._axis_door_min_abs_lateral_m = 1.0
        result = f._door_with_axis_aligned_approach(msg)
        self.assertTrue(result.handle_detected)
        self.assertEqual(result.handle_detection_method, 'yolo_primary')
        self.assertAlmostEqual(result.handle_position.point.x, 6.3)
        self.assertAlmostEqual(result.door_pose.pose.position.y, 0.8)
        self.assertEqual(f._door_identity_xy(result), (6.0, 2.0))
        self.assertAlmostEqual(msg.door_pose.pose.position.y, 2.0)

    def test_nonfinite_observation_is_not_an_identity(self):
        msg = door(body=(float('nan'), 2.0))
        self.assertIsNone(self.fsm._observed_door_xy(msg))

    def test_workspace_checks_payload_not_parking_fallback(self):
        f = self.fsm
        f._current_map_pose = lambda: (6.0, 1.0, math.pi / 2)
        f._door_open_ready_max_dist_m = .2
        f._door_open_ready_lateral_tolerance_m = .1
        f._door_handle_xy = lambda msg: (6.0, 2.0)
        self.assertIn('handle_forward=', f._door_open_workspace_invalid_reason(door(handle=(4.0, .1))))
        self.assertEqual(f._door_open_workspace_invalid_reason(door(handle=(6.0, 2.0))), '')
        f._current_map_pose = lambda: None
        self.assertTrue(f._door_open_workspace_invalid_reason(door()))

    def test_final_refresh_requires_fresh_same_station_measured_point(self):
        f = self.fsm
        f.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))
        f._door_open_fresh_blue_max_age_sec = 8.0
        f._door_open_fresh_blue_max_dist_m = 1.2
        f._door_handle_xy = lambda msg: (6.0, 2.0)
        f._is_blue_xy_recordable_wall_observation = lambda xy: True
        stale = door()
        stale.handle_position.header.stamp.sec = 80
        other_wall = door(body=(6.0, -2.0), handle=(6.0, -2.0))
        other_wall.handle_position.header.stamp.sec = 99
        other_door = door(body=(8.0, 2.0), handle=(8.0, 2.0))
        other_door.handle_position.header.stamp.sec = 99
        projected = door()
        projected.handle_detected = False
        projected.handle_position.header.stamp.sec = 99
        f.detected_doors = [stale, other_wall, other_door, projected]
        self.assertIsNone(f._fresh_handle_for_locked_station(door()))
        fresh = door(handle=(6.3, 2.0))
        fresh.door_id = 'a_different_camera_track'
        fresh.handle_position.header.stamp.sec = 99
        f.detected_doors.append(fresh)
        self.assertIs(f._fresh_handle_for_locked_station(door()), fresh)

    def test_refresh_copies_geometry_and_provenance_without_changing_id(self):
        f = self.fsm
        target = door(handle=(3.0, 0.0))
        f.target_door = target
        f.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100_000_000_000))
        f._door_open_fresh_blue_max_age_sec = 8.0
        fresh = door(handle=(6.3, 2.0))
        fresh.handle_position.header.stamp.sec = 99
        f._fresh_handle_for_locked_station = lambda target: fresh
        f._blue_target_open_pose_aligned_for_safe_memory = lambda msg: True
        self.assertFalse(f._refresh_close_handle_before_opening())
        self.assertEqual(f.target_door.door_id, target.door_id)
        self.assertEqual(f.target_door.handle_position, fresh.handle_position)
        self.assertEqual(f.target_door.handle_detection_method, fresh.handle_detection_method)
        self.assertEqual(f.target_door.observed_door_position, target.observed_door_position)
        self.assertEqual(target.handle_position.point.x, 3.0)


if __name__ == '__main__':
    unittest.main()
