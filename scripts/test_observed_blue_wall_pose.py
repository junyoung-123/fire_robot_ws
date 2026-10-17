import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
from rclpy.clock import ClockType
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from fire_robot_interfaces.msg import DoorInfo
from fire_robot_fsm.state_machine_node import StateMachineNode


class ObservedBlueWallPoseTests(unittest.TestCase):
    def node_and_door(self):
        node = object.__new__(StateMachineNode)
        node._axis_door_lane_bounds_enabled = False
        node._axis_door_approach_enabled = True
        node._axis_door_side_standoff_m = 1.1
        node._door_open_axis_min_side_lateral_m = .95
        node._observed_blue_min_abs_wall_y_m = 0.
        node._observed_blue_max_abs_wall_y_m = 0.
        node._axis_door_max_handle_pose_progress_delta_m = 2.
        node._explore_center_y = 0.
        node._axis_lateral_xy = lambda x, y: y
        node._axis_progress_xy = lambda x, y: x
        node._door_has_map_identity = lambda door: True
        node.get_clock = lambda: NS(now=lambda: Time(seconds=10, clock_type=ClockType.ROS_TIME))
        node.get_logger = lambda: Mock()
        node._tf_buffer = NS(lookup_transform=Mock(return_value=NS(transform=NS(
            translation=NS(x=10., y=-1.5), rotation=NS(x=0., y=0., z=0., w=1.)))))
        scan = LaserScan()
        scan.header.frame_id = 'lidar'
        scan.header.stamp = Time(seconds=9.9, clock_type=ClockType.ROS_TIME).to_msg()
        scan.angle_min = -math.pi
        scan.angle_increment = 2*math.pi/720
        scan.range_min, scan.range_max = .15, 12.
        angles = scan.angle_min+np.arange(720)*scan.angle_increment
        scan.ranges = [float(2/math.sin(a)) if math.sin(a) > .2 else math.inf for a in angles]
        node._latest_scan = scan
        door = DoorInfo()
        door.door_color = 'blue'
        door.door_id = 'observed-blue-test'
        door.header.frame_id = door.door_pose.header.frame_id = 'map'
        door.observed_door_position.header.frame_id = 'map'
        door.observed_door_position.header.stamp = scan.header.stamp
        door.observed_door_position.point.x, door.observed_door_position.point.y = 11., .5
        door.handle_position.header.frame_id = 'map'
        door.handle_position.point.x, door.handle_position.point.y = 11., .5
        return node, door

    def test_wall_near_initial_centerline_is_supported_by_actual_scan(self):
        node, door = self.node_and_door()
        self.assertTrue(node._blue_door_has_openable_wall_lateral(door))
        x, y, yaw = node._live_observed_blue_parking_pose(door)
        self.assertAlmostEqual(x, 11., places=5)
        self.assertAlmostEqual(y, -.6, places=5)
        self.assertAlmostEqual(yaw, math.pi/2, places=5)

    def test_rotated_translated_map_does_not_change_relative_parking(self):
        node, door = self.node_and_door()
        a = .8
        c, s = math.cos(a), math.sin(a)
        tx, ty = -4., 7.
        def transform(x, y):
            return tx+c*x-s*y, ty+s*x+c*y
        for stamped in (door.observed_door_position, door.handle_position):
            stamped.point.x, stamped.point.y = transform(stamped.point.x, stamped.point.y)
        p = node._tf_buffer.lookup_transform.return_value.transform
        p.translation.x, p.translation.y = transform(10., -1.5)
        p.rotation.z, p.rotation.w = math.sin(a/2), math.cos(a/2)
        x, y, yaw = node._live_observed_blue_parking_pose(door)
        ex, ey = transform(11., -.6)
        self.assertAlmostEqual(x, ex, places=5)
        self.assertAlmostEqual(y, ey, places=5)
        self.assertAlmostEqual(yaw, a+math.pi/2, places=5)

    def test_missing_noisy_or_stale_sensor_does_not_override_lateral_gate(self):
        for case in ('missing', 'unknown', 'stale', 'visual_stale', 'future', 'unmapped'):
            node, door = self.node_and_door()
            if case == 'missing': node._latest_scan = None
            if case == 'unknown': node._latest_scan.ranges = [math.inf]*720
            if case == 'stale': node._latest_scan.header.stamp.sec = 8
            if case == 'visual_stale': door.observed_door_position.header.stamp.sec = 8
            if case == 'future': door.observed_door_position.header.stamp.sec = 11
            if case == 'unmapped': door.observed_door_position.header.frame_id = 'odom'
            self.assertIsNone(node._live_observed_blue_parking_pose(door), case)
            self.assertFalse(node._blue_door_has_openable_wall_lateral(door), case)

    def test_adjustment_preserves_panel_identity_and_handle_coordinates(self):
        node, door = self.node_and_door()
        adjusted = node._door_with_axis_aligned_approach(door)
        self.assertAlmostEqual(adjusted.door_pose.pose.position.y, -.6, places=5)
        self.assertEqual(adjusted.observed_door_position, door.observed_door_position)
        self.assertEqual(adjusted.handle_position, door.handle_position)
        self.assertEqual(node._door_identity_xy(adjusted), (11., .5))
        self.assertAlmostEqual(node._door_open_target_pose_xy_yaw(adjusted)[1], -.6, places=5)

    def test_cached_normal_still_requires_fresh_scan(self):
        node, door = self.node_and_door()
        for _ in range(2): self.assertIsNotNone(node._live_observed_blue_parking_pose(door))
        node._tf_buffer.lookup_transform.assert_called_once()
        node._latest_scan.header.stamp.sec = 8
        self.assertIsNone(node._live_observed_blue_parking_pose(door))

    def test_memory_target_retains_body_and_original_observation_time(self):
        node, door = self.node_and_door()
        node._cluster_target_xy = lambda row: (row['x'], row['y'])
        node._add_observed_blue_cluster_id = Mock()
        node._observed_blue_door_id_to_cluster = {}
        msg = node._observed_blue_cluster_to_door(dict(x=11., y=.5, last_seen=8.25))
        msg.door_pose.pose.position.y = -.6
        self.assertEqual(node._door_body_xy(msg), (11., .5))
        self.assertAlmostEqual(Time.from_msg(msg.observed_door_position.header.stamp).nanoseconds/1e9, 8.25)
        self.assertIsNone(node._live_observed_blue_parking_pose(msg))


if __name__ == '__main__':
    unittest.main()
