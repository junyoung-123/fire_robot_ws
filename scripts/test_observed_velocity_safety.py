import math
import random
import unittest
from unittest.mock import Mock

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from fire_robot_navigation.cmd_vel_safety_node import CmdVelSafetyNode


class ObservedVelocitySafetyTests(unittest.TestCase):
    def node(self):
        node = object.__new__(CmdVelSafetyNode)
        node._max_linear_x = .25
        node._max_angular_z = 1.0
        node._allow_reverse = True
        node._manual_hold_sec = .75
        node._manual_active_until_sec = 0.
        node._lidar_stop_manual_commands = True
        node._stop_on_missing_scan = True
        node._coupled_collision_stop = True
        node._scan_timeout_sec = .75
        node._latest_scan_time_sec = 10.
        node._now_sec = lambda: 10.
        node._front_stop_distance_m = .48
        node._rear_stop_distance_m = .40
        node._rotation_stop_distance_m = .32
        node._drive_stop_half_angle = math.radians(35.)
        scan = LaserScan()
        scan.header.stamp.sec = 10
        scan.angle_min = -math.pi
        scan.angle_increment = math.pi/180
        scan.range_min = .1
        scan.range_max = 12.
        scan.ranges = [3.]*361
        node._latest_scan = scan
        node._pub = Mock()
        node._log_obstacle_clamp = Mock()
        return node

    def test_manual_forward_command_stops_at_obstacle(self):
        node = self.node()
        node._latest_scan.ranges[180] = .3
        command = Twist()
        command.linear.x = .13
        node._on_manual_cmd_vel(command)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.)

    def test_missing_and_stale_scan_stop_translation_and_rotation(self):
        for missing in (True, False):
            node = self.node()
            if missing:
                node._latest_scan = None
            else:
                node._latest_scan_time_sec = 8.
            command = Twist()
            command.linear.x = .13
            command.angular.z = .4
            node._on_manual_cmd_vel(command)
            safe = node._pub.publish.call_args.args[0]
            self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def test_clear_scan_preserves_command(self):
        node = self.node()
        command = Twist()
        command.linear.x = .13
        node._on_manual_cmd_vel(command)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, .13)

    def test_contact_distance_bypass_still_requires_fresh_scan(self):
        for case in ('missing', 'stale_receive', 'stale_measurement', 'future_measurement'):
            node = self.node()
            node._lidar_stop_manual_commands = False
            if case == 'missing':
                node._latest_scan = None
            elif case == 'stale_receive':
                node._latest_scan_time_sec = 8.
            else:
                node._latest_scan.header.stamp.sec = 8 if case == 'stale_measurement' else 12
            command = Twist()
            command.linear.x, command.angular.z = .03, -.02
            node._on_manual_cmd_vel(command)
            safe = node._pub.publish.call_args.args[0]
            self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.), case)

    def test_fresh_near_contact_scan_allows_separately_checked_manual_motion(self):
        node = self.node()
        node._lidar_stop_manual_commands = False
        node._latest_scan.ranges[180] = .35
        command = Twist()
        command.linear.x, command.angular.z = .03, -.02
        node._on_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (.03, -.02))

    def test_guarded_fsm_command_cannot_use_contact_distance_bypass(self):
        node = self.node()
        node._lidar_stop_manual_commands = False
        node._latest_scan.ranges[270] = .28
        command = Twist()
        command.linear.x, command.angular.z = .13, .4
        node._on_guarded_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))
        self.assertEqual(node._manual_active_until_sec, 10.75)

    def test_guarded_fsm_command_requires_fresh_scan(self):
        node = self.node()
        node._lidar_stop_manual_commands = False
        node._latest_scan.header.stamp.sec = 8
        command = Twist()
        command.linear.x, command.angular.z = .13, .4
        node._on_guarded_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def test_blocked_turn_must_not_become_unchecked_straight_motion(self):
        node = self.node()
        node._latest_scan.ranges[270] = .28
        command = Twist()
        command.linear.x, command.angular.z = .13, .4
        node._on_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def test_clear_straight_reverse_remains_available_near_side_obstacle(self):
        node = self.node()
        node._latest_scan.ranges[270] = .28
        command = Twist()
        command.linear.x = -.08
        node._on_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (-.08, 0.))

    def test_front_stop_retains_turn_only_after_surround_clearance_check(self):
        for angular in (-.4, .4):
            node = self.node()
            node._latest_scan.ranges[180] = .4
            command = Twist()
            command.linear.x, command.angular.z = .13, angular
            node._on_manual_cmd_vel(command)
            safe = node._pub.publish.call_args.args[0]
            self.assertEqual((safe.linear.x, safe.angular.z), (0., angular))

    def test_front_stop_with_blocked_side_also_stops_rotation(self):
        node = self.node()
        node._latest_scan.ranges[180] = .4
        node._latest_scan.ranges[270] = .3
        command = Twist()
        command.linear.x, command.angular.z = .13, .4
        node._on_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def test_translation_stop_without_rotation_check_does_not_allow_turn(self):
        node = self.node()
        node._rotation_stop_distance_m = 0.
        node._latest_scan.ranges[180] = .4
        command = Twist()
        command.linear.x, command.angular.z = .13, .4
        node._on_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def swept_node(self, allowed=True):
        node = self.node()
        node._rotation_stop_distance_m = .5942
        node._latest_scan.ranges[270] = .54
        node._nav_swept_clearance_enabled = True
        node._footprint_guard = Mock()
        node._footprint_guard.snapshot.return_value.safe.return_value = allowed
        node._footprint_guard.snapshot.return_value.clearance.return_value = .1
        node._footprint_guard.snapshot.return_value.last_failure = None if allowed else 'UNKNOWN_OR_OCCUPIED_SWEEP'
        return node

    def test_nav_arc_uses_body_sweep_when_full_circle_is_blocked(self):
        node = self.swept_node()
        command = Twist()
        command.linear.x, command.angular.z = .20, .80
        node._on_nav_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertAlmostEqual(safe.linear.x, .12)
        self.assertAlmostEqual(safe.angular.z, .48)
        self.assertAlmostEqual(safe.linear.x/safe.angular.z, .25)
        node._footprint_guard.snapshot.return_value.safe.assert_called_once_with(.12, .48)

    def test_nav_in_place_turn_uses_measured_sweep(self):
        node = self.swept_node()
        command = Twist()
        command.angular.z = -.80
        node._on_nav_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., -.5))

    def test_nav_straight_also_requires_body_sweep(self):
        node = self.swept_node(False)
        command = Twist()
        command.linear.x = .20
        node._on_nav_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))
        node._footprint_guard.snapshot.return_value.safe.assert_called_once_with(.12, 0.)

    def test_scaled_velocity_cannot_round_outside_strict_sweep_bounds(self):
        rng = random.Random(1)
        for _ in range(10000):
            command = Twist()
            command.linear.x = rng.uniform(-.25, .25)
            command.angular.z = rng.uniform(-1., 1.)
            bounded = CmdVelSafetyNode._bounded_sweep_twist(command)
            self.assertLessEqual(abs(bounded.linear.x), .12)
            self.assertLessEqual(abs(bounded.angular.z), .5)

    def test_guarded_mission_motion_is_scaled_before_body_check(self):
        node = self.swept_node()
        command = Twist()
        command.linear.x, command.angular.z = .24, .8
        node._on_guarded_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertAlmostEqual(safe.linear.x, .12)
        self.assertAlmostEqual(safe.angular.z, .4)
        node._footprint_guard.snapshot.return_value.safe.assert_called_once_with(.12, .4)

    def test_guarded_mission_stops_despite_contact_channel_distance_exception(self):
        node = self.swept_node(False)
        node._lidar_stop_manual_commands = False
        command = Twist()
        command.linear.x = .12
        node._on_guarded_manual_cmd_vel(command)
        safe = node._pub.publish.call_args.args[0]
        self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def test_straight_nav_missing_scene_does_not_assume_front_inf_is_clear(self):
        node = self.swept_node()
        node._footprint_guard.snapshot.return_value = None
        node._latest_scan.ranges = [float('inf')]*361
        command = Twist()
        command.linear.x = .1
        node._on_nav_cmd_vel(command)
        self.assertEqual(node._pub.publish.call_args.args[0].linear.x, 0.)

    def test_nav_unknown_or_colliding_sweep_stops_both_components(self):
        for missing in (True, False):
            node = self.swept_node(False)
            if missing:
                node._footprint_guard.snapshot.return_value = None
            command = Twist()
            command.linear.x, command.angular.z = .20, .80
            node._on_nav_cmd_vel(command)
            safe = node._pub.publish.call_args.args[0]
            self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))

    def test_nav_sweep_never_bypasses_front_or_rear_stop(self):
        for velocity, index, distance in ((.2, 180, .3), (-.2, 0, .3)):
            node = self.swept_node()
            node._latest_scan.ranges[index] = distance
            command = Twist()
            command.linear.x, command.angular.z = velocity, .4
            node._on_nav_cmd_vel(command)
            safe = node._pub.publish.call_args.args[0]
            self.assertEqual((safe.linear.x, safe.angular.z), (0., 0.))
            node._footprint_guard.snapshot.assert_not_called()

    def test_nav_sweep_never_bypasses_stale_scan(self):
        node = self.swept_node()
        node._latest_scan.header.stamp.sec = 8
        command = Twist()
        command.angular.z = .4
        node._on_nav_cmd_vel(command)
        self.assertEqual(node._pub.publish.call_args.args[0].angular.z, 0.)
        node._footprint_guard.snapshot.assert_not_called()

    def test_sweep_fallback_not_applied_to_manual_contact_channel(self):
        node = self.swept_node()
        command = Twist()
        command.angular.z = .4
        node._on_manual_cmd_vel(command)
        self.assertEqual(node._pub.publish.call_args.args[0].angular.z, 0.)
        node._footprint_guard.snapshot.assert_not_called()

    def test_manual_owner_excludes_nav_sweep(self):
        node = self.swept_node()
        node._manual_active_until_sec = 11.
        node._on_nav_cmd_vel(Twist())
        node._pub.publish.assert_not_called()
        node._footprint_guard.snapshot.assert_not_called()

    def test_opt_out_retains_existing_radial_stop(self):
        node = self.swept_node()
        node._nav_swept_clearance_enabled = False
        command = Twist()
        command.angular.z = .4
        node._on_nav_cmd_vel(command)
        self.assertEqual(node._pub.publish.call_args.args[0].angular.z, 0.)
        node._footprint_guard.snapshot.assert_not_called()


if __name__ == '__main__':
    unittest.main()
