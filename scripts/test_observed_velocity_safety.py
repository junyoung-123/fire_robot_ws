import math
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


if __name__ == '__main__':
    unittest.main()
