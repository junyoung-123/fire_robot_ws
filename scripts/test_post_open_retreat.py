import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rclpy.time import Time
from fire_robot_fsm.state_machine_node import StateMachineNode


class PostOpenRetreatTests(unittest.TestCase):
    def node(self, error_deg):
        node = object.__new__(StateMachineNode)
        node._post_open_side_retreat_pending = True
        node._post_open_side_retreat_enabled = True
        node._post_open_side_retreat_m = 1.1
        node._post_open_side_retreat_target_abs_y_m = .48
        node._post_open_side_retreat_start_time = Time(seconds=9)
        node._post_open_side_retreat_start_lateral = .9
        node._post_open_side_retreat_timeout_sec = 14.
        node._post_open_side_retreat_yaw_tolerance = math.radians(10.)
        node._post_open_side_retreat_linear_vel = .18
        node._post_open_reorient_angular_vel = .5
        node._post_open_clearance_angular_vel_limit = .2
        node._post_open_clearance_heading_kp = 1.
        node._explore_center_y = 0.
        node._current_map_pose = lambda: (0., .9, -math.pi/2 + math.radians(error_deg))
        node._axis_lateral_xy = lambda x, y: y
        node._mission_forward_yaw = lambda: 0.
        node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=10))
        node.get_logger = lambda: Mock()
        node.cmd_vel_pub = Mock()
        return node

    def test_diagonal_heading_cannot_translate_toward_neighboring_obstacle(self):
        node = self.node(30.)
        self.assertTrue(node._run_post_open_side_retreat())
        command = node.cmd_vel_pub.publish.call_args[0][0]
        self.assertEqual(command.linear.x, 0.)
        self.assertLess(command.angular.z, 0.)

    def test_aligned_heading_can_translate(self):
        node = self.node(5.)
        self.assertTrue(node._run_post_open_side_retreat())
        self.assertGreater(node.cmd_vel_pub.publish.call_args[0][0].linear.x, 0.)

    def test_measured_clearance_ends_motion_before_distance_cap(self):
        node = self.node(0.)
        node._current_map_pose = lambda: (0., .47, -math.pi/2)
        self.assertFalse(node._run_post_open_side_retreat())
        self.assertFalse(node._post_open_side_retreat_pending)
        command = node.cmd_vel_pub.publish.call_args[0][0]
        self.assertEqual(command.linear.x, 0.)
        self.assertEqual(command.angular.z, 0.)


if __name__ == '__main__':
    unittest.main()
