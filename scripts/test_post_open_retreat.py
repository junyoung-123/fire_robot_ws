import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rclpy.time import Time
from fire_robot_fsm.state_machine_node import StateMachineNode, State
from fire_robot_fsm.alignment_safety import (
    retreat_scan_clearance, observed_retreat_velocity, retreat_progress_budget, retreat_escape_budget,
    remembered_return_heading)
from sensor_msgs.msg import LaserScan


class PostOpenRetreatTests(unittest.TestCase):
    def test_completed_wheel_escape_subgoal_renews_bounded_deadline(self):
        self.assertEqual(retreat_escape_budget(119.,120.,500.,499.7),139.)

    def test_escape_progress_never_extends_expired_deadline_or_ceiling(self):
        self.assertEqual(retreat_escape_budget(121.,120.,500.,499.7),120.)
        self.assertEqual(retreat_escape_budget(179.,180.,500.,499.7),180.)
        self.assertEqual(retreat_escape_budget(181.,180.,500.,499.7),180.)

    def test_missing_stale_future_or_invalid_escape_progress_earns_no_time(self):
        for stamp in (None,498.,500.1,math.nan):
            self.assertEqual(retreat_escape_budget(119.,120.,500.,stamp),120.)

    def test_progress_credit_extends_only_for_new_inward_wheel_supported_motion(self):
        anchor = (2.2, 0., 0., 110.)
        changed, deadline = retreat_progress_budget(anchor, 2.0, (0., .2), 115., 120.)
        self.assertEqual(deadline, 135.)
        self.assertEqual(changed, (2.0, 0., .2, 115.))

    def test_stationary_localization_jump_does_not_extend(self):
        anchor = (2.2, 0., 0., 110.)
        for lateral, odom in ((1.8, (0., 0.)), (1.8, (0., .1)), (2.19, (0., .2))):
            _,deadline=retreat_progress_budget(anchor,lateral,odom,115.,120.)
            self.assertEqual(deadline,120.)

    def test_stationary_map_jump_reanchors_without_credit_then_accepts_new_wheel_progress(self):
        anchor=(2.2,0.,0.,110.)
        changed,deadline=retreat_progress_budget(anchor,1.8,(0.,0.),115.,120.)
        self.assertEqual(changed,(1.8,0.,0.,115.))
        self.assertEqual(deadline,120.)
        changed,deadline=retreat_progress_budget(changed,1.7,(0.,.1),117.,deadline)
        self.assertEqual(changed,(1.7,0.,.1,117.))
        self.assertEqual(deadline,137.)

    def test_repeated_map_only_jumps_never_extend_budget(self):
        anchor=(2.2,0.,0.,110.)
        deadline=120.
        for i in range(1,5):
            anchor,deadline=retreat_progress_budget(anchor,2.2-i*.2,(0.,0.),110.+i,deadline)
            self.assertEqual(deadline,120.)

    def test_oscillation_or_outward_motion_does_not_extend(self):
        anchor = (2.2, 0., 0., 110.)
        for lateral in (2.2, 2.4, -2.2, -2.4):
            self.assertEqual(retreat_progress_budget(anchor, lateral, (0., .2), 115., 120.),
                             (anchor, 120.))

    def test_missing_invalid_or_expired_measurements_do_not_extend(self):
        anchor = (2.2, 0., 0., 110.)
        for odom, elapsed in ((None, 115.), ((0., math.nan), 115.),
                              ((0., .2), 121.), ((0., .2), 109.)):
            self.assertEqual(retreat_progress_budget(anchor, 2., odom, elapsed, 120.),
                             (anchor, 120.))

    def test_odom_jump_cannot_extend(self):
        anchor = (2.2, 0., 0., 114.9)
        self.assertEqual(retreat_progress_budget(anchor, 1.2, (0., 1.), 115., 120.),
                         (anchor, 120.))

    def test_progress_never_exceeds_hard_ceiling(self):
        anchor = (2.2, 0., 0., 290.)
        _, deadline = retreat_progress_budget(anchor, 2., (0., .2), 295., 298.)
        self.assertEqual(deadline, 300.)

    def test_only_inward_wheel_progress_can_extend_beyond_initial_escape_ceiling(self):
        anchor=(2.2,0.,0.,170.)
        _,deadline=retreat_progress_budget(anchor,2.,(0.,.2),178.,180.)
        self.assertEqual(deadline,198.)
        self.assertEqual(retreat_escape_budget(178.,180.,500.,499.8),180.)
        _,stationary=retreat_progress_budget(anchor,2.,(0.,0.),178.,180.)
        self.assertEqual(stationary,180.)

    def test_return_bearing_uses_own_odom_after_map_correction(self):
        anchor,error=remembered_return_heading(None,(10.,2.,0.),(1.,3.,math.pi/2),(11.,2.),0.)
        self.assertAlmostEqual(error,0.)
        _,error=remembered_return_heading(anchor,(20.,-8.,1.),(1.,3.,math.pi/2),(11.,2.),0.)
        self.assertAlmostEqual(error,0.)
        _,error=remembered_return_heading(anchor,(20.,-8.,1.),(1.,3.,math.pi/2+.2),(11.,2.),0.)
        self.assertAlmostEqual(error,-.2)

    def test_reached_return_target_is_not_reselected_after_moving_inward(self):
        anchor,_=remembered_return_heading(None,(0.,0.,0.),(1.,2.,0.),(.5,0.),math.pi/2)
        anchor,error=remembered_return_heading(anchor,(0.,0.,0.),(1.5,2.,0.),(.5,0.),math.pi/2)
        self.assertIsNone(anchor[0])
        self.assertAlmostEqual(error,math.pi/2)
        _,error=remembered_return_heading(anchor,(0.,0.,0.),(1.5,2.5,math.pi/2),(.5,0.),math.pi/2)
        self.assertAlmostEqual(error,0.)

    def test_no_return_waypoint_anchors_only_observed_inward_heading(self):
        anchor,error=remembered_return_heading(None,(0.,0.,0.),(1.,2.,.1),None,math.pi/2)
        self.assertIsNone(anchor[0])
        self.assertAlmostEqual(error,math.pi/2)

    def test_invalid_return_pose_or_waypoint_cannot_generate_heading(self):
        for pose,waypoint in (((0.,0.),(1.,2.)),((0.,math.nan,0.),(1.,2.)),
                              ((0.,0.,0.),(math.nan,2.))):
            with self.assertRaises(ValueError):
                remembered_return_heading(None,pose,(0.,0.,0.),waypoint,0.)

    def test_first_valid_sample_does_not_add_time(self):
        anchor, deadline = retreat_progress_budget(None, 2., (0., .2), 115., 120.)
        self.assertEqual(anchor, (2., 0., .2, 115.))
        self.assertEqual(deadline, 120.)

    def test_precontact_pose_refreshes_return_waypoint_only_during_opening(self):
        from geometry_msgs.msg import PoseStamped
        node = self.node(0.)
        node.state = State.OPENING_DOOR
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp.sec = 'map', 10
        msg.pose.position.x, msg.pose.position.y = 6.1, -1.4
        msg.pose.orientation.w = 1.
        node._contact_return_pose_callback(msg)
        self.assertEqual(node._post_open_return_waypoint, (6.1, -1.4, 0.))
        for state, frame, stamp in ((State.EXPLORING, 'map', 10),
                                    (State.OPENING_DOOR, 'evaluation', 10),
                                    (State.OPENING_DOOR, 'map', 8),
                                    (State.OPENING_DOOR, 'map', 11)):
            node.state = state
            msg.header.frame_id, msg.header.stamp.sec = frame, stamp
            msg.pose.position.x = 8.
            node._contact_return_pose_callback(msg)
            self.assertEqual(node._post_open_return_waypoint, (6.1, -1.4, 0.))

    def test_precontact_odom_pose_survives_later_map_correction(self):
        from geometry_msgs.msg import PoseStamped
        node = self.guarded_node(0.)
        node.state = State.OPENING_DOOR
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp.sec = 'odom', 10
        msg.pose.position.x, msg.pose.position.y = 6.1, -1.4
        msg.pose.orientation.z = math.sin(-math.pi/4)
        msg.pose.orientation.w = math.cos(-math.pi/4)
        node._contact_return_pose_callback(msg)
        node._post_open_side_retreat_start_time = None
        scene = SimpleNamespace(pose=(15., -3., 1.), odom_pose=(6.6, -2.4, 1.),
                                safe=Mock(return_value=True), clearance=Mock(return_value=.1))
        node._current_map_pose = lambda: scene.pose
        guard = SimpleNamespace(snapshot=Mock(return_value=scene),
                                recovery_command=Mock(return_value=(0., 0., 'test')),
                                reset_recovery=Mock())
        node._post_open_footprint_guard = guard
        node._run_post_open_side_retreat()
        self.assertEqual(node._post_open_return_odom_anchor[0], (6.1, -1.4))
        self.assertAlmostEqual(node._post_open_return_odom_anchor[1], math.pi/2)
        expected = math.atan2(1., -.5)-1.
        self.assertAlmostEqual(guard.recovery_command.call_args.args[1], expected)
        scene.pose = (30., -10., -2.)
        node._run_post_open_side_retreat()
        self.assertAlmostEqual(guard.recovery_command.call_args.args[1], expected)
        node._finish_post_open_side_retreat('test')
        self.assertIsNone(node._post_open_precontact_odom)

    def test_odom_return_pose_rejects_stale_future_and_wrong_state(self):
        from geometry_msgs.msg import PoseStamped
        node = self.node(0.)
        node._post_open_precontact_odom = None
        msg = PoseStamped()
        msg.header.frame_id = 'odom'
        msg.pose.orientation.w = 1.
        for state, stamp in ((State.EXPLORING, 10), (State.OPENING_DOOR, 8),
                             (State.OPENING_DOOR, 11)):
            node.state, msg.header.stamp.sec = state, stamp
            node._contact_return_pose_callback(msg)
            self.assertIsNone(node._post_open_precontact_odom)

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

    def test_return_heading_uses_measured_preopening_doorway_pose(self):
        node = self.node(0.)
        node._post_open_return_waypoint = (3., -1., -.8)
        heading = node._post_open_return_heading((3.7, -2.3, 1.5), math.pi/2)
        self.assertAlmostEqual(heading, math.atan2(1.3, -.7))
        self.assertGreater(heading, math.pi/2)

    def test_reaching_measured_waypoint_resumes_inward_clearance(self):
        node = self.node(0.)
        node._post_open_return_waypoint = (3., -1., -.8)
        self.assertEqual(node._post_open_return_heading((3.1, -1., 1.5), 1.5), 1.5)
        self.assertIsNone(node._post_open_return_waypoint)

    def test_no_recorded_waypoint_preserves_existing_dispatch(self):
        node = self.node(0.)
        self.assertEqual(node._post_open_return_heading((0., 0., 0.), .9), .9)

    def test_measured_clearance_ends_motion_before_distance_cap(self):
        node = self.node(0.)
        node._current_map_pose = lambda: (0., .47, -math.pi/2)
        self.assertFalse(node._run_post_open_side_retreat())
        self.assertFalse(node._post_open_side_retreat_pending)
        command = node.cmd_vel_pub.publish.call_args[0][0]
        self.assertEqual(command.linear.x, 0.)
        self.assertEqual(command.angular.z, 0.)

    def guarded_node(self, error_deg):
        node = self.node(error_deg)
        node._post_open_observed_clearance_enabled = True
        node._post_open_scan_turn_min_m = .60
        node._latest_scan_received_sec = 10.
        scan = LaserScan()
        scan.header.stamp.sec = 10
        scan.angle_min, scan.angle_increment = -math.pi, math.pi/180
        scan.range_min, scan.range_max = .1, 12.
        scan.ranges = [3.]*361
        node._latest_scan = scan
        node._transition = Mock()
        return node

    def test_guarded_dispatch_clears_side_before_turning(self):
        node = self.guarded_node(30.)
        node._latest_scan.ranges[270] = .30
        self.assertTrue(node._run_post_open_side_retreat())
        command = node.cmd_vel_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (.06, 0.))

    def test_guarded_dispatch_uses_rear_escape_only_when_forward_blocked(self):
        node = self.guarded_node(30.)
        node._latest_scan.ranges[180] = .30
        self.assertTrue(node._run_post_open_side_retreat())
        command = node.cmd_vel_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (-.06, 0.))

    def test_guarded_dispatch_never_turns_or_hands_off_with_stale_scan(self):
        node = self.guarded_node(0.)
        node._latest_scan.header.stamp.sec = 8
        node._current_map_pose = lambda: (0., .47, -math.pi/2)
        self.assertTrue(node._run_post_open_side_retreat())
        command = node.cmd_vel_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0., 0.))
        self.assertTrue(node._post_open_side_retreat_pending)

    def test_timeout_is_not_successful_navigation_handoff(self):
        node = self.guarded_node(0.)
        node._post_open_side_retreat_start_time = Time(seconds=0)
        node._post_open_side_retreat_timeout_sec = 5.
        self.assertTrue(node._run_post_open_side_retreat())
        node._transition.assert_called_once_with(State.EMERGENCY_STOP)
        self.assertTrue(node._post_open_side_retreat_pending)

    def test_clear_surround_allows_original_centerward_turn(self):
        node = self.guarded_node(30.)
        self.assertTrue(node._run_post_open_side_retreat())
        command = node.cmd_vel_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0., -.5))

    def test_post_open_uses_guarded_publisher_not_contact_channel(self):
        node = self.guarded_node(30.)
        node._post_open_guarded_pub = Mock()
        self.assertTrue(node._run_post_open_side_retreat())
        node.cmd_vel_pub.publish.assert_not_called()
        command = node._post_open_guarded_pub.publish.call_args.args[0]
        self.assertEqual((command.linear.x, command.angular.z), (0., -.5))

    def test_invalid_and_partial_scan_never_certifies_clearance(self):
        node = self.guarded_node(0.)
        for change in ('nan', 'short', 'future', 'stale_receive'):
            scan = self.guarded_node(0.)._latest_scan
            received = 10.
            if change == 'nan':
                scan.ranges[180] = float('nan')
            elif change == 'short':
                scan.ranges = scan.ranges[:180]
            elif change == 'future':
                scan.header.stamp.sec = 12
            else:
                received = 8.
            clearance = retreat_scan_clearance(scan, 10., received)
            if change == 'nan':
                self.assertEqual(clearance[0], 0.)
            else:
                self.assertIsNone(clearance)

    def test_both_blocked_directions_produce_stop_not_a_rotation(self):
        command = observed_retreat_velocity((.30, .30, .30), .8, .18, .5, .17, .60)
        self.assertEqual(command, (0., 0., 'BLOCKED'))


if __name__ == '__main__':
    unittest.main()
