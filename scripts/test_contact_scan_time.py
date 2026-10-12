from collections import deque
import json
from types import SimpleNamespace as NS
import time
import unittest
import numpy as np
from unittest.mock import Mock, patch
from builtin_interfaces.msg import Time as TimeMessage
from tf2_ros import TransformException
from fire_robot_manipulation.robot_observation_manipulation_node import RobotObservationManipulation
from fire_robot_manipulation.contact_feedback import SceneObservationPending


def scan(stamp):
    sec=int(stamp)
    return NS(header=NS(stamp=TimeMessage(sec=sec,nanosec=round((stamp-sec)*1.e9)),frame_id='lidar_link'),
              ranges=[2.,2.,2.],angle_min=-.5,angle_increment=.5,range_min=.15,range_max=12.)


class ContactScanTimeTests(unittest.TestCase):
    def lower_bound_fixture(self):
        node=self.fixture(np.linspace(19.08,19.98,11))
        for msg,_ in node._contact_scans:
            msg.angle_min=.53; msg.angle_increment=.01; msg.ranges=[.28,.28,.28]
        node._tf_buffer.lookup_transform_full.return_value.header=NS(stamp=TimeMessage(sec=20))
        node._publish_sim_cmd_vel=Mock();node._feedback_event=Mock()
        return node

    def test_positive_bound_below_sensor_minimum_is_checked_against_actual_body(self):
        node=self.lower_bound_fixture()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.temporal_scan_consensus',
                   return_value=(np.full(3,.149),3,.10,np.zeros(3))):
            node._feedback_scene_constraints()
        self.assertGreater(node._last_scanned_body_gap,.03)
        self.assertLess(node._last_scanned_body_gap,.08)
        node._publish_sim_cmd_vel.assert_not_called()

    def test_low_confidence_bound_that_reaches_body_still_stops(self):
        node=self.lower_bound_fixture()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.temporal_scan_consensus',
                   return_value=(np.full(3,.08),3,.10,np.zeros(3))):
            with self.assertRaises(SceneObservationPending):
                node._feedback_scene_constraints()
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_nonpositive_confidence_bound_is_never_treated_as_free_space(self):
        node=self.lower_bound_fixture()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.temporal_scan_consensus',
                   return_value=(np.full(3,-.01),3,.30,np.zeros(3))):
            with self.assertRaises(ValueError):
                node._feedback_scene_constraints()
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def test_invalid_current_returns_cannot_be_filled_by_old_clear_history(self):
        node=self.lower_bound_fixture()
        node._contact_scans[-1][0].ranges=[.1,.1,.1]
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.temporal_scan_consensus',
                   return_value=(np.full(3,.28),3,.10,np.zeros(3))):
            with self.assertRaises(ValueError):
                node._feedback_scene_constraints()

    def test_raw_close_observation_stops_before_temporal_model_can_hide_it(self):
        node=self.fixture(np.linspace(19.08,19.98,11))
        for msg,_ in node._contact_scans:
            msg.angle_min=1.08; msg.angle_increment=.01; msg.ranges=[.5,.5,.5]
        node._contact_scans[-1][0].ranges[1]=.19
        node._tf_buffer.lookup_transform_full.return_value.header=NS(stamp=TimeMessage(sec=20))
        node._publish_sim_cmd_vel=Mock(); node._feedback_event=Mock()
        with patch('fire_robot_manipulation.robot_observation_manipulation_node.temporal_scan_consensus',
                   return_value=(np.full(3,.5),3,.08,np.zeros(3))):
            with self.assertRaises(SceneObservationPending):
                node._feedback_scene_constraints()
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)

    def fixture(self,stamps):
        node=RobotObservationManipulation.__new__(RobotObservationManipulation)
        node._contact_scans=deque((scan(stamp),time.monotonic()) for stamp in stamps)
        node._sim_time_sec=Mock(return_value=20.)
        node._manipulation_frame='base_link'
        node._contact_reference_frame='odom'
        node._tf_buffer=Mock()
        node._tf_buffer.lookup_transform_full.return_value=NS(transform=NS(
            rotation=NS(x=0.,y=0.,z=0.,w=1.),translation=NS(x=.25,y=0.,z=.248)))
        return node

    def test_new_future_callback_does_not_replace_usable_past_scan(self):
        node=self.fixture([19.96,20.02])
        node._feedback_scene_constraints()
        self.assertAlmostEqual(node._last_scene_scan_stamp,19.96)
        args=node._tf_buffer.lookup_transform_full.call_args.args
        self.assertEqual(args[4],'odom')
        self.assertEqual(args[3].nanoseconds,19960000000)

    def test_only_future_or_stale_scans_cannot_drive(self):
        for stamps in ([20.02],[19.7],[]):
            node=self.fixture(stamps)
            with self.assertRaises(ValueError):
                node._feedback_scene_constraints()
            node._tf_buffer.lookup_transform_full.assert_not_called()

    def test_newest_nonfuture_scan_is_selected(self):
        node=self.fixture([19.90,19.98,19.94])
        node._feedback_scene_constraints()
        self.assertAlmostEqual(node._last_scene_scan_stamp,19.98)

    def test_newest_scan_ahead_of_tf_uses_aligned_older_scan(self):
        node=self.fixture([19.90,19.98])
        valid=node._tf_buffer.lookup_transform_full.return_value
        node._tf_buffer.lookup_transform_full.side_effect=[TransformException('future extrapolation'),valid]
        node._feedback_scene_constraints()
        self.assertAlmostEqual(node._last_scene_scan_stamp,19.90)

    def test_absent_transform_does_not_bypass_scene_guard(self):
        node=self.fixture([19.90,19.98])
        node._tf_buffer.lookup_transform_full.side_effect=TransformException('unavailable TF')
        with self.assertRaises(ValueError):
            node._feedback_scene_constraints()

    def test_close_return_diagnostic_serializes_invalid_neighbors_without_changing_scan(self):
        node=self.fixture([19.98])
        msg=node._contact_scans[0][0]
        msg.ranges=[.18,float('inf'),2.,2.]
        msg.angle_min=0.
        node._feedback_event=Mock()
        node._feedback_scene_constraints()
        payload=node._feedback_event.call_args.kwargs
        json.dumps(payload,allow_nan=False)
        self.assertIn(None,payload['returns'][0]['neighbor_ranges_m'])
        self.assertTrue(np.isinf(msg.ranges[1]))

    def test_new_scan_invalidates_plan_without_driving_or_lowering_clearance(self):
        node=self.fixture([19.9])
        node._feedback_scene_constraints=Mock(return_value=(np.array([[1.,0.]]),np.array([.005])))
        node._last_scanned_body_gap=.05
        node._last_scene_scan_stamp=19.9
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        self.assertEqual(node._feedback_scene_command_safe(.02,0.),'replan')
        node._publish_sim_cmd_vel.assert_called_once_with(0.,0.)

    def test_actual_clearance_violation_still_stops_not_replans(self):
        node=self.fixture([19.9])
        node._feedback_scene_constraints=Mock(return_value=(np.array([[1.,0.]]),np.array([.005])))
        node._last_scanned_body_gap=.02
        node._feedback_stop=Mock(return_value=False)
        self.assertFalse(node._feedback_scene_command_safe(.02,0.,active=True))
        node._feedback_stop.assert_called_once()

    def test_active_probe_stops_during_scene_reobservation(self):
        node=self.fixture([19.9])
        node._feedback_scene_constraints=Mock(side_effect=SceneObservationPending('recheck'))
        node._publish_sim_cmd_vel=Mock()
        node._feedback_event=Mock()
        self.assertEqual(node._feedback_scene_command_safe(.02,0.,active=True),'replan')
        node._publish_sim_cmd_vel.assert_called_with(0.,0.)


if __name__ == '__main__':
    unittest.main()
