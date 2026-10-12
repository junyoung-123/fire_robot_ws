import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from geometry_msgs.msg import PointStamped, TransformStamped
from fire_robot_manipulation.mission_contact_adapter import MissionContactAdapter


class SelectedHandleViewTests(unittest.TestCase):
    def fixture(self):
        point = PointStamped()
        point.header.frame_id = 'contact_scan'
        point.header.stamp.sec = 20
        point.point.x, point.point.y, point.point.z = .93, .44, .77
        current = TransformStamped()
        current.header.stamp.sec = 40
        current.transform.rotation.w = 1.
        camera = TransformStamped()
        camera.transform.translation.x = .25
        camera.transform.translation.y = .0175
        # Optical z forward, x right, y down.
        camera.transform.rotation.x = -.5
        camera.transform.rotation.y = .5
        camera.transform.rotation.z = -.5
        camera.transform.rotation.w = .5
        node = SimpleNamespace(tf=Mock(), now=lambda: 40., reference_frame='contact_scan',
            view_hint=None, request_target=point,
            guarded_view_command=Mock(side_effect=lambda v, w: (v, w)))
        node.tf.lookup_transform.side_effect = [current, camera]
        return node, point, current

    def test_remembered_bearing_can_turn_before_confident_handle_exists(self):
        node, point, current = self.fixture()
        candidate, applied, basis = MissionContactAdapter.reobserve_selected_view(node)
        self.assertEqual(candidate, (0., .12))
        self.assertEqual(candidate, applied)
        self.assertEqual(basis, 'remembered_target_bearing_only')
        self.assertEqual(point.header.stamp.sec, 20)

    def test_memory_only_never_commands_translation(self):
        node, point, current = self.fixture()
        point.point.y = .0175
        candidate, applied, _ = MissionContactAdapter.reobserve_selected_view(node)
        self.assertEqual(candidate, (0., 0.))

    def test_fresh_weak_hint_can_back_away_only_through_footprint_guard(self):
        node, point, current = self.fixture()
        node.view_hint = (39.9, np.array([.93, .0175, .77]), .33)
        node.guarded_view_command = Mock(return_value=(0., 0.))
        candidate, applied, basis = MissionContactAdapter.reobserve_selected_view(node)
        self.assertEqual(candidate, (-.04, 0.))
        self.assertEqual(applied, (0., 0.))
        self.assertEqual(basis, 'fresh_registered_depth_view_hint')

    def test_stale_transform_stops_camera_recovery(self):
        node, point, current = self.fixture()
        current.header.stamp.sec = 39
        candidate, applied, _ = MissionContactAdapter.reobserve_selected_view(node)
        self.assertEqual(candidate, (0., 0.))

    def test_old_or_future_view_hint_cannot_authorize_backing(self):
        for stamp in (38., 41.):
            node, point, current = self.fixture()
            point.point.y = .0175
            node.view_hint = (stamp, np.array([.93, .0175, .77]), .33)
            candidate, _, basis = MissionContactAdapter.reobserve_selected_view(node)
            self.assertEqual(candidate, (0., 0.))
            self.assertEqual(basis, 'remembered_target_bearing_only')

    def test_map_landmark_reframed_with_current_tf_without_freshening_observation(self):
        node, point, current = self.fixture()
        point.header.frame_id = 'map'
        current.header.frame_id = 'contact_scan'
        current.transform.translation.x = 1.
        node.tf.lookup_transform.side_effect = None
        node.tf.lookup_transform.return_value = current
        node.command, node.event = Mock(), Mock()
        with patch('fire_robot_manipulation.mission_contact_adapter.rclpy.ok', return_value=True):
            result = MissionContactAdapter.wait_selection_target(node, point)
        self.assertAlmostEqual(result.point.x, 1.93)
        self.assertEqual(result.header.stamp.sec, 20)
        self.assertEqual(node.tf.lookup_transform.call_args.args[2].nanoseconds, 0)
        self.assertEqual(node.event.call_args.kwargs['basis'], 'selection_only_not_contact_measurement')

    def test_camera_frame_measurement_keeps_original_exposure_tf(self):
        node, point, current = self.fixture()
        point.header.frame_id = 'camera_color_optical_frame'
        node.wait_transform = Mock(return_value=point)
        self.assertIs(MissionContactAdapter.wait_selection_target(node, point), point)
        node.wait_transform.assert_called_once_with(point)
        node.tf.lookup_transform.assert_not_called()


if __name__ == '__main__':
    unittest.main()
