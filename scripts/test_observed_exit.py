import math
import unittest
from unittest.mock import Mock

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


if __name__ == '__main__':
    unittest.main()
