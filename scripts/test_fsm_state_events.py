#!/usr/bin/env python3
"""Explicit transitions, including short-lived success, reach ROS observers."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fire_robot_fsm.state_machine_node import State, StateMachineNode


class StateEventTests(unittest.TestCase):
    def recovery_node(self):
        node = object.__new__(StateMachineNode)
        node.state = State.NAVIGATING
        node._locked_target_refine_enabled = True
        node._local_obstacle_escape_start_time = None
        node._nav_start_pose_recovery_active = False
        node._door_retarget_backoff_until = None
        node.target_door = SimpleNamespace(door_id='locked', door_color='blue')
        node.target_door_pub = Mock()
        return node

    def test_each_manual_recovery_defers_goal_refinement(self):
        for flag in ('_local_obstacle_escape_start_time',
                     '_nav_start_pose_recovery_active', '_door_retarget_backoff_until'):
            with self.subTest(flag=flag):
                node = self.recovery_node()
                setattr(node, flag, True)
                target = node.target_door
                node._maybe_refine_locked_blue_target(SimpleNamespace(door_color='blue'))
                node.target_door_pub.publish.assert_not_called()
                self.assertIs(node.target_door, target)

    def test_observation_memory_still_updates_during_escape(self):
        node = self.recovery_node()
        node._local_obstacle_escape_start_time = True
        observation = SimpleNamespace(door_color='blue')
        existing = SimpleNamespace(door_color='blue')
        node.detected_doors = [existing]
        node._door_with_side_wall_progress_bias = lambda msg: msg
        node._record_semantic_door_observation = Mock()
        node._is_door_opened_for_observation = lambda msg: False
        node._is_door_abandoned_for_observation = lambda msg: False
        node._is_same_door_id_observation = lambda a, b: True
        node._merge_door_observation = Mock()
        node._record_observed_physical_blue_door = Mock()
        node.door_callback(observation)
        node._record_semantic_door_observation.assert_called_once_with(observation)
        node._merge_door_observation.assert_called_once_with(existing, observation)
        node._record_observed_physical_blue_door.assert_called_once_with(existing)
        node.target_door_pub.publish.assert_not_called()

    def test_normal_navigation_can_process_new_observation(self):
        node = self.recovery_node()
        node._door_open_settle_start_time = None
        node._door_open_align_start_time = None
        node._door_has_map_identity = Mock(return_value=False)
        observation = SimpleNamespace(door_color='blue')
        node._maybe_refine_locked_blue_target(observation)
        node._door_has_map_identity.assert_called_once_with(observation)

    def test_short_lived_opened_state_is_published_without_timer_tick(self):
        node = object.__new__(StateMachineNode)
        node.state = State.OPENING_DOOR
        node.target_door = SimpleNamespace(door_id='observed_handle_42')
        node.get_logger = lambda: Mock()
        node._clear_explore_nav_stuck_watch = Mock()
        node._clear_door_nav_stuck_watch = Mock()
        events = []
        node._publish_state = lambda: events.append((node.state, node.target_door.door_id))
        node._transition(State.DOOR_OPENED)
        node._transition(State.EMERGENCY_STOP)
        self.assertEqual(events, [(State.DOOR_OPENED, 'observed_handle_42'),
                                  (State.EMERGENCY_STOP, 'observed_handle_42')])


if __name__ == '__main__':
    unittest.main()
