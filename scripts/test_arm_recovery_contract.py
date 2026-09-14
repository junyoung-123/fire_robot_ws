#!/usr/bin/env python3
"""No command or timer can substitute for finite joint feedback at home."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fire_robot_manipulation.manipulation_node import ManipulationNode
from fire_robot_fsm.state_machine_node import StateMachineNode, State


class ArmRecoveryTests(unittest.TestCase):
    def test_feedback_must_contain_every_joint(self):
        targets = {'joint1': 0.0, 'joint2': 0.0}
        self.assertEqual(ManipulationNode._sim_arm_target_error(targets, {'joint1': 0.0}), float('inf'))
        self.assertEqual(ManipulationNode._sim_arm_target_error(targets, {'joint1': 0.0, 'joint2': float('nan')}), float('inf'))
        self.assertAlmostEqual(ManipulationNode._sim_arm_target_error(targets, {'joint1': .01, 'joint2': 1.72}), 1.72)

    def test_stale_zero_positions_cannot_pass_home(self):
        arm = object.__new__(ManipulationNode)
        arm._sim_arm_positions = {f'joint{i}': 0.0 for i in range(1, 7)}
        arm._sim_arm_feedback_time = 0.0
        arm._sim_arm_home_timeout = 0.2
        arm._sim_arm_home_tolerance = .10
        arm.get_logger = lambda: SimpleNamespace(info=lambda *a: None, error=lambda *a: None)
        arm._publish_sim_arm_targets = lambda _: None
        clock = [10.0]
        def advance(delay):
            clock[0] += delay
        with patch('fire_robot_manipulation.manipulation_node.time.monotonic', side_effect=lambda: clock[0]), patch('fire_robot_manipulation.manipulation_node.time.sleep', side_effect=advance):
            self.assertFalse(arm._wait_for_sim_arm_home())

    def test_fresh_stable_feedback_passes_home(self):
        arm = object.__new__(ManipulationNode)
        arm._sim_arm_positions = {f'joint{i}': .01 for i in range(1, 7)}
        arm._sim_arm_feedback_time = 10.0
        arm._sim_arm_home_timeout = 2.0
        arm._sim_arm_home_tolerance = .10
        arm.get_logger = lambda: SimpleNamespace(info=lambda *a: None, error=lambda *a: None)
        arm._publish_sim_arm_targets = lambda _: None
        clock = [10.0]
        def advance(delay):
            clock[0] += delay
            arm._sim_arm_feedback_time = clock[0]
        with patch('fire_robot_manipulation.manipulation_node.time.monotonic', side_effect=lambda: clock[0]), patch('fire_robot_manipulation.manipulation_node.time.sleep', side_effect=advance):
            self.assertTrue(arm._wait_for_sim_arm_home())
            self.assertGreaterEqual(clock[0], 10.5)

    def test_recovery_failure_stops_fsm_instead_of_trying_next_door(self):
        fsm = object.__new__(StateMachineNode)
        transitions = []
        fsm.get_logger = lambda: SimpleNamespace(error=lambda *a: None)
        fsm._transition = transitions.append
        future = SimpleNamespace(result=lambda: SimpleNamespace(success=False, message='ARM_NOT_STOWED: timeout'))
        fsm._door_open_result(future)
        self.assertEqual(transitions, [State.EMERGENCY_STOP])


if __name__ == '__main__':
    unittest.main()
