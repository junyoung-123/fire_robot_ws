import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'src/fire_robot_manipulation/fire_robot_manipulation/preserved_contact_policy.py'
tree = ast.parse(SOURCE.read_text())
method = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == '_feedback_latch_clear_push')


class Deadline:
    def __init__(self, *args, **kwargs):
        pass

    def failure(self, *args):
        return None


scope = dict(SimMotionDeadline=Deadline, time=SimpleNamespace(monotonic=lambda: 0.,sleep=lambda _: None),
             rclpy=SimpleNamespace(ok=lambda: True))
exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)


class LatchInterruptedTests(unittest.TestCase):
    def fixture(self, angle=0., fresh=True, lever=.3):
        node = SimpleNamespace(
            _sim_door_initial_position=0., _sim_door_position=angle,
            _sim_lever_initial_position=0., _sim_lever_position=lever,
            _sim_lever_press_min_angle=.2, _sim_latch_clear_min_angle=.25,
            _feedback_config=dict(press_sign=1.), probes=0, observations=0, events=[], stops=[])
        node._sim_time_sec = lambda: 0.
        node._feedback_tool = lambda: [0., 0., 0.]
        node._feedback_begin_contact_follow = lambda: True
        node._feedback_latch_motion = lambda: (.02,0.,None)
        node._feedback_event = lambda name, **values: node.events.append((name, values))
        node._publish_sim_cmd_vel = lambda v, w: node.stops.append((v, w))
        node._feedback_stop = lambda reason: ('stop', reason)

        def observe():
            node.observations += 1
            return fresh or node.observations == 1

        def probe(*args):
            node.probes += 1
            if node.probes > 1:
                raise AssertionError('A released latch must not be pushed again')
            node._sim_door_position = .26
            return 'sensor_wait'

        node._feedback_fresh = observe
        node._feedback_probe_contact_motion = probe
        return node

    def test_partial_sensor_paused_pulse_can_transition_on_next_observation(self):
        node = self.fixture()
        self.assertIs(scope['_feedback_latch_clear_push'](node), True)
        self.assertEqual(node.probes, 1)
        self.assertIn(('latch_released', dict(door_delta_rad=.26)), node.events)
        self.assertTrue(all(command == (0., 0.) for command in node.stops))

    def test_initially_released_latch_does_not_receive_another_pulse(self):
        node = self.fixture(angle=.26)
        self.assertIs(scope['_feedback_latch_clear_push'](node), True)
        self.assertEqual(node.probes, 0)

    def test_stale_observation_never_allows_transition(self):
        node = self.fixture(angle=.26, fresh=False)
        self.assertEqual(scope['_feedback_latch_clear_push'](node),
                         ('stop', 'Lost fresh latch release observation'))
        self.assertEqual(node.probes, 0)

    def test_lost_lever_contact_never_allows_transition(self):
        node = self.fixture(angle=.26, lever=.1)
        result = scope['_feedback_latch_clear_push'](node)
        self.assertEqual(result[0], 'stop')
        self.assertEqual(node.probes, 0)


if __name__ == '__main__':
    unittest.main()
