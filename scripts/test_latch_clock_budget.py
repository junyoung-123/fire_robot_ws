import ast
from pathlib import Path
import unittest
from fire_robot_manipulation.contact_feedback import SimMotionDeadline


class LatchClockTests(unittest.TestCase):
    def test_low_real_time_factor_does_not_spend_simulated_motion_budget(self):
        deadline = SimMotionDeadline(0., 0., 30., stall_limit=12., wall_limit=360.)
        self.assertIsNone(deadline.failure(10., 150.))
        self.assertIsNone(deadline.failure(20., 300.))
        self.assertEqual(deadline.failure(30., 350.), 'Simulation motion deadline exceeded')

    def test_stalled_clock_and_absolute_cap_are_still_bounded(self):
        deadline = SimMotionDeadline(0., 0., 30., stall_limit=12., wall_limit=360.)
        self.assertEqual(deadline.failure(0., 12.), 'Simulation clock stalled')
        deadline = SimMotionDeadline(0., 0., 30., stall_limit=12., wall_limit=360.)
        self.assertEqual(deadline.failure(29., 360.), 'Absolute wall deadline exceeded')

    def test_latch_control_uses_the_two_clock_guard(self):
        source = Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation/fire_robot_manipulation/preserved_contact_policy.py'
        tree = ast.parse(source.read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == '_feedback_latch_clear_push')
        call = next(node for node in ast.walk(method) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'SimMotionDeadline')
        self.assertEqual(ast.literal_eval(call.args[2]), 30.)
        self.assertEqual({item.arg: ast.literal_eval(item.value) for item in call.keywords},
                         dict(stall_limit=12., wall_limit=360.))


if __name__ == '__main__':
    unittest.main()
