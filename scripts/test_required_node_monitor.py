import tempfile
import unittest
from pathlib import Path
from run_world1_full import failed_required_node


class RequiredNodeMonitorTests(unittest.TestCase):
    def test_critical_child_exit_is_detected_even_if_launch_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'simulation.log'
            path.write_text('[ERROR] [robot_observation_manipulation_node-16]: process has died [exit code 1]\n')
            self.assertEqual(failed_required_node(path), 'robot_observation_manipulation_node')

    def test_normal_stop_event_is_not_a_process_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'simulation.log'
            path.write_text('[INFO] [robot_observation_manipulation_node]: CONTACT_FEEDBACK stopped\n')
            self.assertIsNone(failed_required_node(path))

    def test_no_log_before_launch_is_not_a_failure(self):
        self.assertIsNone(failed_required_node(Path('/nonexistent/simulation.log')))

    def test_navigation_perception_and_safety_exit_stop_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'simulation.log'
            for name in ('door_detection_node', 'state_machine_node', 'cmd_vel_safety_node'):
                with self.subTest(node=name):
                    path.write_text(f'[ERROR] [{name}-7]: process has died [exit code 1]\n')
                    self.assertEqual(failed_required_node(path), name)


if __name__ == '__main__':
    unittest.main()
