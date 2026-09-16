#!/usr/bin/env python3
"""A failed test harness must not orphan its Gazebo launch process."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py'
SPEC = importlib.util.spec_from_file_location('physical_fixture_runner_under_test', PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class LifecycleTests(unittest.TestCase):
    def args(self, root):
        return SimpleNamespace(
            feedback_contact=False, use_yolo_observation=False,
            attach_grasp_joint=False, extra_sweep=False, mobile_follow=False,
            output_root=root, attach=False, headless=True, door_open_motion='push',
            min_angle_rad=2.05, handle_x=.43, handle_y=.24, handle_z=.66,
            launch_settle_sec=0., ready_timeout_sec=1.)

    def test_failed_ros_initialization_never_launches_gazebo(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.object(RUNNER.rclpy, 'init', side_effect=RuntimeError('invalid DDS configuration')), \
                patch.object(RUNNER, '_start_launch') as launch:
            with self.assertRaisesRegex(RuntimeError, 'invalid DDS'):
                RUNNER.run(self.args(root))
            launch.assert_not_called()

    def test_cleanup_error_cannot_skip_owned_launch_shutdown(self):
        node = Mock()
        node.destroy_node.side_effect = RuntimeError('node cleanup failed')
        process = object()
        with tempfile.TemporaryDirectory() as root, \
                patch.object(RUNNER.rclpy, 'init'), \
                patch.object(RUNNER, 'ContactProbe', return_value=node), \
                patch.object(RUNNER, '_start_launch', return_value=process), \
                patch.object(RUNNER, '_wait_for_ready', side_effect=RuntimeError('readiness failed')), \
                patch.object(RUNNER, '_stop_launch') as stop:
            with self.assertRaisesRegex(RuntimeError, 'node cleanup failed'):
                RUNNER.run(self.args(root))
            stop.assert_called_once_with(process)


if __name__ == '__main__':
    unittest.main()
