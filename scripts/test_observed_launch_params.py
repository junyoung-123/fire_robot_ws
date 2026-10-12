"""Exercise generated ROS parameter files, not only their Python templates."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from launch import LaunchContext


class ObservedLaunchParamsTests(unittest.TestCase):
    def test_generated_contact_parameters_have_no_yaml_aliases(self):
        path = Path(__file__).resolve().parents[1]/'src/fire_robot_bringup/launch/observed_navigation.launch.py'
        spec = importlib.util.spec_from_file_location('observed_navigation_launch', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        context = LaunchContext()
        context.launch_configurations.update({
            'guarded_post_open': 'true', 'robot_xacro_file': 'fire_robot_actual_piper.urdf.xacro',
            'guard_static_free_history_sec': '30.0',
            'lidar_tf_x': '0.25', 'lidar_tf_z': '0.248', 'world': 'world1_full_contact.world',
            'physical_contact': 'true', 'enable_depth_camera': 'true', 'spawn_z': '0.07',
            'spawn_x': '-3.0', 'spawn_y': '0.0', 'spawn_yaw': '0.0',
            'enable_mission_fsm': 'true', 'raw_scan_topic': '/scan_raw',
            'clock_output_topic': '/clock', 'joint_states_output_topic': '/joint_states'})
        original = tempfile.NamedTemporaryFile
        paths = []

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            paths.append(Path(result.name))
            return result

        try:
            with patch.object(module.tempfile, 'NamedTemporaryFile', capture):
                module.start(context)
            self.assertEqual(len(paths), 3)
            for generated in paths:
                content = generated.read_text()
                self.assertFalse(any(isinstance(event, yaml.events.AliasEvent)
                                     for event in yaml.parse(content)), generated.name)
            app = yaml.safe_load(paths[-1].read_text())
            for name in ('cmd_vel_safety_node', 'state_machine_node'):
                params = app[name]['ros__parameters']
                self.assertTrue(params['guarded_footprint_enabled'])
                self.assertEqual(params['guard_static_free_history_sec'],30.)
                self.assertEqual(len(params['guard_footprint_bounds_m']), 4)
        finally:
            for generated in paths:
                generated.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
