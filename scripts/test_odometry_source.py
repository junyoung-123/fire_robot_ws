import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from launch import LaunchContext


PATH = (Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/launch/gazebo.launch.py')
spec = importlib.util.spec_from_file_location('gazebo_launch_under_test', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OdometrySourceTests(unittest.TestCase):
    def test_wheel_mode_never_bridges_ground_truth(self):
        self.assertEqual(module.odometry_topics('wheel'), ('/wheel_odom', '/wheel_tf'))

    def test_baseline_mode_is_preserved(self):
        self.assertEqual(module.odometry_topics('ground_truth'), ('/odom', '/tf'))

    def test_invalid_source_is_not_silently_ground_truth(self):
        with self.assertRaises(ValueError):
            module.odometry_topics('typo')

    def test_selected_bridge_keeps_ros_interface_and_sim_time(self):
        for source in ('wheel', 'ground_truth'):
            context = LaunchContext()
            context.launch_configurations.update(odometry_source=source, use_sim_time='true')
            with patch.object(module, 'Node', wraps=module.Node) as constructor:
                nodes = module.create_odometry_bridge(context)
            self.assertEqual(len(nodes), 1)
            config = constructor.call_args.kwargs
            self.assertEqual(config['remappings'],
                             list(zip(module.odometry_topics(source), ('/odom', '/tf'))))
            clock = config['parameters'][0]['use_sim_time']
            self.assertEqual(clock.perform(context), 'true')
            self.assertEqual(config['arguments'][0].split('@')[0], module.odometry_topics(source)[0])


if __name__ == '__main__':
    unittest.main()
