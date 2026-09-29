import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml
from launch import LaunchContext

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'observed_launch_test', ROOT/'src/fire_robot_bringup/launch/observed_navigation.launch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ObservedNavigationLaunchTest(unittest.TestCase):
    def test_no_prior_map_or_truth_odometry_and_unknown_is_not_free(self):
        context = LaunchContext()
        context.launch_configurations['world'] = 'obstacle_wall_doors_v5.world'
        with patch.object(module, 'IncludeLaunchDescription') as include:
            module.start(context)
        args = dict(include.call_args.kwargs['launch_arguments'])
        self.assertEqual(args['localization_mode'], 'mapping')
        self.assertEqual(args['odometry_source'], 'wheel')
        self.assertEqual(args['navigation_only'], 'true')
        self.assertEqual(args['min_opened_doors_before_exit'], '0')
        overrides = yaml.safe_load(Path(args['app_overrides']).read_text())
        self.assertFalse(overrides['state_machine_node']['ros__parameters']['axis_door_lane_bounds_enabled'])
        self.assertFalse(overrides['state_machine_node']['ros__parameters']['nav_start_center_recovery_enabled'])
        self.assertTrue(overrides['cmd_vel_safety_node']['ros__parameters']['lidar_stop_manual_commands'])
        self.assertTrue(overrides['cmd_vel_safety_node']['ros__parameters']['stop_on_missing_scan'])
        params = yaml.safe_load(Path(args['nav2_params_file']).read_text())
        costmap = params['global_costmap']['global_costmap']['ros__parameters']
        self.assertEqual(costmap['static_layer']['map_topic'], '/map')
        self.assertTrue(costmap['track_unknown_space'])
        self.assertNotIn('fixed_obstacle_layer', costmap['plugins'])
        self.assertNotIn('map_server', params)
        self.assertFalse(params['planner_server']['ros__parameters']['GridBased']['allow_unknown'])
        Path(args['nav2_params_file']).unlink()
        slam = yaml.safe_load(Path(args['slam_params_file']).read_text())['slam_toolbox']['ros__parameters']
        self.assertEqual(slam['minimum_travel_distance'], 0.0)
        self.assertGreaterEqual(slam['minimum_time_interval'], .2)
        self.assertFalse(slam['do_loop_closing'])
        self.assertEqual(slam['scan_topic'], '/slam_scan')
        self.assertEqual(slam['transform_publish_period'], 0.0)
        self.assertTrue(slam['use_scan_matching'])
        Path(args['slam_params_file']).unlink()


if __name__ == '__main__':
    unittest.main()
