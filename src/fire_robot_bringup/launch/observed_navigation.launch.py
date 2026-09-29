"""Navigation-only experiment: live SLAM, wheel odometry, no saved world map."""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    bringup = get_package_share_directory('fire_robot_bringup')
    navigation = get_package_share_directory('fire_robot_navigation')
    with open(os.path.join(navigation, 'config', 'nav2_params.yaml')) as stream:
        params = yaml.safe_load(stream)
    costmap = params['global_costmap']['global_costmap']['ros__parameters']
    costmap['track_unknown_space'] = True
    # The SLAM grid ends at finite returns. Keep planning space around the
    # robot so valid no-return rays can clear beyond an observed open doorway.
    costmap.update(rolling_window=True, width=24, height=24,
                   update_frequency=2.0, publish_frequency=1.0)
    costmap.pop('origin_x', None)
    costmap.pop('origin_y', None)
    costmap['plugins'] = ['static_layer', 'obstacle_layer', 'inflation_layer']
    costmap.pop('fixed_obstacle_layer', None)
    costmap['static_layer']['map_topic'] = '/map'
    costmap['static_layer']['subscribe_to_updates'] = True
    for layer in (costmap, params['local_costmap']['local_costmap']['ros__parameters']):
        layer['obstacle_layer']['scan']['inf_is_valid'] = True
    params['planner_server']['ros__parameters']['GridBased']['allow_unknown'] = False
    # Unknown remains blocked; only measured ray paths can clear unknown cells.
    params.pop('map_server', None)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     prefix='observed_nav_', delete=False) as stream:
        yaml.safe_dump(params, stream, sort_keys=False)
        runtime_params = stream.name
    with open(os.path.join(navigation, 'config', 'slam_toolbox_params.yaml')) as stream:
        slam = yaml.safe_load(stream)
    slam['slam_toolbox']['ros__parameters'].update({
        # Wheel slip during skid-steer turns requires more frequent scan matching
        # and a larger angular search than the saved-map experiment ever used.
        # Humble shouldProcessScan has a translation-only pre-gate. Zero lets
        # in-place turns reach the matcher; time throttling bounds the workload.
        'minimum_time_interval': 0.25,
        'minimum_travel_distance': 0.0,
        'minimum_travel_heading': 0.08,
        'coarse_search_angle_offset': 0.90,
        'coarse_angle_resolution': 0.035,
        'fine_search_angle_offset': 0.01,
        'map_update_interval': 1.0,
        'enable_interactive_mode': False,
        # Repeated corridor panels can form false loop constraints several
        # metres apart. Keep online local scan matching, but do not deform
        # the map from unverified place-recognition matches in this profile.
        'do_loop_closing': False,
        'scan_topic': '/slam_scan',
        # Installed Humble 2.6.10 predates restamp_tf. The correction relay
        # below is the sole map-to-odom publisher; scans retain their stamps.
        'transform_publish_period': 0.0,
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     prefix='observed_slam_', delete=False) as stream:
        yaml.safe_dump(slam, stream, sort_keys=False)
        runtime_slam = stream.name
    return [Node(package='fire_robot_navigation', executable='motion_scan_filter',
                 parameters=[{'use_sim_time': True}], output='screen'),
            Node(package='fire_robot_navigation', executable='slam_correction_tf',
                 parameters=[{'use_sim_time': True}], output='screen'),
            IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'simulation.launch.py')),
        launch_arguments={
            'world': LaunchConfiguration('world').perform(context),
            'use_rviz': 'false', 'headless': 'true',
            'localization_mode': 'mapping', 'odometry_source': 'wheel',
            'navigation_only': 'true', 'launch_moveit': 'false',
            'nav2_params_file': runtime_params,
            'slam_params_file': runtime_slam,
            'app_overrides': os.path.join(bringup, 'config', 'observed_navigation.yaml'),
            'enable_segformer': 'false', 'start_without_fire': 'true',
            'min_opened_doors_before_exit': '0',
            'log_detection_candidates': 'true',
        }.items(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='obstacle_wall_doors_v5.world'),
        OpaqueFunction(function=start),
    ])
