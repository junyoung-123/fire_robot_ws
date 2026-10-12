"""Navigation-only experiment: live SLAM, wheel odometry, no saved world map."""
import os
import math
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
        if LaunchConfiguration('robot_xacro_file').perform(context) == 'fire_robot_actual_piper.urdf.xacro':
            from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE
            # Include the calibrated fixed-link cameras, not just the base box.
            layer['footprint'] = repr([[float(x), float(y)] for x, y in PIPER_BODY_OUTLINE])
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
    app_overrides = os.path.join(bringup, 'config', 'observed_navigation.yaml')
    if LaunchConfiguration('guarded_post_open').perform(context).lower() == 'true':
        from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE
        scan_x = float(LaunchConfiguration('lidar_tf_x').perform(context))
        # Bound a full rotation about base_link from the actual body outline
        # and calibrated scanner origin, not a world/door position.
        turn_clearance = max(math.hypot(x-scan_x, y) for x, y in PIPER_BODY_OUTLINE) + .05
        with open(app_overrides) as stream:
            app = yaml.safe_load(stream)
        app['cmd_vel_safety_node']['ros__parameters']['rotation_stop_distance_m'] = turn_clearance
        app['cmd_vel_safety_node']['ros__parameters']['nav_swept_clearance_enabled'] = True
        bounds = [float(PIPER_BODY_OUTLINE[:,0].min()), float(PIPER_BODY_OUTLINE[:,0].max()),
                  float(PIPER_BODY_OUTLINE[:,1].min()), float(PIPER_BODY_OUTLINE[:,1].max())]
        for name in ('cmd_vel_safety_node', 'state_machine_node'):
            app[name]['ros__parameters'].update({
                'guarded_footprint_enabled': True, 'guard_footprint_bounds_m': list(bounds),
                'guard_static_free_history_sec': float(
                    LaunchConfiguration('guard_static_free_history_sec').perform(context))})
        app['state_machine_node']['ros__parameters'].update({
            # All mission motions need body sweeps. Only the separately
            # checked arm/contact controller uses the near-contact channel.
            'manual_cmd_vel_topic': '/cmd_vel_manual_guarded',
            'post_open_observed_clearance_enabled': True,
            'post_open_guarded_cmd_vel_topic': '/cmd_vel_manual_guarded',
            'post_open_scan_turn_min_m': turn_clearance,
            'post_open_side_retreat_timeout_sec': 90.0,
        })
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         prefix='observed_contact_app_', delete=False) as stream:
            yaml.safe_dump(app, stream, sort_keys=False)
            app_overrides = stream.name
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
            'enable_lifecycle_recovery': 'true',
            'nav2_params_file': runtime_params,
            'slam_params_file': runtime_slam,
            'app_overrides': app_overrides,
            'enable_segformer': 'false', 'start_without_fire': 'true',
            'min_opened_doors_before_exit': '0',
            'log_detection_candidates': 'true',
            'physical_contact': LaunchConfiguration('physical_contact').perform(context),
            'robot_xacro_file': LaunchConfiguration('robot_xacro_file').perform(context),
            'enable_depth_camera': LaunchConfiguration('enable_depth_camera').perform(context),
            'spawn_z': LaunchConfiguration('spawn_z').perform(context),
            'spawn_x': LaunchConfiguration('spawn_x').perform(context),
            'spawn_y': LaunchConfiguration('spawn_y').perform(context),
            'spawn_yaw': LaunchConfiguration('spawn_yaw').perform(context),
            'enable_mission_fsm': LaunchConfiguration('enable_mission_fsm').perform(context),
            'lidar_tf_x': LaunchConfiguration('lidar_tf_x').perform(context),
            'lidar_tf_z': LaunchConfiguration('lidar_tf_z').perform(context),
            'raw_scan_topic': LaunchConfiguration('raw_scan_topic').perform(context),
            'clock_output_topic': LaunchConfiguration('clock_output_topic').perform(context),
            'joint_states_output_topic': LaunchConfiguration('joint_states_output_topic').perform(context),
            'publish_debug_image': 'true',
        }.items(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='obstacle_wall_doors_v5.world'),
        DeclareLaunchArgument('physical_contact', default_value='false'),
        DeclareLaunchArgument('robot_xacro_file', default_value='fire_robot.urdf.xacro'),
        DeclareLaunchArgument('enable_depth_camera', default_value='false'),
        DeclareLaunchArgument('spawn_z', default_value='0.07'),
        DeclareLaunchArgument('spawn_x', default_value='-3.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument('enable_mission_fsm', default_value='true'),
        DeclareLaunchArgument('guarded_post_open', default_value='false'),
        DeclareLaunchArgument('guard_static_free_history_sec', default_value='2.0'),
        DeclareLaunchArgument('lidar_tf_x', default_value='0.2'),
        DeclareLaunchArgument('lidar_tf_z', default_value='0.07'),
        DeclareLaunchArgument('raw_scan_topic', default_value='/scan'),
        DeclareLaunchArgument('clock_output_topic', default_value='/clock'),
        DeclareLaunchArgument('joint_states_output_topic', default_value='/joint_states'),
        OpaqueFunction(function=start),
    ])
