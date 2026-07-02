from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz     = LaunchConfiguration('use_rviz',     default='true')
    headless     = LaunchConfiguration('headless',     default='false')
    world_file   = LaunchConfiguration('world',        default='obstacle_wall_doors_v5.world')
    start_without_fire = LaunchConfiguration('start_without_fire', default='true')
    launch_moveit = LaunchConfiguration('launch_moveit', default='false')
    localization_mode = LaunchConfiguration('localization_mode', default='localization')
    enable_segformer = LaunchConfiguration('enable_segformer', default='true')
    localization_start_delay = LaunchConfiguration('localization_start_delay', default='8.0')
    nav2_start_delay = LaunchConfiguration('nav2_start_delay', default='12.0')
    app_start_delay = LaunchConfiguration('app_start_delay', default='18.0')
    map_file = LaunchConfiguration('map', default=PathJoinSubstitution([
        FindPackageShare('fire_robot_navigation'), 'maps', 'obstacle_wall_doors_v5_static.yaml',
    ]))
    nav2_params = PathJoinSubstitution([
        FindPackageShare('fire_robot_navigation'), 'config', 'nav2_params.yaml',
    ])

    slam_params = PathJoinSubstitution([
        FindPackageShare('fire_robot_navigation'), 'config', 'slam_toolbox_params.yaml',
    ])

    # ── 1. Gazebo + 로봇 스폰 + 브릿지 ─────────────────────
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fire_robot_bringup'), 'launch', 'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'headless':     headless,
            'world':        world_file,
        }.items(),
    )

    # ── 2a. 초기 지도 작성 모드: SLAM mapping ────────────
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'), 'launch', 'online_async_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time':      use_sim_time,
            'slam_params_file':  slam_params,
        }.items(),
        condition=IfCondition(PythonExpression(["'", localization_mode, "' == 'mapping'"])),
    )

    # ── 2b. 저장 지도 기반 위치추정 모드: map_server + AMCL ─
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'localization_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'map': map_file,
            'params_file': nav2_params,
        }.items(),
        condition=IfCondition(PythonExpression(["'", localization_mode, "' == 'localization'"])),
    )

    # ── 3. Nav2 자율주행 ──────────────────────────────────
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params,
        }.items(),
    )

    # ── 4. MoveIt2 (시뮬레이션 모드 — manipulation_node는 sim_mode=True) ──
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fire_robot_manipulation'), 'launch', 'move_group.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'launch_manipulation': 'false',
        }.items(),
        condition=IfCondition(launch_moveit),
    )

    # ── 5. 애플리케이션 노드 (Gazebo 완전 기동 후 3초 지연) ─
    app_nodes = TimerAction(
        period=app_start_delay,
        actions=[
            Node(
                package='fire_robot_perception',
                executable='sensor_fusion_node',
                name='sensor_fusion_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'publish_rate': 1.0,
                    'enable_segformer': enable_segformer,
                    'segformer_torch_threads': 1,
                    'max_segmentation_jobs_per_cycle': 1,
                    'map_frame': 'map',
                    'base_frame': 'base_link',
                    'lidar_frame': 'lidar_link',
                    'map_resolution': 0.05,
                    'map_width_m': 40.0,
                    'map_height_m': 20.0,
                    'map_origin_x': -10.0,
                    'map_origin_y': -10.0,
                    'lidar_max_range_m': 8.0,
                    'lidar_obstacle_memory_sec': 2.0,
                    'obstacle_dilation_radius_m': 0.06,
                    'camera_obstacle_dilation_radius_m': 0.18,
                    'min_obstacle_mark_range_m': 0.85,
                    'robot_clear_radius_m': 0.85,
                    'segmentation_min_interval_sec': 3.0,
                    'camera_sources': [
                        'front|/camera/color/image_raw|/camera/color/camera_info|0.0',
                        'front_left|/camera/front_left/image_raw|/camera/front_left/camera_info|70.0',
                        'front_right|/camera/front_right/image_raw|/camera/front_right/camera_info|-70.0',
                    ],
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_perception',
                executable='door_detection_node',
                name='door_detection_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'door_approach_offset_m': 1.5,
                    'nav_goal_max_abs_y_m': 0.40,
                    'publish_map_frame': True,
                    'camera_sources': [
                        'front|/camera/color/image_raw|/camera/color/camera_info|0.0',
                        'front_left|/camera/front_left/image_raw|/camera/front_left/camera_info|70.0',
                        'front_right|/camera/front_right/image_raw|/camera/front_right/camera_info|-70.0',
                    ],
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_navigation',
                executable='navigation_node',
                name='navigation_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'max_goal_step_m': 1.20,
                    'adjust_goal_to_free_space': True,
                    'goal_adjust_search_radius_m': 1.6,
                    'goal_adjust_max_cost': 20,
                    'goal_adjust_max_abs_y_m': 0.60,
                    'goal_adjust_min_progress_m': 0.35,
                    'fallback_goal_reached_dist_m': 0.85,
                    'lane_biased_intermediate_goals': True,
                    'lane_goal_min_abs_y_m': 0.35,
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_manipulation',
                executable='manipulation_node',
                name='manipulation_node',
                parameters=[{
                    'use_sim_time':     use_sim_time,
                    'sim_mode':         True,   # Gazebo 시뮬레이션: 슬립 기반 시뮬
                    'planning_group':   'piper_arm',
                    'gripper_group':    'piper_gripper',
                    'velocity_scaling': 0.5,
                }],
                output='screen',
            ),
            TimerAction(
                period=25.0,
                actions=[
                    Node(
                        package='fire_robot_fsm',
                        executable='state_machine_node',
                        name='state_machine_node',
                        parameters=[{
                            'use_sim_time':        use_sim_time,
                            'nav_timeout_sec':     120.0,
                            'exit_x':              17.2,   # obstacle_wall_doors_v5.world 비상구 접근 X
                            'exit_y':               0.0,
                            'exit_yaw':             0.0,
                            'explore_timeout_sec': 60.0,
                            'explore_linear_vel':   0.0,
                            'explore_angular_vel':  0.0,
                            'explore_obstacle_stop_m': 0.75,
                            'explore_obstacle_slow_m': 1.85,
                            'explore_avoid_angular_vel': 0.72,
                            'explore_side_clearance_m': 0.78,
                            'explore_scan_min_range_m': 0.85,
                            'explore_color_lane_bias': True,
                            'explore_forward_yaw':  0.0,
                            'explore_heading_kp':   0.9,
                            'explore_heading_soft_limit_deg': 50.0,
                            'explore_heading_hard_limit_deg': 115.0,
                            'explore_center_y':     0.0,
                            'explore_y_soft_limit_m': 1.25,
                            'explore_y_hard_limit_m': 1.60,
                            'explore_center_lookahead_m': 1.20,
                            'prefer_forward_doors': True,
                            'mission_start_x':     -1.0,
                            'next_door_min_forward_m': 2.50,
                            'opened_door_merge_dist_m': 2.50,
                            'min_opened_doors_before_exit': 1,
                            'complete_when_past_exit_x': True,
                            'exit_complete_margin_m': 0.60,
                            'final_scan_before_exit_sec': 8.0,
                            'final_scan_angular_vel': 0.45,
                            'pre_nav_scan_sec': 1.5,
                            'pre_nav_scan_angular_vel': 0.0,
                            'explore_nav_enabled': True,
                            'explore_nav_step_m': 1.50,
                            'explore_nav_lane_y_m': 0.35,
                            'explore_nav_lane_width_m': 0.65,
                            'explore_nav_scan_lookahead_m': 3.00,
                            'explore_nav_color_bias_score': 0.55,
                            'explore_nav_keep_lane_bias_score': 0.18,
                            'explore_nav_front_obstacle_width_m': 1.00,
                            'explore_nav_front_obstacle_bias_score': 3.0,
                            'explore_nav_blue_door_bias_score': 2.4,
                            'explore_nav_blue_lane_min_clearance_m': 0.78,
                            'max_explore_nav_failures_before_exit': 8,
                            'explore_nav_timeout_sec': 60.0,
                            'min_target_door_abs_y_m': 0.35,
                            'target_door_direct_nav_max_dist_m': 2.0,
                            'nav_start_max_abs_y_m': 1.35,
                            'nav_start_max_heading_error_deg': 80.0,
                            'start_without_fire':   start_without_fire,
                        }],
                        output='screen',
                    ),
                ],
            ),
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('fire_robot_bringup'), 'rviz', 'simulation.rviz'
        ])],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('use_rviz',     default_value='true'),
        DeclareLaunchArgument('world',        default_value='obstacle_wall_doors_v5.world'),
        DeclareLaunchArgument('start_without_fire', default_value='true'),
        DeclareLaunchArgument('launch_moveit', default_value='false'),
        DeclareLaunchArgument(
            'enable_segformer',
            default_value='true',
            description='Enable camera semantic segmentation. Disable for CPU-only navigation stress tests.',
        ),
        DeclareLaunchArgument(
            'localization_mode',
            default_value='localization',
            description='mapping: run SLAM to create/update a map; localization: use saved map + AMCL.',
        ),
        DeclareLaunchArgument(
            'localization_start_delay',
            default_value='8.0',
            description='Delay localization/SLAM startup until Gazebo has spawned the robot and TF begins.',
        ),
        DeclareLaunchArgument(
            'nav2_start_delay',
            default_value='12.0',
            description='Delay Nav2 startup until localization and odom TF are available.',
        ),
        DeclareLaunchArgument(
            'app_start_delay',
            default_value='18.0',
            description='Delay perception/navigation/FSM app nodes until Nav2 startup is underway.',
        ),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_navigation'), 'maps', 'obstacle_wall_doors_v5_static.yaml',
            ]),
            description='Map yaml used when localization_mode:=localization.',
        ),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Gazebo without GUI when true.',
        ),

        gazebo_launch,   # Gazebo + robot_state_publisher + spawn + bridge
        TimerAction(
            period=localization_start_delay,
            actions=[
                slam_launch,           # mapping 모드: SLAM → /map 발행
                localization_launch,   # localization 모드: static map + AMCL
            ],
        ),
        TimerAction(
            period=nav2_start_delay,
            actions=[
                nav2_launch,           # Nav2 → /navigate_to_pose
            ],
        ),
        moveit_launch,   # optional move_group (sim door opening uses manipulation_node sim_mode)
        rviz_node,       # RViz2 시각화 (use_rviz:=false 로 비활성화 가능)
        app_nodes,       # 인지/주행/조작/FSM 노드 (3초 후 기동)
    ])
