from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                             IncludeLaunchDescription, TimerAction)
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
    min_opened_doors_before_exit = LaunchConfiguration('min_opened_doors_before_exit', default='0')
    launch_moveit = LaunchConfiguration('launch_moveit', default='false')
    localization_mode = LaunchConfiguration('localization_mode', default='sim_odom')
    spawn_x = LaunchConfiguration('spawn_x', default='-3.0')
    spawn_y = LaunchConfiguration('spawn_y', default='0.0')
    spawn_yaw = LaunchConfiguration('spawn_yaw', default='0.0')
    enable_segformer = LaunchConfiguration('enable_segformer', default='false')
    localization_start_delay = LaunchConfiguration('localization_start_delay', default='8.0')
    localization_recover_delay = LaunchConfiguration('localization_recover_delay', default='18.0')
    nav2_start_delay = LaunchConfiguration('nav2_start_delay', default='12.0')
    nav2_recover_delay = LaunchConfiguration('nav2_recover_delay', default='45.0')
    app_start_delay = LaunchConfiguration('app_start_delay', default='28.0')
    enable_lifecycle_recovery = LaunchConfiguration('enable_lifecycle_recovery', default='false')
    map_file = LaunchConfiguration('map', default=PathJoinSubstitution([
        FindPackageShare('fire_robot_navigation'), 'maps', 'obstacle_wall_doors_v5_static.yaml',
    ]))
    door_model_path = LaunchConfiguration('door_model_path', default=PathJoinSubstitution([
        FindPackageShare('fire_robot_perception'), 'models', 'best.pt',
    ]))
    handle_model_path = LaunchConfiguration('handle_model_path', default=PathJoinSubstitution([
        FindPackageShare('fire_robot_perception'), 'models', 'handle_best_v2.pt',
    ]))
    publish_debug_image = LaunchConfiguration('publish_debug_image', default='false')
    log_handle_detections = LaunchConfiguration('log_handle_detections', default='false')
    log_detection_candidates = LaunchConfiguration('log_detection_candidates', default='false')
    sim_world_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_bringup'), 'worlds', world_file,
    ])
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
            'spawn_x':      spawn_x,
            'spawn_y':      spawn_y,
            'spawn_yaw':    spawn_yaw,
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

    # Gazebo's differential-drive odometry is deterministic relative to the
    # configured spawn pose. Use it as the localization source for repeatable
    # policy validation while still serving the immutable static map. AMCL
    # remains available through localization_mode:=localization for dedicated
    # localization stress tests and the real-robot launch path.
    sim_odom_map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_file,
        }],
        condition=IfCondition(PythonExpression([
            "'", localization_mode, "' == 'sim_odom'",
        ])),
        output='screen',
    )
    sim_odom_map_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server'],
        }],
        condition=IfCondition(PythonExpression([
            "'", localization_mode, "' == 'sim_odom'",
        ])),
        output='screen',
    )
    sim_odom_map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='sim_odom_map_tf',
        arguments=[
            '0.0', '0.0', '0.0',
            '0.0', '0.0', '0.0',
            'map', 'odom',
        ],
        condition=IfCondition(PythonExpression([
            "'", localization_mode, "' == 'sim_odom'",
        ])),
        output='screen',
    )

    # ── 3. Nav2 자율주행 ──────────────────────────────────
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fire_robot_bringup'),
                'launch', 'navigation_no_waypoint_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': nav2_params,
        }.items(),
    )

    localization_lifecycle_recovery = TimerAction(
        period=localization_recover_delay,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash',
                    '-lc',
                    (
                        'echo "[localization_lifecycle_recovery] checking localization lifecycle"; '
                        'map_ready() { timeout 5s ros2 topic echo /map nav_msgs/msg/OccupancyGrid --once >/dev/null 2>&1; }; '
                        'for attempt in 1 2 3 4; do '
                        'if timeout 5s ros2 lifecycle get /amcl 2>/dev/null | grep -Eq "^active" && map_ready; then '
                        'echo "[localization_lifecycle_recovery] AMCL/map already active"; exit 0; '
                        'fi; '
                        'echo "[localization_lifecycle_recovery] recovery attempt ${attempt}"; '
                        'for node in /map_server /amcl; do '
                        'timeout 8s ros2 lifecycle set "${node}" configure >/dev/null 2>&1 || true; '
                        'timeout 8s ros2 lifecycle set "${node}" activate >/dev/null 2>&1 || true; '
                        'done; '
                        'sleep 2; '
                        'done; '
                        'if timeout 5s ros2 lifecycle get /amcl 2>/dev/null | grep -Eq "^active" && map_ready; then '
                        'echo "[localization_lifecycle_recovery] AMCL/map active after recovery"; '
                        'else '
                        'echo "[localization_lifecycle_recovery] AMCL/map still inactive after recovery attempts"; '
                        'fi'
                    ),
                ],
                output='screen',
            ),
        ],
        condition=IfCondition(PythonExpression([
            "'", enable_lifecycle_recovery, "' == 'true' and '",
            localization_mode, "' == 'localization'",
        ])),
    )

    sim_odom_map_lifecycle_recovery = TimerAction(
        period=localization_recover_delay,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash',
                    '-lc',
                    (
                        'echo "[sim_odom_map_lifecycle_recovery] checking map_server"; '
                        'map_ready() { timeout 5s ros2 topic echo /map nav_msgs/msg/OccupancyGrid --once >/dev/null 2>&1; }; '
                        'for attempt in 1 2 3 4; do '
                        'if timeout 5s ros2 lifecycle get /map_server 2>/dev/null | grep -Eq "^active" && map_ready; then '
                        'echo "[sim_odom_map_lifecycle_recovery] map_server active"; exit 0; '
                        'fi; '
                        'echo "[sim_odom_map_lifecycle_recovery] recovery attempt ${attempt}"; '
                        'timeout 8s ros2 lifecycle set /map_server configure >/dev/null 2>&1 || true; '
                        'timeout 8s ros2 lifecycle set /map_server activate >/dev/null 2>&1 || true; '
                        'sleep 2; '
                        'done; '
                        'if timeout 5s ros2 lifecycle get /map_server 2>/dev/null | grep -Eq "^active" && map_ready; then '
                        'echo "[sim_odom_map_lifecycle_recovery] map active after recovery"; '
                        'else '
                        'echo "[sim_odom_map_lifecycle_recovery] map still unavailable"; '
                        'fi'
                    ),
                ],
                output='screen',
            ),
        ],
        condition=IfCondition(PythonExpression([
            "'", enable_lifecycle_recovery, "' == 'true' and '",
            localization_mode, "' == 'sim_odom'",
        ])),
    )

    # ── 4. MoveIt2 (시뮬레이션 모드 — manipulation_node는 sim_mode=True) ──
    nav2_lifecycle_recovery = TimerAction(
        period=nav2_recover_delay,
        actions=[
            ExecuteProcess(
                cmd=[
                    'bash',
                    '-lc',
                    (
                        'echo "[nav2_lifecycle_recovery] checking Nav2 lifecycle"; '
                        'nav2_action_ready() { '
                        'for node in /controller_server /planner_server /bt_navigator; do '
                        'timeout 5s ros2 lifecycle get "${node}" 2>/dev/null | grep -Eq "^active" || return 1; '
                        'done; '
                        'timeout 5s ros2 action info /navigate_to_pose 2>/dev/null | '
                        'grep -Eq "Action servers:[[:space:]]*[1-9]"; '
                        '}; '
                        'for attempt in 1 2 3 4; do '
                        'if nav2_action_ready; then '
                        'echo "[nav2_lifecycle_recovery] Nav2 already active"; exit 0; '
                        'fi; '
                        'echo "[nav2_lifecycle_recovery] recovery attempt ${attempt}"; '
                        'for node in /controller_server /smoother_server /planner_server /behavior_server /bt_navigator /waypoint_follower /velocity_smoother; do '
                        'timeout 8s ros2 lifecycle set "${node}" configure >/dev/null 2>&1 || true; '
                        'timeout 8s ros2 lifecycle set "${node}" activate >/dev/null 2>&1 || true; '
                        'done; '
                        'sleep 2; '
                        'done; '
                        'if nav2_action_ready; then '
                        'echo "[nav2_lifecycle_recovery] Nav2 active after recovery"; '
                        'else '
                        'echo "[nav2_lifecycle_recovery] Nav2 still inactive after recovery attempts"; '
                        'fi'
                    ),
                ],
                output='screen',
            ),
        ],
        condition=IfCondition(PythonExpression([
            "'", enable_lifecycle_recovery, "' == 'true'",
        ])),
    )

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
                package='fire_robot_navigation',
                executable='initial_static_map_node',
                name='initial_static_map_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'source_map_topic': '/map',
                    'static_map_topic': '/initial_static_map',
                    'occupied_threshold': 55,
                    'min_occupied_cells': 1,
                    'capture_delay_sec': 2.0,
                    'lock_after_capture': True,
                    'map_frame': 'map',
                    'base_frame': 'base_link',
                    'pad_forward_m': 34.0,
                    'pad_backward_m': 4.0,
                    'pad_lateral_m': 5.0,
                    'clear_axis_band_m': 0.0,
                    'clear_axis_forward_m': 34.0,
                    'clear_axis_backward_m': 4.0,
                    'min_map_width_m': 34.0,
                    'min_map_height_m': 8.0,
                    'fallback_after_sec': 0.0,
                    'fallback_resolution': 0.05,
                    'fallback_value': 0,
                    'publish_rate_hz': 0.10,
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_navigation',
                executable='fixed_obstacle_map_node',
                name='fixed_obstacle_map_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'static_map_topic': '/initial_static_map',
                    'scan_topic': '/scan',
                    'obstacle_map_topic': '/fixed_obstacle_map',
                    'map_frame': 'map',
                    'base_frame': 'base_link',
                    'min_range_m': 0.15,
                    'max_range_m': 8.0,
                    'mark_radius_m': 0.10,
                    'hit_count_threshold': 3,
                    'robot_clear_radius_m': 1.05,
                    'center_y_m': 0.0,
                    'max_abs_y_m': 1.65,
                    'unknown_value': 0,
                    'publish_rate_hz': 0.25,
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_navigation',
                executable='mission_axis_node',
                name='mission_axis_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'map_topic': '/initial_static_map',
                    'map_frame': 'map',
                    'base_frame': 'base_link',
                    'lock_origin_to_initial_robot_pose': True,
                    'lock_axis_after_first_estimate': True,
                    'fallback_yaw': 0.0,
                    'max_robot_heading_deviation_deg': 3.0,
                    'fallback_without_map_after_sec': 10.0,
                    'publish_rate_hz': 1.0,
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_perception',
                executable='sensor_fusion_node',
                name='sensor_fusion_node',
                parameters=[{
                    'use_sim_time': use_sim_time,
                    'publish_rate': 0.25,
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
                    'lock_origin_to_initial_robot_pose': True,
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
                    'model_path': door_model_path,
                    'handle_model_path': handle_model_path,
                    'confidence_threshold': 0.20,
                    'yolo_min_interval_sec': 2.40,
                    'yolo_imgsz': 320,
                    'torch_num_threads': 1,
                    'reuse_yolo_detections': False,
                    'image_process_min_interval_sec': 1.20,
                    'front_image_process_min_interval_sec': 0.90,
                    'side_image_process_min_interval_sec': 1.40,
                    'front_yolo_min_interval_sec': 1.80,
                    'side_yolo_min_interval_sec': 2.80,
                    'publish_debug_image': publish_debug_image,
                    'yolo_far_confidence_distance_m': 4.0,
                    'yolo_far_min_confidence': 0.20,
                    'yolo_side_far_min_confidence': 0.20,
                    'door_approach_offset_m': 1.5,
                    'nav_goal_max_abs_y_m': 0.85,
                    'side_door_min_abs_y_m': 0.35,
                    'side_door_standoff_m': 1.25,
                    'side_handle_max_abs_y_m': 1.85,
                    'detect_handle_enabled': True,
                    'handle_confidence_threshold': 0.25,
                    'handle_yolo_imgsz': 160,
                    'handle_yolo_roi_expand_x_px': 44,
                    'handle_yolo_roi_expand_y_px': 32,
                    'handle_yolo_min_area_px': 3,
                    'handle_yolo_max_area_ratio': 0.16,
                    'handle_yolo_fallback_hsv': True,
                    'handle_search_expand_x_px': 20,
                    'handle_search_expand_y_px': 14,
                    'handle_min_area_px': 5,
                    'handle_max_area_ratio': 0.08,
                    'handle_min_fill_ratio': 0.16,
                    'handle_default_z_m': 0.90,
                    'handle_min_z_m': 0.65,
                    'handle_max_z_m': 1.25,
                    'log_handle_detections': log_handle_detections,
                    'side_range_max_disagreement_ratio': 1.8,
                    'side_range_max_disagreement_m': 0.8,
                    'side_lidar_short_visual_ratio': 0.70,
                    'side_lidar_short_visual_margin_m': 0.90,
                    'side_visual_fallback_max_distance_m': 4.5,
                    'side_wall_projection_enabled': False,
                    'side_wall_projection_min_abs_angle_deg': 35.0,
                    'side_wall_projection_min_extend_m': 0.35,
                    'front_lateral_lidar_prefer_angle_deg': 12.0,
                    'hsv_fallback_max_door_distance_m': 6.8,
                    'hsv_fallback_side_requires_range': True,
                    'side_camera_requires_range': True,
                    'yolo_reject_edge_clipped_doors': True,
                    'yolo_edge_clip_min_width_ratio': 0.08,
                    'yolo_edge_clip_min_height_ratio': 0.50,
                    'yolo_side_edge_clip_allow_max_dist_m': 2.8,
                    'yolo_side_edge_clip_allow_min_conf': 0.22,
                    'yolo_reject_top_clipped_far_doors': True,
                    'yolo_top_clip_min_distance_m': 4.0,
                    'yolo_top_clip_max_bottom_ratio': 0.68,
                    'yolo_top_clip_min_confidence': 0.50,
                    'yolo_far_min_width_px': 20,
                    'yolo_far_min_height_px': 80,
                    'yolo_far_small_allow_min_confidence': 0.55,
                    'hsv_fallback_min_width_px': 30,
                    'hsv_fallback_min_height_px': 85,
                    'hsv_fallback_green_min_height_px': 45,
                    'hsv_fallback_green_min_width_px': 12,
                    'hsv_fallback_green_min_area_ratio': 0.006,
                    'hsv_fallback_green_min_fill_ratio': 0.10,
                    'hsv_fallback_clipped_max_width_ratio': 0.56,
                    'hsv_fallback_min_fill_ratio': 0.35,
                    'hsv_fallback_min_aspect_ratio': 0.75,
                    'blue_min_color_ratio': 0.34,
                    'blue_min_dominance_margin': 0.18,
                    'blue_max_red_ratio': 0.035,
                    'blue_max_green_ratio': 0.055,
                    'red_min_color_ratio': 0.18,
                    'green_min_color_ratio': 0.14,
                    'log_detection_candidates': log_detection_candidates,
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
                    'max_goal_step_m': 0.0,
                    'adjust_goal_to_free_space': True,
                    'goal_adjust_search_radius_m': 2.6,
                    'goal_adjust_max_cost': 35,
                    'goal_adjust_max_abs_y_m': 1.65,
                    'goal_adjust_min_progress_m': 0.35,
                    'goal_adjust_require_reachable': True,
                    'goal_adjust_connectivity_max_cost': 95,
                    'goal_adjust_start_search_radius_m': 1.20,
                    'blue_goal_adjust_lateral_bias': 0.30,
                    'blue_goal_adjust_min_abs_y_m': 0.35,
                    'blue_goal_adjust_min_abs_y_slack_m': 0.85,
                    'blue_goal_adjust_max_abs_y_overshoot_m': 0.16,
                    'blue_goal_adjust_max_frame_lateral_m': 0.60,
                    'fallback_goal_reached_dist_m': 0.85,
                    'precise_goal_reached_dist_m': 0.18,
                    'precise_goal_reached_yaw_tolerance_rad': 0.21,
                    'explore_goal_reached_dist_m': 0.35,
                    'explore_navigation_timeout_sec': 45.0,
                    'green_navigation_timeout_sec': 28.0,
                    'lane_biased_intermediate_goals': True,
                    'lane_goal_min_abs_y_m': 0.35,
                    'retarget_preempt_enabled': True,
                    'retarget_cancel_settle_sec': 0.05,
                  'duplicate_goal_replan_after_sec': 0.0,
                    'early_success_cancel_enabled': True,
                }],
                output='screen',
            ),
            Node(
                package='fire_robot_manipulation',
                executable='manipulation_node',
                name='manipulation_node',
                parameters=[{
                    'use_sim_time':     use_sim_time,
                    'sim_mode':         True,
                    'door_open_motion': 'push',
                    'press_handle_before_push': True,
                    'lever_press_distance_m': 0.07,
                    'sim_physical_door_opening': True,
                    'sim_door_physics_required': True,
                    'sim_open_away_from_centerline': True,
                            'sim_door_open_angle_rad': 2.09439510239,
                            'sim_door_command_steps': 1,
                            'sim_door_max_match_distance_m': 0.65,
                            'sim_door_axis_match_max_progress_m': 0.65,
                            'sim_door_axis_match_max_lateral_m': 0.50,
                            'sim_door_world_file': sim_world_path,
                    'sim_arm_motion_enabled': True,
                    'post_open_backoff_enabled': True,
                    'post_open_backoff_linear_x': -0.05,
                    'post_open_backoff_sec': 1.2,
                    'planning_group':   'piper_arm',
                    'gripper_group':    'piper_gripper',
                    'velocity_scaling': 0.5,
                }],
                remappings=[('/cmd_vel', '/cmd_vel_manual')],
                output='screen',
            ),
            TimerAction(
                period=32.0,
                actions=[
                    Node(
                        package='fire_robot_fsm',
                        executable='state_machine_node',
                        name='state_machine_node',
                        parameters=[{
                            'use_sim_time':        use_sim_time,
                            'nav_timeout_sec':     55.0,
                            'max_nav_retries':     2,
                            'explore_timeout_sec': 60.0,
                            'explore_linear_vel':   0.0,
                            'explore_angular_vel':  0.0,
                            'explore_obstacle_stop_m': 0.55,
                            'explore_obstacle_slow_m': 1.55,
                            'explore_avoid_angular_vel': 0.72,
                            'require_mission_axis': True,
                            'explore_side_clearance_m': 0.62,
                            'explore_scan_min_range_m': 0.95,
                            'explore_color_lane_bias': True,
                            'explore_forward_yaw':  0.0,
                            'explore_heading_kp':   0.9,
                            'explore_heading_soft_limit_deg': 50.0,
                            'explore_heading_hard_limit_deg': 115.0,
                            'explore_center_y':     0.0,
                            'explore_y_soft_limit_m': 0.95,
                            'explore_y_hard_limit_m': 1.30,
                            'explore_center_lookahead_m': 1.20,
                            'prefer_forward_doors': False,
                            'mission_start_x':      0.0,
                            'next_door_min_forward_m': 0.0,
                            'target_door_min_robot_forward_m': 0.0,
                            'blue_target_max_robot_ahead_m': 5.0,
                            'side_wall_door_progress_bias_m': 0.0,
                            'opened_door_merge_dist_m': 1.20,
                            'min_opened_doors_before_exit': min_opened_doors_before_exit,
                            'opened_physical_door_merge_dist_m': 1.10,
                            'abandoned_physical_door_merge_dist_m': 0.85,
                            'observed_physical_door_merge_dist_m': 1.65,
                            'observed_blue_min_observations': 2,
                            'observed_blue_target_min_confidence': 0.34,
                            'observed_blue_low_conf_target_min_observations': 4,
                            'observed_blue_stable_open_min_confidence': 0.52,
                            'observed_blue_fresh_evidence_max_progress_gap_m': 1.70,
                            'observed_blue_fresh_evidence_max_lateral_gap_m': 1.85,
                            'semantic_door_memory_enabled': True,
                            'semantic_door_merge_dist_m': 1.10,
                            'semantic_same_wall_merge_progress_m': 0.0,
                            'semantic_same_wall_merge_lateral_m': 0.0,
                            'semantic_blue_min_observations': 2,
                            'semantic_blue_min_ratio': 0.58,
                            'semantic_exit_tail_blue_min_observations': 4,
                            'semantic_exit_min_green_observations': 3,
                            'semantic_exit_min_green_ratio': 0.50,
                            'blue_target_immediate_min_confidence': 0.55,
                            'exit_interrupt_blue_min_confidence': 0.55,
                            'use_observed_blue_clusters_as_targets': True,
                            'detected_door_max_age_sec': 8.0,
                            'observed_blue_same_id_max_jump_m': 0.9,
                            'observed_blue_same_id_max_total_drift_m': 1.20,
                            'observed_blue_stale_abandon_sec': 120.0,
                            'observed_blue_stale_behind_m': 0.0,
                            'observed_blue_target_max_behind_m': -1.0,
                            'observed_blue_same_side_after_opened_min_progress_m': 0.0,
                            'blue_same_side_after_opened_min_progress_m': 0.0,
                            'blue_red_conflict_dist_m': 2.10,
                            'blue_red_conflict_strict_dist_m': 0.85,
                            'observed_blue_min_abs_wall_y_m': 0.95,
                            'observed_blue_max_abs_wall_y_m': 2.20,
                            'blue_handle_max_wall_overshoot_m': 0.30,
                            'blue_handle_validation_overshoot_m': 0.35,
                            'red_blue_conflict_memory_sec': 240.0,
                            'blue_red_conflict_after_exit_dist_m': 2.0,
                            'red_observation_min_confidence': 0.24,
                            'red_observation_min_count_for_blue_suppression': 1,
                            'confirmed_blue_bypass_red_conflict_count': 6,
                            'confirmed_blue_bypass_red_conflict_confidence': 0.60,
                            'opened_station_blue_suppression_progress_m': 0.0,
                            'opened_station_same_side_blue_suppression_progress_m': 1.20,
                            'opened_station_opposite_side_blue_suppression_progress_m': 0.0,
                            'opened_station_same_id_blue_suppression_progress_m': 0.0,
                            'abandoned_station_same_side_blue_suppression_progress_m': 2.0,
                            'abandoned_station_opposite_side_blue_suppression_progress_m': 0.0,
                            'max_door_approach_failures_before_abandon': 4,
                            'pre_exit_blue_approach_failures_before_abandon': 3,
                            'door_open_requires_fresh_blue': True,
                            'door_open_fresh_blue_max_age_sec': 10.0,
                            'door_open_fresh_blue_max_dist_m': 1.45,
                            'door_open_fresh_blue_min_confidence': 0.24,
                            'allow_mapped_memory_open_without_live': False,
                            'allow_fallback_exit_goal': False,
                            'allow_near_backtracking_blue_revisit': False,
                            'max_explore_past_last_opened_blue_m': 4.0,
                            'abandoned_blue_extends_explore_limit': False,
                            'blue_target_max_beyond_explore_limit_m': 2.20,
                            'observed_exit_explore_grace_m': 1.20,
                            'observed_exit_blue_candidate_margin_m': 0.45,
                            'allow_front_wall_exit_fallback': True,
                            'front_wall_exit_min_range_m': 0.45,
                            'front_wall_exit_max_range_m': 8.0,
                            'front_wall_exit_half_angle_deg': 70.0,
                            'front_wall_exit_min_coverage': 0.14,
                            'front_wall_exit_max_spread_m': 1.20,
                            'front_wall_exit_min_after_opened_blue_m': 1.75,
                            'axis_door_side_standoff_m': 1.10,
                            'axis_door_min_side_goal_lateral_m': 0.75,
                            'axis_door_max_handle_pose_progress_delta_m': 1.6,
                            'axis_door_use_detected_pose_lateral': False,
                            'locked_target_refine_enabled': True,
                            'locked_target_refine_min_shift_m': 0.18,
                            'locked_target_refine_max_progress_jump_m': 1.05,
                            'locked_target_refine_max_total_progress_drift_m': 1.05,
                            'locked_target_refine_min_period_sec': 1.5,
                            'locked_target_refine_same_detector_forward_max_progress_jump_m': 0.0,
                            'locked_target_refine_same_detector_forward_max_total_drift_m': 0.0,
                            'locked_target_close_station_max_forward_correction_m': 4.0,
                            'explore_observation_wait_sec': 2.0,
                            'explore_observation_wait_angular_vel': 0.0,
                            'nav_target_republish_sec': 0.0,
                            'explore_nav_failed_goal_memory_sec': 24.0,
                            'explore_nav_failed_goal_progress_window_m': 1.60,
                            'explore_waypoint_scan_sec': 2.4,
                            'explore_waypoint_scan_angular_vel': 0.60,
                            'explore_interrupt_min_confidence': 0.20,
                            'explore_interrupt_max_distance_m': 8.8,
                            'detected_exit_use_expected_region': False,
                            'detected_exit_merge_dist_m': 2.4,
                            'detected_exit_max_abs_lateral_m': 2.4,
                            'detected_exit_min_robot_forward_m': 0.0,
                            'no_blue_exit_min_robot_forward_m': 12.0,
                            'detected_exit_max_center_y_m': 0.85,
                            'detected_exit_min_after_opened_blue_m': 1.75,
                            'detected_exit_min_robot_past_opened_blue_m': 1.20,
                            'detected_exit_require_front_wall_confirmation': False,
                            'detected_exit_front_wall_tolerance_m': 1.80,
                            'blue_exit_suppression_margin_m': 1.1,
                            'pre_exit_blue_requires_fresh_detection': True,
                            'pre_exit_blue_suppression_margin_m': 3.0,
                            'pre_exit_blue_min_confidence': 0.18,
                            'pre_exit_blue_fresh_max_age_sec': 4.0,
                            'exit_pass_through_m': 0.90,
                            'exit_nav_standoff_m': 0.80,
                            'exit_visible_nav_max_dist_m': 4.0,
                            'exit_direct_cross_max_dist_m': 2.8,
                            'exit_cross_linear_vel': 0.40,
                            'exit_cross_heading_kp': 1.8,
                            'exit_cross_angular_vel_limit': 0.75,
                            'exit_cross_timeout_sec': 45.0,
                            'exit_cross_rotate_in_place_threshold_deg': 70.0,
                            'exit_interrupt_blue_before_exit_margin_m': 0.35,
                            'max_door_open_failures_before_abandon': 2,
                            'complete_when_past_exit_x': False,
                            'exit_complete_margin_m': 0.35,
                            'door_open_ready_max_dist_m': 0.20,
                            'door_open_straight_ready_max_dist_m': 0.26,
                            'door_open_ready_yaw_tolerance_deg': 5.0,
                            'door_open_ready_lateral_tolerance_m': 0.16,
                            'door_open_fine_control_max_dist_m': 1.10,
                            'door_open_fine_lateral_tolerance_m': 0.75,
                            'door_open_fine_position_kp': 0.75,
                            'door_open_fine_linear_vel': 0.075,
                            'door_open_fine_min_linear_vel': 0.020,
                            'door_open_fine_allow_reverse': True,
                            'door_open_fine_exception_reverse_linear_x': -0.08,
                            'door_open_fine_exception_reverse_sec': 0.8,
                            'door_open_fine_exception_reverse_max_sec': 5.0,
                            'door_open_fine_yaw_stuck_sec': 0.0,
                            'door_open_axis_min_side_lateral_m': 0.95,
                            'door_open_align_timeout_sec': 36.0,
                            'door_open_align_angular_vel': 0.25,
                            'door_open_align_kp': 0.75,
                            'post_open_reorient_enabled': False,
                            'post_open_reorient_yaw_tolerance_deg': 14.0,
                            'post_open_reorient_angular_vel': 0.50,
                            'post_open_reorient_timeout_sec': 6.0,
                            'post_open_clearance_advance_enabled': False,
                            'post_open_side_retreat_enabled': True,
                            'post_open_side_retreat_m': 1.10,
                            'post_open_side_retreat_target_abs_y_m': 0.48,
                            'post_open_side_retreat_linear_vel': 0.18,
                            'post_open_side_retreat_timeout_sec': 14.0,
                            'post_open_side_retreat_yaw_tolerance_deg': 42.0,
                            'post_open_clearance_advance_m': 0.45,
                            'post_open_clearance_linear_vel': 0.12,
                            'post_open_clearance_front_min_m': 0.85,
                            'post_open_clearance_front_angle_deg': 24.0,
                            'post_open_clearance_timeout_sec': 8.0,
                            'post_open_clearance_heading_kp': 0.8,
                            'post_open_clearance_angular_vel_limit': 0.30,
                            'door_open_settle_sec': 2.8,
                            'exit_retry_delay_sec': 3.0,
                            'final_scan_before_exit_sec': 2.0,
                            'final_scan_angular_vel': 0.45,
                            'pre_nav_scan_sec': 1.1,
                            'pre_nav_scan_angular_vel': 0.0,
                            'door_nav_stuck_watch_enabled': True,
                            'door_nav_stuck_timeout_sec': 32.0,
                            'door_nav_stuck_min_progress_m': 0.08,
                            'door_nav_stuck_min_goal_dist_m': 0.90,
                            'explore_nav_enabled': True,
                            'explore_nav_step_m': 1.20,
                            'explore_nav_min_step_m': 0.30,
                            'explore_nav_reduce_step_clearance_m': 1.70,
                            'explore_nav_goal_clearance_margin_m': 0.45,
                            'explore_nav_clearance_deferral_limit': 1,
                            'explore_nav_lane_y_m': 0.65,
                            'explore_nav_lane_width_m': 1.10,
                            'explore_nav_scan_lookahead_m': 3.20,
                            'explore_nav_color_bias_score': 0.55,
                            'explore_nav_keep_lane_bias_score': 0.18,
                            'explore_nav_lane_alternate_success_interval': 2,
                            'explore_nav_front_obstacle_width_m': 1.20,
                            'explore_nav_front_obstacle_bias_score': 3.4,
                            'explore_nav_blue_door_bias_score': 2.4,
                            'explore_nav_blue_lane_min_clearance_m': 0.75,
                            'explore_nav_blue_lane_bias_max_dist_m': 0.0,
                            'explore_nav_front_block_scan_clearance_m': 0.55,
                            'explore_nav_front_block_scan_after_opened_m': 2.0,
                            'explore_nav_recenter_abs_y_m': 0.80,
                            'explore_nav_max_lateral_step_m': 0.65,
                            'max_explore_nav_failures_before_exit': 3,
                            'explore_nav_timeout_sec': 32.0,
                            'explore_nav_stuck_wall_timeout_sec': 28.0,
                            'explore_nav_stuck_min_progress_m': 0.10,
                            'min_target_door_abs_y_m': 0.35,
                            'target_door_direct_nav_max_dist_m': 0.0,
                            'nav_start_max_abs_y_m': 0.95,
                            'nav_start_max_heading_error_deg': 80.0,
                            'nav_start_center_recovery_enabled': True,
                            'nav_start_center_recovery_target_abs_y_m': 0.75,
                            'nav_start_center_recovery_linear_vel': 0.13,
                            'nav_start_center_recovery_angular_vel': 0.40,
                            'nav_start_center_recovery_yaw_tolerance_deg': 32.0,
                            'nav_start_center_recovery_max_sec': 22.0,
                            'start_without_fire':   start_without_fire,
                        }],
                        remappings=[('/cmd_vel', '/cmd_vel_manual')],
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
        DeclareLaunchArgument('min_opened_doors_before_exit', default_value='0'),
        DeclareLaunchArgument('launch_moveit', default_value='false'),
        DeclareLaunchArgument('log_detection_candidates', default_value='false'),
        DeclareLaunchArgument(
            'enable_segformer',
            default_value='false',
            description='Enable camera semantic segmentation. Disable for CPU-only navigation stress tests.',
        ),
        DeclareLaunchArgument(
            'localization_mode',
            default_value='sim_odom',
            description='sim_odom: fixed map plus Gazebo odometry; mapping: SLAM; localization: saved map plus AMCL.',
        ),
        DeclareLaunchArgument('spawn_x', default_value='-3.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument(
            'localization_start_delay',
            default_value='8.0',
            description='Delay localization/SLAM startup until Gazebo has spawned the robot and TF begins.',
        ),
        DeclareLaunchArgument(
            'localization_recover_delay',
            default_value='18.0',
            description='Delay before retrying map_server/AMCL lifecycle activation if localization autostart stalls.',
        ),
        DeclareLaunchArgument(
            'nav2_start_delay',
            default_value='12.0',
            description='Delay Nav2 startup until localization and odom TF are available.',
        ),
        DeclareLaunchArgument(
            'nav2_recover_delay',
            default_value='22.0',
            description='Delay before retrying Nav2 lifecycle activation if autostart stalls.',
        ),
        DeclareLaunchArgument(
            'enable_lifecycle_recovery',
            default_value='false',
            description='Enable manual lifecycle recovery timers for debugging only.',
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
            'door_model_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_perception'), 'models', 'best.pt',
            ]),
            description='YOLO door detector weight path. Leave the file absent to use HSV fallback.',
        ),
        DeclareLaunchArgument(
            'handle_model_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_perception'), 'models', 'handle_best_v2.pt',
            ]),
            description='YOLO door-handle detector weight path. If absent, HSV handle fallback is used.',
        ),
        DeclareLaunchArgument(
            'publish_debug_image',
            default_value='false',
            description='Publish door_detection debug image topics for visual validation.',
        ),
        DeclareLaunchArgument(
            'log_handle_detections',
            default_value='false',
            description='Log accepted/rejected handle detector candidates during validation.',
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
                sim_odom_map_server,
                sim_odom_map_lifecycle,
                sim_odom_map_tf,
            ],
        ),
        localization_lifecycle_recovery,
        sim_odom_map_lifecycle_recovery,
        TimerAction(
            period=nav2_start_delay,
            actions=[
                nav2_launch,           # Nav2 → /navigate_to_pose
            ],
        ),
        nav2_lifecycle_recovery,
        moveit_launch,   # optional move_group (sim door opening uses manipulation_node sim_mode)
        rviz_node,       # RViz2 시각화 (use_rviz:=false 로 비활성화 가능)
        app_nodes,       # 인지/주행/조작/FSM 노드 (Nav2/lifecycle 준비 후 순차 기동)
    ])
