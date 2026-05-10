from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz     = LaunchConfiguration('use_rviz',     default='true')

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
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # ── 2. SLAM (slam_toolbox online async) ──────────────
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
            'params_file':  PathJoinSubstitution([
                FindPackageShare('fire_robot_navigation'), 'config', 'nav2_params.yaml'
            ]),
        }.items(),
    )

    # ── 4. MoveIt2 (시뮬레이션 모드 — manipulation_node는 sim_mode=True) ──
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fire_robot_manipulation'), 'launch', 'move_group.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    # ── 5. 애플리케이션 노드 (Gazebo 완전 기동 후 3초 지연) ─
    app_nodes = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='fire_robot_perception',
                executable='sensor_fusion_node',
                name='sensor_fusion_node',
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen',
            ),
            Node(
                package='fire_robot_perception',
                executable='door_detection_node',
                name='door_detection_node',
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen',
            ),
            Node(
                package='fire_robot_navigation',
                executable='navigation_node',
                name='navigation_node',
                parameters=[{'use_sim_time': use_sim_time}],
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
            Node(
                package='fire_robot_fsm',
                executable='state_machine_node',
                name='state_machine_node',
                parameters=[{
                    'use_sim_time':        use_sim_time,
                    'exit_x':              16.0,   # corridor.world 비상구 X
                    'exit_y':               0.0,
                    'exit_yaw':             0.0,
                    'explore_timeout_sec': 30.0,
                }],
                output='screen',
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

        gazebo_launch,   # Gazebo + robot_state_publisher + spawn + bridge
        slam_launch,     # SLAM → /map 발행
        nav2_launch,     # Nav2 → /navigate_to_pose
        moveit_launch,   # move_group (sim 환경에서도 플래닝 가능)
        rviz_node,       # RViz2 시각화 (use_rviz:=false 로 비활성화 가능)
        app_nodes,       # 인지/주행/조작/FSM 노드 (3초 후 기동)
    ])
