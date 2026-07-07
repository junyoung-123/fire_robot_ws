from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                             IncludeLaunchDescription, RegisterEventHandler,
                             TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                   PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """실제 하드웨어 런치파일.
    실행 전 확인:
      - 2D LiDAR driver: /scan(sensor_msgs/LaserScan), frame_id=lidar_link 발행
      - 첫 지도 작성은 enable_slam:=true, 이후 주행은 enable_localization:=true + map:=... 사용
      - J100/Jackal 베이스는 Clearpath stack(roas2_bringup)이 platform/odom, cmd_vel mux를 담당
      - RealSense D435: USB3.0 연결 확인 (realsense2_camera 자동 인식)
      - PIPER CAN: can-auto-setup.sh 사용 시 left_arm, 수동 설정 시 can0 사용
    """

    # ---------- 인자 ----------
    use_sim_time     = LaunchConfiguration('use_sim_time',     default='false')
    lidar_serial_port = LaunchConfiguration('lidar_serial_port', default='/dev/ttyUSB0')
    piper_can_port   = LaunchConfiguration('piper_can_port',   default='left_arm')
    enable_camera    = LaunchConfiguration('enable_camera',    default='true')
    enable_lidar    = LaunchConfiguration('enable_lidar',    default='true')
    enable_piper     = LaunchConfiguration('enable_piper',     default='true')
    enable_base      = LaunchConfiguration('enable_base',      default='false')
    enable_clearpath_platform = LaunchConfiguration(
        'enable_clearpath_platform', default='false')
    enable_clearpath_ekf = LaunchConfiguration(
        'enable_clearpath_ekf', default='true')
    enable_moveit    = LaunchConfiguration('enable_moveit',    default='false')
    enable_slam      = LaunchConfiguration('enable_slam',      default='true')
    enable_nav2      = LaunchConfiguration('enable_nav2',      default='true')
    enable_localization = LaunchConfiguration('enable_localization', default='false')
    enable_app_nodes = LaunchConfiguration('enable_app_nodes', default='true')
    enable_fire_robot_description = LaunchConfiguration(
        'enable_fire_robot_description', default='true')
    lidar_frame = LaunchConfiguration('lidar_frame', default='sick_tim_57x')
    map_file = LaunchConfiguration('map', default=PathJoinSubstitution([
        FindPackageShare('fire_robot_navigation'), 'maps', 'corridor.yaml',
    ]))
    nav2_params = PathJoinSubstitution([
        FindPackageShare('fire_robot_navigation'), 'config', 'nav2_params.yaml',
    ])
    door_model_path = LaunchConfiguration('door_model_path', default=PathJoinSubstitution([
        FindPackageShare('fire_robot_perception'), 'models', 'best.pt',
    ]))
    odom_topic = LaunchConfiguration('odom_topic', default='/odom')
    clearpath_setup_path = LaunchConfiguration(
        'clearpath_setup_path',
        default='/home/roas/jackal_ws/src/roas2_bringup/',
    )
    slam_params_file = LaunchConfiguration(
        'slam_params_file',
        default=PathJoinSubstitution([
            FindPackageShare('fire_robot_navigation'), 'config', 'slam_toolbox_params.yaml',
        ]),
    )

    urdf_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_description'), 'urdf', 'fire_robot.urdf.xacro',
    ])
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', urdf_path]),
        value_type=str,
    )

    # ---------- Clearpath J100/Jackal platform ----------
    # 실제 모바일 베이스는 roas2_bringup/Clearpath stack이 ros2_control,
    # twist_mux, micro-ROS, platform/odom을 담당한다. 프로젝트 단독
    # 검증 환경에서는 패키지가 없을 수 있어 기본값은 비활성화한다.
    clearpath_jackal_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('roas2_bringup'), 'launch', 'jackal.launch.py'
            ])
        ]),
        condition=IfCondition(enable_clearpath_platform),
    )
    clearpath_ekf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('roas2_bringup'),
                'launch', 'include', 'localization.launch.py'
            ])
        ]),
        launch_arguments={
            'setup_path': clearpath_setup_path,
            'use_sim_time': use_sim_time,
            'enable_ekf': enable_clearpath_ekf,
        }.items(),
        condition=IfCondition(enable_clearpath_platform),
    )

    # ---------- 하드웨어 드라이버 ----------

    # 1) Intel RealSense D435 (realsense2_camera: RGB + Depth 동시 지원)
    #    설치: sudo apt install -y ros-humble-realsense2-camera
    camera_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        parameters=[{
            'depth_module.profile':  '640x480x30',
            'rgb_camera.profile':    '640x480x30',
            'enable_depth':          True,
            'enable_color':          True,
            'pointcloud.enable':     True,
            'align_depth.enable':    True,
            'base_frame_id':         'camera_link',
            'use_sim_time':          use_sim_time,
        }],
        remappings=[
            ('color/image_raw',         '/camera/color/image_raw'),
            ('color/camera_info',       '/camera/color/camera_info'),
            ('depth/image_rect_raw',    '/camera/depth/image_rect_raw'),
            ('depth/camera_info',       '/camera/depth/camera_info'),
            ('depth/color/points',      '/camera/depth/points'),
        ],
        condition=IfCondition(enable_camera),
        output='screen',
    )

    # 2) 2D LiDAR — Nav2, SLAM, AMCL의 주 거리 센서.
    #    기본 frame_id는 lidar_link이며 /scan(sensor_msgs/LaserScan)을 발행한다.
    lidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='lidar_node',
        parameters=[{
            'product_name':  'LDLiDAR_LD19',
            'topic_name':    'scan',
            'port_name':     lidar_serial_port,
            'port_baudrate': 230400,
            'frame_id':      'lidar_link',
            'use_sim_time':  use_sim_time,
        }],
        condition=IfCondition(enable_lidar),
        output='screen',
    )

    # 3) PIPER 6DoF 매니퓰레이터 드라이버 (AgileX piper_ros, humble branch)
    #    piper_ros 패키지가 설치되어 있어야 함:
    #    https://github.com/agilexrobotics/piper_ros
    piper_driver_node = Node(
        package='piper',
        executable='piper_single_ctrl',
        name='piper_driver',
        parameters=[{
            'can_port':      piper_can_port,
            'auto_enable':   True,
            'gripper_exist': True,
            'gripper_val_mutiple': 1,
            'use_sim_time':  use_sim_time,
        }],
        condition=IfCondition(enable_piper),
        output='screen',
    )

    # 4) 개발용 모바일 베이스 ros2_control (diff_drive_controller)
    #    실제 J100/Jackal에서는 Clearpath stack을 사용하므로 기본 비활성화.
    robot_controllers_yaml = PathJoinSubstitution([
        FindPackageShare('fire_robot_description'), 'config', 'robot_controllers.yaml',
    ])
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        parameters=[
            {'robot_description': robot_description},
            robot_controllers_yaml,
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(enable_base),
        output='screen',
    )
    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        condition=IfCondition(enable_base),
        output='screen',
    )
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        condition=IfCondition(enable_base),
        output='screen',
    )
    # controller_manager가 준비된 뒤에만 spawner 실행 (race condition 방지)
    spawn_controllers_on_ready = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager_node,
            on_start=[joint_state_broadcaster_spawner, diff_drive_spawner],
        )
    )

    # ---------- TF / 로봇 상태 ----------
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time':      use_sim_time,
        }],
        condition=IfCondition(PythonExpression([
            "'", enable_fire_robot_description, "' == 'true' and '",
            enable_clearpath_platform, "' != 'true'"
        ])),
        output='screen',
    )

    # ---------- MoveIt2 ----------
    # 실제 PIPER SDK 노드는 FollowJointTrajectory 액션 서버를 제공하지 않는다.
    # ros2_control 브리지를 검증하기 전까지 실제 런치에서는 기본 비활성화한다.
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fire_robot_manipulation'), 'launch', 'move_group.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
        condition=IfCondition(enable_moveit),
    )

    # ---------- SLAM (online async) ----------
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('slam_toolbox'), 'launch', 'online_async_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time':  use_sim_time,
            'slam_params_file': slam_params_file,
        }.items(),
        condition=IfCondition(enable_slam),
    )

    # ---------- Localization (저장 지도 + AMCL) ----------
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
        condition=IfCondition(enable_localization),
    )

    # ---------- Nav2 ----------
    nav2_launch = GroupAction([
        SetRemap(src='/odom', dst=odom_topic),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('nav2_bringup'),
                    'launch', 'navigation_launch.py'
                ])
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'params_file':  nav2_params,
            }.items(),
            condition=IfCondition(enable_nav2),
        ),
    ])

    # ---------- 애플리케이션 노드 ----------
    # 드라이버가 완전히 준비된 뒤에 올라오도록 2초 지연
    app_nodes = TimerAction(
        period=2.0,
        condition=IfCondition(enable_app_nodes),
        actions=[
            Node(package='fire_robot_perception', executable='sensor_fusion_node',
                 parameters=[{'use_sim_time': use_sim_time, 'lidar_frame': lidar_frame}], output='screen'),
            Node(package='fire_robot_perception', executable='door_detection_node',
                 parameters=[{
                     'use_sim_time': use_sim_time,
                     'model_path': door_model_path,
                 }], output='screen'),
            Node(package='fire_robot_navigation',  executable='navigation_node',
                 parameters=[{'use_sim_time': use_sim_time}], output='screen'),
            Node(package='fire_robot_manipulation', executable='manipulation_node',
                 parameters=[{
                     'use_sim_time': use_sim_time,
                     'sim_mode': False,
                     'allow_sim_fallback': False,
                     'control_backend': 'piper_sdk',
                     'manipulation_frame': 'left_base_link',
                     'piper_pos_cmd_topic': '/pos_cmd',
                     'piper_enable_topic': '/enable_flag',
                     'piper_command_settle_sec': 1.2,
                 }], output='screen'),
            Node(package='fire_robot_fsm',          executable='state_machine_node',
                 parameters=[{'use_sim_time': use_sim_time}], output='screen'),
        ],
    )

    return LaunchDescription([
        # 인자 선언
        DeclareLaunchArgument('use_sim_time',       default_value='false'),
        DeclareLaunchArgument('lidar_serial_port',  default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('piper_can_port',     default_value='left_arm'),
        DeclareLaunchArgument('enable_camera',      default_value='true'),
        DeclareLaunchArgument('enable_lidar',       default_value='true'),
        DeclareLaunchArgument('enable_piper',       default_value='true'),
        DeclareLaunchArgument('enable_base',        default_value='false'),
        DeclareLaunchArgument('enable_clearpath_platform', default_value='false',
            description='Launch roas2_bringup Jackal/J100 platform stack when available.'),
        DeclareLaunchArgument('enable_clearpath_ekf', default_value='true',
            description='Enable Clearpath robot_localization EKF for odom->base_link TF.'),
        DeclareLaunchArgument('enable_moveit',      default_value='false'),
        DeclareLaunchArgument('enable_slam',        default_value='true'),
        DeclareLaunchArgument('enable_nav2',        default_value='true'),
        DeclareLaunchArgument('enable_localization', default_value='false'),
        DeclareLaunchArgument('enable_app_nodes',   default_value='true'),
        DeclareLaunchArgument('enable_fire_robot_description', default_value='true',
            description='Publish the simplified fire_robot_description TF tree. Disable if Clearpath roas2 description is already active.'),
        DeclareLaunchArgument('lidar_frame', default_value='sick_tim_57x',
            description='LaserScan frame on the real Clearpath/ROAS2 robot.'),
        DeclareLaunchArgument('odom_topic', default_value='/odom',
            description='Odometry topic used by Nav2; use /platform/odom/filtered on Clearpath J100.'),
        DeclareLaunchArgument(
            'door_model_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_perception'), 'models', 'best.pt',
            ]),
            description='YOLO door detector weight path. Leave the file absent to use HSV fallback.',
        ),
        DeclareLaunchArgument('clearpath_setup_path',
            default_value='/home/roas/jackal_ws/src/roas2_bringup/',
            description='Path to roas2_bringup setup directory on the J100 onboard PC.'),
        DeclareLaunchArgument(
            'map',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_navigation'), 'maps', 'corridor.yaml',
            ]),
            description='Map yaml used when enable_localization:=true.',
        ),
        DeclareLaunchArgument('slam_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_navigation'), 'config', 'slam_toolbox_params.yaml',
            ])),

        # 하드웨어
        clearpath_jackal_launch,
        clearpath_ekf_launch,
        robot_state_publisher_node,
        camera_node,
        lidar_node,
        piper_driver_node,
        controller_manager_node,
        spawn_controllers_on_ready,  # controller_manager 기동 후 spawner 실행

        # 미들웨어
        moveit_launch,
        slam_launch,
        localization_launch,
        nav2_launch,

        # 앱 노드 (지연 기동)
        app_nodes,
    ])
