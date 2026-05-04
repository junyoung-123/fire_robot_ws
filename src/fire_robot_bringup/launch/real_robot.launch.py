from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                   PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """실제 하드웨어 런치파일.
    실행 전 확인:
      - Radar USB: /dev/ttyUSB0 (권한: sudo chmod 777 /dev/ttyUSB0)
      - RealSense D435: USB3.0 연결 확인 (realsense2_camera 자동 인식)
      - PIPER CAN: sudo ip link set can0 type can bitrate 1000000 && sudo ip link set up can0
    """

    # ---------- 인자 ----------
    use_sim_time     = LaunchConfiguration('use_sim_time',     default='false')
    radar_serial_port = LaunchConfiguration('radar_serial_port', default='/dev/ttyUSB0')
    piper_can_port   = LaunchConfiguration('piper_can_port',   default='can0')
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
        output='screen',
    )

    # 2) Radar — LD19/LD06 2D Radar (ldlidar_stl_ros2)
    #    실제 레이더 하드웨어에 맞는 패키지로 교체 필요.
    #    LD19 설치: https://github.com/LihanChen2004/ldlidar_stl_ros2
    #    sudo apt install -y ros-humble-ldlidar-stl-ros2  (또는 소스 빌드)
    radar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='radar_node',
        parameters=[{
            'product_name':  'LDLiDAR_LD19',
            'topic_name':    'scan',
            'port_name':     radar_serial_port,
            'port_baudrate': 230400,
            'frame_id':      'radar_link',
            'use_sim_time':  use_sim_time,
        }],
        output='screen',
    )

    # 3) PIPER 6DoF 매니퓰레이터 드라이버 (AgileX piper_sdk)
    #    piper_ros2 패키지가 설치되어 있어야 함:
    #    https://github.com/agilexrobotics/piper_ros2
    piper_driver_node = Node(
        package='piper_sdk',
        executable='piper_ctrl_single_node',
        name='piper_driver',
        parameters=[{
            'can_port':     piper_can_port,
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # 4) 모바일 베이스 ros2_control (diff_drive_controller)
    #    robot_controllers.yaml 에 DiffDriveController 설정 필요
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
        output='screen',
    )
    diff_drive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
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
        output='screen',
    )

    # ---------- MoveIt2 ----------
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('fire_robot_manipulation'), 'launch', 'move_group.launch.py'
            ])
        ]),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
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
    )

    # ---------- Nav2 ----------
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

    # ---------- 애플리케이션 노드 ----------
    # 드라이버가 완전히 준비된 뒤에 올라오도록 2초 지연
    app_nodes = TimerAction(
        period=2.0,
        actions=[
            Node(package='fire_robot_perception', executable='sensor_fusion_node',
                 parameters=[{'use_sim_time': use_sim_time}], output='screen'),
            Node(package='fire_robot_perception', executable='door_detection_node',
                 parameters=[{'use_sim_time': use_sim_time}], output='screen'),
            Node(package='fire_robot_navigation',  executable='navigation_node',
                 parameters=[{'use_sim_time': use_sim_time}], output='screen'),
            Node(package='fire_robot_fsm',          executable='state_machine_node',
                 parameters=[{'use_sim_time': use_sim_time}], output='screen'),
        ],
    )

    return LaunchDescription([
        # 인자 선언
        DeclareLaunchArgument('use_sim_time',       default_value='false'),
        DeclareLaunchArgument('radar_serial_port',  default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('piper_can_port',     default_value='can0'),
        DeclareLaunchArgument('slam_params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('fire_robot_navigation'), 'config', 'slam_toolbox_params.yaml',
            ])),

        # 하드웨어
        robot_state_publisher_node,
        camera_node,
        radar_node,
        piper_driver_node,
        controller_manager_node,
        spawn_controllers_on_ready,  # controller_manager 기동 후 spawner 실행

        # 미들웨어
        moveit_launch,
        slam_launch,
        nav2_launch,

        # 앱 노드 (지연 기동)
        app_nodes,
    ])
