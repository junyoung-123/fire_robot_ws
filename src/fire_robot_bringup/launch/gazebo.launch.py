from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import (Command, FindExecutable,
                                   LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file   = LaunchConfiguration('world',        default='corridor.world')

    world_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_bringup'), 'worlds', world_file,
    ])

    urdf_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_description'), 'urdf',
        'fire_robot.urdf.xacro',
    ])

    # xacro → URDF 문자열 변환
    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', urdf_path]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world',        default_value='corridor.world'),

        # ── 1. Gazebo Ignition 실행 ───────────────────────
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_path],
            output='screen',
        ),

        # ── 2. robot_state_publisher (URDF 내용 전달) ────
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_description,
            }],
            output='screen',
        ),

        # ── 3. 로봇 스폰 (Ignition 월드에 URDF 삽입) ────
        #   /robot_description 토픽을 읽어 Ignition 월드에 모델 생성
        #   복도 중앙(x=0)에서 시작, z=0.07(wheel_radius)
        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_fire_robot',
            arguments=[
                '-name',  'fire_robot',
                '-topic', '/robot_description',
                '-x', '-3.0',
                '-y', '0.0',
                '-z', '0.07',
                '-Y', '0.0',
            ],
            output='screen',
        ),

        # ── 4. ros_gz_bridge (Ignition ↔ ROS2) ──────────
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge',
            arguments=[
                # 시뮬레이션 시간
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
                # Radar (Ignition /scan → ROS2 /scan, LaserScan 형식 그대로)
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                # RGB 카메라 이미지
                '/camera_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera_raw/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                # Depth 카메라 (rgbd_camera → /camera/depth/*)
                '/depth_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/depth_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                # 구동 명령 (ROS2 → Ignition)
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                # 오도메트리
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                # TF (diff drive 플러그인이 odom→base_footprint 발행)
                '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                # 조인트 상태
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            remappings=[
                ('/camera_raw',              '/camera/color/image_raw'),
                ('/camera_raw/camera_info',  '/camera/color/camera_info'),
                ('/depth_camera/image',      '/camera/depth/color/image_raw'),
                ('/depth_camera/depth_image','/camera/depth/image_rect_raw'),
                ('/depth_camera/points',     '/camera/depth/points'),
                ('/depth_camera/camera_info','/camera/depth/camera_info'),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
    ])
