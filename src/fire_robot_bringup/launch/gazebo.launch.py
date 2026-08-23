from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (Command, FindExecutable,
                                  LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file = LaunchConfiguration('world', default='corridor.world')
    headless = LaunchConfiguration('headless', default='false')
    render_engine_gui = LaunchConfiguration('render_engine_gui', default='ogre2')
    enable_depth_camera = LaunchConfiguration('enable_depth_camera', default='false')
    gazebo_env = {
        'LIBGL_ALWAYS_SOFTWARE': '1',
        'MESA_GL_VERSION_OVERRIDE': '3.3',
    }

    world_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_bringup'), 'worlds', world_file,
    ])

    urdf_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_description'), 'urdf',
        'fire_robot.urdf.xacro',
    ])

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', urdf_path]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world', default_value='corridor.world'),
        DeclareLaunchArgument(
            'headless',
            default_value='false',
            description='Run Gazebo server-only mode without GUI when true.',
        ),
        DeclareLaunchArgument(
            'render_engine_gui',
            default_value='ogre2',
            description='Gazebo GUI rendering engine, for example ogre2 or ogre.',
        ),
        DeclareLaunchArgument(
            'enable_depth_camera',
            default_value='false',
            description='Bridge the simulated depth camera topics when enabled.',
        ),

        ExecuteProcess(
            cmd=[
                'ign', 'gazebo', '-r',
                '--render-engine-gui', render_engine_gui,
                world_path,
            ],
            output='screen',
            additional_env=gazebo_env,
            condition=UnlessCondition(headless),
        ),
        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', '-s', world_path],
            output='screen',
            additional_env=gazebo_env,
            condition=IfCondition(headless),
        ),

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

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='sim_lidar_sensor_tf',
            arguments=[
                '0.2', '0.0', '0.07',
                '0.0', '0.0', '0.0',
                'base_link',
                'fire_robot/base_footprint/lidar_sensor',
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),

        Node(
            package='fire_robot_navigation',
            executable='cmd_vel_safety_node',
            name='cmd_vel_safety_node',
            parameters=[{
                'use_sim_time': use_sim_time,
                'input_topic': '/cmd_vel',
                'output_topic': '/cmd_vel_safe',
                'allow_reverse': False,
                'max_blocked_reverse_linear_x': 0.05,
                'max_linear_x': 0.25,
                'max_angular_z': 1.0,
            }],
            output='screen',
        ),

        Node(
            package='ros_gz_sim',
            executable='create',
            name='spawn_fire_robot',
            arguments=[
                '-name', 'fire_robot',
                '-topic', '/robot_description',
                '-x', '-3.0',
                '-y', '0.0',
                '-z', '0.07',
                '-Y', '0.0',
            ],
            output='screen',
        ),

        # Keep clock and motion-critical bridges separate from camera streams.
        # A busy image bridge can otherwise stall ROS /clock in WSL headless runs.
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_clock_bridge',
            arguments=[
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_motion_bridge',
            arguments=[
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            ],
            remappings=[
                ('/cmd_vel', '/cmd_vel_safe'),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_rgb_camera_bridge',
            arguments=[
                '/camera_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera_raw/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera_front_left_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera_front_left_raw/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
                '/camera_front_right_raw@sensor_msgs/msg/Image[gz.msgs.Image',
                '/camera_front_right_raw/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            remappings=[
                ('/camera_raw', '/camera/color/image_raw'),
                ('/camera_raw/camera_info', '/camera/color/camera_info'),
                ('/camera_front_left_raw', '/camera/front_left/image_raw'),
                ('/camera_front_left_raw/camera_info',
                 '/camera/front_left/camera_info'),
                ('/camera_front_right_raw', '/camera/front_right/image_raw'),
                ('/camera_front_right_raw/camera_info',
                 '/camera/front_right/camera_info'),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_depth_camera_bridge',
            arguments=[
                '/depth_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/depth_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
                '/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
                '/depth_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            ],
            remappings=[
                ('/depth_camera/image', '/camera/depth/color/image_raw'),
                ('/depth_camera/depth_image', '/camera/depth/image_rect_raw'),
                ('/depth_camera/points', '/camera/depth/points'),
                ('/depth_camera/camera_info', '/camera/depth/camera_info'),
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(enable_depth_camera),
            output='screen',
        ),
    ])
