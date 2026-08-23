from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')
    door_open_motion = LaunchConfiguration('door_open_motion')
    contact_min_angle = LaunchConfiguration('contact_min_angle_rad')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('fire_robot_bringup'), 'launch', 'gazebo.launch.py'])),
        launch_arguments={
            'world': 'physical_contact_door_test.world',
            'headless': headless,
            'render_engine_gui': 'ogre',
            'use_sim_time': 'true',
        }.items())

    command_topics = [
        f'/fire_robot/sim_arm/joint{i}_cmd@std_msgs/msg/Float64]gz.msgs.Double'
        for i in range(1, 7)]
    command_topics += [
        '/fire_robot/sim_arm/gripper_left_joint_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/fire_robot/sim_arm/gripper_right_joint_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/door_joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        '/proof/overhead/image@sensor_msgs/msg/Image[gz.msgs.Image',
    ]
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='physical_contact_command_bridge',
        arguments=command_topics,
        output='screen')

    manipulation = TimerAction(period=4.0, actions=[Node(
        package='fire_robot_manipulation',
        executable='manipulation_node',
        name='manipulation_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'sim_mode': True,
            'sim_control_enabled': True,
            'sim_command_transport': 'ros',
            'sim_step_sec': 1.6,
            'sim_return_home': False,
            'door_open_motion': door_open_motion,
            'press_handle_before_push': True,
            'lever_press_distance_m': 0.07,
            'sim_physical_door_opening': True,
            'sim_door_physics_required': True,
            'sim_door_contact_only': True,
            'sim_door_contact_min_angle_rad': contact_min_angle,
            'sim_verify_door': True,
            'sim_feedback_timeout_sec': 22.0,
            'sim_push_follow_enabled': True,
            'sim_push_follow_linear_x': 0.13,
            'sim_push_follow_angular_z': -0.13,
            'sim_push_follow_stage_sec': 1.7,
            'post_open_backoff_enabled': True,
            'post_open_backoff_linear_x': -0.05,
            'post_open_backoff_sec': 1.2,
        }])])

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='physical_contact_rviz',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('fire_robot_bringup'), 'rviz',
            'manipulation_demo.rviz'])],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(use_rviz),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('door_open_motion', default_value='push'),
        DeclareLaunchArgument('contact_min_angle_rad', default_value='0.75'),
        gazebo,
        bridge,
        manipulation,
        rviz,
    ])
