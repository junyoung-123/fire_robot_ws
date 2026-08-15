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
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('fire_robot_bringup'), 'launch', 'gazebo.launch.py'])),
        launch_arguments={'world': 'manipulation_demo.world',
                          'headless': headless,
                          # Ogre2 crashes in some WSLg drivers while copying
                          # render textures. Ogre1 keeps the GUI usable there.
                          'render_engine_gui': 'ogre',
                          'use_sim_time': 'true'}.items())

    command_topics = [
        f'/sim/joint{i}_position_cmd@std_msgs/msg/Float64]gz.msgs.Double'
        for i in range(1, 7)]
    command_topics += [
        '/sim/gripper_left_position_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/sim/gripper_right_position_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/sim/door_hinge_position_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/door_joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
    ]
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  name='manipulation_command_bridge', arguments=command_topics,
                  output='screen')
    manipulation = TimerAction(period=4.0, actions=[Node(
        package='fire_robot_manipulation', executable='manipulation_node',
        name='manipulation_node', output='screen', parameters=[{
            'use_sim_time': True, 'sim_mode': True,
            'sim_control_enabled': True, 'sim_step_sec': 1.5,
            'sim_return_home': False, 'sim_verify_door': True}])])
    rviz = Node(
        package='rviz2', executable='rviz2', name='manipulation_rviz',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('fire_robot_bringup'), 'rviz',
            'manipulation_demo.rviz'])],
        parameters=[{'use_sim_time': True}], condition=IfCondition(use_rviz),
        output='screen')

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        gazebo, bridge, manipulation, rviz])
