from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (Command, FindExecutable,
                                   LaunchConfiguration, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    urdf_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_description'), 'urdf',
        'fire_robot.urdf.xacro',
    ])
    srdf_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_manipulation'), 'config',
        'fire_robot.srdf',
    ])

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', urdf_path]),
        value_type=str,
    )
    robot_description_semantic = ParameterValue(
        Command(['cat ', srdf_path]),
        value_type=str,
    )

    kinematics_yaml       = PathJoinSubstitution([
        FindPackageShare('fire_robot_manipulation'), 'config', 'kinematics.yaml'])
    joint_limits_yaml     = PathJoinSubstitution([
        FindPackageShare('fire_robot_manipulation'), 'config', 'joint_limits.yaml'])
    ompl_planning_yaml    = PathJoinSubstitution([
        FindPackageShare('fire_robot_manipulation'), 'config', 'ompl_planning.yaml'])
    moveit_controllers_yaml = PathJoinSubstitution([
        FindPackageShare('fire_robot_manipulation'), 'config', 'moveit_controllers.yaml'])

    move_group_params = {
        'robot_description':          robot_description,
        'robot_description_semantic': robot_description_semantic,
        'robot_description_kinematics': kinematics_yaml,
        'robot_description_planning':   joint_limits_yaml,
        'use_sim_time': use_sim_time,
        'planning_pipelines': ['ompl'],
        'ompl': {
            'planning_plugin':
                'ompl_interface/OMPLPlanner',
            'request_adapters':
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        },
    }

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # MoveIt2 move_group 서버
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            name='move_group',
            parameters=[
                move_group_params,
                ompl_planning_yaml,
                moveit_controllers_yaml,
            ],
            output='screen',
        ),

        # manipulation_node (문 개방 서비스 서버)
        Node(
            package='fire_robot_manipulation',
            executable='manipulation_node',
            name='manipulation_node',
            parameters=[{
                'use_sim_time': use_sim_time,
                'sim_mode': False,             # 실제 MoveIt2 사용
                'planning_group': 'piper_arm',
                'gripper_group':  'piper_gripper',
                'velocity_scaling': 0.3,
            }],
            output='screen',
        ),
    ])
