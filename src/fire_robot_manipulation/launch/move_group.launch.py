from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def load_yaml(package_name, *paths):
    package_path = get_package_share_directory(package_name)
    file_path = os.path.join(package_path, *paths)
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_text(package_name, *paths):
    package_path = get_package_share_directory(package_name)
    file_path = os.path.join(package_path, *paths)
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    launch_manipulation = LaunchConfiguration('launch_manipulation', default='true')

    description_share = get_package_share_directory('fire_robot_description')
    urdf_path = os.path.join(description_share, 'urdf', 'fire_robot.urdf.xacro')

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', urdf_path]),
        value_type=str,
    )

    robot_description_semantic = load_text(
        'fire_robot_manipulation', 'config', 'fire_robot.srdf')
    kinematics_yaml = load_yaml(
        'fire_robot_manipulation', 'config', 'kinematics.yaml')
    joint_limits_yaml = load_yaml(
        'fire_robot_manipulation', 'config', 'joint_limits.yaml')
    ompl_planning_yaml = load_yaml(
        'fire_robot_manipulation', 'config', 'ompl_planning.yaml')
    moveit_controllers_yaml = load_yaml(
        'fire_robot_manipulation', 'config', 'moveit_controllers.yaml')

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
    move_group_params['ompl'].update(ompl_planning_yaml)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('launch_manipulation', default_value='true'),

        # MoveIt2 move_group 서버
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            name='move_group',
            parameters=[
                move_group_params,
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
            condition=IfCondition(launch_manipulation),
            output='screen',
        ),
    ])
