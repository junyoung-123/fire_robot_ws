"""One live-SLAM mission with sensor-driven PIPER contact opening."""
import os
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('fire_robot_bringup')
    arguments = [f'/fire_robot/sim_arm/joint{i}_cmd@std_msgs/msg/Float64]gz.msgs.Double'
                 for i in range(1, 7)]
    arguments += [f'/fire_robot/sim_arm/{name}_cmd@std_msgs/msg/Float64]gz.msgs.Double'
                  for name in ('gripper_left_joint', 'gripper_right_joint')]
    arguments += [f'/proof/{name}/image@sensor_msgs/msg/Image[gz.msgs.Image'
                  for name in ('perspective', 'overhead', 'handle')]
    for index in range(1, 4):
        arguments.append(f'/evaluation/door_blue{index}/joints@sensor_msgs/msg/JointState[gz.msgs.Model')
        arguments += [f'/proof/audit/door_blue{index}/{part}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'
                      for part in ('lever', 'latch', 'panel', 'jamb', 'strike')]
    parameters = {
        'use_sim_time': True, 'sim_mode': True, 'sim_control_enabled': True,
        'sim_command_transport': 'ros', 'sim_step_sec': 2.2,
        'sim_arm_target_timeout_sec': 12.0, 'sim_contact_joint_error_tolerance_rad': .11,
        'sim_arm_clearance_retract_m': .18, 'sim_pre_stow_backoff_sec': 0.0,
        'sim_return_home': True, 'piper_mount_y_m': .15,
        'contact_reference_frame': LaunchConfiguration('contact_reference_frame'),
        'contact_motion_policy': LaunchConfiguration('contact_motion_policy'),
        'handle_surface_inset_m': 0.0, 'door_open_motion': 'push',
        'press_handle_before_push': True, 'lever_press_distance_m': .11,
        'sim_gripper_lever_width_m': .045, 'sim_grasp_search_step_m': .02,
        'sim_physical_door_opening': True, 'sim_door_physics_required': True,
        'sim_door_contact_only': True, 'sim_door_contact_min_angle_rad': 2.05,
        'sim_verify_door': True,
        'sim_door_feedback_joint_name': 'camera_door_rotation',
        'sim_door_feedback_topic': '/observed_door/rotation',
        'sim_lever_feedback_joint_name': 'encoder_lever_rotation',
        'sim_feedback_timeout_sec': 45.0, 'sim_lever_press_min_angle_rad': .08,
        'sim_lever_press_timeout_sec': 8.0,
        'sim_latch_clear_min_angle_rad': .25, 'sim_latch_clear_timeout_sec': 12.0,
        'sim_gripper_release_timeout_sec': 3.0, 'sim_gripper_release_half_width_m': .034,
        'sim_handle_refine_timeout_sec': 6.0, 'sim_handle_refine_min_samples': 3,
        'sim_handle_refine_max_shift_m': .04, 'sim_push_follow_enabled': True,
        'sim_push_follow_cmd_vel_topic': '/cmd_vel_manual',
        'sim_push_follow_linear_x': .08, 'sim_push_follow_angular_z': -.10,
        'sim_push_follow_radius_m': .60, 'sim_push_follow_stage_sec': 1.4,
        # The mission FSM owns the observed post-open retreat. Do not add a
        # second, time-only reverse command inside the contact service.
        'post_open_backoff_enabled': False, 'post_open_backoff_linear_x': -.10,
        'post_open_backoff_sec': 3.0, 'sim_release_handle_before_base_push': False,
        'hold_handle_during_base_open': True, 'require_detected_handle': True,
        'require_yolo_handle': True, 'feedback_contact_enabled': True,
        'feedback_encoder_contact_follow': True, 'feedback_clearance_min_m': .05,
        'feedback_clearance_max_m': .15, 'feedback_press_max_travel_m': .11,
        'feedback_press_sign': -1.0,
        'feedback_opening_speed_mps': .03, 'feedback_opening_priority_weight': 20.0,
    }
    plugin = os.path.join(get_package_prefix('fire_robot_sim_evidence'), 'lib')
    return LaunchDescription([
        DeclareLaunchArgument('spawn_x', default_value='-3.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        DeclareLaunchArgument('enable_mission_fsm', default_value='true'),
        DeclareLaunchArgument('contact_reference_frame', default_value='odom'),
        DeclareLaunchArgument('contact_motion_policy', default_value='measured'),
        SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', plugin + ':' + os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'observed_navigation.launch.py')),
            launch_arguments={'world': 'world1_full_contact.world', 'physical_contact': 'true',
                'robot_xacro_file': 'fire_robot_actual_piper.urdf.xacro',
                'enable_depth_camera': 'true', 'spawn_z': '0.0',
                'spawn_x': LaunchConfiguration('spawn_x'),
                'spawn_y': LaunchConfiguration('spawn_y'),
                'spawn_yaw': LaunchConfiguration('spawn_yaw'),
                'enable_mission_fsm': LaunchConfiguration('enable_mission_fsm'),
                'guarded_post_open': 'true',
                'lidar_tf_x': '0.25', 'lidar_tf_z': '0.248', 'raw_scan_topic': '/scan_raw',
                'clock_output_topic': '/clock_raw',
                'joint_states_output_topic': '/joint_states_raw'}.items()),
        Node(package='fire_robot_navigation', executable='sensor_message_gate', output='screen'),
        Node(package='fire_robot_navigation', executable='robot_self_scan_filter',
             parameters=[{'use_sim_time': True}], output='screen'),
        Node(package='fire_robot_manipulation', executable='contact_scan_localization_node',
             parameters=[{'use_sim_time': True}], output='screen',
             condition=IfCondition(PythonExpression(["'", LaunchConfiguration('contact_reference_frame'),
                                                    "' == 'contact_scan'"]))),
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='physical_evidence_bridge',
             arguments=arguments, parameters=[{'use_sim_time': True}], output='screen'),
        TimerAction(period=4.0, actions=[
            Node(package='fire_robot_manipulation', executable='robot_observation_manipulation_node',
                 name='robot_observation_manipulation_node', parameters=[parameters],
                 remappings=[('/open_door', '/physical_open_door'),
                             ('/detected_door', '/contact/target_observation'),
                             ('/manipulation_done', '/contact/manipulation_done')], output='screen'),
            Node(package='fire_robot_manipulation', executable='observed_door_angle_node',
                 parameters=[{'use_sim_time': True, 'observation_reference_frame': LaunchConfiguration('contact_reference_frame')}],
                 remappings=[('/detected_door', '/contact/target_observation')], output='screen'),
            Node(package='fire_robot_manipulation', executable='mission_contact_adapter',
                 parameters=[{'use_sim_time': True, 'contact_reference_frame': LaunchConfiguration('contact_reference_frame')}], output='screen'),
        ]),
    ])
