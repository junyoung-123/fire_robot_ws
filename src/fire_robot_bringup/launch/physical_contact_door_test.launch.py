from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, AppendEnvironmentVariable
from ament_index_python.packages import get_package_prefix
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    headless = LaunchConfiguration('headless')
    use_rviz = LaunchConfiguration('use_rviz')
    enable_perception = LaunchConfiguration('enable_perception')
    require_yolo_handle = LaunchConfiguration('require_yolo_handle')
    feedback_contact = LaunchConfiguration('feedback_contact_enabled')
    door_open_motion = LaunchConfiguration('door_open_motion')
    contact_min_angle = LaunchConfiguration('contact_min_angle_rad')
    door_model_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_perception'), 'models', 'best.pt'])
    handle_model_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_perception'), 'models',
        'handle_best_v3_CANDIDATE.pt'])
    handle_fallback_model_path = PathJoinSubstitution([
        FindPackageShare('fire_robot_perception'), 'models',
        'handle_best_v2.pt'])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('fire_robot_bringup'), 'launch', 'gazebo.launch.py'])),
        launch_arguments={
            'world': 'physical_contact_door_test.world',
            'headless': headless,
            'render_engine_gui': 'ogre',
            'use_sim_time': 'true',
            'robot_xacro_file': 'fire_robot_actual_piper.urdf.xacro',
            'enable_depth_camera': feedback_contact,
            'spawn_x': '-3.80',
            'spawn_z': '0.0',
            # This dedicated test intentionally permits low-speed contact with
            # the panel after the lever latch has been released.
            'front_stop_distance_m': '0.0',
        }.items())

    command_topics = [
        f'/fire_robot/sim_arm/joint{i}_cmd@std_msgs/msg/Float64]gz.msgs.Double'
        for i in range(1, 7)]
    command_topics += [
        '/fire_robot/sim_arm/gripper_left_joint_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/fire_robot/sim_arm/gripper_right_joint_cmd@std_msgs/msg/Float64]gz.msgs.Double',
        '/door_joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        '/proof/overhead/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/proof/perspective/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/proof/handle/image@sensor_msgs/msg/Image[gz.msgs.Image',
    ]
    command_topics += [
        f'/proof/audit/{part}@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts'
        for part in ('lever', 'latch', 'panel', 'jamb', 'strike')]
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='physical_contact_command_bridge',
        arguments=command_topics,
        output='screen')

    manipulation = TimerAction(period=4.0, actions=[Node(
        package='fire_robot_manipulation',
        executable='physical_contact_manipulation_node',
        name='manipulation_node',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'sim_mode': True,
            'sim_control_enabled': True,
            'sim_command_transport': 'ros',
            'sim_step_sec': 2.2,
            'sim_arm_target_timeout_sec': 12.0,
            # Exact joint convergence is impossible once the fingers load the
            # lever. This tolerance is used only together with fresh measured
            # lever rotation, so panel contact alone cannot pass the grasp.
            'sim_contact_joint_error_tolerance_rad': 0.11,
            'sim_arm_clearance_retract_m': 0.18,
            'sim_pre_stow_backoff_linear_x': -0.08,
            'sim_pre_stow_backoff_sec': 0.0,
            'sim_return_home': True,
            'piper_mount_y_m': 0.15,
            'handle_surface_inset_m': 0.0,
            'door_open_motion': door_open_motion,
            'press_handle_before_push': True,
            'lever_press_distance_m': 0.11,
            'sim_gripper_lever_width_m': 0.045,
            'sim_grasp_search_step_m': 0.02,
            'sim_physical_door_opening': True,
            'sim_door_physics_required': True,
            'sim_door_contact_only': True,
            'sim_door_contact_min_angle_rad': contact_min_angle,
            'sim_verify_door': True,
            'sim_feedback_timeout_sec': 45.0,
            'sim_lever_press_min_angle_rad': 0.08,
            'sim_lever_press_timeout_sec': 8.0,
            'sim_latch_clear_min_angle_rad': 0.25,
            'sim_latch_clear_timeout_sec': 12.0,
            'sim_gripper_release_timeout_sec': 3.0,
            'sim_gripper_release_half_width_m': 0.034,
            'sim_handle_refine_timeout_sec': 6.0,
            'sim_handle_refine_min_samples': 3,
            'sim_handle_refine_max_shift_m': 0.04,
            'sim_push_follow_enabled': True,
            'sim_push_follow_cmd_vel_topic': '/cmd_vel_manual',
            'sim_push_follow_linear_x': 0.08,
            'sim_push_follow_angular_z': -0.10,
            'sim_push_follow_radius_m': 0.60,
            'sim_push_follow_stage_sec': 1.4,
            'post_open_backoff_enabled': True,
            'post_open_backoff_linear_x': -0.10,
            'post_open_backoff_sec': 3.0,
            'sim_release_handle_before_base_push': False,
            'hold_handle_during_base_open': True,
            'require_detected_handle': require_yolo_handle,
            'require_yolo_handle': require_yolo_handle,
            'feedback_contact_enabled': feedback_contact,
            'feedback_encoder_contact_follow': feedback_contact,
            # Bounds, not predetermined motion endpoints. The measured tool
            # position and lever rotation decide completion at runtime.
            'feedback_clearance_min_m': 0.05,
            'feedback_clearance_max_m': 0.15,
            'feedback_press_max_travel_m': 0.11,
            'feedback_press_sign': -1.0,
        }])])

    perception = TimerAction(period=4.0, actions=[Node(
        package='fire_robot_perception',
        executable='door_detection_node',
        name='physical_contact_door_detection',
        condition=IfCondition(enable_perception),
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'model_path': door_model_path,
            'handle_model_path': handle_model_path,
            'handle_fallback_model_path': handle_fallback_model_path,
            'confidence_threshold': 0.20,
            'handle_confidence_threshold': 0.20,
            'yolo_min_interval_sec': 0.35,
            'handle_yolo_imgsz': 640,
            'handle_yolo_min_area_px': 400,
            'torch_num_threads': 2,
            'publish_debug_image': True,
            'publish_map_frame': False,
            'handle_require_registered_depth': feedback_contact,
            'handle_yolo_full_frame': feedback_contact,
            'camera_sources': [
                'front|/camera/color/image_raw|/camera/color/camera_info|0.0',
            ],
            'min_door_aspect_ratio': 0.65,
            'yolo_reject_edge_clipped_doors': False,
            'hsv_fallback_clipped_max_width_ratio': 0.95,
            'handle_yolo_max_area_ratio': 0.25,
            'handle_yolo_fallback_hsv': True,
            'log_handle_detections': True,
            'log_detection_candidates': True,
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
        AppendEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH',
                                  get_package_prefix('fire_robot_sim_evidence')+'/lib'),
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument('enable_perception', default_value='false'),
        DeclareLaunchArgument('require_yolo_handle', default_value='false'),
        DeclareLaunchArgument('feedback_contact_enabled', default_value='false'),
        DeclareLaunchArgument('door_open_motion', default_value='push'),
        DeclareLaunchArgument('contact_min_angle_rad', default_value='2.05'),
        gazebo,
        bridge,
        manipulation,
        perception,
        rviz,
    ])
