"""Run the focused guards before starting the expensive physical probe."""
import importlib.util
import argparse
from pathlib import Path
import sys
import unittest


TESTS = (
    'test_contact_clearance_projection', 'test_chassis_supplied_opening',
    'test_contact_feedback', 'test_coordinated_contact',
    'test_body_contact_envelope', 'test_held_open_guards',
    'test_observed_neutral_return', 'test_release_ik_tolerance',
    'test_partial_gripper_release', 'test_gripper_servo_limits',
    'test_released_base_withdrawal', 'test_release_recovery_order',
    'test_released_wrist_unload', 'test_released_grip_clearance',
    'test_feedback_gripper_closure', 'test_post_open_retreat',
    'test_confirmed_handle_track', 'test_integrated_handle_confidence',
    'test_manipulation_global_bindings', 'test_grasp_alignment_gate',
    'test_stopped_contact_anchor', 'test_required_node_monitor',
    'test_observed_velocity_safety',
    'test_active_contact_heading',
    'test_reachable_contact_posture', 'test_rotating_contact_workspace',
    'test_observed_orbit_model',
    'test_observed_contact_geometry', 'test_integrated_video_label',
    'test_scanned_contact_scene', 'test_physical_contact_evidence',
    'test_contact_scan_time',
    'test_measured_chassis_progress',
    'test_measured_chassis_rest',
    'test_cad_scan_occlusion',
    'test_stopped_scene_recheck',
    'test_temporal_scan_consensus',
    'test_actual_held_servo',
    'test_arm_contact_velocity_barriers',
    'test_confirmed_angle_progress',
    'test_encoder_articulation',
    'test_panel_stream_recovery',
    'test_observed_preload_release',
    'test_held_height_recheck',
    'test_observed_press_budget',
    'test_camera_reacquisition',
    'test_selected_handle_view',
    'test_observed_panel_scene_motion',
    'test_observed_door_plane',
    'test_panel_support_accumulation',
    'test_wrist_yaw_lead',
    'test_contact_supervisor_budget',
    'test_precontact_pose_clock',
    'test_contact_geometry_stance',
    'test_measured_heading_reference',
    'test_contact_scene_reserve',
    'test_contact_plan_reobservation',
    'test_contact_lookahead',
    'test_contact_scan_registration',
    'test_static_line_yaw',
    'test_contact_static_reference',
    'test_contact_scan_lifecycle',
    'test_latch_clock_budget',
    'test_latch_interrupted_transition',
    'test_observed_latch_motion',
    'test_contact_commit_time',
    'test_pregrasp_docking_feasibility',
    'test_atomic_panel_observation',
    'test_projection_equivalence',
    'test_minimum_contact_displacement',
    'test_preparing_stance_drift',
    'test_observed_panel_docking',
    'test_missing_visual_occlusion',
    'test_independent_panel_barrier',
    'test_observed_lidar_panel',
    'test_wheel_pose_reobservation',
    'test_contact_local_ik',
    'test_r34_reference_comparison',
    'test_robot_encoder_tf_stamp',
    'test_observed_footprint',
    'test_observed_launch_params',
    'test_guard_sensor_contract',
    'test_world_contact_clearance',
    'test_released_home_feedback',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--tests', nargs='+', choices=TESTS)
    args = parser.parse_args()
    sys.path.insert(0,str(args.tests_dir))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
    suite = unittest.TestSuite()
    for name in TESTS:
        spec = importlib.util.spec_from_file_location(name, args.tests_dir/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if args.tests is None or name in args.tests:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
