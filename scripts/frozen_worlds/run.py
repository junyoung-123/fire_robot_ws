#!/usr/bin/env python3
"""Bounded same-run integration test with isolated ROS/Gazebo process groups."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_observed_navigation import ReadinessWatch, stop_group
from physical_contact_evidence import chassis_contact


def failed_required_node(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    required = ('robot_observation_manipulation_node', 'observed_door_angle_node',
                'mission_contact_adapter', 'contact_scan_localization_node',
                'door_detection_node', 'state_machine_node', 'cmd_vel_safety_node')
    for line in lines:
        if '[ERROR]' not in line or 'process has died' not in line:
            continue
        for name in required:
            if '['+name+'-' in line:
                return name
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-label', default='R243')
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--contact-probe', action='store_true')
    parser.add_argument('--probe-spawn',type=float,nargs=3,metavar=('X','Y','YAW'),
                        help='Diagnostic fixture spawn only; never a measured target or full-mission input')
    parser.add_argument('--full-reference-validation', action='store_true',
                        help='Explicit experimental full mission with archived opening motion and robot-only sensors')
    parser.add_argument('--contact-motion-policy', choices=('measured', 'r34_reference'), default='measured')
    parser.add_argument('--contact-reference-frame', choices=('odom','map','contact_scan'), default='odom')
    parser.add_argument('--dds-udp-only', action='store_true')
    parser.add_argument('--static-return-memory-sec', type=float, default=2.)
    args = parser.parse_args()
    if not math.isfinite(args.static_return_memory_sec) or not 2.<=args.static_return_memory_sec<=30.:
        parser.error('--static-return-memory-sec must be bounded to 2..30 seconds')
    if args.probe_spawn is not None and (not args.contact_probe or not all(math.isfinite(v) for v in args.probe_spawn)):
        parser.error('--probe-spawn requires diagnostic --contact-probe and finite fixture coordinates')
    if args.full_reference_validation and (args.contact_probe or args.contact_motion_policy!='r34_reference'):
        parser.error('--full-reference-validation requires r34_reference and original full-mission spawn')
    if args.contact_motion_policy=='r34_reference' and not (args.contact_probe or args.full_reference_validation):
        parser.error('R34 reference requires --contact-probe or explicit --full-reference-validation')
    ws = Path(__file__).resolve().parents[2]
    baseline = json.loads(args.baseline.read_text())
    baseline_checks = {path: (ws/path).is_file() and hashlib.sha256((ws/path).read_bytes()).hexdigest() == digest
        for group in ('source_sha256', 'model_sha256', 'script_sha256')
        for path, digest in baseline[group].items()}
    if not all(baseline_checks.values()):
        raise RuntimeError('Frozen source baseline mismatch: ' + str([p for p, ok in baseline_checks.items() if not ok]))
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.fixture, args.output/'world_snapshot.world')
    world = ET.parse(args.fixture).getroot().find('world')
    expected = sorted(model.get('name') for model in world.findall('model')
                      if model.get('name', '').startswith('door_blue'))
    env = dict(os.environ, ROS_DOMAIN_ID=str(100 + os.getpid() % 100), IGN_PARTITION=f'frozen_w{args.world}_{os.getpid()}',
               GZ_PARTITION=f'frozen_w{args.world}_{os.getpid()}', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               LIBGL_ALWAYS_SOFTWARE='1')
    env['FROZEN_VALIDATION_WORLD'] = str(args.fixture.resolve())
    if args.dds_udp_only:
        env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(ws/'scripts/recording_udp.xml')
        env['FASTDDS_DEFAULT_PROFILES_FILE'] = env['FASTRTPS_DEFAULT_PROFILES_FILE']
    manifest = {'scope': 'live_slam_navigation_plus_sensor_piper_contact', 'prior_map': None,
                'contact_reference_frame': args.contact_reference_frame,
                'contact_motion_policy': args.contact_motion_policy,
                'odometry': 'wheel_encoders', 'environment_truth': 'evaluation_only',
                'static_post_open_free_memory_sec': args.static_return_memory_sec,
                'static_memory_assumption': 'fixed obstacles; completed door; fresh hits override history',
                'domain': env['ROS_DOMAIN_ID'], 'partition': env['IGN_PARTITION'],
                'dds_transport': 'udp_only' if args.dds_udp_only else 'default',
                'source_sha256': {str(p.relative_to(ws)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ws/'src').rglob('*') if p.is_file()
                                  and p.suffix in ('.py', '.yaml', '.world', '.xacro', '.cpp')}}
    manifest.update(world_id=args.world, evaluation_blue_models=expected,
        frozen_baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        baseline_all_hashes_match=True, baseline_label=args.baseline_label,
        no_policy_tuning=args.baseline_label == 'R243', no_changes_during_run=True)
    weights = [ws/'src/fire_robot_perception/models'/name for name in
               ('best.pt', 'handle_best_v3_CANDIDATE.pt', 'handle_best_v2.pt')]
    manifest['model_sha256'] = {str(path.relative_to(ws)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in weights}
    script_names=('run_world1_full.py','full_mission_recorder.py','observed_navigation_recorder.py',
                  'run_observed_navigation.py','physical_contact_evidence.py')
    manifest['script_sha256']={str(Path('scripts')/name):hashlib.sha256((ws/'scripts'/name).read_bytes()).hexdigest()
                              for name in script_names}
    manifest['script_sha256'].update({str(path.relative_to(ws)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).parent.glob('*.py')})
    if args.full_reference_validation:
        manifest['scope'] = 'experimental_full_mission_archived_motion_robot_only_sensors'
        manifest['reference_contract'] = {
            'archived_motion_core_unchanged': True,
            'archived_ground_truth_body_odometry_restored': False,
            'approach_release_and_sensor_provider': 'current_observed_adapters',
            'nominal_short_probe_interpolation': True,
            'probe_pause_handling': 'abort_partial_step_reobserve_replan_no_virtual_sensor_clock',
            'completion_target': 'task_angle_plus_bounded_fusion_and_raw_sensor_confidence_reserves',
            'contact_pose_reference': 'stationary_original_lidar_beam_temporal_median',
            'contact_reference_extension': 'repeated_new_static_returns_original_anchor_only_no_old_xy_replacement',
            'extension_pose_confidence': 'original_anchor_translation_0.025m_yaw_0.5deg_and_8_distinct_scans',
            'uncertain_scan_translation': 'not_applied_not_certified_own_wheel_translation_retained',
            'contact_yaw': 'qualified_static_scan_yaw_with_original_timestamp',
            'panel_angle_confidence': 'independent_camera_lidar_3sigma_rss_with_correlated_yaw_group',
            'contact_correction_filters_sec': {'translation':2.,'yaw':.4},
            'correction_rotation_pivot': 'current_timestamped_lidar_origin_not_spawn_origin',
            'partial_gripper_release': 'axial_only_until_both_jaws_open_parent_budget_includes_bounded_children',
            'gripper_actuator': 'original_pd_400_0_6.3_8N_cap_no_contact_integral_windup',
            'latch_execution_adapter': 'archived_press_probe_arithmetic_sim_30s_wall_600s_stall_3s',
            'loaded_preload_adapter': 'persist_depth_only_within_observed_unloading_support_otherwise_original_stopped_servo_and_tracking_gate',
            'neutral_return_servo': 'own_encoder_bias_gain_1_period_0.25s_original_2mm_and_0.03rad_gates',
            'stopped_stream_recovery': 'all_streams_and_clock_frozen_only_wall_5s_resume_sim_0.15s_no_old_sample_refresh',
            'held_press_height_contract': 'accepted_hold_correction_updates_follow_height_without_resetting_depth',
            'reference_retention_dispatch': 'robot_feedback_adapter_not_frozen_reference',
            'docking_press_reserve': 'observed_xyz_actual_ik_joint_margin_0.20_no_prescribed_press_endpoint',
            'initial_panel_geometry': 'static_reference_initialization_only_not_angle_completion',
            'newer_scene_stance_guards_in_reference_core': False,
            'real_robot_deployment_authorized': False,
            'acceptance': 'unchanged_physical_contacts_angles_all_jobs_whole_chassis_exit'}
    if args.dds_udp_only:
        shutil.copy2(ws/'scripts/recording_udp.xml', args.output/'recording_udp.xml')
        manifest['dds_profile_sha256'] = hashlib.sha256(
            (ws/'scripts/recording_udp.xml').read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for relative in (*manifest['source_sha256'],*manifest['script_sha256']):
        destination = args.output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ws/relative, destination)
    commands = [(['ros2', 'launch', str(Path(__file__).parent/'runtime.launch.py'),
                  'contact_reference_frame:='+args.contact_reference_frame,
                  'guard_static_free_history_sec:='+str(args.static_return_memory_sec),
                  'contact_motion_policy:='+args.contact_motion_policy], 'simulation.log'),
                (['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                  '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry', '--ros-args',
                  '-r', '__node:=evaluation_truth_bridge', '-r', '/odom:=/evaluation/ground_truth_odom'], 'evaluation_bridge.log'),
                (['python3', str(Path(__file__).parent/'recorder.py'), '--output', str(args.output/'evidence'),
                  '--world', str(args.world), '--fixture', str(args.fixture.resolve())], 'recorder.log')]
    if args.contact_probe:
        manifest['scope'] = 'diagnostic_contact_only_spawn_changed_not_full_mission'
        spawn=args.probe_spawn or [3.2,-.9,-1.57079632679]
        manifest['diagnostic_fixture_spawn']=spawn
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        commands[0][0].extend([f'spawn_x:={spawn[0]}',f'spawn_y:={spawn[1]}',
                              f'spawn_yaw:={spawn[2]}','enable_mission_fsm:=false'])
        commands.append((['python3', str(ws/'scripts/request_observed_contact.py'),
            '--output', str(args.output/'probe_result.json'), '--ros-args',
            '-p', 'use_sim_time:=true'], 'probe_client.log'))
    streams, processes = [], []
    status, start, watch = 'TIMEOUT', time.monotonic(), ReadinessWatch()
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for command, name in commands:
            stream = (args.output/name).open('w')
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=ws, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True))
        print(json.dumps({'output': str(args.output), 'partition': env['IGN_PARTITION']}), flush=True)
        while time.monotonic()-start < args.timeout:
            failed = failed_required_node(args.output/'simulation.log')
            if failed:
                status = 'REQUIRED_NODE_EXIT:'+failed
                break
            audit_file = args.output/'evidence/input_audit.json'
            if not audit_file.exists() and time.monotonic()-start > 540:
                status = 'RECORDER_SENSOR_STARTUP_TIMEOUT'
                break
            if audit_file.exists():
                try:
                    audit = json.loads(audit_file.read_text())
                except json.JSONDecodeError:
                    audit = None
                if audit is not None:
                    readiness_failure = watch.check(time.monotonic()-start, audit)
                    if readiness_failure:
                        status = readiness_failure
                        break
            probe_result = args.output/'probe_result.json'
            if args.contact_probe and probe_result.exists():
                status = 'CONTACT_PROBE_SUCCESS' if json.loads(probe_result.read_text())['success'] else 'CONTACT_PROBE_FAILED'
                time.sleep(3)
                break
            if (args.output/'stop_requested').exists():
                status = 'STOPPED_FOR_DIAGNOSIS'
                break
            terminal = args.output/'evidence/terminal_state.json'
            if terminal.exists():
                status = json.loads(terminal.read_text())['state']
                time.sleep(3)
                break
            if any(p.poll() is not None for p in processes):
                status = 'PROCESS_EXIT'
                break
            time.sleep(2)
    except KeyboardInterrupt:
        status = 'STOPPED_FOR_DIAGNOSIS'
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process in processes:
            stop_group(process)
        for stream in streams:
            stream.close()
        # ros2 CLI discovery can detach its daemon from the launch group.
        # Stop only this trial's domain so later runs cannot inherit it.
        with (args.output/'discovery_cleanup.log').open('w') as stream:
            try:
                subprocess.run(['ros2','daemon','stop'],env=env,stdout=stream,
                               stderr=subprocess.STDOUT,timeout=10.,check=False)
            except subprocess.TimeoutExpired:
                stream.write('Domain discovery daemon stop timed out\n')
        evidence = args.output/'evidence/full_checkpoint.json'
        checkpoint = json.loads(evidence.read_text()) if evidence.exists() else {}
        angles = checkpoint.get('evaluation_max_angles_rad', {})
        contacts = checkpoint.get('contact_pairs', {})
        jobs = checkpoint.get('physical_jobs', [])
        all_open = all(angles.get(name, 0) >= 2.05 for name in expected)
        gripper_contact = all(any('gripper' in ' '.join(pair) for pair in contacts.get(name+'/lever', []))
                              for name in expected)
        body_contacts = [pair for pairs in contacts.values() for pair in pairs if chassis_contact(pair)]
        exit_clear = False
        truth_file = args.output/'evidence/evaluation_poses.csv'
        if truth_file.exists():
            rows = list(csv.DictReader(truth_file.open()))
            if rows:
                pose = rows[-1]
                world = ET.parse(args.output/'world_snapshot.world').getroot().find('world')
                wall = world.find("model[@name='wall_east_upper']")
                portal_x = float(wall.findtext('pose').split()[0])
                portal_x += .5*float(wall.findtext('.//collision/geometry/box/size').split()[0])
                from fire_robot_manipulation.piper_actual_kinematics import PIPER_BODY_OUTLINE
                yaw = float(pose['yaw'])
                rear_extent = min(x*math.cos(yaw)-y*math.sin(yaw) for x,y in PIPER_BODY_OUTLINE)
                exit_clear = bool(float(pose['x'])+rear_extent > portal_x)
        source_unchanged = all(hashlib.sha256((ws/path).read_bytes()).hexdigest() == digest
                               for path, digest in manifest['source_sha256'].items())
        model_unchanged = all(hashlib.sha256((ws/path).read_bytes()).hexdigest() == digest
                              for path, digest in manifest['model_sha256'].items())
        scripts_unchanged=all(hashlib.sha256((ws/path).read_bytes()).hexdigest()==digest
                              for path,digest in manifest['script_sha256'].items())
        result = {'status': status, 'world_id': args.world,
                  'evaluation_expected_blue_count': len(expected),
                  'physical_opening_applicable': bool(expected), 'wall_seconds': time.monotonic()-start,
                  'all_physical_doors_open': all_open, 'gripper_lever_contacts': gripper_contact,
                  'successful_physical_jobs': sum(j.get('success', False) for j in jobs),
                  'body_door_contacts': body_contacts,
                  'whole_chassis_past_exit': exit_clear, 'source_unchanged': source_unchanged,
                  'model_unchanged': model_unchanged,
                  'scripts_unchanged': scripts_unchanged,
                  'pass': status == 'MISSION_COMPLETE' and all_open and gripper_contact
                          and sum(j.get('success', False) for j in jobs) >= len(expected)
                          and not body_contacts and exit_clear and source_unchanged and model_unchanged and scripts_unchanged}
        (args.output/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
