"""Streaming offline full-pipeline audit. Evaluation truth never enters control."""
import argparse
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def check(folder):
    if not (folder/'result.json').exists():
        raise ValueError('Audit requires a closed trial with a result')
    module_path=Path(__file__).resolve().parents[1]/'physical_contact_evidence.py'
    spec=importlib.util.spec_from_file_location('contact_classifier',module_path)
    classifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classifier)
    manifest=json.loads((folder/'manifest.json').read_text())
    result=json.loads((folder/'result.json').read_text())
    graph=json.loads((folder/'evidence/ros_graph.json').read_text())
    audit=json.loads((folder/'evidence/input_audit.json').read_text())
    truth_consumers={row['node']:[topic for topic,_ in row['subscriptions']
        if topic.startswith(('/evaluation/','/proof/audit/'))] for row in graph
        if row['node']!='observed_navigation_recorder'}
    truth_consumers={node:topics for node,topics in truth_consumers.items() if topics}
    angles,bodies,jobs,active={},set(),[],None
    wall_contacts=set()
    with (folder/'evidence/events.jsonl').open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row=json.loads(line)
            kind=row['kind']
            if kind=='evaluation_articulation':
                angles[row['model']]=abs(row['door_angle_rad'])
            elif kind=='evaluation_contact':
                bodies.update(tuple(pair) for pair in row['pairs'] if classifier.chassis_contact(pair))
                if row['part'] in ('panel','lever','latch'):
                    wall_contacts.update(tuple(pair) for pair in row['pairs']
                        if any(part.startswith('wall_') for part in pair))
                if active is not None and row['part']=='lever':
                    for side in ('left','right'):
                        if any('fire_robot::gripper_finger_'+side+'::' in ' '.join(pair)
                               for pair in row['pairs']):
                            active['touched'].add(row['model'])
                            active['contacts'].setdefault((row['model'],side),[]).append(row['sim_time'])
            elif kind=='contact_feedback' and active is not None:
                active['feedback'].add(row['payload']['event'])
            elif kind=='physical_phase' and active is not None:
                if row['phase'].startswith('HANDLE_HELD_BASE_OPEN:'):
                    if active['held_start'] is None:
                        active['held_start']=row['sim_time']
                elif active['held_start'] is not None and active['held_end'] is None:
                    active['held_end']=row['sim_time']
            elif kind=='physical_job':
                payload=row['payload']
                if payload['event']=='observe_selected_handle':
                    active=dict(session=payload['session'],observed_door_id=payload['door_id'],
                        feedback=set(),touched=set(),contacts={},held_start=None,held_end=None)
                elif payload['event']=='physical_job_result' and active is not None:
                    touched=sorted(active['touched'])
                    gaps={}
                    if len(touched)==1 and active['held_start'] is not None and active['held_end'] is not None:
                        for side in ('left','right'):
                            start,end=active['held_start'],active['held_end']
                            stamps=sorted({s for s in active['contacts'].get((touched[0],side),[])
                                           if start<=s<=end})
                            stamps=[start,*stamps,end]
                            gaps[side]=max(b-a for a,b in zip(stamps,stamps[1:]))
                    names=active['feedback']
                    jobs.append(dict(session=active['session'],observed_door_id=active['observed_door_id'],
                        successful=bool(payload['success']),contacted_models=touched,
                        maximum_contact_gap_sim_sec=gaps,
                        opposed_lever_contact_confirmed=len(gaps)==2 and max(gaps.values())<=.5,
                        observed_opening_confirmed='opening_sensor_reconfirmed' in names,
                        actual_finger_release_confirmed='gripper_release_verified' in names,
                        actual_wrist_clearance_confirmed='released_handle_clearance_verified' in names,
                        actual_arm_stow_confirmed='home_verified' in names,
                        evaluation_angles_at_job_end=dict(angles)))
                    active=None
    success=[job for job in jobs if job['successful']]
    expected={m.get('name') for m in ET.parse(folder/'world_snapshot.world').getroot().find('world').findall('model')
              if m.get('name','').startswith('door_blue')}
    completed={job['contacted_models'][0] for job in success if len(job['contacted_models'])==1}
    processed=set()
    repeated_after_success=[]
    for job in jobs:
        repeated=processed.intersection(job['contacted_models'])
        if repeated:
            repeated_after_success.append(dict(session=job['session'],models=sorted(repeated)))
        if job['successful']:
            processed.update(job['contacted_models'])
    reference=manifest.get('reference_contract',{})
    scope_ok=(manifest['scope']=='live_slam_navigation_plus_sensor_piper_contact' or
        manifest['scope']=='experimental_full_mission_archived_motion_robot_only_sensors'
        and reference.get('archived_ground_truth_body_odometry_restored') is False)
    checks=dict(full_original_spawn_scope=scope_ok,
        mission_complete=result['status']=='MISSION_COMPLETE',runner_pass=result['pass'],
        source_unchanged=result['source_unchanged'],models_unchanged=result.get('model_unchanged') is True,
        recorder_scripts_unchanged=(result.get('scripts_unchanged') is True if manifest.get('script_sha256') else True),
        live_slam_map_publisher=audit['map_publishers']==['slam_toolbox'],
        no_prior_map=manifest['prior_map'] is None,wheel_odometry=manifest['odometry']=='wheel_encoders',
        no_control_truth_subscription=not truth_consumers,
        frozen_run_baseline=manifest.get('baseline_all_hashes_match') is True,
        all_expected_still_open_at_end=all(angles.get(name,0.)>=2.05 for name in expected),
        all_distinct_physical_doors_processed=completed==expected,
        no_physical_revisit_after_success=not repeated_after_success,
        every_success_has_open_release_clearance_stow_and_opposed_contact=len(success)>=len(expected) and all(
            all(job[key] for key in ('observed_opening_confirmed','actual_finger_release_confirmed',
                'actual_wrist_clearance_confirmed','actual_arm_stow_confirmed','opposed_lever_contact_confirmed'))
            for job in success),
        no_recorded_chassis_door_contact=not bodies,
        no_recorded_moving_door_wall_contact=not wall_contacts,
        whole_chassis_past_exit=result['whole_chassis_past_exit'])
    report=dict(scope='offline_only_not_control',world_id=manifest['world_id'],
        evaluation_expected_blue_models=sorted(expected),physical_opening_applicable=bool(expected),
        pass_all=all(checks.values()),checks=checks,jobs=jobs,
        physical_revisits_after_success=repeated_after_success,
        final_evaluation_angles_rad=angles,control_truth_subscriptions=truth_consumers,
        independently_classified_chassis_contacts=sorted(bodies),moving_door_wall_contacts=sorted(wall_contacts),limitations=[
            'Sampled contact evidence is not continuous force measurement.',
            'Door sensors do not prove absence of every arm-obstacle collision.',
            'Experimental archived motion is not authorized for real-robot deployment.'])
    (folder/'strict_integrated_audit.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    print(json.dumps(check(parser.parse_args().folder),indent=2))
