#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1])/'runtime_inputs'
out.mkdir(parents=True,exist_ok=False)
(out/'COLCON_IGNORE').touch()
paths=list((root/'src/fire_robot_manipulation').rglob('*.py'))
paths += [root/p for p in (
    'src/fire_robot_bringup/launch/physical_contact_door_test.launch.py',
    'src/fire_robot_bringup/launch/gazebo.launch.py',
    'src/fire_robot_bringup/worlds/physical_contact_door_test.world',
    'src/fire_robot_bringup/worlds/physical_contact_door_mirrored.world',
    'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py',
    'src/fire_robot_description/urdf/fire_robot_actual_piper.urdf.xacro',
    'src/fire_robot_perception/fire_robot_perception/door_detection_node.py',
    'scripts/run_observed_angle_demo.py','scripts/run_corridor_demo.py',
    'scripts/start_observed_angle.sh','scripts/capture_angle_sensor_inputs.py',
    'scripts/start_minimal_angle.sh','scripts/test_minimal_angle_policy.py',
    'scripts/start_robot_observation.sh','scripts/test_encoder_articulation.py',
    'scripts/audit_robot_observation_graph.py','ROBOT_OBSERVATION_ONLY.md',
    'scripts/test_piper_ik_boundary.py',
    'scripts/replay_panel_quality.py','src/fire_robot_description/urdf/piper_arm_actual.xacro',
    'scripts/replay_lidar_panel.py','scripts/test_observed_lidar_panel.py',
    'scripts/test_observed_panel_mask.py',
    'scripts/prepare_mirrored_contact_world.py',
    'scripts/test_observed_door_plane.py','scripts/test_observed_angle_input_contract.py',
    'scripts/test_observed_contact_view.py','scripts/test_contact_probe_selector.py',
    'scripts/test_released_base_withdrawal.py',
    'scripts/test_observed_freshness.py',
    'scripts/test_confirmed_handle_track.py','REAL_SENSOR_INTERFACE.md',
    'OBSERVED_ANGLE_EXPERIMENT.md')]
manifest={}
for path in paths:
    relative=path.relative_to(root)
    dest=out/relative
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,dest)
    manifest[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest['base_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(snapshot=str(out),files=len(paths))))
