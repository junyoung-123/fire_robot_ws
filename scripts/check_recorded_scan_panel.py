"""Offline feasibility check; recorded pose/scan are not exact-time TF replay."""
import bisect
import json
import math
from pathlib import Path
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src/fire_robot_manipulation'))
from fire_robot_manipulation.observed_door_plane import rotation_matrix
from scan_panel_prototype import fit_scan_panel

source=Path(sys.argv[1])
records=[json.loads(line) for line in (source/'observed_angle_samples.jsonl').read_text().splitlines()]
obs=[r for r in records if r['kind']=='observation' and r['valid']]
times=[r['depth_stamp'] for r in obs]
rows=[json.loads(line) for line in (source/'demo_timeline.jsonl').read_text().splitlines()]
extrinsics=json.loads(Path(sys.argv[2]).read_text())['lidar_link']
rng=np.random.default_rng(5)
for goal in [10,50,75,100,120,130,134]:
    row=min(rows,key=lambda r:abs(r['sim_time']-goal))
    index=bisect.bisect_right(times,row['sim_time'])-1
    if index<0 or not row['scan']:
        continue
    prior=obs[index]
    scan=row['scan']; pose=row['pose']
    ranges=np.array([np.nan if r is None else r for r in scan['ranges']])
    angles=scan['angle_min']+np.arange(len(ranges))*scan['angle_increment']
    good=np.isfinite(ranges)&(ranges>scan['range_min'])&(ranges<scan['range_max'])
    points=np.column_stack([ranges[good]*np.cos(angles[good]),ranges[good]*np.sin(angles[good]),np.zeros(good.sum())])
    points=points@rotation_matrix(extrinsics['xyzw']).T+extrinsics['xyz']
    points=points@rotation_matrix(pose['q']).T+[pose['x'],pose['y'],pose['z']]
    try:
        fit=fit_scan_panel(points,prior['center'],prior['normal'],rng,
                           threshold=float(sys.argv[3]) if len(sys.argv)>3 else .012)
        agreement=math.degrees(math.acos(np.clip(abs(fit['normal']@np.array(prior['normal'])),0,1)))
        result=dict(sim_time=row['sim_time'],valid=True,inliers=fit['inliers'],
                    support=fit['support_std_m'],residual=fit['residual_m'],normal=fit['normal'].tolist(),
                    camera_normal_difference_deg=agreement)
    except ValueError as exc:
        result=dict(sim_time=row['sim_time'],valid=False,reason=str(exc))
    print(json.dumps(result))
