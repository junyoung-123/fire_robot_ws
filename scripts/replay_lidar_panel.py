"""Offline sensor feasibility study. Simulator angles are evaluation-only."""
import bisect
import json
import math
from pathlib import Path
import sys
import numpy as np
from fire_robot_manipulation.observed_lidar_panel import fit_lidar_panel

source=Path(sys.argv[1]); output=Path(sys.argv[2]); rng=np.random.default_rng(5)
rows=[json.loads(x) for x in (source/'demo_timeline.jsonl').read_text().splitlines()]
observations=[json.loads(x) for x in (source/'observed_angle_samples.jsonl').read_text().splitlines()]
camera=[x for x in observations if x['kind']=='observation' and x['valid']]
times=[x['depth_stamp'] for x in camera]; result=[]; clouds=[]; center=None; normal=None
reference=math.atan2(camera[0]['normal'][1],camera[0]['normal'][0]); previous=reference
for row in rows:
    if row['stage']!='HANDLE_HELD_BASE_OPEN' or not row.get('scan') or not row.get('pose'): continue
    now=row['sim_time']; scan=row['scan']; pose=row['pose']; q=pose['q']
    yaw=math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
    c,s=math.cos(yaw),math.sin(yaw); rotation=np.array([[c,-s],[s,c]])
    ranges=np.asarray(scan['ranges'],dtype=float); angles=scan['angle_min']+np.arange(len(ranges))*scan['angle_increment']
    valid=np.isfinite(ranges)&(ranges>scan['range_min'])&(ranges<scan['range_max'])
    p=np.column_stack((ranges[valid]*np.cos(angles[valid])+.25,ranges[valid]*np.sin(angles[valid])))
    p=p@rotation.T+[pose['x'],pose['y']]
    clouds.append((now,p)); clouds=[item for item in clouds if now-item[0]<.4]
    i=bisect.bisect_right(times,now)-1
    if i>=0 and (center is None or now-times[i]<.25):
        center=np.array(camera[i]['center']); normal=np.array(camera[i]['normal'])
        previous=reference+camera[i]['angle_rad']
    if center is None: continue
    try:
        fit=fit_lidar_panel(np.vstack([item[1] for item in clouds]),center,normal,rng)
        angle=math.atan2(fit['normal'][1],fit['normal'][0]); angle+=round((previous-angle)/math.pi)*math.pi
        center=np.r_[fit['center'],center[2]]; normal=fit['normal']; previous=angle
        estimate=angle-reference
        result.append(dict(sim_time=now,valid=True,estimate_deg=math.degrees(estimate),
            error_deg=math.degrees(estimate-row['door_angle_rad']),inliers=fit['inliers'],
            uncertainty_deg=math.degrees(fit['orientation_uncertainty_rad'])))
    except ValueError as exc: result.append(dict(sim_time=now,valid=False,reason=str(exc)))
valid=[r for r in result if r['valid']]; errors=np.array([r['error_deg'] for r in valid])
summary=dict(samples=len(result),valid=len(valid),rmse_deg=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
    max_error_deg=float(np.max(np.abs(errors))) if len(errors) else None,
    scope='Offline replay; recorded pose/scan pairing approximate; 0.25m lidar mounting from robot URDF, not door coordinates')
output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(dict(summary=summary,rows=result),indent=2))
print(json.dumps(summary))
