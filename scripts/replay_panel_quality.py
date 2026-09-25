"""Offline replay of saved sensor inputs; no controller or simulator commands."""
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np
from fire_robot_manipulation.observed_door_plane import fit_panel, rotation_matrix
from fire_robot_manipulation.observed_panel_mask import panel_color_support

folder=Path(sys.argv[1]); rows=[]; rng=np.random.default_rng(42)
for path in sorted(folder.glob('*deg.json')):
    meta=json.loads(path.read_text()); stem=path.with_suffix('')
    depth=np.load(str(stem)+'_depth.npz')['depth']
    bgr=cv2.imread(str(stem)+'_rgb.png')
    v,u=np.mgrid[0:depth.shape[0]:4,0:depth.shape[1]:4]
    z=depth[::4,::4]; mask=panel_color_support(bgr,'blue')[::4,::4]
    mask &= np.isfinite(z)&(z>.12)&(z<4.)
    z,u,v=z[mask],u[mask],v[mask]; k=meta['k']; tf=meta['depth_to_odom']
    points=np.column_stack(((u-k[2])*z/k[0],(v-k[5])*z/k[4],z))
    points=points@rotation_matrix(tf['xyzw']).T+tf['translation']
    previous=meta['observation']
    fit=fit_panel(points,previous['center'],rng)
    disagreement=math.degrees(math.acos(float(np.clip(abs(fit['normal']@previous['normal']),0.,1.))))
    rows.append(dict(file=path.name,support_std_m=fit['support_std_m'],
        uncertainty_deg=math.degrees(fit['orientation_uncertainty_rad']),
        disagreement_with_original_fit_deg=disagreement))
result=dict(samples=len(rows),max_disagreement_deg=max(r['disagreement_with_original_fit_deg'] for r in rows),rows=rows)
out=Path(sys.argv[2]); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
