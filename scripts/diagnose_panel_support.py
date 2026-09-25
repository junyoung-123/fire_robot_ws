import json
import math
from pathlib import Path
import cv2
import numpy as np
from fire_robot_manipulation.observed_door_plane import fit_panel, rotation_matrix
from fire_robot_manipulation.observed_panel_mask import panel_color_support

root=Path(__file__).resolve().parents[1]/'artifacts/validation'
for run in ('observed_angle_r21','observed_angle_r22'):
    folder=root/run/'sensor_inputs'
    for file in sorted(folder.glob('*.json')):
        meta=json.loads(file.read_text()); prefix=file.stem
        z=np.load(folder/(prefix+'_depth.npz'))['depth'][::4,::4]
        v,u=np.mgrid[0:meta['height']:4,0:meta['width']:4]
        mask=panel_color_support(cv2.imread(str(folder/(prefix+'_rgb.png'))),'blue')[::4,::4]
        valid=mask&np.isfinite(z)&(z>.12)&(z<4)
        z,u,v=z[valid],u[valid],v[valid]; k=meta['k']
        points=np.column_stack(((u-k[2])*z/k[0],(v-k[5])*z/k[4],z))
        tf=meta['depth_to_odom']; rotation=rotation_matrix(tf['xyzw'])
        points=points@rotation.T+np.array(tf['translation'])
        try:
            fit=fit_panel(points,meta['observation']['center'],np.random.default_rng(4))
            a=math.atan2(fit['normal'][1],fit['normal'][0])
            before=math.atan2(meta['observation']['normal'][1],meta['observation']['normal'][0])
            delta=(a-before+math.pi/2)%math.pi-math.pi/2
            print(run,prefix,'difference_deg',round(math.degrees(delta),3),
                  'support',fit['inliers'],'residual',round(fit['residual_m'],5))
        except ValueError as exc:
            print(run,prefix,'INVALID',str(exc))
