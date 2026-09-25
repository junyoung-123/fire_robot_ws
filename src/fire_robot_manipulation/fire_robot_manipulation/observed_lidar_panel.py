"""Track an already visually identified door plane using robot LiDAR points."""
import math
import numpy as np


def fit_lidar_panel(points,center,normal,rng,threshold=.06):
    points=np.asarray(points,dtype=float)[:,:2]
    center=np.asarray(center)[:2]; prior=np.asarray(normal)[:2]
    prior=prior/np.linalg.norm(prior)
    points=points[np.all(np.isfinite(points),axis=1)]
    points=points[np.linalg.norm(points-center,axis=1)<.75]
    if len(points)<30: raise ValueError('lidar_panel_insufficient_points')
    best=None
    for _ in range(80):
        pair=points[rng.choice(len(points),2,replace=False)]
        direction=pair[1]-pair[0]; length=np.linalg.norm(direction)
        if length<.15: continue
        n=np.array([-direction[1],direction[0]])/length
        if abs(n@prior)<math.cos(.20) or abs(n@(center-pair[0]))>.15: continue
        mask=np.abs((points-pair[0])@n)<threshold
        if best is None or mask.sum()>best.sum(): best=mask
    if best is None or best.sum()<30: raise ValueError('no_lidar_panel_line')
    selected=points[best]
    for _ in range(2):
        c=selected.mean(axis=0)
        _,sv,vh=np.linalg.svd(selected-c,full_matrices=False)
        n=vh[-1]
        selected=points[np.abs((points-c)@n)<threshold]
    c=selected.mean(axis=0)
    _,sv,vh=np.linalg.svd(selected-c,full_matrices=False); n=vh[-1]
    support=sv[0]/math.sqrt(len(selected)); residual=sv[1]/math.sqrt(len(selected))
    uncertainty=3*max(residual,.005)/(math.sqrt(len(selected))*max(support,1.e-9))
    if support<.10 or uncertainty>math.radians(2.): raise ValueError('lidar_panel_orientation_uncertain')
    if abs(n@prior)<math.cos(.20) or abs(n@(center-c))>.15: raise ValueError('lidar_panel_association_failed')
    return dict(normal=np.array([n[0],n[1],0.]),center=c,residual_m=float(residual),
        support_std_m=float(support),inliers=len(selected),orientation_uncertainty_rad=float(uncertainty))
