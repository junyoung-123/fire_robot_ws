"""Offline-only prototype: NOT connected to the controller."""
import math
import numpy as np


def fit_scan_panel(points, anchor, previous_normal, rng, threshold=.012):
    """Track the same vertical panel from a horizontal range-scan cross section."""
    points = np.asarray(points, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    points = points[np.all(np.isfinite(points),axis=1)]
    points = points[np.linalg.norm(points[:,:2]-anchor[:2],axis=1)<.95]
    if len(points)<12:
        raise ValueError('too_few_scan_points_near_tracked_panel')
    previous = np.asarray(previous_normal,dtype=float)[:2]
    previous /= np.linalg.norm(previous)
    xy = points[:,:2]
    best = None
    for _ in range(80):
        sample = xy[rng.choice(len(xy),2,replace=False)]
        direction = sample[1]-sample[0]
        length = np.linalg.norm(direction)
        if length<.10:
            continue
        normal = np.array([-direction[1],direction[0]])/length
        if abs(normal@previous)<math.cos(math.radians(20)):
            continue
        if abs((anchor[:2]-sample[0])@normal)>.16:
            continue
        mask = np.abs((xy-sample[0])@normal)<threshold
        count = int(mask.sum())
        if best is None or count>best[0]:
            best=count,mask
    if best is None or best[0]<12 or best[0]/len(xy)<.30:
        raise ValueError('no_supported_scan_panel')
    inliers=xy[best[1]]
    center=inliers.mean(axis=0)
    _,singular,vh=np.linalg.svd(inliers-center,full_matrices=False)
    normal=vh[-1]
    support=float(singular[0]/math.sqrt(len(inliers)))
    residual=float(np.sqrt(np.mean(((inliers-center)@normal)**2)))
    if support<.06 or residual>threshold:
        raise ValueError('scan_panel_support_or_residual_invalid')
    if abs(normal@previous)<math.cos(math.radians(20)):
        raise ValueError('scan_panel_orientation_disagrees_with_track')
    return dict(normal=np.r_[normal,0.],center=np.r_[center,anchor[2]],
                residual_m=residual,inliers=len(inliers),
                inlier_ratio=len(inliers)/len(xy),support_std_m=support)

