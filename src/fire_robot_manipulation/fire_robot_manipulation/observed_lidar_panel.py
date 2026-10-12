"""Track an already visually identified door plane using robot LiDAR points."""
import math
import numpy as np


def panel_offset_uncertainty(points, center, normal, residual):
    """Separate fitted plane location error from individual range scatter."""
    points=np.asarray(points,float)
    center=np.asarray(center,float); normal=np.asarray(normal,float)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<30
            or center.shape!=(2,) or normal.shape!=(2,)
            or not all(np.all(np.isfinite(v)) for v in (points,center,normal))
            or not math.isfinite(residual) or not 0.<=residual<=.06
            or abs(np.linalg.norm(normal)-1.)>1.e-6):
        raise ValueError('invalid_lidar_plane_offset_support')
    # Cap effective samples and leave out spatial halves/quarters to expose bias;
    # the full raw scatter still belongs in the collision clearance reserve.
    bound=max(.005,3.*residual/math.sqrt(min(len(points),64)))
    tangent=np.array([-normal[1],normal[0]])
    ordered=points[np.argsort((points-center)@tangent)]
    omissions=list(np.array_split(np.arange(len(ordered)),4))
    if len(ordered)>=60:
        omissions.extend(np.array_split(np.arange(len(ordered)),2))
    for omitted in omissions:
        subset=np.delete(ordered,omitted,axis=0)
        if len(subset)<30:
            raise ValueError('too_few_points_for_lidar_spatial_cross_check')
        other_center=subset.mean(axis=0)
        _,_,axes=np.linalg.svd(subset-other_center,full_matrices=False)
        bound=max(bound,abs(float((center-other_center)@axes[-1])))
    if bound>.03:
        raise ValueError('lidar_panel_location_uncertain')
    return float(bound)


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
    offset_uncertainty=panel_offset_uncertainty(selected,c,n,float(residual))
    tangent=np.array([-n[1],n[0]])
    lower,upper=np.quantile((selected-c)@tangent,[.05,.95])
    segment=np.array([c+lower*tangent,c+upper*tangent])
    return dict(normal=np.array([n[0],n[1],0.]),center=c,residual_m=float(residual),
        plane_offset_uncertainty_m=offset_uncertainty,
        segment_xy=segment,segment_edge_margin_m=max(.05,3.*float(residual)),
        support_std_m=float(support),inliers=len(selected),orientation_uncertainty_rad=float(uncertainty))
