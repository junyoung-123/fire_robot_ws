"""Relative lever estimate under a rigid grasp, from measured wrist rotation."""
import math
import numpy as np


def gripper_bounds(contact_offset,front_extent,tolerance):
    # Union of the supplied piper_arm_actual.xacro palm and fully open fingers.
    return (np.array([-.040,-.0725,-contact_offset])-tolerance,
            np.array([.0305,.0725,front_extent-contact_offset])+tolerance)


def inside_gripper_envelope(point,tool,rotation,contact_offset,front_extent,tolerance):
    local=np.asarray(rotation).T@(np.asarray(point)-np.asarray(tool))
    lower,upper=gripper_bounds(contact_offset,front_extent,tolerance)
    return bool(np.all(local>=lower) and np.all(local<=upper))


def gripper_occludes_ray(camera,point,tool,rotation,contact_offset,front_extent,tolerance):
    """Segment-box test in the measured wrist frame, not a world-coordinate mask."""
    matrix=np.asarray(rotation).T
    start=matrix@(np.asarray(camera)-np.asarray(tool))
    end=matrix@(np.asarray(point)-np.asarray(tool))
    if not np.all(np.isfinite([start,end])): return False
    lower,upper=gripper_bounds(contact_offset,front_extent,tolerance)
    near,far=0.,1.
    for axis,delta in enumerate(end-start):
        if abs(delta)<1.e-9:
            if start[axis]<lower[axis] or start[axis]>upper[axis]: return False
            continue
        a,b=sorted(((lower[axis]-start[axis])/delta,(upper[axis]-start[axis])/delta))
        near=max(near,a); far=min(far,b)
        if near>far: return False
    return near<1. and far>0.


def lever_from_wrist(reference, measured, door_delta, base_delta, residual_limit=.12):
    reference=np.asarray(reference,dtype=float)
    measured=np.asarray(measured,dtype=float)
    if reference.shape!=(3,3) or measured.shape!=(3,3):
        raise ValueError('invalid_wrist_rotation_shape')
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(measured)):
        raise ValueError('nonfinite_wrist_rotation')
    if not all(math.isfinite(v) for v in (door_delta,base_delta)):
        raise ValueError('nonfinite_door_or_base_rotation')
    angle=door_delta-base_delta
    c,s=math.cos(angle),math.sin(angle)
    yaw=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    relative=yaw.T@measured@reference.T
    roll=math.atan2(relative[2,1],relative[1,1])
    c,s=math.cos(roll),math.sin(roll)
    expected=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    residual=math.acos(float(np.clip((np.trace(expected.T@relative)-1)/2,-1,1)))
    if residual>residual_limit or abs(roll)>.7:
        raise ValueError('wrist_motion_inconsistent_with_lever_constraint')
    return roll,residual


def articulation_from_wrist(reference,measured,base_delta,residual_limit=.12):
    """Absolute FK decomposition under grasp: door yaw and lever roll, no commands."""
    reference=np.asarray(reference,dtype=float); measured=np.asarray(measured,dtype=float)
    if reference.shape!=(3,3) or measured.shape!=(3,3): raise ValueError('invalid_wrist_shape')
    if not np.all(np.isfinite([reference,measured])) or not math.isfinite(base_delta):
        raise ValueError('nonfinite_wrist_measurement')
    c,s=math.cos(base_delta),math.sin(base_delta)
    relative=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@measured@reference.T
    yaw=math.atan2(relative[1,0],relative[0,0])
    roll=math.atan2(relative[2,1],relative[2,2])
    c,s=math.cos(yaw),math.sin(yaw); cr,sr=math.cos(roll),math.sin(roll)
    expected=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@np.array([[1.,0.,0.],[0.,cr,-sr],[0.,sr,cr]])
    residual=math.acos(float(np.clip((np.trace(expected.T@relative)-1)/2,-1,1)))
    if residual>residual_limit or abs(roll)>.7:
        raise ValueError('wrist_pose_inconsistent_with_grasped_door_and_lever')
    return yaw,roll,residual


def fused_grasped_panel_angle(panel_angle,reference,measured,base_delta,door_baseline,
                              encoder_weight=.8,maximum_disagreement=.12,previous_estimate=None,
                              panel_uncertainty_rad=None):
    """Measured FK smooths panel noise only under independently supported grasp.

    This is not opening-completion evidence. The caller must keep raw panel
    observations for independent confirmation and stop on disagreement/slip.
    """
    if (not all(math.isfinite(v) for v in (panel_angle,base_delta,door_baseline,
                                         encoder_weight,maximum_disagreement))
            or not 0.<=encoder_weight<=1. or not 0.<maximum_disagreement<=.12):
        raise ValueError('invalid_grasped_panel_fusion')
    yaw,roll,residual=articulation_from_wrist(reference,measured,base_delta)
    encoder_angle=door_baseline+yaw
    prediction=encoder_angle
    if previous_estimate is not None:
        old_angle,old_encoder=previous_estimate
        if not all(math.isfinite(v) for v in (old_angle,old_encoder)):
            raise ValueError('invalid_grasped_panel_filter_baseline')
        # Predict from measured increments, not an absolute wrist yaw which
        # can differ slightly from the panel through grasp compliance.
        prediction=old_angle+math.atan2(math.sin(encoder_angle-old_encoder),
                                       math.cos(encoder_angle-old_encoder))
    disagreement=math.atan2(math.sin(panel_angle-prediction),math.cos(panel_angle-prediction))
    if abs(disagreement)>maximum_disagreement:
        raise ValueError('independent_panel_disagrees_with_measured_grasp')
    estimate = prediction+(1.-encoder_weight)*disagreement
    if panel_uncertainty_rad is not None:
        if (not math.isfinite(panel_uncertainty_rad)
                or not 0. <= panel_uncertainty_rad <= math.radians(2.)):
            raise ValueError('invalid_independent_panel_confidence')
        # A compliant wrist is not the door joint. Prediction may smooth
        # within the independent sensor interval, never move that interval.
        sensor_angle = prediction+disagreement
        estimate = float(np.clip(estimate, sensor_angle-panel_uncertainty_rad,
                                 sensor_angle+panel_uncertainty_rad))
    return float(estimate),float(disagreement),roll,residual
