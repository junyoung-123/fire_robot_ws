"""Bounded base-heading correction to keep the observed door panel in view."""
import math
import numpy as np
from fire_robot_manipulation.contact_feedback import coordinated_contact_endpoint


def joint_limited_view_obliquity(joints, limits, bias, nominal=.50):
    joints,limits,bias=(np.asarray(v,float) for v in (joints,limits,bias))
    if (joints.shape!=(6,) or limits.shape!=(6,2) or bias.shape!=(6,)
            or not all(np.all(np.isfinite(v)) for v in (joints,limits,bias))):
        raise ValueError('Invalid joint margin observation')
    margin=float(np.min(np.minimum(joints-limits[:,0],limits[:,1]-joints)-np.abs(bias)))
    # Turn the base earlier as encoder-measured wrist headroom is consumed.
    # This reduces arm twist without changing any physical joint limits.
    maximum=nominal*float(np.clip(margin/.25,.4,1.))
    return maximum,margin


def view_preserving_probe(point, normal, outline, speed, yaw_rate, opening_speed,
                          max_obliquity=.50):
    point,normal,outline=(np.asarray(v,float) for v in (point,normal,outline))
    if (point.shape!=(3,) or normal.shape!=(3,) or outline.ndim!=2
            or outline.shape[1]!=2 or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(normal)) or not np.all(np.isfinite(outline))
            or not all(math.isfinite(v) for v in (speed,yaw_rate,opening_speed))):
        raise ValueError('Invalid observation-preserving probe input')
    angle=math.atan2(normal[1],normal[0])
    if abs(angle)<=max_obliquity:
        return speed,yaw_rate,None
    requested=float(np.clip(.30*(angle-math.copysign(max_obliquity,angle)),-.06,.06))
    yaw=min(yaw_rate,requested) if angle<0 else max(yaw_rate,requested)
    target_speed=float(np.clip(speed+(yaw-yaw_rate)*point[1],0.,.06))
    clearance=float(point[:2]@normal[:2]-np.max(outline@normal[:2]))
    # This is the conservative tool plane, not the recessed door surface.
    # If the estimate is already negative, allow only clearance recovery.
    minimum=min(clearance,.035) if clearance>=0. else clearance+.001
    for candidate in np.linspace(target_speed,0.,9):
        endpoint,rotation=coordinated_contact_endpoint(
            point,normal,float(candidate),yaw,.35,.10,opening_speed)
        future_normal=rotation@normal
        future=float(endpoint[:2]@future_normal[:2]-np.max(outline@future_normal[:2]))
        if future>=minimum:
            return float(candidate),float(yaw),dict(obliquity_rad=angle,
                predicted_clearance_m=future,current_clearance_m=clearance,
                nominal_linear_x=speed,nominal_angular_z=yaw_rate)
    return speed,yaw_rate,None
