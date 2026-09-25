"""Short-horizon base probes checked against calibrated PIPER kinematics."""
import math
import numpy as np
from .contact_feedback import coordinated_contact_endpoint
from .observed_contact_view import view_preserving_probe
from .piper_actual_kinematics import PIPER_JOINT_LIMITS, PIPER_BODY_OUTLINE


def select_contact_probe(model, point, normal, rotation, seed, bias, preferred,
                         speed, yaw_rate, opening_speed, max_step=.008):
    point,normal,rotation,seed,bias,preferred=(np.asarray(v,float) for v in
                                            (point,normal,rotation,seed,bias,preferred))
    if not all(np.all(np.isfinite(v)) for v in (point,normal,rotation,seed,bias,preferred)):
        raise ValueError('Nonfinite contact planning observation')
    view_speed,view_yaw,_=view_preserving_probe(
        point,normal,PIPER_BODY_OUTLINE,speed,yaw_rate,opening_speed)
    velocities=sorted(set((0.,float(np.clip(speed,0.,.04)),.02)))
    recovery_turn=-math.copysign(.06,normal[1] if abs(normal[1])>1.e-9 else -1.)
    turns=sorted(set((-.04,0.,.04,recovery_turn,float(yaw_rate),float(view_yaw))))
    candidates=[]
    for v in velocities:
        for w in turns:
            duration=min(.35,max_step/max(v,.001))
            for _ in range(8):
                try:
                    endpoint,inverse_yaw=coordinated_contact_endpoint(
                        point,normal,v,w,duration,max_step,opening_speed)
                    break
                except ValueError as exc:
                    if 'exceeds bounded contact step' not in str(exc): raise
                    duration*=.5
            else:
                continue
            future_normal=inverse_yaw@normal
            obliquity=abs(math.atan2(future_normal[1],future_normal[0]))
            if obliquity>.80:
                continue
            solution=model.solve(endpoint,seed=seed,position_tolerance_m=.001,
                orientation_tolerance_rad=.03,target_rotation=inverse_yaw@rotation,
                prefer_seed_solution=True)
            if solution is None: continue
            commanded=solution.positions+bias
            margin=float(np.min(np.minimum(commanded-PIPER_JOINT_LIMITS[:,0],
                                            PIPER_JOINT_LIMITS[:,1]-commanded)))
            if margin<.025:
                continue
            clearance=float(endpoint[:2]@future_normal[:2]-np.max(PIPER_BODY_OUTLINE@future_normal[:2]))
            score=(4.*max(0.,.12-margin)+4.*max(0.,obliquity-.50)**2
                   +.2*float(np.sum((endpoint-preferred)**2))
                   +4.*max(0.,.03-clearance)**2
                   +.003*((v-speed)/.02)**2+.0002*((w-view_yaw)/.04)**2)
            candidates.append((score,v,w,margin,obliquity,clearance))
    if not candidates:
        raise ValueError('No camera-visible PIPER-safe contact probe')
    score,v,w,margin,obliquity,clearance=min(candidates)
    return v,w,dict(predicted_joint_margin_rad=margin,predicted_obliquity_rad=obliquity,
                    predicted_tool_plane_clearance_m=clearance,feasible_candidates=len(candidates))
