"""Unloaded hand withdrawal; no base motion or door-model coordinates."""
import math
import numpy as np
from .piper_actual_kinematics import PIPER_JOINT_LIMITS


def plan_released_retreat(model, joints, clearance, step=.006):
    seed=np.asarray(joints,float)
    if seed.shape!=(6,) or not np.all(np.isfinite(seed)) or not 0.<clearance<.30:
        raise ValueError('Invalid measured released posture or clearance')
    origin,initial=model.forward(seed)
    yaw=math.atan2(initial[1,2],initial[0,2])
    direction=-math.copysign(1.,yaw)
    maximum=min(abs(yaw),.6)
    count=int(math.ceil(clearance/step))
    chosen_turn=0.
    path=[]
    for index in range(1,count+1):
        point=origin-clearance*index/count*initial[:,2]
        found=None
        # Rotate only toward the base forward axis, by at most 0.06 rad per
        # step. The gripper is already open; fixed contact orientation is no
        # longer a constraint, but joint continuity and limits still are.
        for turn in np.linspace(chosen_turn,min(maximum,chosen_turn+.06),4):
            angle=direction*turn
            c,s=math.cos(angle),math.sin(angle)
            rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@initial
            solution=model.solve(point,seed=seed,position_tolerance_m=.002,
                orientation_tolerance_rad=.03,target_rotation=rotation,
                prefer_seed_solution=True,max_iterations=120)
            if solution is None: continue
            q=solution.positions
            margin=float(np.min(np.minimum(q-PIPER_JOINT_LIMITS[:,0],PIPER_JOINT_LIMITS[:,1]-q)))
            if margin<.025 or np.max(np.abs(q-seed))>.15: continue
            found=(point,rotation,q,turn)
            break
        if found is None:
            raise ValueError('No joint-safe released wrist retreat')
        point,rotation,seed,chosen_turn=found
        path.append((point.copy(),rotation.copy(),seed.copy()))
    return origin,initial[:,2].copy(),path
