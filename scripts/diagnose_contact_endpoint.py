"""Offline IK sensitivity from recorded robot feedback; not a controller."""
import json
from pathlib import Path
import re
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_LIMITS
from fire_robot_manipulation.contact_feedback import coordinated_contact_endpoint

lines=Path(sys.argv[1]).read_text().splitlines()
events=[json.loads(line.split('CONTACT_FEEDBACK ',1)[1]) for line in lines if 'CONTACT_FEEDBACK ' in line]
event=next(e for e in reversed(events) if e['event']=='contact_command')
bias=next((np.array(e['joint_bias_rad']) for e in reversed(events) if 'joint_bias_rad' in e),np.zeros(6))
line=next(line for line in reversed(lines) if 'Arm target stage=encoder_contact_follow:' in line)
seed=np.array([float(x) for x in re.findall(r'joint\d=([-\d.]+)',line)])
point=np.array(event['requested_tool_xyz']); rotation=np.array(event['requested_rotation'])
normal=rotation[:,2].copy(); normal[2]=0.; normal/=np.linalg.norm(normal)
model=PiperActualKinematics()
rows=[]
for v in (0.,.007,.015,.025,.04):
    for w in (-.06,-.03,0.,.03,.06):
        endpoint,change=coordinated_contact_endpoint(point,normal,v,w,.35,.10,.012)
        solution=model.solve(endpoint,seed=seed,position_tolerance_m=.003,
            orientation_tolerance_rad=.03,target_rotation=change@rotation,prefer_seed_solution=True)
        if solution is None: continue
        command=solution.positions+bias
        margin=float(np.min(np.minimum(command-PIPER_JOINT_LIMITS[:,0],PIPER_JOINT_LIMITS[:,1]-command)))
        rows.append(dict(v=v,w=w,margin=margin,q5=float(command[4]),endpoint=endpoint.tolist()))
print(json.dumps(dict(point=point.tolist(),seed=seed.tolist(),bias=bias.tolist(),
                      candidates=sorted(rows,key=lambda r:r['margin'],reverse=True)),indent=2))
