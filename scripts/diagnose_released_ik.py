"""Compare unloaded retreat IK against recorded loaded-servo compensation."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/fire_robot_manipulation'))
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics,PIPER_JOINT_LIMITS

root=Path(sys.argv[1])
events=[]
for line in (root/'launch.log').read_text().splitlines():
    if 'CONTACT_FEEDBACK {' in line:
        events.append(json.JSONDecoder().raw_decode(line.split('CONTACT_FEEDBACK ',1)[1])[0])
release=next(e for e in events if e['event']=='handle_released_after_full_open')
bias=next((np.array(e['joint_bias_rad']) for e in reversed(events)
           if 'joint_bias_rad' in e and e['sim_time_sec']<=release['sim_time_sec']),np.zeros(6))
data=json.loads((root/'motion_diagnostics.json').read_text())
row=min(data['joints'],key=lambda x:abs(x['sim_time_sec']-release['sim_time_sec']))
q=np.array([row['positions'][row['names'].index('joint'+str(i))] for i in range(1,7)])
model=PiperActualKinematics()
point,rotation=model.forward(q)
print('release',release['sim_time_sec'],'q',q.tolist(),'bias',bias.tolist())
for length in ():
    for tolerance in (.03,.10,.20):
        solution=model.solve(point-length*rotation[:,2],seed=q,position_tolerance_m=.003,
            orientation_tolerance_rad=tolerance,target_rotation=rotation,prefer_seed_solution=True)
        if solution is None:
            print(length,tolerance,'no IK'); continue
        margin=lambda v:float(np.min(np.minimum(v-PIPER_JOINT_LIMITS[:,0],PIPER_JOINT_LIMITS[:,1]-v)))
        print(length,tolerance,'margin',margin(solution.positions),'loaded_bias_margin',
              margin(solution.positions+bias),'error',solution.orientation_error_rad)

for height in (0.,):
    for roll in (.1,.2,.3,.4,.5):
        c,s=np.cos(roll),np.sin(roll)
        desired=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])@rotation
        solution=model.solve(point-.08*rotation[:,2]+[0.,0.,height],seed=q,
            position_tolerance_m=.003,orientation_tolerance_rad=.04,
            target_rotation=desired,prefer_seed_solution=True,max_iterations=120)
        print('lift',height,'roll',roll,'q',None if solution is None else solution.positions.tolist())
