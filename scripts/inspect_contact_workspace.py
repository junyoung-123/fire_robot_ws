import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_LIMITS
from fire_robot_manipulation.contact_feedback import measured_wrist_rotation
arm = PiperActualKinematics()
seed = np.array([0.,1.8,-1.2,0.,-.65,1.5])
for angle in (0.,-.3,-.6,-.8,-1.):
    for x in (.45,.503,.55,.6):
        for y in (-.2,-.15,-.1,0.,.15,.2):
            rot = measured_wrist_rotation(arm.approach_rotation,angle,0.,-.3)
            sol = arm.solve([x,y,.73], seed=seed, target_rotation=rot,
                            position_tolerance_m=.003, prefer_seed_solution=True)
            if sol is None: continue
            margin = float(np.min(np.minimum(sol.positions-PIPER_JOINT_LIMITS[:,0],
                                             PIPER_JOINT_LIMITS[:,1]-sol.positions)))
            if margin >= .15:
                print(angle,x,y,round(margin,3),round(arm.translation_condition_ratio(sol.positions),3))
