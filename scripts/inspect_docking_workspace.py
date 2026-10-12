import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_LIMITS

arm = PiperActualKinematics()
for z in (.73,.76,.80):
    for y in (.0175,.07,.10,.145):
        available=[]
        for x in np.linspace(.4628,.695,17):
            solution=arm.solve([x,y,z], max_iterations=100, position_tolerance_m=.003)
            if solution is None:
                continue
            margin=float(np.min(np.minimum(solution.positions-PIPER_JOINT_LIMITS[:,0],
                                            PIPER_JOINT_LIMITS[:,1]-solution.positions)))
            available.append((round(x,3),round(margin,3),
                              round(arm.translation_condition_ratio(solution.positions),3)))
        print(z,y,available)
