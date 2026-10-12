#!/usr/bin/env python3
"""Numerical reach checks on recorded sensor coordinates, not world positions."""
import json
import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_JOINT_LIMITS

kinematics = PiperActualKinematics()
for x in (.80, .75, .70, .67, .64, .60, .56, .52):
    solution = kinematics.solve((x, .0175, .767), max_iterations=100)
    print(json.dumps(dict(x=x, solved=solution is not None,
        joints=solution.positions.tolist() if solution else [],
        ratio=kinematics.translation_condition_ratio(solution.positions) if solution else 0,
        margin=float(np.minimum(solution.positions-PIPER_JOINT_LIMITS[:,0], PIPER_JOINT_LIMITS[:,1]-solution.positions).min()) if solution else 0)))
