#!/usr/bin/env python3
"""Check intermediate IK from robot stow to a recorded live pre-grasp point."""
import numpy as np
from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics, PIPER_STOW


def main():
    model = PiperActualKinematics()
    start, _ = model.forward(PIPER_STOW)
    goal = np.array([.69-.071, model.mount_xyz[1], .767])
    seed = PIPER_STOW.copy()
    for fraction in np.linspace(0., 1., 25)[1:]:
        target = start + fraction * (goal-start)
        result = model.solve(target, seed=seed, position_tolerance_m=.003)
        assert result is not None, (fraction, target)
        seed = result.positions
    print('PASS: all intermediate IK points in the measured arm-aligned approach')


if __name__ == '__main__':
    main()
