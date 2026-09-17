import unittest
import numpy as np

from fire_robot_manipulation.piper_actual_kinematics import PiperActualKinematics


class ContactWorkspaceConditionTests(unittest.TestCase):
    def test_reachable_is_not_necessarily_a_good_contact_pose(self):
        kin = PiperActualKinematics()
        stretched = kin.solve([.87, .15, .77], position_tolerance_m=.003)
        working = kin.solve([.78, .15, .77], position_tolerance_m=.003)
        self.assertIsNotNone(stretched)
        self.assertIsNotNone(working)
        self.assertLess(kin.translation_condition_ratio(stretched.positions), .15)
        self.assertGreater(kin.translation_condition_ratio(working.positions), .15)

    def test_condition_uses_robot_chain_not_a_world_origin(self):
        kin = PiperActualKinematics()
        shifted = PiperActualKinematics(mount_xyz=kin.mount_xyz + [12., -5., 2.])
        joints = np.array([.1, 2., -1.4, .1, -.4, 1.5])
        self.assertAlmostEqual(kin.translation_condition_ratio(joints),
                               shifted.translation_condition_ratio(joints), places=8)

    def test_nonfinite_configuration_cannot_qualify(self):
        self.assertEqual(PiperActualKinematics().translation_condition_ratio([float('nan')]*6), 0.)


if __name__ == '__main__':
    unittest.main()
