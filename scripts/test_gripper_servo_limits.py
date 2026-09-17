import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


class GripperServoLimitsTests(unittest.TestCase):
    def test_gravity_compensation_keeps_force_and_physics_limits(self):
        path = (Path(__file__).resolve().parents[1] / 'src'
                / 'fire_robot_description/urdf/fire_robot_actual_piper.urdf.xacro')
        root = ET.parse(path)
        for joint in ('gripper_left_joint', 'gripper_right_joint'):
            plugin = next(p for p in root.iter('plugin')
                          if p.findtext('joint_name') == joint)
            self.assertEqual(float(plugin.findtext('cmd_max')), 8.0)
            self.assertEqual(float(plugin.findtext('cmd_min')), -8.0)
            self.assertNotEqual(plugin.findtext('use_velocity_commands'), 'true')
            # A 25 g jaw must sag less than the existing 1 mm release tolerance.
            self.assertLess(.025 * 9.81 / float(plugin.findtext('p_gain')), .001)
            self.assertGreater(float(plugin.findtext('d_gain')), 0)


if __name__ == '__main__':
    unittest.main()
