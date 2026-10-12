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
            # Gazebo otherwise supplies a nonzero integral default. A blocked
            # grasp must not wind up and oppose a later open command.
            self.assertEqual(float(plugin.findtext('i_gain')), 0.)
            # A 25 g jaw must sag less than the existing 1 mm release tolerance.
            self.assertLess(.025 * 9.81 / float(plugin.findtext('p_gain')), .001)
            self.assertGreater(float(plugin.findtext('d_gain')), 0)

    def test_loaded_grasp_bias_and_free_space_damping_reserve(self):
        path = (Path(__file__).resolve().parents[1] / 'src'
                / 'fire_robot_description/urdf/fire_robot_actual_piper.urdf.xacro')
        for plugin in ET.parse(path).iter('plugin'):
            if plugin.findtext('joint_name') not in ('gripper_left_joint', 'gripper_right_joint'):
                continue
            p, d = (float(plugin.findtext(name)) for name in ('p_gain', 'd_gain'))
            # 0.7 N covers the R193 measured 1.735 mm bias at P=400;
            # the unchanged 34 mm release criterion leaves 1 mm reserve.
            self.assertLess(.7/p, .001)
            self.assertGreaterEqual(d/(2*(.025*p)**.5), 1.)


if __name__ == '__main__':
    unittest.main()
