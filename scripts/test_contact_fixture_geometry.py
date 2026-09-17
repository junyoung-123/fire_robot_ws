import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


WORLD = (Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/worlds'
         / 'physical_contact_door_test.world')


def numbers(element, path):
    return [float(value) for value in element.findtext(path).split()]


class ContactFixtureGeometryTests(unittest.TestCase):
    def setUp(self):
        self.model = ET.parse(WORLD).find(".//model[@name='blue_hinged_contact_door']")

    def test_leaf_clears_floor_without_moving_hinge(self):
        panel = self.model.find("link[@name='panel']")
        pose = numbers(panel, 'pose')
        size = numbers(panel, "collision[@name='panel_collision']/geometry/box/size")
        self.assertGreaterEqual(pose[2] - size[2] / 2, .005)
        self.assertEqual(numbers(self.model, "joint[@name='door_hinge']/pose"),
                         [0, -.4, 0, 0, 0, 0])
        self.assertEqual(numbers(panel, "visual[@name='panel_visual']/geometry/box/size"), size)

    def test_latch_stays_inside_panel_thickness(self):
        lever = self.model.find("link[@name='lever']")
        bolt = lever.find("collision[@name='latch_bolt_collision']")
        center = numbers(lever, 'pose')[0] + numbers(bolt, 'pose')[0]
        half = numbers(bolt, 'geometry/box/size')[0] / 2
        self.assertGreaterEqual(center - half, -.03)
        self.assertLessEqual(center + half, .03)
        # Preserve the surface facing the strike, not an easier opening gap.
        self.assertAlmostEqual(center + half, .025)
        visual = lever.find("visual[@name='latch_bolt_visual']")
        self.assertEqual(numbers(bolt, 'pose'), numbers(visual, 'pose'))
        self.assertEqual(numbers(bolt, 'geometry/box/size'),
                         numbers(visual, 'geometry/box/size'))

    def test_latch_and_strike_still_exist(self):
        strike = self.model.find("link[@name='frame']/collision[@name='latch_strike_collision']")
        self.assertIsNotNone(strike)
        self.assertEqual(numbers(strike, 'pose'), [.055, .84, .83, 0, 0, 0])
        self.assertEqual(numbers(strike, 'geometry/box/size'), [.04, .08, .05])
        self.assertEqual(self.model.findtext('self_collide'), 'true')


if __name__ == '__main__':
    unittest.main()
