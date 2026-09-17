import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class ContactFrameGeometryTests(unittest.TestCase):
    def test_visible_jamb_is_physical_and_clear_of_door_sweep(self):
        world = Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/worlds/physical_contact_door_test.world'
        model = ET.parse(world).find(".//model[@name='blue_hinged_contact_door']")
        frame = model.find("link[@name='frame']")
        collision = frame.find("collision[@name='hinge_post_collision']")
        visual = frame.find("visual[@name='hinge_post']")
        self.assertIsNotNone(collision)
        self.assertEqual(collision.findtext('pose'), visual.findtext('pose'))
        self.assertEqual(collision.findtext('geometry/box/size'), visual.findtext('geometry/box/size'))
        center = np.fromstring(collision.findtext('pose'), sep=' ')[:2]
        half = np.fromstring(collision.findtext('geometry/box/size'), sep=' ')[:2] / 2.
        post = np.array([center + half * [x, y] for x, y in ((-1,-1),(-1,1),(1,1),(1,-1))])
        # Separating-axis check includes the finite leaf thickness, not only its centerline.
        leaf = np.array([[-.03,0.],[-.03,.8],[.03,.8],[.03,0.]])
        for angle in np.linspace(0., -2.05, 150):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c,-s],[s,c]])
            rotated = leaf @ rotation.T
            axes = np.vstack((np.eye(2), rotation.T))
            separated = any((rotated@axis).max() <= (post@axis).min()+1e-10
                            or (post@axis).max() <= (rotated@axis).min()+1e-10
                            for axis in axes)
            self.assertTrue(separated, f'Jamb overlaps leaf at {angle} rad')


if __name__ == '__main__':
    unittest.main()
