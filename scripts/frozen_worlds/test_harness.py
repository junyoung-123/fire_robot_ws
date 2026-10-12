"""Offline checks of fixture adaptation; these are not simulation PASS results."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
from build_world import build, WORKSPACE


class HarnessTests(unittest.TestCase):
    def test_world1_geometry_reproduces_preserved_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'world1.world'
            build(1, path)
            original = WORKSPACE / 'src/fire_robot_bringup/worlds/world1_full_contact.world'
            self.assertEqual(ET.canonicalize(from_file=original, strip_text=True),
                             ET.canonicalize(from_file=path, strip_text=True))

    def test_all_layouts_preserve_obstacles_and_use_same_contact_fixture(self):
        for number, expected in ((2, 3), (3, 4), (4, 0), (5, 6)):
            with self.subTest(world=number), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'world.world'
                report = build(number, path)
                self.assertEqual(len(report['blue_models']), expected)
                self.assertTrue(report['construction_clearance_audit']['pass_clearance'])
                self.assertTrue(report['unchanged_models'])
                self.assertEqual(report['scope'], 'evaluation_only_never_loaded_by_control')


if __name__ == '__main__':
    unittest.main()
