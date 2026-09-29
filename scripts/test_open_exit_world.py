import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from make_open_exit_world import generate
from render_obstacle_validation_evidence import WORLD_FILES


class OpenExitTests(unittest.TestCase):
    def test_all_five_keep_interior_models_unchanged(self):
        worlds = Path(__file__).resolve().parents[1]/'src/fire_robot_bringup/worlds'
        with tempfile.TemporaryDirectory() as directory:
            for filename in WORLD_FILES.values():
                source, target = worlds/filename, Path(directory)/filename
                before = source.read_bytes()
                manifest = generate(source, target)
                self.assertEqual(source.read_bytes(), before)
                old = {m.get('name'): ET.tostring(m) for m in ET.parse(source).findall('./world/model')}
                new = {m.get('name'): ET.tostring(m) for m in ET.parse(target).findall('./world/model')}
                for name in old.keys()-{'exit_green', 'handle_exit'}:
                    self.assertEqual(new[name], old[name], name)
                self.assertNotIn('handle_exit', new)
                model = ET.parse(target).find('./world/model[@name="exit_green"]')
                for name in ('jamb_left', 'jamb_right', 'lintel'):
                    visual = model.find(f'./link/visual[@name="{name}"]')
                    collision = model.find(f'./link/collision[@name="{name}"]')
                    self.assertEqual(visual.findtext('pose'), collision.findtext('pose'))
                    self.assertEqual(visual.findtext('geometry/box/size'), collision.findtext('geometry/box/size'))
                self.assertEqual(manifest['changed_models'], ['exit_green', 'handle_exit'])


if __name__ == '__main__':
    unittest.main()
