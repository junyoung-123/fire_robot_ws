import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'export_full_mission_video', Path(__file__).with_name('export_full_mission_video.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IntegratedVideoLabelTests(unittest.TestCase):
    def test_success_label_requires_independent_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(module.verified_completion(path, {'pass':False}))
            with self.assertRaises(RuntimeError):
                module.verified_completion(path, {'pass':True})
            (path/'strict_integrated_audit.json').write_text(json.dumps(
                {'pass_all':False,'checks':{'all_three_still_open_at_end':False}}))
            self.assertFalse(module.verified_completion(path, {'pass':True}))
            (path/'strict_integrated_audit.json').write_text(json.dumps(
                {'pass_all':True,'checks':{'all_three_still_open_at_end':True}}))
            self.assertTrue(module.verified_completion(path, {'pass':True}))


if __name__ == '__main__':
    unittest.main()
