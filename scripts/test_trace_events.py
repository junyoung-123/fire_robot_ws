import csv
import tempfile
import unittest
from pathlib import Path

from check_trace_events import check


class TraceEventsTests(unittest.TestCase):
    def run_check(self, rows, expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['event', 'target_id', 'x', 'y', 'yaw'])
                writer.writeheader()
                for event, target, x in rows:
                    writer.writerow(dict(event=event, target_id=target, x=x, y=0, yaw=0))
            return check(path, expected)[0]

    def test_unique_openings_and_completion(self):
        self.assertTrue(self.run_check([
            ('DOOR_OPENED', 'observed_a', 1), ('DOOR_OPENED', 'observed_b', 2),
            ('MISSION_COMPLETE', '', 3)], 2))

    def test_no_blue_world_still_requires_completion(self):
        self.assertTrue(self.run_check([('MISSION_COMPLETE', '', 3)], 0))
        self.assertFalse(self.run_check([], 0))

    def test_success_log_cannot_replace_missing_open_event(self):
        self.assertFalse(self.run_check([('MISSION_COMPLETE', '', 3)], 1))

    def test_duplicate_and_empty_ids_fail(self):
        self.assertFalse(self.run_check([
            ('DOOR_OPENED', 'same', 1), ('DOOR_OPENED', 'same', 2),
            ('MISSION_COMPLETE', '', 3)], 2))
        self.assertFalse(self.run_check([
            ('DOOR_OPENED', '', 1), ('MISSION_COMPLETE', '', 3)], 1))

    def test_nonfinite_evidence_pose_fails(self):
        self.assertFalse(self.run_check([
            ('DOOR_OPENED', 'observed_a', 'nan'), ('MISSION_COMPLETE', '', 3)], 1))

    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(check(Path(directory) / 'missing.csv', 0)[0])


if __name__ == '__main__':
    unittest.main()
