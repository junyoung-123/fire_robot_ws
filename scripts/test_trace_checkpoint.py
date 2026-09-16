import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from validation_trace_logger import ValidationTraceLogger


class TraceCheckpointTests(unittest.TestCase):
    def test_checkpoint_appends_without_duplicate_header_or_rows(self):
        with TemporaryDirectory() as folder:
            node = SimpleNamespace(output_dir=Path(folder), _persisted_rows={})
            rows = [{'t': 1.0, 'x': 2.0}]
            for _ in range(2):
                ValidationTraceLogger.write_csv(node, 'poses.csv', rows, ['t', 'x'])
            rows.append({'t': 2.0, 'x': 3.0})
            ValidationTraceLogger.write_csv(node, 'poses.csv', rows, ['t', 'x'])
            with (Path(folder)/'poses.csv').open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 2)
            self.assertEqual([r['x'] for r in saved], ['2.0', '3.0'])

    def test_empty_stream_gets_header_before_first_observation(self):
        with TemporaryDirectory() as folder:
            node = SimpleNamespace(output_dir=Path(folder), _persisted_rows={})
            ValidationTraceLogger.write_csv(node, 'events.csv', [], ['event'])
            ValidationTraceLogger.write_csv(node, 'events.csv', [{'event':'DOOR_OPENED'}], ['event'])
            self.assertEqual((Path(folder)/'events.csv').read_text().splitlines(),
                             ['event', 'DOOR_OPENED'])


if __name__ == '__main__':
    unittest.main()
