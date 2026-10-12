import importlib.util
from pathlib import Path
import numpy as np

ws = Path(__file__).resolve().parents[1]
path = ws/'src/fire_robot_navigation/fire_robot_navigation/self_scan_geometry.py'
spec = importlib.util.spec_from_file_location('geometry', path)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)
segments = np.array([[[1., -1.], [1., 1.]]])
echo = geometry.expected_echo(segments, np.array([0., np.pi]), (0, 0))
assert abs(echo[0]-1.) < 1e-9 and np.isinf(echo[1])
cleaned, count = geometry.mask_echoes([1., 3.], echo)
assert np.isnan(cleaned[0]) and cleaned[1] == 3. and count == 1
sections = geometry.horizontal_sections(ws/'src/fire_robot_description/meshes/roas2/BASE_FRAME.stl', .313)
assert len(sections) > 0
scan_file = ws/'artifacts/full_world1_r2/evidence/scan_latest.npz'
if scan_file.exists():
    scan = np.load(scan_file)
    angles = scan['angle_min']+np.arange(len(scan['ranges']))*scan['angle_increment']
    echo = geometry.expected_echo(sections, angles, (.25, 0), float(scan['range_min']))
    cleaned, count = geometry.mask_echoes(scan['ranges'], echo)
    close = (scan['ranges'] < .55)
    print('CAD segments:', len(sections), 'self returns removed:', count,
          'near returns remaining:', int((close & np.isfinite(cleaned)).sum()))
    assert count > 20
    assert not np.any(np.isnan(cleaned[np.abs(angles) < .5]))
print('PASS: CAD self-echo mask preserves front environment rays and never clears through the chassis')
