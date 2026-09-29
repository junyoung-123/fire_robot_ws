"""Offline replay of recorded scan/target pairs; no live control outputs."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/fire_robot_fsm'))
from fire_robot_fsm.observed_wall_normal import fit_wall_normal


def main(directory, stop):
    events = [json.loads(line) for line in (directory/'evidence/events.jsonl').read_text().splitlines()]
    targets = [e for e in events if e['kind'] == 'target' and e['color'] == 'blue']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plotted = set()
    for file in sorted((directory/'evidence').glob('scan_0*.npz')):
        scan = np.load(file)
        t = float(scan['sim_time'])
        if t > stop:
            break
        previous = [e for e in targets if e['sim_time'] <= t]
        if not previous or 'sensor_pose' not in scan:
            continue
        target = previous[-1]
        x, y, yaw = scan['sensor_pose']
        dx, dy = target['handle'][0]-x, target['handle'][1]-y
        c, s = math.cos(yaw), math.sin(yaw)
        anchor = (c*dx+s*dy, -s*dx+c*dy)
        normal = fit_wall_normal(scan['ranges'], float(scan['angle_min']), float(scan['angle_increment']),
                                 float(scan['range_min']), float(scan['range_max']), anchor)
        print(json.dumps(dict(t=round(t, 1), target=target['id'], sensor=[x,y,yaw],
                              anchor=anchor, map_normal=None if normal is None else math.degrees(normal+yaw))))
        for slot, when in enumerate((50, 86, 136)):
            if slot in plotted or abs(t-when) > 2:
                continue
            plotted.add(slot)
            r = scan['ranges']
            angles = float(scan['angle_min'])+np.arange(len(r))*float(scan['angle_increment'])
            valid = np.isfinite(r) & (r > float(scan['range_min'])) & (r < float(scan['range_max']))
            points = np.column_stack((r[valid]*np.cos(angles[valid]), r[valid]*np.sin(angles[valid])))
            near = np.linalg.norm(points-np.asarray(anchor), axis=1) < .9
            mapped = points @ np.array([[c, s], [-s, c]])+np.array([x,y])
            axes[slot].scatter(*mapped.T, s=5, color='#9ba3ab')
            axes[slot].scatter(*mapped[near].T, s=9, color='#1678ac')
            axes[slot].plot(*target['handle'], 'x', color='#c14818', ms=12)
            axes[slot].plot(x, y, '^', color='black')
            axes[slot].set(xlim=(0, 5), ylim=(-.5, 3.5), aspect='equal',
                           title=f't={t:.1f}: normal={normal}', xlabel='map x [m]', ylabel='map y [m]')
    fig.tight_layout()
    fig.savefig(directory/'wall_normal_diagnosis.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--until', type=float, default=180)
    args = parser.parse_args()
    main(args.directory, args.until)
