#!/usr/bin/env python3
"""Offline evidence only: recorded TF/events, never a navigation input."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import Polygon
import yaml

from render_obstacle_validation_evidence import (
    WORLD_FILES, obstacle_footprints, read_csv, render_world)


def rotated(points, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*a-s*b, y+s*a+c*b) for a, b in points]


def polygon_path(points):
    return PlotPath(points + [points[0]], closed=True)


def analyze(trace, obstacles, footprint):
    poses = read_csv(trace / 'poses.csv')
    targets = read_csv(trace / 'targets.csv')
    events = read_csv(trace / 'events.csv')
    states = read_csv(trace / 'states.csv')
    length = 0.0
    max_step = 0.0
    max_dt = 0.0
    overlap = []
    previous = None
    obstacle_paths = [(name, polygon_path(points)) for name, points in obstacles]
    for row in poses:
        x, y, yaw, t = (float(row[k]) for k in ('x', 'y', 'yaw', 't'))
        if not all(math.isfinite(v) for v in (x, y, yaw, t)):
            raise ValueError('Nonfinite recorded pose')
        if previous:
            step = math.hypot(x-previous[0], y-previous[1])
            length += step
            max_step = max(max_step, step)
            max_dt = max(max_dt, t-previous[2])
        previous = (x, y, t)
        shape = polygon_path(rotated(footprint, x, y, yaw))
        for name, obstacle in obstacle_paths:
            if shape.intersects_path(obstacle, filled=True):
                overlap.append(dict(t=t, x=x, y=y, obstacle=name))
    alignments = []
    for event in events:
        if event['event'] != 'DOOR_OPENED':
            continue
        matching = [r for r in targets if r['id'] == event['target_id']
                    and float(r['t']) <= float(event['t'])]
        item = dict(target_id=event['target_id'], t=float(event['t']),
                    x=float(event['x']), y=float(event['y']), yaw=float(event['yaw']))
        starts = [s for s in states if s['state'] == 'OPENING_DOOR'
                  and s['target_id'] == event['target_id']
                  and float(s['t']) <= item['t']]
        if starts:
            start_time = float(starts[-1]['t'])
            samples = [p for p in poses if 0 <= start_time-float(p['t']) <= .7]
            if samples:
                sample = samples[-1]
                item['before_opening'] = {k: float(sample[k]) for k in ('t', 'x', 'y', 'yaw')}
                item['before_opening']['state_stamp'] = start_time
        if matching:
            target = matching[-1]
            tx, ty, yaw = (float(target[k]) for k in ('x', 'y', 'yaw'))
            hx, hy = (float(target[k]) for k in ('handle_x', 'handle_y'))
            delta = math.atan2(math.sin(item['yaw']-yaw), math.cos(item['yaw']-yaw))
            item.update(target_x=tx, target_y=ty, target_yaw=yaw,
                        target_age_sec=float(event['t'])-float(target['t']),
                        parking_error_m=math.hypot(item['x']-tx, item['y']-ty),
                        heading_error_deg=abs(math.degrees(delta)))
            if math.isfinite(hx) and math.isfinite(hy):
                item.update(handle_x=hx, handle_y=hy,
                            base_to_estimated_handle_xy_m=math.hypot(item['x']-hx, item['y']-hy))
            if 'before_opening' in item:
                before = item['before_opening']
                delta = math.atan2(math.sin(before['yaw']-yaw), math.cos(before['yaw']-yaw))
                before.update(parking_error_m=math.hypot(before['x']-tx, before['y']-ty),
                              heading_error_deg=abs(math.degrees(delta)))
                if math.isfinite(hx) and math.isfinite(hy):
                    before['base_to_estimated_handle_xy_m'] = math.hypot(before['x']-hx, before['y']-hy)
        alignments.append(item)
    return dict(pose_samples=len(poses), path_length_m=length,
                max_pose_step_m=max_step, max_sample_gap_sec=max_dt,
                sampled_footprint_overlap_count=len(overlap), overlaps=overlap,
                openings=alignments,
                scope='Saved-map + sim-odom mission; simplified arm/hinge backend. '
                      'Footprint screen is sampled 2D geometry, not a physical contact test. '
                      'Alignment compares the last published navigation goal, not an independently measured door plane.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('validation_dir', type=Path)
    parser.add_argument('--world', type=int, required=True, choices=range(1, 6))
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--output-name', default='mission_evidence')
    parser.add_argument('--fail-on-overlap', action='store_true')
    args = parser.parse_args()
    worlds_dir = args.workspace / 'src/fire_robot_bringup/worlds'
    config = yaml.safe_load((args.workspace / 'src/fire_robot_navigation/config/nav2_params.yaml').read_text())
    footprint = json.loads(config['local_costmap']['local_costmap']['ros__parameters']['footprint'])
    trace = args.validation_dir / f'world{args.world}_trace'
    if Path(args.output_name).name != args.output_name:
        raise ValueError('Output name must be a filename stem')
    output = trace / f'{args.output_name}.png'
    metrics = trace / f'{args.output_name}_metrics.json'
    if output.exists() or metrics.exists():
        raise FileExistsError('Do not overwrite an existing evidence result')
    obstacles = obstacle_footprints(worlds_dir / WORLD_FILES[args.world])
    report = analyze(trace, obstacles, footprint)
    metrics.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fig, (ax, lower) = plt.subplots(2, 1, figsize=(16, 8), dpi=160,
                                  gridspec_kw={'height_ratios': [3, 1.5]})
    render_world(ax, args.world, args.validation_dir, worlds_dir)
    ax.set_ylim(-2.8, 2.8)
    for i, item in enumerate(report['openings'], 1):
        pose = item.get('before_opening', item)
        ax.add_patch(Polygon(rotated(footprint, pose['x'], pose['y'], pose['yaw']),
                             fill=False, edgecolor='#007a70', linewidth=1.4, zorder=8))
        ax.annotate(f'open {i}', (pose['x'], pose['y']), xytext=(0, 15),
                    textcoords='offset points', ha='center', fontsize=9)
        ax.arrow(pose['x'], pose['y'], .5*math.cos(pose['yaw']), .5*math.sin(pose['yaw']),
                 width=.02, color='#007a70', zorder=8)
    if report['overlaps']:
        ax.scatter([r['x'] for r in report['overlaps']], [r['y'] for r in report['overlaps']],
                   color='red', s=12, label='sampled footprint overlap', zorder=9)
        ax.legend(fontsize=7, ncol=3, loc='upper left')
    lower.axis('off')
    rows = []
    for i, item in enumerate(report['openings'], 1):
        before = item.get('before_opening', {})
        rows.append([str(i), item['target_id'],
                     f"{before['parking_error_m']:.3f} / {item['parking_error_m']:.3f}"
                     if 'parking_error_m' in before else 'unavailable',
                     f"{before['heading_error_deg']:.1f} / {item['heading_error_deg']:.1f}"
                     if 'heading_error_deg' in before else 'unavailable',
                     f"{before['base_to_estimated_handle_xy_m']:.3f}"
                     if 'base_to_estimated_handle_xy_m' in before else 'unavailable'])
    if rows:
        table = lower.table(cellText=rows,
                            colLabels=['Open', 'Observed target ID', 'Goal error [m]\nbefore / after',
                                       'Heading [deg]\nbefore / after', 'Before: base to estimated handle [m]'],
                            colWidths=[.05, .38, .15, .17, .25], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.65)
        for column in range(5):
            table[(0, column)].set_height(table[(0, column)].get_height()*1.55)
    else:
        lower.text(.5, .5, 'No DOOR_OPENED events recorded.', ha='center')
    fig.suptitle(f"World {args.world}: recorded mission evidence", fontsize=18)
    fig.text(.07, .15,
             f"Path {report['path_length_m']:.2f} m | pose samples {report['pose_samples']} | "
             f"sampled obstacle overlaps {report['sampled_footprint_overlap_count']} | "
             f"largest pose step {report['max_pose_step_m']:.2f} m", fontsize=11)
    fig.text(.07, .035,
             'Saved map + sim odometry + simplified arm/hinge backend. Not physical arm-contact proof.\n'
             'Obstacles are offline Gazebo ground truth. 2D footprint sampling does not prove absence of collisions.\n'
             'Box/arrow: before opening; star: after opening and recovery. Goal errors use the last published navigation goal.\n'
             'Estimated handle distance is not a wall clearance or a proof of arm reachability.',
             fontsize=10)
    fig.tight_layout(rect=(0, .21, 1, .96))
    fig.savefig(output)
    plt.close(fig)
    print(output)
    print(json.dumps({k: v for k, v in report.items() if k not in ('overlaps', 'scope')}))
    return 1 if args.fail_on_overlap and report['sampled_footprint_overlap_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
