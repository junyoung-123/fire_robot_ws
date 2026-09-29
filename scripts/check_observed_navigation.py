#!/usr/bin/env python3
"""Independent post-run evaluator; SDF and simulator pose never go to control."""
import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Polygon
from render_obstacle_validation_evidence import obstacle_footprints


def read_csv(path, keys):
    with path.open() as stream:
        return np.array([[float(row[key]) for key in keys] for row in csv.DictReader(stream)])


def exit_jamb_footprints(root):
    model = root.find('./world/model[@name="exit_green"]')
    if model is None:
        return []
    mx, my, _, _, _, yaw = map(float, model.findtext('pose', '0 0 0 0 0 0').split())
    footprints = []
    for part in model.findall('./link/collision'):
        if not part.get('name', '').startswith('jamb_'):
            continue
        px, py, _, _, _, pyaw = map(float, part.findtext('pose').split())
        sx, sy, _ = map(float, part.findtext('geometry/box/size').split())
        c, s = math.cos(yaw), math.sin(yaw)
        center = np.array([mx+c*px-s*py, my+s*px+c*py])
        c, s = math.cos(yaw+pyaw), math.sin(yaw+pyaw)
        local = np.array([[-sx/2, -sy/2], [sx/2, -sy/2], [sx/2, sy/2], [-sx/2, sy/2]])
        footprints.append(('exit_'+part.get('name'), local @ np.array([[c, s], [-s, c]])+center))
    return footprints


def structure_footprints(root):
    """Static low-height wall/panel boxes, used only by the offline evaluator."""
    def pose(element):
        return np.asarray(list(map(float, element.findtext('pose', '0 0 0 0 0 0').split())))

    def compose(parent, child):
        c, s = math.cos(parent[5]), math.sin(parent[5])
        result = parent + child
        result[:2] = parent[:2] + np.array([[c, -s], [s, c]]) @ child[:2]
        return result

    footprints = []
    for model in root.findall('./world/model'):
        if not model.get('name', '').startswith(('wall_', 'door_')):
            continue
        for link in model.findall('link'):
            base = compose(pose(model), pose(link))
            for index, collision in enumerate(link.findall('collision')):
                box = collision.find('geometry/box/size')
                if box is None:
                    continue
                p = compose(base, pose(collision))
                sx, sy, sz = map(float, box.text.split())
                # Do not project overhead lintels down onto the mobile base.
                if p[2]-sz/2 > .50 or p[2]+sz/2 < .02:
                    continue
                if abs(p[3])+abs(p[4]) > 1e-5:
                    raise ValueError('Non-planar structure requires 3D collision evaluation')
                local = np.array([[-sx/2, -sy/2], [sx/2, -sy/2], [sx/2, sy/2], [-sx/2, sy/2]])
                c, s = math.cos(p[5]), math.sin(p[5])
                name = model.get('name')+'/'+collision.get('name', str(index))
                footprints.append((name, local @ np.array([[c, s], [-s, c]])+p[:2]))
    return footprints


def evaluate(directory, world):
    evidence = directory/'evidence'
    result = json.loads((directory/'result.json').read_text())
    events = [json.loads(line) for line in (evidence/'events.jsonl').read_text().splitlines()]
    visits = [event['payload'] for event in events if event['kind'] == 'visit']
    root = ET.parse(world).getroot()
    doors = []
    for model in root.findall('./world/model'):
        name = model.get('name', '')
        if not (name.startswith('door_') or name == 'exit_green'):
            continue
        pose = [float(v) for v in model.findtext('pose', '0 0 0 0 0 0').split()]
        color = next((color for color in ('blue', 'red', 'green') if color in name), None)
        if color:
            doors.append({'name': name, 'color': color, 'xy': pose[:2]})
    failures = []
    audit_path = evidence/'input_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    graph = json.loads((evidence/'ros_graph.json').read_text())
    forbidden = {'map_server', 'amcl', 'sim_odom_map_tf', 'initial_static_map_node',
                 'fixed_obstacle_map_node', 'manipulation_node'}
    if any(node['node'] in forbidden for node in graph):
        failures.append('forbidden prior-map/manipulation node')
    if audit.get('map_publishers') != ['slam_toolbox']:
        failures.append('live SLAM map publisher not established')
    if audit.get('odom_publishers') != ['gz_selected_odometry_bridge']:
        failures.append('wheel odometry publisher not established')
    if audit.get('evaluation_truth_subscribers') != ['observed_navigation_recorder']:
        failures.append('evaluation truth isolation not established')
    truth_path = evidence/'evaluation_poses.csv'
    truth = read_csv(truth_path, ('sim_time', 'x', 'y', 'yaw')) if truth_path.exists() else np.empty((0, 4))
    matches = []
    if not len(truth):
        failures.append('missing independent simulator-pose evaluation')
    else:
        for visit in visits:
            row = truth[np.argmin(np.abs(truth[:, 0]-visit['sim_time']))]
            door = min(doors, key=lambda d: np.linalg.norm(row[1:3]-d['xy']))
            dx, dy = np.array(door['xy'])-row[1:3]
            distance = math.hypot(dx, dy)
            bearing_error = abs(math.atan2(math.sin(math.atan2(dy, dx)-row[3]),
                                           math.cos(math.atan2(dy, dx)-row[3])))
            # A visit must be beside the correct panel and face its wall normal.
            longitudinal_error = abs(dx)
            normal_yaw = math.copysign(math.pi/2, dy)
            normal_error = abs(math.atan2(math.sin(normal_yaw-row[3]), math.cos(normal_yaw-row[3])))
            ok = (door['color'] == 'blue' and longitudinal_error <= .70
                  and distance <= 1.60 and normal_error <= math.radians(15))
            matches.append({'id': visit['door_id'], 'matched_door': door['name'],
                            'color': door['color'], 'distance_m': distance,
                            'longitudinal_error_m': longitudinal_error,
                            'normal_error_deg': math.degrees(normal_error),
                            'bearing_error_deg': math.degrees(bearing_error), 'pass': bool(ok)})
            if not ok:
                failures.append('visit alignment/color mismatch: '+door['name'])
    expected = {door['name'] for door in doors if door['color'] == 'blue'}
    reached = {match['matched_door'] for match in matches if match['pass']}
    if reached != expected:
        failures.append('unvisited blue doors: '+str(sorted(expected-reached)))
    if len(matches) != len(reached):
        failures.append('duplicate or invalid visits')
    exits = [door for door in doors if door['color'] == 'green']
    if len(truth) and exits:
        exit_door = exits[0]
        crossed = truth[-1, 1] > exit_door['xy'][0] + .20 and abs(truth[-1, 2]-exit_door['xy'][1]) < 1.0
        if not crossed:
            failures.append('exit crossing not confirmed by independent pose')
    else:
        failures.append('exit crossing evaluation unavailable')
    if result['status'] != 'MISSION_COMPLETE':
        failures.append('mission terminal state: '+result['status'])
    footprints = obstacle_footprints(world)
    retained_obstacle_count = len(footprints)
    jambs = exit_jamb_footprints(root)
    footprints += jambs
    structures = structure_footprints(root)
    footprints += structures
    overlap_names = set()
    overlap_events = {}
    if len(truth):
        shapes = [(name, PolygonPath(np.vstack([corners, corners[0]]), closed=True))
                  for name, corners in footprints]
        local = np.array([[-.26, -.23], [.26, -.23], [.26, .23], [-.26, .23]])
        for row in truth:
            c, s = math.cos(row[3]), math.sin(row[3])
            corners = local @ np.array([[c, s], [-s, c]]) + row[1:3]
            robot = PolygonPath(np.vstack([corners, corners[0]]), closed=True)
            for name, obstacle in shapes:
                if robot.intersects_path(obstacle, filled=True):
                    overlap_names.add(name)
                    event = overlap_events.setdefault(name, {
                        'first_sim_time': float(row[0]), 'first_pose': row[1:].tolist(),
                        'sample_count': 0})
                    event['last_sim_time'] = float(row[0])
                    event['sample_count'] += 1
        if overlap_names:
            failures.append('navigation-footprint/obstacle overlap: '+str(sorted(overlap_names)))
        fig, ax = plt.subplots(figsize=(14, 4))
        for name, corners in footprints:
            ax.add_patch(Polygon(corners, facecolor='#aaaaaa', edgecolor='#444444', alpha=.7))
        for door in doors:
            ax.plot(*door['xy'], 's', color=door['color'], ms=9)
            ax.annotate(door['name'], door['xy'], xytext=(0, 9), textcoords='offset points',
                        ha='center', fontsize=7)
        ax.plot(truth[:, 1], truth[:, 2], color='#b95d00', lw=1.5, label='Simulator pose (evaluation only)')
        for visit in visits:
            row = truth[np.argmin(np.abs(truth[:, 0]-visit['sim_time']))]
            ax.plot(row[1], row[2], '*', color='#111111', ms=12)
            ax.arrow(row[1], row[2], .4*math.cos(row[3]), .4*math.sin(row[3]),
                     head_width=.09, color='#111111')
        ax.set_aspect('equal')
        ax.autoscale()
        ax.margins(x=.04, y=.22)
        ax.set_xlabel('Gazebo world x [m]')
        ax.set_ylabel('Gazebo world y [m]')
        ax.set_title(f'{directory.name} | Obstacles retained, NOT supplied to control | Stars: visits, not opening', pad=18)
        ax.legend(loc='lower right', fontsize=8)
        fig.tight_layout()
        fig.savefig(directory/'evaluation_world_trajectory.png', dpi=150)
        plt.close(fig)
        if visits:
            fig, axes = plt.subplots(1, len(visits), figsize=(5*len(visits), 5), squeeze=False)
            for ax, visit, match in zip(axes[0], visits, matches):
                row = truth[np.argmin(np.abs(truth[:, 0]-visit['sim_time']))]
                door = next(d for d in doors if d['name'] == match['matched_door'])
                dx, dy = door['xy']
                ax.plot([dx-.425, dx+.425], [dy, dy], lw=7, color=door['color'])
                c, s = math.cos(row[3]), math.sin(row[3])
                body = local @ np.array([[c, s], [-s, c]]) + row[1:3]
                ax.add_patch(Polygon(body, facecolor='#aaaaaa', edgecolor='#333333'))
                ax.arrow(row[1], row[2], .45*c, .45*s, head_width=.08, color='#cc6400')
                ax.plot([row[1], dx], [row[2], dy], '--', color='#777777', lw=1)
                ax.set_title(f"{door['name']} | navigation visit\n"
                             f"distance={match['distance_m']:.2f} m | yaw error={match['normal_error_deg']:.1f} deg")
                ax.set_xlim(dx-1.4, dx+1.4)
                ax.set_ylim(min(dy, row[2])-.45, max(dy, row[2])+.45)
                ax.set_aspect('equal')
                ax.set_xlabel('Gazebo world x [m]')
                ax.set_ylabel('Gazebo world y [m]')
                ax.grid(alpha=.2)
            fig.suptitle('Independent visit-position evaluation. NOT arm-reach / physical-opening validation.')
            fig.tight_layout()
            fig.savefig(directory/'door_visit_alignment.png', dpi=150)
            plt.close(fig)
    # Contact sensors are not present in these legacy worlds: never claim physical collision-free.
    report = {'status': 'PASS_NAV_VISITS_EXIT' if not failures else 'FAIL',
              'failures': failures, 'expected_blue_count_evaluator_only': len(expected),
              'unique_valid_visits': len(reached), 'matches': matches, 'input_audit': audit,
              'physical_opening_tested': False, 'physical_contact_collision_audit': 'not instrumented'}
    report['navigation_footprint_overlap_obstacles'] = sorted(overlap_names)
    report['navigation_footprint_overlap_events'] = overlap_events
    report['retained_obstacles_evaluator_only'] = retained_obstacle_count
    report['exit_jambs_evaluated'] = len(jambs)
    report['wall_panel_boxes_evaluated'] = len(structures)
    (directory/'independent_check.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--world-file', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.directory, args.world_file) else 1)
