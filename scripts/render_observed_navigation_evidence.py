#!/usr/bin/env python3
"""Plot recorded sensor maps and robot poses; never fabricate an observed map."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def render(directory):
    evidence = directory/'evidence'
    maps = sorted(evidence.glob('map_*.npz'))
    with (evidence/'poses.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    poses = np.array([[float(row[key]) for key in ('sim_time', 'x', 'y')] for row in rows])
    events = [json.loads(line) for line in (evidence/'events.jsonl').read_text().splitlines() if line]
    visits = [event['payload'] for event in events if event['kind'] == 'visit']
    result_path = directory/'result.json'
    status = json.loads(result_path.read_text())['status'] if result_path.exists() else 'IN PROGRESS'
    if not maps:
        return
    indices = sorted(set([0, len(maps)//2, len(maps)-1]))
    latest = np.load(maps[-1])
    last_origin = latest['origin']
    last_res = float(latest['resolution'])
    limits = [last_origin[0], last_origin[0]+latest['data'].shape[1]*last_res,
              last_origin[1], last_origin[1]+latest['data'].shape[0]*last_res]

    def draw_map(ax, index):
        saved = np.load(maps[index])
        grid, origin, res, t = saved['data'], saved['origin'], float(saved['resolution']), float(saved['sim_time'])
        pixels = np.where(grid < 0, 0, np.where(grid >= 65, 2, 1))
        ax.imshow(pixels, origin='lower', cmap=ListedColormap(['#cbd0d5', '#ffffff', '#252c32']),
                  vmin=0, vmax=2, extent=[origin[0], origin[0]+grid.shape[1]*res,
                                         origin[1], origin[1]+grid.shape[0]*res], interpolation='nearest')
        if poses.size:
            path = poses[poses[:, 0] <= t]
            if len(path):
                ax.plot(path[:, 1], path[:, 2], color='#cc6400', lw=1.6, label='Sensor-localized trajectory')
                ax.plot(path[-1, 1], path[-1, 2], 'o', color='#cc6400', ms=5)
        for visit in visits:
            if visit['sim_time'] <= t:
                ax.plot(*visit['robot'][:2], '*', ms=13, color='#0871c1')
        ax.set_aspect('equal')
        ax.set_facecolor('#cbd0d5')
        ax.set_xlim(limits[:2])
        ax.set_ylim(limits[2:])
        ax.set_title(f'Live SLAM | t={t:.1f} s | known={(grid>=0).sum()} cells', fontsize=11)
        ax.set_xlabel('map x [m]')
        ax.set_ylabel('map y [m]')

    fig, axes = plt.subplots(len(indices), 1, figsize=(14, 3.7*len(indices)), squeeze=False,
                             layout='constrained')
    fig.set_constrained_layout_pads(h_pad=.15, hspace=.08)
    for ax, index in zip(axes[:, 0], indices):
        draw_map(ax, index)
    fig.suptitle(f'{directory.name} | {status} | No prior map / wheel odometry\n'
                 f'Gray: unknown, white: observed free, black: observed occupied. '
                 f'Blue stars: {len(visits)} visits (not physical opening)', fontsize=12)
    fig.savefig(directory/'observed_map_progress.png', dpi=150)
    plt.close(fig)
    for label, index in zip(('initial', 'middle', 'latest'), indices):
        fig, ax = plt.subplots(figsize=(14, 4))
        draw_map(ax, index)
        fig.suptitle(f'{directory.name} | {label} sensor-built map | No saved map')
        fig.tight_layout()
        fig.savefig(directory/f'observed_map_{label}.png', dpi=150)
        plt.close(fig)
    cameras = [event for event in events if event['kind'] == 'camera']
    if cameras:
        selected = sorted(set([0, len(cameras)//2, len(cameras)-1]))
        fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 4), squeeze=False)
        for ax, index in zip(axes[0], selected):
            camera = cameras[index]
            ax.imshow(plt.imread(evidence/camera['file']))
            ax.set_title(f"Front camera | t={camera['sim_time']:.1f} s", fontsize=11)
            ax.axis('off')
        fig.suptitle(f'{directory.name} | Recorded camera images, no synthetic overlays')
        fig.tight_layout()
        fig.savefig(directory/'camera_observations.png', dpi=150)
        plt.close(fig)
    graph_file = evidence/'ros_graph.json'
    graph = json.loads(graph_file.read_text()) if graph_file.exists() else []
    names = {entry['node'] for entry in graph}
    forbidden = {'map_server', 'amcl', 'sim_odom_map_tf', 'initial_static_map_node',
                 'fixed_obstacle_map_node', 'manipulation_node'}
    summary = {'status': status, 'visit_count': len(visits), 'snapshots': len(maps),
               'forbidden_nodes_present': sorted(names & forbidden),
               'slam_node_present': any('slam_toolbox' in name for name in names),
               'visits_are_not_door_openings': True}
    (directory/'evidence_summary.json').write_text(json.dumps(summary, indent=2))
    costmap_path = evidence/'costmap_latest.npz'
    scan_path = evidence/'scan_latest.npz'
    if costmap_path.exists() and scan_path.exists():
        costmap = np.load(costmap_path)
        scan = np.load(scan_path)
        grid, origin, res = costmap['data'], costmap['origin'], float(costmap['resolution'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        classes = np.where(grid < 0, 0, np.where(grid == 0, 1, np.where(grid >= 99, 3, 2)))
        axes[0].imshow(classes, origin='lower', interpolation='nearest', vmin=0, vmax=3,
                       cmap=ListedColormap(['#cbd0d5', 'white', '#e8bd77', '#20252a']),
                       extent=[origin[0], origin[0]+grid.shape[1]*res,
                               origin[1], origin[1]+grid.shape[0]*res])
        if len(poses):
            path = poses[poses[:, 0] <= float(costmap['sim_time'])]
            axes[0].plot(path[:, 1], path[:, 2], color='#0871c1', lw=1.5)
        axes[0].set_title(f"Nav2 costmap | t={float(costmap['sim_time']):.1f} s\n"
                          'Gray: unknown / white: observed free / amber: clearance cost')
        axes[0].set_xlabel('map x [m]')
        axes[0].set_ylabel('map y [m]')
        axes[0].set_facecolor('#cbd0d5')
        axes[0].set_xlim(origin[0], origin[0]+grid.shape[1]*res)
        axes[0].set_ylim(origin[1], origin[1]+grid.shape[0]*res)
        r = scan['ranges']
        angle = float(scan['angle_min']) + np.arange(len(r))*float(scan['angle_increment'])
        valid = np.isfinite(r) & (r > float(scan['range_min'])) & (r < float(scan['range_max']))
        axes[1].scatter(r[valid]*np.cos(angle[valid]), r[valid]*np.sin(angle[valid]),
                        s=5, color='#283e45', label='Finite LiDAR returns')
        no_return = np.flatnonzero(np.isposinf(r))[::max(1, int(np.isposinf(r).sum()/20))]
        for i in no_return:
            limit = float(scan['range_max'])
            axes[1].plot([0, limit*np.cos(angle[i])], [0, limit*np.sin(angle[i])],
                         '--', color='#b5c6cb', lw=.6)
        axes[1].plot(0, 0, '^', color='#0871c1', ms=8, label='LiDAR origin')
        axes[1].set_title(f"Raw LiDAR | t={float(scan['sim_time']):.1f} s\n"
                          'Dashed: positive-infinity no-return rays, not obstacle points')
        axes[1].set_xlabel('sensor x [m]')
        axes[1].set_ylabel('sensor y [m]')
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_aspect('equal')
        fig.suptitle(f'{directory.name} | Recorded sensors and planning grid; no SDF overlay')
        fig.tight_layout()
        fig.savefig(directory/'lidar_and_live_costmap.png', dpi=150)
        plt.close(fig)
    truth_path = evidence/'evaluation_poses.csv'
    if truth_path.exists() and len(rows):
        truth = np.genfromtxt(truth_path, delimiter=',', skip_header=1)
        if truth.ndim == 2 and len(truth):
            t = poses[:, 0]
            tx, ty = (np.interp(t, truth[:, 0], truth[:, i]) for i in (1, 2))
            tyaw = np.interp(t, truth[:, 0], np.unwrap(truth[:, 3]))
            myaw = np.array([float(row['yaw']) for row in rows])
            angle = myaw[0] - tyaw[0]
            c, s = np.cos(angle), np.sin(angle)
            delta = np.column_stack([tx-tx[0], ty-ty[0]])
            aligned = delta @ np.array([[c, s], [-s, c]]) + poses[0, 1:3]
            position_error = np.linalg.norm(poses[:, 1:3]-aligned, axis=1)
            heading_error = np.degrees(np.arctan2(np.sin(myaw-tyaw-angle), np.cos(myaw-tyaw-angle)))
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            axes[0].plot(*poses[:, 1:3].T, label='Robot estimate', color='#cc6400')
            axes[0].plot(*aligned.T, label='Simulator (evaluation only)', color='#007b9e')
            axes[0].axis('equal')
            axes[0].legend(fontsize=8)
            axes[0].set_title('Initial-pose-aligned trajectory')
            axes[1].plot(t, position_error, color='#b34f10')
            axes[1].set_title('Position error [m]')
            axes[2].plot(t, heading_error, color='#007b9e')
            axes[2].set_title('Heading error [deg]')
            for ax in axes[1:]:
                ax.set_xlabel('Simulation time [s]')
                ax.grid(alpha=.2)
            fig.suptitle(f'{directory.name}: independent localization evaluation (not control input)')
            fig.tight_layout()
            fig.savefig(directory/'localization_error.png', dpi=150)
            plt.close(fig)
            summary.update(max_position_error_m=float(position_error.max()),
                           max_abs_heading_error_deg=float(np.abs(heading_error).max()),
                           max_estimated_pose_gap_sim_sec=float(np.diff(t).max()) if len(t) > 1 else None)
            (directory/'evidence_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    render(parser.parse_args().directory)
