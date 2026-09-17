#!/usr/bin/env python3
"""Plot passive contact events next to the same-run control feedback."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(folder, output_name='contact_timeline.png'):
    result = json.loads((folder / 'result.json').read_text())
    diagnostic = json.loads((folder / 'contact_diagnostics.json').read_text())
    output = folder / output_name
    if output.exists():
        raise FileExistsError(output)
    samples = [s for s in diagnostic['samples'] if s['phase'] == 'LATCH_CLEAR_PUSH']
    if not samples:
        raise ValueError('No latch-clear contact observations to plot')
    start, end = min(s['sim_time_sec'] for s in samples), max(s['sim_time_sec'] for s in samples)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=160, sharex=True)
    fig.suptitle('Physical contact diagnosis: ' + ('PASS' if result['pass'] else 'FAIL'), fontsize=16)
    series = {'Lever / finger': (2, '#216b9b'), 'Bolt / strike': (1, '#b43240'),
              'Panel / floor': (0, '#7b7b7b')}
    times = {key: [] for key in series}
    for sample in samples:
        for contact in sample['contacts']:
            names = contact['collision1'] + ' ' + contact['collision2']
            key = None
            if sample['part'] == 'lever' and 'gripper_finger' in names:
                key = 'Lever / finger'
            elif sample['part'] == 'latch' and 'latch_strike_collision' in names:
                key = 'Bolt / strike'
            elif sample['part'] == 'panel' and 'ground::' in names:
                key = 'Panel / floor'
            if key:
                times[key].append(sample['sim_time_sec'])
    for key, (level, color) in series.items():
        axes[0].scatter(times[key], [level] * len(times[key]), marker='|', color=color, s=90)
    axes[0].set_yticks([0, 1, 2], ['Panel / floor', 'Bolt / strike', 'Lever / finger'])
    axes[0].set_ylim(-.5, 2.5)
    axes[0].set_title('Recorded collision pairs (passive Gazebo sensors, not control inputs)', fontsize=11)
    probes = [e for e in result['feedback_events'] if e['event'] == 'latch_probe']
    axes[1].plot([e['sim_time_sec'] for e in probes],
                 [math.degrees(e['door_delta_rad']) for e in probes], color='#222222', label='hinge motion')
    axes[1].plot([e['sim_time_sec'] for e in probes],
                 [math.degrees(e['lever_delta_rad']) for e in probes], color='#278047', label='lever at probe start')
    axes[1].set_ylabel('Measured angle [deg]')
    axes[1].legend(loc='upper left')
    corrections = [e for e in result['feedback_events']
                   if e['event'] in ('latch_press_correction', 'lever_hold_correction')]
    axes[2].step([e['sim_time_sec'] for e in corrections],
                 [e['commanded_depth_m'] * 1000 for e in corrections], where='post', color='#345fad')
    axes[2].set_ylabel('Commanded press budget [mm]')
    axes[2].set_xlabel('Recorded simulation time [s]')
    axes[2].set_xlim(start - .5, end + .5)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.text(.08, .025,
             'Marks show sampled contacts, not measured contact force. Missing marks alone are not a contact-loss sensor.\n'
             'Compare the measured response with each correction; these contact marks alone do not prove a firm grip.\n'
             'A depressed lever is not sufficient evidence of latch release or successful opening.', fontsize=10)
    fig.tight_layout(rect=(0., .10, 1., .95))
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output-name', default='contact_timeline.png')
    args=parser.parse_args()
    render(args.folder, args.output_name)
