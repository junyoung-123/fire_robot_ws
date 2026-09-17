#!/usr/bin/env python3
"""Plot measured free-space finger opening, separately from door-test results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(before, after, output):
    if output.exists():
        raise FileExistsError(output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=160, sharey=True)
    for axis, folder, title in zip(axes, (before, after), ('Before: P=80 / D=0.8', 'After: P=400 / D=6.3')):
        result = json.loads((folder / 'gripper_result.json').read_text())
        samples = [sample for sample in result['samples'] if sample['stage'] == 'open_free']
        start = samples[0]['sim_time']
        times = [sample['sim_time'] - start for sample in samples]
        axis.plot(times, [1000 * sample['left'] for sample in samples], color='#276da5', label='Left finger')
        axis.plot(times, [-1000 * sample['right'] for sample in samples], color='#b84735', label='Right finger')
        axis.axhline(35., color='#333333', linestyle=':', label='Command / stroke limit: 35 mm')
        axis.axhline(34., color='#53833e', linestyle='--', label='Unchanged release gate: 34 mm')
        axis.set_title(title)
        axis.set_xlabel('Simulation time since open command [s]')
        axis.set_ylim(28., 35.5)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('Measured per-finger opening [mm]')
    fig.suptitle('Gripper calibration: gravity sag reduced, force limit unchanged', fontsize=14)
    fig.text(.07, .025, 'FREE-SPACE ACTUATOR TEST ONLY. No lever contact or full-door PASS is claimed.\n'
             'Original PIPER geometry, 35 mm finger stroke and 8 N controller force cap retained.', fontsize=10)
    fig.tight_layout(rect=(0., .1, 1., .94))
    fig.savefig(output)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.before, args.after, args.output)
