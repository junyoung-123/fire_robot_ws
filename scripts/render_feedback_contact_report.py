#!/usr/bin/env python3
"""Plot recorded feedback only; never generate or alter Gazebo scene images."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source, output):
    result = json.loads(source.read_text(encoding='utf-8'))
    events = result.get('feedback_events', [])
    following = [row for row in events if row.get('event') == 'contact_follow_sample']
    probes = [row for row in events if row.get('event') == 'latch_probe']
    presses = [row for row in events if row.get('event') in ('press_step', 'latch_press_correction')]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=160)
    state = 'PASS' if result.get('pass') else 'FAIL'
    fig.suptitle(f'Physical contact feedback: {state}', fontsize=16)
    fig.text(.08, .93, result.get('service_message', 'No service result'), fontsize=10)
    if following:
        t = [row['sim_time_sec'] for row in following]
        actual = np.asarray([row['actual_tool_xyz'] for row in following])
        requested = np.asarray([row['requested_tool_xyz'] for row in following])
        axes[0].plot(t, 1000*np.linalg.norm(actual-requested, axis=1), color='#bc3344', label='tool tracking error')
        axes[0].set_ylabel('Position error [mm]')
        axes[0].legend(loc='upper left')
        axes[0].set_title('Actual joint encoders / FK vs requested tool position', fontsize=11)
    else:
        axes[0].text(.5, .5, 'Contact-follow stage was not reached', transform=axes[0].transAxes, ha='center')
    if presses:
        axes[1].step([row['sim_time_sec'] for row in presses],
                     [1000*row['commanded_depth_m'] for row in presses], where='post', label='commanded press budget', color='#345fad')
    for row in events:
        if row.get('event') == 'press_verified':
            axes[1].scatter(row['sim_time_sec'], 1000*row['measured_tool_down_m'],
                            color='#247d49', marker='D', label='measured descent at press confirmation')
    axes[1].set_ylabel('Press depth [mm]')
    axes[1].set_title('A command is not the same as measured motion', fontsize=11)
    if axes[1].has_data():
        axes[1].legend(loc='upper left', fontsize=9)
    if probes:
        t = [row['sim_time_sec'] for row in probes]
        axes[2].plot(t, [math.degrees(row['door_delta_rad']) for row in probes], label='measured hinge motion', color='#222222')
        axes[2].plot(t, [math.degrees(row['lever_delta_rad']) for row in probes], label='measured lever motion', color='#247d49')
        axes[2].legend(loc='upper left', fontsize=9)
    else:
        axes[2].text(.5, .5, 'No latch-motion probe was executed', transform=axes[2].transAxes, ha='center')
    axes[2].set_ylabel('Angle [deg]')
    for axis in axes:
        axis.set_xlabel('Recorded simulation time [s]')
        axis.grid(alpha=.2)
    maximum = math.degrees(float(result.get('max_delta_rad', 0.)))
    required = math.degrees(float(result['min_angle_rad']))
    modes = sorted({row.get('observation_source', 'unspecified') for row in following})
    fig.text(.08, .015,
             f'Max hinge {maximum:.2f} deg / full-open requirement {required:.2f} deg. '
             f'Follow source: {", ".join(modes) if modes else "not reached"}.\n'
             'Instrumented Gazebo joint feedback; not a real force-sensor test. Source: same-run result.json.',
             fontsize=9)
    fig.tight_layout(rect=(0., .06, 1., .91))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Preserving existing evidence: ' + str(output))
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == '__main__':
    main()
