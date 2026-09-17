#!/usr/bin/env python3
"""Attach a stricter contact audit without rewriting historical result.json."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(folder, review_image=False):
    runner_path = Path(__file__).resolve().parents[1] / 'src/fire_robot_bringup/scripts/run_physical_contact_door_test.py'
    spec = importlib.util.spec_from_file_location('contact_evidence_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    result = json.loads((folder / 'result.json').read_text())
    contacts = json.loads((folder / 'contact_diagnostics.json').read_text())
    held_mode = result.get('opening_load_path') == 'held_handle'
    audit = (runner.audit_handle_held_contacts(contacts['samples']) if held_mode
             else runner.audit_panel_push_contacts(contacts['samples']))
    audit['original_result_pass'] = result['pass']
    audit['overall_safe_contact_pass'] = bool(result['pass'] and audit['pass'])
    audit['original_result_unchanged'] = True
    audit['source'] = str(folder / 'result.json')
    output = folder / 'contact_safety_audit.json'
    image_output = folder / ('opening_evidence_review.png' if review_image else 'opening_evidence.png')
    if image_output.exists():
        raise FileExistsError(image_output)
    if review_image:
        if json.loads(output.read_text()) != audit:
            raise ValueError('Existing audit differs; do not relabel historical evidence')
    else:
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(audit, indent=2))

    phases = folder / 'phases'
    press = sorted(phases.glob('closeup_*_press_handle.png'))
    latch = sorted(phases.glob('closeup_*_latch_clear_push.png'))
    angle = result.get('max_delta_rad', 0.) * 180. / 3.141592653589793
    panels = [
        ('1  YOLO handle + registered depth', folder / 'yolo_primary_detection.png'),
        ('2  Actual PIPER fingers / lever press', press[-1] if press else phases / 'press_handle.png'),
        ('3  Lever held while door starts opening', latch[-1] if latch else phases / 'latch_clear_push.png'),
        ('4  Hand retained / base supplies motion' if held_mode else '4  Arm stowed / base-driven push',
         phases / ('handle_held_base_open.png' if held_mode else 'base_push_open.png')),
        (f'5  Final overhead / peak {angle:.1f} deg', folder / 'after_overhead.png'),
        ('6  Recorded hinge motion (same run)', folder / 'door_hinge_angle.png'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.8), dpi=130)
    fig.suptitle('Single-door Gazebo evidence | ' + folder.name, fontsize=17, y=.97)
    for axis, (label, path) in zip(axes.flat, panels):
        if path.exists():
            axis.imshow(plt.imread(path))
        else:
            axis.text(.5, .5, 'No recorded frame for this stage', ha='center', va='center')
        axis.axis('off')
        axis.set_title(label, fontsize=12, loc='left', pad=8)
    overall = audit['overall_safe_contact_pass']
    summary = ('Overall: ' + ('PASS' if overall else 'FAIL')
               + '    Motion/FSM: ' + ('PASS' if result.get('motion_contract_pass', result['pass']) else 'FAIL')
               + '    Contact audit: ' + ('PASS' if audit['pass'] else 'FAIL'))
    reason = 'See contact_safety_audit.json for recorded load-path checks.'
    if not result['pass']:
        reason = result.get('service_message') or result.get('error') or 'Full motion not completed.'
    if audit['forbidden_contact_parts']:
        reason = ('Direct robot/panel contact recorded; this does not qualify as handle-held opening.'
                  if held_mode else 'Camera/sensor housing contacted the panel; not accepted as safe chassis pushing.')
    fig.text(.04, .095, summary, fontsize=15, weight='bold', color='#1d6b46' if overall else '#ad2525')
    fig.text(.04, .056, textwrap.fill(reason, width=135), fontsize=11)
    fig.text(.04, .030, 'Real rendered Gazebo frames and recorded joints. Instrumented simulation, not real hardware or force-control proof.', fontsize=11)
    fig.subplots_adjust(left=.03, right=.98, top=.9, bottom=.16, wspace=.09, hspace=.15)
    fig.savefig(image_output)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--review-image', action='store_true',
                        help='Render a new review image; preserve the old audit and image.')
    args = parser.parse_args()
    main(args.folder, args.review_image)
