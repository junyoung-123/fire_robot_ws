#!/usr/bin/env python3
"""Retiming is by recorded simulation stamps, never by joining other runs."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import cv2
import numpy as np


def verified_completion(run, result):
    if not result.get('pass', False):
        return False
    audit_file = run/'strict_integrated_audit.json'
    if not audit_file.exists():
        raise RuntimeError('Run the independent integrated acceptance audit before exporting a success movie')
    audit = json.loads(audit_file.read_text())
    return bool(audit.get('pass_all') and audit.get('checks') and all(audit['checks'].values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive')
    evidence = args.run/'evidence'
    result = json.loads((args.run/'result.json').read_text())
    complete = verified_completion(args.run, result)
    timeline = [json.loads(line) for line in (evidence/'video_timeline.jsonl').read_text().splitlines()]
    if not timeline:
        raise RuntimeError('No recorded frames to export')
    capture = cv2.VideoCapture(str(evidence/'full_mission_raw.mp4'))
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f'world1_{"full" if complete else "diagnostic"}_same_run_{args.speed:g}x'
    temporary = args.output/f'{stem}_encoding.mp4'
    destination = args.output/f'{stem}.mp4'
    fps = 30.
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1600, 900))
    emitted = 0
    seen = set()
    first = timeline[0]['sim_time']
    intervals = np.diff([r['sim_time'] for r in timeline])
    if len(intervals) and np.any(intervals <= 0):
        raise RuntimeError('Recording contains non-increasing simulation stamps')
    final_interval = float(np.median(intervals)) if len(intervals) else 1./15.
    peaks = {}
    for line in (evidence/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['kind'] == 'evaluation_door_angle' and abs(event['angle_rad']) > .02:
            previous = peaks.get(event['model'])
            if previous is None or abs(event['angle_rad']) > abs(previous['angle_rad']):
                peaks[event['model']] = event
    peak_frames = {min(range(len(timeline)), key=lambda i: abs(timeline[i]['sim_time']-e['sim_time'])):
                   (name, e) for name, e in peaks.items()}
    for index, row in enumerate(timeline):
        valid, frame = capture.read()
        if not valid:
            raise RuntimeError(f'Movie ended before timeline frame {index}')
        assert row['frame'] == index
        frame[:45, 920:] = 255
        cv2.putText(frame, 'ROBOT CAMERA | RGB / model detections', (935, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .60, (35, 35, 35), 2)
        frame[855:] = 255
        stage = (row['phase'] if row['state'] in ('STARTING', 'OPENING_DOOR')
                 and row['phase'] != 'WAITING' else row['state'])
        cv2.putText(frame, f'Sim time {row["sim_time"]:.1f}s | {stage[:65]} | {args.speed:g}x',
                    (20, 883), cv2.FONT_HERSHEY_SIMPLEX, .58, (35, 35, 35), 2)
        if not complete:
            cv2.putText(frame, 'DIAGNOSTIC | NOT A COMPLETE MISSION', (940, 882),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (40, 40, 190), 1)
        if args.speed == 1. and stage not in seen:
            seen.add(stage)
            name = ''.join(c if c.isalnum() or c=='_' else '_' for c in stage)[:95]
            cv2.imwrite(str(args.output/f'{len(seen):02d}_{row["sim_time"]:.1f}s_{name}.jpg'), frame)
        if args.speed == 1. and index in peak_frames:
            name, peak = peak_frames[index]
            cv2.imwrite(str(args.output/f'90_{name}_peak_evaluation_only.jpg'), frame)
        if args.speed == 1. and index == len(timeline)-1:
            cv2.imwrite(str(args.output/'99_final_recorded_frame.jpg'), frame)
        end = timeline[index+1]['sim_time'] if index+1<len(timeline) else row['sim_time']+final_interval
        target_count = round((end-first)*fps/args.speed)
        for _ in range(target_count-emitted):
            writer.write(frame)
        emitted = target_count
    writer.release()
    capture.release()
    def external_path(path):
        if os.name == 'posix' and args.ffmpeg.suffix.lower() == '.exe':
            return subprocess.check_output(['wslpath', '-w', str(path.resolve())], text=True).strip()
        return str(path)
    subprocess.run([str(args.ffmpeg), '-hide_banner', '-loglevel', 'error', '-n',
        '-i', external_path(temporary), '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', external_path(destination)], check=True)
    temporary.unlink()
    metadata = {'source_run': str(args.run.resolve()), 'source_frames': len(timeline),
        'simulation_interval_sec': [first, timeline[-1]['sim_time']],
        'uniform_playback_speed': args.speed, 'output_frames': emitted,
        'duration_sec': emitted/fps, 'output_fps': fps, 'cross_run_splicing': False,
        'actual_capture_fps': (len(timeline)-1)/(timeline[-1]['sim_time']-first),
        'capture_gap_p95_sec': float(np.quantile(intervals,.95)),
        'capture_gap_max_sec': float(np.max(intervals)),
        'complete_mission_verified': complete, 'status': result['status'],
        'missing_camera_frames': sum(r.get('camera_stamp') is None for r in timeline)}
    (args.output/f'{stem}.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
