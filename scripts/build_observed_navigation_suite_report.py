#!/usr/bin/env python3
"""Create a same-version, five-world summary from retained run evidence."""
import argparse
import csv
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
import numpy as np

from check_observed_navigation import exit_jamb_footprints, structure_footprints
from render_obstacle_validation_evidence import obstacle_footprints


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def poses(path):
    with path.open() as stream:
        return np.array([[float(row[k]) for k in ('sim_time', 'x', 'y')]
                         for row in csv.DictReader(stream)])


def build(root, revision):
    runs = [root/f'world{i}_{revision}' for i in range(1, 6)]
    if not all((run/'independent_check.json').exists() for run in runs):
        raise RuntimeError('Five completed independent evaluations are required')
    manifests = [read(run/'manifest.json') for run in runs]
    identical = all(m['source_sha256'] == manifests[0]['source_sha256'] for m in manifests)
    harness_identical = bool(manifests[0].get('harness_sha256')) and all(
        m.get('harness_sha256') == manifests[0]['harness_sha256'] for m in manifests)
    checks = [read(run/'independent_check.json') for run in runs]
    all_passed = identical and all(c['status'] == 'PASS_NAV_VISITS_EXIT' for c in checks)
    report = dict(revision=revision, same_control_source=identical,
                  same_recorded_harness=harness_identical,
                  status='ALL_FIVE_PASS_NAVIGATION_ONLY' if all_passed else 'INCOMPLETE_OR_FAILED',
                  prior_map=False, ground_truth_control=False, physical_opening_tested=False,
                  exit_condition='user-approved open green passage', worlds=[])
    for i, (run, check) in enumerate(zip(runs, checks), 1):
        summary, result = read(run/'evidence_summary.json'), read(run/'result.json')
        audits = [json.loads(line)['audit'] for line in
                  (run/'evidence/events.jsonl').read_text().splitlines()
                  if json.loads(line).get('kind') == 'input_audit']
        violations = []
        for audit in audits:
            for key, allowed in (
                    ('map_publishers', {'slam_toolbox'}),
                    ('odom_publishers', {'gz_selected_odometry_bridge'}),
                    ('evaluation_truth_subscribers', {'observed_navigation_recorder'})):
                unexpected = set(audit.get(key, [])) - allowed
                if unexpected:
                    violations.append(dict(sim_time=audit.get('sim_time'), field=key,
                                           unexpected=sorted(unexpected)))
        report['worlds'].append(dict(world=i, run=run.name, status=check['status'],
            valid_visits=check['unique_valid_visits'],
            blue_count_evaluation_only=check['expected_blue_count_evaluator_only'],
            wall_seconds=result.get('wall_seconds'), failures=check['failures'],
            max_position_error_m=summary.get('max_position_error_m'),
            max_heading_error_deg=summary.get('max_abs_heading_error_deg'),
            max_estimated_pose_gap_sim_sec=summary.get('max_estimated_pose_gap_sim_sec'),
            recorded_audit_count=len(audits), input_audit_violations=violations,
            overlap=check['navigation_footprint_overlap_obstacles']))
    if any(item['input_audit_violations'] for item in report['worlds']):
        report['status'] = 'INPUT_ISOLATION_FAILED'
    (root/f'suite_{revision}.json').write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(5, 3, figsize=(22, 13), squeeze=False)
    for i, run in enumerate(runs):
        maps = sorted((run/'evidence').glob('map_*.npz'))
        trace = poses(run/'evidence/poses.csv')
        last = np.load(maps[-1])
        origin, res, shape = last['origin'], float(last['resolution']), last['data'].shape
        bounds = (origin[0]-.2, origin[0]+shape[1]*res+.2,
                  origin[1]-.2, origin[1]+shape[0]*res+.2)
        for j, index in enumerate((0, len(maps)//2, len(maps)-1)):
            saved, ax = np.load(maps[index]), axes[i, j]
            grid, xy, resolution = saved['data'], saved['origin'], float(saved['resolution'])
            pixels = np.where(grid < 0, 0, np.where(grid >= 65, 2, 1))
            ax.imshow(pixels, origin='lower', interpolation='nearest', vmin=0, vmax=2,
                      cmap=ListedColormap(['#cbd0d5', 'white', '#252c32']),
                      extent=(xy[0], xy[0]+grid.shape[1]*resolution,
                              xy[1], xy[1]+grid.shape[0]*resolution))
            path = trace[trace[:, 0] <= float(saved['sim_time'])]
            if len(path):
                ax.plot(path[:, 1], path[:, 2], color='#bd5b11', lw=1.3)
                ax.plot(path[-1, 1], path[-1, 2], 'o', color='#bd5b11', ms=3)
            ax.set(xlim=bounds[:2], ylim=bounds[2:], aspect='equal',
                   title=f"World {i+1} | {('Initial', 'Middle', 'Last recorded map')[j]} | {float(saved['sim_time']):.0f} s")
            ax.set_facecolor('#cbd0d5')
            ax.set_xlabel('map x [m]')
            if j == 0:
                ax.set_ylabel('map y [m]')
    fig.suptitle(f'{revision}: recorded online SLAM growth, no saved map\n'
                 'Gray: unknown | White: observed free | Black: observed occupied | Orange: robot estimate', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root/f'00_slam_map_evolution_{revision}.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(17, 15), squeeze=False)
    for i, (run, check) in enumerate(zip(runs, checks)):
        ax = axes[i, 0]
        world = run/'open_exit.world'
        root_xml = ET.parse(world).getroot()
        for _, poly in obstacle_footprints(world)+exit_jamb_footprints(root_xml)+structure_footprints(root_xml):
            ax.add_patch(Polygon(poly, facecolor='#b1b3b5', edgecolor='#686d70', lw=.5))
        for model in root_xml.findall('./world/model'):
            name = model.get('name', '')
            if not (name.startswith('door_') or name == 'exit_green'):
                continue
            color = next((c for c in ('blue', 'red', 'green') if c in name), None)
            if color:
                p = list(map(float, model.findtext('pose').split()))
                ax.plot(p[0], p[1], 's', color=color, ms=6)
        trace = poses(run/'evidence/evaluation_poses.csv')
        ax.plot(trace[:, 1], trace[:, 2], color='#bd5b11', lw=1.4)
        events = [json.loads(line) for line in (run/'evidence/events.jsonl').read_text().splitlines()]
        for event in events:
            if event['kind'] == 'visit':
                p = trace[np.argmin(abs(trace[:, 0]-event['payload']['sim_time']))]
                ax.plot(p[1], p[2], '*', color='#15191d', ms=10)
        ax.set_aspect('equal')
        ax.autoscale()
        ax.set(xlabel='Gazebo world x [m]', ylabel='y [m]',
               title=f"World {i+1} | {check['status']} | blue visits {check['unique_valid_visits']}/{check['expected_blue_count_evaluator_only']}")
    fig.suptitle(f'{revision}: independent navigation evaluation\n'
                 'World geometry and simulator pose are evaluation-only, NOT control inputs. Stars: visits, NOT physical opening.', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(root/f'01_world_trajectories_{revision}.png', dpi=170)
    plt.close(fig)
    rows, sections = [], []
    for item, run in zip(report['worlds'], runs):
        rows.append(f'<tr><td><a href="#world{item["world"]}">World {item["world"]}</a></td>'
                    f'<td>{html.escape(item["status"])}</td>'
                    f'<td>{item["valid_visits"]}/{item["blue_count_evaluation_only"]}</td>'
                    f'<td>{item["wall_seconds"]:.0f} s</td></tr>')
        figures = []
        for name, caption in (
            ('observed_map_progress.png', '초기 → 중간 → 마지막 기록 지도: 실제 /map 관측 결과'),
            ('evaluation_world_trajectory.png', '실제 월드 배치와 궤적: 정답 정보는 사후 평가에만 사용'),
            ('door_visit_alignment.png', '문별 주행 방문 위치와 방향: 로봇팔 근접 정렬 검증 아님'),
            ('camera_observations.png', '동일 실행의 전방 카메라 원본'),
            ('lidar_and_live_costmap.png', 'LiDAR와 Nav2 지도 기록'),
            ('localization_error.png', '위치추정 오차: 평가용 시뮬레이터 위치와 비교')):
            if (run/name).exists():
                figures.append(f'<figure><figcaption>{caption}</figcaption>'
                               f'<a href="{run.name}/{name}"><img loading="lazy" src="{run.name}/{name}"></a></figure>')
        sections.append(f'<section id="world{item["world"]}"><h2>World {item["world"]}</h2>'
                        f'<p><a href="{run.name}/independent_check.json">독립 판정</a> · '
                        f'<a href="{run.name}/manifest.json">입력 조건·소스 해시</a> · '
                        f'<a href="{run.name}/simulation.log">원본 로그</a></p>'
                        + ''.join(figures) + '</section>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>빈 지도 SLAM 주행 검증 {revision}</title>
<style>body{{font-family:Arial,'Malgun Gothic',sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;line-height:1.65;color:#222}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:10px;text-align:left}}
th{{background:#eee}}section{{border-top:2px solid #ccc;margin-top:40px;padding-top:15px}}
img{{width:100%;height:auto}}figure{{margin:25px 0}}figcaption{{font-weight:bold}}a{{color:#00649b}}</style>
<h1>World 1–5 · 빈 지도에서 시작한 SLAM 주행</h1>
<p><strong>{html.escape(report['status'])}</strong> · {revision} · 동일 제어 소스: {identical}</p>
<p>저장 지도, 사전 장애물·문 좌표, 문 개수를 제어에 전달하지 않음. 휠 오도메트리와 LiDAR로 지도를 만들고,
카메라 관측으로 문 후보를 기억·선택함. 월드의 문과 장애물 배치는 유지하고, 승인받은 출구만 초록 표식의 열린 통로로 구성함.</p>
<p>SLAM 지도는 시작 위치 기준의 map 좌표계, 배치·궤적 평가는 Gazebo world 좌표계임.
두 그림은 원점이 다르므로 좌표 숫자를 직접 비교하지 않음. 초기 지도는 첫 기록 시점까지 실제 스캔된 영역임.</p>
<p><strong>주행 전용 검증.</strong> 파란문 방문·방향 정렬·중복 제외·비상구 통과를 확인함.
문 개방 서비스는 방문 완료를 반환하는 시험용 어댑터이며, 물리적 문 개방이나 로봇팔 도달 거리는 검증하지 않음.</p>
<table><tr><th>월드</th><th>판정</th><th>유효 방문 / 평가상 파란문</th><th>실행 시간</th></tr>{''.join(rows)}</table>
<p>판정 기준: 문 중심 진행축 오차 ≤0.70 m, 방문 거리 ≤1.60 m, 벽면 정면 방향 오차 ≤15°,
모든 파란문 중복 없이 방문, 실제 출구 통과, 기록된 위치에서 Nav2 외곽선과 장애물·벽·패널 겹침 없음.
접촉 센서 기반 충돌 검증은 아님. 실행 시간에는 시작·종료 정리 시간이 포함됨.</p>
<p>카메라 보정, 문 높이 기반 거리 보조 추정, 주행 여유 거리와 병합 허용오차 같은 일반 파라미터는 사용함.
반복 성공률·임의의 새 환경·실제 로봇 성능을 보증하지 않으며, 재접근과 부정확한 후보 제외에 따른 지연은 남아 있음.</p>
<p>위치추정 기록의 중단 간격은 전체 판정 JSON의 max_estimated_pose_gap_sim_sec에 별도 기록함.
TF·관측 단절을 완전히 해소했다는 주장은 하지 않음. 운용 중 점검상 단절이 60초 이상 지속되면 검증을 중단함.</p>
<p><a href="00_slam_map_evolution_{revision}.png">5개 월드 지도 생성 과정</a> ·
<a href="01_world_trajectories_{revision}.png">5개 월드 평가 궤적</a> ·
<a href="suite_{revision}.json">전체 판정 원본</a> · <a href="00_검증결과.html">이전 실행·실패 기록</a></p>
{''.join(sections)}</html>'''
    (root/'00_최신결과.html').write_text(page, encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('revision')
    args = parser.parse_args()
    build(args.root, args.revision)
