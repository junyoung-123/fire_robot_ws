#!/usr/bin/env python3
"""Build an offline evidence index without modifying trial records."""
import argparse
import html
import json
from pathlib import Path
import re
from urllib.parse import quote


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def build(root):
    rows, sections = [], []
    trials = []
    for folder in root.glob('world*_r*'):
        match = re.fullmatch(r'world(\d+)_r(\d+)', folder.name)
        if not match:
            continue
        world, revision = map(int, match.groups())
        report = read(folder/'independent_check.json')
        result = read(folder/'result.json')
        manifest = read(folder/'manifest.json')
        summary = read(folder/'evidence_summary.json')
        status = report.get('status', result.get('status', 'IN PROGRESS / NOT AUDITED'))
        trials.append((world, revision, folder, report, manifest, summary, status))
    trials.sort(key=lambda t: (t[0], -t[1]))
    for world, revision, folder, report, manifest, summary, status in trials:
        prefix = quote(folder.name)
        name = html.escape(folder.name)
        rows.append(f'<tr><td><a href="#{name}">{name}</a></td><td>{html.escape(status)}</td>'
                    f'<td>{report.get("unique_valid_visits", "-")} / '
                    f'{report.get("expected_blue_count_evaluator_only", "-")}</td>'
                    f'<td>{html.escape(manifest.get("started", ""))}</td></tr>')
        figures = []
        for filename, caption in (
            ('observed_map_progress.png', '센서로 생성한 초기 → 중간 → 마지막 지도. 회색은 미관측 영역.'),
            ('evaluation_world_trajectory.png', '별도 평가용 실제 배치와 궤적. 이 배치 정보는 제어에 입력하지 않음.'),
            ('door_visit_alignment.png', '문별 방문 거리와 방향. 로봇팔 도달·문 개방 성공을 뜻하지 않음.'),
            ('camera_observations.png', '같은 실행에서 기록한 전방 카메라 원본.'),
            ('lidar_and_live_costmap.png', '기록한 LiDAR 거리와 실제 Nav2 지도. 회색 미관측 영역과 센서로 확인한 빈 공간 구별.'),
            ('localization_error.png', '시작 자세를 맞춘 위치추정 오차. 시뮬레이터 위치는 비교에만 사용.')):
            if (folder/filename).exists():
                figures.append(f'<figure><a href="{prefix}/{filename}"><img src="{prefix}/{filename}" '
                               f'alt="{html.escape(caption)}" loading="lazy"></a><figcaption>{caption}</figcaption></figure>')
        failures = '<br>'.join(html.escape(s) for s in report.get('failures', []))
        fixture = ('열린 출구 시험: 초록 문틀의 시각·충돌 형상 일치. 내부 문·장애물 배치 유지.'
                   if manifest.get('exit_fixture') else
                   '이전 출구 형상: 충돌체 없는 초록 패널. LiDAR에는 판으로 관측되는 한계가 있음.')
        metrics = ('위치추정 최대 오차: '
                   f'{summary.get("max_position_error_m", 0):.2f} m / '
                   f'{summary.get("max_abs_heading_error_deg", 0):.1f}°') if 'max_position_error_m' in summary else ''
        links = ' · '.join(f'<a href="{prefix}/{file}">{label}</a>' for file, label in (
            ('independent_check.json', '독립 판정'), ('manifest.json', '실행 설정·해시'),
            ('simulation.log', 'ROS 로그'), ('evidence/events.jsonl', '관측·목표·방문 기록'),
            ('evidence/poses.csv', '로봇 추정 위치'), ('evidence/evaluation_poses.csv', '평가용 실제 위치'))
                          if (folder/file).exists())
        sections.append(f'<section id="{name}"><h2>{name} · {html.escape(status)}</h2><p>{fixture}</p><p>{metrics}</p>'
                        f'<p class="failure">{failures}</p><p>{links}</p>{"".join(figures)}</section>')
    intro = '''<h1>사전 지도 없는 SLAM 주행 검증</h1>
<p>별도 코드 사본 · 기존 검증 자료 보존 · 실행별 날짜와 원본 해시는 아래 기록 참조</p>
<p><strong>월드 내부의 문과 장애물 배치 유지.</strong> 열린 출구 조건은 각 실행에 별도 표시함.
저장 지도·사전 장애물 좌표·문 개수는 제어에 입력하지 않음.
휠 오도메트리와 LiDAR로 온라인 SLAM을 수행하고, 카메라 관측으로 문 후보를 생성함.</p>
<p><strong>주행 전용 시험.</strong> 방문 완료를 기존 FSM의 조작 서비스 경계에서 반환하며, 팔이나 문 힌지는 움직이지 않음.
로그의 DOOR_OPENED는 이 시험에서 방문 완료를 뜻함. 물리적 문 개방 증거와 혼용 금지.</p>
<p>PASS_NAV_VISITS_EXIT: 독립 기록 기준 모든 파란 문 방문, 잘못된 색상·중복 방문 없음, 방문 시 문 중심 진행축 오차 ≤0.70 m,
거리 ≤1.60 m, 벽면 정면 방향 오차 ≤15°, 비상구 위치 통과, 지정 장애물과 Nav2 외곽선의 겹침 없음.
물리 접촉 센서는 계측하지 않았으며, 로봇팔 작업 거리·정밀 정렬·반복 신뢰도를 보증하지 않음.</p>
<p>카메라 보정값, 문 높이 기반 거리 보조 추정, 기존 복도 정책의 여유 거리·병합 허용오차는 남아 있음.
‘사전 지도 없음’은 ‘모든 모델 가정·휴리스틱 제거’를 의미하지 않음. 열린 출구는 출구 개방 동작의 검증을 뜻하지 않음.</p>'''
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>사전 지도 없는 SLAM 주행 검증</title><style>
body{font-family:Arial,'Malgun Gothic',sans-serif;max-width:1200px;margin:32px auto;padding:0 22px;color:#222;line-height:1.6}
h1{font-size:28px}h2{font-size:22px}section{border-top:2px solid #ddd;margin-top:40px;padding-top:12px}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:9px;text-align:left}
th{background:#eee}figure{margin:24px 0}img{width:100%;height:auto}figcaption{color:#555}a{color:#006da3}
.failure{color:#a02c22}p{max-width:1120px}</style><body>'''+intro+(
        '<table><thead><tr><th>실행</th><th>판정</th><th>유효 방문 / 평가 대상</th><th>시작 시각</th></tr></thead><tbody>'
        +''.join(rows)+'</tbody></table>'+''.join(sections)+'</body></html>')
    (root/'00_검증결과.html').write_text(page, encoding='utf-8')
    print(root/'00_검증결과.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
