#!/usr/bin/env python3
"""Package one completed run without replacing older validation material."""
import argparse
import datetime
import html
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run/'result.json').read_text())
    checkpoint = json.loads((args.run/'evidence/full_checkpoint.json').read_text())
    status = 'PASS' if result['pass'] else 'NOT_COMPLETE'
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    records = root/'records'
    records.mkdir(exist_ok=True)
    for name in ('result.json', 'manifest.json', 'world_snapshot.world',
                 'simulation.log', 'recorder.log', 'evaluation_bridge.log'):
        shutil.copyfile(args.run/name, records/name)
    for name in ('full_checkpoint.json', 'input_audit.json', 'ros_graph.json',
                 'poses.csv', 'evaluation_poses.csv', 'events.jsonl',
                 'video_timeline.jsonl', 'terminal_state.json'):
        source = args.run/'evidence'/name
        if source.exists():
            shutil.copyfile(source, records/name)
    angles = checkpoint['evaluation_max_angles_rad']
    summary = {
        'packaged_at': datetime.datetime.now().astimezone().isoformat(),
        'run': args.run.name, 'status': status, 'result': result,
        'evaluation_max_angles_deg': {k: round(v*180/3.141592653589793, 2)
                                     for k, v in angles.items()},
        'per_door_gripper_lever_contact_recorded': {
            name: any('gripper' in ' '.join(pair)
                      for pair in checkpoint.get('contact_pairs', {}).get(name+'/lever', []))
            for name in angles},
        'same_run_continuous_recording': True,
        'older_navigation_results_replaced': False,
        'cross_run_splicing': False,
    }
    (root/'recording_summary.json').write_text(json.dumps(summary, indent=2))
    success_count = result['successful_physical_jobs']
    jobs = checkpoint.get('physical_jobs', [])
    reason = jobs[-1].get('message', result['status']) if jobs else result['status']
    angle_summary = ', '.join(f'{name}: {value:.2f}°' for name, value in summary['evaluation_max_angles_deg'].items())
    meaning = ('처음 출발부터 모든 물리 문 개방과 전체 차체의 출구 통과까지 확인했습니다.'
               if result['pass'] else
               '전체 임무는 완료하지 못했습니다. 성공 데모가 아닌 같은 실행의 진단 녹화입니다.')
    text = f'''# World 1 통합 녹화: {status}

{meaning}

- 실행: {args.run.name}
- 종료 상태: {result['status']}
- 종료 이유: {reason}
- 평가용 실제 문 최대 회전각: {angle_summary}
- 물리 개방 성공 작업: {success_count}
- 전체 차체 출구 통과: {result['whole_chassis_past_exit']}
- 제어 소스 실행 중 변경: {not result['source_unchanged']}

## 영상 구성

왼쪽은 Gazebo에서 렌더링한 외부 시점입니다. 오른쪽 위는 로봇 전방 RGB 영상이며,
실제 검출 결과가 최신일 때 YOLO 표시 영상을 사용합니다. 오른쪽 아래는 실제 온라인
SLAM 지도와 관측한 문 기억, 선택 목표, 로봇 궤적입니다. 외부 시점의 촬영 카메라만
시뮬레이터 위치값으로 움직이며, 이 위치값은 로봇 제어에 전달하지 않습니다.

1x/4x는 기록한 시뮬레이션 시각에 맞춘 재생 속도입니다. 저성능 실행 중 프레임
누락은 동일 실행의 앞 프레임 유지로 처리합니다. 다른 실행의 장면을 이어붙이지 않습니다.
음성은 없습니다. 원본 프레임과 시각 대응은 records/video_timeline.jsonl에 있습니다.

## 검증 조건

원래 출발점, World 1 장애물·문 배치, 저장 지도 없는 온라인 SLAM,
바퀴 엔코더 기반 주행, 카메라 문·손잡이 관측, 로봇 엔코더 기반 팔 제어를 사용했습니다.
파란 문 패널은 동일 위치의 실제 접촉·힌지·레버·걸쇠 모델로 교체했습니다.
초록 표식 출구는 실제 열린 통로입니다.

문·레버의 Gazebo 정답 관절값은 제어에 사용하지 않고 검증 기록에만 사용합니다.
레버 회전의 엔코더 추정은 손잡이가 미끄러지지 않고 파지된다는 가정이 있으며,
힘/토크 센서로 파지를 입증한 것과는 다릅니다.

## 이전 자료와의 구분

기존 World 1~5 주행 성공과 R34 단일문 개방 자료는 수정하지 않았습니다.
이번 결과가 실패한 경우, 그 두 결과가 통합 성공으로 바뀌었다는 의미가 아닙니다.
상태, 각도, 접촉 및 성공 판정의 원본은 records에 보관했습니다.
'''
    (root/'README_녹화결과.md').write_text(text, encoding='utf-8')
    videos = sorted(root.glob('*.mp4'))
    images = sorted(root.glob('*.jpg'))
    video_cards = ''.join(
        f'<section><h2>{html.escape(p.name)}</h2><video controls preload="metadata" src="{html.escape(p.name)}"></video>'
        f'<p><a href="{html.escape(p.name)}" download>MP4 파일</a></p></section>' for p in videos)
    photo_cards = ''.join(
        f'<figure><a href="{html.escape(p.name)}"><img loading="lazy" src="{html.escape(p.name)}"></a>'
        f'<figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in images)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>World 1 통합 녹화</title><style>
body{{font:16px/1.6 "Malgun Gothic",sans-serif;margin:32px auto;padding:0 24px;max-width:1400px;color:#20252a;background:#f7f8fa}}
h1{{font-size:28px}}h2{{font-size:18px}}section{{border-top:1px solid #ccd2d8;padding:18px 0}}
video{{width:100%;max-height:760px;background:#111}}.status{{font-weight:bold;color:{'#167344' if result['pass'] else '#b62424'}}}
.photos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}figure{{margin:0}}img{{width:100%}}figcaption{{font-size:13px}}
</style><h1>World 1 주행 + 물리 문 개방</h1>
<p class="status">{status} | {html.escape(meaning)}</p>
<p>{args.run.name} · {html.escape(result['status'])} · 개방 성공 {success_count} · 출구 통과 {result['whole_chassis_past_exit']}</p>
<p>종료 이유: {html.escape(reason)}<br>평가용 실제 최대 회전각: {html.escape(angle_summary)}</p>
<p>외부 Gazebo / 전방 카메라 / 실제 SLAM·문 기억 지도. 동일 실행만 사용.</p>
<p><a href="README_녹화결과.md">검증 조건과 한계</a> · <a href="records/result.json">판정 원본</a> · <a href="recording_summary.json">요약</a></p>
{video_cards}<section><h2>단계별 기록</h2><div class="photos">{photo_cards}</div></section></html>'''
    (root/'00_녹화결과보기.html').write_text(page, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
