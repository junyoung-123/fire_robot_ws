"""Package verified r29 media while keeping historical evidence separate."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from urllib.parse import quote
from zipfile import ZipFile, ZIP_DEFLATED

DATE = '2026-09-29'
BRANCH = 'codex/observed-nav-20260928'
GITHUB = 'https://github.com/junyoung-123/fire_robot_ws'
SCOPE = 'NAV_SLAM_R29_20260929'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def dump(file, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)


def publish(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.previous / '01_ASSET_INDEX.json').read_text(encoding='utf-8'))
    suite = json.loads((args.results / 'suite_r29.json').read_text(encoding='utf-8'))
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    assert suite['status'] == 'ALL_FIVE_PASS_NAVIGATION_ONLY' and audit['pass']
    assert suite['same_control_source'] and suite['same_recorded_harness']
    assert not suite['prior_map'] and not suite['ground_truth_control'] and not suite['physical_opening_tested']
    assets = []
    for old in original['assets']:
        asset = dict(old)
        source = args.previous / asset['file']
        assert sha(source) == asset['sha256']
        if asset['scope_id'] == 'NAV_20260916':
            asset['file'] = 'archive/images/NAV_20260916/' + source.name
            asset['usage_status'] = 'historical_reference'
            asset['limitations'] = '이전 저장 지도·sim_odom·단순 힌지 backend 주행. 최신 온라인 SLAM 증거로 사용하지 않음. ' + asset['limitations']
        asset['packaged_date'] = DATE
        asset['slide_pages_basis'] = 'legacy feedback deck, not final 40-slide deck'
        copy(source, out / asset['file'])
        assets.append(asset)
        old_note = args.previous / 'asset_notes' / f"{asset['id']}.md"
        if old_note.exists():
            note = old_note.read_text(encoding='utf-8')
            note = f"# {DATE} 자료 사용 범위\n\n상태: {asset['usage_status']}\n\n현재 파일: `{asset['file']}`\n\n{asset['limitations']}\n\n---\n\n" + note
            dest = out / 'asset_notes' / old_note.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(note, encoding='utf-8')
    for file in (args.previous / 'evidence').rglob('*'):
        if file.is_file():
            copy(file, out / 'evidence/historical_20260926' / file.relative_to(args.previous/'evidence'))
    copy(args.previous/'05_PRESENTATION_GUIDE.pdf', out/'archive/05_PRESENTATION_GUIDE_20260926.pdf')

    def add(identifier, source, title, kind, limitations, pages, category='08_online_slam_navigation'):
        file = Path('assets/images') / category / (identifier + source.suffix)
        copy(source, out/file)
        assets.append(dict(id=identifier, file=file.as_posix(), media_type='image', title=title,
                           category=category, keywords=['관측 기반', '온라인 SLAM', '사전 지도 없음', 'r29', identifier, title],
                           slide_pages=pages, slide_pages_basis='final 40-slide deck', evidence_date=DATE,
                           packaged_date=DATE, version='r29', scope_id=SCOPE, usage_status='preferred',
                           priority='high', caption=title, supports=title, limitations=limitations,
                           kind=kind, sha256=sha(source), bytes=source.stat().st_size,
                           source_in_project='00_최종자료/'+args.results.name+'/'+source.relative_to(args.results).as_posix()))
        note = f"# {identifier}: {title}\n\n- 파일: `{file.as_posix()}`\n- 실행: {DATE} / r29\n- 유형: {kind}\n- 권장 페이지: {pages}\n\n{limitations}\n\n원본 SHA256: `{sha(source)}`\n"
        (out/'asset_notes'/f'{identifier}.md').write_text(note,encoding='utf-8')

    nav_limit = '주행·문 방문·방향 정렬 시험. 로봇팔 도달 거리와 물리적 문 개방은 미검증. 월드별 1회 결과.'
    truth_limit = '월드 배치와 정답 로봇 궤적은 사후 평가용이며 제어에 전달하지 않음. 별표는 개방이 아닌 방문. ' + nav_limit
    map_limit = '실제 LiDAR/휠 odometry 온라인 SLAM 기록. 회색=미관측, 흰색=빈 공간, 검정=점유. 시작점 map 좌표계이며 평가 궤적의 world 원점과 다름. ' + nav_limit
    add('N_all_paths',args.results/'01_world_trajectories_r29.png','월드 1~5 실제 주행 궤적과 실험 배치','evaluation_plot',truth_limit,[30])
    add('N_all_slam',args.results/'00_slam_map_evolution_r29.png','월드 1~5 초기·중간·마지막 SLAM 지도','sensor_map_plot',map_limit,[28,30])
    specs = [('path','evaluation_world_trajectory.png','주행 궤적과 월드 배치','evaluation_plot',truth_limit),
             ('map_initial','observed_map_initial.png','초기 SLAM 지도','sensor_map_plot',map_limit),
             ('map_middle','observed_map_middle.png','중간 SLAM 지도','sensor_map_plot',map_limit),
             ('map_latest','observed_map_latest.png','마지막 저장 SLAM 지도','sensor_map_plot',map_limit),
             ('map_progress','observed_map_progress.png','SLAM 지도 생성 과정','sensor_map_plot',map_limit),
             ('alignment','door_visit_alignment.png','문 방문 위치와 방향','evaluation_plot',truth_limit),
             ('camera','camera_observations.png','정면 카메라 관측 모음','recorded_camera',nav_limit),
             ('lidar','lidar_and_live_costmap.png','LiDAR 스캔과 실시간 코스트맵','sensor_plot',nav_limit),
             ('error','localization_error.png','위치추정 오차','evaluation_plot',truth_limit),
             ('start','evidence/camera_000.jpg','시작 정면 카메라 원본','recorded_camera',nav_limit)]
    for w in range(1,6):
        directory = args.results/f'world{w}_r29'
        for suffix, file, title, kind, limit in specs:
            if w == 4 and suffix == 'alignment':
                # No blue-door visit occurred, so no alignment plot exists.
                continue
            add(f'N{w}_{suffix}',directory/file,f'World {w}: {title}',kind,limit,[30+w])
        shutil.copytree(directory,out/'evidence/nav_r29'/directory.name,dirs_exist_ok=True)
    for filename in ['suite_r29.json','README_최신검증_r29.md','00_최신결과.html','00_slam_map_evolution_r29.png','01_world_trajectories_r29.png']:
        copy(args.results/filename,out/'evidence/nav_r29'/filename)
    (out/'evidence/nav_r29/00_검증결과.html').write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>이전 기록 보존 안내</title>'
        '<h1>이전 실행·실패 기록</h1><p>이 전달 묶음에는 최종 r29의 다섯 실행 원본과 기존 발표용 참고 자료를 포함합니다. '
        'r29 이전의 모든 반복 실행 로그는 용량 때문에 이 ZIP에 포함하지 않았으며 원래 프로젝트 폴더에 보존했습니다.</p>'
        '<p>보존 위치: <code>00_최종자료/36_열린출구_관측주행검증_20260929/00_검증결과.html</code></p>'
        '<p><a href="../../00_GALLERY.html">이 자료집의 이전·실패·참고 이미지 보기</a></p></html>',encoding='utf-8')
    copy(args.audit,out/'evidence/release_source_audit.json')
    dump(out/'evidence/publication_regression.json',{
        'date':DATE,'environment':'Ubuntu2204Recovered / ROS2 Humble',
        'command':'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/test_observed_*.py scripts/test_motion_scan_filter.py scripts/test_open_exit_world.py scripts/test_slam_correction_tf.py -q',
        'result':'96 passed in 4.10s','new_gazebo_run':False,
        'environment_notes':['External pytest plugin autoload disabled after anyio/system-pytest incompatibility.',
                             'install setup reported missing optional fire_robot_sim_evidence/local_setup.bash; selected tests still collected and passed.']})
    dump(out/'01_ASSET_INDEX.json',dict(package_version=DATE,github_repo=GITHUB,source_branch=BRANCH,
         latest_navigation_scope=SCOPE,new_simulation_run_during_packaging=False,
         not_applicable={'N4_alignment':'World 4 has no blue doors and no blue-door visits.'},assets=assets))
    fields=['id','file','media_type','title','evidence_date','version','scope_id','usage_status','slide_pages','slide_pages_basis','kind','caption','limitations','sha256','source_in_project']
    with (out/'01_ASSET_INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for asset in assets:
            writer.writerow({k:'; '.join(map(str,asset[k])) if isinstance(asset.get(k),list) else asset.get(k,'') for k in fields})
    oldscope=(args.previous/'02_EVIDENCE_SCOPE.md').read_text(encoding='utf-8').rstrip()
    (out/'02_EVIDENCE_SCOPE.md').write_text(f'''# 최신 검증 범위 ({DATE})

## {SCOPE}

매번 저장 지도 없이 시작한 World 1~5 주행이다. 사전 문 개수·문/장애물 좌표·시뮬레이터 위치는 제어에서 제외했다.
LiDAR, 휠 오도메트리, 온라인 SLAM과 카메라 관측으로 문 후보를 누적했다. 문 기억과 재선택 제외 정책은 유지했다.
실제 문과 장애물은 유지하고 출구만 초록 표식의 열린 통로로 바꿨다.

유효 방문 3/3, 3/3, 4/4, 0/0, 6/6, 모든 출구 통과. 분모는 평가자만 아는 문 개수다.
방문 기준은 진행축 오차 0.70 m, 거리 1.60 m, 벽면 정면 방향 오차 15도 이내다.
문 방문 어댑터를 사용하므로 이번 실행에서 문을 물리적으로 열지 않았다. `DOOR_OPENED` 로그도 방문 완료를 뜻한다.
이 자료를 PIPER 정밀 주차, 손잡이 조작 또는 다문 통합 물리 개방의 증거로 쓰면 안 된다.

동일 r29 소스로 월드별 1회 완료한 결과다. W3 재접근과 W4 긴 회피 재시도는 남아 있다.
최대 위치추정 오차는 월드별 0.52~0.79 m이며 최적 궤적·무오차·실물 성공·반복 성공률을 입증하지 않는다.
접촉 센서 충돌 시험이 아닌 기록된 Nav2 외곽선 겹침 검사다.

## 날짜가 다른 자료의 결합 금지

- N* = 최신 온라인 SLAM 주행. 실제 제어 지도를 보려면 N*_map_* 사용.
- W* = 이전 저장 지도 주행. archive에서 참고용으로만 제공.
- OV1/OV2와 R34 = 기존 단일문 센서 기반 접촉 개방. 최신 N* 주행과 같은 실행이 아니다.
- M*/Q*/PV2 = 기존 문 기억 시연. 빈 지도 SLAM 성장의 증거로 바꾸어 설명하지 않는다.
- E1과 실제 손잡이 평가 수치는 기존 평가 범위를 유지한다. 새 모델 평가를 수행한 것이 아니다.

---

## 기존 자료의 상세 범위 (2026-09-26 보존)

{oldscope}
''',encoding='utf-8')
    (out/'00_START_HERE.md').write_text(f'''# 최신 이미지·영상·실험결과 자료집

정리 날짜: {DATE}. 최신 주행은 r29 온라인 SLAM, 물리 개방은 별도 R34 단일문이다.

1. `02_EVIDENCE_SCOPE.md`에서 시험 범위를 먼저 읽는다.
2. `03_TOPIC_AND_SLIDE_MAP.md`에서 발표 페이지에 맞는 ID를 찾는다.
3. `01_ASSET_INDEX.csv` 또는 JSON에서 ID·날짜·버전·설명을 검색한다.
4. `assets/` 원본과 `asset_notes/ID.md`를 함께 사용한다.
5. `00_GALLERY.html`을 브라우저로 열면 이미지와 영상을 확인할 수 있다.

## 먼저 사용할 자료

- World 1~5 궤적: `N1_path`~`N5_path`, 시작 카메라: `N1_start`~`N5_start`.
- 지도 생성 과정: `N_all_slam`, 각 `N*_map_initial`, `N*_map_middle`, `N*_map_latest`.
- World 4는 파란 문이 없어 방문·정렬 그래프가 없다. `N4_alignment`는 해당 없음이다.
- 통합 궤적: `N_all_paths`. 별표는 문 방문 위치이며 물리 개방 표시가 아니다.
- 단일문 접촉 개방 영상: 기존 `OV1`, `OV2`의 노트·범위를 확인한다. 주행 기록과 이어 붙여 한 실행으로 표현하지 않는다.
- 실제 근거: `evidence/nav_r29/00_최신결과.html`, 각 world*_r29 폴더의 JSON·카메라·지도·로그.

전체 자료 {len(assets)}개. 이전 W* 주행은 `archive/images/NAV_20260916/`로 구분했다.
이전 폴더와 보고서 원본은 그대로 보존했다. 과거 PDF 안내서는 archive에 있으며 최신 결과 설명서는 이 폴더의 Markdown이다.

## 코드

- 저장소: {GITHUB}
- 최신 주행: `{BRANCH}`
- 최신 결과: {GITHUB}/blob/{BRANCH}/docs/RELEASE_20260929.md
- R34 보존: `codex/observed-door-angle-20260922`, `27359f7`
- `master` 소스는 이전 기준선이다. 새 주행은 observed_navigation 실행 경로를 사용한다.

시뮬레이터 정답 배치·위치를 제어 입력으로 쓰지 않았다는 것과 모든 파라미터가 없다는 것은 다르다.
서로 다른 시험의 사진을 같은 연속 시연으로 합치거나, 주행 성공을 다문 물리 개방 성공으로 바꾸지 않는다.
발표에 쓴 ID와 파일명을 남겨 추적 가능하게 한다. 기존 사진의 라이선스·출처는 각 자료 노트를 유지한다.
''',encoding='utf-8')
    (out/'03_TOPIC_AND_SLIDE_MAP.md').write_text('''# 발표 40쪽 기준 자료 선택

| 페이지 | 목적 | 권장 ID |
|---|---|---|
| 28 | 시험 조건 | N_all_slam, 검증 범위 문서 |
| 30 | World 1~5 요약 | N_all_paths 또는 N1_path~N5_path |
| 31 | World 1 | N1_start, N1_path |
| 32 | World 2 | N2_start, N2_path |
| 33 | World 3 | N3_start, N3_path |
| 34 | World 4 | N4_start, N4_path |
| 35 | World 5 | N5_start, N5_path |
| 추가 설명 | 실제 SLAM 성장 | N*_map_initial, N*_map_middle, N*_map_latest |
| 추가 설명 | 스캔·회피·정렬 한계 | N*_lidar, N*_alignment, N*_error |
| 21~26, 36 | 단일문 접촉 개방 | 기존 O*, OV*, E2. R34 별도 시험 |

N*의 페이지 번호는 최종 40쪽 발표 기준이다. 기존 자료의 slide_pages는 과거 피드백 PPT 기준이므로
페이지 숫자보다 title, scope_id, asset_notes를 우선한다. 이미지의 초기 위치와 궤적은 같은 월드 실행에서 가져왔다.
''',encoding='utf-8')
    (out/'CLAUDE.md').write_text('''# Evidence selection

Read 00_START_HERE.md, 02_EVIDENCE_SCOPE.md and 01_ASSET_INDEX.json.
Use N* for the latest no-prior-map navigation; W* is historical saved-map evidence.
R34 opening videos and N* navigation are separate experiments, not one integrated run.
Keep scope, date, version, source IDs and attribution. Do not conceal loops or detection errors.
No physical door opening, close manipulation parking, stress repeatability or real hardware claim is supported by N*.
''',encoding='utf-8')
    (out/'04_REQUEST_FOR_CLAUDE.txt').write_text('발표 자료의 기존 틀을 유지하며 필요한 이미지·영상을 찾아 사용해 주세요. 먼저 00_START_HERE.md와 02_EVIDENCE_SCOPE.md를 읽고, 최종 40쪽 PPT의 28·30~35쪽에는 N* 자료를 사용해 주세요. W*는 이전 저장 지도 기록입니다. 단일문 R34 개방과 최신 온라인 SLAM 주행은 별도 검증으로 표시하고, 사용한 ID와 파일명·출처를 기록해 주세요.\n',encoding='utf-8')

    sections=[]
    for label, chosen in [('최신 관측 주행 r29',[a for a in assets if a['scope_id']==SCOPE]),
                          ('기존 검출·기억·단일문 개방 자료',[a for a in assets if a['scope_id']!=SCOPE and a['usage_status']=='preferred']),
                          ('이전·실패·참고 자료',[a for a in assets if a['usage_status']!='preferred'])]:
        cards=[]
        for asset in chosen:
            url=quote(asset['file'])
            media=(f'<video src="{url}" controls preload="none"></video>' if asset['media_type']=='video' else
                   f'<a href="{url}"><img src="{url}" loading="lazy" alt="{html.escape(asset["title"])}"></a>')
            cards.append(f'<article><h3>{html.escape(asset["id"])} · {html.escape(asset["title"])}</h3>{media}<p>{html.escape(asset["evidence_date"])} / {html.escape(asset["version"])}</p><p>{html.escape(asset["limitations"])}</p><a href="asset_notes/{quote(asset["id"])}.md">설명·출처</a></article>')
        sections.append(f'<section><h2>{label}</h2><div class="grid">'+''.join(cards)+'</div></section>')
    (out/'00_GALLERY.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>최신 검증 이미지·영상</title><style>body{font-family:Arial,"맑은 고딕",sans-serif;margin:32px;color:#202932;background:#fff}h1{font-size:30px}h2{margin-top:40px;border-bottom:2px solid #20745b;padding-bottom:10px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:28px}article{border:1px solid #ddd;padding:16px;min-width:0}h3{font-size:18px;overflow-wrap:anywhere}img,video{width:100%;height:250px;object-fit:contain;background:#f5f5f5}p{font-size:14px;line-height:1.65}a{color:#12644d}</style><h1>관측 주행 r29 · 이미지와 영상</h1><p>2026-09-29 / N*: 저장 지도 없는 주행·문 방문 / R34: 별도 단일문 물리 개방 / W*: 이전 저장 지도 기록</p><p><a href="00_START_HERE.md">읽는 순서</a> · <a href="02_EVIDENCE_SCOPE.md">검증 범위</a> · <a href="01_ASSET_INDEX.csv">자료 색인</a> · <a href="evidence/nav_r29/00_최신결과.html">최신 실험결과</a></p>'''+''.join(sections)+'</html>',encoding='utf-8')
    dump(out/'SHA256.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256.json'})
    assert len(assets)==len({a['id'] for a in assets})
    assert all(sha(out/a['file'])==a['sha256'] for a in assets)
    repo_handoff=args.repo/'docs/handoff'/DATE
    shutil.copytree(out,repo_handoff,dirs_exist_ok=True)
    validation=args.repo/'docs/validation'/DATE
    validation.mkdir(parents=True,exist_ok=True)
    copy(args.results/'suite_r29.json',validation/'suite_r29.json')
    readme=(args.results/'README_최신검증_r29.md').read_text(encoding='utf-8')
    prefix='''# 공개 검증 근거

- [최신 자료집](../../handoff/2026-09-29/00_START_HERE.md)
- [실험 원본과 실행별 로그](../../handoff/2026-09-29/evidence/nav_r29)
- [SLAM 지도 성장 이미지](../../handoff/2026-09-29/assets/images/08_online_slam_navigation/N_all_slam.png)
- [월드 1~5 실제 궤적](../../handoff/2026-09-29/assets/images/08_online_slam_navigation/N_all_paths.png)

아래 상대 파일명은 위 실험 원본 폴더 기준입니다. 원래 실험 파일명·판정은 유지했습니다.

'''
    (validation/'README.md').write_text(prefix+readme,encoding='utf-8')
    copy(args.ppt,args.repo/'docs/reports'/DATE/args.ppt.name)
    zip_path=out.parent/f'Claude_이미지영상_실험결과_최신_{DATE.replace("-","")}.zip'
    with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file():
                z.write(file,out.name+'/'+file.relative_to(out).as_posix())
    with ZipFile(zip_path) as z:
        assert z.testzip() is None
    print(json.dumps({'assets':len(assets),'latest_nav_assets':sum(a['scope_id']==SCOPE for a in assets),
                      'zip_bytes':zip_path.stat().st_size,'output':str(out),'zip':str(zip_path)},ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['previous','results','output','repo','audit','ppt']:
        parser.add_argument('--'+key,type=Path,required=True)
    publish(parser.parse_args())
