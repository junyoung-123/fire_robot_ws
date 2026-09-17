"""Build a dated, portable evidence report without changing robot software."""
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tmp/report_tools'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Spacer, Preformatted, Paragraph, Table, TableStyle

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
FINAL = ROOT/'00_최종자료'
NAV = FINAL/'10_통합재검증_보고서개정_20260916'
BATCH = NAV/'08_동일코드_월드1_5_검증집계_20260916'
HIST = FINAL/'02_FSM별_검증증빙'
PAIR = FINAL/'04_보고서_추가분석_20260913'
RUN = FINAL/'14_주행보존_단일문개방_20260917/contact_copy_r40_handle_held/physical_contact_door_test_20260917_032440'
WS = Path(r'\\wsl.localhost\Ubuntu2204Recovered\home\junyoung\fire_robot_ws_test')
COPY = Path(r'\\wsl.localhost\Ubuntu2204Recovered\home\junyoung\fire_robot_ws_world1_contact_20260916')
PDF_NAME = '화재대응_로봇_핵심아이디어와_검증_20260917_REV6.pdf'
spec = importlib.util.spec_from_file_location('rev4', NAV/'tools/build_report_rev4.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.styles['body'].fontSize = 11.5
base.styles['body'].leading = 18
base.styles['cell'].fontSize = 10
base.styles['cell'].leading = 15
base.styles['th'].fontSize = 10
base.styles['th'].leading = 15
base.styles['small'].fontSize = 9.2
base.styles['small'].leading = 13.8
base.styles['title'].fontSize = 22
base.styles['title'].leading = 29
base.pdfmetrics.registerFont(base.TTFont('CodeMono', 'C:/Windows/Fonts/consola.ttf'))
base.styles['code'] = ParagraphStyle('verified_code', fontName='CodeMono', fontSize=9,
                                    leading=12, backColor=colors.HexColor('#f0f3f5'),
                                    borderPadding=9, spaceBefore=7, spaceAfter=12)

pages, assets, sources, excerpts = [], {}, {}, []
current = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(path, folder, filename=None):
    path = Path(path)
    dest = OUT/folder/(filename or path.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    key = dest.relative_to(OUT).as_posix()
    assets[key] = {'original': str(path), 'sha256': sha(dest), 'bytes': dest.stat().st_size}
    return key


def old(name, folder='01_인식과지도'):
    matches = list(HIST.rglob(name))
    if len(matches) != 1:
        raise ValueError((name, matches))
    return collect(matches[0], folder)


def browser_video(key):
    original=OUT/key
    ffmpeg=ROOT/'tmp/media_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    probe=subprocess.run([str(ffmpeg),'-hide_banner','-i',str(original)],capture_output=True,
                         text=True,encoding='utf-8',errors='replace')
    if 'Video: h264' in probe.stderr:
        return key
    dest=original.with_name(original.stem+'_browser_h264.mp4')
    if not dest.exists():
        command=[str(ffmpeg),'-y','-hide_banner','-loglevel','error','-i',str(original),
                 '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                 '-fps_mode','passthrough','-an','-movflags','+faststart',str(dest)]
        subprocess.run(command,check=True,capture_output=True,timeout=180)
    newkey=dest.relative_to(OUT).as_posix()
    assets[newkey]={'original':str(original),'sha256':sha(dest),'bytes':dest.stat().st_size,
                    'type':'H.264 compatibility transcode; same frame order/rate; original retained'}
    return newkey


def page(title, subtitle, tags):
    global current
    current = dict(number=len(pages)+1, title=title, subtitle=subtitle, tags=tags, blocks=[])
    pages.append(current)


def para(text, kind='body'):
    current['blocks'].append(dict(kind=kind, text=text))


def heading(text):
    para(text, 'h')


def tab(headers, rows, weights=None):
    current['blocks'].append(dict(kind='table', headers=headers, rows=rows, weights=weights))


def fig(path, caption, height=280):
    current['blocks'].append(dict(kind='image', path=path, caption=caption, height=height))


def link(path, label):
    current['blocks'].append(dict(kind='link', path=path, text=label))


def snippet(key, start, count, explanation, after=None):
    lines = (OUT/sources[key]['copy']).read_text(encoding='utf-8').splitlines()
    lower = next(i for i, line in enumerate(lines) if after in line) if after else 0
    index = next(i for i, line in enumerate(lines) if i >= lower and start in line)
    code = textwrap.dedent('\n'.join(lines[index:index+count]))
    current['blocks'].append(dict(kind='code', text=code, source=key,
                                  first=index+1, last=index+count, explanation=explanation))
    excerpts.append(dict(page=current['number'], source=key, first=index+1,
                         last=index+count, sha256=sources[key]['sha256']))


def pseudo(code, note):
    current['blocks'].append(dict(kind='pseudo', text=textwrap.dedent(code).strip(), explanation=note))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def make_graph(run):
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 11, 'figure.facecolor': 'white'})
    samples = [e for e in run['feedback_events'] if e['event']=='handle_held_base_sample']
    verified = next(e for e in run['feedback_events'] if e['event']=='handle_held_open_verified')
    rows = samples+[verified]
    t = [e['sim_time_sec'] for e in rows]
    door = [math.degrees(e['door_delta_rad']) for e in rows]
    lever = [math.degrees(e['lever_delta_rad']) for e in rows]
    fig_, axs = plt.subplots(2, 1, figsize=(10, 6.1), sharex=True,
                              gridspec_kw={'height_ratios':[2,1]})
    axs[0].plot(t, door, color='#127c70', lw=2.2, label='측정 문 회전각')
    axs[0].axhline(math.degrees(run['min_angle_rad']), color='#c03c48', ls='--',
                   label='판정 기준 117.46도')
    axs[0].scatter([t[-1]], [door[-1]], color='#127c70', s=45, zorder=3)
    axs[0].set(ylabel='문 개방각 [deg]', title='손잡이를 놓기 전에 목표 개방각 도달 | r40 실측 사건 로그')
    axs[0].legend(loc='upper left', fontsize=10)
    axs[1].plot(t, lever, color='#806332', lw=1.6)
    axs[1].set(xlabel='시뮬레이션 시간 [s] - 벽시계/영상 시간이 아님', ylabel='레버 누름각 [deg]')
    for ax in axs:
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig_.tight_layout()
    dest=OUT/'06_문개방/feedback_sim_time.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig_.savefig(dest, dpi=200)
    plt.close(fig_)
    key=dest.relative_to(OUT).as_posix()
    assets[key]={'original':str(RUN/'result.json'), 'sha256':sha(dest),
                 'bytes':dest.stat().st_size, 'type':'measured-data plot; no interpolated success claims'}
    return key


def make_design_and_semantic_figures(worlds):
    diagram=OUT/'00_전체구조/observation_memory_action.png'
    diagram.parent.mkdir(parents=True,exist_ok=True)
    fig_,ax=plt.subplots(figsize=(11,3.1))
    ax.set(xlim=(0,11),ylim=(0,3.1))
    ax.axis('off')
    labels=[('관측','RGB · LiDAR · depth','#e5f2f0'),('의미 기억','위치 · 색 · opened','#eaf0fa'),
            ('목표 잠금','미처리 문 하나','#f4ecf6'),('이동 / 정렬','Nav2 · 제어권','#eef2e3'),
            ('개방 / 확인','피드백 · 복귀','#fbecee')]
    for i,(title,sub,color) in enumerate(labels):
        x=.1+i*2.2
        ax.add_patch(Rectangle((x,1.55),2,.93,facecolor=color,edgecolor='#87989f',lw=1))
        ax.text(x+1,2.18,title,ha='center',va='center',fontsize=14,fontweight='bold')
        ax.text(x+1,1.8,sub,ha='center',va='center',fontsize=10)
        if i<4:
            ax.add_patch(FancyArrowPatch((x+2,2.02),(x+2.2,2.02),arrowstyle='-|>',mutation_scale=13,color='#536771'))
    ax.annotate('',xy=(3.3,.8),xytext=(9.9,.8),arrowprops={'arrowstyle':'-|>','color':'#137f7a','lw':2})
    ax.plot([9.9,9.9],[1.55,.8],color='#137f7a',lw=2)
    ax.plot([3.3,3.3],[.8,1.55],color='#137f7a',lw=2)
    ax.text(6.6,.32,'개방 결과를 기억에 반영 → 다음 목표 선택',ha='center',fontsize=12,color='#137f7a')
    ax.set_title('설계 도식 | 목표는 하나, 관측과 기억은 계속',fontsize=15,loc='left',pad=8)
    fig_.tight_layout()
    fig_.savefig(diagram,dpi=200)
    plt.close(fig_)
    key=diagram.relative_to(OUT).as_posix()
    assets[key]={'original':'report explanation; not a measured experiment','sha256':sha(diagram),
                 'bytes':diagram.stat().st_size,'type':'conceptual diagram'}
    detections=list(csv.DictReader((BATCH/'world1_trace/detections.csv').open(encoding='utf-8')))
    poses=list(csv.DictReader((BATCH/'world1_trace/poses.csv').open(encoding='utf-8')))
    openings=worlds[1]['metrics']['openings']
    times=[openings[0]['t']-5,openings[1]['t']-5,float(poses[-1]['t'])]
    start=float(poses[0]['t'])
    fig_,axs=plt.subplots(3,1,figsize=(10.5,9.7),sharex=True,sharey=True)
    for idx,(ax,end) in enumerate(zip(axs,times)):
        ps=[r for r in poses if float(r['t'])<=end]
        selected=[r for r in detections if float(r['t'])<=end]
        for color,label,ink in [('red','빨강 문 관측','#d74553'),('blue','파랑 문 관측','#2487d4'),('green','초록 문 관측','#239457')]:
            ds=[r for r in selected if r['color']==color]
            ax.scatter([float(r['handle_x']) for r in ds],[float(r['handle_y']) for r in ds],s=13,alpha=.38,c=ink,label=label)
        ax.plot([float(r['x']) for r in ps],[float(r['y']) for r in ps],c='#26333a',lw=1.7,label='기록된 로봇 경로')
        ax.scatter([float(ps[-1]['x'])],[float(ps[-1]['y'])],marker='^',s=60,c='#151f23',zorder=4)
        ax.set_title(f"{idx+1}. 누적 관측 {len(selected)}회 | 기록 시작 후 {end-start:.1f}초 (벽시계)",loc='left',fontsize=12)
        ax.set(ylabel='map y [m]',ylim=(-6,6),xlim=(-4,21))
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axs[0].legend(loc='upper left',ncol=2,fontsize=9)
    axs[-1].set_xlabel('map x [m] - 보존된 sim_odom 실행 좌표계')
    fig_.suptitle('World1 | 색 의미 관측이 이동 중 누적되는 과정',fontsize=16,x=.08,ha='left')
    fig_.tight_layout(rect=(0,0,1,.965))
    dest=OUT/'01_인식과지도/world1_observed_semantic_history.png'
    fig_.savefig(dest,dpi=200)
    plt.close(fig_)
    skey=dest.relative_to(OUT).as_posix()
    assets[skey]={'original':str(BATCH/'world1_trace/detections.csv'),'sha256':sha(dest),
                  'bytes':dest.stat().st_size,'type':'raw observation CSV; no ground-truth door overlay'}
    for name in ('detections.csv','poses.csv','targets.csv','events.csv'):
        collect(BATCH/'world1_trace'/name,'07_원본판정/주행/월드1')
    return key,skey


def build_content():
    module_paths = {
        'fsm':'src/fire_robot_fsm/fire_robot_fsm/state_machine_node.py',
        'subfsm':'src/fire_robot_fsm/fire_robot_fsm/fsm_subsystems.py',
        'perception':'src/fire_robot_perception/fire_robot_perception/door_detection_node.py',
        'initial_map':'src/fire_robot_navigation/fire_robot_navigation/initial_static_map_node.py',
        'obstacle_map':'src/fire_robot_navigation/fire_robot_navigation/fixed_obstacle_map_node.py',
        'nav_config':'src/fire_robot_navigation/config/nav2_params.yaml',
        'launch':'src/fire_robot_bringup/launch/simulation.launch.py',
        'nav':'src/fire_robot_navigation/fire_robot_navigation/navigation_node.py',
        'safety':'src/fire_robot_navigation/fire_robot_navigation/cmd_vel_safety_node.py',
        'physical':'src/fire_robot_manipulation/fire_robot_manipulation/physical_contact_manipulation_node.py',
        'contact':'src/fire_robot_manipulation/fire_robot_manipulation/contact_feedback.py',
        'kinematics':'src/fire_robot_manipulation/fire_robot_manipulation/piper_actual_kinematics.py',
        'contact_perception':'src/fire_robot_perception/fire_robot_perception/door_detection_node.py',
    }
    for key, path in module_paths.items():
        root = COPY if key in ('physical','contact','kinematics','contact_perception') else WS
        copied=collect(root/path, '08_코드발췌와원본/'+('접촉복사본' if root==COPY else '주행보존본'))
        sources[key]={'path':str(root/path), 'copy':copied, 'sha256':sha(root/path)}
    # Dated media are kept byte-identical; the report supplies the interpretation.
    sensor=old('01_World1_초기센서_종합.png')
    initial=old('01_초기맵.png')
    growth=old('02_이동중_LiDAR_맵확장.png')
    semantic=old('03_파란문_의미기억.png')
    close=old('04_근접관측_좌표정밀화.png')
    doorbox=old('06_YOLO_문BBox_HSV색상인식.png')
    fusion=old('07_카메라_LiDAR_좌표융합.png')
    newdoor=old('05_새문_관측과_지도등록.png')
    lock=old('01_가장가까운_파란문_목표잠금.png','02_목표와문기억')
    opened=old('02_개방완료_ID_기억.png','02_목표와문기억')
    reobserve=old('03_재관측_개방문_후보제외.png','02_목표와문기억')
    costmap=old('03_LiDAR_장애물_costmap_반영.png','03_회피와정렬')
    navalign=collect(NAV/'figures/world1_alignment_1.png', '03_회피와정렬')
    comparison=collect(PAIR/'figures/02_v2_v3_same_frame_bbox.png','04_YOLO모델')
    comparechart=collect(PAIR/'figures/01_same_images_model_comparison.png','04_YOLO모델')
    collect(PAIR/'handle_v2_v3_paired_comparison.json','07_원본판정/모델')
    vids={
        'perception':old('08_인식과정_실행영상.mp4','09_영상'),
        'map':old('07_Gazebo_RViz_맵생성과정.mp4','09_영상'),
        'convergence':old('08_첫문_좌표수렴_영상.mp4','09_영상'),
        'memory':old('05_목표잠금_개방기억_영상.mp4','09_영상'),
        'short':collect(RUN/'gazebo_handle_held_overview_6x.mp4','09_영상','05_최신_r40_손잡이유지개방_요약6x.mp4'),
        'full':collect(RUN/'gazebo_handle_held_h264.mp4','09_영상','06_최신_r40_손잡이유지개방_전체기록.mp4'),
    }
    vids={key:browser_video(path) for key,path in vids.items()}
    ph={name:collect(RUN/name,'06_문개방') for name in
        ('single_door_storyboard.png','yolo_primary_detection.png','before_overhead.png',
         'after_overhead.png','after_perspective.png','door_hinge_angle.png')}
    for name in ('closeup_00064.50_press_handle.png','closeup_00131.50_handle_held_base_open.png',
                 'closeup_00192.50_handle_held_base_open.png','closeup_00204.50_complete.png'):
        ph[name]=collect(RUN/'phases'/name,'06_문개방')
    for name in ('result.json','contact_safety_audit.json','recovery_evidence.json',
                 'contact_diagnostics.json','motion_diagnostics.json','launch.log','storyboard_sources.json'):
        collect(RUN/name,'07_원본판정/단일문_r40')
    collect(BATCH/'results.tsv','07_원본판정/주행')
    collect(BATCH/'validation_set_manifest.json','07_원본판정/주행')
    collect(NAV/'figures/closed_panel_alignment_reference.json','07_원본판정/주행')
    collect(FINAL/'14_주행보존_단일문개방_20260917/r39_r40_source_comparison.json','07_원본판정/단일문_r40')
    collect(RUN.parent/'source_snapshot_with_evidence_plugin/manifest.json','07_원본판정/단일문_r40','source_manifest.json')
    run=load(RUN/'result.json')
    recovery=load(RUN/'recovery_evidence.json')
    worlds={n:base.read_world(BATCH,n) for n in range(1,6)}
    references=load(NAV/'figures/closed_panel_alignment_reference.json')['worlds']
    for n in worlds:
        collect(BATCH/f'world{n}_trace/mission_evidence_final_metrics.json',f'07_원본판정/주행/월드{n}')
        collect(BATCH/f'world{n}.log',f'07_원본판정/주행/월드{n}')
    graph=make_graph(run)
    design,semantic_history=make_design_and_semantic_figures(worlds)
    peak=math.degrees(run['max_delta_rad'])
    held=next(e for e in run['feedback_events'] if e['event']=='handle_held_open_verified')

    page('관측하고, 기억하고, 실행 결과로 확인한다', '화재 대응 모바일 매니퓰레이터 | 핵심 설계·증거 보고서 REV6 | 2026-09-17', ['overview'])
    para('카메라가 문의 의미를 찾고, 거리 센서가 위치를 보완한다. 지도 위에 문을 기억하여 다음 행동을 정하고, 개방은 명령이 아니라 측정된 반응으로 판단한다.','h')
    fig(ph['single_door_storyboard.png'],'최신 단일문 시험 r40의 실제 Gazebo 렌더링과 계측 요약. 주행 전체 검증과는 별도 시험이다.',300)
    tab(['검증 축','현재 근거','발표 시 표현'],[
        ['관측·의미 기억','2026-09-09~10 보존 이미지/영상','지도 확장·문 기억의 동작 예시'],
        ['월드 1~5 주행','2026-09-16, 5개 월드 검사 PASS','저장 지도 기반 주행 회귀 성공'],
        ['손잡이 유지 개방','2026-09-17 r40, PASS','단일 fixture의 접촉 개방·복귀 성공'],
    ],[1,1.4,1.6])
    para('기존 성공 코드·REV4/REV5·과거 원본은 보존했다. 이번 작업은 보고서 재구성과 증거 재분석이며 새 전체 주행을 수행했다고 표시하지 않는다.','small')

    page('보고서 읽는 순서', '설명용 도식과 실제 계측을 구분하면 핵심이 선명해진다', ['index'])
    tab(['질문','확인할 장'],[
        ['어떻게 처음 보는 환경을 지도화하는가?','초기 센서 → 초기 맵 → 관측 확장 → 고정층과 갱신층'],
        ['어떻게 문을 구별하고 다시 열지 않는가?','문 YOLO/HSV → 의미 좌표 → 목표 잠금 → 개방 기억 → 재관측 제외'],
        ['어떻게 장애물을 피하고 문 앞으로 가는가?','회피 우선순위 → Nav2 → 제어권 → 정렬'],
        ['어디까지 관측 기반이고 무엇이 아직 고정인가?','변경 전후 비교 → IK 접근 → 누름 피드백 → 남은 설정값'],
        ['어떤 실험이 성공했는가?','월드 1~5 각각의 주행 근거 → 최신 단일문 개방 → 한계'],
    ],[1.4,2.5])
    heading('증거 유형')
    para('실제 영상/렌더: Gazebo 카메라로 받은 픽셀이다. 진단 화면: 실제 ROS 데이터와 카메라를 조합한 표시이며 Gazebo GUI 원본 화면과 다르다. 계측 그래프: 로그 수치를 다시 그린 것이다. 설명 도식/의사코드: 구조 설명용이며 성공의 독립 증거가 아니다.')
    heading('판정 범위')
    para('PASS는 해당 날짜·코드·설정·검사 항목에서 통과했다는 뜻이다. 센서 예시, 주행 회귀, 단일문 접촉을 묶어서 “실제 로봇이 미지의 5개 월드에서 모두 문을 열었다”라고 표현하지 않는다.')
    para('영상은 PDF에 재생기를 내장하지 않고 같은 폴더의 MP4와 연결했다. 공유 ZIP을 압축 해제한 뒤 START_보고서와영상.html을 열면 장별 이미지·코드·영상을 함께 볼 수 있다.')

    page('우리 시스템의 핵심 구조', '다섯 책임을 분리하고, 관측은 이동·조작 중에도 계속 쌓는다', ['fsm','ideas'])
    fig(design,'설명 도식. 로봇이 실제로 실행한 특정 시점의 캡처가 아니라 코드 책임의 연결 관계다.',150)
    tab(['대 단계','핵심 책임','다음 단계로 가는 조건'],[
        ['1 Perception','문 bbox·색·거리·손잡이·TF를 결합','유효한 관측 자료가 준비됨'],
        ['2 Memory / Target','문 위치·색 투표·상태를 기억하고 하나 선택','확정 가능한 미처리 파란문을 잠금'],
        ['3 Navigation / Align','장애물 비용을 반영해 이동하고 자세 조정','근접 관측·자세·안전 조건 충족'],
        ['4 Manipulation','파지·레버 누름·손잡이 유지 개방·복귀','해당 backend의 개방 결과 확인'],
        ['5 Next / Exit','opened 기록, 다음 후보 또는 출구','미처리 후보 재검토 후 종료'],
    ],[.95,1.65,1.45])
    heading('왜 목표 잠금과 계속 관측이 동시에 필요한가?')
    para('새 파란문이 보일 때마다 목표를 바꾸면 경로가 흔들린다. 반대로 관측 자체를 중단하면 이미 본 다음 문을 잊는다. 따라서 현재 실행 목표 하나는 유지하되, 나머지 문 관측은 지도와 기억에 축적한다.')
    para('기여점은 새 YOLO/Nav2 알고리즘의 발명보다, 의미 기억·목표 고정·제어권 분리·측정 기반 전이 조건을 연결한 통합 설계에 있다. 기존 라이브러리와 자체 정책을 구분하여 설명한다.')

    page('관측값과 시험 정답의 경계', '센서로 얻은 정보와 평가자가 알고 있는 월드 정보를 섞지 않는다', ['observation','limits'])
    tab(['정보','사용 위치','해석'],[
        ['RGB / scan / 등록 depth / TF','문·장애물·손잡이 위치 추정','로봇 입력 경로'],
        ['landmark / opened 상태','목표 제외·다음 문·출구 전환','관측에서 만든 작업 기억'],
        ['저장 지도 / sim_odom','보존 월드 1~5의 위치·경로 계산','그 실행이 미지 환경 최초 SLAM은 아님'],
        ['SDF 문·장애물 위치 / 정답 문 수','주행 그림·checker·기존 hinge backend 매칭','평가/시뮬레이션 지원 정보'],
        ['레버·문 Gazebo joint state','최신 단일문 피드백과 성공 판정','계측된 시뮬레이션 입력, 실물 힘 센서 아님'],
    ],[1.15,1.4,1.55])
    heading('관측 기반이라는 표현의 정확한 범위')
    para('문 후보와 손잡이 픽셀은 관측으로 생성된다. 그러나 시험에서 제공한 지도, 시뮬레이션 위치추정, 문 backend의 모델 매칭까지 없었다는 뜻은 아니다. 이 보고서는 이 세 수준을 별도 표시한다.')
    para('실제 로봇에서는 카메라/깊이 등록·외부 TF 교정, odometry/localization, 접촉 또는 레버 반응 센싱이 필요하다. Gazebo joint state를 그대로 실물에서 받을 수 있다고 가정하지 않는다.')

    page('1. 초기 장면과 센서 입력', '2026-09-09 | 보존된 World1 초기 센서 증거', ['initial_map','perception'])
    fig(sensor,'그림 1. 전방·전방 좌측·전방 우측 카메라와 2D LiDAR 관측을 함께 정리한 기존 증거.',315)
    tab(['입력','알 수 있는 것','알 수 없는 것'],[
        ['카메라','문 bbox, 색상, 손잡이 외형','단안 bbox만으로 신뢰할 수 있는 3D 위치'],
        ['2D LiDAR','스캔 평면의 벽·장애물 거리','문 색, 스캔 높이 밖의 모든 물체 형상'],
        ['등록 RGB-D','손잡이 픽셀에 대응한 표면 3D 점','가림 뒤 형상·접촉력'],
    ],[.8,1.65,1.65])
    para('초기 원점은 좌표계 선택일 뿐 환경 정답을 제공하지 않는다. 첫 장면에서 가려진 공간은 아직 모른다. 세 카메라의 배치가 실제 장착과 일치하는지는 실물 외부 파라미터 측정으로 별도 확인해야 한다.')

    page('2. 처음 가진 지도는 부분 지도다', '2026-09-10 | 실시간 SLAM 진단 화면 t=0.2초', ['initial_map'])
    fig(initial,'그림 2. LIVE GAZEBO ROS DATA에서 첫 SLAM grid 191×81, locked_cells 297이 표시된다.',315)
    tab(['화면 요소','읽는 방법'],[
        ['밝은/어두운 격자','점유·자유·미관측 공간을 구분하는 OccupancyGrid'],
        ['청록 점','그 시점의 LiDAR 거리 관측'],
        ['자홍색 점','진단 화면이 첫 스캔에서 별도로 보존해 그린 점유 셀'],
        ['문 색 점 / 주황 선','문 관측 overlay / 기록된 로봇 이동'],
    ],[1.1,3])
    para('이 그림은 실제 ROS 데이터를 조합한 진단 화면이지 실제 Gazebo GUI 캡처는 아니다. 표시된 첫 스캔 잠금 점만으로 production /initial_static_map과 모든 셀이 같다고 단정할 수 없다. 현재 노드의 고정 동작은 다음 코드 장에서 별도로 확인한다.','small')

    page('3. 이동하면 새 공간이 드러난다', '같은 관측 과정의 기록 | t=16.2초 → t=29.0초', ['map_growth','color_map'])
    fig(semantic,'그림 3a. t=16.2초: live_map 200×81, 파란 관측 군집 n=3. 화면 손잡이 표시는 당시 HSV 방식이다.',218)
    fig(growth,'그림 3b. t=29.0초: live_map 211×81, 빨강/파랑 관측 군집과 잠긴 파란 목표가 함께 보인다.',218)
    para('첫 191×81 grid보다 표시 범위가 커지고, 보이지 않던 구역에서 LiDAR와 문 관측이 추가된다. 이는 지도와 의미 관측이 이동 중 갱신된 사례다. 격자 크기 증가만으로 모든 방을 완전히 관측했다거나 기존 셀 오차가 전혀 없다는 결론은 낼 수 없다.')
    link(vids['perception'],'영상: 실제 관측·문 좌표 진단 과정 (2026-09-10)')

    page('4. 고정층과 갱신층을 분리한다', '현재 코드 확인 | 한 장의 절대 정답 지도를 계속 덮어쓰는 구조가 아니다', ['map_growth','code'])
    tab(['층','역할','현 구현의 주의점'],[
        ['/initial_static_map','첫 유효 지도를 복사해 고정 참조로 유지','lock 후 source /map 수신을 무시'],
        ['/map','SLAM 모드에서는 새로운 관측으로 갱신','SLAM 자체의 기존 영역 보정은 가능'],
        ['/fixed_obstacle_map','scan+TF로 별도 장애물 overlay 누적','grid geometry가 바뀌면 overlay 재초기화'],
        ['문 의미 기억','색·위치·opened를 별도 구조체로 관리','OccupancyGrid의 색 픽셀 자체가 아님'],
    ],[1.05,1.4,1.7])
    snippet('initial_map','    def _map_callback(self, msg: OccupancyGrid):',6,
            '초기 캡처 완료·잠금 활성·fallback 아님이면 후속 source map을 받지 않는다.')
    snippet('obstacle_map','                lx = math.cos(angle) * distance',4,
            '스캔 평면의 점을 현재 scan TF로 map 좌표로 옮긴 뒤 overlay에 반영한다.')
    para('중요한 구현 한계: “한 번 본 모든 셀은 영구 불변이고 미관측 셀만 확장”이 전체 맵에 완벽히 구현됐다고 말하면 안 된다. 초기 참조 보존과 live 지도/별도 overlay 갱신이 구현된 것이다.','small')

    page('5. 문은 YOLO, 색은 HSV', 'Door 1-class 검출과 색상 분류를 분리하여 문제를 나눈다', ['door_yolo','color_map'])
    fig(doorbox,'그림 5. 2026-09-10 보존 문 bbox·색 분류 자료. 각 bbox의 점수는 검출 confidence이며 정답 보증이 아니다.',300)
    para('문 YOLO는 Door 범주를 찾는다. 그 bbox 안의 색 분포를 HSV로 분류하여 red/blue/green 의미를 붙인다. 파란문은 작업 대상, 빨간문은 위험 회피 참고, 초록문은 출구 후보로 구분한다.')
    tab(['단계','왜 분리했는가'],[
        ['형태: YOLO','단순 색 덩어리와 문 형태를 구분하기 위한 단서'],
        ['색: HSV','Door 학습 모델을 유지하면서 색에 따른 임무 정책 변경'],
        ['시간: 반복 관측','한 프레임의 오탐·색 흔들림을 문 기억에서 재검토'],
    ],[.9,3])
    para('이 이미지에는 검은 장애물 위 낮은 점수 Door 0.27 오탐도 보인다. 우측 handle:hsv는 과거 손잡이 fallback이므로 최신 손잡이 YOLO 증거로 사용하지 않는다. 모델이 완벽하다는 사례가 아니라 처리 과정의 실제 예시다.','small')

    page('6. 색 점은 LiDAR가 본 색이 아니다', '카메라 의미 + 거리 관측 + TF → 지도 위 문 관측', ['color_map','coordinates'])
    fig(fusion,'그림 6. 카메라와 거리 정보의 좌표 융합을 설명하는 보존 자료. 렌더 위 주석/도식은 설명용이다.',310)
    para('청록 LiDAR 점은 거리 센서의 공간 표본이다. 빨강·파랑·초록 문 점은 카메라에서 분류한 의미를 추정 위치에 붙인 관측 기록이다. 색 점 수는 문 개수가 아니며, 같은 문을 여러 번 보면 여러 점이 생길 수 있다.')
    heading('2D 문 위치 계산의 기본 관계')
    pseudo('''p_scan = [range * cos(theta), range * sin(theta)]
p_map = T_map_scan(time) * p_scan
door_observation = {p_map, color, confidence, time}''','거리와 방향을 같은 시점·같은 기준계로 맞춘다는 의미의 설명식. 실제 코드는 카메라별 TF, 거리 일치 검사, fallback 분기를 포함한다.')
    para('점이 흩어지는 원인은 픽셀 bbox 변화, 잘못 연결된 거리, 가림, TF/위치추정 오차 등이다. 그림만 보고 특정 원인 하나로 확정하지 않고 관측 시각·입력 방법·변환 프레임을 함께 기록한다.')

    page('6a. 실제 색 관측 CSV의 누적 지도', 'World1 | 2026-09-16 보존 주행 자료를 이번 개정에서 재시각화', ['color_map','memory','coordinates'])
    fig(semantic_history,'그림 6a. 각 시점까지 수신된 handle_x/y 관측 전부와 로봇 pose 경로. 색 점은 확정 문 개수나 정답 문 위치가 아니다.',425)
    para('최종 기록은 파랑 407회·빨강 275회·초록 173회, 총 855회 관측이다. 이동하면서 의미 관측이 추가되는 모습과 흩어짐을 그대로 남겼다. y가 벽에서 벗어난 관측도 숨기지 않았다.')
    para('이 그림은 실측 관측 로그의 사후 표시다. 저장 지도+sim_odom 조건이며, 온라인 SLAM 전체 확장 또는 모든 점의 수렴 완료를 입증하는 그림이 아니다. 추정 좌표·주차 목표·실제 문 위치는 서로 다르다.','small')

    page('7. 가까워지며 위치를 보완한다', '좌표 안정성과 실제 정답 위치 정확도는 다른 주장이다', ['coordinates','limits'])
    fig(close,'그림 7. 첫 파란문 근접 관측의 보존 진단 화면. 추정 문 위치·주차 목표·로봇 거리·표본 분산을 따로 읽는다.',310)
    para('초기 의미 관측 화면에서는 n=3, RMS spread 0.341m가 표시된다. 이는 소수 관측의 흩어짐이며 위치 정답 오차가 아니다. 이후 목표가 잠기더라도 같은 문에 대한 새 관측으로 좌표를 보완할 수 있다.')
    heading('완전 수렴을 주장하려면 추가로 필요한 것')
    para('충분한 유효 표본 수, 일정 시간의 작은 분산, 최근 관측 유지, 독립 기준면/실측 좌표와의 오차를 함께 제시해야 한다. 화면에서 update 1 또는 spread 0.000이 나와도 표본이 하나라면 수렴 증거가 아니다.')
    link(vids['convergence'],'영상: 첫 문 접근 중 관측 좌표 변화 (2026-09-09)')
    para('이 보고서는 “좌표가 관측으로 갱신되는 과정”을 증명하며, 모든 문이 오차 0으로 수렴했다는 표현은 사용하지 않는다.','small')

    page('8. 문을 점이 아닌 작업 기억으로', 'position + color votes + status + last_seen', ['memory','code'])
    tab(['기억 항목','왜 필요한가'],[
        ['x, y / count / confidence','반복 관측을 하나의 공간 landmark로 요약'],
        ['blue_count, red_count, green_count','단발 색 결과 대신 누적 관측으로 판단'],
        ['last_seen','오래된 추정과 최근 관측을 구분'],
        ['opened / abandoned','완료 문과 제한 횟수 이후 포기한 문을 구분'],
        ['door ID lineage / 물리 station','회전·시점 변화로 ID가 바뀌어도 같은 문 연결'],
    ],[1.45,2.65])
    snippet('fsm',"        count = int(landmark.get('count', 0.0)) + 1",8,
            '관측 횟수에 따라 위치를 갱신하고 색별 표 수와 마지막 관측 시각을 저장한다.')
    para('alpha = 1/min(count, 8)이므로 초기에는 평균처럼 반영하고, 이후에도 새 관측 가중치 1/8을 남긴다. 무한히 평균을 굳히지 않아 최신 관측을 반영할 수 있지만, 잘못된 관측을 완전히 없애는 통계적 보장은 아니다.')
    para('기억은 현재 프로세스의 임무 상태다. 재시작 후 자동 복원되는 영속 DB라고 주장하지 않는다. 관측·선택·처리 이력을 분리한 것이 핵심이다.')

    page('9. 가장 가까운 미처리 문을 잠근다', '다음 후보를 관측해도 현재 처리할 문이 수시로 바뀌지 않도록', ['target','memory'])
    fig(lock,'그림 9. 보존된 목표 잠금 증거. 화면의 좌표/ID는 해당 실행에서 생성된 값이며 다음 월드에 복사하는 좌표가 아니다.',295)
    para('열리지 않았고, 포기 상태가 아니며, 관측 신뢰·색 충돌·공간 필터를 통과한 파란문 중 목표를 선택한다. 목표를 잠근 뒤 다른 후보는 기억에 남기되 현재 작업의 대체 목표로 즉시 쓰지 않는다.')
    para('현재 _nearest_forward_blue_door는 로봇 pose가 있으면 문 identity까지 유클리드 거리를 사용한다. 출구 근처 불확실 후보에는 우선순위 penalty가 있고, pose가 없으면 진행축 값을 사용한다. 따라서 “항상 경로 비용의 전역 최단 문”을 고르는 알고리즘은 아니다.')
    link(vids['memory'],'영상: 목표 잠금 → 개방 기억 → 재관측 제외')

    page('10. 열린 문을 기억하는 장면', '2026-09-09 보존 자료 | World5 성공 기록 재구성', ['opened_memory'])
    fig(opened,'그림 10. 개방 결과 후 저장된 문 ID와 열린 station의 표시. 지도 위 마커는 별도 의미 기억 overlay다.',325)
    para('개방 결과를 받으면 ID 집합과 지도상 물리 station을 기록한다. 이후 개별 카메라가 새 ID를 발급하더라도 같은 위치/같은 벽면 관계라면 이미 처리한 문으로 연결할 수 있다.')
    heading('중요한 구분')
    para('이 장의 “개방 완료”는 당시 주행 backend 결과다. 최신 PIPER 물리 접촉 성공 여부와는 별도다. 기억 로직의 동작을 보여주기 위해 보존했으며, 과거 장면을 새 r40 접촉 시험 장면으로 바꾸어 설명하지 않는다.')
    para('그림에 보이는 fixed map은 해당 sim_odom 실행에서 쓰인 저장 OccupancyGrid다. 앞의 live SLAM 확장 그림과 실행 조건이 다르다.','small')

    page('11. 다시 봐도 목표로 삼지 않았다', '동일 실행의 재관측 사건이 핵심 증거다', ['reobserve','opened_memory'])
    fig(reobserve,'그림 11. t=188.0초: OPENED DOOR IS NOT TARGETED AGAIN. 카메라 재관측과 FSM 제외 사건을 함께 표시한다.',325)
    tab(['항목','기록'],[
        ['저장된 opened station','observed_blue_5p3_m1p7'],
        ['다시 관측된 ID','door_blue_6e60a2 → IGNORED'],
        ['당시 상태 / 활성 목표','EXPLORING / NONE'],
    ],[1.2,2.9])
    para('ID가 달라졌는데도 같은 문으로 제외된 사례라는 점이 중요하다. 한 프레임에서 목표가 없다는 사실만이 아니라, 열린 station과 재관측 ID를 연결한 제외 기록이 증거다. 다만 이 한 사례가 모든 위치오차 상황의 재선택 방지를 보장하지는 않는다.')

    page('12. 재선택 방지 실제 코드', '관측 callback의 후보 등록 전에 완료 문을 차단한다', ['reobserve','code'])
    snippet('fsm',"        elif msg.door_color == 'blue':",13,
            'opened 판정을 먼저 하고 기존 중복 후보를 지운 뒤 return한다. 아래 신규 후보 등록까지 내려가지 않는다.')
    snippet('fsm','    def _is_door_opened_for_observation(self, door: DoorInfo) -> bool:',12,
            'ID만 보는 것이 아니라 공간 station·관측 위치·투영 잔여 후보 관련 조건도 검사한다.')
    heading('이 방식의 장점과 위험')
    para('장점: 카메라 방향이 바뀌어 ID가 변해도 중복 처리 방지가 가능하다. 위험: 병합 범위를 너무 크게 잡으면 가까운 다른 문을 이미 연 문으로 오인할 수 있다. 현재 코드는 위치·벽면 관계와 추가 필터를 사용하며, 이 병합 거리 역시 조정 가능한 휴리스틱이다.')
    para('문 개수를 미리 정해서 횟수만 채우는 구조와 다르다. 단, “휴리스틱이 전혀 없다”는 주장은 현재 코드와 맞지 않는다.','small')

    page('12a. 최신 주행 로그에서도 재확인', '2026-09-16 World5 | 원본 world5.log의 사건 순서', ['reobserve','code','opened_memory'])
    tab(['원본 라인','기록된 사건','의미'],[
        ['513','door_blue4, angle=2.1000, target=2.0944','그 backend에서 실제 문 모델 회전 확인'],
        ['525~528','door_blue_b177f7 개방 성공 → DOOR_OPENED → EXPLORING','opened 기록 후 다음 탐색'],
        ['537','door_blue_15a45f ignored: physical/opened door station','다른 ID의 재관측도 처리 완료 station으로 제외'],
        ['540~541','YOLO handle observed (primary), conf=0.36','같은 주행 로그에 주 모델 손잡이 검출도 존재'],
    ],[.6,2,1.6])
    para('재관측 제외 로그의 좌표는 (-0.55, 1.76)m다. 이 값은 해당 실행의 관측 위치이며 새 목표로 하드코딩해 넣은 값이 아니다. “이미 열었다”라는 기억이 후보 등록 전에 적용되었음을 보여준다.')
    heading('함께 드러나는 사실')
    para('같은 구간에 HSV handle observed도 여러 차례 기록되어 있다. 그러므로 보존 주행이 매 프레임 v3 YOLO만으로 손잡이를 찾았다고 설명하지 않는다. 최신 r40은 주 모델 기원 관측을 요구하는 별도 조건이다.')
    link('07_원본판정/주행/월드5/world5.log','원본 로그 열기: World5 (라인 513~541)')
    para('이 사건을 예전 t=188.0초 영상의 동일 실행이라고 섞지 않는다. 두 날짜의 독립 증거가 같은 기억 정책의 동작을 보여준다.','small')

    page('13. 처음 안 보이던 문도 기록한다', '다음 문은 개방 직후에만 찾는 것이 아니다', ['map_growth','memory'])
    fig(newdoor,'그림 13. 새 문이 시야에 들어오면서 지도 위 후보로 등록되는 과거 증거 자료.',310)
    para('door_callback은 임무 상태와 별개로 관측을 기억에 반영한다. 현재 문으로 이동하는 동안에도 새 문 위치·색·신뢰도는 누적된다. 다음 단계에서는 새로 검색만 하는 것이 아니라 이미 쌓인 미처리 후보를 다시 평가한다.')
    para('열린 문 주변의 후보 제거와, 아직 열지 않은 이웃 문 보존이 동시에 필요하다. 같은 ID라도 위치가 멀면 다른 문인지 확인하고, 다른 ID라도 같은 색·근접 station이면 병합한다. 이 때문에 ID 문자열만 세는 방법보다 복잡하지만 관측 변화에 대응할 수 있다.')
    para('과거 지도 이미지·영상은 원본 그대로 제공한다. 이번 개정에서 모든 과거 녹화의 원 ROS bag을 재생한 것은 아니므로, 현재 코드 전체와 프레임 단위 일치한다고 확대하지 않는다.','small')

    page('14. 장애물 회피의 실제 우선순위', '안전 제약·탐색 편향·경로 추종은 서로 다른 층이다', ['avoidance'])
    tab(['층','현 구현','의도와 한계'],[
        ['안전 / 실행 가능성','scan·costmap·footprint·정지 조건','막힌 곳을 목표 선호 때문에 강행하지 않도록 검사'],
        ['탐색 차선 선택','여유 거리 + 색상/장애물/차선 유지 점수','파랑 방향 선호, 빨강 쪽 회피, 좌우 흔들림 완화'],
        ['목표 이동 경로','SmacPlanner2D 비용 지도 계획','벽/장애물 비용을 피해 잠긴 목표로 이동'],
        ['실시간 추종','Rotation Shim + DWB + cmd_vel 안전 계층','방향 조정·속도 선택·필요한 정지/복구'],
    ],[.95,1.6,1.55])
    para('현재 탐색 차선 정책은 “빨강에서 먼 쪽 무조건 1위, 파랑 쪽 무조건 2위” 같은 엄격한 순위표가 아니라 가중 점수와 상황별 분기다. 파란문 목표 Nav2 경로의 모든 점에 빨강 위험 비용이 직접 들어가는 것도 아니다.')
    pseudo('''score = observed_lane_clearance
score += blue_side_preference + red_avoidance_bias
score += obstacle_side_bias + keep_lane_bias
score -= short_clearance_penalty + recent_failure_penalty
choose_best_available_lane()''','실제 점수식의 역할을 요약한 의사코드. 각 상황의 조기 반환과 세부 수치는 원본 _select_explore_goal_y에 있다.')
    para('따라서 보고서에는 “관측 여유 거리와 의미 편향을 함께 사용하는 회피 정책”으로 기술한다. 최적성·완전 무정지·빨간문 접근 절대 금지를 증명한 것으로 쓰지 않는다.')

    page('15. 회피 점수와 비용 지도의 증거', '관측한 장애물의 위치가 계획 비용으로 이어져야 한다', ['avoidance','code'])
    fig(costmap,'그림 15. LiDAR 장애물 관측과 costmap 반영의 보존 설명 자료.',245)
    snippet('fsm','            if red_avoid_sign != 0.0:',7,
            '탐색 lane scoring의 빨강 반대 방향 보상과 빨강 쪽 감점. 함수 내 해당 점수 분기의 실제 코드.',
            after='        best: tuple[float, float, float, float] | None = None')
    para('이 수식의 가중치는 고정 정책 파라미터다. 관측 기반인 것은 입력되는 장애물 여유·문 방향이며, 점수 계수까지 센서에서 저절로 학습되는 것은 아니다. 경로 계획을 위해 별도의 YOLO식 planning 학습은 사용하지 않는다.')
    para('현재 costmap은 2D 스캔에 기반한다. 낮은 LiDAR가 보지 못하는 높이의 돌출물은 이 자료만으로 회피 보장이 되지 않는다. RGB가 물체를 봤다는 사실과 3D collision volume이 계획기에 들어갔다는 사실을 구분한다.','small')

    page('16. Nav2 설정과 제어권', '현재 보존 코드: SmacPlanner2D + Rotation Shim + DWB', ['navigation','code'])
    tab(['구성','역할','현재 설정 예'],[
        ['SmacPlanner2D','global costmap 위 경로 계획','cost_travel_multiplier 4.0'],
        ['Rotation Shim','큰 방향 차이가 있으면 먼저 회전','angular_dist_threshold 0.45 rad'],
        ['DWB','속도 후보 궤적을 평가해 local 추종','min_vel_x 0.0 / max_vel_x 0.32'],
        ['수동/조작 제어','정렬·예외 후진·팔 동작 시 Nav2와 분리','수동 제어 진입 전 목표 취소 요청'],
    ],[1,1.45,1.55])
    snippet('nav_config','    FollowPath:',8,'Pure Pursuit가 아니라 Rotation Shim이 DWB를 감싸는 현재 설정이다.')
    snippet('fsm','    def _request_navigation_cancel_for_manual_control(self, reason: str):',7,
            'Nav2와 정렬/조작이 동시에 서로 다른 속도를 내지 않게 제어권 전환을 요청한다.')
    para('일반 전진 주행과 필요한 예외 후진을 분리했다. 이 값과 예외 처리가 있다는 사실은 “경로가 한 번도 멈추지 않는다”는 보장이 아니다. 센서 대기·목표 변경·정렬·복구는 정지를 만들 수 있다.')

    page('17. 문 앞 정렬을 어떻게 읽는가', '주행 주차 pose와 팔이 잡을 수 있는 pose는 구분한다', ['alignment'])
    fig(navalign,'그림 17. World1 주행 개방 직전 차체·목표·문 위치 비교. 장애물/문 기준은 사후 SDF 참조이며 센서 입력을 그린 것은 아니다.',300)
    para('정렬은 목표 위치 오차와 방향 오차를 줄여 문 쪽을 향하게 하는 단계다. 하지만 Nav2 goal에 도착했다는 사실만으로 손잡이에 손이 닿거나 실제 문 법선과 완벽히 일치한다고 판정할 수 없다.')
    tab(['실측 범위','정확한 의미'],[
        ['World1 첫 문: 차체 외곽-닫힌 문판 약 0.734m','주행 정렬 결과. 30cm 이내 접촉 준비 완료라는 뜻 아님'],
        ['World1 첫 문: 문 법선 대비 약 8.91도','“정면 방향”을 지향하지만 수학적으로 완전 일자는 아님'],
        ['최신 단일문: 추가 IK 기반 접근','관측 손잡이를 실제 PIPER 작업공간으로 가져오는 별도 단계'],
    ],[1.7,2.2])
    para('현재 자료의 개선점을 숨기지 않고, 주행 정렬 검증과 팔 작업공간 접근 검증을 분리한 것이 설명의 핵심이다.','small')

    page('18. 하드코딩에서 무엇이 바뀌었나', '좌표를 얻는 방식·동작 종료 기준·남은 상수를 나누어 비교', ['before_after'])
    tab(['항목','기존/남아 있는 계산 경로','현재 단일문 관측·피드백 경로'],[
        ['손잡이 위치','bbox + 추정 높이/거리 또는 side-wall projection','주 YOLO 기원 픽셀 + 등록 depth + TF'],
        ['팔 접근','미리 정한 approach/offset·단계 pose','관측 손잡이와 실제 PIPER IK/Jacobian 조건으로 접근'],
        ['레버 누름','설정 거리만큼 누름 명령','작은 단계 이동 후 레버 반응·FK 추종 오차 확인'],
        ['문 개방','주행 backend에서 모델 매칭 후 hinge 명령','손잡이 유지, 팔·본체 명령, 문 각도 결과 확인'],
        ['완료 판단','서비스 성공/대기만으로 판단할 위험','신선한 레버/문 각도·접촉 감사·복귀 계측 결합'],
    ],[.75,1.7,1.9])
    para('이 비교는 현재 보존 코드의 기존 경로와 분리 복사본의 r40 실행 경로를 비교한 것이다. 모든 과거 버전이 동일했다는 뜻이나, r40이 월드 1~5에 이미 통합되었다는 뜻은 아니다.')
    heading('여전히 필요한 상수')
    para('로봇 치수·관절 제한·센서 보정값·제어 이득·충돌 여유·최대 누름량·timeout·개방각 기준은 남는다. “관측 기반”의 핵심은 매번 얻는 목표/반응을 사용한다는 것이지 숫자를 코드에서 없애는 것이 아니다.')

    page('19. 의심했던 벽 투영을 정확히 설명', 'max_abs_y / sin(angle)는 실측 wall_y가 아니다', ['before_after','code'])
    snippet('perception','        max_abs_y = self._side_handle_max_abs_y',13,
            '_side_wall_projected_distance의 실제 계산. 설정된 가로 거리 밴드와 방위각으로 거리를 추정한다.',
            after='    def _side_wall_projected_distance(')
    para('이 경로가 켜져 있으면 “LiDAR가 벽 위치를 실측했다”라고 설명하면 안 된다. max_abs_y는 파라미터이며, 함수 이름/주석만으로 관측 값이라고 해석할 수 없다.')
    snippet('launch',"                    'side_wall_projection_enabled': False,",3,
            '현재 보존 simulation.launch에서는 이 경로를 비활성화했다. 모든 과거 실행이나 다른 launch에도 자동 적용되는 것은 아니다.')
    para('다른 범위 제한·목표 standoff·기본 손잡이 높이·후보 병합 파라미터는 주행 baseline에 남아 있다. 최신 단일문 registered_depth 경로는 유효 depth가 없을 때 벽/높이 추정으로 성공시키지 않고 관측을 거부한다.')
    heading('발표 문장')
    para('“기존에는 일부 좌표를 설정된 벽 밴드로 보완했지만, 현재 단일문 실험에서는 YOLO와 등록 깊이의 3D 관측을 사용하며 데이터가 없으면 진행하지 않도록 분리했다.”')

    page('20. 손잡이 픽셀이 실제 목표 좌표로', '등록 RGB-D + 카메라 보정 + TF', ['handle_yolo','coordinates','code'])
    snippet('contact_perception','        patch = depth[y-2:y+3, x-2:x+3]',16,
            'bbox 중심의 5×5 깊이 patch를 검사하고 중앙값을 카메라 광학 좌표로 역투영한 뒤 기준 프레임으로 변환한다.')
    tab(['검사','필요한 이유'],[
        ['RGB-depth 시각 차이·영상 크기·보정값','다른 픽셀/다른 시점의 깊이를 붙이지 않기'],
        ['유효 깊이 수 / patch 깊이 불연속','손잡이와 뒤 문판이 섞인 경계를 무턱대고 채택하지 않기'],
        ['카메라 TF 변환','픽셀 위치와 로봇팔 기준 좌표를 혼동하지 않기'],
    ],[1.35,2.75])
    para('이 점은 눈에 보이는 손잡이 표면점이다. 손잡이의 정확한 회전축·접촉면·재질·힘까지 검출한 것은 아니다. 실제 RGB-D에서는 정합 오차와 금속 반사/깊이 누락을 별도 검증해야 한다.')

    page('21. 얼마나 다가갈지는 IK로 확인', '최신 r40 | 관측 좌표 + 교정된 PIPER 크기와 관절 제한', ['observation','alignment'])
    fig(ph['yolo_primary_detection.png'],'그림 21. r40에 실제 사용된 v3 YOLO 손잡이 검출. 노란 bbox는 모델 출력, 점은 3D 위치 계산에 쓰는 픽셀이다.',275)
    tab(['r40 계측','값'],[
        ['초기 관측 base_link XYZ','(1.120, 0.249, 0.767) m'],
        ['접근 후 채택 XYZ','(0.666, 0.146, 0.766) m'],
        ['실제 추가 접근 이동','약 0.468m'],
        ['IK condition ratio / 기준','0.2701 / 0.25'],
    ],[1.4,2.6])
    para('이번 실행은 “항상 0.468m 앞으로 가라”라는 명령이 아니다. 관측한 손잡이를 기준으로 실제 링크 길이·관절 한계·IK 조건을 확인해 접근했고, 표의 이동량은 실행 후 계측 결과다. 로봇 치수와 condition 기준은 고정 교정/안전 설정으로 남는다.')

    page('22. 얼마나 누를지는 반응을 본다', '설정 최대량과 실제 멈추는 조건을 구분', ['press_feedback','code'])
    snippet('physical','            stable = stable_lever_motion(',5,
            '최근 레버 반응이 최소 기준을 안정적으로 충족하면 press_verified로 전환하고 추가 깊이를 늘리지 않는다.')
    snippet('physical','            try:',9,
            '아직 반응이 부족하면 작은 단계의 다음 깊이를 계산하되 최대 이동 한계와 오류 처리를 적용한다.',
            after='    def _feedback_press(self, handle_pos):')
    tab(['r40 누름 사건','값과 해석'],[
        ['commanded_depth_m','0.020m: 이번 실행에서 쌓인 명령 깊이'],
        ['measured_tool_down_m','0.01161m: FK로 측정한 실제 공구 하강'],
        ['lever_delta_rad','0.09725rad, 약 5.57도에서 누름 확인'],
    ],[1.3,2.7])
    para('명령 깊이와 실제 움직임은 같지 않다. 위치 servo가 부하를 유지하려면 잔여 오차가 필요하므로 안정 확인 후 기존 목표를 유지한다. 레버 반응이 없거나 추종 오차가 과하면 더 밀어 성공 처리하지 않고 중단한다.')
    para('현재 반응은 Gazebo 레버 joint state다. 실물에서는 엔코더/접촉·힘/시각 피드백으로 대체해야 한다.','small')

    page('23. 손잡이 YOLO는 무엇이 달라졌나', '기존 v2는 보조, 혼합 학습 v3를 주 모델로 사용', ['handle_yolo'])
    fig(comparison,'그림 23. 동일 Gazebo 프레임에서 기존 v2와 v3의 bbox 비교. 2026-09-13 보존 재평가.',310)
    para('팀원은 실제 레버 데이터와 Gazebo 레버 데이터를 섞어 v2에서 추가 학습했다. v3는 시뮬레이션 외형과 도메인 차이를 줄이는 데 효과가 있었지만 실제 데이터의 기준을 통과하지 못해 CANDIDATE 명칭을 유지했다.')
    para('런타임은 주 모델 검출을 먼저 사용하고 설정에 따라 보조 모델/추적/HSV 경로를 가질 수 있다. 따라서 한 장의 bbox가 보인다고 모든 시점이 주 YOLO 직접 검출이었다고 주장하면 안 된다.')
    tab(['최신 r40 기록','의미'],[
        ['use_yolo_observation = true / primary 요구','주 모델 기원 관측이 필요'],
        ['yolo_observation_count = 38','실행 중 기록된 주 모델 기원 관측 수'],
        ['track:primary_yolo:lk:registered_depth','최종 채택은 YOLO에서 시작한 LK 추적 + 등록 깊이'],
    ],[1.5,2.5])

    page('24. 인식 효과를 숫자로 말하기', '고정 threshold의 사후 짝비교, 실시간 성공률과는 다르다', ['handle_yolo','metrics'])
    fig(comparechart,'그림 24. 동일 160장(positive 100, negative 60), confidence 0.25, IoU 0.5 평가.',250)
    tab(['지표','v2','v3'],[
        ['TP / FP / FN','0 / 0 / 100','100 / 54 / 0'],
        ['Recall','0%','100%'],
        ['Precision','예측 없음','64.9%'],
        ['오탐이 있는 negative 이미지','0 / 60','53 / 60'],
    ],[1.6,1,1])
    para('해석: Gazebo 손잡이를 놓치는 문제는 이 표본에서 크게 개선됐지만, 배경 오탐이 남는다. “99%니까 완벽” 대신 문 ROI·시간 추적·depth 정합·조작 전 신선도 검사와 함께 쓴다는 점을 강조한다.')
    para('팀원 전달 실제 데이터 Recall은 약 0.4091로 부족하다. 문 Door 모델 전달 지표는 mAP50 0.6088, Precision 0.6184, Recall 0.5817이다. 서로 다른 데이터셋/평가 조건의 숫자를 같은 성능으로 직접 비교하지 않는다.','small')

    page('25. 월드 1~5 주행 결과 요약', '2026-09-16 보존 자료 | 같은 입력 해시 확인, 물리 팔 FULL은 아님', ['nav_worlds'])
    names={1:'혼합 문·밀집 장애물',2:'대체 문·장애물 배열',3:'파란문 4개',4:'파란문 없음',5:'좌우 파란문 6개'}
    counts={1:3,2:3,3:4,4:0,5:6}
    tab(['월드','구성','개방 사건','경로 m','중첩 시료'],[
        [str(n),names[n],str(len(worlds[n]['metrics']['openings'])),f"{worlds[n]['metrics']['path_length_m']:.2f}",
         str(worlds[n]['metrics']['sampled_footprint_overlap_count'])] for n in range(1,6)
    ],[.5,1.7,.8,.8,.9])
    para('5개 월드 모두 보존 checker 집계의 runner/checker/attitude/events/exclusion/geometry 반환값 0, mission_complete. World1·2는 r9, World3·4·5는 r8 자료이며 manifest로 런타임·검사 입력의 같은 해시를 확인했다.')
    heading('무엇을 증명했는가')
    para('저장 지도와 sim_odom 위치추정 조건에서 관측 문 후보 선택·이동·주행 정렬·처리 사건·다음 문·출구 도달이 완료됐다. 그림의 장애물 위치/크기는 사후 SDF 참조이며 검은 선은 실제 기록된 이동이다.')
    heading('무엇을 증명하지 않았는가')
    para('새 r40 접촉 로봇팔로 16개 문을 모두 연 통합 임무, 미지 환경의 완전 SLAM, 실제 로봇의 충돌·접촉 안전성은 아니다. 샘플링 footprint 중첩 0도 연속 시간 전체의 무충돌 보장과 다르다.')
    para('정답 문 수 3·3·4·0·6은 검증용 수치다. 같은 숫자를 FSM의 사전 문 개수로 공급해 순서대로 방문했다는 의미가 아니다.','small')

    for n in range(1,6):
        m=worlds[n]['metrics']
        path=collect(NAV/f'figures/world{n}_path.png','05_주행_월드1_5')
        page(f'26.{n} World {n} | 주행 PASS', names[n]+' | 보존 실행 2026-09-16', [f'world{n}','nav_worlds'])
        fig(path,f'그림 W{n}. 검은 선=기록 궤적, 회색=벽, 번호 사각형=장애물 크기/위치, 색 선=닫힌 문, 화살표=개방 직전 자세.',255)
        tab(['검사','결과'],[
            ['고유 파란문 처리 사건 / 정답 수',f"{len(m['openings'])} / {counts[n]}"],
            ['누적 경로 / pose 시료',f"{m['path_length_m']:.2f}m / {m['pose_samples']}개"],
            ['샘플링 차체-장애물 중첩',str(m['sampled_footprint_overlap_count'])],
            ['종료','MISSION_COMPLETE / 검사 반환값 0'],
        ],[1.7,2.2])
        if n==4:
            para('파란문이 없는 경우: 파란 작업 0회를 정상으로 받아들이고 탐색·출구 정책으로 종료한다. 로봇에게 “0개”를 미리 알려주고 파란문 검색을 건너뛴다는 뜻이 아니다. 이 경로의 흔들림은 보존 자료에 그대로 남겼다.')
        else:
            rows=references[str(n)]
            para(f"사후 닫힌 문판 기준 차체 외곽 거리 {min(x['footprint_to_closed_panel_plane_m'] for x in rows):.3f}~{max(x['footprint_to_closed_panel_plane_m'] for x in rows):.3f}m. 문 법선 대비 자세 오차 {min(x['heading_to_closed_panel_normal_deg'] for x in rows):.2f}~{max(x['heading_to_closed_panel_normal_deg'] for x in rows):.2f}도. 이는 주행 자세 검사이며 파지 성공률이 아니다.")
            para('문 접근·후퇴·다음 목표 방향 전환이 겹쳐 궤적이 일부 휘거나 교차한다. 이 한 그림만으로 모든 꼬임을 정상 동작으로 단정하지 않는다. 총 임무 성공과 경로의 간결함은 별도 평가 대상이다.')
        link(f'07_원본판정/주행/월드{n}/mission_evidence_final_metrics.json',f'원본 지표: World{n}')

    page('27. 주행 성공과 경로 품질을 분리', '임무 완료 ≠ 최단 경로 ≠ 정렬 완벽 ≠ 실물 검증', ['navigation','limits'])
    tab(['관찰','현재 확인된 사실','다음 평가'],[
        ['문 근처 작은 루프','기록 궤적에 실제 남아 있음','목표 갱신·후진·선회 사건과 시각 맞춰 원인 분리'],
        ['주행 중 정지','정렬·센서대기·복구 등이 가능','정지시간, 계획시간, 재계획 횟수 정량화'],
        ['정면 주차','문 쪽 자세로 접근했으나 각도 오차 존재','실제 문 법선/손잡이 작업공간 기준 오차 추가'],
        ['문 모델 회전','주행 backend 회전 확인','신규 접촉 backend 연결 후 전체 재시험'],
    ],[.85,1.55,1.7])
    heading('기존 성공 자료를 남기는 이유')
    para('주행 보존본이 있으면 로봇팔 통합 후 문제를 주행 회귀와 조작 문제로 나누어 확인할 수 있다. 이번 보고서는 주행 성공을 폐기하지 않되, 새 물리 팔 성공으로 재해석하지 않는다.')
    heading('headless와 GUI')
    para('GUI가 없어도 물리·센서·노드가 실행되는 headless 시험은 유효하다. GUI·녹화는 관찰과 부하 조건을 추가한다. 화면이 있다는 것 자체가 더 엄격한 정책 증명은 아니며, 타이밍·센서 신선도·부하 변화는 별도 조건으로 기록해야 한다.')

    page('28. 최신 단일문 개방의 전체 순서', 'r40 | 실제 PIPER 링크 형상과 접촉을 사용한 별도 fixture', ['physical_sequence','video'])
    fig(ph['single_door_storyboard.png'],'그림 28. 순서: YOLO 기원 관측 → 파지·레버 누름 → 손잡이 유지·본체 이동 → 목표각 → 해제·회수·후진.',395)
    para('문을 몸통으로 밀거나 손잡이를 놓은 상태에서 팔을 늘려 여는 방식이 아니다. 손을 하중 전달 경로로 유지하고, 본체 이동/선회와 팔 자세 보정을 함께 수행한다. 팔 관절을 완전히 고정한 채 전진만 하는 방식도 아니다.')
    link(vids['short'],'최신 영상: 손잡이 유지 개방·복귀 요약 (녹화 기준 6배속)')
    para('약 34초 요약은 녹화 프레임의 6배속이다. 센서 렌더 수신률과 기록 주기가 달라 시뮬레이션 실시간 배속을 뜻하지 않는다. 프레임 보간·생성 이미지는 사용하지 않았다.','small')

    page('29. 파지한 뒤 레버를 눌렀다', 'r40 t=64.5초 렌더 + t=64.897초 측정 사건', ['physical_sequence','press_feedback'])
    fig(ph['closeup_00064.50_press_handle.png'],'그림 29. 실제 Gazebo closeup. 손가락이 레버를 잡은 상태에서 누름 단계가 진행된다.',325)
    tab(['근거','기록'],[
        ['press_verified','sim_time 64.897s'],
        ['레버 변화','약 5.57도'],
        ['명령 하강 / 측정 하강','20.0mm / 11.61mm'],
        ['피드백 출처','Gazebo joint states + 팔 관절 FK'],
    ],[1.1,2.9])
    para('그림은 접촉의 시각적 맥락이고, 누름 판정은 로그로 확인한다. 실제 레버가 잠금장치를 해제하는 데 필요한 각도/토크는 문마다 다르므로 이 fixture의 기준을 그대로 현장 표준으로 쓸 수 없다.')

    page('30. 손잡이를 유지하며 본체 이동', 'r40 중간 개방 장면 | 완전 고정 팔보다 접촉 자세 유지가 중요', ['physical_sequence'])
    fig(ph['closeup_00131.50_handle_held_base_open.png'],'그림 30. t=131.5초, 약 60도 개방 중인 실제 렌더. 손가락-레버 접촉을 유지한다.',325)
    para('문은 경첩을 중심으로 원호 운동을 한다. 본체가 직선으로만 전진하고 팔을 완전히 고정하면 손잡이의 원호와 손의 경로가 달라져 미끄러짐·과부하가 생길 수 있다. 현재 구현은 관측된 접촉 자세·문/본체 방향 차이를 보고 선속도·각속도와 팔 목표를 보정한다.')
    para('문이 정지하거나 레버가 풀리거나 손가락/관절 상태가 유효하지 않으면 중단한다. 문을 붙잡는 가상의 weld joint를 붙이거나 문 hinge 위치 명령을 직접 보내 개방한 시험이 아니다.')
    para('result.json: attach_grasp_joint=false, door_hinge_command_topic_used=false, arm_joint_commands_used=true, mobile_base_commands_used=true.','small')

    page('31. 손잡이를 놓기 전에 목표각 도달', 'r40 개방 확인 사건 | 관측 피드백 기반 전이', ['physical_sequence','code'])
    fig(ph['closeup_00192.50_handle_held_base_open.png'],'그림 31. 개방 확인 직전 실제 closeup. 추적 카메라라 문판이 늘 정면처럼 보일 수 있다. 전체 회전은 다음 고정 카메라로 확인한다.',285)
    snippet('physical','                if delta >= min_delta:',4,
            '측정 문 각도가 기준을 넘은 뒤에 handle_held_open_verified를 남기고 다음 해제 단계로 간다.')
    tab(['사건','실측'],[
        ['손잡이 유지 개방 확인',f"t={held['sim_time_sec']:.3f}s / {math.degrees(held['door_delta_rad']):.2f}도"],
        ['요구 각도 / 최종 최대',f"117.46도 / {peak:.2f}도"],
        ['동시 레버 변화',f"{math.degrees(held['lever_delta_rad']):.2f}도"],
    ],[1.4,2.7])
    para('완전 개방의 뜻은 이 시험에서 설정한 최소 2.05rad 도달이다. 모든 실제 문의 기계적 끝점에 닿았다는 의미는 아니다.','small')

    page('32. 고정 시점에서 본 개방 전후', '추적 closeup의 시각적 착시를 고정 overhead로 보완', ['physical_sequence','hinge'])
    fig(ph['before_overhead.png'],'그림 32a. 개방 전, 고정 overhead 카메라.',235)
    fig(ph['after_overhead.png'],'그림 32b. 같은 고정 overhead 시점의 개방 후. 문판의 회전과 로봇의 위치 변화가 보인다.',235)
    para('최신 모델은 측면 힌지 fixture다. 손잡이는 힌지에서 떨어진 문판 쪽에 있고, 본체/팔은 손잡이를 통해 미는 방향으로 개방한다. 위·아래 영상에서 카메라가 바뀐 것처럼 보이는 closeup과 달리, 이 두 장은 고정 시점의 전후 비교다.')
    para('방향은 카메라 시점만으로 “오른쪽/왼쪽”이라고 외우기보다 문 힌지 기준으로 설명한다. 로그에서 열린 방향 각도는 음수이며, 판정은 초기값 대비 절대 변화량을 사용한다.','small')

    page('33. 개방각과 레버 반응의 계측', 'r40 feedback_events를 시뮬레이션 시간으로 다시 그림', ['physical_metrics'])
    fig(graph,'그림 33. 손잡이 유지 단계의 문 각도와 레버 각도. 실제 기록 시점만 표시한 계측 그래프다.',345)
    para('상단은 문 회전이 진행되어 117.46도 기준을 넘는 과정을 보여준다. 하단은 같은 구간의 레버 반응이다. 영상·서비스 응답만이 아니라 물리 상태를 함께 확인한다.')
    para('이 그래프는 Gazebo joint 계측이며 실물 힘/토크 센서 데이터가 아니다. 시간축은 simulation seconds다. 기존 door_hinge_angle.png의 wall-clock 시간축과 혼동하지 않는다.')

    page('34. 개방 후 해제·회수·후진까지', '성공 문을 연 뒤 다음 주행을 위한 자세로 돌아오는 단계', ['recovery','video'])
    fig(ph['closeup_00204.50_complete.png'],'그림 34. r40 완료 시점 렌더. 팔 회수·후진 완료는 이미지와 실제 관절/odom 결과를 함께 읽는다.',295)
    tab(['복귀 검사','결과'],[
        ['실제 후진 이동',f"{recovery['signed_backoff_m']:.4f}m / {recovery['backoff_duration_sim_sec']:.3f} sim s"],
        ['최종 stow 최대 관절 오차',f"{recovery['final_stow_max_error_rad']:.4f}rad (기준 0.070rad)"],
        ['복귀 접촉 감사','금지된 문판/문틀 접촉 검사 PASS'],
    ],[1.5,2.6])
    para('현재 후진 제어는 설정 속도와 시뮬레이션 시간으로 실행하고, 그 결과를 odom으로 확인한다. 따라서 이 단계까지 이동 거리 피드백 제어로 완전히 바뀌었다고 설명하지 않는다.')
    link(vids['full'],'영상: r40 전체 기록 (원 녹화 프레임 속도, H.264)')

    page('35. 접촉 성공을 어디까지 믿을 수 있나', '시각 자료 + 상태 계측 + 금지 접촉 검사 + 입력 보존', ['physical_metrics','limits'])
    tab(['근거','r40 결과'],[
        ['목표 개방·레버 피드백','PASS'],
        ['손가락-레버 접촉 기록','4,654개 기록'],
        ['본체/카메라/팔-문판/문틀 금지 접촉','검사 범위에서 미검출'],
        ['접촉 기록 생존 확인','5채널 각 4,080메시지, 저장 최대 간격 0.15sim s'],
        ['팔 회수·후진','실제 관절/odom 후검사 PASS'],
        ['입력·단위검증','보존 기록: 212 tests, colcon 8 packages PASS'],
    ],[1.75,2.25])
    para('r39는 실제 개방/복귀 후 접촉 토픽 가용성 검사에서 실패했다. Gazebo contact는 접촉이 없을 때 메시지가 안 나올 수 있다. r40은 제어를 바꾸지 않고 20Hz 빈 heartbeat와 매 physics-step 접촉쌍 집계가 있는 읽기 전용 증거 plugin을 추가했다.')
    para('r39/r40 비교에서 제어·로봇·인지·주행 62개 파일 해시가 같고 world XML 차이는 증거 plugin이었다. 무접촉을 곧바로 센서 장애나 성공으로 혼동하지 않도록 검사의 관측 가능성을 보완한 것이다.')
    para('접촉 수는 실제 힘/토크의 크기나 가해진 하중의 안전 인증을 뜻하지 않는다. 자기충돌 전체와 실물 내구성은 별도 과제다.','small')

    page('36. 현재 물리 시험의 조건과 한계', '한 개 fixture 성공을 일반적인 모든 문으로 확대하지 않는다', ['limits'])
    tab(['조건','현재 값/범위','실제 적용 전 필요'],[
        ['문판','18kg / 0.8×1.98×0.06m','실제 질량·힌지·래치 부하 실측'],
        ['레버','직경 52mm / 길이 260mm','실물 레버 단면과 그리퍼 간섭 검토'],
        ['접촉 마찰','μ=5.0, 높은 마찰 가정','재질별 미끄러짐·그립력 실험'],
        ['그리퍼 force cap','8N 설정','실물 집게 힘/안전 제한 측정'],
        ['힌지','마찰 2 / damping 0.8','실제 힌지 응답과 비교'],
        ['접근 정지 설정','fixture front stop 0.0m','실제 로봇에 그대로 쓰지 않음'],
    ],[.8,1.5,1.8])
    para('현재 완전 개방은 “노란 손잡이니까 HSV로 바로 좌표를 줬다”는 시험이 아니라 주 YOLO 기원 관측을 요구했다. 다만 손잡이 색·외형이 시험 데이터 분포와 유사하므로 임의 재질/조명 일반화는 추가 확인이 필요하다.')
    para('최신 접촉 제어는 문/레버 joint state를 읽는다. 이 값을 실제 로봇에서 무엇으로 관측할지 정하지 않으면 같은 코드를 하드웨어에 연결하는 것만으로 동일 기능이 생기지 않는다.')
    para('이번 보고서에는 실패를 숨기거나 물리 fixture를 임의로 팔 길이를 늘린 모델처럼 설명하지 않았다. 실제 링크 기구학과 관절 제한을 사용하되, 모델 정확도·마찰 가정의 한계를 명시한다.','small')

    page('37. 개방 다음 상태도 핵심이다', '결과 확인 → 열린 문 기록 → 제어권 반환 → 다음 후보', ['fsm','exit'])
    pseudo('''if opening_result_confirmed:
    mark_opened(id_lineage, observed_station)
    retract_arm_and_finish_recovery()
    release_manual_control()
    candidates = memory.unopened_confirmed_blue()
    if candidates:
        lock_next_eligible_target(candidates)
    else:
        recheck_search_coverage_and_exit_candidate()''','상위 설계 요약. baseline 주행 backend와 신규 단일문 접촉 backend가 이미 한 임무로 통합 완료됐다는 의미는 아니다.')
    para('출구 판단은 “지금 화면에 파란색이 없다”와 다르다. 누적 미처리 후보, 재탐색/재확인 조건, 초록 출구 관측을 함께 확인한다. 월드 4는 문 0개 조건을 보여주지만 임의 형상의 모든 방에서 탐색 완전성을 증명하지는 않는다.')
    heading('실패 문은 왜 따로 기억하는가?')
    para('접근/개방 실패를 제한 없이 재시도하면 같은 문에서 영원히 멈출 수 있다. 실패·포기와 opened를 별도 관리하여 재시도 상한 이후 진행 여부를 정책으로 결정한다. “문을 못 열었는데 열었다고 카운트”하는 것과는 구분해야 한다.')
    heading('통합에서 우선 확인할 계약')
    para('문 station 동일성, 신선한 손잡이, 조작 중 Nav2 비활성, 결과·복귀 완료 전 목표 해제 금지, opened 기록 시점, 다음 탐색의 재진입 조건을 시험한다.')

    page('38. 강조할 만한 구현 기여', '기존 도구를 사용한 부분과 팀이 설계한 연결 논리를 분리', ['ideas'])
    tab(['핵심 아이디어','왜 중요한가','증거'],[
        ['기하 지도와 의미 기억 분리','장애물 격자와 문 작업 상태를 다른 목적에 사용','지도 확장 화면 + landmark 코드'],
        ['관측 계속 / 실행 목표 하나','정보를 놓치지 않으면서 목표 변경 진동 완화','잠금·새문·재관측 제외 자료'],
        ['ID와 위치를 함께 기억','같은 문에 새 ID가 붙어도 중복 작업 방지','opened station / IGNORED 사례'],
        ['단계별 제어권과 진입 조건','Nav2·미세정렬·팔이 동시에 충돌하지 않도록','cancel 함수 + 상태 계약'],
        ['명령보다 실제 반응으로 판정','동작 호출만으로 성공하는 거짓 PASS를 줄임','누름 반응·힌지각·복귀·접촉 감사'],
        ['날짜·코드·검사 범위 보존','과거 성공과 현재 실험을 재현 가능하게 구분','manifest·원본 로그·별도 workspace'],
    ],[1.2,1.75,1.25])
    para('YOLO, HSV, SLAM, Nav2, PIPER 기구학 도구는 선행 기술/도구다. 차별점은 이들을 화재 대응 임무의 의미·기억·안전 전이 조건으로 조합하고, 불확실한 관측과 실행 결과를 다루는 방식이다. 세계 최초·일반 최적성 같은 주장은 하지 않는다.')

    page('39. 다음 검증의 우선순위', '발표에 넣을 현재 성과와 이후 과제를 함께 정리', ['next','limits'])
    tab(['우선','해야 할 일','통과 기준 예'],[
        ['1','신규 접촉 backend를 주행 정렬 후 연결','한 문에서 이동→재관측→파지→개방→복귀→다음 목표'],
        ['2','마찰·레버 크기·접근 오차를 바꾼 반복 시험','실패율·정지 이유·금지 접촉 기록을 함께 제출'],
        ['3','손잡이 negative/실물 인식 개선','시뮬+실물 분리 precision/recall, 고정 holdout'],
        ['4','mapping 모드와 저장 지도 주행 분리 검증','초기 지도·확장·localization 오차의 독립 계측'],
        ['5','실제 PIPER 제한 범위 시험','CAN/관절/그리퍼/TF/비상정지 확인 후 저속 동작'],
    ],[.45,1.6,2.0])
    heading('발표용 결론')
    para('“관측 기반 문 의미 기억과 다중 월드 주행 회귀를 확보했고, 별도 PIPER 접촉 시험에서 YOLO 기원 손잡이 좌표로 레버를 누른 채 본체를 이동해 약 118도 개방한 후 복귀했다. 다음 단계는 두 성공 범위를 통합하고 실물 센싱·접촉 조건을 검증하는 것이다.”')
    para('이 표현은 실제 증거와 일치하면서 핵심 아이디어를 드러낸다. 현 상태를 완성도 한 숫자로 단정하거나 실제 시연 성공을 보장하지 않는다.')

    page('부록. 자주 쓰는 용어', '발표 중 설명하기 쉽도록, 이 보고서에서의 의미로 정리', ['glossary'])
    tab(['용어','쉬운 설명'],[
        ['bbox / ROI','검출된 사각형 / 그중 자세히 검사할 관심 영역'],
        ['TF / frame','센서·로봇·지도 사이 좌표 변환 / 좌표 기준'],
        ['OccupancyGrid / costmap','공간이 막혔는지 나타내는 격자 / 주행 위험·여유를 비용으로 표현한 지도'],
        ['landmark / station','여러 관측을 묶어 기억하는 대상 / 같은 실제 문으로 간주하는 지도상 위치'],
        ['목표 잠금','현재 처리할 문 하나를 유지하는 정책. 새 관측 자체를 막는 뜻은 아님'],
        ['IK / FK','원하는 손 위치에서 관절 각도를 계산 / 현재 관절 각도로 실제 손 위치 계산'],
        ['registered depth','RGB 픽셀과 대응하도록 정합된 깊이 영상'],
        ['backend / fixture','실제 실행을 담당하는 구현 / 제한된 조건의 검증용 장치·월드'],
        ['feedback / open-loop','측정 반응으로 다음 명령·종료 결정 / 정한 명령·시간으로 실행'],
        ['Recall / Precision','실제 손잡이를 얼마나 찾았는가 / 찾았다고 한 것 중 진짜가 얼마나 되는가'],
    ],[1.1,3.0])
    para('예: “YOLO bbox의 중심을 registered depth로 3D화하고 TF로 base_link에 옮긴 뒤 IK로 접근한다”는 말은 “영상에서 손잡이를 찾고, 깊이를 이용해 로봇 기준 위치로 바꾼 뒤 실제 팔이 닿는 자세를 계산한다”는 뜻이다.')

    page('40. 자료·영상 찾아보기', '보고서와 같은 폴더 안 상대 경로로 연결됨', ['evidence_index'])
    tab(['폴더','내용'],[
        ['01_인식과지도','초기 센서·부분 지도·관측 확장·의미 색 지도'],
        ['02_목표와문기억','목표 잠금·opened ID·재관측 제외'],
        ['03_회피와정렬 / 04_YOLO모델','costmap·정렬·동일 이미지 모델 비교'],
        ['05_주행_월드1_5 / 06_문개방','5개 월드 궤적과 최신 r40 단계별 원본'],
        ['07_원본판정 / 08_코드발췌와원본','결과 JSON·로그·코드 복사본·라인 근거'],
        ['09_영상','보존 인식/기억 영상과 최신 개방 영상'],
    ],[1.6,2.45])
    for key,label in [('perception','인식·의미 좌표 과정'),('map','과거 Gazebo/RViz 지도 생성'),
                       ('memory','목표 잠금과 재관측 제외'),('short','최신 r40 개방 요약'),('full','최신 r40 전체 기록')]:
        link(vids[key],label)
    para('evidence_manifest.json은 각 파일의 원본 경로·SHA-256·크기를 담고, code_excerpt_index.json은 코드 발췌의 페이지·파일·라인·해시를 담는다. 00_읽는순서.md에는 증거 범위와 공유 방법이 있다.')
    para('REV6은 읽기 쉬운 해설을 새로 작성한 개정본이다. 이전 보고서·성공 코드·실패 기록을 수정하거나 삭제하지 않았다.','small')


def flowables(block):
    k=block['kind']
    if k in ('body','h','small'):
        return [base.p(block['text'],k)]
    if k=='table':
        return [base.table(block['headers'],block['rows'],block['weights']), Spacer(1,9)]
    if k=='image':
        return [base.picture(OUT/block['path'],height=block['height']), Spacer(1,5), base.p(block['caption'],'small')]
    if k=='link':
        s=f'<link href="{html.escape(block["path"],quote=True)}" color="#137f7a"><u>{html.escape(block["text"])}</u></link>'
        return [Paragraph(s,base.styles['body'])]
    if k in ('code','pseudo'):
        title='실제 코드 발췌' if k=='code' else '설명용 의사코드'
        if k=='code':
            role='접촉 복사본' if '접촉복사본' in sources[block['source']]['copy'] else '주행 보존본'
            title+=f" | {role} | {Path(sources[block['source']]['path']).name} : {block['first']}-{block['last']}"
        # Never modify source lines to make them look simpler; wrap only for display.
        lines=[]
        for line in block['text'].splitlines():
            lines.extend(textwrap.wrap(line, width=91, replace_whitespace=False, drop_whitespace=False,
                                        subsequent_indent='    ') or [''])
        return [base.p(title,'small'),Preformatted('\n'.join(lines),base.styles['code']),
                base.p(block['explanation'],'body')]
    raise ValueError(k)


def render_pdf():
    writer=PdfWriter()
    for pg in pages:
        memory=io.BytesIO()
        doc=SimpleDocTemplate(memory,pagesize=(base.W,base.H),leftMargin=40,rightMargin=40,
                             topMargin=45,bottomMargin=44)
        story=[base.p(pg['title'],'title'),base.p(pg['subtitle'],'small'),Spacer(1,7)]
        for b in pg['blocks']:
            story.extend(flowables(b))
        def furniture(canvas, doc):
            canvas.setFillColor(colors.HexColor('#147867'))
            canvas.rect(40,base.H-25,42,3,fill=1,stroke=0)
            canvas.setFont('KR',8)
            canvas.setFillColor(colors.HexColor('#627078'))
            canvas.drawString(40,23,'FIRE ROBOT | CORE IDEAS & EVIDENCE | REV6 | 2026-09-17')
            canvas.drawRightString(base.W-40,23,f"{pg['number']:02d} / {len(pages):02d}")
        doc.build(story,onFirstPage=furniture,onLaterPages=furniture)
        reader=PdfReader(memory)
        if len(reader.pages)!=1:
            raise ValueError(f"Page {pg['number']} overflow: {pg['title']} -> {len(reader.pages)} pages")
        writer.add_page(reader.pages[0])
        writer.add_outline_item(pg['title'],pg['number']-1)
    writer.add_metadata({'/Title':'관측하고 기억하고 실행 결과로 확인하는 화재 대응 로봇',
                         '/Author':'Fire Robot Project', '/Subject':'검증 범위가 구분된 핵심 설계와 실제 증거 REV6'})
    with (OUT/PDF_NAME).open('wb') as stream:
        writer.write(stream)


def make_companions():
    body=[]
    md=['# 화재 대응 로봇 핵심 아이디어와 검증 REV6','', '2026-09-17 | 과거 성공 자료 보존 | 코드/정책 수정 없음','']
    nav=[]
    for pg in pages:
        i=pg['number'];title=pg['title']
        nav.append(f'<a href="#p{i}">{i:02d} {html.escape(title)}</a>')
        body.append(f'<section id="p{i}"><header><span>{i:02d}</span><h2>{html.escape(title)}</h2></header><p class="meta">{html.escape(pg["subtitle"])}</p>')
        md+=['',f'## {i:02d}. {title}',pg['subtitle'],'']
        for b in pg['blocks']:
            k=b['kind']
            if k in ('body','h','small'):
                tag='h3' if k=='h' else 'p'
                body.append(f'<{tag} class="{k}">{html.escape(b["text"])}</{tag}>')
                md.append(('### ' if k=='h' else '')+b['text']+'\n')
            elif k=='table':
                body.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in b['headers'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in b['rows'])+'</tbody></table></div>')
                md+=['| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |']
                md+=['| '+' | '.join(map(str,row))+' |' for row in b['rows']]
            elif k=='image':
                body.append(f'<figure><a href="{html.escape(b["path"])}"><img loading="lazy" src="{html.escape(b["path"])}" alt="{html.escape(b["caption"])}"></a><figcaption>{html.escape(b["caption"])}</figcaption></figure>')
                md.append(f'![{b["caption"]}]({b["path"]})\n')
            elif k=='link':
                if b['path'].endswith('.mp4'):
                    body.append(f'<h3>{html.escape(b["text"])}</h3><video controls preload="none" src="{html.escape(b["path"])}"></video>')
                else:
                    body.append(f'<p><a href="{html.escape(b["path"])}">{html.escape(b["text"])}</a></p>')
                md.append(f'[{b["text"]}]({b["path"]})\n')
            elif k in ('code','pseudo'):
                src='설명용 의사코드'
                if k=='code':
                    src=f"실제 코드: {sources[b['source']]['copy']}:{b['first']}-{b['last']}"
                body.append('<p class="meta">'+html.escape(src)+'</p><pre><code>'+html.escape(b['text'])+'</code></pre><p>'+html.escape(b['explanation'])+'</p>')
                md+=[src,'```python',b['text'],'```',b['explanation'],'']
        body.append('</section>')
    css='''*{box-sizing:border-box}body{margin:0;background:#f7f9fa;color:#20272c;font:16px/1.8 "Malgun Gothic",sans-serif;letter-spacing:0}nav{position:fixed;inset:0 auto 0 0;width:285px;overflow:auto;padding:24px 18px;background:#fff;border-right:1px solid #ccd7db}nav a{display:block;font-size:12px;line-height:1.65;margin:9px 0}a{color:#0d746f}main{margin-left:285px;max-width:1210px;padding:32px 40px}section{padding:20px 0 36px;border-bottom:2px solid #ccd7db;scroll-margin-top:12px}header{display:flex;align-items:baseline;gap:15px}header span{font-size:18px;color:#137f7a;font-weight:bold}h1{font-size:29px}h2{font-size:25px;line-height:1.45;margin:0}h3{font-size:19px}p{margin:16px 0}.meta,.small,figcaption{font-size:13px;color:#56666d}figure{margin:22px 0}img,video{display:block;max-width:100%;height:auto;border:1px solid #d4dce0}video{width:100%;max-height:650px;background:#101617}figcaption{margin-top:9px}table{width:100%;border-collapse:collapse;font-size:14px}th{background:#147867;color:#fff;text-align:left}td,th{padding:12px;border:1px solid #d0d9de;vertical-align:top}tr:nth-child(even){background:#eef3f5}.table{overflow:auto}pre{padding:18px;background:#eaf0f3;overflow:auto;font:13px/1.6 Consolas,"Malgun Gothic",monospace;white-space:pre}#top{border-bottom:3px solid #147867;margin-bottom:20px;padding-bottom:20px}@media(max-width:800px){nav{position:static;width:auto;max-height:240px;border-bottom:1px solid #ccd7db}main{margin:0;padding:24px 18px}h2{font-size:22px}header{gap:9px}}'''
    document='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>화재 대응 로봇 REV6 증거 보고서</title><style>'+css+'</style><nav><strong>핵심 아이디어 · 증거</strong>'+''.join(nav)+'</nav><main><div id="top"><h1>화재 대응 로봇 설계·검증</h1><p>REV6 · 2026-09-17 · 주행 보존 자료와 최신 단일문 접촉 시험을 분리한 보고서</p><a href="'+PDF_NAME+'">PDF 보고서</a></div>'+''.join(body)+'</main></html>'
    (OUT/'START_보고서와영상.html').write_text(document,encoding='utf-8')
    (OUT/'보고서_편집원문.md').write_text('\n'.join(md),encoding='utf-8')
    (OUT/'report_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'code_excerpt_index.json').write_text(json.dumps({'sources':sources,'excerpts':excerpts},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'evidence_manifest.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2),encoding='utf-8')
    coverage={tag:[p['number'] for p in pages if tag in p['tags']] for tag in sorted({t for p in pages for t in p['tags']})}
    (OUT/'coverage_index.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'00_읽는순서.md').write_text('''# REV6 핵심 아이디어·증거 자료

1. PDF로 보고서를 읽거나 START_보고서와영상.html을 열어 영상을 함께 봅니다.
2. PDF의 영상 링크가 뷰어에서 차단되면 HTML 또는 09_영상 폴더를 사용합니다.
3. 공유할 때는 ZIP 전체를 전달하고 압축 해제 후 엽니다. PDF만 전달하면 상대경로 영상은 빠집니다.

## 세 가지 증거 범위
- 2026-09-09~10: 관측 지도 확장·문 기억·재관측 제외의 보존 예시. 최신 r40 접촉 성공 자료가 아닙니다.
- 2026-09-16: 월드 1~5 저장 지도+sim_odom 주행 회귀 PASS. 실제 PIPER 접촉으로 16개 문 개방한 FULL은 아닙니다.
- 2026-09-17 r40: 주 YOLO 기원 손잡이 관측, 등록 깊이, 접촉 개방·복귀의 단일 fixture PASS. 실제 하드웨어 미검증.

## 자료 무결성
원본 사진/영상은 byte-identical 복사입니다. 브라우저 재생용 _browser_h264 사본은 프레임 순서/속도를 유지한 코덱 변환이며 원본도 함께 보존합니다.
추가 feedback_sim_time.png와 색 관측 그래프는 JSON/CSV 실측 자료를 재시각화한 것입니다. 설계 흐름도는 설명 도식이라고 표시했습니다.
증거 영상·사진을 AI로 생성하거나 실패를 성공 이미지로 바꾸지 않았습니다.
코드 발췌의 파일·라인·해시는 code_excerpt_index.json에 있습니다.
기존 보고서·이전 증거·성공 주행 소프트웨어는 수정/삭제하지 않았습니다.
이번 개정은 보고서 재작성과 보존 데이터 재분석입니다. 새 전체 시뮬레이션 결과라고 표시하지 않습니다.

## 다음 편집
보고서_편집원문.md에 장별 설명과 코드가 있습니다. tools/build_report_rev6.py가 생성 스크립트입니다.
''',encoding='utf-8')
    return coverage


def verify_and_package(coverage):
    required=['initial_map','map_growth','color_map','opened_memory','reobserve','avoidance',
              'before_after','world1','world2','world3','world4','world5','door_yolo','handle_yolo','video']
    assert all(coverage.get(k) for k in required)
    reader=PdfReader(OUT/PDF_NAME)
    assert len(reader.pages)==len(pages)
    for pg in pages:
        assert len(reader.pages[pg['number']-1].extract_text())>150
        for b in pg['blocks']:
            if b['kind'] in ('image','link'):
                assert (OUT/b['path']).exists(),b['path']
    for path,data in assets.items():
        assert sha(OUT/path)==data['sha256']
    integrity={'pages':len(pages),'asset_count':len(assets),'source_modules':len(sources),
               'code_excerpts':len(excerpts),'required_coverage_pass':True,'relative_assets_exist':True,
               'sha256_assets_pass':True,'pdf_sha256':sha(OUT/PDF_NAME),
               'visual_qa':'pending rendered-page inspection'}
    (OUT/'report_integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(integrity,ensure_ascii=False,indent=2))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    build_content()
    render_pdf()
    coverage=make_companions()
    verify_and_package(coverage)
    print(OUT/PDF_NAME)
