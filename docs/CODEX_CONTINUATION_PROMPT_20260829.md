# Codex 이어받기 프롬프트 (2026-08-29)

아래 내용을 팀원 Codex 새 작업에 그대로 붙여 넣으세요.

---

너는 ROS2 Humble + Gazebo + Nav2 + PIPER 로봇팔 졸업작품 프로젝트를 이어받는 Codex야. 목표는 현재 작업본을 이해하고, 손잡이 YOLO/문개방 FSM/관측 기반 주행 검증을 이어서 완료하는 것이다.

## 프로젝트 경로

Windows 원본 후보:

```text
C:\Users\황준영\Documents\졸업작품
C:\Users\황준영\Desktop\졸업작품
```

WSL 검증 워크스페이스:

```text
/home/junyoung/fire_robot_ws_test
```

현재 WSL distro:

```text
Ubuntu2204Recovered
```

기본 명령 형식:

```powershell
wsl -d Ubuntu2204Recovered -u junyoung --cd /home/junyoung/fire_robot_ws_test -- bash -lc 'COMMAND'
```

## 먼저 지켜야 할 것

- 사용자 변경을 임의로 되돌리지 말 것.
- `git reset --hard`, `git checkout -- .` 같은 전체 되돌리기 금지.
- 자율주행 성공 코드가 따로 보존되어 있으므로 삭제하지 말 것.
- 좌표 하드코딩으로 문 위치를 알려주는 방식 금지.
- 문/장애물/출구는 카메라 + 2D LiDAR 관측 결과를 map 좌표에 누적해 사용해야 함.
- 문개방 성공 로그만 보고 손잡이 YOLO 성공이라고 판단하지 말 것. 반드시 `handle_detected=True`와 `handle_method=yolo:*`를 확인해야 함.

## 보존된 자율주행 성공 코드

이 버전은 로봇팔/손잡이 strict 기준을 붙이기 전, 관측 기반 자율주행 full validation 성공 기준이다.

```text
branch: codex/navigation-success-before-arm
tag: navigation-success-before-arm-20260826
commit: 41ec296 feat: finalize observation-based navigation validation
```

현재 작업본이 너무 꼬이면 이 branch/tag를 참고해서 주행 정책만 비교하되, 현재 변경을 무작정 되돌리지는 말 것.

## 현재 작업본 목표

1. 시작 후 front/front_left/front_right 카메라와 2D LiDAR로 전방/측면을 관측한다.
2. 관측된 벽, 장애물, 문 색상, 손잡이 후보를 map 좌표 semantic map에 누적한다.
3. 가장 가까운 unopened blue door를 선택하고 target lock을 건다.
4. 해당 파란문을 열기 전까지 다른 파란 후보로 목표가 흔들리지 않게 한다.
5. Nav2가 장애물을 피해 문까지 이동한다.
6. 문 근처에서는 fine alignment로 문 정면 바로 앞에 일자로 정렬한다.
   - 목표 기준: 문까지 약 20~30cm대
   - lateral 약 10cm대
   - yaw error 10도 이내
7. manipulation FSM이 손잡이를 누르고 본체/팔 동작으로 push-open을 수행한다.
8. 열린 문은 opened station으로 기록해 다시 선택하지 않는다.
9. 더 이상 열 파란문이 없을 때만 초록 비상구로 이동해 통과한다.

## 현재 수정된 주요 파일

읽어야 할 파일:

```text
README.md
docs/TEAM_HANDOFF_20260829.md
docs/HANDLE_MODEL_V2_VALIDATION_20260829.md
docs/HANDLE_RETRAIN_REQUEST_20260829.md
src/fire_robot_bringup/launch/simulation.launch.py
src/fire_robot_fsm/fire_robot_fsm/state_machine_node.py
src/fire_robot_fsm/fire_robot_fsm/fsm_subsystems.py
src/fire_robot_perception/fire_robot_perception/door_detection_node.py
src/fire_robot_manipulation/fire_robot_manipulation/manipulation_node.py
tools/convert_blue_doors_to_hinged.py
scripts/check_validation_log.py
```

중요 변경:

- `simulation.launch.py`
  - `handle_model_path` 기본값이 `fire_robot_perception/models/handle_best_v2.pt`.
  - `publish_debug_image`, `log_handle_detections` launch argument 추가.
  - locked target refine 비활성화.

- `door_detection_node.py`
  - Door YOLO는 문 bbox만 검출.
  - 색상 red/blue/green은 HSV.
  - handle YOLO는 door ROI 내부/주변에서 먼저 실행.
  - 실패하면 HSV handle, 이후 wall projection/memory fallback 사용.

- `state_machine_node.py`, `fsm_subsystems.py`
  - 관측 semantic map 기반 파란문 후보 선택.
  - nearest unopened blue target lock.
  - opened/failed/abandoned station memory.
  - 미개방 파란문이 남아 있으면 EXITING 우선 금지.
  - fine door approach 이후 manipulation 검증.

- `manipulation_node.py`
  - `/open_door` 요청을 받아 handle pose를 기준으로 Gazebo 힌지문 registry와 매칭.
  - 레버 누름/문 밀기 단계 FSM.
  - Gazebo에서는 matched blue door joint topic에 열림 각도를 보냄.

- `tools/convert_blue_doors_to_hinged.py`
  - 파란 visual-only door를 힌지문으로 변환.
  - 2026-08-29 기준 손잡이를 얇은 원통에서 `handle_backplate` + 수평 레버 박스로 생성하도록 변경.

## 새 손잡이 모델 v2 상태

팀원이 준 파일:

```text
C:\Users\황준영\Documents\카카오톡 받은 파일\handle_best_v2.pt
```

적용 위치:

```text
src/fire_robot_perception/models/handle_best_v2.pt
install/fire_robot_perception/share/fire_robot_perception/models/handle_best_v2.pt
```

GitHub에는 중복 방지를 위해 ROS 기본 경로인 `src/fire_robot_perception/models/handle_best_v2.pt`만 추적하면 된다. 예전 `handle_best.pt` alias는 기본 실행에서 사용하지 않는다.

모델 확인 결과:

```text
sha256=0c2681fd954da54b71aba2234cfecf1e19832a312c18ba8d156a9a02de463449
names={0: 'lever_handle'}
```

정적 검증:

```text
artifacts/validation/handle_v2_static_20260829/summary.json
결과: 기존 Gazebo camera sample 8장 detection 0건
```

레버형 Gazebo 손잡이 수정 후 live run:

```text
artifacts/validation/handle_v2_lever_quick_20260829_01/stress.log
```

중간 결과:

- 사용자 요청으로 run 중간 종료.
- 모델 로드는 정상.
- 파란문 2개는 fallback 좌표로 문개방 성공.
- `Open door request`에서 `handle_detected=False`.
- `YOLO handle observed` 로그는 안 나왔고 `HSV handle observed`만 반복됨.
- 따라서 현재 Gazebo 기준으로는 손잡이 YOLO 효과가 입증되지 않음.

가능한 원인:

- Gazebo 손잡이 형상이 실제 레버 사진과 다름.
- 320x240 카메라에서 손잡이가 너무 작게 보임.
- door bbox가 화면 상단/가장자리에서 잘려 ROI가 손잡이를 충분히 포함하지 못할 수 있음.
- v2 학습 데이터가 실제 시연 손잡이와는 맞을 수 있으나 Gazebo 도메인과는 맞지 않을 수 있음.

실제 손잡이에서는 효과가 있을 수 있다. 단, 실제 로봇 카메라 높이/거리/조명에서 찍은 사진으로 inference 확인이 필수다.

## 바로 해야 할 일

1. 현재 코드 상태 확인

```bash
cd ~/fire_robot_ws_test
git status --short
```

2. 빌드/문법 확인

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
python3 -m py_compile \
  tools/convert_blue_doors_to_hinged.py \
  scripts/check_validation_log.py \
  src/fire_robot_bringup/launch/simulation.launch.py \
  src/fire_robot_fsm/fire_robot_fsm/state_machine_node.py \
  src/fire_robot_fsm/fire_robot_fsm/fsm_subsystems.py \
  src/fire_robot_manipulation/fire_robot_manipulation/manipulation_node.py \
  src/fire_robot_perception/fire_robot_perception/door_detection_node.py
```

3. 손잡이 모델 확인

```bash
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('src/fire_robot_perception/models/handle_best_v2.pt')
print(m.names)
PY
```

4. stress world부터 검증

```bash
wsl --terminate Ubuntu2204Recovered
wsl -d Ubuntu2204Recovered -u junyoung --cd /home/junyoung/fire_robot_ws_test -- bash -lc '
source /opt/ros/humble/setup.bash
source install/setup.bash
mkdir -p artifacts/validation/team_check/stress
TRACE_DIR=artifacts/validation/team_check/stress \
scripts/run_headless_validation_once.sh \
observation_fsm_quick_validation.world \
artifacts/validation/team_check/stress.log \
1100 \
publish_debug_image:=true \
log_handle_detections:=true
python3 scripts/check_validation_log.py \
artifacts/validation/team_check/stress.log \
--expected-open-count 3 \
--max-rejected 0
'
```

5. 로그 판정

```bash
grep -aE 'State:|Fine door approach accepted|Open door request|handle_detected|YOLO handle observed|HSV handle observed|문 개방 성공|Rejected Gazebo|MISSION_COMPLETE|timeout|failed|EMERGENCY' \
artifacts/validation/team_check/stress.log | tail -240
```

PASS 조건:

- `MISSION_COMPLETE` 존재.
- `문 개방 성공` 3회.
- `Rejected Gazebo door match` 0회.
- 빨간문 opened 없음.
- 문앞 정렬 accepted 로그가 20~30cm대, yaw 10도 이내.

손잡이 YOLO 성공 조건:

- `YOLO handle observed` 존재.
- `/open_door` 요청에서 `handle_detected=True`.
- `handle_method=yolo:lever_handle` 또는 그에 준하는 yolo method.

6. stress 통과 후 world1~5 순차 검증

월드와 기대 open count:

```text
world1: obstacle_wall_doors_v5.world, expected_open=3
world2: obstacle_door_layout_alt_v1.world, expected_open=3
world3: obstacle_door_layout_world3_v1.world, expected_open=4
world4: obstacle_no_blue_world4_v1.world, expected_open=0
world5: obstacle_all_blue_world5_v1.world, expected_open=6
```

검증 예시:

```bash
TRACE_DIR=artifacts/validation/team_check/world1 \
scripts/run_headless_validation_once.sh \
obstacle_wall_doors_v5.world \
artifacts/validation/team_check/world1.log \
1800 \
publish_debug_image:=false \
log_handle_detections:=true

python3 scripts/check_validation_log.py \
artifacts/validation/team_check/world1.log \
--expected-open-count 3 \
--max-rejected 0
```

## 실패 시 디버깅 기준

주행 실패라면:

- target lock이 다른 문/빨간문/출구 후보로 흔들렸는지 확인.
- opened station remnant가 새 문으로 잘못 선택됐는지 확인.
- Nav2 timeout인지, controller stuck인지, costmap obstacle 문제인지 분리.
- 기존 자율주행 성공 branch/tag와 주행 정책 차이를 비교.

문앞 정렬 실패라면:

- `Fine door approach started/accepted` 로그의 dist/lateral/yaw를 본다.
- 목표가 handle 좌표인지 door station 좌표인지 확인.
- 정렬 상태에서 문이 화면 가장자리로 잘려 handle ROI가 손잡이를 놓치는지 확인.

손잡이 YOLO 실패라면:

- `publish_debug_image:=true`로 `/door_detection/debug` 저장.
- 실제 카메라 장면 또는 Gazebo 문앞 장면에서 `handle_best_v2.pt`를 직접 inference.
- ROI가 손잡이를 포함하는지 확인.
- 모델이 실제 손잡이에는 잡히는데 Gazebo만 못 잡으면, Gazebo synthetic 이미지를 추가 학습하거나 시뮬레이션은 HSV/fallback 기준으로 분리한다.

출구 실패라면:

- 미개방 파란문 후보가 남아 있어 EXITING을 막는지 확인.
- 초록 비상구가 camera bbox/HSV 조건을 만족하는지 확인.
- quick stress world에서는 v2 live run 중 출구 확정이 늦어진 로그가 있었다.

## 팀원에게 부탁할 재학습 데이터

파일 `docs/HANDLE_RETRAIN_REQUEST_20260829.md`를 읽고 그대로 진행하면 된다.

핵심:

- class는 `lever_handle` 1개.
- 실제 시연 손잡이와 Gazebo 문앞 정렬 이미지 둘 다 포함.
- 거리 0.2m, 0.4m, 0.8m, 1.5m.
- 정면/좌우 30도/좌우 60도.
- 밝은 조명/어두운 조명/역광.
- 목표는 Precision보다 Recall 우선.

## 최종적으로 남겨야 할 증거

검증 통과 시 아래를 남겨라.

```text
artifacts/validation/final_YYYYMMDD/world1/trajectory.png
artifacts/validation/final_YYYYMMDD/world1/door_alignment.png
artifacts/validation/final_YYYYMMDD/world1/summary.txt
artifacts/validation/final_YYYYMMDD/world1.log
...
world5까지
```

그리고 README에 다음을 명확히 적어라.

- 어떤 월드가 PASS인지.
- expected opened count와 실제 opened count.
- 손잡이 YOLO가 성공했는지 fallback인지.
- 문앞 정렬 수치.
- 미해결 리스크.

## 현재 사용자에게 보고해야 할 요지

- 문 YOLO는 문 bbox 검출에 효과가 있고, 색상은 HSV가 담당한다.
- 새 손잡이 YOLO v2는 모델 로드는 정상이나 Gazebo에서는 아직 효과가 입증되지 않았다.
- 실제 손잡이가 학습 데이터와 비슷하면 실제에서는 효과가 있을 수 있다.
- 하지만 실제 카메라 위치 샘플 검증과 추가 fine-tuning이 필요하다.
- 현재 주행 성공 코드는 보존되어 있고, 현재 작업본은 손잡이/로봇팔 strict 통합 기준을 맞추는 단계다.

---
