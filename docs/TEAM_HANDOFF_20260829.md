# 팀원 인수 문서 (2026-08-29)

## 현재 목표

졸업작품 로봇은 단순히 문을 발견할 때마다 즉시 접근하는 구조가 아니라, 주행 중 관측한 문/장애물/출구 정보를 map 좌표계에 누적해 semantic map을 만들고, 그 지도에서 가장 가까운 미개방 파란문을 선택해 순서대로 여는 구조로 정리 중임.

최종 동작 목표:

1. 시작 후 카메라 3대(front/front_left/front_right)와 2D LiDAR로 전방/측면을 관측한다.
2. 벽, 장애물, 문 색상, 손잡이 후보를 map 좌표에 누적한다.
3. semantic map에서 가장 가까운 unopened blue door를 선택하고 target lock을 건다.
4. 해당 문을 열기 전까지 다른 blue 후보로 목표가 흔들리지 않게 한다.
5. Nav2가 static map + LiDAR/costmap 기반으로 장애물을 피해 이동한다.
6. 문 근처에서는 fine alignment로 문 정면 20~30cm대, lateral 약 10cm 이내, yaw 약 10도 이내 정렬을 목표로 한다.
7. manipulation FSM이 손잡이 위치를 받아 `LOCALIZE_HANDLE -> PRE_GRASP -> GRASP_HANDLE -> PRESS_HANDLE -> PUSH_OPEN -> RETURN_HOME -> POST_OPEN_BACKOFF -> COMPLETE` 순서로 문 개방을 수행한다.
8. 열린 문은 map 좌표/semantic landmark 기준으로 opened 처리해 재탐지 후보에서 제외한다.
9. 모든 파란문을 처리한 뒤 초록 비상구를 관측 기반으로 선택해 통과한다.

## 지금까지 반영한 주요 변경

- `state_machine_node.py`
  - main FSM에 문 접근/문 개방/출구 전환 조건을 계속 보강.
  - target lock 후 다른 파란문 후보로 goal이 드리프트되는 문제를 줄이기 위해 locked target refine를 simulation launch에서 비활성화.
  - live detection을 항상 우선하던 선택 로직을 제거하고 semantic map 후보도 현재 로봇 위치 기준 nearest target으로 선택하도록 수정.
  - opened/abandoned/failed 문 후보는 map station 기준으로 다시 선택되지 않게 억제.
  - 문 앞 정렬 후 live blue 검증이 약간 흔들려도 반복 관측된 semantic map evidence가 있으면 조작 단계로 넘어갈 수 있게 완화.

- `fsm_subsystems.py`
  - Door target selection과 live blue evidence matching 정책을 분리.
  - observed_blue target의 live 후보 매칭 허용 범위를 정렬 상태 기준으로 재조정.
  - pre-exit 영역에서 같은 physical station으로 인정된 경우 마지막 point-distance 중복 검사 때문에 false reject되는 문제를 수정.

- `door_detection_node.py`
  - Door YOLO + HSV 색상 분류 구조 유지.
  - handle YOLO를 door ROI에서 먼저 사용하고, 실패하면 HSV/fallback으로 손잡이 위치를 만든다.
  - 현재 v2 handle 모델도 로드는 되지만, 저장된 Gazebo 샘플 이미지와 live stress run 기준으로 YOLO 손잡이 검출은 아직 확인되지 않았다.

- `manipulation_node.py`
  - Gazebo 힌지문 command topic 기반 문 개방 검증.
  - 요청된 handle 좌표와 월드 내 blue door registry를 axis/progress/lateral 기준으로 매칭한다.
  - 너무 멀리 투영된 문은 `Rejected Gazebo door match`로 실패 처리한다.

- `simulation.launch.py`
  - `handle_model_path` 기본값은 `fire_robot_perception/models/handle_best_v2.pt`.
  - 새 `handle_best_v2.pt`를 기본 손잡이 모델로 적용.
  - `publish_debug_image`, `log_handle_detections` launch argument를 추가해 손잡이 검출 증거를 남길 수 있게 했다.
  - 목표 lock 후 refine 비활성화.
  - 문 앞 정렬/출구 판단/관측 메모리 관련 파라미터를 스트레스 검증 기준으로 조정.

- `tools/convert_blue_doors_to_hinged.py`
  - 앞으로 visual-only blue door를 힌지문으로 변환할 때 손잡이를 얇은 원통이 아니라 `handle_backplate` + 수평 레버 박스로 생성하도록 수정했다.

- `fire_robot_bringup/worlds`
  - `obstacle_wall_doors_v5.world`, `obstacle_door_layout_alt_v1.world`, `obstacle_door_layout_world3_v1.world`, `obstacle_all_blue_world5_v1.world`, `corridor.world`, stress/random seed worlds, quick validation world의 파란 힌지문 손잡이를 레버형으로 정규화했다.

- `observation_fsm_quick_validation.world`
  - 기존 쉬운 검증 월드를 스트레스 월드로 변경.
  - 파란문 3개, 빨간문 2개, 중앙 장애물 5개, 문 앞 장애물, 양쪽 벽 문 배치로 구성.
  - 목표 드리프트, 열린 문 잔상, 손잡이 fallback, 문 앞 정렬 문제를 빠르게 드러내는 용도.

## 최신 검증 상태

### v1/기존 handle 모델 기반 스트레스 월드

검증 로그:

```text
artifacts/validation/semantic_stress_20260828_02/stress.log
```

결과:

```text
PASS mission_complete=True open_success_count=3 matched_blue_doors=3 rejected_gazebo_matches=0
matched_topics=/fire_robot/door/blue/x1040/yn186/sp/cmd,/fire_robot/door/blue/x280/yn186/sp/cmd,/fire_robot/door/blue/x640/yp186/sn/cmd
```

확인된 내용:

- 스트레스 월드 파란문 3개 모두 opened.
- 빨간문 opened 없음.
- 초록 비상구 `MISSION_COMPLETE`.
- 문앞 정렬 로그 예:
  - 1번 문: dist 0.22m, lateral 0.11m, yaw 6deg.
  - 2번 문: dist 0.25m, lateral 0.04m, yaw 3deg.
  - 3번 문: dist 0.20m, lateral 0.11m, yaw 7deg.
- 단, 손잡이 YOLO는 검출하지 못했고 `observed_wall_projection` 또는 `direct_wall_projection` fallback으로 문 개방이 진행됨.

### handle_best_v2.pt 적용 직후 정적 이미지 검사

모델:

```text
src/fire_robot_perception/models/handle_best_v2.pt
sha256=0c2681fd954da54b71aba2234cfecf1e19832a312c18ba8d156a9a02de463449
class={0: 'lever_handle'}
```

GitHub에는 중복 방지를 위해 ROS 기본 경로인 `src/fire_robot_perception/models/handle_best_v2.pt`만 추적한다. 예전 `handle_best.pt` alias는 기본 실행에서 사용하지 않는다.

정적 이미지 테스트:

```text
artifacts/validation/handle_v2_static_20260829/summary.json
```

결과:

- 기존 Gazebo camera sample 8장에 대해 conf 0.05, imgsz 960으로 inference.
- detection 0건.
- 이 샘플들이 손잡이를 작게 보거나 아예 잘 안 보이는 장면일 수 있으므로, Gazebo 손잡이 형상을 레버형으로 수정한 뒤 live ROS/Gazebo 검증을 추가했다.

### handle_best_v2.pt + 레버형 Gazebo 손잡이 live run

검증 명령:

```bash
TRACE_DIR=artifacts/validation/handle_v2_lever_quick_20260829_01/stress \
  scripts/run_headless_validation_once.sh \
  observation_fsm_quick_validation.world \
  artifacts/validation/handle_v2_lever_quick_20260829_01/stress.log \
  700 \
  publish_debug_image:=false \
  log_handle_detections:=true
```

중간 결과:

- 사용자가 GitHub/전달자료 우선 요청해서 run을 중간 종료했다.
- 모델 로드 확인: `classes=['lever_handle']`.
- 파란문 2개는 fallback 좌표로 개방 성공.
- `Open door request` 로그는 둘 다 `handle_detected=False`.
- 로그에는 `YOLO handle observed`가 없고 `HSV handle observed`만 반복됨.
- 3번째 문/출구 처리까지는 이 run에서 완료 확인하지 못함.

핵심 판단:

- v2 모델 파일 자체는 정상이다.
- 하지만 현재 Gazebo 장면에서는 아직 손잡이 YOLO 효과가 입증되지 않았다.
- 원인은 모델이 실제 레버 사진 위주이고, Gazebo 손잡이는 단순 색상/기하/해상도/ROI가 달라 domain gap이 큰 것으로 보인다.
- 실제 시연 손잡이가 학습 데이터와 비슷하면 실제에서는 성능이 더 나올 수 있지만, 실제 카메라 장착 위치에서 30~50장 이상 확인해야 한다.

다음 검증에서 봐야 할 핵심 로그:

```bash
grep -aE 'State:|Locked blue|Navigation succeeded|Navigation failed|Fine door approach accepted|문 앞 정렬|문 개방 요청 손잡이|문 개방 성공|handle_detected|Matched observed door|Rejected Gazebo|MISSION_COMPLETE|EMERGENCY|timeout|failed|stuck' \
  artifacts/validation/handle_v2_lever_quick_20260829_01/stress.log | tail -200
```

판정:

- `MISSION_COMPLETE` 있어야 함.
- `문 개방 성공` 3회여야 함.
- `Rejected Gazebo door match` 없어야 함.
- `Open door request`에서 `handle_detected=True`, `handle_method=yolo` 계열이면 v2가 실제 도움을 준 것.
- 계속 `handle_detected=False`면 주행/문개방은 fallback으로 가능한 상태지만, YOLO 손잡이 검출 모델은 추가 데이터가 더 필요함.

## 왜 오류가 갑자기 많이 나온 것처럼 보였는지

이전 full pass는 자율주행 기준 통과에 가까웠다. 즉 파란문 후보를 찾고, 장애물을 회피하고, 문 근처 정렬 후 문 개방 상태 전환까지 확인하는 기준이었다.

로봇팔 FSM을 붙인 뒤에는 요구 조건이 더 엄격해졌다.

- 문 근처 방문이 아니라 문 정면 가까이 정렬해야 함.
- 손잡이 좌표가 arm workspace 안에 있어야 함.
- 손잡이/문 station 좌표가 Gazebo 실제 힌지문 registry와 맞아야 함.
- 열린 문 잔상이나 색 번짐이 새 파란문으로 재선택되면 안 됨.
- 출구가 보여도 미개방 파란문이 남아 있으면 EXITING보다 파란문 처리가 우선임.

그래서 기존에는 드러나지 않던 target drift, false fresh-blue reject, handle fallback 의존, open-match reject가 새로 드러난 것이다.

## 팀원에게 부탁할 것

우선순위:

1. 손잡이 모델 검증/보강
   - 현재 `handle_best_v2.pt`는 `lever_handle` 클래스 로드는 정상.
   - Gazebo 정적 샘플 8장과 live stress run에서는 YOLO 검출이 아직 안 보임.
   - 실제 시연 장소 레버 손잡이를 로봇 카메라 높이/거리/각도와 비슷하게 촬영해 `handle_best_v2.pt`에 inference를 돌려봐야 함.
   - 실제 사진에서도 Recall이 낮으면 dataset에 실제 손잡이/시뮬레이션 손잡이 이미지를 추가해 fine-tuning 필요.

2. 엄격 기준 full validation 재실행
   - quick stress world부터 다시 끝까지 돌린다.
   - stress PASS 후 world1~5를 순서대로 검증한다.
   - 실패 시 policy 전체를 갈아엎지 말고, target lock/opened station/exit scan/handle detection 중 어디서 깨졌는지 로그로 좁힌다.

3. 실제 로봇 준비
   - PIPER CAN, MoveIt action server, gripper, camera/LiDAR TF를 확인한다.
   - 실제 손잡이 workspace와 문앞 정렬 거리 20~30cm가 물리적으로 가능한지 실측한다.

## 팀원이 이어서 보면 좋은 순서

1. `git status --short`로 현재 변경 파일 확인.
2. `python3 -m py_compile`로 FSM/perception/manipulation 문법 확인.
3. v2 모델이 적용됐는지 확인:

```bash
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('src/fire_robot_perception/models/handle_best_v2.pt')
print(m.names)
PY
```

4. 스트레스 월드부터 검증:

```bash
wsl --terminate Ubuntu2204Recovered
wsl -d Ubuntu2204Recovered -u junyoung --cd /home/junyoung/fire_robot_ws_test -- bash -lc '
mkdir -p artifacts/validation/team_check/stress
TRACE_DIR=artifacts/validation/team_check/stress \
scripts/run_headless_validation_once.sh \
observation_fsm_quick_validation.world \
artifacts/validation/team_check/stress.log \
1100 \
publish_debug_image:=true \
log_handle_detections:=true
python3 scripts/check_validation_log.py artifacts/validation/team_check/stress.log --expected-open-count 3 --max-rejected 0
'
```

5. 스트레스 월드 PASS 후 월드 1~5 순차 검증:

```bash
python3 scripts/check_validation_log.py LOG --expected-open-count N --max-rejected 0
```

기대 open count:

| 월드 | 파일 | 기대 open |
| --- | --- | ---: |
| stress | `observation_fsm_quick_validation.world` | 3 |
| world1 | `obstacle_wall_doors_v5.world` | 3 |
| world2 | `obstacle_door_layout_alt_v1.world` | 3 |
| world3 | `obstacle_door_layout_world3_v1.world` | 4 |
| world4 | `obstacle_no_blue_world4_v1.world` | 0 |
| world5 | `obstacle_all_blue_world5_v1.world` | 6 |

## 앞으로 남은 핵심 작업

1. handle_best_v2 live Gazebo 검증 결과 확인.
2. v2가 계속 `handle_detected=False`면 데이터셋을 추가해야 함.
3. 추가 데이터는 “우리 시연 레버 손잡이 + Gazebo 문앞 정렬 카메라 이미지 + 공개 lever-handle dataset” 중심.
4. semantic map-first 구조를 더 명확히 하려면 main FSM을 `BUILD_SEMANTIC_MAP -> SELECT_BLUE_TARGET -> NAV_TO_TARGET -> FINE_ALIGN -> OPEN_DOOR -> UPDATE_MAP -> EXIT_SCAN -> EXIT`로 더 명시적으로 분리.
5. 실제 로봇 전에는 PIPER CAN, MoveIt action server, gripper 값, camera/LiDAR TF, HSV 조명 튜닝 확인 필요.
