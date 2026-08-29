# Fire Robot Workspace

화재/문 탐지 로봇 시뮬레이션 및 AgileX PIPER 연동 준비용 ROS2 Humble 워크스페이스입니다.

목표 동작은 시작 위치 기준으로 관측한 벽, 문, 장애물 구조를 map 좌표에 축적하고, 가장 가까운 파란문을 선택해 장애물을 피해 접근한 뒤 문 개방 FSM을 수행하는 것입니다. 더 이상 열 파란문이 없으면 초록 비상구를 관측 기반으로 선택해 통과합니다.

## 현재 상태 (2026-08-29)

- `colcon build --symlink-install` PASS
- Python 문법 검사 PASS
- Gazebo headless full validation PASS 이력 있음
- World 1~5 전체 `MISSION_COMPLETE` 이력 있음
- 자율주행 성공 기준 코드는 보존됨
  - 보존 브랜치: `codex/navigation-success-before-arm`
  - 보존 태그: `navigation-success-before-arm-20260826`
  - 보존 커밋: `41ec296 feat: finalize observation-based navigation validation`
- 현재 작업본은 더 엄격한 기준으로 재검증 중
  - 목표: semantic map 기반 문 후보 누적, nearest unopened blue target lock, 문앞 정면 정렬, 레버 누름 + push-open, opened/failed station 메모리, 모든 파란문 처리 후 초록 비상구 통과
  - 스트레스 검증 월드: `observation_fsm_quick_validation.world`
  - 스트레스 월드 기존 모델/fallback 기준 PASS: 파란문 3/3 opened, `MISSION_COMPLETE`
  - 2026-08-29 새 손잡이 모델 v2 + 레버형 Gazebo 손잡이 기준 검증은 진행 중이며, 현재까지는 손잡이 YOLO가 아니라 fallback 좌표로 문개방이 수행됨
- YOLOv8s Door 1-class 모델 적용
  - 모델: `src/fire_robot_perception/models/best.pt`
  - 학습 성능: mAP50 `0.6088`, mAP50-95 `0.3923`, Precision `0.6184`, Recall `0.5817`
  - 색상 분류는 YOLO가 아니라 HSV 로직에서 `red/blue/green`으로 별도 처리
- 손잡이 YOLO 연결 경로 추가
  - 기본 경로: `src/fire_robot_perception/models/handle_best_v2.pt`
  - 2026-08-29 팀원 전달 `handle_best_v2.pt`를 기본 손잡이 모델로 적용
  - 모델 클래스: `lever_handle`
  - 기존 Gazebo 정적 샘플 8장 1차 테스트에서는 detection 0건
  - Gazebo의 얇은 원통형 손잡이가 학습 데이터의 레버 손잡이와 달라서, 파란 힌지문의 손잡이 형상을 `handle_backplate` + 직사각 레버로 수정
  - 수정 후 live stress run에서도 아직 `handle_detected=False`가 반복되어, 실제 레버 손잡이 사진 또는 새 Gazebo 정렬 장면으로 추가 fine-tuning 필요
  - 손잡이 YOLO 모델이 없거나 미검출이면 HSV 손잡이 blob, 이후 문 위치 기반 추정값으로 fallback

## 문 개방 시뮬레이션 (2026-08-24)

- 파란문 Gazebo 모델을 visual-only marker에서 `static=false` 힌지 문으로 변경했습니다.
- 각 파란문은 패널 collision, 손잡이 collision, `hinge` revolute joint, `JointPositionController`를 가집니다.
- 2026-08-29에는 손잡이를 얇은 원통에서 실제 레버에 가까운 `handle_backplate` + 수평 레버 박스 형상으로 바꿨습니다.
- `manipulation_node`는 `/open_door` 요청을 받으면 현재 월드 파일의 힌지문 registry를 읽고, 관측된 손잡이 좌표와 가장 가까운 파란문 joint topic에 열림 각도를 보냅니다.
- Gazebo sim 전용 PIPER joint position controller를 추가해 pre-grasp, grasp, lever-press, push-open, home 단계에서 팔이 움직이는 모습을 확인할 수 있습니다.
- `door_detection_node`는 문 bbox 내부/주변에서 손잡이 YOLO 모델을 먼저 실행하고, 실패하면 노란/금색 손잡이 blob, 이후 기존 문 위치 기반 추정값으로 fallback합니다.
- `manipulation_node`는 `LOCALIZE_HANDLE -> PRE_GRASP -> GRASP_HANDLE -> PRESS_HANDLE -> PUSH_OPEN -> RETURN_HOME -> COMPLETE` sub-FSM 단계를 `/manipulation_phase`로 발행합니다.
- 스모크 테스트에서 `/open_door` 호출 후 `door_blue1` pose가 실제로 회전/이동하는 것을 확인했습니다.
- 팀원 로봇팔 데모 아이디어를 반영해 `manipulation_demo.launch.py`와 `validate_manipulation_demo.sh`를 추가했습니다.
- 로봇팔 단독 데모는 ROS-Gazebo command bridge로 arm/gripper/door hinge를 움직이고, `/door_joint_states`에서 문 힌지가 목표각까지 열렸을 때만 `/open_door` 성공으로 판정합니다.
- `manipulation_demo.world`에는 검증용 Gazebo overhead camera가 있고, `scripts/capture_manipulation_visual_proof.sh`로 실제 Gazebo 렌더링 전/후 이미지와 짧은 mp4를 생성합니다.
- 전체 FSM 시뮬레이션에서는 파란문을 관측 기반으로 모두 열고, 더 이상 파란문이 없을 때 초록 비상구 통과까지 확인했습니다.

## 주요 구조

### Perception

- 카메라: `front`, `front_left`, `front_right`
- 문 검출: YOLO Door bbox + HSV 색상 분류
- 손잡이 검출: Door bbox ROI 안에서 YOLO handle 모델 우선, HSV blob fallback
- 장애물/거리: 2D LiDAR 기반 observation/costmap
- 문 후보는 map 좌표의 관측 메모리로 축적하고, 열린 문/실패 문/빨간문과 충돌하는 후보를 억제합니다.

### Navigation

- Planner: `nav2_smac_planner/SmacPlanner2D`
- Controller: `nav2_rotation_shim_controller::RotationShimController`
- Primary controller: `nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController`
- 초기 static map + localization 기반 주행
- LiDAR 관측 맵으로 장애물을 반영하고, 문/비상구 목표는 관측된 map 좌표에서 생성합니다.

### FSM

- 파란문 후보를 하나 선택하면 target lock을 걸고, 열기 전까지 다른 후보가 끼어들지 않게 합니다.
- 문 앞에서는 Nav2 접근 후 fine alignment로 정면 정렬합니다.
- 열린 문은 map 기준 위치로 기록해 중복 접근을 막습니다.
- 접근/열기 실패는 제한 횟수 이후 abandoned 처리할 수 있습니다.
- 파란문 후보가 더 없으면 마지막 측면 스캔 후 초록 비상구로 전환합니다.

### Manipulation Simulation

- 기본 시뮬레이션 launch에서는 `sim_physical_door_opening=True`입니다.
- `fire_robot_bringup/worlds` 안에서 파란문이 있는 월드는 힌지 기반으로 열립니다.
- 월드 4는 파란문이 없는 검증 월드라 문 개방 없이 비상구로 이동합니다.
- 힌지 문 변환 도구는 `tools/convert_blue_doors_to_hinged.py`입니다.
- 로봇팔 단독 검증은 `ros2 launch fire_robot_bringup manipulation_demo.launch.py` 또는 `bash scripts/validate_manipulation_demo.sh`로 실행합니다.
- 시각 증거 캡처는 `bash scripts/capture_manipulation_visual_proof.sh`로 실행합니다.

## 검증 결과

### 이전 전체 검증 (2026-08-24)

| 월드 | 조건 | 결과 |
| --- | --- | --- |
| World 1 `corridor.world` | 파란문 3개, 빨간문/장애물 혼합 | PASS, 파란문 3/3, `MISSION_COMPLETE` |
| World 2 `obstacle_door_layout_alt_v1.world` | 다른 문 배열/장애물 배치 | PASS, 파란문 3/3, `MISSION_COMPLETE` |
| World 3 `obstacle_door_layout_world3_v1.world` | 파란문 4개, 빨간문 2개 | PASS, 파란문 4/4, `MISSION_COMPLETE` |
| World 4 `obstacle_no_blue_world4_v1.world` | 파란문 없음 | PASS, 문 개방 0/0, no-blue fallback 후 `MISSION_COMPLETE` |
| World 5 `obstacle_all_blue_world5_v1.world` | 좌우 3개씩 모든 문 파란색 | PASS, 파란문 6/6, `MISSION_COMPLETE` |

증빙:

- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\all_worlds_validation_summary.png`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world*\trajectory.png`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world*\door_alignment.png`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world*\world*.log`

### 현재 스트레스 검증 (2026-08-29)

| 월드 | 조건 | 결과 |
| --- | --- | --- |
| `observation_fsm_quick_validation.world` | 파란문 3개, 빨간문 2개, 중앙/문앞 장애물 | PASS, 파란문 3/3, `MISSION_COMPLETE` |

로그:

- `artifacts/validation/semantic_stress_20260828_02/stress.log`

주의:

- 이 PASS는 기존 handle 모델/fallback 조합 기준이다.
- 새 `handle_best_v2.pt`와 레버형 Gazebo 손잡이 적용 후 1차 live run은 중간 종료했다.
- 해당 run에서는 파란문 2개가 fallback 좌표로 개방 처리됐고, 로그상 손잡이 YOLO 검출은 확인되지 않았다.
- 따라서 현재 미해결 핵심은 주행 전체보다 “손잡이 YOLO가 실제/Gazebo 레버를 안정적으로 잡는지”와 “새 엄격 기준에서 stress/world1~5를 다시 끝까지 PASS시키는 것”이다.

## 실행 명령

WSL 정리:

```bash
wsl --terminate Ubuntu2204Recovered
```

빌드:

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

Gazebo + RViz2 실행:

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
source install/setup.bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=true headless:=false \
  world:=obstacle_wall_doors_v5.world
```

Headless full validation:

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
source install/setup.bash
TRACE_DIR=artifacts/validation/latest/world1 \
  bash scripts/run_headless_validation_once.sh corridor.world artifacts/validation/latest/world1.log 3000
```

손잡이 YOLO 학습:

```bash
cd ~/fire_robot_ws_test/src/fire_robot_perception/scripts
python3 train_handle_detector.py \
  --dataset ~/datasets/door_handle_detection/dataset.yaml \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --install
```

손잡이 YOLO 강제 실제 실행:

```bash
ros2 launch fire_robot_bringup real_robot.launch.py \
  handle_model_path:=~/fire_robot_ws_test/src/fire_robot_perception/models/handle_best_v2.pt \
  require_yolo_handle:=true
```

손잡이 검증 디버그 실행:

```bash
ros2 launch fire_robot_bringup simulation.launch.py \
  world:=observation_fsm_quick_validation.world \
  publish_debug_image:=true \
  log_handle_detections:=true
```

## 실제 로봇 전 남은 작업

- 실제 PIPER CAN `can0` 연결 확인
- `ros2 action list | grep piper`로 `/piper_arm_controller/follow_joint_trajectory` 제공 여부 확인
- PIPER MoveIt 단독 동작 확인
- 그리퍼 open/close 값 실측
- 실제 모바일 베이스의 `/odom -> base_link` TF 확인
- 실제 카메라 장착 위치와 camera TF 확인
- 실제 2D LiDAR 높이/각도 기준 costmap 파라미터 튜닝
- 연구실 조명 기준 HSV 임계값 튜닝
- 손잡이 전용 YOLO 모델 학습 후 `models/handle_best_v2.pt` 설치

## 주의

Gazebo/ROS 프로세스가 남아 있으면 `/clock`, TF, Nav2 action result가 꼬여 가짜 실패가 생길 수 있습니다. 검증 전에는 WSL을 한 번 종료하고 새로 시작하는 것을 권장합니다.
