# Fire Robot Workspace

화재/문 탐지 로봇 시뮬레이션 및 AgileX PIPER 연동 준비용 ROS2 Humble 워크스페이스입니다.

목표 동작은 시작 위치 기준으로 관측한 벽, 문, 장애물 구조를 map 좌표에 축적하고, 가장 가까운 파란문을 선택해 장애물을 피해 접근한 뒤 문 개방 FSM을 수행하는 것입니다. 더 이상 열 파란문이 없으면 초록 비상구를 관측 기반으로 선택해 통과합니다.

## 최신 코드와 자료 (2026-09-17)

**최신 개발 브랜치: `codex/world1-contact-integration-20260916`.** 이전 성공 주행 코드는 `codex/feedback-resume-20260914`의 `d1ffb39`로 별도 보존합니다. 기본 `master`의 로봇 소스는 이전 버전이므로 최신 접촉 시험을 실행할 때는 개발 브랜치를 사용하세요.

- [최신 검증 범위와 한계](docs/VALIDATION_20260917.md) / [팀원 인계](docs/TEAM_HANDOFF_20260917.md)
- [REV6 보고서·영상 읽는 순서](docs/reports/2026-09-17/00_읽는순서.md) / [보고서 PDF](docs/reports/2026-09-17/화재대응_로봇_핵심아이디어와_검증_20260917_REV6.pdf)
- 월드 1~5 주행: 2026-09-16 동일 입력 코드 기준 5/5 PASS, 문 회전 16/16. **저장 지도 + sim_odom + 기존 힌지 명령 backend** 조건입니다.
- 단일문 r40: 2026-09-17 **손잡이 접촉 유지 → 레버 누름 → 본체·팔 협조 개방 → 해제·팔 회수·후진** PASS. 유지 개방 117.72도, 최종 117.76도. 문 힌지 직접 명령/가상 파지 연결은 사용하지 않았습니다.
- 두 결과는 별도 시험입니다. 새 PIPER 물리 접촉을 모든 월드에 결합한 FULL, 실제 하드웨어 검증, 다양한 마찰 조건의 반복 신뢰도 검증은 미완료입니다. 후진은 아직 설정 속도·시간 제어이며 거리 피드백 제어가 아닙니다.
- 이번 동기화에서 `212`개 스크립트 테스트 PASS를 재확인했습니다. 새 Gazebo FULL 실행 결과를 추가한 것은 아닙니다.

아래 내용은 날짜별 보존 기록입니다. 특히 과거 차체로 문 패널을 미는 방식과 r21 FAIL은 최신 r40 동작 설명이 아닙니다.

## 재개 검증 안내 (2026-09-16 보존)

**최신 완료 결과:** r8(월드 3~5) + r9(월드 1~2)의 동일 런타임/검사 해시를 확인하여 통합 5/5 PASS, 실제 문 모델 회전 16/16을 기록했습니다. 별도 PIPER 접촉 r21은 최대 3.50도로 FAIL입니다. 상세 범위·수정·사진 위치는 [2026-09-16 결과](docs/VALIDATION_20260916.md)를 참고하세요. 아래 r5/r7은 보존된 중간 실행 기록입니다.

**팀원 작업 기준:** `codex/feedback-resume-20260914`. [최신 인계 안내](docs/TEAM_HANDOFF_20260916.md)와 [대표 검증 이미지](docs/validation/2026-09-16/README.md)를 먼저 확인하세요. 기본 `master`는 이전 성공 코드 보존용이며 새 피드백 작업본과 같지 않습니다. 새 접촉 개방은 미완료이므로 완성판으로 표시하지 않습니다.

이전 성공 코드와 검증 자료는 보존하고 `codex/feedback-resume-20260914`에서 수정/재검증 중입니다.
아래 9월 13일 기록은 당시 코드의 결과이며, 새 피드백 제어의 성공을 의미하지 않습니다.

- 새 검사에서 이전 full의 서비스 성공이 실제 문 회전을 보장하지 않는 결함을 확인했습니다. 문의 실제 Gazebo 모델 회전을 확인하도록 수정했고, 문 뒤 연속 벽은 문틀 개구부/힌지 회전 여유로 수정했습니다. 문·장애물 위치는 유지했습니다.
- r5: 실제 문 회전 16/16 확인, 월드 1~4 강화 검사 PASS, 월드 5는 차체 영역과 장애물 겹침 115개 시료로 FAIL입니다.
- r7: 정렬 전 코스트맵 검사 및 전체 맵 주기 발행 후 월드 1·2 PASS. 월드 3은 3개 문 회전 확인 뒤 중단되어 PASS가 아닙니다. 원본 로그/카메라는 보존했지만 종료 시 저장되던 궤적 CSV는 남지 않았습니다.
- 후속 수정: 코스트맵 차단을 시각적 문 재확인 실패와 분리하고 동일 목표의 제한된 재계획으로 처리합니다. 궤적 CSV는 5초마다 증분 저장/동기화하도록 보강했습니다. 회귀 테스트 97개 PASS, FSM 재빌드 PASS입니다.
- r21 독립 접촉 시험: v3 YOLO 65회, 최대 문 각도 3.50도, 추가 누름 반응 없음으로 안전 정지했습니다. 보호 분기는 확인했지만 완전 개방 FAIL입니다.

- 통합 미션 회귀: 저장된 static map + sim odometry + 단순화 팔/힌지 명령 backend 조건으로 월드 1~5를 순차 실행합니다. 실제 PIPER 접촉 개방을 결합한 full 검증은 아닙니다.
- 새 관측/피드백 기반 PIPER 독립 시험: YOLO/RGB-D 접근과 레버 누름은 확인했지만 r20에서 최대 개방 6.59도로 FAIL입니다. 실제 레버/문 관절 반응이 없는 추가 누름을 중단하도록 보강했습니다.
- 주행 수정: 수동 회피/복구 중 관측 갱신이 Nav2 목표를 재시작하지 않도록 차단하고, 개방 후 중앙 복귀는 선회 오차 10도 이내에서만 전진하도록 변경했습니다.
- 검증 보강: 즉시 `DOOR_OPENED` 발행, 고유 개방 이벤트 검사, 수동 제어 충돌 검사, 장애물 실제 크기와 차체 footprint의 샘플링 중첩 검사, 상태별 원본 카메라 수집을 추가했습니다.
- 실행별 소스 해시/설정/스크립트와 결과를 새로운 `artifacts/validation/resumed_*` 및 `feedback_resume_*` 경로에 보존합니다. 최종 결과는 실행 완료 후 갱신합니다.

## 이전 상태 (2026-09-13, 당시 코드 기준)

- `colcon build --symlink-install` PASS
- Python 문법 검사 PASS
- Gazebo headless strict full validation World 1~5 연속 PASS (2026-09-07)
  - 모든 월드 `MISSION_COMPLETE`, 파란문 16/16, rejected door match 0건
  - World 5 + CPU worker 2개 no-GUI 부하 조건에서도 파란문 6/6 PASS
- Handle v3 YOLO + 실제 PIPER 형상 단일 문 물리 접촉 검증 PASS (2026-09-13)
  - 강화된 완전 개방 기준 재검증: `yolo:primary:item` 관측 46회, 선택 손잡이 좌표 `base_link=(0.712, 0.335, 0.752)m`
  - 레버 최대 회전 `0.287rad (16.4°)`, 문 최대/최종 회전 `2.059rad (118.0°)`
  - 차체 변위 `1.630m`, 서비스 성공, door hinge 직접 명령 0회
- 자율주행 성공 기준 코드는 보존됨
  - 보존 브랜치: `codex/navigation-success-before-arm`
  - 보존 태그: `navigation-success-before-arm-20260826`
  - 보존 커밋: `41ec296 feat: finalize observation-based navigation validation`
- 현재 작업본은 semantic map 기반 문 후보 누적, nearest unopened blue target lock, 문앞 정면 정렬, 레버 누름 + push-open, opened/failed station 메모리, 모든 파란문 처리 후 초록 비상구 통과 구조입니다.
- World 1~5 주행 검증과 실제 PIPER 물리 접촉 검증은 각각 완료했지만, 실제 PIPER 접촉 동작을 5개 월드 전체 미션에 결합한 검증은 아직 수행하지 않았습니다.

### 인식 모델

- YOLOv8s Door 1-class 모델 적용
  - 모델: `src/fire_robot_perception/models/best.pt`
  - 학습 성능: mAP50 `0.6088`, mAP50-95 `0.3923`, Precision `0.6184`, Recall `0.5817`
  - 색상 분류는 YOLO가 아니라 HSV 로직에서 `red/blue/green`으로 별도 처리
- 손잡이 검출은 주·보조 모델 체인으로 구성
  - 시뮬레이션 주 모델: `handle_best_v3_CANDIDATE.pt` (Gazebo 평가 99%)
  - 보조 모델 및 실제 로봇 기본값: `handle_best_v2.pt`
  - v3는 팀원 분리 평가에서 실제 손잡이 약 41%로 실제 기준 72%를 통과하지 못했으므로 실제 로봇 기본값으로 승격하지 않음
  - 두 YOLO 모델이 미검출일 때만 HSV 손잡이와 문 기하 추정 fallback 사용
  - `require_yolo_handle:=true`이면 fallback 성공을 허용하지 않고 YOLO 관측을 필수로 검사

## 문 개방 시뮬레이션 (2026-09-13)

- 파란문 Gazebo 모델을 visual-only marker에서 `static=false` 힌지 문으로 변경했습니다.
- 전용 접촉 월드는 측면 힌지, 회전 레버, 래치 볼트/스트라이크, 무게 18kg 패널 collision을 포함합니다.
- `fire_robot_actual_piper.urdf.xacro`는 사용자 제공 PIPER/J100 자료의 메쉬, 관절축, 질량, 관성, 관절 제한을 사용합니다.
- `physical_contact_manipulation_node`는 YOLO 손잡이 관측을 `base_link`로 변환하고 실제 PIPER 체인의 수치 IK로 pre-grasp, grasp, lever-press를 수행합니다.
- 래치 해제 뒤 팔을 회수하고 차체가 문 패널을 밀며 힌지 궤적을 따라갑니다. 성공은 경과 시간이 아니라 측정된 레버/문 joint state로 판정합니다.
- 완전 개방 판정은 문 회전 `2.05rad (117.5°)` 이상을 요구하며, 단순히 문이 조금 열린 상태는 PASS로 처리하지 않습니다.
- pre-grasp는 위치 제어 잔차 `0.075rad` 이내를 허용합니다. grasp는 자유공간에서 `0.035rad` 이내이거나, 접촉 하중 잔차 `0.11rad` 이내이면서 grasp 시작 뒤 레버가 실제로 회전해야 통과합니다.
- 팔이 카메라/LiDAR를 가린 재관측값은 이전 관측에서 `0.04m` 넘게 이동하면 폐기하며, grasp 성공은 grasp 시작 뒤 새로 측정된 레버 회전을 요구합니다.
- 엄격 접촉 모드에서는 `/fire_robot/door/.../cmd`를 발행하지 않습니다. 즉, 검증 중 door hinge를 직접 회전시키지 않습니다.
- `door_detection_node`는 문 bbox 내부/주변에서 손잡이 YOLO 모델을 먼저 실행하고, 실패하면 노란/금색 손잡이 blob, 이후 기존 문 위치 기반 추정값으로 fallback합니다.
- 전용 검증 스크립트는 외부 카메라 영상, YOLO debug bbox, 단계별 프레임, 레버/문 각도와 JSON 결과를 같은 실행에서 저장합니다.

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
- Primary controller: `dwb_core::DWBLocalPlanner`
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

### Handle v3 + 실제 PIPER 물리 접촉 검증 (2026-09-13)

| 항목 | 결과 |
| --- | --- |
| Handle v3 직접 YOLO 관측 | PASS, `yolo:primary:item`, 49회 |
| 실제 PIPER 메쉬/관절 기반 IK | PASS |
| 레버 물리 누름 | PASS, 최대 `26.4°` |
| 차체 접촉 문 밀기 | PASS, 변위 `0.359m` |
| 문 완전 개방 | PASS, 최종 `128.9°` |
| door hinge 직접 명령 | 사용 안 함 |

상세 결과와 증거 이미지는 [`docs/VALIDATION_20260913.md`](docs/VALIDATION_20260913.md)에 정리했습니다.

### 최종 엄격 전체 검증 (2026-09-07)

| 월드 | 조건 | 결과 |
| --- | --- | --- |
| World 1 `obstacle_wall_doors_v5.world` | 파란문 3개, 빨간문/장애물 혼합 | PASS, 파란문 3/3, `MISSION_COMPLETE` |
| World 2 `obstacle_door_layout_alt_v1.world` | 다른 문 배열/장애물 배치 | PASS, 파란문 3/3, `MISSION_COMPLETE` |
| World 3 `obstacle_door_layout_world3_v1.world` | 파란문 4개, 빨간문 2개 | PASS, 파란문 4/4, `MISSION_COMPLETE` |
| World 4 `obstacle_no_blue_world4_v1.world` | 파란문 없음 | PASS, 문 개방 0/0, no-blue fallback 후 `MISSION_COMPLETE` |
| World 5 `obstacle_all_blue_world5_v1.world` | 좌우 3개씩 모든 문 파란색 | PASS, 파란문 6/6, `MISSION_COMPLETE` |

증빙: `artifacts/validation/final_full_20260907_r6` 및
`C:\Users\황준영\Documents\졸업작품\미팅자료_20260907`

### 저성능 스트레스 검증 (2026-09-06)

| 월드 | 조건 | 결과 |
| --- | --- | --- |
| World 5 + CPU worker 2개 | 파란문 6개, Gazebo/YOLO/Nav2 동시 실행 | PASS, 파란문 6/6, rejected 0, `MISSION_COMPLETE` |

로그:

- `artifacts/validation/final_cpu_stress_20260906`

주의:

- CPU worker 6개 조건은 전체 load average가 logical CPU 수를 넘어 Nav2 action timeout이 누적되어 3/6에서 중단했습니다. 이 한계 기록은 `artifacts/validation/overload_boundary_6workers_20260906`에 보존했습니다.
- 2026-09-07 추가 2-worker 관찰 실행은 2/6 진행 중 목표 재계획 지연을 확인한 뒤 중단했습니다. 사용자 요청에 따라 이 부하 관찰만을 이유로 주행 정책은 추가 수정하지 않았고 참고 로그로 보존했습니다.
- 실제 로봇은 Gazebo 물리 엔진을 같이 실행하지 않으므로 이 과부하 조건과 동일하지 않습니다.
- 2026-09-07 World 1~5 결과는 자율주행/FSM 기준입니다. 당시 문 개방에는 fallback과 Gazebo 문 연동이 포함됐으므로 실제 PIPER 접촉 증거로 해석하지 않습니다.
- Handle v3 직접 검출과 실제 PIPER 접촉 문 개방은 2026-09-13 전용 단일 문 시험에서 별도로 확인했습니다.

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
bash scripts/run_full_world_validation_set.sh artifacts/validation/full_worlds_latest
```

CPU contention validation:

```bash
CPU_WORKERS=2 bash scripts/run_cpu_stress_validation.sh \
  artifacts/validation/cpu_stress_latest
```

Handle v3 + 실제 PIPER 형상 물리 접촉 검증:

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/fire_robot_bringup/scripts/run_physical_contact_door_test.py \
  --headless \
  --use-yolo-observation \
  --minimum-yolo-confidence 0.20 \
  --service-timeout-sec 240 \
  --output-root artifacts/validation/physical_contact_latest
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
- Handle v2/v3를 실제 레버 데이터로 추가 학습하고 분리 validation에서 실제 기준 72% 이상 확보
- 실제 PIPER에서 레버 누름, 차체 밀기, 비상 정지를 저속 단계별로 확인
- 실제 PIPER 접촉 동작과 World 1~5 전체 FSM을 결합한 통합 검증

## 주의

Gazebo/ROS 프로세스가 남아 있으면 `/clock`, TF, Nav2 action result가 꼬여 가짜 실패가 생길 수 있습니다. 검증 전에는 WSL을 한 번 종료하고 새로 시작하는 것을 권장합니다.
