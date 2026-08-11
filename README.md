# Fire Robot Workspace

화재/문 탐지 로봇 시뮬레이션 및 AgileX PIPER 연동 준비용 ROS2 Humble 워크스페이스입니다.

목표 동작은 시작 위치 기준으로 관측한 벽, 문, 장애물 구조를 map 좌표에 축적하고, 가장 가까운 파란문을 선택해 장애물을 피해 접근한 뒤 문 개방 FSM을 수행하는 것입니다. 더 이상 열 파란문이 없으면 초록 비상구를 관측 기반으로 선택해 통과합니다.

## 현재 상태 (2026-08-11)

- `colcon build --symlink-install` PASS
- Python 문법 검사 PASS
- Gazebo headless full validation PASS
- World 1~5 전체 `MISSION_COMPLETE`
- 최종 검증 증빙: `docs/validation/2026-08-11/`
- YOLOv8s Door 1-class 모델 적용
  - 모델: `src/fire_robot_perception/models/best.pt`
  - 학습 성능: mAP50 `0.6088`, mAP50-95 `0.3923`, Precision `0.6184`, Recall `0.5817`
  - 색상 분류는 YOLO가 아니라 HSV 로직에서 `red/blue/green`으로 별도 처리

## 주요 구조

### Perception

- 카메라: `front`, `front_left`, `front_right`
- 문 검출: YOLO Door bbox + HSV 색상 분류
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

## 최종 검증 결과 (2026-08-11)

| 월드 | 조건 | 결과 |
| --- | --- | --- |
| World 1 | 파란문 3개, 빨간문/장애물 혼합 | PASS, 파란문 3/3, false open 없음 |
| World 2 | 다른 문 배열/장애물 배치 | PASS, 파란문 3/3, false open 없음 |
| World 3 | 파란문 4개, 빨간문 2개 | PASS, 파란문 4/4, false open 없음 |
| World 4 | 파란문 없음 | PASS, 문 개방 없이 비상구 이동 |
| World 5 | 좌우 3개씩 모든 문 파란색 | PASS, 파란문 6/6, false open 없음 |

전체 요약은 [docs/validation/2026-08-11/summary.txt](docs/validation/2026-08-11/summary.txt)에 있습니다.

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
bash /mnt/c/Users/황준영/Documents/졸업작품/run_full_worlds_20260802.sh \
  full_worlds_12345_exit_tail_red_guard_20260811
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

## 주의

Gazebo/ROS 프로세스가 남아 있으면 `/clock`, TF, Nav2 action result가 꼬여 가짜 실패가 생길 수 있습니다. 검증 전에는 WSL을 한 번 종료하고 새로 시작하는 것을 권장합니다.
