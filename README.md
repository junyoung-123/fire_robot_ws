# Fire Robot Workspace

화재 상황에서 이동 로봇과 AgileX PIPER 매니퓰레이터를 이용해 안전 문을 찾고, 문을 개방한 뒤 비상구로 이동하는 ROS2 Humble 기반 졸업작품 프로젝트입니다.

## 현재 상태

2026-07-02 기준으로 Gazebo 장애물 회피/문 개방/비상구 도착 시뮬레이션 검증은 통과했습니다.

| 항목 | 상태 |
| --- | --- |
| ROS2 Humble 빌드 | 통과 |
| Gazebo 실행 | 통과 |
| Nav2/SLAM 실행 | 통과 |
| FSM 전체 흐름 | 통과 |
| 벽 부착형 좌우 문 + 정면 비상구 월드 | 통과 |
| 파란 문 이동 및 sim_mode 문 개방 | 통과 |
| 초록 비상구 이동 및 `MISSION_COMPLETE` | 통과 |
| 정적 벽 지도 + AMCL localization 주행 | 통과 |
| 2D LiDAR 기반 동적 장애물 회피 | 통과 |
| YOLOv8 학습 모델 `best.pt` | 미완료 |
| 실제 PIPER/CAN/모바일 베이스 검증 | 미완료 |

최신 장시간 검증 증거는 `artifacts/mission_20260702_041116/mission_path.png`, `mission_path.mp4`, `mission_summary.txt`에 저장했습니다.

## 동작 흐름

```text
카메라/라이다 입력
-> 문 탐지 및 색상 분류
-> 빨간 문 감지 시 미션 시작
-> 복도 탐색 중 파란 문 후보 탐지
-> Nav2로 파란 문 앞 이동
-> PIPER sim_mode 문 개방
-> 다시 탐색
-> 초록 비상구 탐지
-> 비상구 이동
-> MISSION_COMPLETE
```

## Navigation Update (2026-07-02)

최신 시뮬레이션 구조는 SLAM을 계속 누적하면서 주행하는 방식이 아니라, 월드 구조를 기준으로 만든 정적 벽 지도와 AMCL localization을 기본으로 사용합니다. 로봇이 움직이는 동안 2D LiDAR와 카메라 인식 결과로 costmap을 갱신하고, Nav2가 반복적으로 경로를 재계획합니다.

핵심 변경:

| 영역 | 내용 |
| --- | --- |
| 모바일 베이스 | Clearpath J100 계열 치수 반영, 4륜 diff-drive joint 연결 |
| 지도 | `obstacle_wall_doors_v5_static.yaml` 정적 벽 지도 추가 |
| localization | 기본 `localization_mode:=localization`, map_server + AMCL 사용 |
| planner | `SmacPlanner2D` 사용 |
| controller | `RotationShimController` + `RegulatedPurePursuitController` 사용 |
| costmap | global은 정적 벽 지도 + LiDAR clearing, local은 LiDAR/segmentation points 사용 |
| 회피 정책 | 가까운 파란문 방향 우선, 먼 파란문은 LiDAR 차선 유지 후 재스캔 |
| Gazebo | 카메라 headless 안정화를 위해 Sensors render engine을 `ogre2`로 변경 |

최신 장시간 검증:

```text
log: logs/headless_verify_20260702_041116.log
result: EXITING -> MISSION_COMPLETE
evidence:
  artifacts/mission_20260702_041116/mission_path.png
  artifacts/mission_20260702_041116/mission_path.mp4
```

검증용 headless 장시간 실행은 CPU 부담을 줄이기 위해 `enable_segformer:=false`로 수행했습니다. 카메라 기반 HSV 문/색상 인식은 켜진 상태였고, 2D LiDAR가 동적 장애물 costmap을 담당했습니다. `simulation.launch.py`의 기본값은 `enable_segformer:=true`라서 SegFormer 연결은 유지됩니다.

문 색상 의미:

| 색상 | 의미 | 동작 |
| --- | --- | --- |
| 빨간색 | 화재/위험 구역 | 미션 시작 트리거 |
| 파란색 | 안전 문 후보 | 문 앞으로 이동 후 개방 |
| 초록색 | 비상구 | 최종 목적지 |

## 패키지 구성

```text
fire_robot_ws/src/
├── fire_robot_bringup        # 시뮬레이션/실제 로봇 launch
├── fire_robot_description    # 로봇 URDF/Xacro, ros2_control 설정
├── fire_robot_fsm            # 전체 임무 FSM
├── fire_robot_interfaces     # 공용 msg/srv
├── fire_robot_manipulation   # MoveIt2/PIPER 문 개방 로직
├── fire_robot_navigation     # Nav2 wrapper, Nav2 설정
└── fire_robot_perception     # 문 탐지, 센서 융합, 데이터셋/학습 스크립트
```

AgileX PIPER 드라이버는 테스트용 WSL 워크스페이스에서 외부 의존 패키지로 추가해 사용했습니다. 원본 프로젝트 패키지 트리에는 포함하지 않습니다.

## 인식 구조

이 프로젝트의 인식은 두 갈래입니다.

```text
YOLOv8 door detector
-> 문 bounding box 검출
-> 세로로 긴 문 형태 필터링
-> bbox 내부 HSV 색상 분류
-> DoorInfo / FireInfo 발행
-> FSM과 Nav2 목표 생성에 사용
```

```text
SegFormer + LaserScan/Depth
-> /segmentation_map OccupancyGrid 생성
-> Nav2 costmap 보조 레이어로 사용
```

현재 시뮬레이션 검증은 YOLO `best.pt` 없이 HSV fallback으로 통과했습니다. 실제 환경에서는 조명과 문 재질이 달라지므로 YOLO 학습 모델 적용이 필요합니다.

## 데이터셋과 학습

문 검출용 OpenImages 기반 데이터셋은 이미 준비되어 있습니다.

```text
src/fire_robot_perception/datasets/door_detection
```

확인된 구성:

```text
train images: 5000
val images: 204
train labels: 5000
val labels: 204
```

팀원 GPU 환경에서 학습:

```bash
cd fire_robot_ws/src/fire_robot_perception
pip install ultralytics

python3 scripts/train_door_detector.py \
  --dataset datasets/door_detection/dataset.yaml
```

학습 후 생성되는 `best.pt`를 `door_detection_node`의 `model_path` 파라미터로 넘겨 사용합니다.

```bash
ros2 run fire_robot_perception door_detection_node \
  --ros-args -p model_path:=/path/to/best.pt
```

문이 아닌 파란/빨간/초록 물체 오탐을 줄이기 위해 bbox 세로/가로 비율 필터가 적용되어 있습니다.

```bash
ros2 run fire_robot_perception door_detection_node \
  --ros-args -p min_door_aspect_ratio:=1.3
```

실제 카메라에서 좌우 문을 너무 많이 놓치면 값을 낮추고, 표지판/가구 오탐이 많으면 값을 높여 조정합니다.

## WSL2 테스트 환경

권장 환경:

```text
Ubuntu 22.04 Jammy
ROS2 Humble
Gazebo Fortress/Ignition
```

Windows의 한글 경로에서 직접 빌드하면 ROSIDL 경로 문제가 생길 수 있으므로, WSL 내부 ASCII 경로를 권장합니다.

```bash
rsync -a \
  --exclude build \
  --exclude install \
  --exclude log \
  --exclude 'src/fire_robot_perception/datasets' \
  '/mnt/c/Users/황준영/Desktop/졸업작품/project/fire_robot_ws/' \
  ~/fire_robot_ws_test/

cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 시뮬레이션 실행

Headless 검증:

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
source install/setup.bash

LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=false \
  headless:=true \
  enable_segformer:=false
```

Gazebo GUI 확인:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=false \
  headless:=false
```

최신 장애물 검증 월드:

```bash
ros2 launch fire_robot_bringup simulation.launch.py \
  world:=obstacle_wall_doors_v5.world \
  localization_mode:=localization \
  use_rviz:=true \
  headless:=false
```

`obstacle_wall_doors_v5.world`는 벽 부착형 빨강/파랑 문, 중앙/측면 장애물, 정면 초록 비상구를 포함합니다.

## 실제 로봇 실행 전 체크리스트

실험실에서 바로 확인할 항목:

```bash
ip link show can0
ls /dev/ttyUSB*
ros2 topic echo /odom
ros2 run tf2_ros tf2_echo odom base_link
ros2 topic echo /camera/color/image_raw
ros2 topic echo /scan
```

PIPER 단독 smoke test:

```bash
ros2 launch fire_robot_bringup real_robot.launch.py \
  enable_camera:=false \
  enable_radar:=false \
  enable_base:=false \
  enable_moveit:=false \
  enable_slam:=false \
  enable_nav2:=false \
  enable_app_nodes:=false \
  enable_piper:=true \
  piper_can_port:=can0
```

로봇 팔은 처음에 낮은 속도부터 시작하세요.

```text
velocity_scaling: 0.1 ~ 0.2 권장
```

## PIPER 문 개방 파라미터

실제 로봇 연결 전 공식 문서와 실측으로 다시 확인해야 합니다.

```python
PRE_GRASP_OFFSET = 0.12
PULL_DISTANCE    = 0.35
GRIPPER_OPEN     = 0.08
GRIPPER_CLOSE    = 0.01
```

관련 파일:

```text
src/fire_robot_manipulation/fire_robot_manipulation/manipulation_node.py
```

## 남은 작업

| 우선순위 | 작업 |
| --- | --- |
| 1 | YOLOv8 `best.pt` 학습 및 적용 |
| 2 | 실제 로봇 `/odom -> base_link` TF 확인 |
| 3 | PIPER `can0` 연결 및 팔 단독 동작 확인 |
| 4 | 실제 카메라/라이다 토픽 확인 |
| 5 | HSV 범위, bbox 비율 필터, 문 접근 거리, 그리퍼 값 현장 튜닝 |
| 6 | 실제 통합 미션 테스트 |

## 알려진 제한

- 실제 로봇 하드웨어는 아직 검증하지 않았습니다.
- 현재 문 인식은 학습 모델 없이도 동작하지만, 실제 환경에서는 YOLO `best.pt` 적용이 필요합니다.
- Gazebo GUI는 WSLg/그래픽 성능에 따라 느릴 수 있습니다.
- MoveIt 로그에 octomap 3D sensor plugin 경고가 뜰 수 있으나, 현재 sim_mode 문 개방 검증에는 영향이 없습니다.
