# 화재 대피 모바일 매니퓰레이터

> 화재 발생 시 대피로 확보를 위한 모바일 매니퓰레이터 주행 및 제어 알고리즘 설계 (졸업작품)

## 동작 흐름

```
카메라로 문 색상 인식
→ 빨간 문(화재) 감지 → 탐색 시작
→ 파란 문 발견 → Nav2 자율주행으로 이동
→ 로봇팔로 문 손잡이 잡고 개방
→ 다음 파란 문 탐색 (반복)
→ 모든 파란 문 개방 완료 → 비상구로 탈출
```

### FSM 상태 흐름

```
IDLE
 └─ 화재(빨간 문) 감지
EXPLORING  ◄──────────────────────────────┐
 ├─ 파란 문 발견 → NAVIGATING             │
 │    └─ 도착 → OPENING_DOOR             │
 │         └─ 개방 성공 → DOOR_OPENED ───┘
 └─ 30 s 동안 새 파란 문 없음
EXITING  (비상구 좌표로 이동)
 └─ 도착
MISSION_COMPLETE
```

**플랫폼:** ROS2 Humble · Gazebo Ignition · Nav2 · MoveIt2  
**센서:** Intel RealSense D435 (RGB + Depth) · 2D Radar  
**로봇팔:** PIPER 6DoF

---

## 역할 분담

| 파트 | 담당 | 패키지 |
|------|------|--------|
| 자율주행 | 나 | `fire_robot_perception` `fire_robot_navigation` `fire_robot_bringup` |
| 로봇 제어 | 팀원 | `fire_robot_manipulation` |
| 공통 | 공동 | `fire_robot_interfaces` `fire_robot_description` `fire_robot_fsm` |

---

## 패키지 구조

```
fire_robot_ws/src/
├── fire_robot_interfaces     # 공용 메시지/서비스 (수정 불필요)
├── fire_robot_description    # URDF 로봇 모델 (수정 불필요)
├── fire_robot_fsm            # 전체 상태 머신 (수정 불필요)
│
├── [자율주행 - 나]
│   ├── fire_robot_perception     # 문 감지, 센서 융합
│   ├── fire_robot_navigation     # Nav2 자율주행
│   └── fire_robot_bringup        # 시뮬레이션/실제로봇 런치파일
│
└── [로봇제어 - 팀원]
    └── fire_robot_manipulation   # MoveIt2 로봇팔 제어, 문 개방
```

---

## 구현 현황

### 자율주행 파트 (나)

| 항목 | 상태 |
|------|------|
| YOLOv8 + HSV 문 탐지 노드 | ✅ 완성 |
| 문 위치 TF 변환 (`base_link` → `map` 프레임) | ✅ 완성 |
| SegFormer + Radar 센서 융합 노드 | ✅ 완성 |
| Depth 카메라 융합 (선택적) | ✅ 완성 |
| Nav2 자율주행 노드 | ✅ 완성 |
| FSM 상태 머신 (전체 파란 문 순차 개방 + 비상구 탈출) | ✅ 완성 |
| 데이터셋 준비 (OpenImages v7 Door ~5000장) | ✅ 완성 |
| 학습 스크립트 (`train_door_detector.py`) | ✅ 완성 |
| **YOLOv8 학습 모델 (best.pt)** | ⏳ 팀원 GPU 학습 예정 |
| 시뮬레이션 통합 테스트 | ⏳ 미완 |

### 자율주행 남은 작업

**1. YOLOv8 모델 학습** (GPU 있는 환경에서)
```bash
pip install ultralytics
cd fire_robot_ws/src/fire_robot_perception
python3 scripts/train_door_detector.py \
  --dataset datasets/door_detection/dataset.yaml
# → runs/detect/door_detector/weights/best.pt 생성
```
학습 후 노드 실행 시 `--ros-args -p model_path:=<best.pt 경로>` 추가.  
현재는 best.pt 없어서 HSV contour 탐지로만 동작 중.

**2. 시뮬레이션 통합 테스트** (WSL2에서)
```bash
ros2 launch fire_robot_bringup simulation.launch.py
```
빨간 문 감지 → 모든 파란 문 순차 개방 → 비상구 탈출 전체 FSM 흐름 검증.

**3. 실측 파라미터 조정** (실제 하드웨어 시)

| 항목 | 파일 | 기본값 |
|------|------|--------|
| Radar 드라이버 패키지 | `real_robot.launch.py` | `ldlidar_stl_ros2` (LD19) → 실제 모델에 맞게 |
| Radar 포트 | `real_robot.launch.py` | `/dev/ttyUSB0` |
| Nav2 최대 속도 | `nav2_params.yaml` | `0.5 m/s` |
| HSV 색상 임계값 | `door_detection_node.py` 상단 | 현장 조명 보고 조정 |
| 카메라–Radar 오프셋 | `sensor_fusion_node` 파라미터 | `radar_to_cam_x: 0.0` |
| 탐색 타임아웃 | FSM `explore_timeout_sec` 파라미터 | `30.0 s` |
| 비상구 좌표 | FSM `exit_x` / `exit_y` / `exit_yaw` 파라미터 | `10.0 / 0.0 / 0.0` |

---

### 로봇 제어 파트 (팀원)

| 항목 | 상태 |
|------|------|
| `manipulation_node.py` — 문 개방 4단계 시퀀스 | ✅ 골격 완성 |
| `fire_robot.srdf` — MoveIt2 플래닝 그룹 설정 | ✅ 완성 |
| `move_group.launch.py` | ✅ 완성 |
| 실제 PIPER 팔 동작 테스트 | ❌ 미완 |

실측 후 수정 필요한 파라미터:
```python
# manipulation_node.py 상단
PRE_GRASP_OFFSET = 0.12   # 손잡이 앞 멈춤 거리 (m)
PULL_DISTANCE    = 0.35   # 문 당기는 거리 (m)
GRIPPER_OPEN     = 0.08   # 그리퍼 열림 폭 (m)
GRIPPER_CLOSE    = 0.01   # 그리퍼 닫힘 폭 (m)
```
```yaml
# move_group.launch.py
velocity_scaling: 0.3   # 처음엔 0.1~0.2로 낮게 시작 권장
```

---

## 파트 간 인터페이스 (반드시 지킬 것)

FSM이 로봇팔을 제어하는 방식. 아래 규격을 맞춰야 두 파트가 연결된다.

### 서비스 서버 — 팀원이 구현
```
/open_door  [fire_robot_interfaces/srv/OpenDoor]

Request:
  string                    door_id
  geometry_msgs/PointStamped handle_position

Response:
  bool   success
  string message
```

> FSM은 `/open_door` 서비스를 호출하고 **서비스 응답(response)**으로 성공/실패를 판단한다.  
> 서비스 응답만 맞추면 자율주행 파트와 독립적으로 개발 가능하다.

---

## 센서 구성

### Radar (2D)
- 시뮬레이션: Ignition `gpu_lidar` — 240° FOV, 0.5~80m, `/scan` (LaserScan)
- 실제: `ldlidar_stl_ros2` 드라이버 (LD19 기본, `/dev/ttyUSB0` 230400 baud)
  - 다른 레이더 모델 사용 시 `real_robot.launch.py`의 드라이버 패키지 교체 필요

### Depth 카메라 (Intel RealSense D435)
- 시뮬레이션: Ignition `rgbd_camera` — 640×480, 30Hz
- 실제: `realsense2_camera` ROS2 드라이버 (USB 3.0)
- 발행 토픽:
  - `/camera/color/image_raw` — 컬러 이미지
  - `/camera/depth/image_rect_raw` — 뎁스 이미지 (m)
  - `/camera/depth/points` — PointCloud2

Depth 카메라 활성화 (기본 꺼짐):
```bash
ros2 run fire_robot_perception door_detection_node \
  --ros-args -p use_depth_camera:=true
```

---

## 환경 세팅

### 1. Ubuntu 22.04 (또는 WSL2)
```powershell
wsl --install -d Ubuntu-22.04
```

### 2. ROS2 Humble 설치
```bash
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt install -y ros-humble-desktop
```

### 3. 의존성 설치
```bash
sudo apt install -y \
  ros-humble-moveit \
  ros-humble-ros-gz \
  ros-humble-ros-gz-bridge \
  ros-humble-nav2-bringup \
  ros-humble-slam-toolbox \
  ros-humble-xacro \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-realsense2-camera \
  ros-humble-tf2-geometry-msgs \
  python3-colcon-common-extensions \
  python3-rosdep
```

### 4. 클론 & 빌드
```bash
git clone https://github.com/junyoung-123/fire_robot_ws.git
cd fire_robot_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 5. 시뮬레이션 실행
```bash
ros2 launch fire_robot_bringup simulation.launch.py
```

### 6. 로봇팔 단독 테스트
```bash
ros2 launch fire_robot_manipulation move_group.launch.py

ros2 run fire_robot_manipulation manipulation_node \
  --ros-args -p sim_mode:=true
```

---

## Git 운용

```bash
# 수정 후 푸시
git add .
git commit -m "수정 내용"
git push

# 상대방 변경사항 받기
git pull
```

---

## 문의

인터페이스 변경이 필요하면 `fire_robot_interfaces` 패키지 수정 전에 상의 후 진행.
