# 화재 대피 로봇 - 로봇팔 문 개방 시뮬레이션 작업 보고

작성일: 2026-08-15
최신 갱신: 2026-08-24
담당 범위: PIPER 로봇팔 문 개방 시뮬레이션 및 검증

## 1. 기존 상태

팀원 작업에서는 YOLO/HSV/LiDAR 기반 문 탐지, Nav2 문 앞 접근, fine alignment,
FSM 연동까지 검증되었다. 기존 `/open_door` 시뮬레이션은 단계별 로그와
대기 시간만 사용했으며, Gazebo 로봇팔이나 문 관절은 실제로 움직이지 않았다.

## 2. 구현 내용

- 기존 자율주행 성공 브랜치 `codex/pre-arm-navigation-success`를 보존한 상태에서 현재 `master`에 로봇팔 통합 작업 반영
- PIPER 팔 관절과 좌우 그리퍼에 Gazebo 위치 제어 인터페이스 추가
- 충돌 형상, 손잡이, 회전 힌지를 가진 파란문 전용 데모 월드 제작
- `/open_door` 호출 시 pre-grasp, grasp, lever-press, push-open 순서로 Gazebo 관절 구동
- 문 힌지 상태를 ROS2로 피드백하여 목표 각도 도달 시에만 성공 반환
- 서비스와 관절 피드백을 동시에 처리하도록 MultiThreadedExecutor 적용
- 조작 전용 launch, RViz2 화면, headless 자동 검증 스크립트 추가
- 반복 실행 후 Gazebo 자식 프로세스가 남지 않도록 자동 정리 구현

## 3. 검증 결과

| 검증 항목 | 결과 |
|---|---|
| 관련 ROS2 패키지 빌드 | PASS |
| `/open_door` 서비스 | `success=True` |
| 문 힌지 목표 | +2.09 rad |
| 문 힌지 실측 | 약 +2.09 rad (약 120도) |
| `joint1` 실측 | 약 0.480 rad |
| `joint2` 실측 | 약 0.726 rad |
| `joint3` 실측 | 약 -0.576 rad |
| 좌우 그리퍼 | 약 0.006 m |
| 문 관절 피드백 판정 | PASS |
| 자동 통합 검증 | `MANIPULATION_DEMO_PASS` |
| 검증 후 잔여 Gazebo 프로세스 | 0개 |

## 4. RViz2 확인 기능

- PIPER 전체 RobotModel
- 로봇팔 링크 및 관절 TF 트리
- `arm_link6` 말단 좌표축
- `gripper_base_link` 좌표축
- `/joint_states` 기반 실시간 자세 갱신

문 모델은 SDF이므로 문 자체의 회전은 Gazebo에서 확인하고, RViz2에서는
로봇팔 링크와 TF 상태를 자세히 확인한다.

## 5. 실행 방법

```bash
cd ~/fire_robot_ws_team
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch fire_robot_bringup manipulation_demo.launch.py \
  headless:=true use_rviz:=true
```

자동 검증:

```bash
bash scripts/validate_manipulation_demo.sh
```

## 6. 구현 범위와 남은 작업

현재 결과는 로봇팔 궤적과 문 힌지 명령을 연동하고 실제 관절 피드백으로
성공을 판정하는 통합 시뮬레이션이다. 그리퍼와 손잡이의 접촉 마찰만으로
문이 열리는 순수 접촉 동역학, 실제 PIPER 치수 기반 정밀 IK, MoveIt 충돌
회피 궤적, 실제 장비 CAN 연동은 후속 작업으로 남아 있다.

현재 프로젝트 반영본은 실제 시연 안정성을 우선해 lever press 후
push-open 방향을 기본 시뮬레이션/실제 로봇 정책으로 사용한다. 팀원
로봇팔 데모의 힌지 명령 검증 방식은 `manipulation_demo.launch.py`에
보존되어 있으며, 순수 접촉 검증은 `physical_contact_door_test.launch.py`와
`run_physical_contact_door_test.py`에서 수행한다.

2026-08-24 기준 World 1~5 full validation은 모두 PASS다. 최신 증거는
`C:\Users\황준영\Documents\졸업작품\검증결과_20260823`와
`C:\Users\황준영\Documents\졸업작품\team_share\fire_robot_team_share_20260824.zip`에 정리했다.

## 7. 로봇팔 전용 변경 파일

- `src/fire_robot_manipulation/fire_robot_manipulation/manipulation_node.py`
- `src/fire_robot_manipulation/package.xml`
- `src/fire_robot_description/urdf/fire_robot.urdf.xacro`
- `src/fire_robot_bringup/worlds/manipulation_demo.world`
- `src/fire_robot_bringup/launch/manipulation_demo.launch.py`
- `src/fire_robot_bringup/rviz/manipulation_demo.rviz`
- `scripts/validate_manipulation_demo.sh`
- `docs/MANIPULATION_VALIDATION.md`

현재 변경분은 주행/FSM, perception, manipulation, 검증 월드까지 통합한
상태로 GitHub `master`에 올리는 것을 기준으로 정리했다.
