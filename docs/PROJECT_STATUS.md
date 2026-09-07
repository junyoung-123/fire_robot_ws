# 프로젝트 진행 현황 (2026-09-07)

## 목표 동작

1. 시작 위치 기준으로 보이는 벽, 문, 장애물을 관측한다.
2. 관측 결과를 map 좌표로 축적해 고정 구조/관측 메모리를 만든다.
3. 가장 가까운 파란문을 미션 대상으로 lock한다.
4. 목표 파란문 앞까지 Nav2가 장애물을 피해 주행한다.
5. 문 앞에서 정면 정렬 후 PIPER 문 개방 FSM을 실행한다.
6. 열린 문은 map 좌표로 기록해 다시 접근하지 않는다.
7. 남은 파란문을 찾고 같은 과정을 반복한다.
8. 더 이상 열 파란문이 없을 때 초록 비상구를 통과한다.

## 현재 구현 상태

| 영역 | 상태 |
| --- | --- |
| 시뮬레이션 빌드 | PASS |
| Python 문법 검사 | PASS |
| World 1~5 headless strict full validation | 연속 PASS, 파란문 16/16, rejected match 0 |
| no-GUI CPU 부하 World 5 | PASS, 파란문 6/6, `MISSION_COMPLETE` |
| YOLO Door 모델 | OpenImages Door 기반 YOLOv8s Door 1-class 적용 |
| 색상 분류 | HSV 기반 red/blue/green 분리 유지 |
| 장애물 회피 | Nav2 costmap + SmacPlanner2D + RotationShim/DWB |
| 문 접근 FSM | 파란문 target lock, 문 앞 fine alignment, 열린 문/실패 문 기록 |
| 손잡이 인식 | YOLO handle 모델 정상 로드, 현재 Gazebo 직접 검출 0건; HSV/벽면 투영 fallback 사용 |
| 문 개방 시뮬레이션 | 파란문 힌지 joint + 손잡이 collision + Gazebo joint topic 개방 |
| 로봇팔 통합 재검증 | World 1~5 전체 FSM PASS, lever press + push-open 48도 순수 접촉 PASS, attach-assisted 120도 PASS |
| 실제 PIPER 연동 | 아직 미검증 |

## 최근 반영된 핵심 변경

- `simulation.launch.py`
  - 월드 선택 LaunchArgument 유지: `world:=...`
  - `start_without_fire` LaunchArgument 노출
  - 3카메라와 YOLO Door 모델 파라미터 연결
  - FSM 시작 지연과 perception/Nav2 안정화 파라미터 조정
  - 초록 비상구 후보가 마지막으로 연 파란문보다 충분히 앞에 있을 때만 exit 후보로 인정
  - Gazebo 힌지 문 개방 시뮬레이션 파라미터 연결
- `door_detection_node.py`
  - YOLO Door bbox와 HSV 색상 분리 결합
  - 손잡이 전용 YOLO `handle_model_path` 연결. Door bbox ROI 내부에서 handle/door_handle/knob/lever 클래스를 우선 검출
  - 손잡이 YOLO 결과를 `DoorInfo.handle_position`, `handle_detected`, `handle_detection_method`, `handle_confidence`로 발행
  - 손잡이 YOLO 모델이 없거나 미검출이면 HSV handle blob, 이후 추정 좌표로 fallback
  - 측면 카메라 edge-clipped bbox 오탐 억제
  - 빨간문/초록문 후보가 파란문으로 번지는 상황을 줄이기 위한 기본 필터 보강
- `sensor_fusion_node.py`
  - SegFormer 미사용 시 LiDAR 기반 observation map 발행
  - 초기 로봇 pose 기준으로 segmentation map origin 고정
- `state_machine_node.py`
  - 파란문 발견 시 해당 후보를 lock하고 열기 전까지 다른 후보 끼어들기 억제
  - 열린 문, 실패 문, abandoned 문을 map 좌표 기준으로 관리
  - 출구 근처에서 빨간문/초록문 색 번짐이 observed_blue로 되살아나는 문제 억제
  - 이미 잠긴 stable observed-blue 후보는 사전 스캔 후에도 실제 문 앞에서 한 번 검증하도록 억제 조건을 완화
  - observed_blue 목표는 Nav2 접근 pose가 아니라 handle/wall evidence 기준으로 색상 충돌을 재판정하도록 수정
  - 출구 직전 live 파란문 evidence가 있으면 EXITING보다 문 개방을 우선하도록 보강
  - no-blue 검증 월드에서 초반 가짜 출구 후보가 생기지 않도록 fallback 최소 진행축을 18m로 지연
  - 실제 green 출구 후보가 있으면 no-blue fallback보다 먼저 확인하도록 exit scan 순서 조정
  - 파란문 후보가 없을 때만 최종 측면 스캔 후 EXITING 전환
  - 비상구 통과 조건을 단순 근접이 아니라 진행축 통과 기준으로 보강
- `manipulation_node.py`
  - sim 모드에서 단순 sleep-success 대신 Gazebo 힌지 문 topic으로 열림 각도 명령
  - `/open_door` 요청의 손잡이 검출 방식/신뢰도를 로그로 남기고, 실제 로봇 모드에서 직접 검출 손잡이를 요구할 수 있는 안전 파라미터 추가
  - 현재 world 파일에서 파란 힌지문 topic registry를 읽어 관측 손잡이 좌표와 가장 가까운 문 선택
  - 문 개방 내부 단계를 `/manipulation_phase` sub-FSM으로 발행
  - sim 전용 PIPER arm pre-grasp/grasp/lever-press/push/pull/home joint pose 명령 추가
  - 레버 파지 유지 시뮬레이션용 DetachableJoint attach topic과 팔/베이스 동시 push-follow 궤적 추가
  - 로봇팔 단독 데모에서는 ROS-Gazebo command bridge와 `/door_joint_states` 피드백을 사용해 문 힌지 목표각 도달을 검증
- `manipulation_demo.launch.py` / `validate_manipulation_demo.sh`
  - 팀원 로봇팔 데모 아이디어를 보존하고, 실제 시연 기본 정책은 push-open으로 전환
  - headless 자동 검증에서 `/open_door` 응답과 실제 door_hinge 피드백을 함께 확인
- `door_detection_node.py`
  - 문 bbox ROI에서 손잡이 YOLO를 먼저 실행해 `handle_position` 보정
  - 손잡이 YOLO가 없거나 실패하면 노란/금색 blob, 이후 기존 문 위치 기반 추정 좌표 사용
- `fire_robot.urdf.xacro`
  - Gazebo sim 전용 PIPER joint position controller 추가
- `worlds/*.world`
  - 월드 1/2/3/5와 corridor의 파란문을 collision/handle/hinge joint가 있는 물리 문으로 변경
- `fsm_subsystems.py`
  - 문 접근/정렬/관측 증거 검사를 보조 FSM으로 분리해 디버깅 가능성 개선
- `navigation_node.py`
  - Nav2 action server 초기화 전 목표 큐잉
  - stale result 처리 보강
  - 목표 좌표/프레임 로그와 map 변환 안정화
- `nav2_params.yaml`
  - Planner를 NavFn에서 SmacPlanner2D로 변경
  - RotationShim + DWBLocalPlanner 조합 적용
  - global/local costmap 및 obstacle layer 파라미터 조정
  - no-backup replanning BT XML 포함
- `initial_static_map_node.py` / `fixed_obstacle_map_node.py` / `mission_axis_node.py`
  - 초기 static map과 mission axis 기반으로 진행축을 잡고 localization 주행에 활용
  - 기존 구조를 무분별하게 덮지 않고 미스캔 영역을 관측으로 보강하는 방향 유지

## 최종 검증 요약

최종 검증 디렉터리:

```text
artifacts/validation/final_full_20260907_r6
```

| 월드 | 기대 파란문 | 개방/매칭 | false open | 결과 |
| --- | ---: | ---: | --- | --- |
| World 1 | 3 | 3/3 | 없음 | PASS |
| World 2 | 3 | 3/3 | 없음 | PASS |
| World 3 | 4 | 4/4 | 없음 | PASS |
| World 4 | 0 | 0/0 | 없음 | PASS |
| World 5 | 6 | 6/6 | 없음 | PASS |

공통 확인:

- 모든 월드에서 `MISSION_COMPLETE=True`
- 총 파란문 16/16 개방, 고유 Gazebo door topic 16/16, rejected match 0
- 일시적인 Nav2 실패가 발생한 실행도 FSM 복구 후 최종 미션 완료
- 월드별 최대 roll `0.021°`, 최대 pitch `0.732°`로 검증 제한 `20°` 이내
- 빨간문 개방 없음
- 월드별 궤적/문 앞 정렬 이미지 생성 완료

검증 증빙:

- `C:\Users\황준영\Documents\졸업작품\미팅자료_20260907\월드1-5_장애물크기_주행궤적.png`
- `C:\Users\황준영\Documents\졸업작품\미팅자료_20260907\월드1-5_전체검증.png`
- `C:\Users\황준영\Documents\졸업작품\최종검증_20260906\manipulation_visual_proof\gazebo_door_open_before_after.png`
- `C:\Users\황준영\Documents\졸업작품\미팅자료_20260907\화재탐지로봇_시뮬레이션_검증결과_20260907.pdf`

## 로봇팔 통합 재검증 요약

검증 일자: `2026-08-24`

| 월드 | 기대 파란문 | 개방 성공 | 출구 판단 | 결과 |
| --- | ---: | ---: | --- | --- |
| World 1 `corridor.world` | 3 | 3 | observed green | PASS |
| World 2 `obstacle_door_layout_alt_v1.world` | 3 | 3 | observed green | PASS |
| World 3 `obstacle_door_layout_world3_v1.world` | 4 | 4 | observed green | PASS |
| World 4 `obstacle_no_blue_world4_v1.world` | 0 | 0 | delayed no-blue fallback | PASS |
| World 5 `obstacle_all_blue_world5_v1.world` | 6 | 6 | observed green | PASS |

확인 내용:

- 파란문이 있는 월드는 모든 기대 파란문이 Gazebo 힌지 joint topic으로 개방됨
- World 1/2/3/5는 실제 green 비상구 관측 기반으로 pass-through 완료
- World 4는 파란문 없음 케이스에서 충분한 전방 스캔 후 no-blue fallback으로 완료
- 마지막 재검증 기준 `colcon build --symlink-install --packages-select fire_robot_fsm fire_robot_bringup` PASS

검증 증빙:

- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\all_worlds_validation_summary.png`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world1\world1.log`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world2\world2.log`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world3\world3.log`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world4\world4.log`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world5\world5.log`

## 실제 로봇 전 리스크

- PIPER driver가 MoveIt의 `/piper_arm_controller/follow_joint_trajectory` 액션 서버를 실제로 제공하는지 확인 필요
- 실제 베이스 footprint, wheel odom, TF가 시뮬레이션과 다를 수 있음
- 실제 카메라 높이/각도와 2D LiDAR 높이에 따라 문 bbox 거리 추정과 costmap 튜닝 필요
- OpenImages Door 모델은 문 일반화용이라 연구실 문/색상판/조명 조건에서 추가 데이터가 필요할 수 있음
- 현재 `best.pt`는 Door 1-class라 손잡이를 학습하지 않았음. 실제 손잡이 인식은 별도 `handle_best_v2.pt` 학습/적용 필요

## 다음 작업

1. 실제 PIPER `can0` 연결 및 MoveIt action server 확인
2. 실제 모바일 베이스 `/odom -> base_link` TF 확인
3. 실제 카메라/2D LiDAR TF 확인
4. 연구실 조명 기준 HSV 튜닝
5. 실제 복도 폭/장애물 기준 Nav2 costmap 튜닝
6. 실제 연구실 레버 손잡이 이미지로 `handle_best_v2.pt` 추가 학습 및 직접 검출 재검증
7. 실제 로봇 launch에서 `require_yolo_handle:=true`로 YOLO-only 문 개방 검증
