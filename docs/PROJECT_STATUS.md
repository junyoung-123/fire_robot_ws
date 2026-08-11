# 프로젝트 진행 현황 (2026-08-11)

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
| World 1~5 headless full validation | PASS |
| YOLO Door 모델 | OpenImages Door 기반 YOLOv8s Door 1-class 적용 |
| 색상 분류 | HSV 기반 red/blue/green 분리 유지 |
| 장애물 회피 | Nav2 costmap + SmacPlanner2D + RotationShim/RPP |
| 문 접근 FSM | 파란문 target lock, 문 앞 fine alignment, 열린 문/실패 문 기록 |
| 실제 PIPER 연동 | 아직 미검증 |

## 최근 반영된 핵심 변경

- `simulation.launch.py`
  - 월드 선택 LaunchArgument 유지: `world:=...`
  - `start_without_fire` LaunchArgument 노출
  - 3카메라와 YOLO Door 모델 파라미터 연결
  - FSM 시작 지연과 perception/Nav2 안정화 파라미터 조정
  - 초록 비상구 후보가 마지막으로 연 파란문보다 충분히 앞에 있을 때만 exit 후보로 인정
- `door_detection_node.py`
  - YOLO Door bbox와 HSV 색상 분리 결합
  - 측면 카메라 edge-clipped bbox 오탐 억제
  - 빨간문/초록문 후보가 파란문으로 번지는 상황을 줄이기 위한 기본 필터 보강
- `sensor_fusion_node.py`
  - SegFormer 미사용 시 LiDAR 기반 observation map 발행
  - 초기 로봇 pose 기준으로 segmentation map origin 고정
- `state_machine_node.py`
  - 파란문 발견 시 해당 후보를 lock하고 열기 전까지 다른 후보 끼어들기 억제
  - 열린 문, 실패 문, abandoned 문을 map 좌표 기준으로 관리
  - 출구 근처에서 빨간문/초록문 색 번짐이 observed_blue로 되살아나는 문제 억제
  - 파란문 후보가 없을 때만 최종 측면 스캔 후 EXITING 전환
  - 비상구 통과 조건을 단순 근접이 아니라 진행축 통과 기준으로 보강
- `fsm_subsystems.py`
  - 문 접근/정렬/관측 증거 검사를 보조 FSM으로 분리해 디버깅 가능성 개선
- `navigation_node.py`
  - Nav2 action server 초기화 전 목표 큐잉
  - stale result 처리 보강
  - 목표 좌표/프레임 로그와 map 변환 안정화
- `nav2_params.yaml`
  - Planner를 NavFn에서 SmacPlanner2D로 변경
  - RotationShim + RegulatedPurePursuit 조합 적용
  - global/local costmap 및 obstacle layer 파라미터 조정
  - no-backup replanning BT XML 포함
- `initial_static_map_node.py` / `fixed_obstacle_map_node.py` / `mission_axis_node.py`
  - 초기 static map과 mission axis 기반으로 진행축을 잡고 localization 주행에 활용
  - 기존 구조를 무분별하게 덮지 않고 미스캔 영역을 관측으로 보강하는 방향 유지

## 최종 검증 요약

최종 검증 태그:

```text
full_worlds_12345_exit_tail_red_guard_20260811
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
- `navigation_failed_logs=0`
- `fresh_blue_fail_logs=0`
- 빨간문 개방 없음
- 월드별 궤적/문 앞 정렬 이미지 생성 완료

검증 증빙:

- `docs/validation/2026-08-11/summary.txt`
- `docs/validation/2026-08-11/world*_trajectory.png`
- `docs/validation/2026-08-11/world*_alignment.png`

## 실제 로봇 전 리스크

- PIPER driver가 MoveIt의 `/piper_arm_controller/follow_joint_trajectory` 액션 서버를 실제로 제공하는지 확인 필요
- 실제 베이스 footprint, wheel odom, TF가 시뮬레이션과 다를 수 있음
- 실제 카메라 높이/각도와 2D LiDAR 높이에 따라 문 bbox 거리 추정과 costmap 튜닝 필요
- OpenImages Door 모델은 문 일반화용이라 연구실 문/색상판/조명 조건에서 추가 데이터가 필요할 수 있음

## 다음 작업

1. 실제 PIPER `can0` 연결 및 MoveIt action server 확인
2. 실제 모바일 베이스 `/odom -> base_link` TF 확인
3. 실제 카메라/2D LiDAR TF 확인
4. 연구실 조명 기준 HSV 튜닝
5. 실제 복도 폭/장애물 기준 Nav2 costmap 튜닝
