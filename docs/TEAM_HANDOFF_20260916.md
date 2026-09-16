# 팀원 인계: 2026-09-16

## 어느 버전을 사용할지

- 현재 수정/검증 작업: `codex/feedback-resume-20260914`.
- 기존 성공 코드: `master`의 2026-09-13 커밋 `1d5c03a`. 새 피드백 방식과 혼동하지 않는다.
- 팔 통합 전 주행 성공: `41ec296`, 로컬 보존 브랜치 `codex/navigation-success-before-arm` 및 태그 `navigation-success-before-arm-20260826`.
- 보고서 기준은 `화재대응_모바일매니퓰레이터_설계및검증_20260916_REV4.pdf`이다. REV3는 이전 조건의 역사 자료로 보존한다. 두 보고서를 모두 최신으로 취급하지 않는다.

## 이전과 현재 차이

| 구분 | 이전 기록 | 현재 작업본 |
|---|---|---|
| 팔 개방 | 기존 설정된 오프셋/궤적을 포함한 별도 접촉 시험의 성공 기록 | YOLO/RGB-D, 관측 퍼짐을 반영한 접근 여유, 실제 관절 반응으로 누름 중단/보정 |
| 통합 성공 판정 | 서비스 성공만으로 실제 문 회전을 보장하지 못한 검사 구간이 있었음 | 대상 Gazebo 문 모델의 신선하고 안정된 실제 회전을 확인해야 성공 |
| 주행/정렬 보호 | 이전 정책의 회귀 기준 | 수동 회피 중 Nav2 목표 재시작 금지, 미세 정렬 전 코스트맵 검사, 차단과 문 오인식 실패 분리 |
| 증빙 | REV3 및 당시 실행 기록 | REV4, 동일 입력 해시의 r8/r9 집계, 이벤트/차체 중첩 검사, 상태별 카메라, 5초마다 궤적 저장 |

문/손잡이 모델만 교체한 버전이 아니다. v3 주 모델/v2 보조 구성은 유지하며 제어와 검증도 변경했다.
관측 기반이라고 해서 모든 상수/안전 한계가 없다는 뜻은 아니다. standoff, 정렬 허용 오차, 누름 증분/상한은 여전히 설정값이다.

## 확인된 결과와 미완료

- 월드 1~5 통합 검사 PASS: 3/3, 3/3, 4/4, 0/0, 6/6. 실제 문 회전 16/16, 이산 차체/장애물 중첩 시료 0.
- r8 월드 3~5와 r9 월드 1~2를 합친 결과다. 코드/검사 해시는 같지만 하나의 연속 실행이나 반복 성공률 시험은 아니다.
- 통합 조건: 저장 지도, sim odometry, 단순 팔/힌지 명령 backend. 실제 PIPER 힘으로 16개 문을 열었다는 뜻이 아니다.
- PIPER 접촉 r21은 FAIL: v3 YOLO 65회, 최대 문 회전 3.50도, 기준 117.46도. 추가 누름의 반응이 없어 정지했다. 영상은 실패/보호 동작의 증거다.
- 통합은 HSV/기하 fallback을 허용하므로 모든 문에서 손잡이 YOLO가 필수 통과했다는 주장을 하지 않는다.
- 회귀 테스트 97개 PASS. 빌드 및 상세 범위는 [검증 결과](VALIDATION_20260916.md)를 따른다.

## 다음 작업

1. 기존 r21 원본의 레버/문/끝단/접촉 데이터를 함께 확인한다. 접촉 센서 표시만으로 안정된 파지나 걸쇠 해제를 단정하지 않는다.
2. 실제 PIPER 형상/관절 한계 안에서 접촉 유지, 레버 해제 후 걸쇠 간섭, 차체 밀기 단계의 원인을 분리한다. 누름/접근 안전 한계만 키워 통과시키지 않는다.
3. 독립 접촉 월드에서 힌지 직접 명령이나 임시 고정 관절 없이 완전 개방을 확인한다. 완료 전 통합 full의 팔 backend를 접촉 성공으로 표현하지 않는다.
4. 독립 성공 후 새 경로에 통합 회귀를 실행한다. 이전 증빙은 덮어쓰지 않는다.
5. 실물 전 센서 무응답 시 정지, 후방/회전 보호, Gazebo 관절 피드백을 대체할 실제 계측을 확인한다.

## 코드와 실행

```bash
git fetch origin
git switch codex/feedback-resume-20260914
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
python3 -m unittest discover -s scripts -p 'test_*.py'
# 기존 출력과 다른 새 경로, 다른 Gazebo 작업이 없는 상태에서만 실행
VALIDATION_WORKSPACE_ROOT="$PWD" bash scripts/run_preserved_full_validation.sh artifacts/validation/team_next_run
```

ROS2 Humble 및 프로젝트의 Gazebo/파이썬 의존성이 준비된 작업공간에서 실행한다. 손잡이/문 가중치와 PIPER 메쉬도 필요하다.

- 주행 정렬/목표: `src/fire_robot_fsm/fire_robot_fsm/state_machine_node.py`, `alignment_safety.py`.
- 힌지 backend 확인: `src/fire_robot_manipulation/fire_robot_manipulation/manipulation_node.py`, `sim_door_feedback.py`.
- 별도 PIPER 접촉: 같은 패키지의 `physical_contact_manipulation_node.py`, `contact_feedback.py`, `piper_actual_kinematics.py`.
- 전용 접촉 실행: `src/fire_robot_bringup/scripts/run_physical_contact_door_test.py`. 인자/launch 조건은 기존 r21 원본과 비교한다.
- 사후 checker의 정답 문 개수/SDF 형상은 평가용이다. FSM에 정답 좌표/문 개수를 주입하지 않는다. 다만 이번 full은 저장 지도를 사용했다.

## 전달할 자료

1. 위 최신 Git 브랜치와 이 문서: 이어서 수정할 코드 기준.
2. REV4 PDF: 보고/설명 기준. 이전 REV3는 필요한 경우에만 '과거 조건'으로 별도 전달.
3. 원본까지 필요한 경우 `00_최종자료/10_통합재검증_보고서개정_20260916` 폴더와 PDF를 상대 구조대로 전달. `evidence_gallery.html`은 이미지가 든 폴더와 함께 있어야 한다.

Git에는 [대표 이미지와 결과](validation/2026-09-16/README.md)를 포함한다. 수백 장의 전체 원본, 영상, 실행 tar, 로컬 보고서는 별도 보관이며 이 저장소의 소형 증빙만으로 대체하지 않는다.
