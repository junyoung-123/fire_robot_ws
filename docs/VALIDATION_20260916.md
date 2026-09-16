# 2026-09-16 재검증 결과

## 결과와 범위

동일 런타임 입력/고정 검사 스크립트로 월드 1~5가 모두 통과했다.
월드 3~5는 `resumed_full_20260916_r8`, 월드 1~2는 `resumed_full_20260916_r9`다.
서로 다른 순차 배치의 집계이며 하나의 중단 없는 배치는 아니다.
입력 SHA256 목록과 검사 스크립트의 일치를 확인한 뒤 집계했다.

| 월드 | 고유 문 실제 회전 확인 | 전체 검사 | 경로 길이 | 샘플 차체/장애물 중첩 |
|---|---:|---|---:|---:|
| 1 | 3/3 | PASS | 31.73 m | 0 |
| 2 | 3/3 | PASS | 28.89 m | 0 |
| 3 | 4/4 | PASS | 29.81 m | 0 |
| 4 | 0/0 | PASS | 33.13 m | 0 |
| 5 | 6/6 | PASS | 31.97 m | 0 |

검사: 미션 완료, 고유 개방 대상/이벤트, 실제 문 회전, 차체 자세,
수동 회피 중 Nav2 재발행, 샘플링 footprint 중첩.
문 개수 3/3/4/0/6은 checker 정답이며 FSM 입력이 아니다.

**범위 제한:** 저장 static map + sim odometry + 단순 팔/힌지 명령 backend.
각 문 모델의 실제 회전은 약 120.3도로 확인했지만 팔의 힘으로 열린 것이 아니다.
손잡이 v3 주 모델/v2 보조를 실행하며 HSV/기하 fallback도 허용한다.
빈 지도 초기 SLAM부터 시작한 임무, 모든 문의 필수 YOLO 검출, 실제 로봇,
새 코드의 반복 성공률/추가 CPU 스트레스 시험을 이번 PASS로 주장하지 않는다.
샘플 중첩 0도 모든 순간의 물리 무충돌 증명은 아니다.

## 수정

- 힌지 명령 송신만으로 성공하지 않고, matching Gazebo 모델의 신선하고 안정된 회전값을 요구한다.
- 문 뒤 연속 벽을 개구부와 힌지 회전 여유로 수정했다. 문/장애물 위치와 수는 유지했다. 단독 문 회전 16/16 확인.
- 수동 회피 중 관측 기억은 계속 갱신하되 Nav2 목표 재발행을 차단했다.
- 개방 후 복귀는 선회 오차 10도 이내에서 전진한다.
- 수동 정렬 전 팽창 코스트맵의 중심 접근 구간을 검사한다. 코스트맵 차단을 문 색 재확인 실패와 분리해 제한된 동일 목표 재계획으로 처리한다.
- 전체 global costmap을 주기 발행하고, ROS 시간 기준 신선도를 확인한다.
- 짧은 DOOR_OPENED 상태도 즉시 발행한다. 검증 CSV는 5초마다 증분 저장/fsync한다.
- 회귀 테스트 97개 PASS. 변경 4개 패키지 빌드 후 최종 FSM 추가 재빌드 PASS.

## PIPER 접촉 시험: 별도 FAIL

`feedback_resume_20260916_r21/physical_contact_door_test_20260916_144018`

- v3 직접 YOLO 관측 65회, registered depth 사용, 선택 신뢰도 0.673.
- 관측 퍼짐 3.56 mm, 팔 형상 안전 하한 70.8 mm, 계산 접근 여유 77.9 mm.
- 누름 확인: 명령 누적 30.0 mm, 실제 끝단 하강 21.2 mm, 레버 회전 6.12도.
- 최대 문 회전 3.50도, 요구 117.46도에 미달. 추가 누름에 실제 반응이 없어 안전 정지.
- 문 힌지 직접 명령/임시 고정 관절/추가 개방 보조 없음. Gazebo 관절 반응을 사용하며 실제 힘 센서 시험이 아니다.
- 보호 분기는 동작했지만 접촉 유지/걸쇠 해제/완전 개방은 해결되지 않았다.

## 관측량과 설정값

본체는 관측 목표와 TF pose의 오차를 줄인다. 목표 standoff와 허용 오차는 설정값이다.
팔 접근 여유는 형상 안전 하한 + 관측 퍼짐의 2배다.
레버 누름은 1cm 명령 증분 뒤 관절 반응으로 중단하며, 필요시 5mm 보정한다.
누름 상한 11cm는 성공 거리도, 검증된 실물 안전 거리도 아니다.
실물에서 Gazebo 레버/힌지 관절값을 대신할 센서와 fail-closed 보호가 필요하다.

## 원본과 보존

- `artifacts/validation/resumed_full_20260916_r8`, `resumed_full_20260916_r9`
- 각 실행: source.patch, commit.txt, runtime_inputs.sha256, runner_snapshot, worldN.log, worldN_trace
- r9: `source_code_without_model_assets.tar.gz` (가중치/큰 메쉬 제외 소스 사본)
- Windows: `00_최종자료/10_통합재검증_보고서개정_20260916`
- REV4 보고서 47쪽, 최신 통합 카메라 PNG 453장 (파일 내용 SHA256 기준 429개), PIPER 단계 사진/영상 및 계측 그래프.
- r5 실패, r6 중단, r7 부분 통과/중단, 기존 REV3와 이전 증빙은 삭제하지 않았다.
- r7 월드 3은 실행 중단으로 종료 시 CSV가 저장되지 않았다. 이를 0 충돌/정렬 PASS로 해석하지 않는다.
- 검증은 `codex/feedback-resume-20260914`의 당시 작업 트리에서 수행했다. 해당 변경은 이후 같은 브랜치에 커밋하여 공유하며, 재현 기준은 실행별 해시/소스 사본이다. 인계 안내는 [TEAM_HANDOFF_20260916.md](TEAM_HANDOFF_20260916.md), 대표 증빙은 [검증 이미지](validation/2026-09-16/README.md)를 참고한다.

## 재현

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m unittest discover -s scripts -p 'test_*.py'
VALIDATION_WORKSPACE_ROOT="$PWD" bash scripts/run_preserved_full_validation.sh artifacts/validation/새로운_실행명
```

기존 출력 경로는 덮어쓰지 않는다. 다른 Gazebo 작업과 동시에 실행하지 않는다.
이번 원본을 재현하려면 실행 당시 소스/파라미터/모델 해시를 먼저 비교한다.
