# 팀원 작업 인계 (2026-09-17)

## 어느 코드를 사용할지

- 최신 단일문 접촉 작업: `codex/world1-contact-integration-20260916`.
- 월드 1~5 보존 주행 성공본: `codex/feedback-resume-20260914`, 커밋 `d1ffb39`.
- 로봇팔 이전 주행 보존본: `codex/navigation-success-before-arm`, `navigation-success-before-arm-20260826`, 커밋 `41ec296`.
- `master`는 이전 소스와 최신 안내를 제공하는 입구입니다. 최신 실험 소스와 동일하지 않습니다.

서로 다른 브랜치의 성공 조건을 합쳐서 "새 접촉 팔로 5개 월드 FULL 완료"라고 표시하지 않습니다. [검증 범위](VALIDATION_20260917.md)를 먼저 읽어 주세요.

## 보고서

[REV6](reports/2026-09-17/00_읽는순서.md)는 PDF·HTML·상대경로 사진/영상을 포함합니다. 저장소를 내려받은 뒤 `START_보고서와영상.html`을 브라우저로 열면 됩니다. GitHub 파일 화면에서는 HTML이 앱처럼 실행되지 않습니다. PDF만 전달하면 영상 파일은 전달되지 않습니다.

## 환경과 코드 검사

Ubuntu 22.04, ROS2 Humble, Gazebo Fortress를 사용했습니다. 기존 성공 작업공간을 덮어쓰지 말고 별도 clone/작업공간을 사용합니다.

```bash
git clone --branch codex/world1-contact-integration-20260916 \
  https://github.com/junyoung-123/fire_robot_ws.git fire_robot_ws_contact
cd fire_robot_ws_contact
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
# 필요한 ROS/Gazebo 의존성을 먼저 설치한 환경에서 수행
colcon build --symlink-install --parallel-workers 1
source install/setup.bash
python3 -m pytest -q scripts
```

실제 동작 옵션과 검증 실행기는 `src/fire_robot_bringup/launch/physical_contact_door_test.launch.py`와 `src/fire_robot_bringup/scripts/run_physical_contact_door_test.py`에서 확인합니다. 기존 실행의 설정/로그는 REV6 증거와 로컬 원본에 보존되어 있습니다. 실물 로봇에 fixture 파라미터를 그대로 적용하지 않습니다.

## 다음 작업 원칙

1. r40 결과와 같은 입력 소스인지 확인하고, 변경할 때마다 별도 실행 ID와 해시를 남깁니다.
2. 단일문 반복 재현과 변화된 물성/관측 조건을 먼저 확인합니다.
3. 주행-접근-파지-개방-해제-회수-후진-다음 문 전환을 결합합니다.
4. 새 backend로 실제 실행한 월드만 새 FULL 결과에 집계합니다.
5. 실패 원본을 지우거나 판정 기준을 낮추지 않습니다. 새로운 판정은 별도 감사 결과로 저장합니다.

이번 GitHub 동기화는 기존 수정 소스의 저장과 문서 갱신입니다. 주행 정책을 새로 바꾼 작업이 아닙니다.
