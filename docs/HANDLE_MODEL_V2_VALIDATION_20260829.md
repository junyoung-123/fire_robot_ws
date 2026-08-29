# handle_best_v2.pt 적용 및 1차 검증 (2026-08-29)

## 적용 파일

팀원 전달 파일:

```text
C:\Users\황준영\Documents\카카오톡 받은 파일\handle_best_v2.pt
```

프로젝트 적용 위치:

```text
src/fire_robot_perception/models/handle_best_v2.pt
install/fire_robot_perception/share/fire_robot_perception/models/handle_best_v2.pt
```

GitHub에는 중복 방지를 위해 ROS 기본 경로인 `src/fire_robot_perception/models/handle_best_v2.pt`만 추적한다. 예전 `handle_best.pt` alias는 기본 실행에서 사용하지 않는다.

모델 확인:

```text
sha256=0c2681fd954da54b71aba2234cfecf1e19832a312c18ba8d156a9a02de463449
names={0: 'lever_handle'}
task=detect
```

## 정적 이미지 검증

명령:

```bash
python3 - <<'PY'
from ultralytics import YOLO
from pathlib import Path
import json
model = YOLO("src/fire_robot_perception/models/handle_best_v2.pt")
imgs = [
 "artifacts/validation/max_sim_recovery_current/live_debug/front.png",
 "artifacts/validation/max_sim_recovery_current/live_debug/front_left.png",
 "artifacts/validation/max_sim_recovery_current/live_debug/front_right.png",
 "artifacts/validation/max_sim_recovery_current/world5_live_debug/front.png",
 "artifacts/validation/max_sim_recovery_current/world5_live_debug/front_left.png",
 "artifacts/validation/max_sim_recovery_current/world5_live_debug/front_right.png",
 "artifacts/validation/max_sim_recovery_current/world5_live_debug2/front.png",
 "artifacts/validation/max_sim_recovery_current/world5_live_debug3/front.png",
]
imgs = [p for p in imgs if Path(p).exists()]
out = []
for p in imgs:
    r = model.predict(p, imgsz=960, conf=0.05, verbose=False)[0]
    out.append({"image": p, "count": len(r.boxes)})
print(json.dumps(out, indent=2))
PY
```

결과:

- 8장 모두 detection 0건.
- 따라서 저장된 기존 Gazebo 샘플 이미지 기준으로는 v2 손잡이 검출 효과가 아직 보이지 않음.
- 다만 해당 이미지들이 손잡이가 작게 보이거나 화면에 거의 없는 샘플일 수 있으므로, live ROS/Gazebo 검증 결과까지 함께 봐야 함.

## live 검증 판정 기준

로그에서 아래 형태가 나오면 v2가 실제로 기여한 것:

```text
Open door request: ..., handle_method=yolo..., handle_detected=True
```

아래 형태가 계속 나오면 v2가 로드되더라도 문개방은 fallback에 의존 중:

```text
Open door request: ..., handle_method=direct_wall_projection, handle_detected=False
Open door request: ..., handle_method=observed_wall_projection, handle_detected=False
```

## Gazebo 손잡이 형상 수정

초기 Gazebo 손잡이는 얇은 금색 원통이었다. 이는 실제 레버형 손잡이 학습 데이터와 형태가 크게 달라 YOLO가 잡기 어려운 조건이다.

2026-08-29 수정:

- 파란 힌지문의 손잡이를 `handle_backplate` + 수평 레버 박스 형상으로 변경.
- `tools/convert_blue_doors_to_hinged.py`도 같은 형상을 생성하도록 수정.
- 기존 검증/스트레스 월드의 파란 힌지문 손잡이 43개를 레버형으로 정규화.

문법 확인:

```text
xml_worlds_ok
python py_compile PASS
colcon build --symlink-install PASS
```

## 레버형 수정 후 live run

명령:

```bash
TRACE_DIR=artifacts/validation/handle_v2_lever_quick_20260829_01/stress \
  scripts/run_headless_validation_once.sh \
  observation_fsm_quick_validation.world \
  artifacts/validation/handle_v2_lever_quick_20260829_01/stress.log \
  700 \
  publish_debug_image:=false \
  log_handle_detections:=true
```

중간 결과:

- 사용자 요청으로 GitHub/전달자료 작업을 우선하기 위해 중간 종료.
- 모델 로드 확인: `classes=['lever_handle']`.
- 파란문 2개는 fallback 좌표로 문 개방 성공.
- `Open door request`는 `handle_detected=False`.
- 로그상 `YOLO handle observed`는 아직 확인되지 않고 `HSV handle observed`만 반복.

## 현재 판단

v2 모델은 파일 형식과 클래스명은 정상이다. 하지만 현재까지의 정적 이미지 검증만 보면 “효과 있음”이라고 말하기 어렵다.

레버형 Gazebo 손잡이로 바꾼 뒤에도 live Gazebo 문앞 장면에서 `handle_detected=False`가 반복되므로, 현재 프로젝트 시뮬레이션에서는 아직 fallback 의존 상태다.

실제 손잡이가 학습 데이터와 비슷하면 실제 환경에서 효과가 있을 수 있다. 다만 실제 로봇 카메라 높이/거리/조명에서 찍은 이미지로 반드시 inference 확인이 필요하다.

실제 이미지에서도 recall이 낮으면 레버 손잡이 데이터셋을 다시 보강해야 한다.

필요 데이터:

- 실제 시연 장소 레버 손잡이 이미지.
- Gazebo 파란문 앞 정렬 전/후 front/front_left/front_right 카메라 이미지.
- 손잡이가 작게 보이는 원거리/측면 이미지.
- 레버가 금색/은색/검정색인 케이스.
- 문 색상이 blue/red/green으로 다른 배경 케이스.
