# 레버 손잡이 YOLO 재검증/재학습 요청 (2026-08-29)

## 목적

`handle_detector.pt` 또는 `handle_best_v2.pt`는 문 bbox 안에서 레버형 손잡이만 검출하는 1-class YOLO 모델입니다.

문 색상(red/blue/green)은 이 모델에서 분류하지 않습니다. 문 색상은 기존 코드의 HSV 분류가 담당합니다.

## 현재 모델 v2 상태

파일:

```text
handle_best_v2.pt
```

ROS 적용명:

```text
src/fire_robot_perception/models/handle_best_v2.pt
```

클래스:

```text
lever_handle
```

현재 확인:

- 모델 로드는 정상.
- 기존 Gazebo 카메라 샘플 8장에서는 detection 0건.
- Gazebo 손잡이를 원통형에서 레버형으로 바꾼 후 live run에서도 아직 `handle_detected=True`는 확인되지 않음.
- 문개방은 현재 fallback 손잡이 좌표로 가능하지만, 손잡이 YOLO가 실제로 기여한다고 보기에는 증거가 부족함.

## 왜 추가 데이터가 필요한가

현재 시뮬레이션과 실제 데이터 사이에 차이가 큽니다.

- Gazebo 손잡이는 단순 금색 형상이라 실제 레버의 질감/윤곽과 다름.
- 로봇 카메라 해상도는 320x240 계열이라 손잡이가 작게 보임.
- 문 bbox가 화면 위/가장자리에서 잘리는 경우가 많음.
- 측면 카메라에서는 손잡이가 긴 레버가 아니라 작은 금색 픽셀 덩어리처럼 보일 수 있음.
- v2 학습 데이터의 각도/거리/조명이 실제 시연 조건과 다르면 recall이 낮아질 수 있음.

## 팀원에게 부탁할 데이터

실제 시연 장소 또는 비슷한 문 손잡이에서 아래 조건으로 촬영해 주세요.

- 거리: 0.2m, 0.4m, 0.8m, 1.5m
- 각도: 정면, 좌측 30도, 우측 30도, 좌측 60도, 우측 60도
- 높이: 실제 로봇 팔 카메라 높이와 비슷하게
- 배경: 파란문/빨간문/초록문 또는 색상 종이 부착 문
- 조명: 연구실 밝은 조명, 약간 어두운 조명, 역광/반사
- 상태: 손잡이만 크게 보이는 이미지, 문 전체 bbox 안에서 작게 보이는 이미지 모두 포함

가능하면 Gazebo에서도 `publish_debug_image:=true`로 저장한 문앞 정렬 장면을 함께 넣어 주세요.

## 라벨링 기준

클래스는 하나만 사용합니다.

```text
lever_handle
```

라벨 bbox는 손잡이 레버 전체를 감싸면 됩니다.

- 레버 막대 포함
- 축/로제트/백플레이트는 보이면 같이 포함해도 됨
- 문 전체, 손, 로봇팔, 경첩은 포함하지 않음
- 너무 작은 원거리 샷도 버리지 말고 라벨링

## 학습 목표

우선순위는 Recall입니다. 손잡이를 놓치면 fallback으로 가기 때문에 모델 효과를 입증하기 어렵습니다.

목표:

```text
Recall >= 0.70
mAP50 >= 0.50
Precision은 0.50 이상이면 우선 허용
```

## 학습 예시

```bash
cd ~/fire_robot_ws_test/src/fire_robot_perception/scripts

python3 train_handle_detector.py \
  --dataset ~/datasets/lever_handle_detection/dataset.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch 8 \
  --install
```

GPU 메모리가 부족하면:

```bash
python3 train_handle_detector.py \
  --dataset ~/datasets/lever_handle_detection/dataset.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 768 \
  --batch 8 \
  --install
```

## 전달 결과물

아래 4개를 전달해 주세요.

```text
runs/detect/handle_detector/weights/best.pt
mAP50
Precision
Recall
테스트 이미지 몇 장의 예측 결과
```

## ROS 적용 후 확인할 로그

성공 기준:

```text
YOLO handle observed: ...
Open door request: ..., handle_method=yolo:lever_handle, handle_detected=True
```

fallback 기준:

```text
Open door request: ..., handle_method=direct_wall_projection, handle_detected=False
Open door request: ..., handle_method=observed_wall_projection, handle_detected=False
```

`문 개방 성공`만으로는 손잡이 YOLO 성공이라고 볼 수 없습니다. 반드시 `handle_detected=True`를 확인해야 합니다.
