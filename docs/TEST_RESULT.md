# 검증 결과 정리 (2026-08-11)

## 빌드 및 정적 검사

```text
python3 -m py_compile \
  src/fire_robot_fsm/fire_robot_fsm/state_machine_node.py \
  src/fire_robot_fsm/fire_robot_fsm/fsm_subsystems.py \
  src/fire_robot_bringup/launch/simulation.launch.py

colcon build --symlink-install --packages-select fire_robot_fsm fire_robot_bringup
Summary: 2 packages finished
결과: PASS
```

이전 단계에서 navigation/perception 포함 전체 워크스페이스 빌드도 통과했습니다.

## 최종 Full Validation

검증 태그:

```text
full_worlds_12345_exit_tail_red_guard_20260811
```

검증 방식:

- Gazebo headless
- ROS2 Humble
- YOLO Door bbox + HSV 색상 분류
- 3카메라 + 2D LiDAR observation map
- Nav2 SmacPlanner2D + RotationShim + DWBLocalPlanner
- 좌표/문 개수 하드코딩 없이 관측된 문 후보와 map 메모리 기반으로 목표 선택

## 월드별 결과

### World 1

- 월드: `obstacle_wall_doors_v5.world`
- 조건: 파란문 3개, 빨간문/장애물 혼합
- 결과: PASS

```text
expected_blue=3
opened_count=3
matched_blue=3/3
false_opened=(none)
mission_complete=True
navigation_failed_logs=0
red_opened=(none)
align_ok_logs=3
```

### World 2

- 월드: `obstacle_door_layout_alt_v1.world`
- 조건: 다른 문 배열/장애물 배치
- 결과: PASS

```text
expected_blue=3
opened_count=3
matched_blue=3/3
false_opened=(none)
mission_complete=True
navigation_failed_logs=0
red_opened=(none)
align_ok_logs=3
```

### World 3

- 월드: `obstacle_door_layout_world3_v1.world`
- 조건: 파란문 4개, 빨간문 2개
- 결과: PASS

```text
expected_blue=4
opened_count=4
matched_blue=4/4
false_opened=(none)
mission_complete=True
navigation_failed_logs=0
red_opened=(none)
align_ok_logs=4
```

### World 4

- 월드: `obstacle_no_blue_world4_v1.world`
- 조건: 파란문 없음
- 결과: PASS

```text
expected_blue=0
opened_count=0
matched_blue=0/0
false_opened=(none)
mission_complete=True
navigation_failed_logs=0
red_opened=(none)
align_ok_logs=0
```

### World 5

- 월드: `obstacle_all_blue_world5_v1.world`
- 조건: 좌우 3개씩 모든 문 파란색
- 결과: PASS

```text
expected_blue=6
opened_count=6
matched_blue=6/6
false_opened=(none)
mission_complete=True
navigation_failed_logs=0
red_opened=(none)
align_ok_logs=6
```

## 검증 증빙 파일

```text
docs/validation/2026-08-11/summary.txt
docs/validation/2026-08-11/progress.txt
docs/validation/2026-08-11/world1_trajectory.png
docs/validation/2026-08-11/world1_alignment.png
...
docs/validation/2026-08-11/world5_trajectory.png
docs/validation/2026-08-11/world5_alignment.png
```

## 검증 명령

```bash
cd ~/fire_robot_ws_test
bash /mnt/c/Users/황준영/Documents/졸업작품/run_full_worlds_20260802.sh \
  full_worlds_12345_exit_tail_red_guard_20260811
```

개별 월드 검증:

```bash
cd ~/fire_robot_ws_test
TAG=world1_exit_tail_red_guard_20260811 \
bash /mnt/c/Users/황준영/Documents/졸업작품/run_single_world_validation_20260802.sh \
  world1 obstacle_wall_doors_v5.world 81
```

## 결론

시뮬레이션에서 요구한 기본 주행 정책은 현재 월드 1~5 기준으로 통과했습니다.

- 파란문/빨간문 구분
- 관측된 파란문 전체 개방
- 장애물 회피 주행
- 문 앞 정면 정렬
- 파란문이 없는 환경에서 바로 비상구 이동
- 모든 파란문 개방 후 초록 비상구 통과

실제 로봇 검증은 아직 남아 있으며, PIPER/베이스/카메라/2D LiDAR TF와 연구실 조명 조건 튜닝이 필요합니다.
