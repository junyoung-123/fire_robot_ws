# 최신 이미지·영상·실험결과 자료집

정리 날짜: 2026-09-29. 최신 주행은 r29 온라인 SLAM, 물리 개방은 별도 R34 단일문이다.

1. `02_EVIDENCE_SCOPE.md`에서 시험 범위를 먼저 읽는다.
2. `03_TOPIC_AND_SLIDE_MAP.md`에서 발표 페이지에 맞는 ID를 찾는다.
3. `01_ASSET_INDEX.csv` 또는 JSON에서 ID·날짜·버전·설명을 검색한다.
4. `assets/` 원본과 `asset_notes/ID.md`를 함께 사용한다.
5. `00_GALLERY.html`을 브라우저로 열면 이미지와 영상을 확인할 수 있다.

## 먼저 사용할 자료

- World 1~5 궤적: `N1_path`~`N5_path`, 시작 카메라: `N1_start`~`N5_start`.
- 지도 생성 과정: `N_all_slam`, 각 `N*_map_initial`, `N*_map_middle`, `N*_map_latest`.
- World 4는 파란 문이 없어 방문·정렬 그래프가 없다. `N4_alignment`는 해당 없음이다.
- 통합 궤적: `N_all_paths`. 별표는 문 방문 위치이며 물리 개방 표시가 아니다.
- 단일문 접촉 개방 영상: 기존 `OV1`, `OV2`의 노트·범위를 확인한다. 주행 기록과 이어 붙여 한 실행으로 표현하지 않는다.
- 실제 근거: `evidence/nav_r29/00_최신결과.html`, 각 world*_r29 폴더의 JSON·카메라·지도·로그.

전체 자료 113개. 이전 W* 주행은 `archive/images/NAV_20260916/`로 구분했다.
이전 폴더와 보고서 원본은 그대로 보존했다. 과거 PDF 안내서는 archive에 있으며 최신 결과 설명서는 이 폴더의 Markdown이다.

## 코드

- 저장소: https://github.com/junyoung-123/fire_robot_ws
- 최신 주행: `codex/observed-nav-20260928`
- 최신 결과: https://github.com/junyoung-123/fire_robot_ws/blob/codex/observed-nav-20260928/docs/RELEASE_20260929.md
- R34 보존: `codex/observed-door-angle-20260922`, `27359f7`
- `master` 소스는 이전 기준선이다. 새 주행은 observed_navigation 실행 경로를 사용한다.

시뮬레이터 정답 배치·위치를 제어 입력으로 쓰지 않았다는 것과 모든 파라미터가 없다는 것은 다르다.
서로 다른 시험의 사진을 같은 연속 시연으로 합치거나, 주행 성공을 다문 물리 개방 성공으로 바꾸지 않는다.
발표에 쓴 ID와 파일명을 남겨 추적 가능하게 한다. 기존 사진의 라이선스·출처는 각 자료 노트를 유지한다.
