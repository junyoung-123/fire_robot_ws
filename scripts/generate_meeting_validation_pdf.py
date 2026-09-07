#!/usr/bin/env python3
"""Generate the Korean project validation handoff PDF from checked artifacts."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


WORLD_INFO = {
    1: ("혼합 복도", "파란문 3, 빨간문 3, 밀집 장애물", 3),
    2: ("대체 배열", "문·장애물 위치를 바꾼 일반화 검증", 3),
    3: ("다문 환경", "파란문 4, 빨간문 2, 출구 앞 장애물", 4),
    4: ("파란문 없음", "추가 탐색 후 곧바로 비상구 전환", 0),
    5: ("모든 문 파랑", "좌우 3개씩 파란문 6개 처리", 6),
}


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")),
        (Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
         Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Korean", str(regular)))
            pdfmetrics.registerFont(TTFont("KoreanBold", str(bold)))
            return "Korean", "KoreanBold"
    raise FileNotFoundError("Korean font not found")


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_run(log_path: Path, status_path: Path, summary_path: Path) -> dict[str, object]:
    log = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    status = (
        status_path.read_text(encoding="utf-8", errors="ignore").strip()
        if status_path.exists() else "missing")
    summary = parse_key_values(summary_path)
    topics = sorted(set(re.findall(r"to (/fire_robot/door/blue/[^ ]+)", log)))
    return {
        "status": status,
        "mission": "MISSION_COMPLETE" in log,
        "opens": len(re.findall(r"문 개방 성공", log)),
        "topics": len(topics),
        "rejected": len(re.findall(r"Rejected Gazebo", log)),
        "roll": float(summary.get("max_abs_roll_deg", "nan")),
        "pitch": float(summary.get("max_abs_pitch_deg", "nan")),
        "handle_yolo": len(re.findall(r"handle_detected=True", log)),
    }


def fitted_image(path: Path, max_w: float, max_h: float) -> Image:
    if not path.exists():
        raise FileNotFoundError(path)
    image = Image(str(path))
    scale = min(max_w / image.imageWidth, max_h / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def make_styles(regular: str, bold: str):
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKo", parent=styles["Title"], fontName=bold, fontSize=25,
            leading=33, textColor=colors.HexColor("#102A43"), alignment=TA_LEFT,
            spaceAfter=7 * mm),
        "subtitle": ParagraphStyle(
            "SubtitleKo", parent=styles["Normal"], fontName=regular, fontSize=11,
            leading=17, textColor=colors.HexColor("#486581"), spaceAfter=5 * mm),
        "h1": ParagraphStyle(
            "H1Ko", parent=styles["Heading1"], fontName=bold, fontSize=17,
            leading=23, textColor=colors.HexColor("#102A43"), spaceBefore=3 * mm,
            spaceAfter=4 * mm),
        "h2": ParagraphStyle(
            "H2Ko", parent=styles["Heading2"], fontName=bold, fontSize=12,
            leading=17, textColor=colors.HexColor("#1F5A7A"), spaceBefore=2 * mm,
            spaceAfter=2 * mm),
        "body": ParagraphStyle(
            "BodyKo", parent=styles["BodyText"], fontName=regular, fontSize=9.3,
            leading=14.2, textColor=colors.HexColor("#243B53"), spaceAfter=2.5 * mm),
        "small": ParagraphStyle(
            "SmallKo", parent=styles["BodyText"], fontName=regular, fontSize=7.8,
            leading=11.5, textColor=colors.HexColor("#486581"), spaceAfter=1.5 * mm),
        "center": ParagraphStyle(
            "CenterKo", parent=styles["BodyText"], fontName=regular, fontSize=8.5,
            leading=12.5, alignment=TA_CENTER, textColor=colors.HexColor("#243B53")),
        "table": ParagraphStyle(
            "TableKo", parent=styles["BodyText"], fontName=regular, fontSize=7.6,
            leading=10.5, textColor=colors.HexColor("#243B53")),
        "table_bold": ParagraphStyle(
            "TableBoldKo", parent=styles["BodyText"], fontName=bold, fontSize=7.6,
            leading=10.5, textColor=colors.white),
    }


def p(text: str, style) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullet(text: str, style) -> Paragraph:
    return Paragraph(f"• {text}", style)


def result_table(runs: dict[int, dict[str, object]], s) -> Table:
    rows = [[p("월드", s["table_bold"]), p("시나리오", s["table_bold"]),
             p("문 개방", s["table_bold"]), p("오매칭", s["table_bold"]),
             p("차체 기울기", s["table_bold"]), p("결과", s["table_bold"])]]
    for idx, (_, desc, expected) in WORLD_INFO.items():
        run = runs[idx]
        passed = (
            run["status"] == "mission_complete" and run["mission"]
            and run["opens"] == expected and run["topics"] == expected
            and run["rejected"] == 0
            and run["roll"] <= 20.0 and run["pitch"] <= 20.0)
        tilt = f'R {run["roll"]:.1f}° / P {run["pitch"]:.1f}°'
        rows.append([
            p(str(idx), s["table"]), p(desc, s["table"]),
            p(f'{run["opens"]}/{expected}<br/>고유 {run["topics"]}', s["table"]),
            p(str(run["rejected"]), s["table"]), p(tilt, s["table"]),
            p("PASS" if passed else "FAIL", s["table"]),
        ])
    table = Table(rows, colWidths=[12*mm, 55*mm, 25*mm, 18*mm, 34*mm, 20*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F5A7A")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BCCCDC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F8FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf(full_dir: Path, stress_dir: Path, manipulation_dir: Path,
              output: Path) -> None:
    regular, bold = register_fonts()
    s = make_styles(regular, bold)
    runs = {
        idx: parse_run(
            full_dir / f"world{idx}.log",
            full_dir / f"world{idx}.log.status",
            full_dir / f"world{idx}_trace/summary.txt")
        for idx in WORLD_INFO
    }
    for idx, (_, _, expected) in WORLD_INFO.items():
        run = runs[idx]
        if not (run["status"] == "mission_complete" and run["mission"]
                and run["opens"] == expected and run["topics"] == expected
                and run["rejected"] == 0):
            raise RuntimeError(f"World {idx} is not a strict PASS: {run}")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=14*mm, leftMargin=14*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="화재 대응 로봇 시뮬레이션 및 통합 검증 결과",
        author="졸업작품 팀")
    story = []

    story += [Spacer(1, 12*mm), p("화재 대응 로봇", s["title"]),
              p("관측 기반 자율주행·파란문 개방·비상구 탈출<br/>시뮬레이션 통합 검증 결과", s["title"]),
              p("ROS 2 Humble · Gazebo · Nav2 · YOLOv8 · AgileX PIPER 연동 준비", s["subtitle"])]
    badge = Table([[p("최신 전체 검증", s["center"]), p("World 1~5 STRICT PASS", s["center"])],
                   [p("검증 기준일", s["center"]), p("2026-09-07", s["center"])],
                   [p("실제 로봇", s["center"]), p("현장 연동 전 단계", s["center"])]],
                  colWidths=[48*mm, 90*mm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9EAF2")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#D9F2E6")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9FB3C8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [badge, Spacer(1, 9*mm), p("검증 범위", s["h1"]),
              bullet("카메라 3대와 2D LiDAR 관측을 map 좌표에 누적하고, 가장 가까운 미개방 파란문을 잠금 대상으로 선택함.", s["body"]),
              bullet("Nav2가 장애물 회피 경로를 계획하고 문 앞 정렬 후 문 개방 sub-FSM을 실행함.", s["body"]),
              bullet("열린 문 위치를 기억하며 모든 파란문 처리 후 관측된 초록 비상구를 통과함.", s["body"]),
              Spacer(1, 5*mm),
              p("주의: 본 결과는 Gazebo 통합 시뮬레이션 검증임. 실제 PIPER 힘 제어·실물 손잡이 검출·모바일 베이스 TF는 현장 검증이 남아 있음.", s["subtitle"]),
              PageBreak()]

    story += [p("1. 시스템 구조", s["h1"])]
    stages = [
        ("1", "관측", "front / front-left / front-right 카메라\n2D LiDAR /scan"),
        ("2", "의미 지도", "YOLO 문 bbox + HSV 색상\n문·장애물 map 좌표 메모리"),
        ("3", "목표 선택", "가장 가까운 미개방 파란문 잠금\n다른 후보는 처리 완료까지 보류"),
        ("4", "경로 계획", "SmacPlanner2D\nRotation Shim + DWBLocalPlanner"),
        ("5", "문 개방", "정면 주차 → 레버 위치화\n누름 → push-open → 이탈"),
        ("6", "종료", "재탐색 반복\n파란문 없음 확인 → 초록 출구 통과"),
    ]
    data = []
    for number, title, detail in stages:
        data.append([p(number, s["center"]), p(f"<b>{title}</b><br/>{detail}", s["center"])])
    flow = Table(data, colWidths=[14*mm, 145*mm], rowHeights=[24*mm]*len(data))
    flow.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1F5A7A")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#F2F7FA")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BCCCDC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [flow, Spacer(1, 5*mm), p("지도와 좌표 사용 원칙", s["h2"]),
              bullet("실행 중 파란문 개수와 문 좌표는 FSM에 미리 입력하지 않음. 월드별 기대 개수는 검증 checker에서만 정답 대조에 사용함.", s["body"]),
              bullet("현재 시뮬레이션은 고정 구조 static map + localization을 사용하고, 문 색상·문 목표·실시간 장애물은 관측으로 갱신함.", s["body"]),
              bullet("실제 운용에서의 초기 SLAM 지도 작성은 별도 준비 단계이며, 저장한 절대 지도를 localization에 투입하는 구조임.", s["body"]),
              PageBreak()]

    story += [p("2. World 1~5 엄격 전체 검증", s["h1"]),
              p("합격 조건: MISSION_COMPLETE, 기대 파란문 개방 수 일치, 고유 Gazebo 파란문 매칭 수 일치, rejected match 0, 차체 roll/pitch 20° 이하.", s["body"]),
              result_table(runs, s), Spacer(1, 6*mm)]
    stress = parse_run(
        stress_dir / "stress.log", stress_dir / "stress.log.status",
        stress_dir / "trace/summary.txt")
    stress_pass = (
        stress["status"] == "mission_complete" and stress["mission"]
        and stress["opens"] == 6 and stress["topics"] == 6
        and stress["rejected"] == 0)
    if math.isfinite(stress["roll"]) and math.isfinite(stress["pitch"]):
        stress_attitude = (
            f"최대 roll/pitch {stress['roll']:.1f}°/{stress['pitch']:.1f}°")
    else:
        stress_attitude = "차체 기울기는 당시 logger 버전에서 미수집"
    story += [p("No-GUI 부하 검증", s["h2"]),
              p(f"World 5에서 CPU worker 2개를 지속 점유한 상태로 실행(2026-09-06): "
                f"문 {stress['opens']}/6, 고유 매칭 {stress['topics']}/6, "
                f"rejected {stress['rejected']}, 상태 {stress['status']}, "
                f"checker {'PASS' if stress_pass else 'FAIL'}, {stress_attitude}.", s["body"]),
              p("과부하 검증은 GUI 렌더링이 없는 실제 운용 노드 실행과 유사한 스케줄링 여유를 확인하기 위한 시험이며, 실제 하드웨어 신뢰도를 대신하지는 않음.", s["small"]),
              p("2026-09-07 추가 CPU 부하 실행은 2/6 문 진행 중 목표 재계획 지연을 관찰한 뒤 중단했으며, 요청에 따라 주행 정책은 추가 수정하지 않고 참고 로그로만 보존함.", s["small"]),
              PageBreak()]

    story += [p("3. 장애물 위치·크기와 실제 주행 궤적", s["h1"]),
              p("주황 사각형은 Gazebo 월드 파일에서 직접 읽은 장애물 충돌체 크기와 위치이며, 검은 선은 map→base_link 관측 궤적임.", s["small"]),
              fitted_image(full_dir / "full_worlds_obstacle_paths.png", 180*mm, 235*mm),
              PageBreak(), p("4. 문 관측·목표·정렬 결과", s["h1"]),
              p("각 월드의 문 색상 관측점, 목표점, 개방 이벤트, 출구 완료점을 한 화면에 정리함. 점군 산포는 연속 프레임 투영값이며 실제 로봇 궤적은 검은 선임.", s["small"]),
              fitted_image(full_dir / "full_worlds_evidence.png", 180*mm, 235*mm),
              PageBreak()]

    combined = manipulation_dir / "gazebo_door_open_before_after.png"
    summary = parse_key_values(manipulation_dir / "summary.txt")
    story += [p("5. Gazebo 문 개방 시각 증거", s["h1"]),
              p("동일한 Gazebo 카메라에서 개방 명령 전·후를 캡처한 원본 이미지임. 파란 힌지문은 레버 누름 단계 뒤 약 120° 목표로 회전하며, joint feedback이 목표 범위에 도달해야 성공으로 처리됨.", s["body"]),
              fitted_image(combined, 180*mm, 120*mm), Spacer(1, 3*mm),
              p(f"검증 요약: hinge={summary.get('after_hinge_rad', '기록 참조')} rad "
                f"({summary.get('after_hinge_deg', '기록 참조')}°), "
                f"service_success={summary.get('service_success', '기록 참조')}. "
                "전체 주행 검증에서는 LOCALIZE_HANDLE → PRE_GRASP → GRASP_HANDLE → PRESS_HANDLE → PUSH_OPEN → RETURN_HOME → COMPLETE 순서가 실행됨.", s["small"]),
              p("한계: 전체 월드 PASS는 Gazebo 힌지 joint 명령과 피드백을 포함하지만, 실제 접촉력·마찰·레버 토크까지 입증하는 완전한 물리 접촉 시험은 아님.", s["subtitle"]),
              PageBreak()]

    story += [p("6. 학습 모델 적용 상태", s["h1"]),
              p("문 검출 모델", s["h2"]),
              bullet("YOLOv8s Door 1-class best.pt가 simulation/real_robot launch 기본 경로에 연결됨.", s["body"]),
              bullet("OpenImages v7 Door 학습 결과: mAP50 0.6088, mAP50-95 0.3923, Precision 0.6184, Recall 0.5817.", s["body"]),
              bullet("빨강/파랑/초록 구분은 YOLO 클래스가 아니라 bbox 내부 HSV로 수행함.", s["body"]),
              p("손잡이 검출 모델", s["h2"]),
              bullet("팀원 학습 handle_best_v2.pt(lever_handle)를 기본 모델로 연결함.", s["body"]),
              bullet("현재 Gazebo의 단순화된 레버 렌더링에서는 직접 YOLO 검출이 안정적으로 발생하지 않아 HSV 손잡이 또는 문·벽 관측 투영 fallback을 사용함.", s["body"]),
              bullet("따라서 현재 결과는 문 탐색·정렬·개방 FSM 통합을 검증하지만, 실물 레버 손잡이 YOLO 정확도는 별도 촬영 데이터로 재검증해야 함.", s["body"]),
              p("이번에 보완한 오류", s["h2"]),
              bullet("문 개방 직전 최신 카메라 관측을 요구해 과거 projection으로 잘못된 문을 여는 요청을 차단함.", s["body"]),
              bullet("열린/실패 문 주변 중복 후보 억제와 출구 전 추가 탐색 조건을 강화함.", s["body"]),
              bullet("Nav2 시작점 lethal-space 복구, LiDAR 기반 전방 충돌 정지, 출구 앞 장애물 탈출 후 재계획을 추가함.", s["body"]),
              bullet("검증 종료 시 ROS/Gazebo 자식 프로세스를 정리해 다음 월드의 clock/TF 오염을 방지함.", s["body"]),
              PageBreak()]

    story += [p("7. 실제 로봇 전 남은 작업", s["h1"]),
              bullet("PIPER CAN can0 연결 및 SDK/드라이버 통신 확인.", s["body"]),
              bullet("ros2 action list로 /piper_arm_controller/follow_joint_trajectory 제공 여부 확인. 없으면 MoveIt 실행 백엔드를 PIPER SDK 또는 ros2_control 경로로 확정.", s["body"]),
              bullet("PIPER MoveIt 단독 실행, 관절 한계, 충돌 모델, 그리퍼 open/close 값을 실측.", s["body"]),
              bullet("모바일 베이스 /odom→base_link와 카메라·2D LiDAR extrinsic TF를 실측 검증.", s["body"]),
              bullet("연구실 조명에서 문 HSV 및 실물 레버 손잡이 YOLO threshold를 튜닝.", s["body"]),
              bullet("실제 복도 폭·바닥 마찰·장애물에 맞춰 footprint, inflation, 속도, 회전 가속도를 저속부터 단계적으로 조정.", s["body"]),
              bullet("실물 문은 스프링 토크와 개방 방향을 확인하고, 힘/토크 제한 및 비상정지 감시를 넣은 뒤 접촉 시험.", s["body"]),
              Spacer(1, 4*mm), p("미팅에서 확인할 결정 사항", s["h2"]),
              bullet("실제 PIPER 제어 표준 경로: MoveIt FollowJointTrajectory vs PIPER SDK 직접 제어.", s["body"]),
              bullet("시연 문 손잡이 형상·높이·개방 방향과 허용 개방 각도.", s["body"]),
              bullet("초기 SLAM 지도 작성 절차와 시연 당일 localization 초기화 방법.", s["body"]),
              Spacer(1, 5*mm), p("결론", s["h2"]),
              p("현재 단계에서는 다섯 가지 월드에 대한 관측 기반 목표 선택, 장애물 회피, 문앞 정렬, Gazebo 문 개방, 출구 통과의 소프트웨어 통합 동작을 확인했다. 다음 단계의 핵심은 시뮬레이션 정책 확대가 아니라 실제 센서 TF·조명·모바일 베이스·PIPER 실행 경로를 순서대로 연결하고 저속 현장 튜닝하는 것이다.", s["body"])]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(regular, 7.5)
        canvas.setFillColor(colors.HexColor("#829AB1"))
        canvas.drawString(14*mm, 8*mm, "화재 대응 로봇 졸업작품 · 시뮬레이션 검증 자료")
        canvas.drawRightString(A4[0] - 14*mm, 8*mm, f"{document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--stress-dir", type=Path, required=True)
    parser.add_argument("--manipulation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_pdf(args.full_dir, args.stress_dir, args.manipulation_dir, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
