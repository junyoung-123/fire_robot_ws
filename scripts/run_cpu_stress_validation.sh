#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/validation/cpu_stress_latest}"
CPU_WORKERS="${CPU_WORKERS:-6}"
WORLD_FILE="${WORLD_FILE:-obstacle_all_blue_world5_v1.world}"
EXPECTED_OPEN_COUNT="${EXPECTED_OPEN_COUNT:-6}"
MAX_SEC="${MAX_SEC:-1500}"
LOAD_PIDS=()

cleanup_load() {
  for pid in "${LOAD_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  wait "${LOAD_PIDS[@]}" 2>/dev/null || true
}
trap cleanup_load EXIT INT TERM

mkdir -p "${OUT_DIR}/trace"
for ((worker = 0; worker < CPU_WORKERS; worker += 1)); do
  yes > /dev/null &
  LOAD_PIDS+=("$!")
done

printf 'world=%s\ncpu_workers=%s\nlogical_cpus=%s\n' \
  "${WORLD_FILE}" "${CPU_WORKERS}" "$(nproc)" > "${OUT_DIR}/stress_environment.txt"

TRACE_DIR="${OUT_DIR}/trace" scripts/run_headless_validation_once.sh \
  "${WORLD_FILE}" "${OUT_DIR}/stress.log" "${MAX_SEC}" \
  publish_debug_image:=false \
  log_handle_detections:=true

cat "${OUT_DIR}/stress.log.status"
python3 scripts/check_validation_log.py \
  "${OUT_DIR}/stress.log" \
  --expected-open-count "${EXPECTED_OPEN_COUNT}" \
  --max-rejected 0 | tee "${OUT_DIR}/checker.txt"
python3 scripts/check_trace_attitude.py \
  "${OUT_DIR}/trace/summary.txt" \
  --max-deg 20 | tee "${OUT_DIR}/attitude_checker.txt"
