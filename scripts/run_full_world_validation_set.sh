#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-artifacts/validation/full_worlds_latest}"
mkdir -p "${OUT_DIR}"

worlds=(
  "1 obstacle_wall_doors_v5.world 3 1100"
  "2 obstacle_door_layout_alt_v1.world 3 1100"
  "3 obstacle_door_layout_world3_v1.world 4 1500"
  "4 obstacle_no_blue_world4_v1.world 0 700"
  "5 obstacle_all_blue_world5_v1.world 6 1300"
)

for item in "${worlds[@]}"; do
  read -r idx world expected timeout <<< "${item}"
  log="${OUT_DIR}/world${idx}.log"
  trace="${OUT_DIR}/world${idx}_trace"
  rm -f "${log}" "${log}.status"
  rm -rf "${trace}"
  mkdir -p "${trace}"

  echo "=== WORLD ${idx}: ${world} ==="
  TRACE_DIR="${trace}" scripts/run_headless_validation_once.sh \
    "${world}" "${log}" "${timeout}" \
    publish_debug_image:=false \
    log_handle_detections:=true
  cat "${log}.status"
  python3 scripts/check_validation_log.py \
    "${log}" \
    --expected-open-count "${expected}" \
    --max-rejected 0
  python3 scripts/check_trace_attitude.py \
    "${trace}/summary.txt" \
    --max-deg 20
done
