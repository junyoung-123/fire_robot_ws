#!/usr/bin/env bash
set -o pipefail

WORLD_FILE="${1:-obstacle_wall_doors_v5.world}"
LOG_FILE="${2:-artifacts/validation/latest_headless.log}"
MAX_SEC="${3:-900}"
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_ROOT}"
mkdir -p "$(dirname "${LOG_FILE}")"
rm -f "${LOG_FILE}.status"

source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install/setup.bash"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=false \
  headless:=true \
  world:="${WORLD_FILE}" \
  "$@" > "${LOG_FILE}" 2>&1 &
launch_pid=$!

trace_pid=""
if [[ -n "${TRACE_DIR:-}" ]]; then
  mkdir -p "${TRACE_DIR}"
  python3 scripts/validation_trace_logger.py \
    --output-dir "${TRACE_DIR}" \
    > "${TRACE_DIR}/trace_logger.log" 2>&1 &
  trace_pid=$!
fi

status="timeout"
start_seconds=${SECONDS}
deadline=$((SECONDS + MAX_SEC))
while kill -0 "${launch_pid}" 2>/dev/null; do
  sleep 5
  if grep -aqE 'MISSION_COMPLETE|State: MISSION_COMPLETE|미션 완료' "${LOG_FILE}"; then
    status="mission_complete"
    break
  fi
  if grep -aqE 'State: .*EMERGENCY_STOP|Traceback|IndexError|process has died' "${LOG_FILE}"; then
    status="failed"
    break
  fi
  elapsed=$((SECONDS - start_seconds))
  if (( elapsed >= 240 )) && ! grep -aqE 'State:' "${LOG_FILE}"; then
    status="failed"
    break
  fi
  if (( elapsed >= 180 )) && grep -aqE 'Invalid frame ID "map"|map_server/change_state.*timeout|AMCL/map still inactive' "${LOG_FILE}" && ! grep -aqE 'State:' "${LOG_FILE}"; then
    status="failed"
    break
  fi
  if (( SECONDS >= deadline )); then
    status="timeout"
    break
  fi
done

if kill -0 "${launch_pid}" 2>/dev/null; then
  kill -INT "${launch_pid}" 2>/dev/null || true
  sleep 5
  kill -TERM "${launch_pid}" 2>/dev/null || true
fi
wait "${launch_pid}" 2>/dev/null || true

if [[ -n "${trace_pid}" ]] && kill -0 "${trace_pid}" 2>/dev/null; then
  kill -INT "${trace_pid}" 2>/dev/null || true
  sleep 2
  kill -TERM "${trace_pid}" 2>/dev/null || true
fi
if [[ -n "${trace_pid}" ]]; then
  wait "${trace_pid}" 2>/dev/null || true
fi

printf '%s\n' "${status}" > "${LOG_FILE}.status"
