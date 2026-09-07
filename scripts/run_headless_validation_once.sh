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

cleanup_sim_processes() {
  pkill -TERM -f '[r]os2cli.daemon.daemonize' 2>/dev/null || true
  pkill -TERM -f '[r]os2 launch fire_robot_bringup simulation.launch.py' 2>/dev/null || true
  pkill -TERM -f '/(robot_state_publisher|parameter_bridge|map_server|amcl|planner_server|controller_server|smoother_server|behavior_server|bt_navigator|waypoint_follower|velocity_smoother|lifecycle_manager|state_machine_node|navigation_node|door_detection_node|manipulation_node|sensor_fusion_node|cmd_vel_safety_node)([[:space:]]|$)' 2>/dev/null || true
  pkill -TERM -f '[i]gn gazebo' 2>/dev/null || true
  pkill -TERM -f '[g]z sim' 2>/dev/null || true
  pkill -TERM -x gzserver 2>/dev/null || true
  pkill -TERM -x gzclient 2>/dev/null || true
  sleep 3
  pkill -KILL -f '[r]os2cli.daemon.daemonize' 2>/dev/null || true
  pkill -KILL -f '[r]os2 launch fire_robot_bringup simulation.launch.py' 2>/dev/null || true
  pkill -KILL -f '/(robot_state_publisher|parameter_bridge|map_server|amcl|planner_server|controller_server|smoother_server|behavior_server|bt_navigator|waypoint_follower|velocity_smoother|lifecycle_manager|state_machine_node|navigation_node|door_detection_node|manipulation_node|sensor_fusion_node|cmd_vel_safety_node)([[:space:]]|$)' 2>/dev/null || true
  pkill -KILL -f '[i]gn gazebo' 2>/dev/null || true
  pkill -KILL -f '[g]z sim' 2>/dev/null || true
  pkill -KILL -x gzserver 2>/dev/null || true
  pkill -KILL -x gzclient 2>/dev/null || true
  killall -q -KILL ign gzserver gzclient 2>/dev/null || true
}

if pgrep -f '[r]os2 launch fire_robot_bringup real_robot.launch.py' >/dev/null; then
  echo '[validation] Refusing to stop ROS processes while real_robot.launch.py is active.' >&2
  exit 2
fi

cleanup_sim_processes
sleep 4

source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install/setup.bash"
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  export ROS_DOMAIN_ID="$((100 + RANDOM % 100))"
fi
ros2 daemon stop >/dev/null 2>&1 || true
echo "[validation] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" > "${LOG_FILE}.domain"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

map_args=()
if [[ " $* " != *" map:="* ]]; then
  map_base="${WORLD_FILE%.world}_static.yaml"
  map_candidate="${WORKSPACE_ROOT}/install/fire_robot_navigation/share/fire_robot_navigation/maps/${map_base}"
  if [[ -f "${map_candidate}" ]]; then
    map_args=(map:="${map_candidate}")
  fi
fi
start_args=()
if [[ " $* " != *" start_without_fire:="* ]]; then
  start_args=(start_without_fire:=true)
fi
lifecycle_args=()
if [[ " $* " != *" enable_lifecycle_recovery:="* ]]; then
  lifecycle_args=(enable_lifecycle_recovery:=true)
fi

ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=false \
  headless:=true \
  world:="${WORLD_FILE}" \
  "${map_args[@]}" \
  "${start_args[@]}" \
  "${lifecycle_args[@]}" \
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
  if (( elapsed >= 360 )) && ! grep -aqE 'State:' "${LOG_FILE}"; then
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

cleanup_sim_processes

printf '%s\n' "${status}" > "${LOG_FILE}.status"
case "${status}" in
  mission_complete) exit 0 ;;
  timeout) exit 124 ;;
  *) exit 1 ;;
esac
