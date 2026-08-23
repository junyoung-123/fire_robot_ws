#!/usr/bin/env bash
set -eo pipefail

WS=${WS:-/home/junyoung/fire_robot_ws_test}
WORLD=${WORLD:-obstacle_wall_doors_v5.world}
WORLD_PATH="${WS}/src/fire_robot_bringup/worlds/${WORLD}"
LOG_PREFIX=${LOG_PREFIX:-/tmp/handle_manip_smoke}
HANDLE_X=${HANDLE_X:-3.24}
HANDLE_Y=${HANDLE_Y:--1.82}
HANDLE_Z=${HANDLE_Z:-0.95}

GAZEBO_LOG="${LOG_PREFIX}_gazebo.log"
MANIP_LOG="${LOG_PREFIX}_manipulation.log"
CALL_LOG="${LOG_PREFIX}_call.log"

cleanup() {
  pkill -f "ign gazebo -r -s ${WORLD_PATH}" 2>/dev/null || true
  pkill -f "ros2 run fire_robot_manipulation manipulation_node" 2>/dev/null || true
  pkill -f "ros2 service call /open_door" 2>/dev/null || true
}
trap cleanup EXIT

cd "${WS}"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

rm -f "${GAZEBO_LOG}" "${MANIP_LOG}" "${CALL_LOG}"

timeout 35s ign gazebo -r -s "${WORLD_PATH}" >"${GAZEBO_LOG}" 2>&1 &
timeout 32s ros2 run fire_robot_manipulation manipulation_node \
  --ros-args \
  -p sim_mode:=true \
  -p sim_physical_door_opening:=true \
  -p sim_door_physics_required:=true \
  -p sim_door_command_steps:=1 \
  -p sim_door_world_file:="${WORLD_PATH}" \
  -p sim_arm_motion_enabled:=true \
  >"${MANIP_LOG}" 2>&1 &

sleep 7

timeout 16s ros2 service call /open_door fire_robot_interfaces/srv/OpenDoor \
  "{door_id: test_blue, handle_position: {header: {frame_id: map}, point: {x: ${HANDLE_X}, y: ${HANDLE_Y}, z: ${HANDLE_Z}}}}" \
  >"${CALL_LOG}" 2>&1 || true

sleep 7

echo "SERVICE_CALL"
cat "${CALL_LOG}" || true
echo "MANIPULATION_LOG"
cat "${MANIP_LOG}" || true
echo "GAZEBO_RELEVANT_LOG"
grep -aE "Err|Error|WARN|Door|JointPositionController|Topic" "${GAZEBO_LOG}" | tail -n 80 || true
