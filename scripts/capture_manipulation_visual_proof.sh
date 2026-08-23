#!/usr/bin/env bash
set -eo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_ROOT}"

OUTPUT_DIR="${1:-/mnt/c/Users/황준영/Documents/졸업작품/validation_images/manipulation_visual_proof_$(date +%Y%m%d_%H%M%S)}"
LAUNCH_LOG="${OUTPUT_DIR}/manipulation_visual_proof.log"
mkdir -p "${OUTPUT_DIR}"

source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install/setup.bash"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

setsid ros2 launch fire_robot_bringup manipulation_demo.launch.py \
  headless:=true use_rviz:=false >"${LAUNCH_LOG}" 2>&1 &
launch_pid=$!

cleanup() {
  kill -- -"${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}
trap cleanup EXIT

python3 scripts/capture_manipulation_visual_proof.py \
  --output-dir "${OUTPUT_DIR}" \
  --video-output "${OUTPUT_DIR}/gazebo_door_open_motion.mp4"

echo "${OUTPUT_DIR}"
