#!/usr/bin/env bash
set -eo pipefail

WORLD_FILE="${1:-obstacle_wall_doors_v5.world}"
LOG_FILE="${2:-logs/record_sim_validation.log}"
if [[ $# -ge 1 ]]; then
  shift
fi
if [[ $# -ge 1 ]]; then
  shift
fi

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WORKSPACE_ROOT}"

mkdir -p "$(dirname "${LOG_FILE}")"

source /opt/ros/humble/setup.bash
source "${WORKSPACE_ROOT}/install/setup.bash"

export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3

ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=true \
  headless:=false \
  world:="${WORLD_FILE}" \
  "$@" \
  2>&1 | tee "${LOG_FILE}"
