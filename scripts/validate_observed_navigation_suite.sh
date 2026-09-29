#!/usr/bin/env bash
# Stop at the first independent failure; never overwrite an existing trial.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${1:?result root required}
REV=${2:?revision label required}
shift 2
EXTRA=()
if [[ ${1:-} == --open-exit ]]; then
  EXTRA=(--open-exit)
  shift
fi
source /opt/ros/humble/setup.bash
source /home/junyoung/fire_robot_ws_observed_nav_20260928/install/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/junyoung/fire_robot_ws_observed_nav_20260928
for NUMBER in "$@"; do
  case "$NUMBER" in
    1) WORLD=obstacle_wall_doors_v5.world ;;
    2) WORLD=obstacle_door_layout_alt_v1.world ;;
    3) WORLD=obstacle_door_layout_world3_v1.world ;;
    4) WORLD=obstacle_no_blue_world4_v1.world ;;
    5) WORLD=obstacle_all_blue_world5_v1.world ;;
    *) printf 'Invalid world number: %s\n' "$NUMBER"; exit 2 ;;
  esac
  OUTPUT="$ROOT/world${NUMBER}_${REV}"
  python3 scripts/run_observed_navigation.py --world "$WORLD" --output "$OUTPUT" --timeout 1500 "${EXTRA[@]}"
  python3 "$HERE/render_observed_navigation_evidence.py" "$OUTPUT"
  EVALUATION_WORLD="src/fire_robot_bringup/worlds/$WORLD"
  if [[ -f "$OUTPUT/open_exit.world" ]]; then EVALUATION_WORLD="$OUTPUT/open_exit.world"; fi
  python3 "$HERE/check_observed_navigation.py" "$OUTPUT" --world-file "$EVALUATION_WORLD"
  if [[ -f "$ROOT/stop_after_current" ]]; then break; fi
done
