#!/usr/bin/env bash
set -euo pipefail
SOURCE=$(cd "$(dirname "$0")/.." && pwd)
DEST=/home/junyoung/fire_robot_ws_observed_nav_20260928
mkdir -p "$DEST/src" "$DEST/scripts"
rsync -a --exclude='maps/*.pgm' --exclude='maps/*.yaml' "$SOURCE/src/" "$DEST/src/"
rsync -a "$SOURCE/scripts/" "$DEST/scripts/"
mkdir -p "$DEST/src/fire_robot_perception/models"
for model in best.pt handle_best_v2.pt handle_best_v3_CANDIDATE.pt; do
  cp -n "/home/junyoung/fire_robot_ws_test/src/fire_robot_perception/models/$model" \
    "$DEST/src/fire_robot_perception/models/$model"
done
cd "$DEST"
set +u
source /opt/ros/humble/setup.bash
set -u
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
colcon build --symlink-install --executor sequential --packages-skip fire_robot_sim_evidence
set +u
source install/setup.bash
set -u
python3 scripts/test_observed_wall_axis.py
python3 scripts/test_observed_bbox_range.py
python3 scripts/test_observed_parking.py
python3 scripts/test_observed_exit.py
python3 scripts/test_observed_exit_aperture.py
python3 scripts/test_observed_wall_normal.py
python3 scripts/test_open_exit_world.py
python3 scripts/test_observed_velocity_safety.py
python3 scripts/test_observed_navigation_launch.py
python3 scripts/test_motion_scan_filter.py
python3 scripts/test_slam_correction_tf.py
python3 scripts/test_observed_navigation_checker.py
python3 scripts/test_observed_navigation_watchdog.py
