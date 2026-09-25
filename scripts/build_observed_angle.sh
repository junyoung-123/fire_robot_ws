#!/usr/bin/env bash
set -eo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/junyoung/fire_robot_ws_observed_angle_20260922
source /opt/ros/humble/setup.bash
colcon build --base-paths src --symlink-install --parallel-workers 2 --cmake-clean-cache
python3 scripts/test_observed_door_plane.py
