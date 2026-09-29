#!/usr/bin/env bash
set -eo pipefail
cd /home/junyoung/fire_robot_ws_observed_nav_20260928
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
exec python3 scripts/run_observed_navigation.py "$@"
