#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/junyoung/fire_robot_ws_corridor_demo_20260922
python3 scripts/prepare_corridor_demo_world.py
python3 -m py_compile scripts/run_corridor_demo.py scripts/prepare_corridor_demo_world.py
set +u
source /opt/ros/humble/setup.bash
set -u
colcon build --symlink-install --parallel-workers 2 --cmake-clean-cache
