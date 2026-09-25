#!/usr/bin/env bash
set -eo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /home/junyoung/fire_robot_ws_corridor_demo_20260922
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=219
export IGN_PARTITION=fire_robot_corridor_demo_20260922
export LIBGL_ALWAYS_SOFTWARE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec python3 scripts/run_corridor_demo.py \
  --headless --use-yolo-observation --feedback-contact \
  --minimum-ik-condition-ratio .25 --minimum-yolo-confidence .45 \
  --ready-timeout-sec 80 --yolo-timeout-sec 90 --service-timeout-sec 900 \
  --observation-approach-timeout-sec 60 --video-fps 3 \
  --output-root artifacts/validation/corridor_demo_r1
