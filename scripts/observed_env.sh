#!/usr/bin/env bash
set -eo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
source /home/junyoung/fire_robot_ws_observed_angle_20260922/install/setup.bash
export ROS_DOMAIN_ID=224
export IGN_PARTITION=fire_robot_observed_angle_20260922
exec "$@"
