#!/usr/bin/env bash
set -eo pipefail
export CONTACT_ANGLE_CONTROLLER=robot_observation_manipulation_node
export CONTACT_ENVIRONMENT_FEEDBACK_TOPIC=/observed_door/rotation
export CONTACT_LEVER_FEEDBACK_JOINT=encoder_lever_rotation
exec bash /home/junyoung/fire_robot_ws_observed_angle_20260922/scripts/start_minimal_angle.sh "$1"
