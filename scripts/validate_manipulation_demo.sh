#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source install/setup.bash

launch_log=$(mktemp)
setsid ros2 launch fire_robot_bringup manipulation_demo.launch.py headless:=true >${launch_log} 2>&1 &
launch_pid=$!
cleanup() { kill -- -${launch_pid} 2>/dev/null || true; wait ${launch_pid} 2>/dev/null || true; rm -f ${launch_log}; }
trap cleanup EXIT

sleep 8
service_output=$(timeout 15 ros2 service call /open_door fire_robot_interfaces/srv/OpenDoor \
  '{door_id: manipulation_demo, handle_position: {header: {frame_id: base_link}, point: {x: 0.60, y: 0.0, z: 0.80}}}')
echo ${service_output}
grep -q 'success=True' <<<${service_output}
echo 'Door hinge state after /open_door:'
timeout 8 ros2 topic echo /door_joint_states sensor_msgs/msg/JointState --once
echo 'Robot joint state after sequence:'
timeout 8 ign topic -e -t /joint_states -n 1
grep -E 'Gazebo joint-control|Open door request|Step [1-4]/4|\[SIM\]' ${launch_log} || true
echo 'MANIPULATION_DEMO_PASS'
