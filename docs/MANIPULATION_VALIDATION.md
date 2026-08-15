# Gazebo Robot-arm Door-opening Validation

This test is separate from the navigation/FSM world validation. It moves the
PIPER joints and gripper in Gazebo and opens a dynamic hinged door.

## Interactive run

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch fire_robot_bringup manipulation_demo.launch.py
```

The launch opens both Gazebo and an arm-focused RViz2 view by default. For an
RViz2-only visualization window backed by headless Gazebo, use:

```bash
ros2 launch fire_robot_bringup manipulation_demo.launch.py headless:=true use_rviz:=true
```

RViz2 shows the robot model, every arm TF, and dedicated axes for `arm_link6`
and `gripper_base_link`. Expand `PIPER Robot Model` in the Displays panel to
inspect individual links. The hinged SDF door remains a Gazebo-only visual.

Call the service from another sourced terminal:

```bash
ros2 service call /open_door fire_robot_interfaces/srv/OpenDoor \
  '{door_id: demo, handle_position: {header: {frame_id: base_link}, point: {x: 0.60, y: 0.0, z: 0.80}}}'
```

## Automated headless validation

```bash
bash scripts/validate_manipulation_demo.sh
```

The test passes only when `/open_door` returns success after feedback from
`door_hinge` reaches 1.15 rad within the configured tolerance. It also prints
the final Gazebo robot joint state. Expected pull-pose values are approximately
`joint1=0.48`, `joint2=0.72`, `joint3=-0.58`, and both gripper joints `0.006` m.

This is a controlled integration simulation: the arm trajectory and hinged
door command are coordinated by `manipulation_node`. It validates ROS-Gazebo
command transport, joint actuation, gripper closure, door motion, feedback, and
service result handling. It is not yet a contact-dynamics proof that friction
between the gripper and handle alone causes the door motion.
