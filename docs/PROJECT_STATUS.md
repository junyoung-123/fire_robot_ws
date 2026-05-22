# Project Status

Date: 2026-05-22

## Current State

The project is now in a ROS2-buildable and full-simulation-testable state.

Core status:

| Area | Status |
| --- | --- |
| ROS2 Humble build | Ready |
| Simulation launch | Full Gazebo FSM flow passes in WSL headless mode; GUI is default, `headless:=true` is available for tests |
| Nav2/SLAM launch | Works in simulation; waits for valid TF/odom when no real base is connected |
| MoveIt launch | Starts without the previous gripper KDL and SRDF link errors |
| PIPER driver launch | Uses AgileX `piper_single_ctrl`; waits for real `can0` hardware |
| Door detection | Runs with HSV fallback; YOLO model is not trained yet |
| Sensor fusion | Runs in LiDAR-only fallback if optional ML dependencies are missing |
| Full simulation mission | Passed: red detection -> blue door navigation -> simulated opening -> green exit -> mission complete |
| Full real-robot mission | Not verified yet |

## Package Map

| Package | Role |
| --- | --- |
| `fire_robot_bringup` | Main launch files for simulation and real robot |
| `fire_robot_description` | URDF/Xacro robot model and ros2_control config |
| `fire_robot_fsm` | Mission state machine |
| `fire_robot_interfaces` | Shared messages and services |
| `fire_robot_manipulation` | MoveIt/PIPER door-opening logic |
| `fire_robot_navigation` | Nav2 navigation wrapper and Nav2 params |
| `fire_robot_perception` | Door detection and sensor fusion |

## Recommended Test Path

Use an ASCII-only WSL path for ROS2 builds. Building directly under a Windows-mounted Korean path can trigger ROSIDL path parsing issues.

```bash
rsync -a \
  --exclude build \
  --exclude install \
  --exclude log \
  --exclude 'src/fire_robot_perception/datasets' \
  '/mnt/c/Users/황준영/Desktop/졸업작품/project/fire_robot_ws/' \
  ~/fire_robot_ws_test/
```

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Run the validation script:

```bash
./scripts/validate_ros2_workspace.sh --skip-rosdep
```

Run simulation in WSL:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
ros2 launch fire_robot_bringup simulation.launch.py use_rviz:=false headless:=true
```

Expected full simulation result:

```text
State: IDLE -> EXPLORING
State: EXPLORING -> NAVIGATING
State: NAVIGATING -> OPENING_DOOR
State: OPENING_DOOR -> DOOR_OPENED
State: DOOR_OPENED -> EXPLORING
State: EXPLORING -> EXITING
State: EXITING -> MISSION_COMPLETE
```

For visual Gazebo inspection, remove `headless:=true`:

```bash
ros2 launch fire_robot_bringup simulation.launch.py use_rviz:=false
```

Run real robot launch with all hardware disabled:

```bash
ros2 launch fire_robot_bringup real_robot.launch.py \
  enable_camera:=false \
  enable_radar:=false \
  enable_base:=false \
  enable_piper:=false \
  enable_moveit:=false \
  enable_slam:=true \
  enable_nav2:=true \
  enable_app_nodes:=false
```

## Real PIPER Checklist

Before testing in the lab:

1. Connect the PIPER CAN interface.
2. Confirm CAN device exists.

```bash
ip link show can0
```

3. If needed, configure CAN.

```bash
sudo ip link set can0 up type can bitrate 1000000
```

4. Start PIPER-only smoke test.

```bash
ros2 launch fire_robot_bringup real_robot.launch.py \
  enable_camera:=false \
  enable_radar:=false \
  enable_base:=false \
  enable_moveit:=false \
  enable_slam:=false \
  enable_nav2:=false \
  enable_app_nodes:=false \
  enable_piper:=true \
  piper_can_port:=can0
```

## Known Limitations

- The trained YOLO model is not available yet. Door detection currently falls back to HSV color detection.
- Real robot motion was not tested because the robot is in the lab.
- The actual base driver is still project-specific. Nav2 needs a valid `/odom` topic and TF chain.
- WSL Gazebo may require software rendering environment variables.
- Gazebo GUI requires a working display. Use `headless:=true` only for non-visual smoke tests.
- RealSense and radar hardware topics need lab verification.
- The simulation currently opens one blue door before timing out and moving to the green exit. This is acceptable for the current corridor world but can be expanded when more door scenarios are added.
