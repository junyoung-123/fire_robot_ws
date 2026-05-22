# ROS2 Test Result

Date: 2026-05-22  
Environment: WSL2 Ubuntu 22.04, ROS 2 Humble  
Workspace used for ROS2 test: `~/fire_robot_ws_test`

## Summary

The project builds and launches in a ROS 2 Humble environment. Core launch files, MoveIt configuration, Nav2/SLAM startup, Gazebo bridge startup, the AgileX PIPER driver path, and the full Gazebo FSM simulation flow were verified without connecting the real robot.

The real PIPER arm and mobile base were not available, so CAN communication, physical motion, camera/radar hardware, and final door-opening behavior must still be verified in the lab.

## Test Results

| Test | Result | Notes |
| --- | --- | --- |
| Python syntax check | PASS | 21 Python files parsed, 0 errors |
| ROS2 workspace build | PASS | 15 packages built with `colcon build --symlink-install` |
| `simulation.launch.py --show-args` | PASS | Launch file is discoverable and arguments parse correctly |
| `real_robot.launch.py --show-args` | PASS | Hardware enable/disable arguments parse correctly |
| PIPER package discovery | PASS | `piper_single_ctrl` and `piper_read_slave_joint` found |
| PIPER no-hardware smoke test | EXPECTED FAIL | Driver starts, then fails because `can0` is not connected |
| Nav2/SLAM no-base smoke test | EXPECTED WAIT | Nav2 starts, then waits for missing `odom` TF |
| Gazebo simulation smoke test | PASS WITH WSL NOTE | ROS nodes and Gazebo bridges start; software rendering is recommended in WSL |
| Full Gazebo FSM simulation | PASS | Red door detection, blue door navigation, simulated door opening, green exit navigation, and `MISSION_COMPLETE` verified |
| MoveIt smoke log check | PASS | No gripper KDL chain error, no missing `lidar_link` warning, no traceback |
| Git whitespace check | PASS | `git diff --check` returned no issues |

## Commands Used

```bash
cd ~/fire_robot_ws_test
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

```bash
source install/setup.bash
ros2 launch fire_robot_bringup simulation.launch.py --show-args
ros2 launch fire_robot_bringup real_robot.launch.py --show-args
ros2 pkg executables piper
```

PIPER smoke test without the robot:

```bash
timeout 8s ros2 launch fire_robot_bringup real_robot.launch.py \
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

Expected result without hardware:

```text
CAN socket can0 does not exist.
```

Nav2/SLAM smoke test without base odometry:

```bash
timeout 12s ros2 launch fire_robot_bringup real_robot.launch.py \
  enable_camera:=false \
  enable_radar:=false \
  enable_base:=false \
  enable_moveit:=false \
  enable_slam:=true \
  enable_nav2:=true \
  enable_app_nodes:=false \
  enable_piper:=false
```

Expected result without base/odom:

```text
Timed out waiting for transform from base_link to odom
```

Gazebo simulation smoke test in WSL:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
timeout 35s ros2 launch fire_robot_bringup simulation.launch.py use_rviz:=false headless:=true
```

Observed important startup messages:

```text
Creating GZ->ROS Bridge: [/scan ... -> /scan ...]
Creating ROS->GZ Bridge: [/cmd_vel ... -> /cmd_vel ...]
Creating GZ->ROS Bridge: [/odom ... -> /odom ...]
SensorFusionNode started | depth=OFF
DoorDetectionNode started | YOLO=FALLBACK_HSV | depth=OFF
ManipulationNode started [SIMULATION]
StateMachineNode started
NavigationNode started
```

Full FSM simulation in WSL:

```bash
LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
timeout -k 10s 260s ros2 launch fire_robot_bringup simulation.launch.py \
  use_rviz:=false \
  headless:=true
```

Observed mission flow:

```text
Fire detected (1 red door(s)). Starting mission.
State: IDLE -> EXPLORING
Navigation goal received: door door_blue_...
Navigation goal accepted by Nav2
Goal succeeded
State: NAVIGATING -> OPENING_DOOR
Open door request: id=door_blue_...
Door open succeeded
State: DOOR_OPENED -> EXPLORING
Green exit detected
State: EXPLORING -> EXITING
Navigation goal received: door door_green_...
Goal succeeded
State: EXITING -> MISSION_COMPLETE
Mission complete
```

## Fixed During Testing

- Corrected ROS2 Python package install locations by adding `setup.cfg` files.
- Fixed MoveIt launch parameter loading so non-ROS YAML files are passed correctly.
- Installed SRDF files with `fire_robot_manipulation`.
- Changed simulation Gazebo command to `ign gazebo` for ROS2 Humble/Fortress.
- Added launch switches to `real_robot.launch.py` so hardware can be disabled for tests.
- Replaced the stale PIPER driver entry with AgileX `piper` package executable `piper_single_ctrl`.
- Updated Nav2 BT plugin names for installed ROS2 Humble packages.
- Fixed red-door/fire position handling to preserve the source TF frame.
- Removed invalid gripper KDL IK setup and corrected SRDF `lidar_link` references to `radar_link`.
- Added a `headless` Gazebo launch option. GUI is the default; WSL smoke tests can use `headless:=true`.
- Fixed the real-robot MoveIt pose target link from a generated non-existent link name to `arm_link6`.
- Tuned simulation-only door approach distance so Nav2 goals are placed inside the explored map.
- Changed the global costmap to a rolling window so the long corridor simulation can navigate to the green exit.

## Remaining Verification

These items require the lab robot or trained model assets:

- PIPER CAN connection through `can0`.
- Actual PIPER arm movement and gripper open/close behavior.
- Mobile base odometry and `odom -> base_link` TF.
- Real camera/radar topics and calibration.
- YOLO door detector trained `best.pt`.
- Visual Gazebo GUI inspection. Run without `headless:=true` when a GUI display is available.
