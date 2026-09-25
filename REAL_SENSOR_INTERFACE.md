# Door-angle observation without Gazebo

The angle estimator has no Gazebo message dependency, hinge-state subscription,
world-file reader, model-name lookup, or simulator fallback. A real RGB-D camera
and calibrated robot TF can provide its inputs. This is an interface contract,
not a claim that the complete contact controller is ready for real hardware.

## Standalone measurement node

After building this isolated workspace and sourcing its install:

```bash
ros2 run fire_robot_manipulation observed_door_angle_node \
  --ros-args -p use_sim_time:=false
```

Required inputs:

| Input | Type / interpretation |
| --- | --- |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo`, RGB intrinsics |
| `/camera/color/image_raw` | `sensor_msgs/Image`, calibrated optical frame |
| `/camera/depth/image_rect_raw` | Depth registered to the same RGB intrinsics and pixel dimensions; `32FC1` metres or `16UC1` millimetres |
| `/detected_door` | `fire_robot_interfaces/DoorInfo`, primary YOLO handle or its visual track, with stamped 3D handle observation |
| TF | Timestamped transform from the sensor/handle frame to `odom`; real robot odometry, not a door-model pose |

ROS topic remappings may adapt camera-driver topic names. Equal image dimensions
alone do not prove registration: extrinsics and pixel alignment must be calibrated.
Input timestamps must use one clock. No current-pose substitution is made when a
measurement's exact TF is missing.

Outputs:

- `/observed_door/rotation`: `sensor_msgs/JointState`, the only joint name is
  `camera_door_rotation`; position is signed rotation in radians from the initial
  stable plane, expressed after odometry motion compensation.
- `/observed_door/status`: JSON in `std_msgs/String`, validity, rejection reason,
  plane normal, residual, support, source timestamp, and angle when valid.

For a calibrated depth pixel (u,v,z), the sensor computes
`p_camera = [(u-cx)*z/fx, (v-cy)*z/fy, z]`, then applies timestamped TF to odom.
A measured plane normal gives `atan2(n_y,n_x)`. Continuous sign unwrapping and
subtraction of the initial stable plane produce door rotation. Thus robot yaw
is compensated before angle measurement, rather than being confused with it.

## Assumptions and limits

- One initially closed, approximately vertical, planar door is selected through
  its observed handle. The first stable plane establishes zero. This does not
  infer the absolute hinge angle of an initially ajar door.
- Fresh plane observations are followed continuously. A gap can make the
  plane's 180-degree sign ambiguity unresolvable; the estimator rejects a large
  unobserved change rather than guessing.
- Plane support and discontinuity gates are retained. From R15, camera freshness
  uses measurement age <=0.25 ROS seconds. A stalled clock is rejected after
  one real second. Camera receipt watchdog is three real seconds only with
  use_sim_time=true, one second on hardware time. Arm/lever receipt limits stay
  at one real second. Slow simulated time is not confused with stale images.
- Camera/robot calibration, depth failures on reflective surfaces, occlusion,
  odometry drift, and multi-door target reset are not validated on hardware.
- The experimental controller adapter still inherits the simulation actuator
  interface. Arm/gripper encoders and especially the lever encoder must be
  replaced by genuine hardware feedback before real operation. A standard door
  does not provide the simulated lever encoder.
- No torque measurement is implemented or claimed here. The new angle signal
  is geometric RGB-D measurement, not inferred force.

Gazebo hinge angle is permitted only in the independent recorder/checker. It
must not be remapped into `camera_door_rotation` or used as a fallback.
