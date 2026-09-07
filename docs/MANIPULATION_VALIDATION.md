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
  '{door_id: manipulation_demo, handle_position: {header: {frame_id: base_link}, point: {x: 0.52, y: -0.30, z: 0.80}}, handle_detected: false, handle_detection_method: manual_demo, handle_confidence: 0.0}'
```

## Automated headless validation

```bash
bash scripts/validate_manipulation_demo.sh
```

The test passes only when `/open_door` returns success after feedback from
`door_hinge` reaches about `+2.09` rad, 120 degrees, within the configured tolerance. It also
prints the final Gazebo robot joint state. The simplified visual IK aims the
arm at the lever near the free edge, closes the gripper, presses the lever,
executes `PUSH_OPEN`, and leaves the door open for visual inspection. The
proof-only door uses a side hinge opposite the lever.

This is a controlled integration simulation: the arm trajectory and hinged
door command are coordinated by `manipulation_node`. It validates ROS-Gazebo
command transport, joint actuation, gripper closure, door motion, feedback, and
service result handling. It is not yet a contact-dynamics proof that friction
between the gripper and handle alone causes the door motion.

## Contact-only lever press and push-open validation

The current project direction is push-open for the real demonstration because
the mobile manipulator can keep the door moving away from the robot instead of
backing up while holding the handle. The contact-only test uses a robot-facing
lever handle near the free side of the door. The manipulation sub-FSM presses
the lever down first, then pushes the door open:

`PRE_GRASP -> GRASP_HANDLE -> PRESS_HANDLE -> PUSH_OPEN -> RETURN_HOME`

It does not publish a Gazebo hinge command; the door must move from arm/gripper
contact.

```bash
python3 src/fire_robot_bringup/scripts/run_physical_contact_door_test.py \
  --min-angle-rad 0.75
```

Latest lever contact-only result: `2026-08-22`, PASS.

- Motion: `push`
- Handle target in `base_link`: `x=0.43, y=0.24, z=0.83`
- Gazebo door hinge command topic: not used
- Max hinge motion: `0.8387 rad` (about 48 degrees)
- Evidence: `C:\Users\황준영\Documents\졸업작품\검증결과\physical_contact_door_test_20260822_110835`

Pure friction contact still does not prove a full 120-degree opening. Without
an explicit grasp constraint, the latest true 120-degree contact-only check
reached about `0.8391 rad` maximum and then lost contact.

## Lever grasp constraint and coordinated push-follow

For a realistic "grip lever, press, then push while following the hinge arc"
visualization, the physical test can enable Gazebo's DetachableJoint system.
This attaches the simulated gripper finger to the door panel after
`GRASP_HANDLE`; it approximates a maintained lever grasp. It is not a pure
friction-only proof, but it validates the higher-level arm/base sequence needed
for a real PIPER attempt.

```bash
python3 src/fire_robot_bringup/scripts/run_physical_contact_door_test.py \
  --min-angle-rad 2.09 \
  --attach-grasp-joint
```

Latest attach-assisted full-open result: `2026-08-22`, PASS.

- Sequence: `PRE_GRASP -> GRASP_HANDLE -> PRESS_HANDLE -> PUSH_OPEN`
- Gazebo door hinge command topic: not used
- Grasp constraint: `/fire_robot/sim_arm/handle_attach`
- Final hinge angle: `2.1287 rad`
- Max hinge motion: `2.2187 rad` (over 120 degrees)
- Evidence: `C:\Users\황준영\Documents\졸업작품\검증결과\physical_contact_door_test_20260822_112407`

## Gazebo visual proof capture

```bash
bash scripts/capture_manipulation_visual_proof.sh
```

`manipulation_demo.world` includes a fixed Gazebo overhead camera on
`/proof/overhead/image`. The capture script launches headless Gazebo, calls
`/open_door`, records the proof camera frames, and writes both visual and
numeric evidence. The run passes only when the same capture sees
`door_hinge` move from closed to about `+2.09` rad, 120 degrees, and `/open_door` returns
success.

Latest corrected visual proof: `2026-09-07`, PASS. The hinge is fixed at one
side of the panel, the lever is near the opposite free edge, and positive
rotation pushes the free edge away from the robot. Final hinge feedback was
`2.096245 rad` (`120.106 degrees`) with `service_success=True`.

Latest visual evidence is collected with the full-world proof images:

- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\all_worlds_validation_summary.png`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world*\trajectory.png`
- `C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world*\door_alignment.png`

## Full FSM integration check

Latest check: `2026-08-24`

```bash
TRACE_DIR=artifacts/validation/full_evidence_20260823/proof_traces/world1 \
  bash scripts/run_headless_validation_once.sh \
  corridor.world \
  artifacts/validation/full_evidence_20260823/proof_traces/world1.log \
  3000
```

Latest multi-world result:

| World | Scenario | Blue doors opened | Exit basis | Result |
| --- | --- | ---: | --- | --- |
| 1 | corridor, 3 blue / red mix | 3/3 | observed green | PASS |
| 2 | alternate layout, 3 blue | 3/3 | observed green | PASS |
| 3 | 4 blue / 2 red | 4/4 | observed green | PASS |
| 4 | no blue doors | 0/0 | delayed no-blue fallback | PASS |
| 5 | all six side doors blue | 6/6 | observed green | PASS |

World 4 is intentionally the no-blue scenario. In the latest run the FSM did
not open any door, completed the delayed no-blue exploration path, and then
passed the exit goal.

Latest evidence:

`C:\Users\황준영\Documents\졸업작품\검증결과_20260823\all_worlds_validation_summary.png`

Per-world proof images and logs:

`C:\Users\황준영\Documents\졸업작품\검증결과_20260823\world*`
