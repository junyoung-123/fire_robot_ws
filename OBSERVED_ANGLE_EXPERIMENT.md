# Observed Door Rotation Experiment, 2026-09-22

This is a separate Git worktree and branch. The prior successful source and
evidence remain unchanged. Base commit: 65f4363. The corridor fixture and passive
media recorder were copied from fire_robot_ws_corridor_demo_20260922.

## Scope

Current status at 2026-09-22 17:02 KST: R18 was intentionally stopped during
opening to review scope after the user requested preserving the prior successful
motion and replacing only its angle input. R18 has no completed PASS/FAIL result;
the new wrist-retreat path has unit coverage but has NOT completed simulation.
The mirrored fixture has NOT been run. R13/R16/R17 opened but failed recovery.
The earlier `fire_robot_ws_corridor_demo_20260922` run `corridor_demo_r1`
has an independent full PASS (maximum 2.0581491249 rad) using simulator angle.
Neither that source nor its evidence is replaced by this experiment.

- Door rotation is measured from registered RGB-D points near the observed YOLO
  handle and a fitted vertical panel plane in odom. The first stable plane is
  the rotation baseline. No door hinge/world pose or model geometry enters it.
- The original contact controller is inherited, with its door angle joint name
  replaced by camera_door_rotation. Incoming door_hinge is ignored. Missing or
  stale observation cannot fall back to simulator door angle.
- Arm and lever measurements still use simulated encoders. This is NOT an
  all-sensors-real deployment, torque sensing, or observed lever-angle result.
- The original test recorder independently logs Gazebo door_hinge for reference
  and success/error auditing only. Those reference values do not feed control.
- Plane orientation is inherently ambiguous by 180 degrees. Continuous fresh
  observations unwrap it; large unobserved turns are rejected, not guessed.
- The baseline assumes the door is initially closed. Absolute open angle of an
  initially ajar door is not identified by this experiment.

The new estimator must be validated before any success claim or replacement
of the preserved presentation evidence.

## Sensor timing

R6 retained valid plane estimates but the 5 Hz simulated depth sensor delivered
two consecutive observations 1.03 real seconds apart on the loaded laptop.
The unchanged one-second controller freshness guard stopped movement. From R7,
only this worktree's depth sensor runs at 10 Hz. No guard, target angle, contact
criterion, or controller motion policy is relaxed. This is not a measured
hardware camera rate guarantee.

## Observation and posture corrections, R8 onward

- New handles still require confidence 0.45. A previously confirmed, fresh,
  odom-consistent 3D track may continue at confidence 0.20 after three consistent
  measurements. Low-confidence detections cannot acquire a new target.
- The previous measured plane is an additional RANSAC hypothesis, not a held
  measurement. Current depth points must support every published estimate;
  residual, point count, spatial support and temporal continuity gates remain.
- R10 reached 51.36 degrees but intermittent plane-fit rejection exceeded the
  original one-second feedback guard.
- R11 reached 101.80 degrees with moving-angle RMSE 0.030 degrees against the
  independent reference. It FAILED when a compensated wrist command exceeded
  a calibrated joint limit. Contact audit passed, but this is not full success.
- R12's headroom-only steering FAILED at 48.46 degrees: turning farther toward
  the door consumed lateral arm workspace. This heuristic is no longer active.
- R13 instead evaluates bounded base probes with the calibrated PIPER IK,
  bias-compensated joint limits, view angle and tool posture. It opened to
  118.04 degrees and released the handle, but FAILED during fixed-wrist arm-only
  retraction. The base travelled 0.952 m during held opening; the arm did not
  perform the entire opening from a stationary chassis.
- R14 prechecks the released-hand IK path. If it cannot retract the arm safely,
  it holds the measured joint posture and reverses the base only after both
  fingers are confirmed open. Measured FK+odom displacement, not elapsed time,
  decides fingertip clearance. Missing feedback, stale odometry, excessive
  travel and timeout stop the motion. A real deployment still needs rear-space
  collision checking; this fixture is also checked by independent contact audit.

Mechanical limits, contact checks, target (2.05 rad) and stale-feedback stop are
not relaxed. Short probe velocities and safety margins are controller settings,
not known door coordinates, hinge radii or commanded opening-angle trajectories.

R14 stopped at 59.73 degrees although all final plane fits remained valid. The
latest measurement was only 0.143 ROS seconds old, but one real second had passed
on the slow simulator. R15 separates the clocks: camera measurement age is now
limited to 0.25 ROS seconds, clock stalls to one real second, and camera transport
silence to three real seconds in simulation only (one on hardware time). This is
an explicit timing-policy change, not an unchanged one-second camera receipt gate.
Arm/lever receipt limits remain unchanged. Six unit tests cover stale timestamps,
slow simulation, frozen/rewound clocks, future stamps and transport silence.

## Opposite-handed fixture

R16 opened past 117 degrees and released the handle, but FAILED recovery:
the fingertip-only retreat still left contact with the lever when shoulder
folding began. Independent audit detected finger/panel and camera/panel contacts.
R17 reserves the full calibrated gripper length plus measured grasp width and
two tool tolerances before folding. This is a conservative robot/observed-object
clearance, not a known door coordinate. The 0.30 m base travel cap remains.

R17's longer base withdrawal also FAILED: reverse motion contacted the door
strike/frame, then folding recontacted the panel. That withdrawal implementation
is removed from the active controller. R18 instead plans a released-hand retreat
with bounded wrist-yaw relaxation toward the robot forward axis. Every 6 mm step
is checked by calibrated IK, joint margin (0.025 rad), joint continuity (0.15 rad)
and wrist turn (0.06 rad per step). Only already-open fingers may use it. The
base stays stopped. Loaded compensation is reset and unloaded encoder tracking
re-estimates servo bias. Final measured hand displacement decides clearance.
This local kinematic check is not full mesh collision planning; the unchanged
independent contact audit remains mandatory.

R15 reached 107.07 degrees (angle RMSE 0.032 degrees) but FAILED after an
uncaught base-TF freshness exception terminated the controller. R16 retries
the original TF validity check for at most 0.5 real seconds while commanding
zero base velocity. No stale/future transform is accepted. Persistent failure
returns a stopped service failure instead of crashing the ROS node. Three
additional tests cover transient delay, bounded failure and exception handling.

`prepare_mirrored_contact_world.py` reflects the door and corridor across y=0.
The calibrated robot is NOT reflected. Both door and lever scalar joint limits
are reversed, retaining the same x/z axes, so their measured signed rotations
are genuinely opposite. The opening threshold and independent checker are the
same for both cases. The mirrored fixture uses `CONTACT_LEVER_PRESS_SIGN=1.0`
and `CONTACT_INITIAL_YAW=0.12` with
`CONTACT_FIXTURE_WORLD=physical_contact_door_mirrored.world`.

Lever depression direction is configured for the fixture, not automatically
classified from the image. Door angle and its sign are measured from depth.
These two independent single-door cases are not a two-door navigation mission.

The active view helper reports a conservative plane through the measured tool.
This is not the recessed panel surface, so a negative tool-plane clearance is
not a direct collision measurement. A correction in that state must improve
the conservative estimate; the independent Gazebo contact audit still rejects
robot/panel load paths. This helper is not full swept-volume collision checking.

## Evidence interpretation

## Preserved-motion verification (R19 onward)

`minimal_observed_angle_node` is the active controller for these runs. Its parent
`preserved_contact_policy.py` is an exact copy of the successful corridor run's
controller (SHA256 ea9a63ec1d7c26fff740d62208abf9e0d9ec01cb0c83b2ee530a9c185b85ac0a).
Approach, lever press, contact following, base steering, release and recovery
methods are inherited without overrides. The expanded experimental motion
changes described above are NOT active in these runs.

The adapter replaces the door-angle input, camera freshness checking and event
labels. Depth acquisition is 10 Hz rather than the original 5 Hz. R19 (original
handedness) and R20 (mirrored) both stopped BEFORE arm approach because final
YOLO confidence was below the unchanged 0.45 confirmation threshold.

Offline inference on the actual final camera frames showed confidence 0.289 /
0.327 at inference size 640, versus 0.769 / 0.793 at native image width 320.
R21 onward therefore use handle inference size 320. This is an additional
perception configuration change, not an unchanged-input replay. The weights,
confidence gate, required fresh confirmations, motion policy, physical model
and contact checker are unchanged. The raw failed runs remain preserved.

Motion equivalence is source-code equivalence, not proof of identical physical
trajectories: changed measured inputs and timing can change a feedback policy's
trajectory and outcome. Every direction must pass the complete checker anew.

The perspective evidence camera follows the hand. Its apparent angle is not
the hinge angle. Use the fixed overhead view and timestamped comparison plot
to assess opening rotation. Both are evidence only, never control inputs.
Raw Gazebo frames are captured at about 3 frames per simulation second. Encoded
videos repeat frames to 12 fps, with explicit 1x or 4x simulation-time playback;
they are not real-time desktop screen recordings or generated motion.
