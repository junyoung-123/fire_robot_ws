# Robot-only observation experiment

## Current scope

The user has narrowed the active task to one baseline door. Mirrored fixtures
and their failures remain archived, but no mirrored rerun is requested now.
Original successful motion/source/media remain unchanged in their old workspace.
R34 passed the complete independent single-door checker. This is a separate
experimental controller, not a new World 1-5 or real-hardware success claim.

## Allowed controller inputs

- Robot arm and gripper encoders through `/joint_states`.
- Robot odometry and calibrated transforms through TF.
- Camera/depth images, intrinsics and observed YOLO handle positions.
- `/scan` from the robot's existing 2D LiDAR.
- `/observed_door/rotation` derived from RGB-D/LiDAR panel points and robot TF.

No controller subscription to `/door_joint_states`, model poses or door/lever
joint truth is allowed. The adapter also rejects raw environmental joint names,
including a mixed message containing a permitted name and a raw lever name.
The independent recorder and physical contact checker still read simulator
truth to evaluate the experiment. Simulated robot encoders are the sensor
equivalent of the encoders available on the real PIPER.

R31/R32 experiment (NOT the current door-angle input): both relative door yaw and lever roll were decomposed from measured
wrist FK after removing measured base rotation. During grasp this is
cross-checked against at least five distinct fresh RGB-D angles (disagreement
limit 0.06 rad). When the arm occludes the camera, opposed finger encoders and
the grasp-rotation residual must remain valid before FK replaces unavailable
camera rotation. No command is used as a measurement. FK also supplies only
a search-region anchor for camera reacquisition, never a synthetic plane.
Final opening must be visually reconfirmed with fresh RGB-D before release.
This remains a rigid-grasp model, not an independently measured slip sensor.
Robot TF/encoders are simulated sensor equivalents; no hardware success is claimed.

Current R34 policy: RGB-D plus the existing robot 2D LiDAR directly observe the
door panel. A visually identified plane seeds the LiDAR association; nearby
scan points must support a continuous line with adequate span and bounded
orientation uncertainty. The newest scan is preferred when its support meets
the unchanged uncertainty gate. Up to three recent scans are accumulated only
when needed, transformed at their actual timestamps; their mean time is used,
not a fabricated current timestamp. Sensor-age and quality gates are unchanged.
Wrist-derived door angle is disabled. Robot FK remains a lever/contact estimate
under a grasp assumption and a camera reacquisition ROI. Final opening still
requires fresh RGB-D reconfirmation. `camera_door_rotation` is the legacy wire
joint-name identifier; the status explicitly labels `rgbd` or `lidar`.

## Lever estimate and limitations

At an opposed-gripper contact candidate, measured FK wrist orientation defines
the relative baseline. The estimator removes measured base rotation and
RGB-D door rotation from subsequent measured wrist orientations. The residual
roll estimates lever depression under a rigid-grasp assumption. No commanded
wrist position is used as measurement. Finger aperture and rotational residual
checks reject obvious contact loss or inconsistent wrist motion.

This is a kinematic contact estimate, not a direct lever encoder, torque sensor
or proof against every possible slip. Latch clearance and opening require
observed panel motion. Independent evaluation must additionally confirm real
lever depression, contact load path, opening, release and recovery. Encoder
threshold crossing alone is not sufficient evidence of actual door opening.

Before grasp the lever estimate is unknown, not initialized to a fake measured
zero. The original early-contact guard assumed a raw lever joint sensor; the
new bounded approach uses observed panel rotation and recent handle displacement
as disturbance guards instead. The original servo steps, tracking limits and
settling criteria are retained in the adapter. This is an explicit sensor-guard
change, not a claim of completely identical code.

## Runs

- R23: mirrored run intentionally interrupted when environmental feedback was
  explicitly prohibited. It still used the old simulated lever input.
- R24: first robot-only baseline run failed with an integration exception in the
  preserved approach guard, which subtracted an unavailable lever value. No
  opening success. Raw logs retained. Fixed by the explicit observation guard.
- R25/R26: failed before grasp because partial gripper occlusion shifted the
  detected handle center. The adapter now checks the camera-to-anchor ray
  against measured FK gripper geometry, preserving the stationary anchor only
  when self-occlusion is geometrically supported. Panel-motion guard remains.
- R27: grasp/press and 17.4-degree opening succeeded, but the sequence stopped
  on a base-TF freshness check. Not full PASS. The replacement queries TF at
  one ROS-clock timestamp with a bounded stopped wait, rather than comparing
  independently received latest TF and clock messages.
- R28: the next approach had a slightly different observed target. IK stopped
  after 240 iterations at a wrist-limit boundary despite a feasible solution.
  Offline replay confirmed convergence after 318 iterations at the unchanged
  3 mm/0.10 rad tolerances and original joint limits. The bounded iteration
  budget in this experimental copy is 480; a recorded-pose regression test
  checks the original acceptance tolerances. No runtime target was added.
- R29: reached 57 degrees, then stopped because the remaining visible RGB-D
  panel patch crossed the old fixed 45 mm support-standard-deviation gate.
  It retained over 700 inliers and about 2.4 mm residual. The estimator now
  bounds orientation uncertainty using noise/support, capped effective sample
  count, a 1 mm noise floor and four spatial leave-one-region-out fits.
  Estimated uncertainty must remain below 2 degrees. This is a quality
  heuristic, not a calibrated statistical guarantee. The full physical-contact
  checker, required opening angle and input freshness limits are unchanged.
- R30: reached 62 degrees; the camera patch became too small even for the
  uncertainty gate. Kept the gate and added the explicitly permitted robot
  encoder observation path rather than accepting unreliable camera angles.
- R31: reached 117.75 degrees, visually reconfirmed opening, released the handle
  and verified fingertip retraction. Failed final stow, so NOT full PASS.
  The original raw-lever feedback loop repeatedly added wrist preload to the
  new wrist-derived lever estimate, allowing its own roll target to accumulate.
  After observed latch release, the adapter now holds that measured press
  posture plus the existing bounded preload; only door/base yaw follows live
  observations. The press loop and strict final recovery tolerances remain.
- R32: held press posture stayed bounded, but a long camera occlusion exposed
  roughly 21 degrees of wrist-yaw/door disagreement. The controller correctly
  stopped when vision returned. This falsifies the rigid-grasp assumption for
  door-angle fallback in this contact geometry. Do not describe this method
  as a generally reliable door sensor. R33 replaces it with direct LiDAR input.
  Offline scan replay: R31 RMSE 1.03 deg, R32 RMSE 1.19 deg; pose/scan pairing
  in that replay is approximate. These are feasibility results, not live PASS.
- R33: directly observed 71.6-degree opening, then stopped on sensor freshness.
  Always accumulating three scans added mean-sample latency and crossed the
  0.25-second measurement-age gate. No stale reading was accepted. R34 prefers
  a supported current scan before falling back to a timestamped accumulation.
- R34: FULL PASS for the baseline door on 2026-09-22. Primary YOLO observation,
  approach/alignment, lever press, held-handle base opening, visual confirmation,
  release and robot home all completed. At opening confirmation: observed
  117.776 degrees versus independent Gazebo reference 117.839 degrees. Maximum
  reference opening 117.879 degrees; final reference 117.864 degrees. Moving
  angle RMSE 0.343 degrees, maximum observation error 2.398 degrees. 133 LiDAR
  and 1829 RGB-D valid samples; no wrist-derived door-angle samples.
  Independent opening-contact and recovery-contact audits both passed; 4648
  finger-contact samples and no forbidden contact parts. Recovery audit is
  not comprehensive self-collision certification. The live controller AND
  angle-observer input audits passed. The 56 focused regression tests passed.
  Evidence: artifacts/validation/observed_angle_r34/
  physical_contact_door_test_20260922_193138. Failed runs remain archived.

The R34 runtime_inputs snapshot is immutable. Three new latest-scan regression
tests were added after launch; the active runtime source was unchanged during
the run. Post-run test output and the supplemental test file are separate.

Run `scripts/start_robot_observation.sh observed_angle_rNN`. Record runtime
inputs with `snapshot_observed_angle_inputs.py`, then save the live input graph
using `audit_robot_observation_graph.py`. Full PASS must come from the unchanged
independent checker, not only the ROS service response.
