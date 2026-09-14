# Feedback manipulation experiment: incomplete

Baseline `a03942e` is preserved on `codex/observed-door-contract-20260914`.
Changes are isolated on `codex/feedback-manipulation-20260914`.
The full-world manipulation node and navigation/FSM policy were not modified.

## Scope

Opt-in actual-PIPER physics fixture only. Set `--feedback-contact` and
`--use-yolo-observation` in `run_physical_contact_door_test.py`.
The node rejects feedback mode outside the instrumented simulation.

- v3 handle YOLO, registered depth, real CameraInfo and TF locate the handle.
- LK image tracking can bridge a brief YOLO dropout; it is labeled separately.
- Body approach uses observed handle position and the supplied PIPER IK limits.
- Approach clearance combines observed repeatability, real gripper dimensions,
  and configured tracking/safety bounds. Repeatability is not absolute accuracy.
- Actual arm encoders/FK close the loop on position and orientation.
- Incremental press ends on measured lever rotation, not a fixed 7 cm stroke.
- Lever movement alone does not prove latch release; door motion is also required.
- During close-range visual occlusion the last stationary handle observation is
  anchored in odom. This does not solve tracking a handle after the door rotates.
- Lever and hinge feedback are instrumented Gazebo joint states, not vision,
  force/torque sensing, or a validated real-hardware sensor interface.

Configured gripper width, approach orientation, tolerances, travel bounds and
legacy chassis push/backoff settings remain. Do not describe all manipulation
parameters as automatically observed or this experiment as a full-world PASS.

## Actual results

Artifacts: `artifacts/validation/feedback_contact_20260914_r*/`.
These are different debugging revisions, not repeated identical-code trials.

- r15: contact press verified (30 mm command, 18.32 mm measured tool descent,
  6.03 degree lever motion); latch-clear push failed. Full result FAIL.
- r16: contact press verified (30 mm command, 19.75 mm measured descent,
  5.66 degree lever motion). Further feedback correction allowed door motion
  to 11.94 degrees; contact was lost and the travel limit ended the run. FAIL.
- No direct hinge commands, attach helper, extended arm links, or extra sweep
  assistance were used in these runs.
- Final changes after r16 stop immediately on contact loss during rotation,
  require fresh door feedback, and require confirmation of the final stow pose.
  These guards passed unit tests but have NOT had a new full Gazebo run.

45 unit/regression tests and a four-package colcon build passed after the final
guards. Next required work is moving-handle reobservation and contact following,
then a new fixture run. Do not integrate into worlds 1-5 as a validated success.

Evidence copied to the Windows project folder:
`00_최종자료/08_피드백기반_문조작_20260914/`.
Video right panel is a same-run detection snapshot, not synchronized live RGB.

## Reproduce

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=232 LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 src/fire_robot_bringup/scripts/run_physical_contact_door_test.py \
  --headless --use-yolo-observation --feedback-contact \
  --service-timeout-sec 420 --yolo-timeout-sec 45 --video-fps 5 \
  --output-root artifacts/validation/feedback_contact_new_run
```
