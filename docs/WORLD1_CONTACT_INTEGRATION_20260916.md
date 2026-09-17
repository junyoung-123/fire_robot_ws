# World 1 contact integration: experimental copy

## 2026-09-17 continuation: navigation preserved, one door isolated

The user requested an updated report and a single physical door demonstration,
not another rerun or replacement of the World 1-5 navigation baseline. The
original workspace remains clean at d1ffb39. REV4 and original navigation
artifacts remain untouched. REV5 separates their navigation scope from this
still-experimental manipulation backend.

| Trial | Full result | New measured evidence |
| --- | --- | --- |
| r32 | FAIL, 56.14 deg | 3578 finger/lever samples, no sampled panel contact. Measured-tip-based press correction added but not exercised in this run. Full motion deadline. |
| r33 | FAIL, 55.43 deg | New fixed strike sensor confirms left front camera against latch strike. Final 15 wall seconds: base moved 0.09 mm while wheels rotated. |
| r34 | FAIL, 49.75 deg | 3412 finger/lever samples, no sampled body/panel/strike contact in held phase. Stopped at unchanged original joint limits. Both-axis following and simultaneous arm/base probes exercised. |
| r35 | FAIL, 64.30 deg | Bounded reference and reach-error-based base speed. 3582 finger/lever samples, no forbidden sampled body/door/frame contact. Full simulation-time deadline. |
| r36 | FAIL, 41.24 deg | Fast local IK and reachable-posture selection exercised. 658 finger/lever samples, no forbidden sampled contact. A 9.8 mm requested downward correction moved the measured tool only about 0.4 mm and did not depress the lever. |
| r37 | Full FAIL; held opening reached 117.62 deg | 2392 finger/lever contact samples; held-phase audit passed. Fixed 180 mm released-hand retraction selected an unreachable folded IK branch. |
| r38 | Full FAIL; held opening reached, peak 128.92 deg | Measured-grip-derived 79.2 mm retraction succeeded (81.1 mm measured). Subsequent legacy shoulder/elbow folding hit the panel (119 sampled contacts) and did not converge. Peak includes unintended recovery motion, NOT extra held opening. |
| r39 | Motion PASS, overall FAIL (evidence availability) | Held opening 117.73 deg, release/stow/backoff completed. No sampled forbidden contacts. Native Gazebo contact topics emit only on contact; zero panel messages incorrectly failed the old availability gate. Original result retained. |
| r40 | PASS, 117.76 deg peak | Held opening 117.72 deg, release/stow/backoff completed. 4654 finger/lever contact records. Opening and recovery audits PASS; continuous contact heartbeat PASS. 28.56 cm reverse measured after stow. |

r35 separates opening speed from base speed using local differential-drive
kinematics. The preferred forward posture is the observed pre-grasp pose;
lateral posture is constrained by observed panel normal and calibrated body
outline, not a preset world coordinate or known hinge radius. It does not
certify clearance from arbitrary jambs: passive jamb/strike contact records
remain part of the independent audit. Robot collision geometry is unchanged.

r36 adds a local-seed IK fast path only for contact updates.
A valid solution must meet position/orientation tolerance and original joint
limits; global seeds remain available if the local solution fails. Bounded
probe duration is checked against the actual endpoint, not a pessimistic sum
of opposing base/arm velocities. No angle or safety gate is relaxed.

Current code tests: 212 passed; all eight colcon packages built. This includes calibration-envelope checks
against xacro and STL collision geometry, post-safety /cmd_vel_safe stop
recording, local IK validity, recovery path FK regression, camera-command
allowlisting and camera orientation. This alone is not a physical full-opening PASS.

r38 releases only after measured full opening. It freezes the measured wrist
orientation after release, estimates retreat from calibrated finger extent plus
half the measured grip span and the unchanged 15 mm tool tolerance, and moves
in small Cartesian steps. This assumes a round lever section; it is not a
general perception-based estimate of arbitrary handle depth.

r39 corrects the supplied-chain shoulder direction during stow, without changing
joint limits or tolerances. Recovery contacts are checked separately from
handle-held opening. The external proof cameras can move only two named,
collision-free camera models; original on-robot RGB-D sensors remain unchanged.
Video frames are wall-clock-throttled to 2 fps. Video duration is not guaranteed
to equal simulator time; timestamped keyframes and joint traces remain available.

### Final isolated result: r40

`contact_copy_r40_handle_held/physical_contact_door_test_20260917_032440`
is a single-fixture physical contact PASS, not a five-world physical mission.
The r39/r40 controller, perception, robot and navigation hashes match (62 files).
The world XML also matches except for the passive evidence plugin. No threshold,
force limit, robot dimension or contact material was changed between these runs.

The new `fire_robot_sim_evidence` plugin reads existing ContactSensorData through
a const ECS interface. It aggregates contact pairs each physics step and sends
20 Hz messages including empty heartbeats. It has no physical-command interface.
The checker rejects missing streams or gaps exceeding 0.5 simulation seconds.
r40 received 4080 messages on each of five channels; maximum stored gap 0.15 s.
Source reference: https://raw.githubusercontent.com/gazebosim/gz-sim/ign-gazebo6/src/systems/contact/Contact.cc

The final reverse command still uses configured velocity and simulation-time
duration; it is not distance-feedback control. Independent saved-odom analysis
measured -0.2856 m in the starting body-forward direction over 2.856 sim seconds.
Final actual-joint maximum stow error was 0.0193 rad (0.070 rad criterion).
The high-friction fixture and lack of real force sensing remain explicit limits.

Build under a Linux-only PATH when WSL inherits Windows Anaconda CMake packages:

```bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers 1
source install/setup.bash
python3 -m pytest -q scripts
```

The first r40 snapshot omitted the new C++ suffix. Preserve it, but use
`source_snapshot_with_evidence_plugin` for the complete 173-input snapshot.
Reports and compatible videos are in the Windows `14_...` evidence directory;
REV4 and original navigation results are unchanged. Original workspace remains
clean at d1ffb39. Test processes were stopped after recording.

The fixture has explicit assumptions, including a 52 mm lever with contact
friction mu=5.0 and an 18 kg panel. These are not measured real-door material
properties. Neither self-collision coverage of the entire robot nor real force
control is certified. The report must retain these limits even after a pass.

Windows evidence/report work is under
`00_최종자료/14_주행보존_단일문개방_20260917`.
Each simulation has its own source archive and result JSON; preserve failures.

## Historical progress through r31

## Corrected user intent: handle-held base motion

The user's clarification supersedes the bumper proposal below. Keep the lever
depressed and the hand on the handle while the mobile base supplies opening
motion. Do NOT release early and push the panel with the body. No bumper is
being added. r23/r24 remain historical demonstrations of the WRONG opening
sequence and must not be presented as completion of the clarified request.

The new fixture setting is `hold_handle_during_base_open=true`, with
`sim_release_handle_before_base_push=false`. It retains the grasp after latch
clearance, follows measured lever/door/base motion, and releases only after
the full angle is measured. Differential-drive motion and a rotating hinge
do not generally permit rigidly freezing every arm joint for the entire sweep;
the hand contact is maintained while bounded arm corrections prevent twisting.

New sequence tests enforce release ordering and failure behavior. The contact
checker requires finger/lever evidence during HANDLE_HELD_BASE_OPEN and rejects
direct panel contact by ANY robot part in that phase. This remains instrumented
Gazebo feedback, not real force sensing. See new r25+ results, not r23/r24, for
validation of this mode. Before r28, 180 unit tests passed; no full contact PASS.

| Handle-held trial | Result | Evidence / next correction |
| --- | --- | --- |
| r25 | FAIL, 60.72 deg peak | 1023 sampled finger/lever contacts; no recorded robot/panel contacts during held opening. Small-angle contact integration rejected its operating range. |
| r26 | FAIL, 102.77 deg peak | Midpoint-tangent projection removes the angle singularity without a known hinge coordinate/radius. Stopped at original PIPER joint limits. Right camera/panel contact also failed the load-path audit. |
| r27 | FAIL, 46.23 deg peak | Following both contact-reference coordinates unloaded the forward pushing servo. Door motion stalled and the joint-limit guard stopped correction. 1166 finger/lever contact samples, no sampled robot/panel contact; still NOT a full PASS. The forward-X change was reverted. |
| r28 | FAIL before grasp, 0 deg | Increased observed approach conditioning to 0.30 and changed steering toward the panel exterior. Approach timed out; last condition ratio 0.293. Opening service was not called. |
| r29 | FAIL before grasp, 0 deg | A 90-second simulation-time approach budget did not help: the near-camera view lost usable handle observations before the 0.30 conditioning target. Opening service was not called. |
| r30 | FAIL, 59.32 deg peak | Approach condition 0.25 reached a live observed handle. Opening stopped at the unchanged 40 mm arm tracking guard. Finger/panel contacts were also recorded, so the load-path audit failed. |
| r31 | FAIL, 53.07 deg peak | Held-arm settling executed twice. 898 finger/lever contact samples; no sampled robot/panel contact during held opening. Lever retention correction produced no measured lever response and stopped. No release/home/full-angle PASS. |

Latest engineering checks: all seven packages built (8.18 s); 184 script tests
passed (5.11 s); `git diff --check` clean. Preserved workspace still clean at
`d1ffb39c4a6d92868e851af0415c4c04e710ed5f`. These are NOT mission PASS results.

The remaining control issue is sustained handle contact with enough whole-body
workspace throughout a large hinge rotation. Neither a larger tracking limit
nor an early handle release qualifies as a fix. Current frozen forward reach
and lateral following are experimental; they are not a validated whole-body
planner. Next work should distinguish measured grasp slip from servo/workspace
constraints, using the saved contact and joint records before another trial.

Only sampled contact evidence is claimed, not a measured grasp force. The
opening criterion remains 2.05 rad, with release and home feedback required.

## Latest status after r24

**NOT an overall safe-contact PASS.** r23 and r24 completed observed-handle
grasp, lever press, release, stow, approximately 118-degree physical opening,
and retreat. However, the recorded panel contacts identify the protruding
right camera housing, not the chassis bumper, as the pushing surface. The old
angle/FSM-only checker reports PASS; the new independent load-path audit FAILS.
Original `result.json` files remain unchanged, with a separate
`contact_safety_audit.json` and `opening_evidence.png` explaining the distinction.
Do not present either run as a hardware-ready or safely executed full opening.

The copied arm dimensions and joint limits were not enlarged. No hinge command,
grasp attachment, or hidden kinematic sweep was used. No protective bumper has
been added: that would be an explicitly labelled hardware-layout assumption,
not a software-only fix or an as-built robot measurement.

## Preserved baseline

- Source workspace: `/home/junyoung/fire_robot_ws_test`.
- Preserved branch/commit: `codex/feedback-resume-20260914`, `d1ffb39`.
- Separate working copy: `/home/junyoung/fire_robot_ws_world1_contact_20260916`.
- Experimental branch: `codex/world1-contact-integration-20260916`.
- Previous report and evidence stay unchanged. No result below is an old PASS.

## Gates

1. Actual-PIPER contact fixture: observed handle, lever press, physical full opening.
2. Navigation-to-manipulation handoff, reachable near approach, stow and departure.
3. World 1 full mission using contact opening, not commanded hinge rotation.
4. Separate sensor-based mapping/localization test, not saved map/sim pose proof.
5. Repeated trials and sensor/contact failure recovery with explicit outcomes.

All gates are pending. A code build or unit test is not a Gazebo mission PASS.
Only the single-door contact fixture has run in this copy so far, not World 1.

## First diagnosis

The r21 contact records include the left finger contacting `latch_bolt_collision`
and continuous panel-floor contacts. The edge bolt previously protruded in front
of the door skin into the gripper corridor. The copied fixture now confines the
bolt inside the leaf thickness, preserving its strike-facing edge and strike
geometry. A 10 mm floor gap removes an unintended floor contact. These are
fixture changes, not a validated control fix. Arm lengths, hinge command policy,
friction coefficients, and the 2.05 rad opening criterion are unchanged.

## Changes limited to the copy

- Separate simulation-time motion deadlines from stalled-clock and absolute
  wall-time limits. A slow simulation must not be mistaken for motion failure.
- Keep the YOLO CPU thread cap after Ultralytics lazily initializes its predictor.
  The installed library reset a requested two threads to eight. A real-weight
  smoke test reproduced that reset and confirmed that the wrapper restores two.
- Process TF/depth callbacks concurrently with one serialized image inference
  stream. Keep only the latest pending image; discard busy image callbacks.
  RGB/depth timestamp and freshness checks are not relaxed.
- Add stopped-base, bounded corrections when the measured lever starts relaxing.
  Each retention correction requires incremental lever response. Door coasting
  alone cannot confirm that a finger still presses the handle. Lost contact,
  stale feedback, tracking error and the press travel limit still stop motion.
- Add selectable wheel odometry to the launch bridge. Ground-truth remains the
  default for compatibility, and this fixture still uses it. This wiring is NOT
  evidence that sensor-only World 1 mapping/localization passed.
- Save source hashes, a diff and source archive for each trial. Model files are
  hashed, not copied into each archive.

## Trial ledger

All paths below are relative to `artifacts/validation` in this copy.

| Trial | Outcome | Actual finding |
| --- | --- | --- |
| `contact_copy_r1` | SETUP ERROR | DDS domain 233 outside usable port range; no mission ran. |
| `contact_copy_r2` | FAIL | Live primary YOLO/depth; fixed 180 wall-second pre-grasp deadline expired. |
| `contact_copy_r3` | FAIL | The base approach still had a separate fixed wall timeout. |
| `contact_copy_r4` | FAIL | No consistent live handle at pre-grasp; CPU oversubscription and stale depth. |
| `contact_copy_r5` | FAIL | YOLO bbox detected, but RGB/depth timestamps diverged; coordinates correctly rejected. |
| `contact_copy_r6` | FAIL | Live YOLO/depth, grasp and press verified; 0.209 rad / 12.0 deg maximum opening, then lever contact lost. |
| `contact_copy_r7` | FAIL | Retention corrections produced lever response, then no further response; safe stop at 0.167 rad / 9.6 deg maximum opening. |

For r6, the first verified press commanded 0.030 m and measured 0.0153 m tool
descent with 0.0988 rad lever motion. These are distinct quantities. The test
never commanded the hinge, and did not achieve the required 2.05 rad opening.
The contact log reports collision pairs but no force samples; do not label it
force-controlled grasping or a successful full physical opening.

The r7 result is not an improvement in maximum opening angle over r6. It proves
that the new retention branch is exercised and stops when a correction does not
produce measured lever motion. It does not prove that grasp retention is solved.
The next diagnostic target is moving-handle contact geometry and tool tracking,
not lower success thresholds or further blind press travel.

## Verification and evidence

- All seven packages built after creating the copy. The three changed packages
  rebuilt successfully after the edits.
- `python3 -m pytest -q scripts`: **122 passed**, 8.84 seconds.
- `git diff --check`: clean.
- Preserved workspace `git status --short`: clean, still on its original branch.
- No Gazebo/ROS validation processes remained after r7.
- r6 and r7 images, videos, result JSON, raw logs and matching source snapshots
  are also copied into Windows `00_최종자료/11_분리복사본_물리접촉실험_20260916`.
- This branch is experimental and uncommitted; no merge or GitHub push was made.

## Remaining boundaries

- The feedback controller uses instrumented simulated joint states. No physical
  robot force/lever sensor adapter is validated.
- The test still has configured safety bounds, low-speed probing, a configured
  chassis follow radius, retraction distance and joint-space stow waypoints.
  Observed handle coordinates and measured press endpoints do not mean every
  operation is parameter-free or fully adaptive.
- World 1 has not been converted and validated with a lever/latch on every
  relevant door. Its prior commanded-hinge PASS must remain separate.
- Navigation handoff, sensor-only SLAM/localization, full mission and repeated
  failure recovery are pending. Do not merge this copy as a proven replacement.

## Reproduce the contact trial

```bash
cd /home/junyoung/fire_robot_ws_world1_contact_20260916
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=224 IGN_PARTITION=world1_contact_copy_20260916
export LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 scripts/snapshot_validation_source.py artifacts/validation/NEW_RUN/source_snapshot
python3 src/fire_robot_bringup/scripts/run_physical_contact_door_test.py \
  --headless --use-yolo-observation --feedback-contact \
  --service-timeout-sec 720 --yolo-timeout-sec 45 --video-fps 2 \
  --summary-only --output-root artifacts/validation/NEW_RUN
```

Use a new output directory for every trial. Run only one Gazebo validation at a
time and check that its owned processes exit before starting another one.

## Single-door continuation (r8 onward)

The older results above remain historical evidence, not the latest status.
All contact trials through r14 FAILED the unchanged 2.05 rad full-opening gate.

| Trial | Maximum opening | Finding |
| --- | --- | --- |
| r8 | 1.21 deg | Occluded optical flow drifted away from the measured tool. Reject it instead of moving to that coordinate. |
| r9 | 1.13 deg | The short base-motion timer included expensive IK planning. Plan stopped, then time the commanded motion. |
| r10 | 1.07 deg | Replacing a loaded servo target with measured FK unloaded the lever. Retain the bounded setpoint. |
| r11 | 4.60 deg | Repeatedly using measured lateral tool position accumulated servo error. |
| r12 | 8.26 deg | An odometry-anchored planar sweep stopped the lateral drift, but lever retention still failed. |
| r13 | 15.11 deg | Latch-clear gate reached 14.42 deg; gripper release failed. This is a partial stage result, NOT full opening. |
| r14 | 8.17 deg | Latch-clear did not repeat; the press correction produced insufficient measured response. |
| r15 | 1.03 deg | Both fingers still contacted the lever, but a correction produced only 0.00226 rad lever motion. Full sequence FAIL. |
| r16 | 0 deg | Three new detections arrived approximately 1.6 wall-seconds apart; the oldest exceeded a 3 wall-second filter before three could be accepted. Failed before grasp; no test of wrist-following motion. |
| r17 | 7.78 deg | Observation refinement passed. Wrist tilt permitted continued lever-held motion, but a retention correction failed. Its initial response sample already differed by -0.01656 rad because the baseline preceded IK planning. |
| r18 | 8.08 deg | Dispatch-time comparison removed that timing error, but measured lever response still failed; this was not only a bookkeeping bug. |
| r19 | 1.52 deg | Correcting the 5-degree tool-normal offset did not resolve contact. A live encoder sample showed the lower finger near its free-space equilibrium rather than supporting the handle. |
| r20 | 0.42 deg | Force-capped full closure produced two opposed finger candidates (19.16 / 32.58 mm), then the press target exceeded the unchanged 40 mm tracking limit. |

Additional safeguards now reject stale contact TF, require lever response after
hold correction, and let an actually moving servo settle before declaring a
correction unresponsive. The moving-door wrist follows measured hinge/base yaw.
This remains instrumented simulation feedback, not a real force-sensor result.

### Free-space gripper diagnosis

`gripper_servo_before` and `gripper_servo_after` are actuator calibration runs,
NOT door-opening or autonomous-navigation evidence. Their fixed robot-local
pose is deliberately clear of the door. Each uses the unchanged PIPER geometry,
35 mm per-finger stroke and 8 N simulated controller force cap.

| Setting | Right-finger opening | Error from 35 mm | Existing 34 mm release gate |
| --- | --- | --- | --- |
| P=80, D=0.8 | 32.011 mm | 2.989 mm | FAIL even without a handle |
| P=400, D=6.3 | 34.395 mm | 0.605 mm | PASS in free space |

The direction and size of the old steady error agree with finger gravity sag
(25 g finger, approximately mg/P). These are simulator servo gains, not measured
real PIPER gains. No force limit, arm size, release threshold, door-angle gate or
physics bypass was changed. A contact trial must still verify actual release.
The regression suite after this edit reports 141 passed; `git diff --check` is clean.

Through r23 the fixture hinge post had visual geometry but no collision geometry.
r24 adds matching visual/collision geometry outside the closed leaf and its
opening sweep. This corrects the fixture; it does not resolve sensor contact.

### Changes after r15/r16

- Wrist orientation now includes measured lever rotation about the panel-normal
  axis, not only door/base yaw. Opposing finger contacts in r15 are consistent
  with the old level wrist resisting the tilted lever; this is the hypothesis
  being tested, not a confirmed full-opening fix.
- Refinement requires three distinct, nonzero exposure timestamps from the last
  3 ROS seconds. It separately rejects a stream whose newest receipt is over
  3 wall seconds old. Duplicate images and freshly delivered old images cannot
  qualify. The 3-sample gate, spatial bounds and source requirements remain.
- r17 starts with 149 passing unit tests and a clean whitespace check. Its exact
  source fingerprint is in `contact_copy_r17/source_snapshot`.
- r18 starts with 151 passing tests. Press-response baselines are captured at
  command dispatch after IK, excluding motion before the correction existed.
  The minimum measured response, maximum press travel and stop-on-stale rules
  are unchanged. Contact commands now log desired/measured wrist rotation to
  distinguish control error from physical blocking.
- r19 corrects the desired approach orientation, not the robot geometry. The
  FK of the actual zero-joint pose tilts the tool approximately five degrees;
  using that pose as the desired normal contradicted the base +X approach.
- r20 replaces a preset 45 mm grip width with closing under the same 8 N force
  cap, accepting only stable, nonzero positions of BOTH fingers. This nominates
  opposed contact but does not measure force or certify a successful grasp.
  The independent lever-motion check remains mandatory.
- r21 adds a bounded 0.035 rad wrist lead relative to measured lever rotation.
  Holding exactly the current lever angle can resist a rotary press after a
  firm grasp. This is a small control increment, not a fixed final press angle.
  The press-travel cap and 40 mm tracking stop remain. 157 unit tests passed
  before r21. Full physical opening remains unproven until the actual trial passes.

### r21-r24: workspace readiness and stricter evidence

| Trial | Peak / final opening | Old motion gate | Additional finding |
| --- | --- | --- | --- |
| r21 | 0.96 deg peak | FAIL | Holding a bounded wrist lead did not retain lever response. |
| r22 | 6.22 deg peak | FAIL | Retaining peak measured wrist depression still failed in a stretched pose. |
| r23 | 118.09 / 118.09 deg | PASS | Full phase sequence, but panel pushed by right camera housing. Safe-contact audit FAIL. |
| r24 | 118.48 / 118.48 deg | PASS | With physical hinge post and bounded next press command; same camera contact. Safe-contact audit FAIL. |

r23 replaces first-reachable parking with live observed handle IK plus a minimum
translational Jacobian singular-value ratio of 0.15. The accepted ratios were
0.1882 and 0.1655 in r23/r24. The measured handle x coordinates at parking were
0.7772 and 0.7962 m in base_link; they are results, not preset door targets.
First verified press in r24 commanded 20 mm and measured 13.12 mm tool descent
with 0.1046 rad lever motion. The subsequent motion is recorded separately.

r24 bounds the NEXT press setpoint by the existing 40 mm tracking budget. It
also restores physical collision geometry to the hinge post without overlap
with the door leaf through the required opening sweep.

The new checker independently audits panel collision pairs during BASE_PUSH_OPEN.
Camera, LiDAR, radar and arm/gripper contacts cannot qualify as chassis pushing;
missing or unknown contact evidence fails closed. Fixed-joint lumping is handled
so a name beginning with base_footprint cannot hide a camera collision.

Verification: 167 script tests passed; all seven colcon packages built. These
are regression/build results, not full mission proof. Old original-workspace
results and reports were not changed. Current experimental edits remain local.
Next step: decide an acceptable physical pushing surface or revise the maneuver.
Do not remove sensor collision geometry or simply count the angle as success.
