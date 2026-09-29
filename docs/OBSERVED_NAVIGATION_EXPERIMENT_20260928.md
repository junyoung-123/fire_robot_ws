# Empty-map navigation experiment (2026-09-28)

Latest evaluated release: **r29, 2026-09-29, all five worlds PASS_NAV_VISITS_EXIT**.
See [release scope](RELEASE_20260929.md) and [published results](validation/2026-09-29/README.md).
This is navigation and door visits only, not physical opening. The sections below preserve
the chronological investigation, including failed and superseded runs.

Experimental branch: `codex/observed-nav-20260928`, baseline `27359f7`.
The original checkout and all earlier PASS evidence are preserved.

## Input contract

- Keep every physical obstacle and door in the original world files unchanged.
- Do not load saved occupancy maps or prebuilt SLAM pose graphs.
- Use differential-drive wheel odometry, not Gazebo pose odometry, for ROS `/odom` and `/tf`.
- Run online `slam_toolbox` on the robot's scan and wheel TF.
- Feed the live `/map` to Nav2. Unknown is not free; planning through unknown is disabled.
- Do not run the saved-map latch, fixed obstacle overlay, or world-reading manipulation node.
- Door identities, colors and target coordinates come from camera / LiDAR observations.
- No expected door count is supplied to the FSM. World geometry may be used only by an
  independent post-run evaluator, never a ROS control node.

## Scope and limits

This is a **navigation-only** test. `navigation_visit_node` replaces the manipulation
service at the existing FSM boundary. Its success means the robot reached a blue
door with the FSM's fine-alignment checks and the adapter's proximity/bearing check.
It does not turn a hinge, move an arm, or claim a door physically opened. The legacy
FSM still names its completed-target state `DOOR_OPENED`; in these runs it must be
read as **VISITED**, not used as physical opening evidence.

Robot calibration, model weights and policy tolerances remain configured constants.
The legacy height-based visual range fallback also remains; an assumed panel height
is not a measured depth. Geometric camera/LiDAR association is still imperfect.
This is not a claim that every heuristic in the legacy corridor policy was removed.
Observation noise, partial-map axis estimation, SLAM drift and persistent door-memory
consistency need verification under this new localization condition.

`MISSION_COMPLETE` by itself is not PASS. Check unique blue visits, wrong-color visits,
alignment, obstacle clearance, map growth, exit crossing and the input contract.
Do not replace the earlier presentation or change its claims before this audit passes.

## Reproduction

Build a separate WSL workspace using `scripts/setup_observed_navigation_workspace.sh`.
Run `scripts/launch_observed_navigation_trial.sh --world WORLD.world --output NEW_DIR`.
The runner uses its own DDS domain and Gazebo transport partition. Cleanup terminates
only process groups it created. Evidence includes sensor maps, camera frames, poses,
observed targets, visits, graph audit, complete logs and the original world hash.

The optional contact-audit C++ package is not needed for navigation and is excluded
from this build. The first all-package build exposed Windows Anaconda protobuf headers
on the WSL search path; no global system package was modified to work around it.

## Findings during the first pilots

- r1: PCA of an incomplete initial map produced about -9 degrees of corridor-axis
  error. The experimental profile now estimates the dominant observed wall-line
  direction and waits for robot TF before publishing its initial origin.
- r2: a vertically clipped close-up door bbox was treated as a full-height door.
  Its invalid height-based range displaced a near target. The experimental profile
  now uses only measured range for vertically clipped bboxes, or rejects them if
  measured range is unavailable.
- r3: independent simulator-pose evaluation found localization drift above 2 m and
  100 degrees after turning. One internal visit was NOT accepted by the independent
  alignment checker. It is not a PASS result.
- r4: faster matching / a wider angular search alone still allowed transient heading
  errors. The [upstream Humble implementation](https://github.com/SteveMacenski/slam_toolbox/blob/humble/src/slam_toolbox_common.cpp)
  pre-filters scans by translation in `shouldProcessScan`, ignoring heading in that
  pre-filter. See also [upstream issue 807](https://github.com/SteveMacenski/slam_toolbox/issues/807).
  The next trial sets `minimum_travel_distance=0` with a 0.25 s time interval so
  in-place rotations can reach scan matching. This remains an experiment, not a
  validated fix until evaluated in Gazebo.

## Evaluated r5 result

World 1 r5 completed online SLAM navigation in 1,014 s elapsed (including cleanup).
The independent navigation checker accepted 3 distinct blue-door visits, no red-door
visit, no duplicate visit, exit-position crossing and no overlap of the configured
navigation footprint with the 12 named obstacle boxes. Door counts and SDF geometry
were read only by that post-run checker. No saved map nodes were present; the sole
subscriber to simulator ground-truth odometry was the passive evidence recorder.

This is `PASS_NAV_VISITS_EXIT`, **not full manipulation PASS**. Panel-center distances
at visits were 0.94 / 0.71 / 1.53 m; wall-normal yaw errors were 10.8 / 11.8 / 6.5 deg.
The last distance is not PIPER-reach validation. Maximum initial-pose-aligned SLAM
position error was 0.89 m and heading error 9.07 deg. A temporary duplicate candidate
caused an unnecessary return before being rejected. These remain quality limitations.

The checker thresholds are longitudinal error <=0.70 m, distance <=1.60 m and
wall-normal yaw error <=15 deg. Exit crossing uses the recorded simulator position,
not a mission-state message alone. The existing green exit is a visual marker with
no panel collision, not a physically opened exit door. Obstacle checking is a 2D
footprint-box test at recorded poses, not instrumented physical contact detection.

Evidence scripts produce sensor-map progress, raw camera views, independent world
trajectory, visit-distance diagrams and localization-error plots. Every view states
whether it is sensor data or evaluator-only data. Four synthetic checker regression
tests separately cover acceptance, wrong-color rejection, obstacle-overlap rejection
and ground-truth subscriber isolation; they are not simulation success evidence.

World 2 r5 was stopped after its third internal visit: the independent position put
the robot too far from that panel. The legacy parking helper clamped its goal to
an initial-axis lateral lane (0.95 m) even when the observed SLAM wall was farther
away. r6 opts out of that lane clamp and uses observed wall lateral minus the
configured standoff instead. Three new tests cover both wall sides, translated
observations and unchanged legacy default behavior. This does not increase the
checker's permitted distance or alter any world geometry.

World 2 r5 also overlapped an obstacle during the legacy manual "return to initial
center" recovery. r6 disables that direct recovery, enables LiDAR safety for manual
commands and stops those commands when scans are missing/stale. r6 independently
accepted all three visits, with no named-obstacle footprint overlap, but was stopped
at the exit: the final crossing code replaced the observed lateral position with
the initial centerline. LiDAR safety correctly stopped the robot near the end wall.
This is a FAIL, not a completed mission.

r7 adds an opt-in `exit_use_observed_lateral` setting. Exit approach, direct-crossing
fallback and completion lateral checks use the observed exit instead of the initial
centerline. The legacy behavior remains the default outside this experiment. Four
tests cover both sides, translated/rotated observations, direct crossing and legacy
compatibility. No checker thresholds or world geometry were changed.

`scripts/validate_observed_navigation_suite.sh ROOT REV 2 3 4 5 1` builds no new map
prior: it starts a fresh SLAM process for each trial, evaluates the completed run,
and stops at the first independent failure. Each revision writes new result folders.

## Pause checkpoint: World 2 only

The user requested stopping after World 2. No World 3/4/5 fresh-SLAM run was started.
The original checkout is still unchanged. World 1 r5 remains an earlier navigation
PASS, not a same-revision five-world result.

- World 2 r7: three independently valid visits, no named-obstacle footprint overlap;
  exit crossing failed. Removing the centerline bias alone was insufficient.
- World 2 r8: three independently valid visits (1.20 / 1.20 / 1.02 m from panel
  centers, 13.0 / 12.6 / 12.1 degree wall-normal errors), no named-obstacle footprint
  overlap. Nav2 crossing was enabled; its final goal was outside the observed map
  and was correctly rejected. This is a FAIL, not complete navigation success.
- r9 code adds a Nav2 approach stage before planning the final crossing. Six focused
  exit tests and the seven-package build passed. The World 2 r9 run was interrupted
  at the pause checkpoint; it is not end-to-end validated and must not be called PASS.

### Remaining exit-fixture inconsistency

The unchanged World 2 `exit_green` model has a 0.06 x 0.9 x 2.0 m visual panel, but
no collision geometry and no opening actuator. The robot's sensor is `gpu_lidar`.
[Gazebo's GPU lidar measures visual geometry](https://gazebosim.org/api/sensors/9/classgz_1_1sensors_1_1GpuLidarSensor.html),
not only physical collision shapes. The recorded SLAM map therefore shows the panel
as an occupied barrier although the chassis could physically pass through it in a
blind direct-drive test. Removing a collision element did not make it a sensor-open
exit. The existing small gaps beside that panel are not a usable robot passage.

Do not hide this by disabling safety, clearing the green panel from the costmap,
injecting world coordinates, enabling planning through all unknown space, or lowering
the exit checker. A future test needs either a genuinely open passage with a visible
green sign, or an observable exit-opening action. That changes the fixture/scope and
must be documented separately; it was NOT silently applied to these world files.
The earlier direct-drive PASS must not be presented as evidence of sensor-safe
crossing of a physically closed exit door.

## Resume: approved open-exit fixture (2026-09-29)

The user approved changing the exit into a genuinely open passage with a green
marker, while retaining interior obstacles. `make_open_exit_world.py` now creates
an explicit per-run SDF copy: the exit panel becomes two green jambs and a lintel
with matching visual/collision geometry; its floating handle is removed. All other
model XML and the original world file are checked to remain unchanged. No fixture
coordinates/dimensions are sent to the navigation, perception or FSM nodes.

Pass `--open-exit` after the revision in the suite command to enable this fixture.
The manifest records both original and derived world hashes and the changed model
names. Independent evaluation also checks the added exit jamb footprints. This is
an open-passage navigation test, NOT physical opening of a closed green exit.
Results are separated under `observed_nav_results_20260929`; earlier trials remain.

### r10 diagnosis and r11 changes

r10 independently accepted all three World 2 visits without obstacle/jamb footprint
overlap, but failed exit crossing. It was stopped for diagnosis, NOT passed. The
finite-return SLAM grid stopped near the end wall; the locked green observation was
also a jamb rather than the aperture center. Repeated attempts accumulated substantial
SLAM drift (maximum 3.01 m / 28.08 degrees in the retained evaluation).

r11 keeps the raw SLAM input unchanged and uses Nav2's standard rolling global
costmap plus `inf_is_valid` ray clearing. Positive infinity means a valid no-return
measurement for this Gazebo LiDAR; NaN is not treated as free space. Unknown planning
remains forbidden. The window size is a planning buffer, not a supplied world map.
The persistent static layer still reads only the live `/map` from SLAM.

An opt-in exit helper associates a green observation with two finite LiDAR edges
surrounding a positive-infinity beam interval. The measured midpoint and wall normal
provide approach/crossing poses. No SDF, exit dimensions, fixed centerline or truth
pose is used by the helper. Unbounded, too-narrow, unrelated or invalid-data gaps
are rejected. These geometric tolerances are robot-clearance/association parameters,
not measured environment coordinates. Four focused tests cover those cases.

References checked for this change: Humble Karto `OccupancyGrid::AddScan` skips
infinite/out-of-range returns; Humble Nav2 `ObstacleLayer::laserScanValidInfCallback`
supports positive-infinity clearing. This does not imply sensor no-return is always
safe in real smoke/glass/drop-off environments; that hardware behavior is untested.

### r11/r12 failures and r13 alignment correction

r11 crossed the open exit but failed independent alignment at the third blue door
(20.48 deg vs the unchanged 15 deg limit). It remains a FAIL. r12 added an opt-in
local wall-normal fit from finite LiDAR points around the observed door position.
Two visits met independent alignment (5.77 / 8.42 deg), but the third candidate was
abandoned after repeated alignment failures. Despite `MISSION_COMPLETE`, r12 is FAIL.

r13 separates temporary normal-estimation unavailability from a navigation-position
failure: position errors stay available for the Nav2 handoff; final alignment stops
and waits for fresh range support, with a bounded retry. This fixes the r12 regression
where an absent fit produced a missing pose error and consumed approach retries.
The observed-normal profile also disables the legacy relaxed 16-degree settling
acceptance. No independent checker limit is relaxed. Finite scan snapshots now retain
their measured timestamp and sensor-to-map TF for later diagnosis.

### r13 World 2 PASS / World 3 failure; r14 mapping isolation

World 2 r13 passed all three independent blue visits and open-exit crossing,
with no tested obstacle/jamb footprint overlap. World 3 r13 failed: blue3 was
missed, another blue door was visited twice, and exit crossing was not completed.
The retained map pose jumped backwards 2.58 m at sim191, then 1.77 m at sim432.5
and 2.11 m at sim443.5. These jumps exceed the local matcher's search window and
are consistent with false place-recognition constraints in the repeated corridor.
Stored semantic locations consequently no longer agreed with current observations.

r14 disables automatic loop closure only in the experimental online-mapping
profile. Wheel odometry and local LiDAR scan matching remain enabled; every map
is still built from scratch. This avoids unverified long-range graph corrections
but does not solve arbitrary large-loop drift or prove general SLAM robustness.
All worlds must be rechecked on this configuration. No visit, collision or exit
evaluation limits have been relaxed.

### r14 results and r15 observed-exit completion

World 3 r14 passed four unique visits and exit crossing. The extended independent
checker also found no mobile-base overlap with low wall/panel collision boxes.
World 4 r14 failed even though the FSM announced completion: the true final x was
14.91 m, before the actual exit at 17.92 m. A stale green projection was locked as
a crossing goal, without any LiDAR aperture having been confirmed.

r15 keeps fresh green surface observations separate from their stand-off goals,
and removes the initial-centerline preference for green observations in this
profile. Until the bounded LiDAR opening is measured, green is only an approach
hint. Completion now requires the robot center to be at least 0.41 m beyond the
measured plane, inside the measured width with 0.28 m lateral body clearance.
These are robot footprint margins, not supplied exit coordinates. A rear view
cannot reverse the saved outward normal. The physical world and independent
acceptance thresholds remain unchanged. All worlds require fresh r15 runs.

### r15 infrastructure failure and r16 stationary-scan gate

World 4 r15 never moved: Nav2's lifecycle configure response timed out and its
navigation action server did not become available. This trial remains FAIL, not
a test of the new exit behavior. While stationary, repeated noisy scan matching
still changed the SLAM pose, reaching 2.50 m / 33.59 degrees of spurious drift.

r16 adds a wheel-odometry motion gate before SLAM only. It retains original scan
timestamps/ranges, passes in-place turns (0.035 rad) or translation (0.03 m), and
sends three startup scans before rejecting motionless duplicates. Safety/Nav2 and
perception continue receiving unfiltered `/scan`. The mapping pre-gate remains
zero to avoid Humble's translation-only rejection of in-place rotations. Four
focused tests check stationary, rotation-only, accumulation and angle wrapping.
The runner now fails a missing Nav2 action server after 180 wall seconds instead
of spending the whole mission budget retrying unavailable navigation. No source
of simulator position is introduced into this filter.

r16 exposed a TF-lifetime integration issue: Humble's default `restamp_tf=false`
keeps map-to-odom stamped at the last admitted scan. Once stationary scans are
gated, that timestamp expires from the odometry TF buffer, blocking Nav2 startup.
r17 sets the supported `restamp_tf=true`: the latest estimated map-to-odom
correction is republished at current time between scans. Raw scan timestamps and
all measured ranges remain untouched. r16 is retained as a failed integration
trial; it is not counted as successful navigation.

r17 then exposed startup scan loss before the TF cache was ready. r18 waits for
an actual SLAM OccupancyGrid before enabling stationary gating, instead of assuming
three scans suffice. Before that acknowledgement it forwards scans at at most 4 Hz.
The added regression covers repeated startup scans, map acknowledgement and later
rotation-only updates. The runner also detects missing initial SLAM output within
180 seconds. This is an initialization handshake, not a fabricated map or pose.

r18 confirmed that the installed binary is slam_toolbox 2.6.10 and does NOT
implement `restamp_tf` (unlike the current upstream Humble branch). r19 disables
its built-in TF publication and adds a sole map-to-odom relay from `/pose` (SLAM
estimate) and stamped odom-to-base TF (wheel integration). The SE(2) correction is
held between scans, while odom continues tracking motion. Scan messages are never
restamped. Stale wheel TF, or motion without a new SLAM correction for 2 seconds,
stops publication. There is no simulator-pose subscription in the relay. Initial
pose, pure translation, rotation and wraparound reconstruction are unit tested.
Upstream version-specific source: https://github.com/SteveMacenski/slam_toolbox/blob/2.6.10/src/slam_toolbox_common.cpp

### r19 outcome and r20 occluded wall association

World 4 r19 passed the independent visit/exit audit. World 5 r19 abandoned its
first blue target after repeated wall-normal rejection; the stopped trial is
retained as FAIL. Recorded scans show a supported door-wall line behind nearby
clutter. Counting foreground returns against wall support rejected this line.

r20 evaluates each fitted line's span, residual, anchor distance and incidence
before selection. Foreground points in front of that line are treated as
occlusions, not contrary wall evidence. Minimum support, line quality and visual
anchor association limits remain enforced. Added tests cover dense occlusion and
an incorrect visual depth that must still be rejected. No world coordinates,
obstacle removal or acceptance-threshold relaxation are involved.

World 5 r20 passed: 6/6 independent blue visits, exit crossing and no tested
footprint overlap with 12 obstacles, 2 exit jambs or 17 wall/panel boxes. Maximum
recorded localization error was 0.722 m / 11.09 degrees. This is navigation-only,
not a physical arm opening test, and its 938-second run includes a slow
reapproach. Worlds 1-4 need separate runs of this same source before suite PASS.

### r21 startup handshake and r22 close-station refresh

World 1 r20 exposed a startup race: the map arrived while the first SLAM pose
could not yet be matched to buffered wheel TF. Motion gating then stopped new
stationary scans, leaving map-to-odom unavailable. r21 waits for BOTH a map and
the SLAM-derived map-to-odom transform before gating stationary scans. A unit
test covers map-before-transform arrival. The runner also reports missing
map-to-base transforms as an infrastructure timeout.

Worlds 1 and 2 r21 passed, with maximum recorded localization errors of
0.394 m / 6.09 degrees and 0.665 m / 5.75 degrees respectively. World 2 spent
additional time rejecting an imprecise candidate; no extra visit was accepted.
World 3 r21 failed the independent audit at its third door (longitudinal error
0.772 m against the unchanged 0.70 m limit). The trial was stopped and retained.

The final close-memory correction previously accepted only `observed_blue_`
IDs, excluding detector-origin IDs. In the observed-navigation profile r22
applies the same correction to either origin and allows measured shifts in
either longitudinal direction. The original profile is unchanged. Freshness,
confidence, same-wall association, bounded correction and one-refresh limit
remain in place. Regression tests cover detector IDs and both shift signs.
The independent evaluator now explicitly serializes NumPy booleans on failures.

World 3 r22 passed all 4 visits and exit crossing without tested footprint
overlap. Its third-door longitudinal error was 0.284 m in this new run. This is
not a controlled identical-trajectory A/B test; other runs of the final source
are still required for the complete suite report.

### r23 coupled velocity safety

World 4 r22 stalled near an obstacle during repeated rotation/recovery attempts.
No sampled navigation-footprint overlap was found, but the mission did not finish;
the stopped trial is retained as FAIL. Review also identified that independently
clamping angular velocity could turn a commanded avoidance arc into an unchecked
straight motion while leaving its linear component active.

r23 enables `coupled_collision_stop` only in the experimental profile. If either
component of a commanded twist is blocked by a LiDAR safety check, both components
stop. A subsequent independently safe reverse command remains allowed. The original
profile retains its prior default. Scan thresholds, the world obstacle geometry and
the independent acceptance criteria are unchanged. Six safety tests cover blocked
turns, blocked forward motion and available rear-clear escape.

Worlds 4, 1 and 2 r23 have passed their independent evaluations. Worlds 3 and 5
remain pending at this checkpoint; these three results alone are not a suite PASS.
World 2 includes repeated short exploration/reapproach motions, so successful
completion must not be described as an optimized or consistently smooth trajectory.

### r24 checked in-place turn after a translation stop

World 3 r23 passed four visits and exit crossing. World 5 r23 stalled before
its first visit: front LiDAR range remained about 0.40 m, below the 0.48 m
translation stop, while fine alignment requested an arc with a 22-degree
position-heading error. r23 also blocked its angular component despite adequate
surround clearance. Repeated approach timeouts returned to the same pose. The
run was stopped and retained as FAIL, not silently retried as a success.

r24 keeps translation stopped, but permits the requested in-place rotation ONLY
when the separate all-direction LiDAR rotation check is enabled and passed.
If rotation is blocked, both components still stop so that no unsafe straight
motion remains. Missing/stale scans stop both. None of the front, rear, rotation
or evaluator thresholds changes. Added tests cover both rotation signs, a front
stop with a blocked side, and an absent rotation check. All five worlds require
new runs of this revised source before a same-version suite PASS can be claimed.

### r25 harness outage classification (r24 control unchanged)

World 5 r24 was stopped by the harness at 180 wall seconds. The recorded estimated
pose stream has a 26-sim-second gap from t=81 to t=107, but had already recovered
and continued through t=119. The latest 15-second graph audit still described the
earlier outage. The old runner interpreted ANY negative TF audit after 180 seconds
as a startup failure, even if the node had initialized and subsequently recovered.

r25 retains r24 control unchanged. The harness now remembers successful startup,
requires 60 consecutive wall seconds of missing runtime readiness before stopping,
and still fails never-ready components after 180 seconds. Three regression tests
cover startup failure, transient recovery and persistent runtime failure. Audit
snapshots are retained in the event stream; the summary reports the maximum gap
in estimated poses. The transient TF issue is NOT claimed to be solved by this
test-harness correction. All failure evidence and independent pass limits remain.

### r26 align before the navigation-visit service, no threshold relaxation

Worlds 5, 4 and 1 r25 passed (6/6, 0/0 and 3/3 visits respectively). World 2
r25 failed its first visit twice: the observed handle bearing was -0.360 and
about -0.349 rad, outside the unchanged 20-degree service limit. The FSM's
close-position exception had accepted a 9-degree wall-normal error against a
relaxed 10-degree limit, despite a stricter configured alignment tolerance.
Combining residual lateral parking error and yaw error violated the service
precondition; repeated service rejection exhausted the target's attempt budget.

r26 removes this relaxed close-yaw exception only for the observed-normal profile.
It configures 4 degrees plus the existing 1-degree control margin, so every ready
branch in this profile uses 5 degrees. The 20-degree visit-service condition and
all independent evaluator limits remain unchanged. A regression verifies that
a 9-degree close pose continues alignment, a 3-degree pose is ready, and the
legacy non-observed profile retains its prior behavior. The r25 failure is kept.
# r27: preserve measured handle against panel-centre fallback

World 2 r26 passed all three visits and the exit. World 3 r26 was stopped
for diagnosis after repeated close-station corrections (one valid visit, not a
completed run). Logs alternated a detected YOLO handle near the panel edge and
an estimated panel-centre handle. Both referred to the same door but generated
different parking stations and restarted alignment repeatedly.

Only in the observed-normal profile, a trusted observed handle is no longer
replaced by an estimated panel centre or a final wall-memory fallback. New
trusted observations can still refine it; normal observation freshness checks
and all independent evaluation limits remain unchanged. Legacy profiles retain
their previous refinement behavior. A regression test covers both source types
and the legacy profile. All five worlds must be rerun with the final source.
# r28: finish the current world review before advancing

World 3 r27 failed with a SLAM transform runtime timeout after three valid
visits. The retained trace also showed repeated returns to the old projection
of an already visited candidate. This is not a completed world result.

The follow-up fixes are grouped before another run:

- A selected observation-memory object is explicitly completed on success,
  even when its refined position moved outside the old proximity window.
  Nearby unrelated clusters are not suppressed and merge radii are unchanged.
- Subsequent trusted-handle corrections during fine alignment require three
  time-separated observations agreeing within 0.15 m over a two-second window.
  The initial upgrade from an estimated panel centre remains available.
- Motion-gated SLAM checks live map-to-odom timestamp freshness on every scan,
  not a latched startup flag. When that TF expires, original scans resume at
  up to 4 Hz while stationary so localization has input with which to recover.
  No simulator pose, stale TF republishing, or fabricated scan data is added.

Tests cover localization recovery while stationary, stale/rapid/inconsistent
handle measurements, and completion of the exact selected memory without
excluding another nearby candidate. World 3 is reviewed before other worlds.
# r29: retain the original selected memory reference

World 3 r28 passed navigation evaluation (4/4 visits, exit, no sampled
footprint overlap, max pose-record gap 0.503 s), but a post-run log review
found attempts to return to older projections of completed candidates. The
no-reselection idea is retained, not bypassed: the selected source memory
object is now held throughout position refinements. The lookup dictionary can
otherwise be rebound to a newer cluster after a large observation correction,
leaving the original candidate pending despite completion of the newer one.

Both the original selected object and its current associated object are marked
complete on success. Unrelated neighbors and failed targets are not marked
opened. A regression test reproduces lookup rebinding after a position change.
World 3 remains the current review world before advancing to the others.
# Final same-source navigation suite: r29 (2026-09-29)

All five independent checks returned `PASS_NAV_VISITS_EXIT` with identical
recorded control-source and harness hashes. No control changes were made
between these five runs. Build: 7 packages; focused regression tests: 62 passed.

| World | Unique blue visits | Exit | Wall time including cleanup | Max position error |
|---|---:|---|---:|---:|
| 1 | 3/3 | crossed | 653.71 s | 0.786 m |
| 2 | 3/3 | crossed | 747.82 s | 0.758 m |
| 3 | 4/4 | crossed | 1362.50 s | 0.523 m |
| 4 | 0/0 | crossed | 1250.38 s | 0.615 m |
| 5 | 6/6 | crossed | 944.05 s | 0.726 m |

The counts above are read from world geometry by the post-run evaluator only.
Across 169 recorded input audits: only slam_toolbox published /map, only the
selected wheel-odometry bridge published /odom, and the evaluation recorder was
the only ground-truth subscriber. No forbidden prior-map/truth-localization
nodes were present. Maximum estimated-pose sample gaps were 0.502-0.535 s.
No sampled Nav2-footprint overlap with evaluated obstacle/wall/panel geometry
was found; this is not a physical contact-sensor collision audit.

Navigation only: VISIT service responses do not physically open a door. These
results do not establish manipulation reach, close parking, physical opening,
repeatability, or real-robot performance. In particular, World 3 retains local
reapproach/ambiguous-candidate checks and World 4 retains a long obstacle
recovery loop. Completion is not evidence of an optimal or uniformly smooth
trajectory. Existing memory-based no-reselection logic is retained; exact
selected source memory is linked to completion even after pose refinement.

Every run starts with online SLAM, wheel odometry and no saved map. The actual
obstacles/interior doors are preserved. The approved exit modification is an
open green-marked passage. Sensor calibration, geometric measurement models,
safety clearances and association tolerances are still configuration values;
"observation-based" does not mean parameter-free.

Evidence: `observed_nav_results_20260929/world{1..5}_r29`, plus
`suite_r29.json`, `00_slam_map_evolution_r29.png`, and
`01_world_trajectories_r29.png`. Historical failures remain alongside them.
Windows evidence is in Documents/졸업작품/00_최종자료/
36_열린출구_관측주행검증_20260929; start at `00_최신결과.html`.
The original fire_robot_ws repository remains clean and unchanged.
