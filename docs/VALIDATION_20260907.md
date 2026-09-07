# Final Simulation Validation - 2026-09-07

## Strict full set

Command:

```bash
bash scripts/run_full_world_validation_set.sh \
  artifacts/validation/final_full_20260907_r6
```

Acceptance criteria for every world:

- mission reaches `MISSION_COMPLETE`
- successful door-open requests equal the scenario expectation
- unique matched Gazebo blue-door topics equal the expectation
- rejected Gazebo door matches equal zero
- maximum absolute roll and pitch remain below 20 degrees

| World | Scenario | Expected/opened | Max roll/pitch | Result |
| --- | --- | ---: | ---: | --- |
| 1 | mixed doors and obstacles | 3/3 | 0.002/0.621 deg | PASS |
| 2 | alternate door/obstacle layout | 3/3 | 0.003/0.732 deg | PASS |
| 3 | four blue, two red | 4/4 | 0.001/0.514 deg | PASS |
| 4 | no blue doors | 0/0 | 0.004/0.316 deg | PASS |
| 5 | all six side doors blue | 6/6 | 0.021/0.121 deg | PASS |

The set completed 16/16 unique blue-door openings with zero rejected Gazebo
door matches. The trace logger recorded 3,672 door observations and 16
door-open events. Transient Nav2 failures are allowed only when the FSM
recovers and the strict terminal criteria still pass.

## CPU contention run

The retained representative stress result is
`artifacts/validation/final_cpu_stress_20260906`. World 5 ran with two
continuous CPU workers and still opened 6/6 unique blue doors, produced zero
rejected matches, and reached `MISSION_COMPLETE`.

A 2026-09-07 observation run was stopped at 2/6 after prolonged target
replanning under load. Per the project decision, the navigation policy was not
changed solely for this synthetic contention case; its incomplete log remains
under `artifacts/validation/final_cpu_stress_20260907` for reference.

The six-worker overload probe is a capacity-boundary record, not a functional
regression result.

## Perception interpretation

- `best.pt` (Door 1-class) loaded successfully and ran with HSV color
  classification on front/front-left/front-right cameras.
- `handle_best_v2.pt` (`lever_handle`) loaded successfully.
- The stylized Gazebo lever produced zero direct handle-YOLO hits in the final
  set, so handle localization used HSV or observed wall/door projection.
- These results validate navigation, target memory, alignment, manipulation FSM,
  Gazebo hinge command/feedback, and exit integration. They do not prove
  handle-YOLO accuracy or contact forces on a physical laboratory door.

## Evidence

- `artifacts/validation/final_full_20260907_r6/full_worlds_obstacle_paths.png`
- `artifacts/validation/final_full_20260907_r6/full_worlds_evidence.png`
- `artifacts/validation/final_full_20260907_r6/door_opening_evidence.png`
- `artifacts/validation/final_cpu_stress_20260906/trace/trajectory.png`
- `artifacts/validation/final_cpu_stress_20260906/checker.txt`

The door-opening schematic is log-derived. Raw Gazebo before/after imagery and
joint feedback are stored in the external meeting artifact folder and must be
described as commanded hinge integration, not full contact-dynamics proof.
