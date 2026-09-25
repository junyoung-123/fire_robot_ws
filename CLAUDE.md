# Project and evidence entry point

Read the dated section at the top of `README.md` and `docs/RELEASE_20260926.md`.
For report/PPT work, read `docs/handoff/2026-09-26/00_START_HERE.md`,
`02_EVIDENCE_SCOPE.md` and `01_ASSET_INDEX.json` in that same folder.
Use asset IDs to select original images/videos and record their provenance.

The R34 single-door sensor-based contact test, historical World 1-5 navigation,
and semantic-memory test are distinct runs. Do not merge them into one full
integrated success claim. Archived false positives are not successful detections.
Robot encoder feedback is permitted; simulator door/lever truth is evaluation
only on the R34 robot-observation entry point. Safety/configuration parameters
still exist. Real hardware and the mirrored fixture are not certified.

Preserve previous successful branches and evidence. Reproducing a simulation
requires ROS2 Humble/Gazebo dependencies, local path adaptation and a new run;
source hashes and unit tests alone do not prove a new simulation succeeded.
