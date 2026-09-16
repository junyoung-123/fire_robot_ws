#!/usr/bin/env bash
# Independent mission regression; never overwrite an existing evidence directory.
set -uo pipefail
cd "${VALIDATION_WORKSPACE_ROOT:-$(dirname "${BASH_SOURCE[0]}")/..}" || exit 2
if [[ ! -f src/fire_robot_fsm/package.xml || ! -f scripts/run_headless_validation_once.sh ]]; then
  echo 'Invalid workspace root. Set VALIDATION_WORKSPACE_ROOT when running an archived runner.' >&2
  exit 2
fi
out="${1:?Provide a new output directory}"
if [[ -e "$out" ]]; then
  echo "Refusing to overwrite existing evidence: $out" >&2
  exit 2
fi
mkdir -p "$out"
export VALIDATION_WORKSPACE_ROOT="$PWD"
mkdir "$out/runner_snapshot"
cp scripts/run_headless_validation_once.sh "$out/runner_snapshot/run_headless_validation_once.sh"
runner_snapshot="$out/runner_snapshot/run_headless_validation_once.sh"
chmod a-w "$runner_snapshot"
helpers=(validation_trace_logger.py capture_passive_mission_images.py
  check_validation_log.py check_trace_attitude.py check_trace_events.py
  check_manual_navigation_exclusion.py render_resumed_world_evidence.py
  render_obstacle_validation_evidence.py)
for helper in "${helpers[@]}"; do
  cp "scripts/$helper" "$out/runner_snapshot/$helper"
  chmod a-w "$out/runner_snapshot/$helper"
done
export VALIDATION_HELPERS_ROOT="$(realpath "$out/runner_snapshot")"
git rev-parse HEAD > "$out/commit.txt"
git diff --binary > "$out/source.patch"
find src -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.world' -o -name '*.xacro' -o -name '*.pt' \) -print0 |
  sort -z | xargs -0 sha256sum > "$out/runtime_inputs.sha256"
printf 'world\trunner_rc\tchecker_rc\tattitude_rc\tevents_rc\texclusion_rc\tgeometry_rc\tstatus\n' > "$out/results.tsv"
printf '%s\n' 'SCOPE: full mission, saved static map, sim odometry, simplified arm and hinge backend.' \
  'Actual door model rotation must be verified after each hinge command.' \
  'NOT: feedback-contact actual-PIPER physics or empty-map SLAM validation.' \
  'Door counts below are checker ground truth only, not FSM inputs.' > "$out/scope.txt"
worlds=(
  '1 obstacle_wall_doors_v5.world 3 1500'
  '2 obstacle_door_layout_alt_v1.world 3 1500'
  '3 obstacle_door_layout_world3_v1.world 4 1800'
  '4 obstacle_no_blue_world4_v1.world 0 900'
  '5 obstacle_all_blue_world5_v1.world 6 2100'
)
if [[ -n "${VALIDATION_WORLD_FILTER:-}" ]]; then
  selected=()
  for specification in "${worlds[@]}"; do
    read -r world _ <<< "$specification"
    if [[ ",${VALIDATION_WORLD_FILTER}," == *",${world},"* ]]; then
      selected+=("$specification")
    fi
  done
  worlds=("${selected[@]}")
  if (( ${#worlds[@]} == 0 )); then exit 2; fi
fi
printf '%s\n' "${worlds[@]}" > "$out/world_limits.txt"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PASSIVE_MISSION_IMAGES=1
for specification in "${worlds[@]}"; do
  read -r world file expected timeout <<< "$specification"
  log="$out/world${world}.log"
  trace="$out/world${world}_trace"
  echo "Starting world $world: $file"
  TRACE_DIR="$trace" bash "$runner_snapshot" \
    "$file" "$log" "$timeout" publish_debug_image:=false log_handle_detections:=true
  runner=$?
  python3 "$VALIDATION_HELPERS_ROOT/check_validation_log.py" "$log" --expected-open-count "$expected" --max-rejected 0 --require-door-feedback > "$out/world${world}_checker.txt" 2>&1
  checker=$?
  python3 "$VALIDATION_HELPERS_ROOT/check_trace_attitude.py" "$trace/summary.txt" --max-deg 20 > "$out/world${world}_attitude.txt" 2>&1
  attitude=$?
  python3 "$VALIDATION_HELPERS_ROOT/check_trace_events.py" "$trace/events.csv" --expected-open-count "$expected" > "$out/world${world}_events_check.txt" 2>&1
  events=$?
  python3 "$VALIDATION_HELPERS_ROOT/check_manual_navigation_exclusion.py" "$log" > "$out/world${world}_exclusion.txt" 2>&1
  exclusion=$?
  python3 "$VALIDATION_HELPERS_ROOT/render_resumed_world_evidence.py" "$out" --world "$world" \
    --output-name mission_evidence_final --fail-on-overlap > "$out/world${world}_geometry.txt" 2>&1
  geometry=$?
  status="missing_runner_status"
  if [[ -f "$log.status" ]]; then status="$(<"$log.status")"; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$world" "$runner" "$checker" "$attitude" "$events" "$exclusion" "$geometry" "$status" >> "$out/results.tsv"
  echo "Finished world $world: runner=$runner checker=$checker attitude=$attitude events=$events exclusion=$exclusion geometry=$geometry $status"
  if (( runner != 0 || checker != 0 || attitude != 0 || events != 0 || exclusion != 0 || geometry != 0 )); then
    echo 'Stopped before the next world: investigate this failure first.'
    exit 1
  fi
done
printf 'All %s selected worlds passed this mission-regression scope.\n' "${#worlds[@]}"
