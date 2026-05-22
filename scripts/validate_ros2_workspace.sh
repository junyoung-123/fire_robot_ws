#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_ROSDEP=0
if [[ "${1:-}" == "--skip-rosdep" ]]; then
  SKIP_ROSDEP=1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[ERROR] ros2 command not found. Source ROS 2 first:"
  echo "        source /opt/ros/humble/setup.bash"
  exit 1
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "[ERROR] colcon command not found. Install python3-colcon-common-extensions."
  exit 1
fi

echo "[1/6] Checking Python syntax"
python3 - <<'PY'
import ast
from pathlib import Path

bad = []
for path in Path("src").rglob("*.py"):
    try:
        ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        bad.append((path, exc))

if bad:
    for path, exc in bad:
        print(f"[BAD] {path}: {exc}")
    raise SystemExit(1)

print("Python syntax OK")
PY

echo "[2/6] Installing ROS dependencies with rosdep when available"
if [[ "$SKIP_ROSDEP" == "1" ]]; then
  echo "[SKIP] rosdep step skipped by --skip-rosdep"
elif command -v rosdep >/dev/null 2>&1; then
  rosdep install --from-paths src --ignore-src -r -y
else
  echo "[WARN] rosdep not found. Skipping dependency installation."
fi

echo "[3/6] Building workspace"
colcon build --symlink-install

echo "[4/6] Sourcing local install"
set +u
source install/setup.bash
set -u

echo "[5/6] Checking generated interfaces and installed packages"
ros2 interface show fire_robot_interfaces/msg/DoorInfo >/dev/null
ros2 interface show fire_robot_interfaces/srv/OpenDoor >/dev/null
ros2 pkg prefix fire_robot_bringup >/dev/null
ros2 pkg prefix fire_robot_manipulation >/dev/null
test -f "$(ros2 pkg prefix fire_robot_manipulation)/share/fire_robot_manipulation/config/fire_robot.srdf"

echo "[6/6] Checking launch files are discoverable"
ros2 launch fire_robot_manipulation move_group.launch.py --show-args >/dev/null
ros2 launch fire_robot_bringup simulation.launch.py --show-args >/dev/null
ros2 launch fire_robot_bringup real_robot.launch.py --show-args >/dev/null

cat <<'EOF'

[OK] ROS 2 workspace validation passed.

Optional runtime smoke test:
  source install/setup.bash
  LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=3.3 \
    timeout 45s ros2 launch fire_robot_bringup simulation.launch.py use_rviz:=false headless:=true

If the smoke test exits because of timeout after Gazebo/Nav2 nodes start, that is expected.
Investigate only if you see immediate Python import errors, missing package errors,
or duplicate /open_door service errors.
EOF
