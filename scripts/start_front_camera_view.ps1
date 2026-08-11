param(
    [string]$Distro = "Ubuntu2204Recovered",
    [string]$Workspace = "/home/junyoung/fire_robot_ws_test",
    [string]$Topic = "/camera/color/image_raw"
)

$cmd = @"
source /opt/ros/humble/setup.bash
cd "$Workspace"
source install/setup.bash
for i in {1..60}; do
  if ros2 topic list 2>/dev/null | grep -qx "$Topic"; then
    break
  fi
  sleep 1
done
python3 scripts/show_ros_image.py --topic "$Topic" --window-name "Front Camera"
"@

$script = @"
`$cmd = @'
$cmd
'@
& wsl.exe -d "$Distro" -- bash -lc `$cmd
"@

$encoded = [Convert]::ToBase64String([System.Text.Encoding]::Unicode.GetBytes($script))
Start-Process -FilePath powershell.exe `
    -ArgumentList @('-NoProfile', '-EncodedCommand', $encoded) `
    -WindowStyle Hidden
