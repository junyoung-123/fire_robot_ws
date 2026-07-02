# Fire robot maps

Use this directory for saved static maps used by AMCL/localization mode.

Typical workflow:

1. Mapping pass: run `simulation.launch.py localization_mode:=mapping` or real robot with `enable_slam:=true`.
2. Save the map: `ros2 run nav2_map_server map_saver_cli -f ~/fire_robot_ws_test/src/fire_robot_navigation/maps/corridor`.
3. Mission pass: run with `localization_mode:=localization map:=~/fire_robot_ws_test/src/fire_robot_navigation/maps/corridor.yaml`.
