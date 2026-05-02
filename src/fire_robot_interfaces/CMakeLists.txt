cmake_minimum_required(VERSION 3.8)
project(fire_robot_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DoorInfo.msg"
  "msg/FireInfo.msg"
  "msg/RobotState.msg"
  "srv/OpenDoor.srv"
  "action/NavigateToDoor.action"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
