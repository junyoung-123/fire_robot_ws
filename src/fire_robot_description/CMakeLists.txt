cmake_minimum_required(VERSION 3.8)
project(fire_robot_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
