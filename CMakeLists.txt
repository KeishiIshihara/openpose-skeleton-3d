cmake_minimum_required(VERSION 2.8.3)
project(openpose_skeleton_3d)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  cv_bridge
  message_generation
  coco_keypoints
)

add_message_files(
   FILES
   Keypoint3d.msg
   COCO3d.msg
   COCO3d_ARR.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  # INCLUDE_DIRS include
)

###########
## Build ##
###########
include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(openpose_skeleton_3d_node src/openpose_skeleton_3d_node.cpp)
add_dependencies(openpose_skeleton_3d_node ${PROJECT_NAME}_gencpp})
target_link_libraries(openpose_skeleton_3d_node
  ${catkin_LIBRARIES}
)
