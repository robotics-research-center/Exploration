cmake_minimum_required(VERSION 2.8.3)
project(explore_direction_estimator_mod)

find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  sensor_msgs 
  nav_msgs 
  message_filters 
  image_transport 
  cv_bridge 
  image_geometry 
  tf 
  pcl_ros 
  std_srvs
  message_generation
 # dynamic_reconfigure
  rospy
)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

catkin_package()

add_definitions(-msse3)
include_directories(  ${catkin_INCLUDE_DIRS}  ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



add_executable(explore_direction_estimation_mod src/explore_direction_estimation.cpp)


target_link_libraries ( explore_direction_estimation_mod ${PCL_LIBRARIES} ${catkin_LIBRARIES} )

add_dependencies(explore_direction_estimation_mod ${catkin_EXPORTED_TARGETS})
add_dependencies(explore_direction_estimation_mod ${PROJECT_NAME}_gencpp)


find_package(Boost REQUIRED COMPONENTS signals thread)
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(explore_direction_estimation_mod  ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

#add_dependencies(road_points_from_dense_3d ${PROJECT_NAME}_gencfg)


