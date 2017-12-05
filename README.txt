EXPLORE_DIRECTION_ESTIMATOR
---------------------------

Description:

This ROS node estimates all possible safe directions for the robot to move by considering the obstacles around the it and it's 
dimensions (only width). 

				  
				  Important features:
					  - Road plane segmentation based on RANSAC (pcl plane fitting used) and hence obstacle segmentation  
					  - genrates possible driveable directions for the vehicle based on available gap
					  - generates smooth directions and produces new directions only when necessary
					  - we can dynamically switch between input mode i.e. use of point cloud or disparity image for direction generation
				  
				  Subscribes To:
				  	  - /camera/left/camera_info
				  	  - /camera/right/camera_info
				  	  - /camera/disparity
				  	  - /camera/left/image_rect
				  	  - /camera/points
				  	  
				  publishes
				  	  - /explore/ground_points
				  	  - /explore/obstacle_points
				  	  - /explore/grid_view 
				  	  - /explore/incomming_point_cloud 
				  	  - /explore/drive_directions	       - unfiltered directions
				  	  - /explore/filtered_drive_directions - filtered and smoothed drive directions - use this
				  	  - /explore/gap_marks                 - gaps are drawn as lines in green color between the end points of every safe gap
				  	  - /explore/directions_as_poses	   - safe directions are published as pose array which is stamped
				  	  - /explore/free_space_marker		   - to visualize free space. free space is drawn as triangles from camera center to safe gap endpoints

				  parameters
				  	  - please see the parameters_info.txt in the project folder. This file will detail on the parameters and
				  	  	what they do.	


To run the node, run the following commands in different terminals:

1. roscore
2. rosbag play <file.bag> <appropriate remappings> 
3. rosrun explore_direction_estimator explore_direction_estimator <appropriate remappings>
4. rosrun ROS_NAMESPACE=<namespace> rosrun stereo_image_proc stereo_image_proc <appropriate remappings for <stereo> and <image> >
5. rviz
   (For rviz just open the rviz configuration file which is in the project folder) 
   

If you want to see the result from the stereo image proce:

6. rosrun image_view stereo_view <appropriate remappings for <stereo> and <image> >

