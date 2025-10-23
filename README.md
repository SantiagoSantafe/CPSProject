#CPS Project Documentation
## First Phase
### 23 Oct
We had problems installing GroundDino he error happened while pip was trying to "get requirements to build wheel" for GroundingDINO.

You can force pip to use your current environment (which already has torch) instead of creating an isolated one.

Run the GroundingDINO installation again using the --no-build-isolation flag:

pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

After that, re-run the segment-anything installation (it likely didn't run because the previous command failed):
pip install git+https://github.com/facebookresearch/segment-anything.git

I was following the whole installation of ROS jazzy,
found that is easier to use:
cpsstudent@cps-wkstn-nuc03:~$ sudo apt install ros-jazzy-desktop

I'm looking the camera documentation installing all the needed packages
https://github.com/orbbec/OrbbecSDK_ROS2

Summary: 365 packages finished [2h 22min 16s]
  106 packages had stderr output: action_tutorials_py ament_clang_format ament_clang_tidy ament_copyright ament_cppcheck ament_cpplint ament_flake8 ament_index_python ament_lint ament_lint_cmake ament_mypy ament_package ament_pclint ament_pep257 ament_pycodestyle ament_pyflakes ament_uncrustify ament_xmllint camera_info_manager_py demo_nodes_py domain_coordinator examples_rclpy_executors examples_rclpy_guard_conditions examples_rclpy_minimal_action_client examples_rclpy_minimal_action_server examples_rclpy_minimal_client examples_rclpy_minimal_publisher examples_rclpy_minimal_service examples_rclpy_minimal_subscriber examples_rclpy_pointcloud_publisher examples_tf2_py foonathan_memory_vendor google_benchmark_vendor gz_cmake_vendor gz_math_vendor gz_utils_vendor iceoryx_posh launch launch_pytest launch_ros launch_testing launch_testing_examples launch_testing_ros launch_xml launch_yaml lifecycle_py mimick_vendor osrf_pycommon qt_gui_cpp quality_of_service_demo_py rclpy ros2action ros2bag ros2cli ros2component ros2doctor ros2interface ros2launch ros2lifecycle ros2multicast ros2node ros2param ros2pkg ros2plugin ros2run ros2service ros2test ros2topic ros2trace rosbag2_examples_py rosbag2_py rosidl_cli rosidl_pycommon rosidl_runtime_py rpyutils rqt rqt_action rqt_bag rqt_bag_plugins rqt_console rqt_graph rqt_gui rqt_gui_py rqt_msg rqt_plot rqt_publisher rqt_py_console rqt_reconfigure rqt_service_caller rqt_shell rqt_srv rqt_topic rviz_ogre_vendor sensor_msgs_py sros2 test_launch_ros test_ros2trace test_tracetools test_tracetools_launch tf2_ros_py tf2_tools topic_monitor tracetools_launch tracetools_read tracetools_test tracetools_trace
