# RTAB-Map Project Documentation
## Complete Guide for RGB-D SLAM with Orbbec Femto Mega Camera

---

## Table of Contents
1. [Phase 1: Initial Setup & Installation](#phase-1-initial-setup--installation)
2. [Phase 2: System Configuration](#phase-2-system-configuration)
3. [Phase 3: RTAB-Map Integration & Debugging](#phase-3-rtab-map-integration--debugging)
4. [Visual Analysis of RTAB-Map Interface](#visual-analysis-of-rtab-map-interface)
5. [Mapping Quality Assessment](#mapping-quality-assessment)
6. [Recommendations & Best Practices](#recommendations--best-practices)

---

## Phase 1: Initial Setup & Installation

### 1.1 ROS 2 & System Packages

#### ROS 2 Jazzy Desktop
Installed using the standard APT repository.

```bash
sudo apt install ros-jazzy-desktop
```

#### Camera Driver (Orbbec)
Installed the OrbbecSDK ROS 2 driver for the Femto Mega camera.

**Reference:** https://github.com/orbbec/OrbbecSDK_ROS2

### 1.2 Perception Libraries (GroundingDINO)

#### Problem
The installation of GroundingDINO failed. The error occurred while pip was trying to "get requirements to build wheel". This is often caused by pip trying to build dependencies in an isolated environment, which can conflict with existing libraries like torch.

#### Solution
We forced pip to use the current environment by using the `--no-build-isolation` flag.

```bash
# 1. Install GroundingDINO without build isolation
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# 2. Re-install Segment Anything (which may have failed due to the previous error)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Phase 2: System Configuration

### 2.1 ROS 2 Environment
To ensure the ROS 2 workspace is always available, the sourcing command was added to the `.bashrc` file for automatic execution.

```bash
# Command added to ~/.bashrc
source ~/ros2_ws/install/setup.bash
```

### 2.2 Launching Camera Node
The Orbbec Femto Mega camera node is launched using its provided launch file.

```bash
ros2 launch orbbec_camera femto_mega.launch.py
```

### 2.3 ROS 2 Bag Utilities
These are the common commands used for recording and replaying experimental data.

#### Record all topics

```bash
ros2 bag record -a <bag_name>
# Example: ros2 bag record -a lab_environment
```

#### Play a bag file (with clock)

```bash
ros2 bag play <bag_name> --clock
# The --clock flag is critical for simulation time
```

---

## Phase 3: RTAB-Map Integration & Debugging

This section details the step-by-step troubleshooting process to get RTAB-Map working with our recorded rosbag data.

### Problem 1: Image Resolution Mismatch

#### Symptom
RTAB-Map would not process the bag data. We identified a resolution mismatch between the RGB image (`/camera/color/image_raw`) and the depth image (`/camera/depth/image_raw`).

#### Solution
Create a custom Python ROS 2 node to subscribe to the raw depth image, resize it to match the RGB sensor's resolution (1280x720), and republish it on a new topic.

#### Implementation: resize_depth.py Node

**File:** `~/resize_depth.py`

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DepthResizer(Node):
    def __init__(self):
        super().__init__('depth_resizer')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Input topic
            self.callback,
            10
        )
        self.pub = self.create_publisher(Image, '/camera/depth/image_resized', 10) # Output topic
        self.get_logger().info('Depth Resizer node started. Subscribing to /camera/depth/image_raw...')

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Resize to 1280x720 (matching the RGB sensor)
        resized = cv2.resize(cv_image, (1280, 720), interpolation=cv2.INTER_NEAREST)
        
        out_msg = self.bridge.cv2_to_imgmsg(resized, encoding='passthrough')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    node = DepthResizer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

#### Setup Instructions

```bash
# Make the script executable
chmod +x ~/resize_depth.py

# Install dependencies if missing
sudo apt install ros-jazzy-cv-bridge python3-opencv
```

---

### Problem 2: Timestamp Synchronization

#### Symptom
After fixing the resolution, RTAB-Map still failed with:
- `[WARN] ... Did not receive data since 5 seconds!`
- `[WARN] ... The time difference between rgb and depth frames is high`

#### Cause
When replaying a rosbag, nodes process data as fast as possible, but the timestamps in the bag are from the past. The system must be told to use the "simulated" clock from the bag file, not the current system time.

#### Solution
- Play the rosbag using the `--clock` flag
- Launch RTAB-Map with the `use_sim_time:=true` parameter

#### Standard Workflow

**Terminal 1: Play Rosbag**
```bash
ros2 bag play lab_environment --clock
```

**Terminal 2: Run Resize Node**
```bash
python3 ~/resize_depth.py
```

**Terminal 3: Run RTAB-Map**
```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    args:="--delete_db_on_start" \
    depth_topic:=/camera/depth/image_resized \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    depth_camera_info_topic:=/camera/depth/camera_info \
    approx_sync:=true \
    frame_id:=camera_link \
    use_sim_time:=true
```

---

### Problem 3: Visual Odometry Failure

#### Symptom
Even with correct synchronization, odometry failed. Logs showed:
- `Odom: quality=0`
- `[ERROR] ... no odometry is provided`
- `Registration failed: "Not enough inliers 0/20"`

#### Cause
The default visual odometry strategy (using visual features, Strategy 0) was failing. This was likely due to rapid motion, poor lighting, or lack of texture in the recorded environment.

#### Solution
Switch the odometry strategy from visual features to **ICP (Iterative Closest Point)** (Strategy 1). ICP uses the 3D point cloud geometry to find the transformation, which is more robust in texture-poor environments.

#### Final Working Launch Command (Using ICP)

**Terminal 3 (Final): Run RTAB-Map with ICP**
```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    args:="--delete_db_on_start --Odom/Strategy 1" \
    depth_topic:=/camera/depth/image_resized \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    depth_camera_info_topic:=/camera/depth/camera_info \
    approx_sync:=true \
    approx_sync_max_interval:=0.2 \
    frame_id:=camera_link \
    use_sim_time:=true \
    queue_size:=50
```

#### Result
 **This solution worked successfully.**

**Note:** The process was observed to be slow. For future tests, we can experiment with playing the bag at a slower rate to give the ICP algorithm more time to process each frame:

```bash
ros2 bag play lab_environment --clock --rate 0.5
```

---

## Visual Analysis of RTAB-Map Interface

###  Orbbec Femto Mega RGB-D Camera

This depth camera simultaneously captures:
- **RGB**: Color image of the environment
- **Depth (D)**: Depth information for each pixel, creating a 3D point cloud

###  RTAB-Map (Real-Time Appearance-Based Mapping)

RTAB-Map is a SLAM (Simultaneous Localization and Mapping) algorithm that enables:
- Real-time environment mapping
- Simultaneous robot/camera localization
- Loop closure detection to correct map drift
- Memory management for large-scale environments

---

## RTAB-Map Interface Components

### 1. Point Cloud View (3D Map - Upper Right Panel)

The **point cloud** represents the captured 3D space:

| Color/Element | Meaning |
|---------------|---------|
| **White/Gray points** | Detected surfaces (ceiling, walls, floor) |
| **Colored points** | Features extracted from environment with RGB information |
| **Point density** | Higher density = better scan quality |

**Observable elements in the images:**
- Ceiling structure with beams
- Building walls
- Environment objects (desks, computer monitors)
- Robot/sensor movement through space

---

### 2. Odometry View (Lower Left Panel)

This view shows the **camera perspective** with information overlays.

#### Color Coding System

| Color | Status | Meaning |
|-------|--------|---------|
|**Dark Red** | **Odometry Lost** | Tracking lost - system cannot determine position (CRITICAL) |
|**Dark Yellow** | **Low Inliers** | Few feature matches between frames (WARNING) |
|**Green** | **Inliers** | Correctly matched features between consecutive frames (GOOD) |
|**Yellow** | **Unmatched Features** | Visible features not matched with previous frames (NORMAL in new areas) |
| **Red** | **Outliers** | Incorrect correspondences or noise (filtered out) |

#### Odometry Loss Analysis (Images 4)

**Complete red background** indicates total odometry loss, typically caused by:
- Very fast camera movement
- Textureless surfaces (smooth walls)
- Poor lighting conditions
- Occlusions or motion blur
- Reflective surfaces (computer screens)

---

### 3. 3D Map View (Right Panel)

Shows the **constructed three-dimensional representation**:

| Element | Description |
|---------|-------------|
| **Floor mesh** | Flat surface (building floor) |
| **Vertical structures** | Walls and columns |
| **Coordinate axes** | Map reference system |
| - Green axis | Y axis |
| - Blue axis | Z axis |
| - Red axis | X axis (not visible in these views) |

#### Map Progression

| Image | Description |
|-------|-------------|
| **Image 1** | Wide view of mapped environment with ceiling and multiple structures |
| **Image 2** | Close-up of work area (desks, computers) |
| **Image 3** | Rotated view showing different perspectives |
| **Image 4** | Indoor area focus with tracking loss |

<p align="center">
  <img src="Images/Imagen1" alt="Image 1" width="500">
  <br>
  </strong> Image 1.
</p>
<p align="center">
  <img src="Images/Imagen2" alt="Image 1" width="500">
  <br>
  </strong> Image 2.
</p>
<p align="center">
  <img src="Images/Imagen3" alt="Image 1" width="500">
  <br>
  </strong> Image 3.
</p>
<p align="center">
  <img src="Images/Imagen4" alt="Image 1" width="500">
  <br>
  </strong> Image 4.
</p>
## Ouster LiDAR Startup Guide for ROS 2

This document details the process for establishing the network connection with the Ouster LiDAR sensor and launching the ROS 2 driver to expose the data (point cloud and IMU) as ROS 2 topics.

-----

## 1\. Host Network Configuration

The Ouster sensor uses a **Link-Local IP address** (in the $169.254.x.x$ range). For your computer (host) to communicate with the sensor for both control (HTTP) and data reception (UDP), it must be on the same network range.

### Step 1.1: Discover the Sensor (Optional)

You can use the `ouster-cli` command-line tool to confirm the sensor's IP address and the expected destination IP (your machine's IP).

```bash
ouster-cli discover
```

**Typical Output:**

```
OS-1-32-U2 - 122228000146
* addresses:
  * IPv4 link-local 169.254.41.35/16  # <-- Sensor IP
* UDP destination address: 169.254.41.100 # <-- Host IP (must be assigned to the host)
* UDP port lidar, IMU: 7502, 7503
```

### Step 1.2: Assign Link-Local IP to the Host

Assign the expected destination IP (`169.254.41.100/16` in the example) to your Ethernet interface (`eno1` or `eth0`):

```bash
sudo ip addr add 169.254.41.100/16 dev eno1
```

> **Note:** Replace `eno1` with your Ethernet interface name if it's different.

-----

## 2\. ROS 2 Configuration File (YAML)

To configure the ROS 2 driver robustly and avoid command-line parsing errors, we'll use a **YAML configuration file**.

### `ouster_params.yaml`

Create this file in your current directory or a dedicated configuration folder:

```yaml
# ouster_params.yaml
/ouster/os_driver:
  ros__parameters:
    # --- Control Parameters ---
    sensor_hostname: "169.254.41.35"     # Sensor's IP (from discover)
    metadata_hostname: "169.254.41.35"  # IP for metadata (same as sensor)

    # --- UDP Data Parameters ---
    lidar_port: 7502                    # UDP Port for LiDAR data
    imu_port: 7503                      # UDP Port for IMU data
    udp_dest_host: "169.254.41.100"     # Destination IP (your host's IP)

    # --- Driver Parameters ---
    auto_start: True                    # Automatically start data streaming
    proc_mask: 1                        # Processing mask (1=PCL, 2=IMU, 3=Both)
    
    # [Other parameters like resolution mode, etc., can be added here]
```

-----

## 3\. Launch the ROS 2 Driver

Use the launch file (`driver.launch.py`) from the **`ouster_ros`** package and pass the path to your configuration file.

### Step 3.1: Execute the Launch File

Ensure your ROS 2 environment is sourced (`source install/setup.bash`).

```bash
ros2 launch ouster_ros driver.launch.py params_file:=./ouster_params.yaml
```

### Step 3.2: Verify the ROS 2 Topics

Once the driver launches successfully (without hostname or UDP timeout errors), the sensor data will be exposed in ROS 2.

Open a new terminal and list the topics:

```bash
ros2 topic list
```

**Expected Data Topics:**

| Topic | Message Type | Description |
| :--- | :--- | :--- |
| `/ouster/points` | `sensor_msgs/PointCloud2` | **Point Cloud** data (LiDAR). |
| `/ouster/imu` | `sensor_msgs/Imu` | **IMU data** (if `imu_port` is used). |
| `/ouster/metadata` | *string* | Sensor configuration metadata. |

You can confirm data publication with the following command:

```bash
ros2 topic echo /ouster/points --once
```

-----

## 4\. Launch SLAM/GLIM (Example)

After the LiDAR and IMU data are available on the `/ouster/points` and `/ouster/imu` topics, you can launch your SLAM package (like GLIM) by providing the correct topic names.

### Step 4.1: Find the GLIM Launch File

Since the launch file `glim.launch.py` was not found, you must list the files to find the correct name (e.g., `glim_slam.launch.py`):

```bash
ls /home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws/install/glim/share/glim/launch
```

### Step 4.2: Execute the Launch File (Example with Corrected Name)

Assuming the correct name is **`glim_slam.launch.py`**:

```bash
ros2 launch glim glim_slam.launch.py \
    pointcloud_topic:=/ouster/points \
    imu_topic:=/ouster/imu
```

-----


-----

## GLIM SLAM: Quick Start Command Guide

### 1\. **Preparation and Configuration**

Before running the node, ensure you are in the correct **ROS 2 workspace directory** and that the configuration files point to the **CPU modules** and **correct topics**.

  * **Location:** `/home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws`
  * **Key Files Status:**
      * **`config.json`** must specify the CPU modules: `config_odometry_cpu.json`, `config_sub_mapping_passthrough.json`, and `config_global_mapping_pose_graph.json`.
      * **`config_ros.json`** must specify the correct sensor topics (e.g., `"/ouster/imu"`, `"/ouster/points"`).

### 2\. **Execute the GLIM Node**

Navigate to your workspace root and run the `glim_rosnode`. You must pass the absolute path to the configuration directory using the `--ros-args` parameter.

1.  **Change Directory (if necessary):**

    ```bash
    cd /home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws
    ```

2.  **Run the Node:**

    ```bash
    ros2 run glim_ros glim_rosnode --ros-args -p config_path:=/home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws/install/glim/share/glim/config
    ```

### 3\. **Expected Output on Successful Launch**

A successful launch will display informational (`[info]`) messages confirming that the node loaded the necessary shared libraries and started the IMU initialization process.

  * **Module Loading:** Confirms the use of CPU modules.
    ```
    [glim] [info] load libodometry_estimation_cpu.so
    [glim] [info] load libsub_mapping_passthrough.so
    [glim] [info] load libglobal_mapping_pose_graph.so
    ```
  * **Initialization:** Confirms the IMU state is estimated and optimized.
    ```
    [odom] [info] estimate initial IMU state
    ...
    [odom] [info] initial IMU state estimation result
    [odom] [info] T_world_imu=se3(...)
    ```

### 4\. **Next Steps (Visualization)**

To visualize the SLAM process and the resulting map/trajectory, you should run RViz and load the GLIM configuration.

  * **Launch RViz2:**
    ```bash
    ros2 run rviz2 rviz2 -d install/glim/share/glim/config/glim.rviz
    ```



The documented troubleshooting process demonstrates the importance of proper synchronization, resolution matching, and odometry strategy selection for successful SLAM deployment.

---

**Last Updated:** November 2025  
**ROS 2 Version:** Jazzy  
**Camera:** Orbbec Femto Mega  
**SLAM Algorithm:** RTAB-Map
