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

#### Odometry Loss Analysis (Images 4 & 5)

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
| **Images 4-5** | Indoor area focus with tracking loss |


The documented troubleshooting process demonstrates the importance of proper synchronization, resolution matching, and odometry strategy selection for successful SLAM deployment.

---

**Last Updated:** November 2025  
**ROS 2 Version:** Jazzy  
**Camera:** Orbbec Femto Mega  
**SLAM Algorithm:** RTAB-Map
