
# CPSProject — SLAM + Open-Vocabulary 3D Semantic Mapping + Offline Evaluation
**ROS 2 Jazzy + Python (offline-capable)**

This repository delivers an end-to-end robotics perception workflow:

1) **RGB-D SLAM (RTAB-Map)** using **Orbbec Femto Mega**  
2) **LiDAR-IMU SLAM (GLIM)** using **Ouster OS-1-32**  
3) **Offline semantic mapping + language navigation** using recorded RGB-D datasets (RealSense D435i recommended)  
4) **Offline evaluation** with metrics + plots + visual evidence suitable for final presentations

> **Main idea:** SLAM is handled in ROS 2. Semantic mapping + evaluation is reproducible offline from recorded datasets, producing debug frames and metrics.

---

## Table of Contents
- [CPSProject — SLAM + Open-Vocabulary 3D Semantic Mapping + Offline Evaluation](#cpsproject--slam--open-vocabulary-3d-semantic-mapping--offline-evaluation)
  - [Table of Contents](#table-of-contents)
  - [Project Deliverables](#project-deliverables)
  - [Hardware / Software Requirements](#hardware--software-requirements)
    - [Supported OS](#supported-os)
    - [Sensors Used (project)](#sensors-used-project)
    - [Python](#python)
  - [Installation Overview](#installation-overview)
    - [Path A — Offline Pipeline (Recommended for graders / reproducibility)](#path-a--offline-pipeline-recommended-for-graders--reproducibility)
    - [Path B — Full ROS 2 SLAM (Ubuntu only)](#path-b--full-ros-2-slam-ubuntu-only)
- [A) Offline Pipeline (Cross-platform: macOS/Linux)](#a-offline-pipeline-cross-platform-macoslinux)
  - [1) Create virtual environment and install dependencies](#1-create-virtual-environment-and-install-dependencies)
  - [2) Model Weights (required)](#2-model-weights-required)
  - [3) Quick sanity check (dry-run)](#3-quick-sanity-check-dry-run)
  - [4) Record a RealSense dataset (offline RGB-D)](#4-record-a-realsense-dataset-offline-rgb-d)
  - [5) Run the offline pipeline on the dataset](#5-run-the-offline-pipeline-on-the-dataset)
- [B) ROS 2 SLAM (Ubuntu: RTAB-Map + GLIM)](#b-ros-2-slam-ubuntu-rtab-map--glim)
  - [2) RTAB-Map SLAM (Orbbec Femto Mega, RGB-D)](#2-rtab-map-slam-orbbec-femto-mega-rgb-d)
  - [2.3 Fix 1 — Depth/RGB resolution mismatch (required in our setup)](#23-fix-1--depthrgb-resolution-mismatch-required-in-our-setup)
  - [2.4 Fix 2 — Timestamp synchronization (required for rosbags)](#24-fix-2--timestamp-synchronization-required-for-rosbags)
  - [2.5 Fix 3 — Visual odometry failure → ICP odometry (final working command)](#25-fix-3--visual-odometry-failure--icp-odometry-final-working-command)
- [3) GLIM SLAM (Ouster OS-1-32, LiDAR + IMU)](#3-glim-slam-ouster-os-1-32-lidar--imu)
  - [3.1 Configure host network (link-local)](#31-configure-host-network-link-local)
  - [3.2 Configure Ouster ROS driver using YAML](#32-configure-ouster-ros-driver-using-yaml)
  - [3.3 Run GLIM](#33-run-glim)
- [Model Weights (Details)](#model-weights-details)
- [Running Evaluation + Generating Evidence](#running-evaluation--generating-evidence)
  - [Linux:](#linux)
  - [macOS:](#macos)
- [Troubleshooting (Known Issues + Fixes)](#troubleshooting-known-issues--fixes)
    - [1) ModuleNotFoundError: No module named 'src'](#1-modulenotfounderror-no-module-named-src)
    - [2) RTAB-Map “Did not receive data since 5 seconds”](#2-rtab-map-did-not-receive-data-since-5-seconds)
    - [3) RTAB-Map “time difference between rgb and depth frames is high”](#3-rtab-map-time-difference-between-rgb-and-depth-frames-is-high)
    - [4) RTAB-Map visual odometry fails (quality=0, not enough inliers)](#4-rtab-map-visual-odometry-fails-quality0-not-enough-inliers)
    - [5) GLIM warning “large time gap between consecutive LiDAR frames”](#5-glim-warning-large-time-gap-between-consecutive-lidar-frames)
    - [Repository Layout](#repository-layout)
- [Authors](#authors)
- [Credits / Licenses](#credits--licenses)

---

## Project Deliverables

This repository includes:

- ✅ **RTAB-Map RGB-D SLAM** with real-world integration fixes:
  - depth/RGB resolution mismatch → resize + republish
  - rosbag time sync → `--clock` + `use_sim_time:=true`
  - odometry failure → ICP odometry (`--Odom/Strategy 1`)
- ✅ **GLIM LiDAR-IMU SLAM** with Ouster network configuration + ROS driver YAML
- ✅ **Open-vocabulary semantic mapping (offline)**:
  - text query → detection + segmentation → 3D projection → semantic map fusion
- ✅ **Text-based navigation demo** (goal selection from semantic map)
- ✅ **Evaluation metrics + plots + debug frames**
- ✅ Visual assets under `Images/` and evaluation plots under `results/evaluations/`

For a longer technical report and screenshots analysis, see **`documentation.md`**.

---

## Hardware / Software Requirements

### Supported OS
- **macOS**: offline semantic pipeline ✅
- **Ubuntu 22.04 / 24.04**: offline pipeline ✅ + ROS 2 SLAM ✅ (recommended for full reproduction)

### Sensors Used (project)
- **Orbbec Femto Mega** (RGB-D)
- **Intel RealSense D435i** (RGB-D, used for offline datasets)
- **Ouster OS-1-32** (LiDAR + IMU)

### Python
- **Python 3.10+** recommended  
- Use a **virtual environment** to avoid system Python conflicts.

---

## Installation Overview

There are two ways to reproduce the project:

### Path A — Offline Pipeline (Recommended for graders / reproducibility)
Runs on macOS or Linux using recorded datasets:
- record RealSense dataset (optional)
- run semantic pipeline
- run evaluation
- generate plots + debug frames + video evidence

### Path B — Full ROS 2 SLAM (Ubuntu only)
Runs RTAB-Map (Orbbec) and GLIM (Ouster):
- install ROS 2 Jazzy
- install sensor drivers
- run SLAM live or from rosbags
- visualize in RTAB-Map / RViz / GLIM viewer

---

# A) Offline Pipeline (Cross-platform: macOS/Linux)

## 1) Create virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

If you see ModuleNotFoundError: No module named 'src', always run with:

PYTHONPATH=. python -m scripts.run_system --dry-run
```

⸻

## 2) Model Weights (required)

Place checkpoints under ./models/:
	•	models/sam_vit_h_4b8939.pth
	•	models/groundingdino_swint_ogc.pth

Optional helper (if available in this repo):

python -m scripts.download_Weights


⸻

## 3) Quick sanity check (dry-run)

Validates the end-to-end pipeline wiring without sensors/datasets:

PYTHONPATH=. python -m scripts.run_system \
  --dry-run --max-frames 4 \
  --results-dir results/demo_run \
  --verbose

Evaluate:

PYTHONPATH=. python -m scripts.evaluate_system \
  --run-dir results/demo_run \
  --queries configs/eval/test_queries.json \
  --scenarios configs/eval/test_scenarios.json \
  --verbose


⸻

## 4) Record a RealSense dataset (offline RGB-D)

This produces:
	•	rgb/000000.png ...
	•	depth/000000.png ...
	•	meta.json with intrinsics + depth scale

Verify device:

python - << 'EOF'
import pyrealsense2 as rs
ctx = rs.context()
print("Devices:", ctx.devices.size())
EOF

Record:

PYTHONPATH=. python -m scripts.record_realsense --frames 120 --fps 15 --viewer

Example output:

data/realsense_runs/20260129_160410/
  rgb/
  depth/
  meta.json


⸻

## 5) Run the offline pipeline on the dataset

PYTHONPATH=. python -m scripts.run_system \
  --data data/realsense_runs/<RUN_TIMESTAMP> \
  --max-frames 30 \
  --results-dir results/realsense_demo_run \
  --fast \
  --save-debug \
  --verbose


⸻

# B) ROS 2 SLAM (Ubuntu: RTAB-Map + GLIM)

This section is required to reproduce the SLAM part of the project (Orbbec + Ouster).
For detailed screenshots and interface analysis see documentation.md.

1) Install ROS 2 Jazzy (Ubuntu)

sudo apt update
sudo apt install ros-jazzy-desktop

Recommended .bashrc additions:

source /opt/ros/jazzy/setup.bash
if using a workspace:
source ~/ros2_ws/install/setup.bash


⸻

## 2) RTAB-Map SLAM (Orbbec Femto Mega, RGB-D)

2.1 Install Orbbec ROS 2 driver

Reference: https://github.com/orbbec/OrbbecSDK_ROS2

Launch camera:

ros2 launch orbbec_camera femto_mega.launch.py

2.2 Record / replay rosbag

Record all topics:

ros2 bag record -a lab_environment

Replay with simulated time (CRITICAL):

ros2 bag play lab_environment --clock


⸻

## 2.3 Fix 1 — Depth/RGB resolution mismatch (required in our setup)

RTAB-Map will fail if RGB and depth resolutions differ.

Create and run a ROS2 Python node to resize depth and republish:

sudo apt install ros-jazzy-cv-bridge python3-opencv
chmod +x ~/resize_depth.py
python3 ~/resize_depth.py

Depth is republished to:
	•	/camera/depth/image_resized

⸻

## 2.4 Fix 2 — Timestamp synchronization (required for rosbags)

When replaying bags, you must use:
	•	ros2 bag play ... --clock
	•	launch RTAB-Map with use_sim_time:=true

Launch RTAB-Map:

ros2 launch rtabmap_launch rtabmap.launch.py \
  args:="--delete_db_on_start" \
  depth_topic:=/camera/depth/image_resized \
  rgb_topic:=/camera/color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  depth_camera_info_topic:=/camera/depth/camera_info \
  approx_sync:=true \
  frame_id:=camera_link \
  use_sim_time:=true


⸻

## 2.5 Fix 3 — Visual odometry failure → ICP odometry (final working command)

If you see errors like:
	•	Odom: quality=0
	•	Not enough inliers
	•	no odometry is provided

Switch to ICP odometry:

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

If processing is slow:

ros2 bag play lab_environment --clock --rate 0.5


⸻

# 3) GLIM SLAM (Ouster OS-1-32, LiDAR + IMU)

## 3.1 Configure host network (link-local)

Ouster uses 169.254.x.x. Host must be in same range.

Example:

sudo ip addr add 169.254.41.100/16 dev eno1

Optional:

ouster-cli discover


⸻

## 3.2 Configure Ouster ROS driver using YAML

Create ouster_params.yaml:

/ouster/os_driver:
  ros__parameters:
    sensor_hostname: "169.254.41.35"
    metadata_hostname: "169.254.41.35"
    lidar_port: 7502
    imu_port: 7503
    udp_dest_host: "169.254.41.100"
    auto_start: True
    proc_mask: 1

Launch driver:

ros2 launch ouster_ros driver.launch.py params_file:=./ouster_params.yaml

Verify topics:

ros2 topic list
ros2 topic echo /ouster/points --once


⸻

## 3.3 Run GLIM

ros2 run glim_ros glim_rosnode --ros-args \
  -p config_path:=/home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws/install/glim/share/glim/config

RViz:

ros2 run rviz2 rviz2 -d install/glim/share/glim/config/glim.rviz

If GLIM warns about large time gaps:
	•	reduce bag playback rate (--rate 0.5)
	•	monitor CPU load
	•	reduce visualization load

⸻

# Model Weights (Details)

GroundingDINO install issue (we hit this)

If pip fails while “getting requirements to build wheel”, install without build isolation:

pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git


⸻

# Running Evaluation + Generating Evidence

Evaluate an offline run

PYTHONPATH=. python -m scripts.evaluate_system \
  --run-dir results/realsense_demo_run \
  --queries configs/eval/test_queries.json \
  --scenarios configs/eval/test_scenarios.json \
  --verbose

Outputs:
	•	results/evaluations/evaluation_<run>.json
	•	plots under results/evaluations/

Create a video from debug frames (optional)

## Linux:

sudo apt install ffmpeg
cd results/<run_name>/debug/frames
ffmpeg -framerate 5 -i %06d.png -pix_fmt yuv420p demo_detection.mp4

## macOS:

brew install ffmpeg
cd results/<run_name>/debug/frames
ffmpeg -framerate 5 -i %06d.png -pix_fmt yuv420p demo_detection.mp4


⸻

# Troubleshooting (Known Issues + Fixes)

### 1) ModuleNotFoundError: No module named 'src'

Run:

PYTHONPATH=. python -m scripts.run_system --dry-run

### 2) RTAB-Map “Did not receive data since 5 seconds”

Cause: rosbag time vs wall time mismatch
Fix:
	•	use ros2 bag play ... --clock
	•	use use_sim_time:=true

### 3) RTAB-Map “time difference between rgb and depth frames is high”

Fix:
	•	--clock, use_sim_time:=true
	•	approx_sync:=true
	•	increase queue size if needed

### 4) RTAB-Map visual odometry fails (quality=0, not enough inliers)

Fix:
	•	switch to ICP odometry: --Odom/Strategy 1
	•	slow down playback: --rate 0.5

### 5) GLIM warning “large time gap between consecutive LiDAR frames”

Fix:
	•	reduce bag playback rate
	•	monitor CPU usage
	•	reduce visualization frequency / point cloud load

⸻

### Repository Layout

```text
configs/eval/          # evaluation configs
data/
├── realsense_runs/    # offline datasets
├── bags/              # rosbags (optional)
models/                # checkpoints
results/               # outputs + evaluation plots
scripts/               # entry points
src/                   # core modules
tests/                 # tests
documentation.md        # extended documentation
Images/                 # screenshots for report/presentation
```
⸻

# Authors
	•	Andrés Santiago Santafé Silva
	•	Ana Maria Oliveros Ossa

⸻

# Credits / Licenses

This project builds on open-source tools including ROS 2, RTAB-Map, GLIM, Segment Anything, and GroundingDINO.
Please refer to each upstream repository for licensing and model usage terms.

---
