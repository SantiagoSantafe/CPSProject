Below are (1) a polished final documentation file you can paste into documentation.md, and (2) an improved, installation-focused README.md (English) so anyone can reproduce your results.

⸻

1) documentation.md (FINAL — copy/paste)

# CPSProject — Final Documentation
## SLAM + Open-Vocabulary Semantic Mapping + Offline Evaluation (ROS 2 + Python)

**Last updated:** January 29, 2026  
**ROS 2:** Jazzy  
**Sensors used:** Orbbec Femto Mega (RGB-D), Intel RealSense D435i (RGB-D), Ouster OS-1-32 (LiDAR + IMU)  
**Main outputs:** SLAM maps (RTAB-Map / GLIM) + semantic map + navigation demo + offline evaluation metrics + visual evidence

---

## Table of Contents
- [CPSProject — Final Documentation](#cpsproject--final-documentation)
  - [SLAM + Open-Vocabulary Semantic Mapping + Offline Evaluation (ROS 2 + Python)](#slam--open-vocabulary-semantic-mapping--offline-evaluation-ros-2--python)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [System Architecture](#system-architecture)
    - [A) Online / Sensor-side (ROS 2)](#a-online--sensor-side-ros-2)
    - [B) Offline / Evaluation-side (Python)](#b-offline--evaluation-side-python)
  - [Repository Structure](#repository-structure)
  - [Environment Setup](#environment-setup)
    - [Python environment (recommended on all OS)](#python-environment-recommended-on-all-os)
  - [2) `README.md` (Improved — installation + quickstart, English)](#2-readmemd-improved--installation--quickstart-english)
- [CPSProject](#cpsproject)
  - [Table of Contents](#table-of-contents-1)
  - [Requirements](#requirements)
    - [Supported OS](#supported-os)
    - [Python](#python)
    - [Hardware](#hardware)
  - [Quickstart (Offline Pipeline)](#quickstart-offline-pipeline)

---

## Project Goal

This project demonstrates an end-to-end robotics perception pipeline:

- **SLAM** to build geometric maps from sensors (RGB-D / LiDAR).
- **Open-vocabulary object detection + segmentation** (GroundingDINO + SAM; CLIP/OpenCLIP-style scoring) to recognize objects from natural-language queries.
- **Semantic mapping** to accumulate detected instances into a global map (3D centroids + confidence + observations).
- **Text navigation** (command → target object → goal pose / waypoint selection).
- **Offline evaluation** producing **quantitative metrics** and **visual evidence** suitable for presentations.

Key design choice: the semantic pipeline is **offline-first**, enabling reproducible demos without having the sensor connected during runtime.

---

## System Architecture

### A) Online / Sensor-side (ROS 2)
- **RGB-D SLAM:** Orbbec → RTAB-Map
- **LiDAR-IMU SLAM:** Ouster → GLIM (+ RViz)

Outputs:
- RTAB-Map database / point clouds / trajectories
- GLIM map + trajectory + debug viewer / RViz

### B) Offline / Evaluation-side (Python)
Using saved RGB-D frames and intrinsics:
1. Detect target objects from **free-text queries**.
2. Segment masks (SAM) for spatial precision.
3. Project masks to 3D using depth + intrinsics.
4. Fuse detections into a semantic map (instances).
5. Run text-navigation scenarios and score results.
6. Save debug frames + JSON outputs + plots.

---

## Repository Structure

Typical structure (matches your current tree):

CPSProject/
Images/                         # screenshots for report/presentation
configs/eval/                   # evaluation queries + scenarios
data/bags/                      # rosbags (ROS side)
data/realsense_runs/            # offline RGB-D datasets (recorded)
models/                         # model checkpoints (SAM, DINO, etc.)
results/                        # pipeline outputs + plots
scripts/                        # entry points (record/run/evaluate/visualize)
src/                            # core modules (perception/mapping/navigation)
tests/                          # unit tests
documentation.md                # this file
README.md
requirements.txt

---

## Environment Setup

### Python environment (recommended on all OS)
Use a virtual environment to avoid `externally-managed-environment` issues:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt

Model weights

Place weights in ./models/ (expected by the pipeline):
	•	SAM checkpoint (example): models/sam_vit_h_4b8939.pth
	•	GroundingDINO weights (example): models/groundingdino_swint_ogc.pth

If your repo includes a helper script:

python -m scripts.download_Weights

(Otherwise download manually and match file names used in your code/config.)

GroundingDINO installation note (important)

If pip fails while “building wheel” due to isolated builds and torch conflicts, install with:

pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/facebookresearch/segment-anything.git


⸻

RGB-D SLAM (Orbbec Femto Mega + RTAB-Map)

1) Install ROS 2 Jazzy

sudo apt update
sudo apt install ros-jazzy-desktop

2) Launch Orbbec driver

Reference driver: https://github.com/orbbec/OrbbecSDK_ROS2

ros2 launch orbbec_camera femto_mega.launch.py

3) Record and replay a bag with simulated time

Record:

ros2 bag record -a lab_environment

Replay:

ros2 bag play lab_environment --clock

4) Fix depth/RGB resolution mismatch (critical)

RTAB-Map requires aligned RGB and depth shapes. If RGB is 1280×720 and depth differs, RTAB-Map may refuse to process.

Run your resizing node (republish resized depth):

sudo apt install ros-jazzy-cv-bridge python3-opencv
chmod +x ~/resize_depth.py
python3 ~/resize_depth.py

5) Run RTAB-Map with sync + simulated time

ros2 launch rtabmap_launch rtabmap.launch.py \
  args:="--delete_db_on_start" \
  depth_topic:=/camera/depth/image_resized \
  rgb_topic:=/camera/color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  depth_camera_info_topic:=/camera/depth/camera_info \
  approx_sync:=true \
  frame_id:=camera_link \
  use_sim_time:=true

6) Fix “visual odometry failed” using ICP strategy

If you see:
	•	Odom: quality=0
	•	Registration failed: Not enough inliers
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

Optional: slow down playback if ICP cannot keep up:

ros2 bag play lab_environment --clock --rate 0.5


⸻

LiDAR SLAM (Ouster + GLIM)

1) Host network configuration (link-local)

Ouster often uses link-local 169.254.x.x. The host must be on the same range.

Example:

sudo ip addr add 169.254.41.100/16 dev eno1

Optional discovery:

ouster-cli discover

2) Ouster ROS2 driver configuration (YAML)

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

Launch:

ros2 launch ouster_ros driver.launch.py params_file:=./ouster_params.yaml

Verify:

ros2 topic list
ros2 topic echo /ouster/points --once

3) Run GLIM (CPU modules)

ros2 run glim_ros glim_rosnode --ros-args \
  -p config_path:=/home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws/install/glim/share/glim/config

RViz:

ros2 run rviz2 rviz2 -d install/glim/share/glim/config/glim.rviz

4) If GLIM warns about time gaps

Example warning:

[glim] [warning] large time gap between consecutive LiDAR frames!!

Typical fixes:
	•	Reduce bag playback rate (--rate 0.5)
	•	Reduce visualization load
	•	Check CPU saturation and ROS QoS / buffering

⸻

Offline Semantic Mapping + Text Navigation

This pipeline is independent from ROS at runtime. It uses a recorded dataset folder containing:
	•	RGB frames
	•	Depth frames
	•	meta.json with intrinsics + depth scale

Key scripts
	•	scripts/run_system.py — runs detection → mapping → navigation → saves outputs + debug
	•	scripts/evaluate_system.py — computes metrics and saves evaluation JSON + plots
	•	scripts/record_realsense.py — records RealSense dataset to disk
	•	scripts/webcam_detect.py — quick 2D demo (live webcam) for open-vocab detection

Outputs (per run)
	•	results/<run>/semantic_map.json
	•	results/<run>/nav_results.json
	•	results/<run>/debug/frames/*.png (overlay evidence)
	•	results/<run>/debug/detections.jsonl (trace)

⸻

RealSense D435i Offline Dataset Recording

1) Verify device

python - << 'EOF'
import pyrealsense2 as rs
ctx = rs.context()
print("Devices:", ctx.devices.size())
EOF

2) Record dataset

python -m scripts.record_realsense --frames 120 --fps 15 --viewer

Example output:

data/realsense_runs/20260129_153237/
  rgb/000000.png ...
  depth/000000.png ...
  meta.json

meta.json stores intrinsics (fx, fy, cx, cy) and depth_scale.

⸻

End-to-End Offline Pipeline

1) Quick dry run

PYTHONPATH=. python -m scripts.run_system \
  --dry-run --max-frames 4 \
  --results-dir results/demo_run \
  --verbose

2) Full run on a recorded dataset

PYTHONPATH=. python -m scripts.run_system \
  --data data/realsense_runs/<RUN_TIMESTAMP> \
  --max-frames 30 \
  --results-dir results/realsense_demo_run \
  --fast \
  --save-debug \
  --verbose

3) Evaluate

PYTHONPATH=. python -m scripts.evaluate_system \
  --run-dir results/realsense_demo_run \
  --queries configs/eval/test_queries.json \
  --scenarios configs/eval/test_scenarios.json \
  --verbose


⸻

Evaluation Metrics (Definition of “Success”)

Object retrieval (per text query)
	•	Top-1 accuracy
	•	Top-K accuracy
	•	MRR
	•	mAP@K
	•	Error modes:
	•	OK
	•	WRONG_INSTANCE (label correct, wrong instance)
	•	WRONG_LABEL
	•	NOT_FOUND
	•	NO_GT

Navigation (per command scenario)
	•	Success rate
	•	Average XY goal error
	•	Status breakdown (e.g., OK, NOT_FOUND, WRONG_TARGET, MISSING)

Evaluation outputs:
	•	results/evaluations/evaluation_<run>.json
	•	plots under results/evaluations/ (e.g., histograms / breakdown charts)

⸻

Visual Evidence (Screenshots + Video)

Debug frames location

If --save-debug was used:

results/<run>/debug/frames/

Create a short demo video (ffmpeg)

Linux:

sudo apt install ffmpeg
cd results/<run>/debug/frames
ffmpeg -framerate 5 -i %06d.png -pix_fmt yuv420p demo_detection.mp4

macOS:

brew install ffmpeg
cd results/<run>/debug/frames
ffmpeg -framerate 5 -i %06d.png -pix_fmt yuv420p demo_detection.mp4


⸻

Troubleshooting

A) ModuleNotFoundError: No module named 'src'

Use:

PYTHONPATH=. python -m scripts.run_system --dry-run

B) Pip issues on Ubuntu (externally-managed-environment)

Use a venv:

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

C) GroundingDINO wheel/build failures

Use:

pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

D) SAM checkpoint missing

Ensure one of these exists:

models/sam_vit_h_4b8939.pth
models/sam_vit_l_0b3195.pth
models/sam_vit_b_01ec64.pth

E) Performance too slow on CPU
	•	Use --fast
	•	Reduce --max-frames
	•	Record at lower resolution if supported by your recorder

⸻

Conclusions

This project integrates:
	•	Practical SLAM pipelines: RTAB-Map (RGB-D) and GLIM (LiDAR-IMU), including real-world fixes (sync, resolution, ICP fallback).
	•	A complete offline semantic mapping pipeline with open-vocabulary queries.
	•	A reproducible evaluation framework producing metrics + failure modes + visual evidence.

⸻

Future Work
	1.	Stronger instance association (3D tracking / clustering) to reduce duplicates.
	2.	Embedding-based retrieval (nearest neighbor over object descriptors) for better ranking.
	3.	Real robot navigation integration (Nav2) for online execution of goal poses.
	4.	Automatic report generation (PDF/HTML) per run with plots + images.
	5.	GPU acceleration + batching for faster detection/segmentation.

⸻

Quick Commands

Record (RealSense)

python -m scripts.record_realsense --frames 120 --fps 15 --viewer

Run pipeline (offline)

PYTHONPATH=. python -m scripts.run_system --data data/realsense_runs/<RUN> --max-frames 30 --fast --save-debug --verbose

Evaluate

PYTHONPATH=. python -m scripts.evaluate_system --run-dir results/<RUN> --verbose

Make a demo video

cd results/<RUN>/debug/frames
ffmpeg -framerate 5 -i %06d.png -pix_fmt yuv420p demo_detection.mp4

---
