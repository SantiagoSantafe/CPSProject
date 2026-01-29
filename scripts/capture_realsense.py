#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np
import cv2

def main():
    import pyrealsense2 as rs

    out_dir = Path("data/realsense_demo").resolve()
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    # 640x480 is plenty for a demo; stable and fast
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Get intrinsics (color stream)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy, "width": intr.width, "height": intr.height}

    # Depth scale (meters per unit)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    intrinsics["depth_scale"] = depth_scale

    with open(out_dir / "intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=2)
    print(f"[OK] Wrote intrinsics: {out_dir/'intrinsics.json'}")
    print(f"Depth scale: {depth_scale} meters/unit")

    # Warmup
    for _ in range(30):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    print("\n[INFO] Capturing 60 frames (~2s). Move slowly and keep objects visible.")
    n_frames = 60
    t0 = time.time()

    try:
        for i in range(n_frames):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())  # BGR
            depth_raw = np.asanyarray(depth.get_data()).astype(np.uint16)  # z16

            # Convert depth to meters float32 (this matches your project_to_3d expecting meters)
            depth_m = depth_raw.astype(np.float32) * depth_scale

            cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), color_img)
            np.save(depth_dir / f"{i:06d}.npy", depth_m)

            if i % 10 == 0:
                print(f"  saved frame {i}/{n_frames}")

    finally:
        pipeline.stop()

    print(f"\n[OK] Capture done in {time.time()-t0:.2f}s")
    print(f"Dataset at: {out_dir}")

if __name__ == "__main__":
    main()