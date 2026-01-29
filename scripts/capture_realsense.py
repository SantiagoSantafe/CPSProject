#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs


def _timestamp_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a short RGB-D sequence from Intel RealSense D435i.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output folder (default: data/realsense_runs/<timestamp>)")
    parser.add_argument("--num-frames", type=int, default=60, help="Frames to capture")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--align-to-color", action="store_true", help="Align depth to color frame")
    parser.add_argument("--visualize", action="store_true", help="Show live preview while capturing")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames before recording")
    args = parser.parse_args()

    run_id = _timestamp_run_id()
    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / "realsense_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    color_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    color_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    align = None
    if args.align_to_color:
        align = rs.align(rs.stream.color)

    # Intrinsics (from color stream)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    intrinsics = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy, "width": intr.width, "height": intr.height}

    meta = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(profile.get_device().get_info(rs.camera_info.name)),
        "serial": str(profile.get_device().get_info(rs.camera_info.serial_number)),
        "streams": {
            "color": {"width": args.width, "height": args.height, "fps": args.fps},
            "depth": {"width": args.width, "height": args.height, "fps": args.fps, "format": "z16"},
        },
        "intrinsics": intrinsics,
        "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
        "align_to_color": bool(args.align_to_color),
        "notes": "Depth is stored as uint16 PNG in raw units (z16). Convert to meters using depth_scale.",
    }

    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saving to: {out_dir.resolve()}")
    print("[i] Warming up...")
    for _ in range(args.warmup):
        frames = pipeline.wait_for_frames()
        if align is not None:
            frames = align.process(frames)

    print(f"[i] Recording {args.num_frames} frames...")
    t0 = time.time()

    try:
        for i in range(args.num_frames):
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())  # BGR8
            depth_img = np.asanyarray(depth.get_data())  # uint16

            cv2.imwrite(str(color_dir / f"{i:06d}.png"), color_img)
            cv2.imwrite(str(depth_dir / f"{i:06d}.png"), depth_img)

            if args.visualize:
                depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.03)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                preview = np.hstack([color_img, depth_vis])
                cv2.imshow("RealSense Capture (color | depth)", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[i] Early stop requested.")
                    break

        dt = time.time() - t0
        print(f"[OK] Captured {i+1} frames in {dt:.2f}s")

    finally:
        pipeline.stop()
        if args.visualize:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
