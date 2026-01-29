#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser("Record RealSense RGB-D frames to disk (offline dataset).")
    parser.add_argument("--out", type=str, default="data/realsense_runs", help="Base output directory")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--frames", type=int, default=None, help="How many frames to record")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to record (alternative to --frames)")

    parser.add_argument("--warmup", type=int, default=30, help="Warmup frames (not saved)")
    parser.add_argument("--viewer", action="store_true", help="Show live preview while recording")
    args = parser.parse_args()

    if args.frames is None and args.duration is None:
        args.frames = 120  # default
    if args.frames is None:
        args.frames = int(args.duration * args.fps)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out).expanduser().resolve() / ts
    rgb_dir = run_dir / "rgb"
    depth_dir = run_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit

    meta = {
        "created_at": ts,
        "device": "realsense",
        "streams": {"color": [args.width, args.height, args.fps], "depth": [args.width, args.height, args.fps]},
        "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy},
        "depth_scale": depth_scale,
        "notes": "depth png is uint16 in sensor units; multiply by depth_scale to get meters",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Writing dataset to: {run_dir}")
    print(f"[OK] Intrinsics: {meta['intrinsics']}")
    print(f"[OK] Depth scale (m/unit): {depth_scale}")

    preview_saved = False

    try:
        for _ in range(args.warmup):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

        print(f"[OK] Warmup done. Recording {args.frames} frames...")

        for i in range(args.frames):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            color_img = np.asanyarray(color.get_data())  # BGR
            depth_img = np.asanyarray(depth.get_data())  # uint16

            rgb_path = rgb_dir / f"{i:06d}.png"
            dep_path = depth_dir / f"{i:06d}.png"

            cv2.imwrite(str(rgb_path), color_img)
            cv2.imwrite(str(dep_path), depth_img)

            if not preview_saved:
                cv2.imwrite(str(run_dir / "preview_rgb.png"), color_img)
                preview_saved = True

            if args.viewer:
                depth_vis = cv2.convertScaleAbs(depth_img, alpha=255.0 / max(1, int(depth_img.max())))
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("RGB", color_img)
                cv2.imshow("Depth (vis)", depth_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Early stop requested.")
                    break

            if (i + 1) % 10 == 0:
                print(f"  saved {i+1}/{args.frames}")

    finally:
        pipeline.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("[OK] Done.")


if __name__ == "__main__":
    main()