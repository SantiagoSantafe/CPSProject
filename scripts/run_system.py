#!/usr/bin/env python3
"""
Run the full pipeline:
1) Load RGB-D frames (offline dataset) OR dry-run synthetic frames
2) Detect open-vocabulary objects (CLIP + SAM) OR stub detection in dry-run
3) Build a semantic map
4) Run navigation queries
5) Write outputs to results folder
6) (Optional) Save debug overlays + detections.jsonl for visual evidence

Usage:
  python -m scripts.run_system --dry-run --max-frames 4 --results-dir results/demo --verbose --save-debug
  python -m scripts.run_system --data data/realsense_runs/<run_id> --max-frames 30 --results-dir results/realsense_demo --fast --save-debug --verbose
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import cv2

from src.mapping.semantic_mapper import SemanticMapper
from src.navigation.navigation_controller import NavigationController


# -----------------------------
# JSON safety (numpy -> python)
# -----------------------------
def _to_py(x: Any) -> Any:
    """Convert numpy scalars/arrays recursively to JSON-serializable Python types."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    return x


# -----------------------------
# Debug visualization helpers
# -----------------------------
def _draw_detections_overlay(rgb_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw bounding boxes + label + score on RGB input.
    Returns BGR image suitable for cv2.imwrite.
    """
    img_bgr = cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR).copy()

    for d in detections or []:
        label = str(d.get("label", "obj"))
        score = float(d.get("score", 0.0))
        box = d.get("box", None)
        if box and len(box) == 4:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                f"{label} {score:.2f}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
    return img_bgr


# -----------------------------
# Data loading (offline dataset)
# -----------------------------
def load_data(data_path: str, max_frames: int | None = None):
    """
    Load RGB-D dataset from disk.

    Supported formats:
    - RealSense capture (meta.json + rgb/*.png + depth/*.png)
    - Legacy format (intrinsics.json + rgb/*.png + depth/*.png or depth/*.npy)

    Returns:
        rgb_images: list[np.ndarray]  (H,W,3) RGB
        depth_images: list[np.ndarray] (H,W) depth in METERS
        poses: None
        intrinsics: dict {fx, fy, cx, cy}
    """
    if data_path is None:
        raise ValueError("--data is required unless --dry-run is set")

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError("Expected subfolders: rgb/ and depth/")

    # Default depth scale (if missing meta): typical RealSense units -> meters
    depth_scale = 0.001

    meta_path = data_path / "meta.json"
    intr_path = data_path / "intrinsics.json"

    intrinsics: Dict[str, float] | None = None

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        intr = meta.get("intrinsics", {})
        intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }
        if meta.get("depth_scale") is not None:
            depth_scale = float(meta["depth_scale"])

    elif intr_path.exists():
        intr = json.loads(intr_path.read_text(encoding="utf-8"))
        intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }
        depth_scale = float(intr.get("depth_scale", depth_scale))

    else:
        raise FileNotFoundError(
            f"Missing intrinsics. Expected one of:\n"
            f"  - {meta_path}\n"
            f"  - {intr_path}"
        )

    rgb_files = sorted(list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.jpeg")))
    depth_files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.npy")))

    if not rgb_files:
        raise FileNotFoundError(f"No RGB images found in: {rgb_dir}")
    if not depth_files:
        raise FileNotFoundError(f"No depth files found in: {depth_dir}")

    if max_frames:
        rgb_files = rgb_files[:max_frames]
        depth_files = depth_files[:max_frames]

    n = min(len(rgb_files), len(depth_files))
    if n == 0:
        raise FileNotFoundError("No paired RGB/Depth frames found.")
    if len(rgb_files) != len(depth_files):
        print(f"[WARN] rgb frames={len(rgb_files)} depth frames={len(depth_files)}; using first {n} pairs", flush=True)

    rgb_images: List[np.ndarray] = []
    depth_images: List[np.ndarray] = []

    for rgb_f, depth_f in zip(rgb_files[:n], depth_files[:n]):
        rgb_bgr = cv2.imread(str(rgb_f), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_f}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb)

        if depth_f.suffix == ".png":
            depth_raw = cv2.imread(str(depth_f), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise RuntimeError(f"Failed to read depth image: {depth_f}")

            if depth_raw.dtype in (np.uint16, np.int16):
                depth_m = depth_raw.astype(np.float32) * depth_scale
            else:
                depth_m = depth_raw.astype(np.float32)
        else:
            # .npy assumed already meters
            depth_m = np.load(depth_f).astype(np.float32)

        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_images.append(depth_m)

    print(f"[OK] Loaded {len(rgb_images)} RGB-D frames from {data_path}", flush=True)
    print(f"[OK] Intrinsics: {intrinsics}", flush=True)
    print(f"[OK] Depth scale (m/unit): {depth_scale}", flush=True)

    poses = None
    return rgb_images, depth_images, poses, intrinsics


def _default_results_dir() -> Path:
    return Path(os.environ.get("RESULTS_DIR", "results")).resolve()


def _dry_run_data(max_frames: int = 2):
    H, W = 64, 64
    rgb_images, depth_images = [], []
    for _ in range(max_frames):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[16:32, 16:32, :] = 255
        depth = np.ones((H, W), dtype=np.float32) * 1.0
        rgb_images.append(rgb)
        depth_images.append(depth)

    poses = None
    intrinsics = {"fx": 60.0, "fy": 60.0, "cx": W / 2.0, "cy": H / 2.0}
    return rgb_images, depth_images, poses, intrinsics


def _resolve_sam_checkpoint(repo_root: Path) -> Optional[str]:
    candidates = [
        repo_root / "models" / "sam_vit_h_4b8939.pth",
        repo_root / "models" / "sam_vit_l_0b3195.pth",
        repo_root / "models" / "sam_vit_b_01ec64.pth",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main(argv: Optional[List[str]] = None) -> int:
    print("[run_system] main() started", flush=True)

    parser = argparse.ArgumentParser(description="Build semantic map + run navigation commands.")
    parser.add_argument("--data", type=str, default=None, help="Path to offline dataset folder (optional)")
    parser.add_argument("--results-dir", type=str, default=None, help="Where to write outputs (default: RESULTS_DIR or ./results)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Run without models/sensors, using synthetic data (for CI/tests)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true", help="CPU-friendly settings (faster demo)")

    # Debug evidence
    parser.add_argument("--save-debug", action="store_true", help="Save per-frame overlays + detections.jsonl")
    parser.add_argument("--debug-dir", type=str, default=None, help="Debug output dir (default: <results-dir>/debug)")

    args = parser.parse_args(argv)
    np.random.seed(args.seed)

    out_dir = Path(args.results_dir).resolve() if args.results_dir else _default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup debug dirs AFTER args/out_dir exist
    dbg_frames_dir = None
    dbg_jsonl_path = None

    if args.save_debug:
        debug_dir = (
            Path(args.debug_dir).expanduser().resolve()
            if args.debug_dir
            else (out_dir / "debug")
        )

        dbg_frames_dir = debug_dir / "frames"
        dbg_frames_dir.mkdir(parents=True, exist_ok=True)

        dbg_jsonl_path = debug_dir / "detections.jsonl"

        print(f"[OK] Debug enabled -> {debug_dir}", flush=True)

    # --- Load data ---
    if args.dry_run:
        rgb_images, depth_images, poses, intrinsics = _dry_run_data(max_frames=args.max_frames or 2)

        class StubDetector:
            def detect_objects(self, image, text_queries, background_queries=None):
                m = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                m[16:32, 16:32] = 1
                return [{"label": "chair", "score": 0.9, "mask": m, "box": [16, 16, 16, 16]}]

            def project_to_3d(self, object_mask, depth_image, camera_intrinsics):
                return {
                    "centroid": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    "dimensions": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    "points_3d": np.zeros((10, 3), dtype=np.float32),
                }

        detector = StubDetector()

    else:
        rgb_images, depth_images, poses, intrinsics = load_data(data_path=args.data, max_frames=args.max_frames)

        from src.perception.open_vocab_detector import OpenVocabularyDetector

        repo_root = Path(__file__).resolve().parents[1]
        ckpt = _resolve_sam_checkpoint(repo_root)
        if ckpt is None:
            raise FileNotFoundError(
                f"SAM checkpoint not found. Put one of these in {repo_root/'models'}:\n"
                f"  - sam_vit_h_4b8939.pth\n  - sam_vit_l_0b3195.pth\n  - sam_vit_b_01ec64.pth"
            )

        detector = OpenVocabularyDetector(
            sam_checkpoint=ckpt,
            fast=args.fast,
        )

    # --- Build semantic map ---
    print("Building semantic map...", flush=True)
    mapper = SemanticMapper()
    queries = ["chair", "table", "computer", "bookshelf"]

    for frame_idx, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
        if args.verbose:
            print(f"\n[Frame {frame_idx+1}/{len(rgb_images)}]", flush=True)

        dets = detector.detect_objects(rgb, queries)
        if args.verbose:
            print(f"Found {len(dets)} objects", flush=True)

        # Save per-frame evidence
        if args.save_debug and dbg_frames_dir is not None and dbg_jsonl_path is not None:
            overlay_bgr = _draw_detections_overlay(rgb, dets)
            cv2.imwrite(str(dbg_frames_dir / f"{frame_idx:06d}.png"), overlay_bgr)

            rec = {
                "frame": int(frame_idx),
                "detections": [
                    {"label": d.get("label"), "score": float(d.get("score", 0.0)), "box": d.get("box")}
                    for d in (dets or [])
                ],
            }
            with open(dbg_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        for det in dets:
            proj = detector.project_to_3d(det["mask"], depth, intrinsics)
            if proj is None:
                continue
            mapper.update(det["label"], proj["centroid"], proj["dimensions"], det.get("score", 0.0))

    semantic_map = mapper.get_semantic_map()
    print("\n" + "=" * 50, flush=True)
    print("BUILD COMPLETE", flush=True)
    print("=" * 50, flush=True)

    if args.verbose:
        print("\nSEMANTIC MAP", flush=True)
        for oid, obj in semantic_map.items():
            c = np.asarray(obj.get("centroid", [0, 0, 0]), dtype=np.float32).reshape(-1)
            print(f"\n[{oid}] {obj.get('label')}", flush=True)
            print(f"    Position: X={c[0]:.2f}m, Y={c[1]:.2f}m, Z={c[2]:.2f}m", flush=True)
            print(f"    Confidence: {float(obj.get('confidence', 0.0)):.2f}", flush=True)
            print(f"    Observations: {int(obj.get('observations', 0))}", flush=True)

    # --- Navigation demo ---
    nav = NavigationController(semantic_map)
    nav_commands = [
        "navigate to the chair",
        "navigate to the table",
        "navigate to the computer",
        "navigate to the bookshelf",
    ]
    current_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    nav_results = []
    for cmd in nav_commands:
        res = nav.execute_navigation_command(cmd, current_pos)
        nav_results.append(res)

    # --- Write outputs (JSON-safe) ---
    semantic_map_path = out_dir / "semantic_map.json"
    nav_results_path = out_dir / "nav_results.json"

    semantic_map_path.write_text(json.dumps(_to_py(semantic_map), indent=2), encoding="utf-8")
    nav_results_path.write_text(json.dumps(_to_py(nav_results), indent=2), encoding="utf-8")

    print(f"[OK] Wrote: {semantic_map_path}", flush=True)
    print(f"[OK] Wrote: {nav_results_path}", flush=True)

    # Optional: run evaluation
    eval_script = Path(__file__).resolve().parent / "evaluate_system.py"
    queries_cfg = Path("configs/eval/test_queries.json")
    scenarios_cfg = Path("configs/eval/test_scenarios.json")
    if eval_script.exists() and queries_cfg.exists() and scenarios_cfg.exists():
        try:
            import subprocess
            eval_out = out_dir.parent / "evaluations"
            eval_out.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "python", "-m", "scripts.evaluate_system",
                    "--run-dir", str(out_dir),
                    "--queries", str(queries_cfg),
                    "--scenarios", str(scenarios_cfg),
                    "--out", str(eval_out / f"evaluation_{out_dir.name}.json"),
                ],
                check=False,
            )
        except Exception as e:
            print(f"[WARN] Evaluation step failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    import sys, traceback

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("[FATAL] Unhandled exception in run_system.py", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)