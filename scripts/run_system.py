#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from src.perception.open_vocab_detector import OpenVocabularyDetector
from src.mapping.semantic_mapper import SemanticMapper
from src.navigation.navigation_controller import NavigationController


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "results")).resolve()


def load_data(data_path: str, max_frames: int | None = None):
    """
    Load RGB-D dataset from disk.

    Supported formats:
    - RealSense capture (meta.json + rgb/*.png + depth/*.png)
    - Legacy format (intrinsics.json + rgb/*.png + depth/*.npy)

    Returns:
        rgb_images: list[np.ndarray]  (H,W,3) RGB
        depth_images: list[np.ndarray] (H,W) depth in METERS
        poses: None
        intrinsics: dict {fx, fy, cx, cy}
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    rgb_dir = data_path / "rgb"
    depth_dir = data_path / "depth"

    if not rgb_dir.exists() or not depth_dir.exists():
        raise FileNotFoundError("Expected subfolders: rgb/ and depth/")

    # ---------------------------
    # Load intrinsics
    # ---------------------------
    intrinsics = None
    depth_scale = 1.0

    meta_path = data_path / "meta.json"
    intr_path = data_path / "intrinsics.json"

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)

        intr = meta["intrinsics"]
        intrinsics = {
            "fx": float(intr["fx"]),
            "fy": float(intr["fy"]),
            "cx": float(intr["cx"]),
            "cy": float(intr["cy"]),
        }
        depth_scale = float(meta.get("depth_scale", 1.0))

    elif intr_path.exists():
        with open(intr_path, "r") as f:
            intrinsics = json.load(f)

    else:
        raise FileNotFoundError(
            f"Missing intrinsics. Expected one of:\n"
            f"  - {meta_path}\n"
            f"  - {intr_path}"
        )

    # ---------------------------
    # Load images
    # ---------------------------
    rgb_files = sorted(rgb_dir.glob("*.png"))
    depth_files = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.npy"))

    if max_frames:
        rgb_files = rgb_files[:max_frames]
        depth_files = depth_files[:max_frames]

    rgb_images = []
    depth_images = []

    for rgb_f, depth_f in zip(rgb_files, depth_files):
        rgb = cv2.imread(str(rgb_f), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_f}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_images.append(rgb)

        if depth_f.suffix == ".png":
            depth_raw = cv2.imread(str(depth_f), cv2.IMREAD_UNCHANGED)
            depth_m = depth_raw.astype(np.float32) * depth_scale
        else:
            depth_m = np.load(depth_f).astype(np.float32)

        depth_images.append(depth_m)

    print(f"[OK] Loaded {len(rgb_images)} RGB-D frames from {data_path}")
    print(f"[OK] Intrinsics: {intrinsics}")

    poses = None
    return rgb_images, depth_images, poses, intrinsics

def _dry_run_data(max_frames: int = 2):
    rgb_images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(max_frames)]
    depth_images = [np.ones((64, 64), dtype=np.float32) * 1.5 for _ in range(max_frames)]
    poses = [np.eye(4, dtype=np.float32) for _ in range(max_frames)]
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0}
    return rgb_images, depth_images, poses, intrinsics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build semantic map + run navigation commands.")
    parser.add_argument("--data", type=str, default=None, help="Path to offline dataset folder (optional)")
    parser.add_argument("--results-dir", type=str, default=None, help="Where to write outputs (default: RESULTS_DIR or ./results)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Run without models/sensors, using synthetic data (for CI/tests)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    np.random.seed(args.seed)

    out_dir = Path(args.results_dir).resolve() if args.results_dir else _default_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    if args.dry_run:
        rgb_images, depth_images, poses, intrinsics = _dry_run_data(max_frames=args.max_frames or 2)
        # In dry-run we also stub detector to avoid heavy models
        class StubDetector:
            def __init__(self):
                self.frame = 0

            def detect_objects(self, image, text_queries, background_queries=None):
                h, w = image.shape[:2]
                m1 = np.zeros((h, w), dtype=np.uint8)
                m2 = np.zeros((h, w), dtype=np.uint8)

                # two boxes/masks
                m1[10:26, 10:26] = 1
                m2[30:46, 30:46] = 1

                if self.frame == 0:
                    dets = [
                        {"label": "chair", "score": 0.90, "mask": m1, "box": [10, 10, 16, 16]},
                        {"label": "table", "score": 0.85, "mask": m2, "box": [30, 30, 16, 16]},
                    ]
                else:
                    dets = [
                        {"label": "computer", "score": 0.88, "mask": m1, "box": [10, 10, 16, 16]},
                        {"label": "bookshelf", "score": 0.82, "mask": m2, "box": [30, 30, 16, 16]},
                    ]

                self.frame += 1
                return dets

            def project_to_3d(self, object_mask, depth_image, camera_intrinsics):
                # centroid depends on which mask block we got (rough heuristic)
                v, u = np.where(object_mask > 0)
                if len(u) == 0:
                    return None
                u_mean = float(np.mean(u))
                if u_mean < 20:
                    centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                else:
                    centroid = np.array([2.0, 0.5, 0.0], dtype=np.float32)

                return {
                    "centroid": centroid,
                    "dimensions": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    "points_3d": np.zeros((10, 3), dtype=np.float32),
                }

        detector = StubDetector()
    else:
        # Call load_data in a monkeypatch-friendly way (supports both signatures):
        #   load_data()
        #   load_data(data_path=..., max_frames=...)
        try:
            rgb_images, depth_images, poses, intrinsics = load_data(data_path=args.data, max_frames=args.max_frames)
        except TypeError:
            # Backwards compatibility for older load_data() implementations/tests
            rgb_images, depth_images, poses, intrinsics = load_data()
        detector = OpenVocabularyDetector()

    mapper = SemanticMapper()

    target_queries = ["chair", "table", "computer", "bookshelf"]
    background_queries = ["wall", "floor", "ceiling", "window"]

    if args.verbose:
        print("Building semantic map...")

    semantic_map = mapper.build_semantic_map(
        rgb_images, depth_images, poses, intrinsics,
        detector, target_queries, background_queries
    )

    # Save semantic map artifact
    _write_json(out_dir / "semantic_map.json", semantic_map)

    nav_controller = NavigationController(semantic_map)

    test_commands = [
        "navigate to the chair",
        "go to the table",
        "find the computer",
        "locate the bookshelf"
    ]

    current_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    nav_results: List[Dict[str, Any]] = []

    for command in test_commands:
        result = nav_controller.execute_navigation_command(command, current_position)
        nav_results.append({"command": command, "start_position": current_position.tolist(), "result": result})

        if result.get("success") and result.get("goal_pose") is not None:
            goal = np.array(result["goal_pose"], dtype=np.float32).reshape(-1)
            current_position = np.array([goal[0], goal[1], 0.0], dtype=np.float32)

    _write_json(out_dir / "nav_results.json", nav_results)

    if args.verbose:
        print(f"[OK] Wrote: {out_dir / 'semantic_map.json'}")
        print(f"[OK] Wrote: {out_dir / 'nav_results.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())