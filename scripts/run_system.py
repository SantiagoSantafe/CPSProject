#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from src.perception.open_vocab_detector import OpenVocabularyDetector
from src.mapping.semantic_mapper import SemanticMapper
from src.navigation.navigation_controller import NavigationController


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "results")).resolve()


def load_data(data_path: Optional[str] = None, max_frames: Optional[int] = None):
    """
    Minimal loader:
    - If data_path is None: raise (unless dry_run)
    - Expected structure if data_path provided (simple offline):
        data_path/
          rgb_000.npy, rgb_001.npy, ...
          depth_000.npy, depth_001.npy, ...
          poses.npy   (N,4,4) optional
          intrinsics.json  with fx,fy,cx,cy
    """
    if data_path is None:
        raise NotImplementedError("Provide --data <path> or use --dry-run")

    p = Path(data_path).resolve()
    intr_path = p / "intrinsics.json"
    if not intr_path.exists():
        raise FileNotFoundError(f"Missing intrinsics.json at {intr_path}")

    intrinsics = json.loads(intr_path.read_text(encoding="utf-8"))

    rgb_files = sorted(p.glob("rgb_*.npy"))
    depth_files = sorted(p.glob("depth_*.npy"))
    if not rgb_files or not depth_files:
        raise FileNotFoundError("No rgb_*.npy or depth_*.npy found in --data folder")

    n = min(len(rgb_files), len(depth_files))
    if max_frames is not None:
        n = min(n, max_frames)

    rgb_images = [np.load(f) for f in rgb_files[:n]]
    depth_images = [np.load(f) for f in depth_files[:n]]

    poses = None
    poses_path = p / "poses.npy"
    if poses_path.exists():
        poses_all = np.load(poses_path)
        poses = [poses_all[i] for i in range(min(n, poses_all.shape[0]))]

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