#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _resolve_repo_root() -> Path:
    # scripts/webcam_detect.py -> repo_root
    return Path(__file__).resolve().parents[1]


def _resolve_sam_checkpoint(repo_root: Path) -> str:
    candidates = [
        repo_root / "models" / "sam_vit_h_4b8939.pth",
        repo_root / "models" / "sam_vit_l_0b3195.pth",
        repo_root / "models" / "sam_vit_b_01ec64.pth",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "SAM checkpoint not found. Put one of these in ./models:\n"
        "  - sam_vit_h_4b8939.pth\n"
        "  - sam_vit_l_0b3195.pth\n"
        "  - sam_vit_b_01ec64.pth"
    )


def _parse_queries(text: str) -> List[str]:
    # allow commas or semicolons
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _draw_detections(img_bgr: np.ndarray, detections: List[dict]) -> np.ndarray:
    out = img_bgr.copy()
    for d in detections or []:
        box = d.get("box", None)
        if not box or len(box) != 4:
            continue
        x, y, w, h = map(int, box)
        label = str(d.get("label", "obj"))
        score = float(d.get("score", 0.0))

        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{label} ({score:.2f})",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return out


def main() -> int:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # Import detector from your project
    repo_root = _resolve_repo_root()
    sys.path.insert(0, str(repo_root))  # allows "src...." imports

    from src.perception.open_vocab_detector import OpenVocabularyDetector

    ckpt = _resolve_sam_checkpoint(repo_root)

    print("[INFO] Loading detector (OpenCLIP + SAM)...")
    detector = OpenVocabularyDetector(
        sam_checkpoint=ckpt,
        fast=True,  # faster CPU demo
    )
    print("[OK] Detector ready.")

    # Default queries
    target_queries = [
        "chair",
        "table",
        "computer",
        "bottle",
        "smartphone",
    ]
    background_queries = [
        "wall",
        "ceiling",
        "floor",
        "window",
        "background",
    ]

    print("\n====================")
    print("WEBCAM DETECTION")
    print("====================")
    print("Keys:")
    print("  [SPACE]  run detection on current frame")
    print("  [T]      edit target queries (comma-separated)")
    print("  [B]      edit background queries (comma-separated)")
    print("  [Q]      quit")
    print("\nCurrent target queries:", target_queries)
    print("Current background queries:", background_queries)
    print("====================\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return 1

    last_result = None  # last frame with detections drawn

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[ERROR] Could not read frame from camera.")
            break

        hud = frame_bgr.copy()
        cv2.putText(hud, "SPACE: detect | T: edit targets | B: edit background | Q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(hud, f"Targets: {', '.join(target_queries[:5])}{'...' if len(target_queries) > 5 else ''}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Live Camera", hud)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("t"):
            print("\n[INPUT] Write target queries (comma or ; separated).")
            print("Example: chair, table, bottle")
            new_text = input("targets> ").strip()
            if new_text:
                target_queries = _parse_queries(new_text)
                print("[OK] Updated targets:", target_queries)

        elif key == ord("b"):
            print("\n[INPUT] Write background queries (comma or ; separated).")
            print("Example: wall, ceiling, floor")
            new_text = input("background> ").strip()
            if new_text:
                background_queries = _parse_queries(new_text)
                print("[OK] Updated background:", background_queries)

        elif key == ord(" "):
            print("\n[INFO] Running detection on current frame...")
            # detector expects RGB in your pipeline? depends on your implementation.
            # Many OpenCV webcam frames are BGR. If your detector expects RGB, convert here:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            detections = detector.detect_objects(frame_rgb, target_queries, background_queries)

            # filter out background labels explicitly (sometimes it may output them)
            bg_set = set([b.lower() for b in background_queries])
            det_filtered = []
            for d in detections:
                lab = str(d.get("label", "")).lower()
                if lab in bg_set:
                    continue
                det_filtered.append(d)

            print(f"[OK] Detected {len(det_filtered)} target objects (filtered).")
            last_result = _draw_detections(frame_bgr, det_filtered)

            cv2.imshow("Detections (last)", last_result)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Exit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())