#!/usr/bin/env python3
"""
scripts/visualize_offline_run.py

Create plots + a small text summary for slides from an evaluation JSON.

Usage:
  python -m scripts.visualize_offline_run --eval results/evaluations/evaluation_<run>.json --outdir results/reports/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _bar_plot(metrics: Dict[str, float], title: str, outpath: Path) -> None:
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _hist_plot(values, title: str, xlabel: str, outpath: Path) -> None:
    v = [float(x) for x in values if x is not None and np.isfinite(x)]
    plt.figure()
    plt.hist(v, bins=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _status_plot(counts: Dict[str, int], title: str, outpath: Path) -> None:
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="Path to evaluation_*.json")
    ap.add_argument("--outdir", default=None, help="Output directory (default: results/reports/<eval_name>)")
    args = ap.parse_args(argv)

    eval_path = Path(args.eval).expanduser().resolve()
    data = load_json(eval_path)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else Path("results/reports") / eval_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    obj = data["object_retrieval"]
    nav = data["navigation"]

    # --- Object metrics plot
    topk = 5
    metrics = {
        "Top-1(ID)": obj.get("top1_accuracy") or 0.0,
        f"Top-{topk}(ID)": obj.get("topk_accuracy") or 0.0,
        "MRR": obj.get("mean_reciprocal_rank") or 0.0,
        f"mAP@{topk}": obj.get("map_at_k") or 0.0,
        f"Top-{topk}(Label)": obj.get("topk_label_accuracy") or 0.0,
    }
    _bar_plot(metrics, "Object Retrieval Metrics", outdir / "object_metrics.png")

    # --- Navigation status/modes
    _status_plot(nav.get("mode_breakdown", {}), "Navigation Mode Breakdown", outdir / "nav_modes.png")
    _status_plot(nav.get("status_breakdown", {}), "Navigation Status Breakdown", outdir / "nav_status.png")

    # --- Goal error hist
    goal_errors = [x.get("goal_xy_error") for x in nav.get("per_scenario", [])]
    if any(x is not None for x in goal_errors):
        _hist_plot(goal_errors, "Goal XY Error Histogram", "Error (m)", outdir / "goal_xy_error_hist.png")

    # --- Retrieval error modes
    _status_plot(obj.get("error_modes", {}), "Retrieval Error Modes", outdir / "retrieval_error_modes.png")

    # --- Slides summary text
    lines = []
    lines.append("=== SLIDES SUMMARY ===")
    lines.append(f"Run: {data.get('run_dir')}")
    lines.append(f"Objects in map: {len(load_json(Path(data['semantic_map_path'])))}")
    if obj.get("top1_accuracy") is not None:
        lines.append(f"Retrieval Top-1(ID): {obj['top1_accuracy']:.3f}")
        lines.append(f"Retrieval Top-5(ID): {obj['topk_accuracy']:.3f}")
        lines.append(f"MRR: {obj.get('mean_reciprocal_rank', 0.0):.3f}")
        lines.append(f"mAP@5: {obj.get('map_at_k', 0.0):.3f}")
    if obj.get("topk_label_accuracy") is not None:
        lines.append(f"Retrieval Top-5(Label): {obj['topk_label_accuracy']:.3f}")
    if obj.get("position_success_rate") is not None:
        lines.append(f"Position Success (<=thr): {obj['position_success_rate']:.3f}")
        if obj.get("avg_position_error") is not None:
            lines.append(f"Avg Position Error: {obj['avg_position_error']:.3f} m")
    lines.append(f"Navigation Success Rate: {nav['success_rate']:.3f}")
    if nav.get("avg_goal_xy_error") is not None:
        lines.append(f"Avg Goal XY Error: {nav['avg_goal_xy_error']:.3f} m")
    lines.append(f"Nav modes: {nav.get('mode_breakdown', {})}")
    lines.append(f"Retrieval modes: {obj.get('error_modes', {})}")

    (outdir / "slides_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] Wrote report to: {outdir}")
    print("[OK] Files:")
    for f in sorted(outdir.glob("*.png")):
        print(f"  - {f.name}")
    print(f"  - slides_summary.txt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())