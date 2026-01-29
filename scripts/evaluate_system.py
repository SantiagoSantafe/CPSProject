#!/usr/bin/env python3
"""
Evaluate System (Final)
======================

Inputs:
- A run folder containing:
    semantic_map.json  (required)
    nav_results.json   (optional, only for extra diagnostics)
- Evaluation configs:
    configs/eval/test_queries.json
    configs/eval/test_scenarios.json

Outputs:
- results/evaluations/evaluation_<run_name>.json
- results/evaluations/object_retrieval_metrics.png
- results/evaluations/navigation_status_breakdown.png
- results/evaluations/navigation_goal_error_hist.png (if applicable)

CPU-only. No GPU/ROS required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# I/O helpers
# -------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "results")).resolve()

def _normalize_id(x: Any) -> str:
    """Canonical id for comparisons across JSON (str) vs python (int)."""
    if x is None:
        return ""
    return str(x)

def _normalize_semantic_map(semantic_map: Dict[Any, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert semantic_map keys to strings and ensure required fields exist.
    This makes evaluation stable w.r.t. JSON serialization.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (semantic_map or {}).items():
        sid = _normalize_id(k)
        out[sid] = dict(v)
    return out


# -------------------------
# Metrics helpers
# -------------------------
def _euclidean(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(a - b))


def _reciprocal_rank(gt_id: Any, ranked_ids: List[Any]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == gt_id:
            return 1.0 / i
    return 0.0


def _precision_recall_ap_at_k(relevant: List[int], k: int) -> Tuple[float, float, float]:
    """relevant: list of 0/1 across ranking positions."""
    if k <= 0:
        return 0.0, 0.0, 0.0
    rel_k = relevant[:k]
    retrieved_rel = sum(rel_k)
    precision = retrieved_rel / k

    total_rel = sum(relevant)
    recall = (retrieved_rel / total_rel) if total_rel > 0 else 0.0

    # AP@k
    ap = 0.0
    hits = 0
    for i, r in enumerate(rel_k, start=1):
        if r == 1:
            hits += 1
            ap += hits / i
    ap = (ap / min(total_rel, k)) if total_rel > 0 else 0.0
    return float(precision), float(recall), float(ap)


# -------------------------
# Retrieval evaluation
# -------------------------
def evaluate_object_retrieval(
    semantic_map: Dict[Any, Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    distance_threshold: float = 0.75,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generic evaluator (no CLIP required). Ranks objects by:
      exact label match > substring match > confidence

    Computes:
      - top1_accuracy, topk_accuracy, MRR
      - precision@k, recall@k, mAP@k
      - position_success_rate, avg_position_error (if gt_position exists)
    """
    if not semantic_map:
        raise ValueError("semantic_map is empty")

    obj_ids = list(semantic_map.keys())
    obj_labels = [str(semantic_map[i].get("label", "")) for i in obj_ids]

    any_id_gt = any(q.get("gt_id") is not None for q in test_queries)
    any_pos_gt = any(q.get("gt_position") is not None for q in test_queries)

    top1_hits = 0
    topk_hits = 0
    rr_list: List[float] = []
    p_list: List[float] = []
    r_list: List[float] = []
    ap_list: List[float] = []

    pos_successes = 0
    pos_errors: List[float] = []

    per_query: List[Dict[str, Any]] = []

    for qi, q in enumerate(test_queries):
        query_text = str(q.get("query", "")).strip()
        if not query_text:
            raise ValueError(f"test_queries[{qi}] missing 'query'")

        gt_id_raw = q.get("gt_id", None)
        gt_id = _normalize_id(gt_id_raw) if gt_id_raw is not None else None
        gt_pos = q.get("gt_position", None)

        q_lower = query_text.lower()

        scores: List[float] = []
        for oid, label in zip(obj_ids, obj_labels):
            l_lower = label.lower()
            exact = 1.0 if l_lower == q_lower else 0.0
            substr = 0.7 if (q_lower in l_lower or l_lower in q_lower) and exact == 0.0 else 0.0
            conf = float(semantic_map[oid].get("confidence", 0.0))
            scores.append(exact * 10.0 + substr * 5.0 + conf)

        ranked_idx = list(np.argsort(scores)[::-1])
        ranked_ids_full = [obj_ids[i] for i in ranked_idx]
        ranked_ids = ranked_ids_full[:top_k]
        pred_id = ranked_ids[0] if ranked_ids else None

        hit_top1 = None
        hit_topk = None
        rr = None
        precision_k = None
        recall_k = None
        ap_k = None

        if gt_id is not None:
            hit_top1 = (pred_id == gt_id)
            hit_topk = (gt_id in ranked_ids)
            if hit_top1:
                top1_hits += 1
            if hit_topk:
                topk_hits += 1

            rr = _reciprocal_rank(gt_id, ranked_ids)
            rr_list.append(rr)

            relevant = [1 if rid == gt_id else 0 for rid in ranked_ids_full]
            precision_k, recall_k, ap_k = _precision_recall_ap_at_k(relevant, top_k)
            p_list.append(precision_k)
            r_list.append(recall_k)
            ap_list.append(ap_k)

        pos_success = None
        pos_error = None
        if gt_pos is not None and pred_id is not None:
            pred_centroid = semantic_map[pred_id].get("centroid", [np.nan, np.nan, np.nan])
            pos_error = _euclidean(pred_centroid, gt_pos)
            pos_success = bool(pos_error <= distance_threshold)
            pos_errors.append(pos_error)
            if pos_success:
                pos_successes += 1

        out = {
            "query": query_text,
            "gt_id": gt_id,
            "gt_position": gt_pos,
            "pred_top1_id": pred_id,
            "top_k_ids": ranked_ids,
            "hit_top1": hit_top1,
            "hit_topk": hit_topk,
            "rr": rr,
            "precision_at_k": precision_k,
            "recall_at_k": recall_k,
            "ap_at_k": ap_k,
            "pos_error": pos_error,
            "pos_success": pos_success,
            "notes": q.get("notes", None),
        }
        per_query.append(out)

        if verbose:
            print(f"\n[Query {qi+1}] {query_text}")
            print(f"  pred_top1={pred_id}, top_k={ranked_ids}")
            if gt_id is not None:
                print(f"  gt_id={gt_id} top1={hit_top1} topk={hit_topk} MRR={rr:.3f} P@K={precision_k:.3f} R@K={recall_k:.3f} AP@K={ap_k:.3f}")
            if gt_pos is not None and pos_error is not None:
                print(f"  pos_error={pos_error:.3f} success={pos_success}")

    n = len(test_queries)

    metrics: Dict[str, Any] = {
        "num_queries": n,
        "top1_accuracy": (top1_hits / n) if any_id_gt and n > 0 else None,
        "topk_accuracy": (topk_hits / n) if any_id_gt and n > 0 else None,
        "mean_reciprocal_rank": float(np.mean(rr_list)) if rr_list else None,
        "precision_at_k": float(np.mean(p_list)) if p_list else None,
        "recall_at_k": float(np.mean(r_list)) if r_list else None,
        "map_at_k": float(np.mean(ap_list)) if ap_list else None,
        "per_query": per_query,
    }

    if any_pos_gt:
        metrics["position_success_rate"] = (pos_successes / n) if n > 0 else 0.0
        metrics["avg_position_error"] = float(np.mean(pos_errors)) if pos_errors else None

    return metrics


# -------------------------
# Navigation evaluation
# -------------------------
def evaluate_navigation_commands(
    nav_controller: Any,
    test_scenarios: List[Dict[str, Any]],
    *,
    position_tolerance: float = 0.75,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not hasattr(nav_controller, "execute_navigation_command"):
        raise AttributeError("nav_controller must have execute_navigation_command()")

    n = len(test_scenarios)
    success_count = 0
    status_breakdown: Dict[str, int] = {}
    goal_errors: List[float] = []

    per_scenario: List[Dict[str, Any]] = []

    for i, sc in enumerate(test_scenarios):
        cmd = str(sc.get("command", "")).strip()
        if not cmd:
            raise ValueError(f"test_scenarios[{i}] missing 'command'")

        start_pos = np.array(sc.get("start_position", [0.0, 0.0, 0.0]), dtype=np.float32)
        expected_goal_xy = sc.get("expected_goal_xy", None)
        expected_target_id_raw = sc.get("expected_target_id", None)
        expected_target_id = _normalize_id(expected_target_id_raw) if expected_target_id_raw is not None else None

        try:
            result = nav_controller.execute_navigation_command(cmd, start_pos)
        except Exception as e:
            status_breakdown["EXCEPTION"] = status_breakdown.get("EXCEPTION", 0) + 1
            per_scenario.append({
                "command": cmd,
                "start_position": start_pos.tolist(),
                "success": False,
                "status": "EXCEPTION",
                "message": str(e),
                "goal_xy_error": None,
                "goal_xy_match": None,
                "target_id_match": None,
                "raw_result": None,
                "notes": sc.get("notes", None),
            })
            continue

        success = bool(result.get("success", False))
        status = str(result.get("status", "UNKNOWN"))
        message = str(result.get("message", ""))

        status_breakdown[status] = status_breakdown.get(status, 0) + 1
        if success:
            success_count += 1

        goal_xy_error = None
        goal_xy_match = None
        if expected_goal_xy is not None and result.get("goal_pose") is not None:
            gp = np.array(result["goal_pose"], dtype=np.float32).reshape(-1)
            goal_xy = gp[:2]
            exp_xy = np.array(expected_goal_xy, dtype=np.float32).reshape(-1)[:2]
            goal_xy_error = float(np.linalg.norm(goal_xy - exp_xy))
            goal_xy_match = bool(goal_xy_error <= position_tolerance)
            goal_errors.append(goal_xy_error)

        target_id_match = None
        if expected_target_id is not None:
            target_id_match = (str(result.get("target_id", None)) == expected_target_id)

        per_scenario.append({
            "command": cmd,
            "start_position": start_pos.tolist(),
            "success": success,
            "status": status,
            "message": message,
            "target_id": result.get("target_id", None),
            "target_label": result.get("target_label", None),
            "goal_pose": result.get("goal_pose", None),
            "goal_xy_error": goal_xy_error,
            "goal_xy_match": goal_xy_match,
            "target_id_match": target_id_match,
            "raw_result": result,
            "notes": sc.get("notes", None),
        })

        if verbose:
            print(f"\n[Scenario {i+1}] {cmd} success={success} status={status}")
            if goal_xy_error is not None:
                print(f"  goal_xy_error={goal_xy_error:.3f} tol={position_tolerance} match={goal_xy_match}")

    metrics = {
        "num_scenarios": n,
        "success_rate": (success_count / n) if n > 0 else 0.0,
        "avg_goal_xy_error": float(np.mean(goal_errors)) if goal_errors else None,
        "status_breakdown": status_breakdown,
        "per_scenario": per_scenario,
    }
    return metrics


# -------------------------
# Plots
# -------------------------
def plot_retrieval(metrics: Dict[str, Any], out_dir: Path, top_k: int) -> None:
    labels = []
    values = []

    for key, lab in [
        ("top1_accuracy", "Top-1"),
        ("topk_accuracy", f"Top-{top_k}"),
        ("mean_reciprocal_rank", "MRR"),
        ("precision_at_k", f"P@{top_k}"),
        ("recall_at_k", f"R@{top_k}"),
        ("map_at_k", f"mAP@{top_k}"),
    ]:
        v = metrics.get(key, None)
        if v is not None:
            labels.append(lab)
            values.append(float(v))

    if not labels:
        return

    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1.0)
    plt.title("Object Retrieval Metrics")
    plt.ylabel("Score")
    plt.savefig(out_dir / "object_retrieval_metrics.png", bbox_inches="tight")
    plt.close()


def plot_navigation(metrics: Dict[str, Any], out_dir: Path) -> None:
    sb = metrics.get("status_breakdown", {}) or {}
    if sb:
        plt.figure()
        plt.bar(list(sb.keys()), [sb[k] for k in sb.keys()])
        plt.title("Navigation Status Breakdown")
        plt.ylabel("Count")
        plt.savefig(out_dir / "navigation_status_breakdown.png", bbox_inches="tight")
        plt.close()

    errs = [s["goal_xy_error"] for s in metrics.get("per_scenario", []) if s.get("goal_xy_error") is not None]
    if errs:
        plt.figure()
        plt.hist(errs, bins=10)
        plt.title("Goal XY Error Histogram")
        plt.xlabel("Error (m)")
        plt.ylabel("Count")
        plt.savefig(out_dir / "navigation_goal_error_hist.png", bbox_inches="tight")
        plt.close()


# -------------------------
# CLI
# -------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + navigation.")
    parser.add_argument("--run-dir", type=str, required=True, help="Folder containing semantic_map.json")
    parser.add_argument("--results-dir", type=str, default=None, help="Base results dir (default: ./results)")
    parser.add_argument("--queries", type=str, default="configs/eval/test_queries.json")
    parser.add_argument("--scenarios", type=str, default="configs/eval/test_scenarios.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-threshold", type=float, default=0.75)
    parser.add_argument("--position-tolerance", type=float, default=0.75)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    base_results = Path(args.results_dir).resolve() if args.results_dir else _default_results_dir()
    eval_dir = base_results / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    semantic_map_path = run_dir / "semantic_map.json"
    if not semantic_map_path.exists():
        raise FileNotFoundError(f"Missing: {semantic_map_path}")

    semantic_map = _read_json(semantic_map_path)
    semantic_map = _normalize_semantic_map(semantic_map)

    queries_path = Path(args.queries).resolve()
    scenarios_path = Path(args.scenarios).resolve()

    test_queries = _read_json(queries_path) if queries_path.exists() else []
    test_scenarios = _read_json(scenarios_path) if scenarios_path.exists() else []

    # NavigationController must be stub-safe (no heavy deps at import)
    from src.navigation.navigation_controller import NavigationController
    nav_controller = NavigationController(semantic_map)

    retrieval_metrics = evaluate_object_retrieval(
        semantic_map,
        test_queries,
        top_k=args.top_k,
        distance_threshold=args.distance_threshold,
        verbose=args.verbose,
    ) if test_queries else None

    nav_metrics = evaluate_navigation_commands(
        nav_controller,
        test_scenarios,
        position_tolerance=args.position_tolerance,
        verbose=args.verbose,
    ) if test_scenarios else None

    report = {
        "run_dir": str(run_dir),
        "semantic_map_path": str(semantic_map_path),
        "queries_path": str(queries_path),
        "scenarios_path": str(scenarios_path),
        "object_retrieval": retrieval_metrics,
        "navigation": nav_metrics,
    }

    out_json = eval_dir / f"evaluation_{run_dir.name}.json"
    _write_json(out_json, report)

    if not args.no_plots:
        if retrieval_metrics:
            plot_retrieval(retrieval_metrics, eval_dir, args.top_k)
        if nav_metrics:
            plot_navigation(nav_metrics, eval_dir)

    print(f"[OK] Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())