#!/usr/bin/env python3
"""scripts/evaluate_system.py

Offline evaluation for CPSProject.

Inputs (produced by scripts/run_system.py):
  - <run_dir>/semantic_map.json
  - <run_dir>/nav_results.json

Evaluation configs (editable by you):
  - configs/eval/test_queries.json
  - configs/eval/test_scenarios.json

Outputs:
  - results/evaluations/evaluation_<run_name>.json (or --out)
  - prints a compact stats block for your presentation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _ensure_str_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x)


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(a - b))


def _ranked_reciprocal_rank(found_id: Optional[str], ranked_ids: List[str]) -> float:
    if found_id is None:
        return 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == found_id:
            return 1.0 / i
    return 0.0


def _precision_at_k(found_id: Optional[str], ranked_ids: List[str], k: int) -> float:
    if found_id is None:
        return 0.0
    topk = ranked_ids[:k]
    return 1.0 / k if found_id in topk else 0.0


def _recall_at_k(found_id: Optional[str], ranked_ids: List[str], k: int) -> float:
    # single relevant item => recall is 1 if present else 0
    if found_id is None:
        return 0.0
    return 1.0 if found_id in ranked_ids[:k] else 0.0


def _ap_at_k(found_id: Optional[str], ranked_ids: List[str], k: int) -> float:
    # single relevant item => AP@K is precision at the rank if present
    if found_id is None:
        return 0.0
    topk = ranked_ids[:k]
    for i, rid in enumerate(topk, start=1):
        if rid == found_id:
            return 1.0 / i
    return 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_object_retrieval(
    semantic_map: Dict[str, Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    distance_threshold: float = 0.75,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate retrieval using your query_engine ranking if available in nav_results.

    If nav_results contains ranked lists per query you can extend this later.
    For now we do a simple label ranking fallback:
      - exact label match > substring match > confidence

    NOTE: IDs are coerced to string to avoid int/str mismatch.
    """

    obj_ids = list(semantic_map.keys())
    obj_labels = [semantic_map[oid].get("label", "") for oid in obj_ids]

    per_query: List[Dict[str, Any]] = []

    any_id_gt = any(q.get("gt_id") is not None for q in test_queries)
    any_pos_gt = any(q.get("gt_position") is not None for q in test_queries)

    top1_hits = 0
    topk_hits = 0
    rrs: List[float] = []
    ps: List[float] = []
    rs_: List[float] = []
    aps: List[float] = []

    pos_successes = 0
    pos_errors: List[float] = []

    for qi, q in enumerate(test_queries):
        query_text = str(q.get("query", "")).strip()
        if not query_text:
            raise ValueError(f"test_queries[{qi}] missing 'query'.")

        gt_id = _ensure_str_id(q.get("gt_id", None))
        gt_pos = q.get("gt_position", None)

        q_lower = query_text.lower()
        scores: List[float] = []

        for oid, label in zip(obj_ids, obj_labels):
            l_lower = (label or "").lower()
            exact = 1.0 if l_lower == q_lower else 0.0
            substr = 0.7 if (q_lower in l_lower or l_lower in q_lower) and exact == 0.0 else 0.0
            conf = float(semantic_map[oid].get("confidence", 0.0))
            score = exact * 10.0 + substr * 5.0 + conf
            scores.append(score)

        ranked_idx = list(np.argsort(scores)[::-1])
        ranked_ids = [obj_ids[i] for i in ranked_idx][:top_k]
        pred_top1 = ranked_ids[0] if ranked_ids else None

        hit_top1 = None
        hit_topk = None
        rr = None
        p_at_k = None
        r_at_k = None
        ap_at_k = None

        if gt_id is not None:
            hit_top1 = (pred_top1 == gt_id)
            hit_topk = (gt_id in ranked_ids)
            rr = _ranked_reciprocal_rank(gt_id, ranked_ids)
            p_at_k = _precision_at_k(gt_id, ranked_ids, top_k)
            r_at_k = _recall_at_k(gt_id, ranked_ids, top_k)
            ap_at_k = _ap_at_k(gt_id, ranked_ids, top_k)

            top1_hits += int(hit_top1)
            topk_hits += int(hit_topk)
            rrs.append(rr)
            ps.append(p_at_k)
            rs_.append(r_at_k)
            aps.append(ap_at_k)

        pos_error = None
        pos_success = None
        if gt_pos is not None and pred_top1 is not None:
            gt_pos_arr = np.array(gt_pos, dtype=np.float32)
            pred_cent = np.array(
                semantic_map[pred_top1].get("centroid", [np.nan, np.nan, np.nan]),
                dtype=np.float32,
            )
            pos_error = _euclidean(pred_cent, gt_pos_arr)
            pos_success = bool(pos_error <= distance_threshold)
            pos_errors.append(pos_error)
            pos_successes += int(pos_success)

        out = {
            "query": query_text,
            "gt_id": gt_id,
            "gt_position": gt_pos,
            "pred_top1_id": pred_top1,
            "top_k_ids": ranked_ids,
            "hit_top1": hit_top1,
            "hit_topk": hit_topk,
            "rr": rr,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "ap_at_k": ap_at_k,
            "pos_error": pos_error,
            "pos_success": pos_success,
            "notes": q.get("notes", None),
        }
        per_query.append(out)

        if verbose:
            print(f"\n[Query {qi+1}] {query_text}")
            print(f"  pred_top1={pred_top1}, top_k={ranked_ids}")
            if gt_id is not None:
                print(
                    f"  gt_id={gt_id} top1={hit_top1} topk={hit_topk} "
                    f"MRR={rr:.3f} P@K={p_at_k:.3f} R@K={r_at_k:.3f} AP@K={ap_at_k:.3f}"
                )
            if gt_pos is not None:
                print(f"  pos_error={pos_error:.3f} success={pos_success}")

    n = len(test_queries)

    metrics: Dict[str, Any] = {
        "num_queries": n,
        "top1_accuracy": (top1_hits / n) if any_id_gt else None,
        "topk_accuracy": (topk_hits / n) if any_id_gt else None,
        "mean_reciprocal_rank": float(np.mean(rrs)) if rrs else None,
        "precision_at_k": float(np.mean(ps)) if ps else None,
        "recall_at_k": float(np.mean(rs_)) if rs_ else None,
        "map_at_k": float(np.mean(aps)) if aps else None,
        "per_query": per_query,
    }

    if any_pos_gt:
        metrics["position_success_rate"] = pos_successes / n if n > 0 else 0.0
        metrics["avg_position_error"] = float(np.mean(pos_errors)) if pos_errors else None

    return metrics


def evaluate_navigation(
    nav_results: List[Dict[str, Any]],
    test_scenarios: List[Dict[str, Any]],
    *,
    position_tolerance: float = 0.75,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate navigation results written by run_system.

    nav_results is expected to be a list in the same order as commands executed.
    We match by index.
    """

    n = len(test_scenarios)
    per_sc: List[Dict[str, Any]] = []

    success_count = 0
    goal_errors: List[float] = []
    goal_hits = 0
    goal_total = 0

    target_hits = 0
    target_total = 0

    status_breakdown: Dict[str, int] = {}

    for i, sc in enumerate(test_scenarios):
        cmd = str(sc.get("command", "")).strip()
        expected_target_id = _ensure_str_id(sc.get("expected_target_id", None))
        expected_goal_xy = sc.get("expected_goal_xy", None)

        result = (
            nav_results[i]
            if i < len(nav_results)
            else {"success": False, "status": "MISSING", "message": "No nav result"}
        )

        success = bool(result.get("success", False))
        status = str(result.get("status", "UNKNOWN"))
        status_breakdown[status] = status_breakdown.get(status, 0) + 1

        if success:
            success_count += 1

        # target id match
        target_id_match = None
        if expected_target_id is not None:
            target_total += 1
            target_id_match = (_ensure_str_id(result.get("target_id", None)) == expected_target_id)
            target_hits += int(bool(target_id_match))

        # goal xy match
        goal_xy_match = None
        goal_xy_error = None
        if expected_goal_xy is not None and result.get("goal_pose") is not None:
            goal_total += 1
            goal_pose = np.array(result["goal_pose"], dtype=np.float32).reshape(-1)
            goal_xy = goal_pose[:2]
            exp_xy = np.array(expected_goal_xy, dtype=np.float32).reshape(-1)[:2]
            goal_xy_error = float(np.linalg.norm(goal_xy - exp_xy))
            goal_errors.append(goal_xy_error)
            goal_xy_match = bool(goal_xy_error <= position_tolerance)
            goal_hits += int(goal_xy_match)

        out = {
            "command": cmd,
            "success": success,
            "status": status,
            "message": str(result.get("message", "")),
            "target_id": _ensure_str_id(result.get("target_id", None)),
            "target_label": result.get("target_label", None),
            "goal_pose": result.get("goal_pose", None),
            "goal_xy_error": goal_xy_error,
            "goal_xy_match": goal_xy_match,
            "target_id_match": target_id_match,
            "raw_result": result,
        }
        per_sc.append(out)

        if verbose:
            print(f"\n[Scenario {i+1}] {cmd} success={success} status={status}")
            if goal_xy_error is not None:
                print(
                    f"  goal_xy_error={goal_xy_error:.3f} "
                    f"tol={position_tolerance} match={goal_xy_match}"
                )

    metrics: Dict[str, Any] = {
        "num_scenarios": n,
        "success_rate": (success_count / n) if n > 0 else 0.0,
        "avg_goal_xy_error": float(np.mean(goal_errors)) if goal_errors else None,
        "status_breakdown": status_breakdown,
        "goal_position_success_rate": (goal_hits / goal_total) if goal_total > 0 else None,
        "target_id_accuracy": (target_hits / target_total) if target_total > 0 else None,
        "per_scenario": per_sc,
    }
    return metrics


def _load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = load_json(path)

    # Your semantic_map.json is expected to be a dict: {id: {label, centroid, ...}}
    # Normalize keys to string.
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        out[str(k)] = v
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an offline run (semantic map + nav results).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory produced by scripts/run_system.py")
    parser.add_argument("--queries", type=str, default="configs/eval/test_queries.json")
    parser.add_argument("--scenarios", type=str, default="configs/eval/test_scenarios.json")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: results/evaluations/evaluation_<run>.json)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--distance-threshold", type=float, default=0.75)
    parser.add_argument("--goal-tol", type=float, default=0.75)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    sem_path = run_dir / "semantic_map.json"
    nav_path = run_dir / "nav_results.json"

    if not sem_path.exists():
        raise FileNotFoundError(f"Missing: {sem_path}")
    if not nav_path.exists():
        raise FileNotFoundError(f"Missing: {nav_path}")

    semantic_map = _load_semantic_map(sem_path)
    nav_results = load_json(nav_path)

    queries = load_json(Path(args.queries))
    scenarios = load_json(Path(args.scenarios))

    print("\n====================")
    print("EVALUATION")
    print("====================")

    obj_metrics = evaluate_object_retrieval(
        semantic_map,
        queries,
        top_k=args.top_k,
        distance_threshold=args.distance_threshold,
        verbose=args.verbose,
    )

    nav_metrics = evaluate_navigation(
        nav_results,
        scenarios,
        position_tolerance=args.goal_tol,
        verbose=args.verbose,
    )

    out = {
        "run_dir": str(run_dir),
        "semantic_map_path": str(sem_path),
        "nav_results_path": str(nav_path),
        "queries_path": str(Path(args.queries).resolve()),
        "scenarios_path": str(Path(args.scenarios).resolve()),
        "object_retrieval": obj_metrics,
        "navigation": nav_metrics,
    }

    # default output path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        eval_dir = Path("results/evaluations").resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / f"evaluation_{run_dir.name}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Print a compact summary block for slides
    print("\n====================")
    print("SUMMARY (copy/paste)")
    print("====================")
    print(f"Run: {run_dir}")
    print(f"Objects in map: {len(semantic_map)}")

    if obj_metrics.get("top1_accuracy") is not None:
        print(f"Retrieval Top-1 Acc: {obj_metrics['top1_accuracy']:.3f}")
        print(f"Retrieval Top-{args.top_k} Acc: {obj_metrics['topk_accuracy']:.3f}")
        if obj_metrics.get("mean_reciprocal_rank") is not None:
            print(f"MRR: {obj_metrics['mean_reciprocal_rank']:.3f}")
        if obj_metrics.get("map_at_k") is not None:
            print(f"mAP@{args.top_k}: {obj_metrics['map_at_k']:.3f}")

    if obj_metrics.get("position_success_rate") is not None:
        print(f"Position Success Rate (<= {args.distance_threshold}m): {obj_metrics['position_success_rate']:.3f}")
        if obj_metrics.get("avg_position_error") is not None:
            print(f"Avg Position Error: {obj_metrics['avg_position_error']:.3f} m")

    print(f"Navigation Success Rate: {nav_metrics['success_rate']:.3f}")
    if nav_metrics.get("avg_goal_xy_error") is not None:
        print(f"Avg Goal XY Error: {nav_metrics['avg_goal_xy_error']:.3f} m")

    print(f"[OK] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())