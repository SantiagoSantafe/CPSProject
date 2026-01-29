#!/usr/bin/env python3
"""scripts/evaluate_system.py

Offline evaluation for CPSProject.

Inputs (produced by scripts/run_system.py):
  - <run_dir>/semantic_map.json
  - <run_dir>/nav_results.json

Evaluation configs:
  - configs/eval/test_queries.json
  - configs/eval/test_scenarios.json

Outputs:
  - results/evaluations/evaluation_<run_name>.json (or --out)
  - prints a compact summary block for slides

Adds:
  - Label-based Top-k success (robust when multiple instances share same label)
  - Error-mode breakdown for retrieval and navigation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ------------------------
# Helpers
# ------------------------

def _ensure_str_id(x: Any) -> Optional[str]:
    return None if x is None else str(x)


def _lower(x: Any) -> str:
    return str(x or "").strip().lower()


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
    return (1.0 / k) if (found_id in ranked_ids[:k]) else 0.0


def _recall_at_k(found_id: Optional[str], ranked_ids: List[str], k: int) -> float:
    if found_id is None:
        return 0.0
    return 1.0 if (found_id in ranked_ids[:k]) else 0.0


def _ap_at_k(found_id: Optional[str], ranked_ids: List[str], k: int) -> float:
    if found_id is None:
        return 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid == found_id:
            return 1.0 / i
    return 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_semantic_map(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        out[str(k)] = v
    return out


# ------------------------
# Retrieval evaluation
# ------------------------

def _rank_fallback_by_label_conf(
    semantic_map: Dict[str, Dict[str, Any]],
    query_text: str,
    top_k: int,
) -> Tuple[List[str], str]:
    """
    Fallback ranking:
      exact label match > substring match > confidence
    Returns (ranked_ids[:top_k], method_name)
    """
    obj_ids = list(semantic_map.keys())
    ql = _lower(query_text)

    scores: List[float] = []
    for oid in obj_ids:
        label = _lower(semantic_map[oid].get("label", ""))
        exact = 1.0 if label == ql else 0.0
        substr = 0.7 if ((ql in label) or (label in ql)) and exact == 0.0 else 0.0
        conf = float(semantic_map[oid].get("confidence", 0.0))
        scores.append(exact * 10.0 + substr * 5.0 + conf)

    ranked_idx = list(np.argsort(scores)[::-1])
    ranked_ids = [obj_ids[i] for i in ranked_idx][:top_k]
    return ranked_ids, "fallback(label/conf)"


def evaluate_object_retrieval(
    semantic_map: Dict[str, Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    distance_threshold: float = 0.75,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Metrics:
      - Top-1 / Top-k (ID)
      - MRR, P@k, R@k, mAP@k (single relevant ID)
      - Top-k by label (robust to multiple instances)
      - Position success (if gt_position provided)

    Error modes for ID-queries:
      - OK
      - WRONG_INSTANCE (label matches but ID differs)
      - WRONG_LABEL
      - NOT_FOUND (gt not in top_k)
    """

    any_id_gt = any(q.get("gt_id") is not None for q in test_queries)
    any_pos_gt = any(q.get("gt_position") is not None for q in test_queries)

    top1_hits = 0
    topk_hits = 0
    rrs: List[float] = []
    ps: List[float] = []
    rs_: List[float] = []
    aps: List[float] = []

    # label-based
    top1_label_hits = 0
    topk_label_hits = 0

    # pos-based
    pos_successes = 0
    pos_errors: List[float] = []

    # error modes
    error_modes: Dict[str, int] = {
        "OK": 0,
        "WRONG_INSTANCE": 0,
        "WRONG_LABEL": 0,
        "NOT_FOUND": 0,
        "NO_GT": 0,
    }

    per_query: List[Dict[str, Any]] = []

    for qi, q in enumerate(test_queries):
        query_text = str(q.get("query", "")).strip()
        if not query_text:
            raise ValueError(f"test_queries[{qi}] missing 'query'.")

        gt_id = _ensure_str_id(q.get("gt_id", None))
        gt_pos = q.get("gt_position", None)

        gt_label = _lower(q.get("gt_label", None))
        if gt_id is not None and not gt_label:
            # if gt_label not provided, derive from semantic_map if possible
            gt_label = _lower(semantic_map.get(gt_id, {}).get("label", ""))

        ranked_ids, rank_method = _rank_fallback_by_label_conf(semantic_map, query_text, top_k=top_k)
        pred_top1 = ranked_ids[0] if ranked_ids else None

        pred_top1_label = _lower(semantic_map.get(pred_top1, {}).get("label", "")) if pred_top1 else ""
        ranked_labels = [_lower(semantic_map.get(rid, {}).get("label", "")) for rid in ranked_ids]

        # ID-metrics
        hit_top1 = None
        hit_topk = None
        rr = None
        p_at_k = None
        r_at_k = None
        ap_at_k = None

        # label-metrics
        hit_top1_label = None
        hit_topk_label = None

        mode = None
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

            # error mode (ID queries)
            if hit_topk:
                if hit_top1:
                    mode = "OK"
                else:
                    # GT is in top-k but not top-1
                    # check if predicted label matches GT label => wrong instance/ordering
                    if gt_label and pred_top1_label == gt_label:
                        mode = "WRONG_INSTANCE"
                    else:
                        mode = "WRONG_LABEL"
            else:
                mode = "NOT_FOUND"

            error_modes[mode] += 1
        else:
            error_modes["NO_GT"] += 1

        # label success (works even without gt_id if gt_label exists)
        if gt_label:
            hit_top1_label = (pred_top1_label == gt_label)
            hit_topk_label = (gt_label in ranked_labels)
            top1_label_hits += int(hit_top1_label)
            topk_label_hits += int(hit_topk_label)

        # position-based
        pos_error = None
        pos_success = None
        if gt_pos is not None and pred_top1 is not None:
            gt_pos_arr = np.array(gt_pos, dtype=np.float32)
            pred_cent = np.array(semantic_map[pred_top1].get("centroid", [np.nan, np.nan, np.nan]), dtype=np.float32)
            pos_error = _euclidean(pred_cent, gt_pos_arr)
            pos_success = bool(pos_error <= distance_threshold)
            pos_errors.append(pos_error)
            pos_successes += int(pos_success)

        out = {
            "query": query_text,
            "gt_id": gt_id,
            "gt_label": gt_label or None,
            "gt_position": gt_pos,
            "rank_method": rank_method,
            "pred_top1_id": pred_top1,
            "pred_top1_label": pred_top1_label or None,
            "top_k_ids": ranked_ids,
            "top_k_labels": ranked_labels,
            "hit_top1": hit_top1,
            "hit_topk": hit_topk,
            "rr": rr,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "ap_at_k": ap_at_k,
            "hit_top1_label": hit_top1_label,
            "hit_topk_label": hit_topk_label,
            "error_mode": mode,
            "pos_error": pos_error,
            "pos_success": pos_success,
            "notes": q.get("notes", None),
        }
        per_query.append(out)

        if verbose:
            print(f"\n[Query {qi+1}] {query_text}")
            print(f"  pred_top1={pred_top1} label='{pred_top1_label}' top_k={ranked_ids}")
            if gt_id is not None:
                print(f"  gt_id={gt_id} mode={mode} top1={hit_top1} topk={hit_topk} MRR={rr:.3f} mAP@{top_k}={ap_at_k:.3f}")
            if gt_label:
                print(f"  gt_label='{gt_label}' top1_label={hit_top1_label} topk_label={hit_topk_label}")
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

        "top1_label_accuracy": (top1_label_hits / n) if any(q.get("gt_label") or q.get("gt_id") for q in test_queries) else None,
        "topk_label_accuracy": (topk_label_hits / n) if any(q.get("gt_label") or q.get("gt_id") for q in test_queries) else None,

        "error_modes": error_modes,
        "per_query": per_query,
    }

    if any_pos_gt:
        metrics["position_success_rate"] = pos_successes / n if n > 0 else 0.0
        metrics["avg_position_error"] = float(np.mean(pos_errors)) if pos_errors else None

    return metrics


# ------------------------
# Navigation evaluation
# ------------------------

def _nav_error_mode(result: Dict[str, Any]) -> str:
    """
    Normalize navigation statuses into a slide-friendly mode.
    """
    if not result:
        return "MISSING"
    if bool(result.get("success", False)):
        return "OK"
    status = str(result.get("status", "UNKNOWN")).upper()
    if "NOT_FOUND" in status:
        return "NOT_FOUND"
    if "WRONG" in status:
        return "WRONG_TARGET"
    if "ERROR" in status or "EXCEPTION" in status:
        return "ERROR"
    return status or "FAILED"


def evaluate_navigation(
    nav_results: List[Dict[str, Any]],
    test_scenarios: List[Dict[str, Any]],
    *,
    position_tolerance: float = 0.75,
    verbose: bool = True,
) -> Dict[str, Any]:
    n = len(test_scenarios)
    per_sc: List[Dict[str, Any]] = []

    success_count = 0
    goal_errors: List[float] = []
    goal_hits = 0
    goal_total = 0

    target_hits = 0
    target_total = 0

    status_breakdown: Dict[str, int] = {}
    mode_breakdown: Dict[str, int] = {}

    for i, sc in enumerate(test_scenarios):
        cmd = str(sc.get("command", "")).strip()
        expected_target_id = _ensure_str_id(sc.get("expected_target_id", None))
        expected_goal_xy = sc.get("expected_goal_xy", None)

        result = nav_results[i] if i < len(nav_results) else {"success": False, "status": "MISSING", "message": "No nav result"}
        success = bool(result.get("success", False))
        status = str(result.get("status", "UNKNOWN"))

        status_breakdown[status] = status_breakdown.get(status, 0) + 1
        mode = _nav_error_mode(result)
        mode_breakdown[mode] = mode_breakdown.get(mode, 0) + 1

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
            "mode": mode,
            "message": str(result.get("message", "")),
            "target_id": _ensure_str_id(result.get("target_id", None)),
            "target_label": result.get("target_label", None),
            "goal_pose": result.get("goal_pose", None),
            "goal_xy_error": goal_xy_error,
            "goal_xy_match": goal_xy_match,
            "target_id_match": target_id_match,
            "raw_result": result,
            "notes": sc.get("notes", None),
        }
        per_sc.append(out)

        if verbose:
            print(f"\n[Scenario {i+1}] {cmd} success={success} status={status} mode={mode}")
            if goal_xy_error is not None:
                print(f"  goal_xy_error={goal_xy_error:.3f} tol={position_tolerance} match={goal_xy_match}")

    metrics: Dict[str, Any] = {
        "num_scenarios": n,
        "success_rate": (success_count / n) if n > 0 else 0.0,
        "avg_goal_xy_error": float(np.mean(goal_errors)) if goal_errors else None,
        "status_breakdown": status_breakdown,
        "mode_breakdown": mode_breakdown,
        "goal_position_success_rate": (goal_hits / goal_total) if goal_total > 0 else None,
        "target_id_accuracy": (target_hits / target_total) if target_total > 0 else None,
        "per_scenario": per_sc,
    }
    return metrics


# ------------------------
# CLI
# ------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate an offline run (semantic map + nav results).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory produced by scripts/run_system.py")
    parser.add_argument("--queries", type=str, default="configs/eval/test_queries.json")
    parser.add_argument("--scenarios", type=str, default="configs/eval/test_scenarios.json")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
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

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        eval_dir = Path("results/evaluations").resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        out_path = eval_dir / f"evaluation_{run_dir.name}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Compact summary block for slides
    print("\n====================")
    print("SUMMARY (copy/paste)")
    print("====================")
    print(f"Run: {run_dir}")
    print(f"Objects in map: {len(semantic_map)}")

    if obj_metrics.get("top1_accuracy") is not None:
        print(f"Retrieval Top-1 (ID) Acc: {obj_metrics['top1_accuracy']:.3f}")
        print(f"Retrieval Top-{args.top_k} (ID) Acc: {obj_metrics['topk_accuracy']:.3f}")
        if obj_metrics.get("mean_reciprocal_rank") is not None:
            print(f"MRR: {obj_metrics['mean_reciprocal_rank']:.3f}")
        if obj_metrics.get("map_at_k") is not None:
            print(f"mAP@{args.top_k}: {obj_metrics['map_at_k']:.3f}")

    if obj_metrics.get("top1_label_accuracy") is not None:
        print(f"Retrieval Top-1 (LABEL) Acc: {obj_metrics['top1_label_accuracy']:.3f}")
        print(f"Retrieval Top-{args.top_k} (LABEL) Acc: {obj_metrics['topk_label_accuracy']:.3f}")

    if obj_metrics.get("position_success_rate") is not None:
        print(f"Position Success (<= {args.distance_threshold}m): {obj_metrics['position_success_rate']:.3f}")
        if obj_metrics.get("avg_position_error") is not None:
            print(f"Avg Position Error: {obj_metrics['avg_position_error']:.3f} m")

    print(f"Navigation Success Rate: {nav_metrics['success_rate']:.3f}")
    if nav_metrics.get("avg_goal_xy_error") is not None:
        print(f"Avg Goal XY Error: {nav_metrics['avg_goal_xy_error']:.3f} m")
    print(f"Nav modes: {nav_metrics.get('mode_breakdown', {})}")
    print(f"Retrieval error modes: {obj_metrics.get('error_modes', {})}")

    print(f"[OK] Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())