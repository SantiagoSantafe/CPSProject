import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(a - b))


def _ranked_reciprocal_rank(found_id: Optional[Any], ranked_ids: List[Any]) -> float:
    """
    Compute Reciprocal Rank (RR) for a single query.
    RR = 1 / rank if found in ranked list, otherwise 0.
    """
    if found_id is None:
        return 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == found_id:
            return 1.0 / i
    return 0.0


def evaluate_object_retrieval(
    semantic_map: Dict[Any, Dict[str, Any]],
    test_queries: List[Dict[str, Any]],
    *,
    top_k: int = 5,
    distance_threshold: float = 0.75,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate object retrieval performance of a semantic map.

    This function measures how well the system retrieves the correct object(s) given
    a natural language query. It supports two common evaluation modes:

    1) ID-based ground truth:
       - Provide 'gt_id' in each test query.
       - A retrieval is correct if the GT object appears in the top-k results.

    2) Position-based ground truth:
       - Provide 'gt_position' (x,y,z) in each test query.
       - A retrieval is correct if the predicted top result is within 'distance_threshold'
         meters of the GT position.

    Input format (each element in test_queries):
        {
          "query": "chair",
          "gt_id": 3,                      # optional
          "gt_label": "chair",             # optional (for reporting)
          "gt_position": [1.2, 0.4, 0.0],   # optional
          "notes": "near window"           # optional
        }

    Requirements:
    - The semantic_map must be a dictionary:
        semantic_map[obj_id] = {
            "label": str,
            "centroid": [x, y, z],
            "dimensions": [dx, dy, dz],    # optional
            "confidence": float,           # optional
            ...
        }

    NOTE:
    This function does not require CLIP; it evaluates retrieval by label matching and/or
    nearest centroid distance depending on your ground truth.

    Parameters
    ----------
    semantic_map:
        Dictionary of objects in the semantic map.
    test_queries:
        List of evaluation queries with ground truth.
    top_k:
        Evaluate Top-K hit rate (default 5).
    distance_threshold:
        For position-based evaluation, success if predicted is within this threshold (meters).
    verbose:
        If True, prints per-query diagnostics.

    Returns
    -------
    metrics: dict
        {
          "num_queries": int,
          "top1_accuracy": float,
          "topk_accuracy": float,
          "mean_reciprocal_rank": float,
          "position_success_rate": float,     # only if position GT provided in any query
          "avg_position_error": float,        # only if position GT provided
          "per_query": [ ... detailed results ... ]
        }
    """
    if not semantic_map:
        raise ValueError("semantic_map is empty. Cannot evaluate retrieval.")

    # Pre-extract objects for faster evaluation.
    obj_ids = list(semantic_map.keys())
    obj_labels = [semantic_map[i].get("label", "") for i in obj_ids]
    obj_centroids = [np.array(semantic_map[i].get("centroid", [np.nan, np.nan, np.nan]), dtype=np.float32) for i in obj_ids]

    per_query_results: List[Dict[str, Any]] = []
    top1_hits = 0
    topk_hits = 0
    rrs: List[float] = []

    # Position-GT aggregated metrics (computed only if any query includes gt_position).
    any_pos_gt = any("gt_position" in q and q["gt_position"] is not None for q in test_queries)
    pos_successes = 0
    pos_errors: List[float] = []

    for qi, q in enumerate(test_queries):
        query_text = str(q.get("query", "")).strip()
        gt_id = q.get("gt_id", None)
        gt_pos = q.get("gt_position", None)

        if not query_text:
            raise ValueError(f"test_queries[{qi}] missing non-empty 'query' field.")

        # --- Retrieval strategy for evaluation ---
        # Since this is a generic evaluator, we rank objects by:
        # 1) Exact label match score (case-insensitive)
        # 2) Otherwise, substring match score
        # 3) Otherwise, fallback to confidence if present
        #
        # If you have a real query engine that returns ranked results, you can pass
        # those results directly instead and adapt this function.
        q_lower = query_text.lower()

        scores = []
        for oid, label in zip(obj_ids, obj_labels):
            l_lower = (label or "").lower()
            exact = 1.0 if l_lower == q_lower else 0.0
            substr = 0.7 if (q_lower in l_lower or l_lower in q_lower) and exact == 0.0 else 0.0
            conf = float(semantic_map[oid].get("confidence", 0.0))
            # Weighted score: exact > substring > confidence
            score = exact * 10.0 + substr * 5.0 + conf
            scores.append(score)

        ranked_indices = list(np.argsort(scores)[::-1])  # descending
        ranked_ids = [obj_ids[i] for i in ranked_indices][:top_k]
        ranked_labels = [obj_labels[i] for i in ranked_indices][:top_k]
        ranked_scores = [float(scores[i]) for i in ranked_indices][:top_k]

        # Determine predicted top-1.
        pred_id = ranked_ids[0] if ranked_ids else None
        pred_label = ranked_labels[0] if ranked_labels else None

        # --- ID-based success ---
        hit_top1 = False
        hit_topk = False
        if gt_id is not None:
            hit_top1 = (pred_id == gt_id)
            hit_topk = (gt_id in ranked_ids)
            if hit_top1:
                top1_hits += 1
            if hit_topk:
                topk_hits += 1
            rrs.append(_ranked_reciprocal_rank(gt_id, ranked_ids))

        # --- Position-based success (if provided) ---
        pos_success = None
        pos_error = None
        if gt_pos is not None:
            gt_pos_arr = np.array(gt_pos, dtype=np.float32)
            if pred_id is not None:
                pred_centroid = np.array(semantic_map[pred_id].get("centroid", [np.nan, np.nan, np.nan]), dtype=np.float32)
                pos_error = _euclidean_distance(pred_centroid, gt_pos_arr)
                pos_success = (pos_error <= distance_threshold)
                pos_errors.append(pos_error)
                if pos_success:
                    pos_successes += 1

        result = {
            "query": query_text,
            "gt_id": gt_id,
            "gt_position": gt_pos,
            "pred_top1_id": pred_id,
            "pred_top1_label": pred_label,
            "top_k_ids": ranked_ids,
            "top_k_labels": ranked_labels,
            "top_k_scores": ranked_scores,
            "hit_top1": hit_top1 if gt_id is not None else None,
            "hit_topk": hit_topk if gt_id is not None else None,
            "pos_error": pos_error,
            "pos_success": pos_success,
            "notes": q.get("notes", None),
        }
        per_query_results.append(result)

        if verbose:
            print(f"\n[Query {qi+1}/{len(test_queries)}] '{query_text}'")
            print(f"  Top-1: id={pred_id}, label='{pred_label}'")
            print(f"  Top-{top_k}: {list(zip(ranked_ids, ranked_labels))}")
            if gt_id is not None:
                print(f"  GT id={gt_id} -> Top1={hit_top1}, Top{top_k}={hit_topk}")
            if gt_pos is not None:
                print(f"  GT pos={gt_pos} -> pos_error={pos_error:.3f}m, success={pos_success}")

    num_queries = len(test_queries)

    metrics: Dict[str, Any] = {
        "num_queries": num_queries,
        "top1_accuracy": (top1_hits / num_queries) if any(q.get("gt_id") is not None for q in test_queries) else None,
        "topk_accuracy": (topk_hits / num_queries) if any(q.get("gt_id") is not None for q in test_queries) else None,
        "mean_reciprocal_rank": (float(np.mean(rrs)) if rrs else None),
        "per_query": per_query_results,
    }

    if any_pos_gt:
        metrics["position_success_rate"] = pos_successes / num_queries
        metrics["avg_position_error"] = float(np.mean(pos_errors)) if pos_errors else None

    return metrics


def evaluate_navigation_commands(
    nav_controller: Any,
    test_scenarios: List[Dict[str, Any]],
    *,
    position_tolerance: float = 0.75,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate navigation command execution success.

    This function assumes your NavigationController exposes:
        execute_navigation_command(command_text: str, current_position: np.ndarray) -> dict

    Expected return dict (recommended convention):
        {
          "success": bool,
          "status": "SUCCESS" | "NOT_FOUND" | "ERROR" | ...,
          "message": str,
          "target_id": int | str,
          "target_label": str,
          "goal_pose": [x, y, theta]
        }

    Test scenario format (each element in test_scenarios):
        {
          "command": "navigate to the chair",
          "start_position": [0.0, 0.0, 0.0],
          "expected_target_id": 3,                 # optional
          "expected_target_label": "chair",        # optional (for reporting)
          "expected_goal_xy": [1.5, 0.2],          # optional
          "notes": "simple target test"            # optional
        }

    Scoring:
    - success_rate: fraction of scenarios where nav_controller returns success=True
    - target_id_accuracy: fraction where returned target_id matches expected_target_id (if provided)
    - goal_position_success_rate: fraction where goal xy is within position_tolerance of expected_goal_xy (if provided)
    - error_breakdown: counts per status code

    Parameters
    ----------
    nav_controller:
        Your NavigationController instance.
    test_scenarios:
        List of navigation tests.
    position_tolerance:
        XY tolerance for goal comparison (meters).
    verbose:
        Print per-scenario diagnostics.

    Returns
    -------
    metrics: dict with aggregated statistics and per-scenario results.
    """
    if not hasattr(nav_controller, "execute_navigation_command"):
        raise AttributeError("nav_controller must have method execute_navigation_command(command_text, current_position).")

    per_scenario: List[Dict[str, Any]] = []
    num = len(test_scenarios)

    success_count = 0
    target_id_hits = 0
    target_id_total = 0
    goal_hits = 0
    goal_total = 0

    error_breakdown: Dict[str, int] = {}

    for i, sc in enumerate(test_scenarios):
        cmd = str(sc.get("command", "")).strip()
        if not cmd:
            raise ValueError(f"test_scenarios[{i}] missing non-empty 'command'.")

        start_pos = np.array(sc.get("start_position", [0.0, 0.0, 0.0]), dtype=np.float32)

        expected_target_id = sc.get("expected_target_id", None)
        expected_goal_xy = sc.get("expected_goal_xy", None)

        try:
            result = nav_controller.execute_navigation_command(cmd, start_pos)
        except Exception as e:
            # Hard failure: count as error.
            status = "EXCEPTION"
            error_breakdown[status] = error_breakdown.get(status, 0) + 1
            per_scenario.append({
                "command": cmd,
                "start_position": start_pos.tolist(),
                "success": False,
                "status": status,
                "message": str(e),
                "raw_result": None,
                "target_id_match": None,
                "goal_xy_error": None,
                "goal_xy_match": None,
                "notes": sc.get("notes", None),
            })
            if verbose:
                print(f"\n[Scenario {i+1}/{num}] '{cmd}' -> EXCEPTION: {e}")
            continue

        success = bool(result.get("success", False))
        status = str(result.get("status", "UNKNOWN"))
        message = str(result.get("message", ""))

        error_breakdown[status] = error_breakdown.get(status, 0) + 1

        if success:
            success_count += 1

        # Target-ID match (optional)
        target_id_match = None
        if expected_target_id is not None:
            target_id_total += 1
            target_id_match = (result.get("target_id", None) == expected_target_id)
            if target_id_match:
                target_id_hits += 1

        # Goal XY match (optional)
        goal_xy_match = None
        goal_xy_error = None
        if expected_goal_xy is not None and result.get("goal_pose") is not None:
            goal_total += 1
            goal_pose = np.array(result["goal_pose"], dtype=np.float32).reshape(-1)
            goal_xy = goal_pose[:2]
            expected_xy = np.array(expected_goal_xy, dtype=np.float32).reshape(-1)[:2]
            goal_xy_error = float(np.linalg.norm(goal_xy - expected_xy))
            goal_xy_match = (goal_xy_error <= position_tolerance)
            if goal_xy_match:
                goal_hits += 1

        scenario_out = {
            "command": cmd,
            "start_position": start_pos.tolist(),
            "success": success,
            "status": status,
            "message": message,
            "target_id": result.get("target_id", None),
            "target_label": result.get("target_label", None),
            "goal_pose": result.get("goal_pose", None),
            "target_id_match": target_id_match,
            "goal_xy_error": goal_xy_error,
            "goal_xy_match": goal_xy_match,
            "notes": sc.get("notes", None),
            "raw_result": result,
        }
        per_scenario.append(scenario_out)

        if verbose:
            print(f"\n[Scenario {i+1}/{num}] '{cmd}'")
            print(f"  success={success}, status={status}")
            if target_id_match is not None:
                print(f"  target_id={result.get('target_id')} (expected {expected_target_id}) -> match={target_id_match}")
            if goal_xy_match is not None:
                print(f"  goal_xy_error={goal_xy_error:.3f}m (tol={position_tolerance}) -> match={goal_xy_match}")
            if message:
                print(f"  message: {message}")

    metrics: Dict[str, Any] = {
        "num_scenarios": num,
        "success_rate": success_count / num if num > 0 else 0.0,
        "status_breakdown": error_breakdown,
        "target_id_accuracy": (target_id_hits / target_id_total) if target_id_total > 0 else None,
        "goal_position_success_rate": (goal_hits / goal_total) if goal_total > 0 else None,
        "per_scenario": per_scenario,
    }

    return metrics