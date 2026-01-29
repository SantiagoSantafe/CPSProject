from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class QueryResult:
    id: Any
    label: str
    centroid: List[float]
    similarity: float
    dimensions: List[float]


class SemanticQueryEngine:
    """
    Query engine with switchable backends.

    backends:
      - "stub" (default): lightweight ranking using label matching + optional confidence (NO sklearn/torch/clip)
      - "clip": real CLIP-based backend (requires torch + clip + sklearn)
    """

    def __init__(self, semantic_map: dict, backend: str = "stub"):
        self.semantic_map = semantic_map or {}
        self.backend = backend

        self.object_ids: List[Any] = []
        self.labels: List[str] = []

        # clip backend state
        self.device = None
        self.model = None
        self.index = None
        self.embedding_cache = None

        if self.semantic_map:
            self.setup_search_index()

    def setup_search_index(self) -> None:
        if not self.semantic_map:
            return

        self.object_ids = list(self.semantic_map.keys())
        self.labels = [self.semantic_map[obj_id].get("label", "") for obj_id in self.object_ids]

        if self.backend == "stub":
            # Nothing to build; we just rank by string similarity
            return

        if self.backend == "clip":
            # Heavy deps only imported here
            import torch
            import clip
            from sklearn.neighbors import NearestNeighbors

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Loading CLIP text encoder...")
            self.model, _ = clip.load("ViT-B/32", device=self.device)

            print(f"Indexing {len(self.labels)} objects...")
            text_inputs = clip.tokenize(self.labels).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            self.embedding_cache = text_features.cpu().numpy()
            self.index = NearestNeighbors(n_neighbors=min(5, len(self.labels)), algorithm="auto").fit(self.embedding_cache)
            print("Search index built successfully")
            return

        raise ValueError(f"Unknown backend='{self.backend}'. Use 'stub' or 'clip'.")

    def query_objects(self, text_query: str, max_results: int = 5, spatial_constraints=None) -> List[Dict[str, Any]]:
        text_query = (text_query or "").strip()
        if not text_query:
            return []

        if not self.semantic_map:
            return []

        if self.backend == "stub":
            return self._query_stub(text_query, max_results=max_results, spatial_constraints=spatial_constraints)

        if self.backend == "clip":
            return self._query_clip(text_query, max_results=max_results, spatial_constraints=spatial_constraints)

        raise ValueError(f"Unknown backend='{self.backend}'.")

    def _query_stub(self, text_query: str, max_results: int = 5, spatial_constraints=None) -> List[Dict[str, Any]]:
        q = text_query.lower()

        scored: List[tuple[float, Any]] = []
        for obj_id in self.object_ids:
            obj = self.semantic_map[obj_id]
            label = str(obj.get("label", "")).lower()

            # simple deterministic scoring
            exact = 1.0 if label == q else 0.0
            substr = 0.7 if (q in label or label in q) and exact == 0.0 else 0.0
            conf = float(obj.get("confidence", 0.0))
            score = exact * 10.0 + substr * 5.0 + conf

            # spatial constraint (optional)
            if spatial_constraints and "near_pos" in spatial_constraints:
                pos_obj = np.array(obj.get("centroid", [np.nan, np.nan, np.nan]), dtype=np.float32)
                pos_ref = np.array(spatial_constraints["near_pos"], dtype=np.float32)
                radius = float(spatial_constraints.get("radius", 2.0))
                if float(np.linalg.norm(pos_obj - pos_ref)) > radius:
                    continue

            scored.append((score, obj_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: List[Dict[str, Any]] = []

        for score, obj_id in scored[:max_results]:
            obj = self.semantic_map[obj_id]
            results.append({
                "id": obj_id,
                "label": obj.get("label", ""),
                "centroid": obj.get("centroid", [0.0, 0.0, 0.0]),
                "similarity": float(min(1.0, max(0.0, score / 10.0))),  # normalized-ish
                "dimensions": obj.get("dimensions", [0.5, 0.5, 0.5]),
            })
        return results

    def _query_clip(self, text_query: str, max_results: int = 5, spatial_constraints=None) -> List[Dict[str, Any]]:
        if self.index is None or self.model is None:
            return []

        import torch
        import clip

        text_inputs = clip.tokenize([text_query]).to(self.device)

        with torch.no_grad():
            qf = self.model.encode_text(text_inputs)
            qf = qf / qf.norm(dim=-1, keepdim=True)
            q_vec = qf.cpu().numpy()

        distances, indices = self.index.kneighbors(q_vec, n_neighbors=min(max_results, len(self.object_ids)))

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            obj_id = self.object_ids[idx]
            obj = self.semantic_map[obj_id]

            similarity = float(1.0 - distances[0][rank])

            # spatial constraint (optional)
            if spatial_constraints and "near_pos" in spatial_constraints:
                pos_obj = np.array(obj.get("centroid", [np.nan, np.nan, np.nan]), dtype=np.float32)
                pos_ref = np.array(spatial_constraints["near_pos"], dtype=np.float32)
                radius = float(spatial_constraints.get("radius", 2.0))
                if float(np.linalg.norm(pos_obj - pos_ref)) > radius:
                    continue

            results.append({
                "id": obj_id,
                "label": obj.get("label", ""),
                "centroid": obj.get("centroid", [0.0, 0.0, 0.0]),
                "similarity": similarity,
                "dimensions": obj.get("dimensions", [0.5, 0.5, 0.5]),
            })
        return results

    def parse_navigation_command(self, command_text: str) -> Dict[str, Any]:
        command_text = (command_text or "").lower().strip()
        verbs = ["go to", "move to", "navigate to", "find", "approach", "locate"]
        target_desc = command_text
        for verb in verbs:
            target_desc = target_desc.replace(verb, "").strip()
        return {"target_desc": target_desc, "spatial_rel": None}