import json
import numpy as np
from pathlib import Path


def test_run_system_smoke(monkeypatch, tmp_results_dir):
    import scripts.run_system as rs

    # Stub load_data
    def _load():
        rgb = [np.zeros((64,64,3), dtype=np.uint8) for _ in range(2)]
        depth = [np.ones((64,64), dtype=np.float32)*1.5 for _ in range(2)]
        poses = [np.eye(4, dtype=np.float32) for _ in range(2)]
        intr = {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0}
        return rgb, depth, poses, intr
    monkeypatch.setattr(rs, "load_data", _load)

    # Stub detector
    class Det:
        def detect_objects(self, rgb, target_queries, background_queries=None):
            m = np.zeros((64,64), dtype=np.uint8); m[16:32,16:32] = 1
            return [{"label": "chair", "score": 0.9, "mask": m, "bbox": [16,16,16,16], "box": [16,16,16,16]}]
        def project_to_3d(self, mask, depth, intr):
            return {"centroid": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    "dimensions": np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    "points_3d": np.zeros((10,3), dtype=np.float32)}
    monkeypatch.setattr(rs, "OpenVocabularyDetector", lambda: Det())

    # Force query engine to stub backend (if you add backend param)
    from src.navigation import navigation_controller as nc_mod
    orig_init = nc_mod.NavigationController.__init__
    def _init(self, semantic_map):
        orig_init(self, semantic_map)
        # patch query engine to deterministic
        self.query_engine = type("QE", (), {
            "parse_navigation_command": lambda self, t: {"target_desc": "chair"},
            "query_objects": lambda self, q, max_results=5: [{"id": 0, "label": "chair", "centroid": [1,0,0], "dimensions":[1,1,1]}],
        })()
    monkeypatch.setattr(nc_mod.NavigationController, "__init__", _init)

    # Run
    exit_code = rs.main(argv=["--results-dir", str(tmp_results_dir), "--max-frames", "2"])
    assert exit_code == 0

    # Check artifacts
    smap = Path(tmp_results_dir) / "semantic_map.json"
    nav = Path(tmp_results_dir) / "nav_results.json"
    assert smap.exists()
    assert nav.exists()

    data = json.loads(smap.read_text(encoding="utf-8"))
    assert isinstance(data, dict)