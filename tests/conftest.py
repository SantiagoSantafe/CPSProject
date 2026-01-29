import os
import json
import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import src...` works when running pytest
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture(autouse=True)
def _determinism():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    np.random.seed(0)


@pytest.fixture
def fake_intrinsics():
    return {"fx": 100.0, "fy": 100.0, "cx": 32.0, "cy": 32.0}


@pytest.fixture
def fake_depth():
    d = np.ones((64, 64), dtype=np.float32) * 1.5
    return d


@pytest.fixture
def fake_mask():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[16:32, 16:32] = 1
    return m


@pytest.fixture
def tmp_results_dir(tmp_path, monkeypatch):
    out = tmp_path / "results"
    out.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RESULTS_DIR", str(out))
    return out


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)