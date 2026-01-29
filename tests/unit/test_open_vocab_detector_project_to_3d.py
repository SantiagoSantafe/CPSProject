import numpy as np
from src.perception.geometry import project_mask_to_3d


def test_project_mask_to_3d_returns_centroid(fake_mask, fake_depth, fake_intrinsics):
    out = project_mask_to_3d(fake_mask, fake_depth, fake_intrinsics)
    assert out is not None
    assert out["centroid"].shape == (3,)
    assert out["dimensions"].shape == (3,)
    assert out["num_points"] > 0


def test_project_mask_to_3d_rejects_invalid_depth(fake_mask, fake_intrinsics):
    depth = np.zeros((64, 64), dtype=np.float32)  # invalid
    out = project_mask_to_3d(fake_mask, depth, fake_intrinsics)
    assert out is None