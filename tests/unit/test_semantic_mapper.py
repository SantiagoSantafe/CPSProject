import numpy as np
from src.mapping.semantic_mapper import SemanticMapper


def test_transform_to_global_single_point():
    m = SemanticMapper()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    p = np.array([0.5, 0.0, -0.5], dtype=np.float32)
    out = m.transform_to_global(p, pose)
    assert np.allclose(out, [1.5, 2.0, 2.5])


def test_integrate_merges_close_same_label():
    m = SemanticMapper()
    m.integrate_semantic_objects([{"label": "chair", "centroid": [0,0,0], "dimensions": [1,1,1], "score": 0.8}], 0)
    assert len(m.semantic_objects) == 1
    oid = next(iter(m.semantic_objects.keys()))
    m.integrate_semantic_objects([{"label": "chair", "centroid": [0.1,0,0], "dimensions": [1,1,1], "score": 0.9}], 1)

    assert len(m.semantic_objects) == 1
    assert m.semantic_objects[oid]["observations"] == 2
    assert m.semantic_objects[oid]["confidence"] == 0.9