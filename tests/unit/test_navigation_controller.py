import numpy as np
from src.navigation.navigation_controller import NavigationController


def test_execute_navigation_command_not_found(monkeypatch):
    semantic_map = {}
    nc = NavigationController(semantic_map)

    # patch query engine to avoid heavy init if needed
    nc.query_engine = type("QE", (), {
        "parse_navigation_command": lambda self, t: {"target_desc": "chair"},
        "query_objects": lambda self, q, max_results=5: [],
    })()

    out = nc.execute_navigation_command("navigate to chair", np.array([0,0,0], dtype=np.float32))
    assert out["success"] is False
    assert out["status"] == "NOT_FOUND"


def test_calculate_approach_goal_geometry():
    semantic_map = {0: {"label": "chair", "centroid": [0.0, 0.0, 0.0], "dimensions": [1.0, 1.0, 1.0]}}
    nc = NavigationController(semantic_map)
    goal = nc.calculate_approach_goal({"centroid": [0,0,0], "dimensions": [1,1,1]}, np.array([2,0,0], dtype=np.float32))
    assert goal.shape == (3,)
    # should be on +x side of object (robot is at +x)
    assert goal[0] > 0