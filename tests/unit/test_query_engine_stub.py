from src.navigation.query_engine import SemanticQueryEngine


def test_parse_navigation_command():
    e = SemanticQueryEngine({}, backend="stub")
    out = e.parse_navigation_command("Navigate to the chair")
    assert out["target_desc"] == "the chair" or out["target_desc"] == "chair"


def test_query_objects_stub_ranks_label_match():
    semantic_map = {
        0: {"label": "chair", "centroid": [0,0,0], "dimensions": [1,1,1]},
        1: {"label": "table", "centroid": [1,0,0], "dimensions": [1,1,1]},
    }
    e = SemanticQueryEngine(semantic_map, backend="stub")
    res = e.query_objects("chair", max_results=2)
    assert res
    assert res[0]["label"] == "chair"