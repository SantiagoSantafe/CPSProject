import numpy as np
from src.navigation.query_engine import SemanticQueryEngine

class NavigationController:
    def __init__(self, semantic_map: dict):
        """
        semantic_map: dict -> {obj_id: {label, centroid, dimensions, ...}}
        """
        self.semantic_map = semantic_map
        self.query_engine = SemanticQueryEngine(semantic_map, backend="stub")

    def execute_navigation_command(self, command_text: str, current_position: np.ndarray):
        """
        Args:
            command_text (str): "Find the red chair"
            current_position (np.array): [x, y, z] robot position

        Returns:
            dict: {success, status, message, target_id, target_label, goal_pose}
        """
        parsed = self.query_engine.parse_navigation_command(command_text)
        target_desc = parsed["target_desc"]

        print(f"Navigation command parsed: searching for '{target_desc}'")

        results = self.query_engine.query_objects(target_desc, max_results=5)

        if not results:
            return {
                "success": False,
                "status": "NOT_FOUND",
                "message": f"Object '{target_desc}' not found"
            }

        best_target = results[0]
        print(f"Navigation: Target identified -> ID {best_target['id']} ({best_target['label']})")

        goal_pose = self.calculate_approach_goal(best_target, current_position)

        return {
            "success": True,
            "status": "SUCCESS",
            "target_id": best_target["id"],
            "target_label": best_target["label"],
            "goal_pose": goal_pose.tolist() if isinstance(goal_pose, np.ndarray) else goal_pose,
            "message": f"Navigating to {best_target['label']}"
        }

    def calculate_approach_goal(self, target_object: dict, current_position: np.ndarray, safe_distance: float = 0.8):
        """
        Returns goal pose [x, y, theta] in map/global frame.
        """
        obj_center = np.array(target_object["centroid"], dtype=np.float32)   # [x,y,z]
        robot_pos = np.array(current_position, dtype=np.float32)             # [x,y,z]

        obj_center_2d = obj_center[:2]
        robot_pos_2d = robot_pos[:2]

        dims = np.array(target_object.get("dimensions", [0.5, 0.5, 0.5]), dtype=np.float32)
        obj_radius = np.linalg.norm(dims[:2]) / 2.0

        approach_vector = robot_pos_2d - obj_center_2d
        dist_to_obj = np.linalg.norm(approach_vector)

        if dist_to_obj < 1e-6:
            approach_direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            approach_direction = approach_vector / dist_to_obj

        total_dist_from_center = obj_radius + safe_distance
        goal_xy = obj_center_2d + (approach_direction * total_dist_from_center)

        look_at_vector = -approach_direction
        theta = float(np.arctan2(look_at_vector[1], look_at_vector[0]))

        return np.array([float(goal_xy[0]), float(goal_xy[1]), theta], dtype=np.float32)