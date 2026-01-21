import numpy as np

from perception.open_vocab_detector import OpenVocabularyDetector
from mapping.semantic_mapper import SemanticMapper
from navigation.navigation_controller import NavigationController

def load_data():
    """
    Placeholder: aquí debes cargar rgb_images, depth_images, poses, intrinsics
    Devuelve:
      rgb_images: list[np.ndarray]
      depth_images: list[np.ndarray]
      poses: list[np.ndarray]  # (N,4,4) o None
      intrinsics: dict {fx, fy, cx, cy}
    """
    raise NotImplementedError("Implement load_data()")

def main():
    detector = OpenVocabularyDetector()
    mapper = SemanticMapper()

    rgb_images, depth_images, poses, intrinsics = load_data()

    target_queries = ["chair", "table", "computer", "bookshelf"]
    background_queries = ["wall", "floor", "ceiling", "window"]

    print("Building semantic map...")
    semantic_map = mapper.build_semantic_map(
        rgb_images, depth_images, poses, intrinsics,
        detector, target_queries, background_queries
    )

    # ✅ Pasamos el dict, no el objeto mapper
    nav_controller = NavigationController(semantic_map)

    test_commands = [
        "navigate to the chair",
        "go to the table",
        "find the computer",
        "locate the bookshelf"
    ]

    current_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for command in test_commands:
        print(f"\nExecuting: {command}")
        result = nav_controller.execute_navigation_command(command, current_position)
        print(f"Result: {result}")

        # ✅ ahora existe "success"
        if result.get("success"):
            # Simulación simple de moverse al goal en XY (manteniendo z=0)
            goal = np.array(result["goal_pose"], dtype=np.float32)
            current_position = np.array([goal[0], goal[1], 0.0], dtype=np.float32)

if __name__ == "__main__":
    main()