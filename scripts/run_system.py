import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import json
import numpy as np
import time

from perception.open_vocab_detector import OpenVocabularyDetector
from mapping.semantic_mapper import SemanticMapper
from navigation.navigation_controller import NavigationController

def main():
    # Initialize components
    detector = OpenVocabularyDetector()
    mapper = SemanticMapper()
    
    # Load your data (from ROS bags or EnvoDat)
    rgb_images, depth_images, poses, intrinsics = load_data()
    
    # Build semantic map
    print("Building semantic map...")
    mapper.build_semantic_map(rgb_images, depth_images, poses, intrinsics)
    
    # Initialize navigation
    nav_controller = NavigationController(mapper)
    
    # Test navigation commands
    test_commands = [
        "navigate to the chair",
        "go to the table near the window", 
        "find the computer on the desk",
        "locate the bookshelf"
    ]
    
    current_position = np.array([0, 0, 0])  # Starting position
    
    for command in test_commands:
        print(f"\nExecuting: {command}")
        result = nav_controller.execute_navigation_command(command, current_position)
        print(f"Result: {result}")
        
        if result['success']:
            # Update position (simulated navigation)
            current_position = result['goal_position']

def ensure_directories():
    """ Ensure  Correct directories """

    
if __name__ == "__main__":
    main()