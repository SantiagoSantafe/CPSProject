from src.navigation.query_engine import SemanticQueryEngine
import numpy as np

class NavigationController:
    def __init__(self, semantic_map):
        self.semantic_map = semantic_map
        self.query_engine = SemanticQueryEngine(semantic_map)
        
    def execute_navigation_command(self, command_text, current_position):
        """Execute natural language navigation command
        
        Args:
            command_text (str): "Find the red chair"
            current_position (np.array): [x, y, z] robot position
            
        Returns:
            dict: Navigation plan {target_pose, target_obj_id, status}
        """
        # Implement:
        # 1. Parse command
        parsed = self.query_engine.parse_navigation_command(command_text)
        target_desc = parsed['target_desc']
        print(f"Navigation command parsed: searching for {target_desc}")
        # 2. Query for target objects
        results = self.query_engine.query_objects(target_desc, max_results=5)

        if not results:
            return {
                'status': 'NOT_FOUND',
                'message': f"Object '{target_desc}' not found"
            }
        
        #Choose the closest object to the robot
        best_target = results[0] 
        print(f"Navigation: Target identified -> ID {best_target['id']} ({best_target['label']})")
        
        # 3. Calculate navigation goal
        goal_pose = self.calculate_approach_goal(best_target, current_position)
        
        # 4. Return navigation plan
        return {
            'status': 'success',
            'target_id': best_target['id'],
            'target_label': best_target['label'],
            'goal_pose': goal_pose,  # [x, y, theta]
            'message': f"Navigating to {best_target['label']}"
        }
        
    def calculate_approach_goal(self, target_object, current_position, safe_distance=0.8):
        """Calculate safe approach position for target object"""
        # Implement:
        # 1. Consider object size and environment
        obj_center = np.array(target_object['centroid']) #x,y,z
        robot_pos = np.array(current_position) #x,y,z
        
        obc_center_2d = obj_center[:2]
        robot_pos_2d = robot_pos[:2]
        
        dims = np.array(target_object['dimensions']) #x,y,z
        #Apprx Radius of the object 
        obj_radius = np.linalg.norm(dims[:2])/2.0

        # 2. Calculate optimal approach distance
        #vector from object to robot
        approach_vector = robot_pos_2d - obj_center_2d
        dist_to_obj = np.linalg.norm(approach_vector)
        
        if dist_to_obj < 1e-3:
            approach_direction = np.array([1.0, 0.0])
        else:
            approach_direction = approach_vector / dist_to_obj
        
        
        # 3. Return goal position

        total_dist_from_center = obj_radius + safe_distance

        goal_xy = obj_center_2d + (approach_direction * total_dist_from_center)

        #Robot must look at the object 
        look_at_vector = -approach_direction
        theta = np.arctan2(look_at_vector[1], look_at_vector[0])
        
        return np.array([goal_xy[0], goal_xy[1], theta])