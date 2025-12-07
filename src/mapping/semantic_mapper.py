import open3d as o3d
import numpy as np
from collections import defaultdict

class SemanticMapper:
    def __init__(self):
        self.semantic_objects = {}  # object_id -> object_data
        self.global_pointcloud = o3d.geometry.PointCloud()
        self.object_counter = 0
        
        
    def build_semantic_map(self, rgb_images, depth_images, poses, camera_intrinsics,detector, target_queries, background_queries):
        """Build complete semantic map from RGB-D sequence"""
        # Implement:
        print(f"Building semantic map with {len(rgb_images)} frames...")
        # 1. Process each frame with object detection
        for frame_num, (rgb,depth) in enumerate(zip(rgb_images,depth_images)):
            print(f"\n[Frame {frame_num + 1}/{len(rgb_images)}]")

            detections = detector.detect_objects(rgb,target_queries,background_queries)

            if not detections:
                print("No objects detected")
                continue
            
            print(f"Found {len(detections)} objects")
            
            # 2. Project objects to 3D
            objects_3d = []
            for obj in detections:
                result_3d = detector.project_to_3d(
                    obj['mask'],
                    depth,
                    camera_intrinsics
                )
                
                if result_3d is not None:
                    centroid = result_3d['centroid']
                    if poses is not None and frame_num < len(poses):
                        centroid = self.transform_to_global(centroid, poses[frame_num])
                    
                    objects_3d.append({
                        'label': obj['label'],
                        'score': obj['score'],
                        'centroid': centroid.tolist() if isinstance(centroid,np.ndarray) else centroid,
                        'dimensions': result_3d['dimensions'].tolist(),
                        'points_3d': result_3d['points_3d']
                    })
                # 3. Integrate into global semantic map
                self.integrate_semantic_objects(objects_3d,frame_num)

        print("\n" + "="*50)
        print("BUILD COMPLETE")
        self.print_map()
        
        return self.semantic_objects

    def add_new_object(self,label,centroid_3d,dimensions,confidence):
        """Add new object to semantic map"""
        # Implement:
        object_id = self.object_counter
        self.object_counter += 1

        self.semantic_objects[object_id] = {
            'label': label,
            'centroid': centroid_3d,
            'dimensions': dimensions,
            'confidence': confidence,
            'observations': 1,
            'last_seen': 0
        }

        print(f"Added new object {object_id}: {label}")
        return object_id
    
    def find_matching_object(self,label,centroid_3d,distance_threshold=0.3):
        """Find matching object in semantic map"""
        # Implement:
        for obj_id, obj_data in self.semantic_objects.items():
            if obj_data['label'] != label:
                continue

            existing_centroid = np.array(obj_data['centroid'])
            new_centroid = np.array(centroid_3d)
            distance = np.linalg.norm(existing_centroid - new_centroid)

            if distance < distance_threshold:
                return obj_id

        return None
    
    def update_object(self,object_id,centroid_3d,confidence,frame_number):
        """Update existing object in semantic map"""
        obj = self.semantic_objects[object_id]
        
        old_centroid = np.array(obj['centroid'])
        new_centroid = np.array(centroid_3d)
        
        n = obj['observations']
        updated_centroid = (old_centroid * n + new_centroid) / (n + 1)
        
        obj['centroid'] = updated_centroid.tolist()
        obj['confidence'] = max(obj['confidence'], confidence)
        obj['observations'] += 1
        obj['last_seen'] = frame_number
        
        print(f"    ~ Actualizado #{object_id}: {obj['label']} (obs: {obj['observations']})")
    
    def integrate_semantic_objects(self,detected_objects_3d,frame_number=0):
        """Integrate detected objects into semantic map"""
        # Implement:
        for obj in detected_objects_3d:
            label = obj['label']
            centroid = obj['centroid']
            dimensions = obj.get('dimensions', [0,0,0])
            confidence = obj['score']
            
            existing_id = self.find_matching_object(label,centroid)
            
            if existing_id is not None:
                self.update_object(existing_id,centroid,confidence,frame_number)
            else:
                self.add_new_object(label,centroid,dimensions,confidence)    
    
    def transform_to_global(self,points_local, pose):
        """Transform point from local to global coordinates"""
        # Implement:
        points_local = np.array(points_local)
    
        # If single point [X, Y, Z]
        if points_local.ndim == 1:
            point_homogeneous = np.append(points_local, 1.0)
            point_global = pose @ point_homogeneous
            return point_global[:3]
        
        # If multiple points (N, 3)
        else:
            N = points_local.shape[0]
            ones = np.ones((N, 1))
            points_homogeneous = np.hstack([points_local, ones])  # (N, 4)
            points_global = (pose @ points_homogeneous.T).T       # (N, 4)
            return points_global[:, :3]                            # (N, 3)
    
    def get_semantic_map(self):
        """Return semantic map as dictionary"""
        return self.semantic_objects
    
    def print_map(self):
        """Print semantic map"""
        print("\n" + "="*50)
        print("SEMANTIC MAP")
        print("="*50)
        
        if not self.semantic_objects:
            print("(empty)")
            return
        
        for obj_id, obj in self.semantic_objects.items():
            c = obj['centroid']
            print(f"\n[{obj_id}] {obj['label']}")
            print(f"    Position: X={c[0]:.2f}m, Y={c[1]:.2f}m, Z={c[2]:.2f}m")
            print(f"    Confidence: {obj['confidence']:.2f}")
            print(f"    Observations: {obj['observations']}")