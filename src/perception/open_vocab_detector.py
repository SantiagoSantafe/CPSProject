import torch
import clip
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from PIL import Image

class OpenVocabularyDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        self.setup_models()
        
    def setup_models(self):
        """Initialize CLIP and SAM models"""
        # Load CLIP model
        print("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load SAM model
        print("Loading SAM model...")
        sam_checkpoint = "models/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        #Make the change to automatic mask generator
        self.sam_predictor = SamAutomaticMaskGenerator(sam.to(self.device))
        print("Models loaded successfully")
        
    def detect_objects(self, image, text_queries, background_queries=None):
        """Detect objects in image using open-vocabulary models
        
        Args:
            image (numpy.ndarray): Input image
            text_queries (list): List of text queries
        
        Returns:
            list: List of detected objects with semantic labels and confidence scores
        """
        # Implement:

        #Indices of background objects
        if background_queries:
            all_queries = text_queries + background_queries
            background_start = len(text_queries)
            background_indices = set(range(background_start, len(all_queries)))
        else:
            all_queries = text_queries
            background_queries = []
            background_indices = set()

        #SAM works better with RGB
        #Preprocess the image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 1. Use SAM to segment image into regions
        print("SAM: Scanning image...")
        sam_results = self.sam_predictor.generate(image)
        print(f"   -> Founded {len(sam_results)} regions.")
        # 2. Extract CLIP embeddings for each region
        print("CLIP: Extracting embeddings...")
        crops = []
        valid_indices = []

        img_h, img_w = image_rgb.shape[:2]
        img_area = img_h * img_w    

        min_area = 400
        max_area_ratio = 0.35
        max_aspect_ratio = 5.0

        for i, res in enumerate(sam_results):
            #Get coordinates x, y, w, h
            x, y, w, h = map(int, res["bbox"])
            area = w * h
            aspect = max(w, h) / (min(w, h) + 1e-6)
            area_ratio = area / img_area

            #Size filters
            if area < min_area:
                continue
            if area_ratio > max_area_ratio:
                continue
            if aspect > max_aspect_ratio:
                continue

            #Crop only the region of interest
            crop = self.get_crop_with_context(image_rgb, res["bbox"],context_ratio=0.1)
            #Convert to PIL and preprocess to clip format
            crop_pil = Image.fromarray(crop)
            processed_crop = self.clip_preprocess(crop_pil)

            crops.append(processed_crop)
            valid_indices.append(i)
        if not crops:
            print("No valid regions found.")
            return []

        #Convert the list into a tensor 
        image_input = torch.stack(crops).to(self.device)
        print("CLIP: Embeddings extracted successfully")


        # 3. Compare with text query embeddings

        print("CLIP: Comparing embeddings...")
        enhanced_prompts, num_templates = self.build_text_prompts(all_queries)
        text_tokens = clip.tokenize(enhanced_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)

            #Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features = text_features.view(len(all_queries), num_templates, -1)
            text_features = text_features.mean(dim=1)  # (num_queries, embed_dim)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(1, dim=-1)
            
        # 4. Return object masks with semantic labels and confidence scores
        detected_objects = []
        confidence_threshold = 0.5 # Minimum confidence threshold

        

        for idx_in_batch, sam_idx in enumerate(valid_indices):
            score = values[idx_in_batch].item()
            label_idx = indices[idx_in_batch].item()
            if label_idx in background_indices:
                continue
            # Only save if it exceeds the confidence threshold
            if score > confidence_threshold:
                mask_data = sam_results[sam_idx]
                detected_objects.append({
                    "box": mask_data['bbox'],
                    "label": all_queries[label_idx],
                    "score": score,
                    "mask": mask_data['segmentation'] # Binary mask
                })

        print(f"DEBUG: Objetos antes de filtrar: {len(detected_objects)}")
        for obj in detected_objects:
            print(f"  - {obj['label']}: {obj['score']:.2f}") 
        
        if not detected_objects:
            return []

        # Edge filtering
        detected_objects = self.filter_edge_detections(detected_objects, image.shape)
        print(f"DEBUG: Después de filter_edge: {len(detected_objects)}")
    
        # NMS per class
        final_objects = self.apply_nms_per_class(detected_objects, nms_threshold=0.10)
        print(f"DEBUG: Después de NMS: {len(final_objects)}")   
        
        
        # Overlapping filtering
        final_objects = self.filter_overlapping_detections(final_objects, iou_threshold=0.6)
        print(f"DEBUG: Después de Overlapping: {len(final_objects)}")
        
        return final_objects
        
    def project_to_3d(self, object_mask, depth_image, camera_intrinsics):
        """Project 2D object mask to 3D using depth information"""
        # Implement:
        #Extract camera intrinsics
        fx = camera_intrinsics['fx']
        fy = camera_intrinsics['fy']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']
        
        # 1. Use factory calibration (aligned RGB-D)
        if object_mask.dtype == np.uint8:
            object_mask = object_mask.astype(bool)
        
        # 2. Convert masked depth pixels to 3D points
        v_coords, u_coords = np.where(object_mask).astype(bool)
        
        if len(v_coords) == 0 or len(u_coords) == 0:
            print("Warning: No valid points found in the mask.")
            return None
        #Get depth values at masked pixels
        depths = depth_image[v_coords, u_coords].astype(np.float32)

        #Filter invalid depth values
        valid_mask = (depths > 0.1) & (depths < 10.0)

        if not np.any(valid_mask):
            print("Warning: No valid depth values found.")
            return None 
        
        u = u_coords[valid_mask].astype(np.float32)
        v = v_coords[valid_mask].astype(np.float32)
        Z = depths[valid_mask]
        
        # Back-projection formula:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        points_3d = np.stack([X, Y, Z], axis=1)   
    
        # 3. Calculate object centroid and bounds

        centroid = np.mean(points_3d, axis=0)
        min_bounds = np.min(points_3d, axis=0)
        max_bounds = np.max(points_3d, axis=0)
        dimensions = max_bounds - min_bounds

        return {
            'points_3d': points_3d,        # (N, 3) point cloud
            'centroid': centroid,           # [X, Y, Z] center in meters
            'min_bounds': min_bounds,       # [X, Y, Z] min corner
            'max_bounds': max_bounds,       # [X, Y, Z] max corner  
            'dimensions': dimensions,       # [width, height, depth] in meters
            'num_points': len(points_3d)
        }
        
    
        

    def get_crop_with_context(self, image_rgb, bbox, context_ratio=0.15):
        """Extract crop with additional context"""
        x, y, w, h = map(int, bbox)
        img_h, img_w = image_rgb.shape[:2]
    
        # Add context margin
        margin_x = int(w * context_ratio)
        margin_y = int(h * context_ratio)
    
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
    
        return image_rgb[y1:y2, x1:x2]

    def build_text_prompts(self, base_queries):
        """Build more descriptive prompts for CLIP"""
        templates = [
            "a photo of a {}",
            "a {} in the scene",
            "an image containing a {}"
        ]
        
        enhanced_queries = []
        for query in base_queries:
            # Use multiple templates and average embeddings
            enhanced_queries.extend([t.format(query) for t in templates])
        
        return enhanced_queries, len(templates)


    def apply_nms_per_class(self, detected_objects, nms_threshold=0.3):
        """Applies NMS per class"""
        
        # Group by class
        by_class = {}
        for obj in detected_objects:
            label = obj['label']
            if label not in by_class:
                by_class[label] = []
            by_class[label].append(obj)
        
        final_objects = []
        
        for label, objects in by_class.items():
            if len(objects) == 1:
                final_objects.append(objects[0])
                continue
            
            boxes = [obj['box'] for obj in objects]
            scores = [float(obj['score']) for obj in objects]
            
            indices = cv2.dnn.NMSBoxes(boxes, scores, 
                                        score_threshold=0.0, 
                                        nms_threshold=nms_threshold)
            
            if len(indices) > 0:
                for i in indices.flatten():
                    final_objects.append(objects[i])
        
        return final_objects

    def filter_by_relative_size(self, detected_objects, image_shape, max_ratio=0.4):
        """Erases objects that cover too much of the image (probably false positives)"""
        img_h, img_w = image_shape[:2]
        img_area = img_h * img_w
        
        filtered = []
        for obj in detected_objects:
            x, y, w, h = obj['box']
            obj_area = w * h
            ratio = obj_area / img_area
            
            # If it covers more than 40% of the image, it's probably a false positive
            if ratio < max_ratio:
                filtered.append(obj)
            else:
                print(f"  Filtered {obj['label']} (covers {ratio*100:.1f}% of image)") 
        
        return filtered
    
    def filter_edge_detections(self, detected_objects, image_shape, edge_threshold=0.3):
        """Filter detections that are mostly on the image border"""
        img_h, img_w = image_shape[:2]
        filtered = []
    
        for obj in detected_objects:
            x, y, w, h = map(int, obj['box'])
        
            # Calcular qué porcentaje del box está dentro de la imagen
            x_end = min(x + w, img_w)
            y_end = min(y + h, img_h)
            x_start = max(x, 0)
            y_start = max(y, 0)
            
            visible_area = (x_end - x_start) * (y_end - y_start)
            total_area = w * h
            
            if total_area > 0:
                visible_ratio = visible_area / total_area
                if visible_ratio > (1 - edge_threshold):  # At least 70% visible
                    filtered.append(obj)
                else:
                    print(f"  Filtered {obj['label']} (only {visible_ratio*100:.0f}% visible)")
    
        return filtered
    
    def filter_overlapping_detections(self, detected_objects, iou_threshold=0.5):
        """Erases detections that are too overlapping between different classes"""
        if len(detected_objects) <= 1:
            return detected_objects
        
        def compute_iou(box1, box2):
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = w1 * h1
            box2_area = w2 * h2
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Sort by score descending
        sorted_objects = sorted(detected_objects, key=lambda x: x['score'], reverse=True)
        
        keep = []
        for obj in sorted_objects:
            should_keep = True
            for kept_obj in keep:
                iou = compute_iou(obj['box'], kept_obj['box'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(obj)
        
        return keep 
    