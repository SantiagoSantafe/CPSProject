import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import clip

class SemanticQueryEngine:
    def __init__(self, semantic_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.semantic_map = semantic_map

        #Load only CLIP text part for fast searchs
        print("Loading CLIP text encoder...")
        self.model, _ = clip.load("ViT-B/32", device=self.device)

        self.object_ids = []
        self.index = None
        self.embedding_cache = None

        if self.semantic_map:
            self.setup_search_index()
        
    def setup_search_index(self):
        """Build search index for semantic queries based on object labels"""
        # Implement:
        # 1. Collect all object embeddings
        if not self.semantic_map:
            print("Warning: Semantic map is empty")
            return
        
        self.object_ids = list(self.semantic_map.keys())
        labels = [self.semantic_map[obj_id]['label'] for obj_id in self.object_ids]

        print(f"Indexing {len(labels)} objects...")

        text_inputs = clip.tokenize(labels).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            #Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        self.embedding_cache = text_features.cpu().numpy()
            
        # 2. Build nearest neighbors index

        #Use Nearest neighbors index to find similar objects
        self.index = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(self.embedding_cache)
        print("Search index built successfully")
        
    def query_objects(self, text_query, max_results=5, spatial_constraints=None):
        """Query objects using natural language"""
        # Implement:
        if self.index is None:
            print("Warning: Search index is not built. Call setup_search_index() first.")
            return []
        # 1. Convert text to CLIP embedding
        text_inputs = clip.tokenize([text_query]).to(self.device)

        with torch.no_grad():
            query_feature = self.model.encode_text(text_inputs)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            query_vec = query_feature.cpu().numpy()
        
        # 2. Find semantically similar objects
        distances, indices = self.index.kneighbors(query_vec)

        results = []
        for rank, idx in enumerate(indices[0]):
            obj_id = self.object_ids[idx]
            obj_data = self.semantic_map[obj_id]
            
            #Translate that distance into coseno or a similarity score
            similarity = 1.0 - distances[0][rank]
            
            # 3. Apply spatial constraints

            if spatial_constraints:
                if "near_pos" in spatial_constraints:
                    pos_obj = np.array(obj_data['centroid'])
                    pos_ref = np.array(spatial_constraints['near_pos'])
                    dist = np.linalg.norm(pos_obj - pos_ref)
                    if dist > spatial_constraints.get('radius',2.0):
                        continue
            results.append({
                'id': obj_id,
                'label': obj_data['label'],
                'centroid': obj_data['centroid'],
                'similarity': similarity,
                'dimensions': obj_data.get('dimensions', [0.5, 0.5, 0.5]) # fallback dims
            })
        
        # 4. Return ranked results
        return results
        
    def parse_navigation_command(self, command_text):
        """Parse natural language navigation command"""
        # Implement:
        # 1. Extract target object description
        command_text = command_text.lower()

        target_desc = command_text
        spatial_rel = None
        # 2. Identify spatial relationships

        spatial_keywords = ['near', 'far', 'left', 'right', 'above', 'below','front','back','center','middle','top','bottom','beside','next to','opposite','across','behind','in front of','between','beside','next to','opposite','across','behind','in front of','between','beside','next to','opposite','across','behind','in front of','between']
        
        verbs = ['go to', 'move to', 'navigate to', 'find', 'approach']
        # 3. Return structured query

        for verb in verbs:
            target_desc = target_desc.replace(verb,"").strip()
        
        return{
            'target_desc': target_desc,
            'spatial_rel': spatial_rel
        }