import os
import torch
import numpy as np
import cv2
from PIL import Image

import open_clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class OpenVocabularyDetector:
    """
    Open-vocabulary object detector using:
      - OpenCLIP for image/text embeddings
      - SAM for segmentation proposals

    Notes:
      - 'fast=True' reduces SAM compute for CPU demos.
      - sam_checkpoint must point to a valid SAM .pth file.
    """

    def __init__(self, sam_checkpoint: str, fast: bool = False, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fast = bool(fast)
        self.sam_checkpoint = sam_checkpoint

        print(f"Device: {self.device}", flush=True)
        self.setup_models()

    def setup_models(self):
        # ---- OpenCLIP ----
        print("Loading OpenCLIP model...", flush=True)
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # ---- SAM ----
        print("Loading SAM model...", flush=True)
        if not os.path.exists(self.sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")

        sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
        sam = sam.to(self.device)

        if self.fast:
            # CPU-friendly settings
            self.sam = SamAutomaticMaskGenerator(
                sam,
                points_per_side=8,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.85,
                crop_n_layers=0,
            )
        else:
            self.sam = SamAutomaticMaskGenerator(
                sam,
                points_per_side=16,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.9,
                crop_n_layers=0,
            )

        print("Models loaded successfully", flush=True)

    def build_text_prompts(self, base_queries):
        templates = [
            "a photo of a {}",
            "a {} in the scene",
            "an image containing a {}",
        ]
        enhanced = []
        for q in base_queries:
            enhanced.extend([t.format(q) for t in templates])
        return enhanced, len(templates)

    def get_crop_with_context(self, image_rgb, bbox, context_ratio=0.15):
        x, y, w, h = map(int, bbox)
        img_h, img_w = image_rgb.shape[:2]
        margin_x = int(w * context_ratio)
        margin_y = int(h * context_ratio)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)
        return image_rgb[y1:y2, x1:x2]

    def detect_objects(self, image, text_queries, background_queries=None):
        if background_queries:
            all_queries = text_queries + background_queries
            background_start = len(text_queries)
            background_indices = set(range(background_start, len(all_queries)))
        else:
            all_queries = text_queries
            background_indices = set()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("SAM: Scanning image...", flush=True)
        sam_results = self.sam.generate(image_rgb)
        print(f"   -> Found {len(sam_results)} regions.", flush=True)

        # Filter and prep crops
        crops = []
        valid_indices = []

        img_h, img_w = image_rgb.shape[:2]
        img_area = img_h * img_w

        min_area = 600 if self.fast else 400
        max_area_ratio = 0.30 if self.fast else 0.35
        max_aspect_ratio = 5.0

        for i, res in enumerate(sam_results):
            x, y, w, h = map(int, res["bbox"])
            area = w * h
            aspect = max(w, h) / (min(w, h) + 1e-6)
            area_ratio = area / img_area

            if area < min_area:
                continue
            if area_ratio > max_area_ratio:
                continue
            if aspect > max_aspect_ratio:
                continue

            crop = self.get_crop_with_context(image_rgb, res["bbox"], context_ratio=0.10)
            crop_pil = Image.fromarray(crop)
            processed = self.clip_preprocess(crop_pil)
            crops.append(processed)
            valid_indices.append(i)

        if not crops:
            print("No valid regions found.", flush=True)
            return []

        image_input = torch.stack(crops).to(self.device)

        # Text embeddings (prompt ensembling)
        enhanced_prompts, num_templates = self.build_text_prompts(all_queries)
        text_tokens = self.clip_tokenizer(enhanced_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features = text_features.view(len(all_queries), num_templates, -1).mean(dim=1)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(1, dim=-1)

        detected = []
        confidence_threshold = 0.55 if self.fast else 0.50

        for idx_in_batch, sam_idx in enumerate(valid_indices):
            score = values[idx_in_batch].item()
            label_idx = indices[idx_in_batch].item()
            if label_idx in background_indices:
                continue
            if score >= confidence_threshold:
                mask_data = sam_results[sam_idx]
                detected.append(
                    {
                        "box": mask_data["bbox"],
                        "label": all_queries[label_idx],
                        "score": float(score),
                        "mask": mask_data["segmentation"],
                    }
                )

        return detected

    def project_to_3d(self, object_mask, depth_image, camera_intrinsics):
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics["fy"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]

        if object_mask.dtype == np.uint8:
            object_mask = object_mask.astype(bool)

        v_coords, u_coords = np.where(object_mask)
        if len(v_coords) == 0:
            return None

        depths = depth_image[v_coords, u_coords].astype(np.float32)
        valid = (depths > 0.1) & (depths < 10.0)
        if not np.any(valid):
            return None

        u = u_coords[valid].astype(np.float32)
        v = v_coords[valid].astype(np.float32)
        Z = depths[valid]

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points_3d = np.stack([X, Y, Z], axis=1)

        centroid = np.mean(points_3d, axis=0)
        min_bounds = np.min(points_3d, axis=0)
        max_bounds = np.max(points_3d, axis=0)
        dimensions = max_bounds - min_bounds

        return {
            "points_3d": points_3d,
            "centroid": centroid,
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
            "dimensions": dimensions,
            "num_points": int(points_3d.shape[0]),
        }
