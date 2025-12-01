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
        
    def detect_objects(self, image, text_queries):
        """Detect objects in image using open-vocabulary models
        
        Args:
            image (numpy.ndarray): Input image
            text_queries (list): List of text queries
        
        Returns:
            list: List of detected objects with semantic labels and confidence scores
        """
        # Implement:

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
        for i, res in enumerate(sam_results):
            #Get coordinates x, y, w, h
            x, y, w, h = map(int, res["bbox"])

            #Ignore small regions
            if w < 10 or h < 10:
                continue

            #Crop only the region of interest
            crop = image_rgb[y:y+h, x:x+w]
            #Convert to PIL and preprocess to clip format
            crop_pil = Image.fromarray(crop)
            processed_crop = self.clip_preprocess(crop_pil)

            crops.append(processed_crop)
            valid_indices.append(i)
        if not crops:
            print("No valid regions found.")
            return []
        #Convert the list into a tensor 
        image_input = torch.tensor(np.stack(crops)).to(self.device)
        print("CLIP: Embeddings extracted successfully")
        # 3. Compare with text query embeddings

        print("CLIP: Comparing embeddings...")
        text_tokens = clip.tokenize(text_queries).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)

            #Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            #Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity.topk(1, dim=-1)
            
        # 4. Return object masks with semantic labels and confidence scores
        detected_objects = []
        confidence_threshold = 0.65 # Minimum confidence threshold
        
        for idx_in_batch, sam_idx in enumerate(valid_indices):
            score = values[idx_in_batch].item()
            label_idx = indices[idx_in_batch].item()
            
            # Only save if it exceeds the confidence threshold
            if score > confidence_threshold:
                mask_data = sam_results[sam_idx]
                detected_objects.append({
                    "box": mask_data['bbox'],
                    "label": text_queries[label_idx],
                    "score": score,
                    "mask": mask_data['segmentation'] # Binary mask
                })


        if not detected_objects:
            return []

        # Prepare data for OpenCV NMS
        boxes = []
        scores = []
        class_indices = []
        
        # Extract data from our object list
        for obj in detected_objects:
            boxes.append(obj['box'])   # [x, y, w, h]
            scores.append(float(obj['score']))
            # Find the numeric index of the text to know the class
            class_indices.append(text_queries.index(obj['label']))

        # --- Cleaning NMS ---
        # nms_threshold: 0.3 means "if two boxes overlap more than 30%, delete the worst"
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=0.3)
        
        final_objects = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_objects.append(detected_objects[i])
        return detected_objects
        
    def project_to_3d(self, object_mask, depth_image, camera_intrinsics):
        """Project 2D object mask to 3D using depth information"""
        # Implement:
        # 1. Use factory calibration (aligned RGB-D)
        # 2. Convert masked depth pixels to 3D points
        # 3. Calculate object centroid and bounds
        pass

if __name__ == "__main__":
    # 1. Instanciar el detector
    detector = OpenVocabularyDetector()
    
    # 2. Iniciar la Webcam (0 suele ser la cámara por defecto)
    print("Iniciando cámara... (Mira la ventana que se abre)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit()

    print("--> PRESIONA 'ESPACIO' PARA TOMAR LA FOTO Y DETECTAR <---")
    print("--> PRESIONA 'Q' PARA SALIR <---")

    frame_to_process = None

    # Bucle para mostrar el video en vivo antes de capturar
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir imagen de la cámara.")
            break

        # Mostrar instrucciones en la pantalla
        cv2.putText(frame, "Presiona ESPACIO para detectar", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Camara en vivo", frame)

        # Esperar tecla
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Tecla Espacio
            frame_to_process = frame.copy() # Guardamos la foto
            break
        elif key == ord('q'): # Tecla Q
            break
    
    # Soltamos la cámara (ya tenemos la foto)
    cap.release()
    cv2.destroyWindow("Camara en vivo")

    # 3. Si tomamos una foto, procesarla
    if frame_to_process is not None:
        print("Procesando imagen... Espera un momento...")
        
        # Definir qué buscar
        queries = [
            "person", "smartphone", "eyeglasses", "bottle", "hand",
            "ceiling", "wall", "lights", "fluorescent lights", "background", "empty space"
        ]
        
        # DETECTAR
        results = detector.detect_objects(frame_to_process, queries)
        
        print(f"\nSe detectaron {len(results)} objetos.")
        
        # DIBUJAR
        for res in results:
            x, y, w, h = map(int, res['box'])
            label = res['label']
            score = res['score']

            basura = ["ceiling", "wall", "lights", "fluorescent lights", "background", "empty space"]
            if res['label'] in basura:
                continue
            # Rectángulo y Texto
            cv2.rectangle(frame_to_process, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({score:.2f})"
            cv2.putText(frame_to_process, text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar resultado final estático
        cv2.imshow("Resultado de Deteccion", frame_to_process)
        print("Presiona cualquier tecla para cerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Saliste sin tomar foto.")