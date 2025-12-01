import gdown
import os

def download_models():
    os.makedirs('models', exist_ok=True)
    
    # Grounding DINO
    if not os.path.exists('models/groundingdino_swint_ogc.pth'):
        print("Downloading Grounding DINO...")
        gdown.download(
            'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-swint-ogc/groundingdino_swint_ogc.pth',
            'models/groundingdino_swint_ogc.pth', 
            quiet=False
        )
    
    # Segment Anything Model
    if not os.path.exists('models/sam_vit_h_4b8939.pth'):
        print("Downloading SAM...")
        gdown.download(
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'models/sam_vit_h_4b8939.pth',
            quiet=False
        )
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()