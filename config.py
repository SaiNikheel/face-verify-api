import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"

# Create directories if they don't exist
STORAGE_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Face recognition settings
FACE_DETECTION_CONFIDENCE = 0.5
FACE_RECOGNITION_THRESHOLD = 0.6
FACE_DETECTION_MODEL = "buffalo_l"  # InsightFace model name

# Storage settings
EMBEDDINGS_FILE = STORAGE_DIR / "embeddings.pkl"

# API settings
API_HOST = "localhost"
API_PORT = 1000
DEBUG = True 