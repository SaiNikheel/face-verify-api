import pickle
import json
import numpy as np
import logging
from pathlib import Path
from config import EMBEDDINGS_FILE, IMAGES_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Storage:
    def __init__(self):
        self.embeddings = {}
        self.load_embeddings()

    def load_embeddings(self):
        """Load embeddings from file if it exists."""
        try:
            if EMBEDDINGS_FILE.exists():
                with open(EMBEDDINGS_FILE, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings)} embeddings from storage")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            self.embeddings = {}

    def save_embeddings(self):
        """Save embeddings to file."""
        try:
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Saved {len(self.embeddings)} embeddings to storage")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")

    def save_image(self, employee_id, image_data):
        """Save image to storage."""
        try:
            image_path = IMAGES_DIR / f"{employee_id}.jpg"
            if isinstance(image_data, str):
                import base64
                image_bytes = base64.b64decode(image_data)
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
            else:
                import cv2
                cv2.imwrite(str(image_path), image_data)
            logger.info(f"Saved image for employee {employee_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False

    def add_embedding(self, employee_id, embedding):
        """Add new embedding to storage."""
        if embedding is not None:
            self.embeddings[employee_id] = embedding
            self.save_embeddings()
            return True
        return False

    def get_embedding(self, employee_id):
        """Get embedding for an employee."""
        return self.embeddings.get(employee_id)

    def get_all_embeddings(self):
        """Get all stored embeddings."""
        return self.embeddings

    def remove_embedding(self, employee_id):
        """Remove embedding for an employee."""
        if employee_id in self.embeddings:
            del self.embeddings[employee_id]
            self.save_embeddings()
            return True
        return False 