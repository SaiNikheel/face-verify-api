import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import logging
from config import FACE_DETECTION_CONFIDENCE, FACE_DETECTION_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(name=FACE_DETECTION_MODEL)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("Face engine initialized with model: %s", FACE_DETECTION_MODEL)

    def _preprocess_image(self, image_data):
        """Convert image data to numpy array."""
        if isinstance(image_data, str):
            # Handle base64 string
            import base64
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Handle numpy array or bytes
            image = image_data
        return image

    def detect_face(self, image_data):
        """Detect and align face in the image."""
        try:
            image = self._preprocess_image(image_data)
            faces = self.app.get(image)
            
            if not faces:
                logger.warning("No faces detected in the image")
                return None
            
            # Get the face with highest detection confidence
            face = max(faces, key=lambda x: x.det_score)
            
            if face.det_score < FACE_DETECTION_CONFIDENCE:
                logger.warning(f"Face detection confidence too low: {face.det_score}")
                return None
            
            return face
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return None

    def get_embedding(self, image_data):
        """Extract face embedding from the image."""
        face = self.detect_face(image_data)
        if face is None:
            return None
        return face.embedding

    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings using cosine similarity."""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity) 