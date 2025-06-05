from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import logging
from typing import Optional
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from face_engine import FaceEngine
from storage import Storage
from config import FACE_RECOGNITION_THRESHOLD, API_HOST, API_PORT, DEBUG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="API for face registration and verification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to increase request size limit
class LargeRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Increase the maximum request size limit
        request._body = await request.body()
        return await call_next(request)

app.add_middleware(LargeRequestMiddleware)

# Initialize face engine and storage
face_engine = FaceEngine()
storage = Storage()

class RegisterRequest(BaseModel):
    employee_id: str
    image: str  # Base64 encoded image

class VerifyRequest(BaseModel):
    image: str  # Base64 encoded image

class VerifyResponse(BaseModel):
    verified: bool
    employee_id: Optional[str] = None
    similarity: Optional[float] = None

@app.post("/register")
async def register_face(request: RegisterRequest):
    """Register a new face."""
    try:
        # Get face embedding
        embedding = face_engine.get_embedding(request.image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        # Save image and embedding
        if storage.save_image(request.employee_id, request.image):
            if storage.add_embedding(request.employee_id, embedding):
                return {
                    "status": "success",
                    "message": f"Face registered successfully for employee {request.employee_id}"
                }
        
        raise HTTPException(status_code=500, detail="Failed to save face data")
    
    except Exception as e:
        logger.error(f"Error in registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify", response_model=VerifyResponse)
async def verify_face(request: VerifyRequest):
    """Verify a face against registered faces."""
    try:
        # Get face embedding
        embedding = face_engine.get_embedding(request.image)
        if embedding is None:
            return VerifyResponse(verified=False)

        # Compare with all stored embeddings
        best_match = None
        best_similarity = 0.0

        for employee_id, stored_embedding in storage.get_all_embeddings().items():
            similarity = face_engine.compare_faces(embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = employee_id

        if best_similarity >= FACE_RECOGNITION_THRESHOLD:
            return VerifyResponse(
                verified=True,
                employee_id=best_match,
                similarity=best_similarity
            )
        
        return VerifyResponse(verified=False)
    
    except Exception as e:
        logger.error(f"Error in verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    ) 