import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from detector import FaceShapeDetector

# --- 1. CONFIGURATION & INITIALIZATION ---
app = FastAPI(
    title="HairstyleHub AI",
    description="API for detecting face shapes using OpenCV and Machine Learning.",
    version="1.1.0"
)

# Paths (Use absolute paths or environment variables for production)
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_PATH = "models/face_landmark_model.dat"

# Initialize Detector (Singleton pattern)
try:
    detector = FaceShapeDetector(MODEL_PATH, CASCADE_PATH)
except Exception as e:
    print(f"CRITICAL: Could not initialize AI model: {e}")
    # In production, you might want to exit if the model fails to load
    detector = None

# --- 2. MIDDLEWARE (CORS) ---
# Define allowed origins for security
origins = [
    "http://localhost:5500",           # Local Live Server
    "http://127.0.0.1:5500",           # Local IP
    "https://your-hairstylehub.com",    # Production domain
    "https://your-username.github.io", # GitHub Pages
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- 3. ENDPOINTS ---

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": detector is not None,
        "message": "Welcome to HairstyleHub AI API"
    }

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """
    Receives an image, identifies face shape, and returns geometric ratios.
    """
    # 1. Validation: File Type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # 2. Validation: Model Status
    if detector is None:
        raise HTTPException(status_code=500, detail="AI Model is not loaded on server.")

    try:
        # 3. Efficient Image Loading
        # Read directly from memory buffer (avoids slow disk I/O)
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # 4. Processing
        shape, ratio = detector.classify_shape(img)

        # 5. Production Result Format
        return {
            "status": "success",
            "data": {
                "detected_shape": shape,
                "metric_ratio": ratio,
            },
            "meta": {
                "filename": file.filename,
                "message": f"Successfully detected {shape} face shape."
            }
        }

    except Exception as e:
        # Log the error here in a real production app
        return {"status": "error", "message": "An internal error occurred during processing."}

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # In production, uvicorn is usually run via CLI, but this works for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)