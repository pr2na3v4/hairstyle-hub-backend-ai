from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from detector import FaceShapeDetector # Importing your logic

# 1. Initialize FastAPI and AI Detector
app = FastAPI(title="HairstyleHub AI - Face Shape Detection")

# Load your model path correctly
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_PATH = "models/face_landmark_model.dat"

# Fix: Pass BOTH arguments to the detector
detector = FaceShapeDetector(MODEL_PATH, CASCADE_PATH)

# 2. Setup CORS (Crucial for website integration)
# Replace "*" with your actual website URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    """
    Receives an image, processes it through OpenCV/Facemark, 
    and returns the detected face shape.
    """
    try:
        # Check if file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image.")

        # Read image bytes and convert to OpenCV format
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data.")

        # 3. Call your FaceShapeDetector logic
        shape, ratio = detector.classify_shape(img)

        # 4. Return results as JSON
        return {
            "status": "success",
            "detected_shape": shape,
            "metric_ratio": ratio,
            "message": f"Successfully detected {shape} face shape."
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Start server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)