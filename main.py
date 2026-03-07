import cv2
import numpy as np
import os
import gc
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- 1. OPTIMIZED DETECTOR CLASS ---
class FaceShapeDetector:
    def __init__(self, model_path, cascade_path):
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade missing: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self.facemark = cv2.face.createFacemarkKazemi()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model missing: {model_path}")
        self.facemark.loadModel(model_path)

    def _get_landmarks(self, image):
        max_dim = 800
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        ok, landmarks = self.facemark.fit(image, faces)
        del gray # Free memory immediately
        
        if not ok or len(landmarks) == 0:
            return None
            
        return landmarks[0][0]

    def classify_shape(self, image):
        points = self._get_landmarks(image)
        if points is None:
            return "No face detected", 0.0

        try:
            # Euclidean distance logic
            def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

            jaw_w = dist(points[4], points[12])
            cheek_w = dist(points[2], points[14])
            forehead_w = dist(points[19], points[24])
            face_len = dist(points[27], points[8])
            ratio_lw = face_len / cheek_w
            
            if ratio_lw > 1.25: shape = "Oblong"
            elif (jaw_w / cheek_w) > 0.9: shape = "Square"
            elif (forehead_w / cheek_w) > 0.85 and (jaw_w / cheek_w) < 0.8: shape = "Heart"
            elif 0.9 <= ratio_lw <= 1.05: shape = "Round"
            else: shape = "Oval"

            return shape, round(float(ratio_lw), 2)
        finally:
            del points
            gc.collect()

# --- 2. API LIFESPAN MANAGEMENT ---
# Loads the model once when the server starts to save time and memory
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    MODEL_PATH = "models/face_landmark_model.dat"
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    detector = FaceShapeDetector(MODEL_PATH, CASCADE_PATH)
    yield
    del detector
    gc.collect()

# --- 3. FASTAPI APP SETUP ---
app = FastAPI(lifespan=lifespan, title="HairstyleHub AI")

# Production CORS - Replace "*" with your actual domain when ready
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
async def health():
    return {"status": "online", "model_ready": detector is not None}

@app.post("/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # Read file into memory buffer efficiently
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        shape, ratio = detector.classify_shape(img)
        
        # Immediate cleanup
        del img
        del nparr
        gc.collect()

        return {
            "status": "success",
            "detected_shape": shape,
            "metric_ratio": ratio
        }

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)