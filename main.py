import sys
import asyncio
import cv2
import numpy as np
import gc
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from detector import FaceMeshDetector

# 1. WINDOWS STABILITY FIX
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 2. APP INITIALIZATION
app = FastAPI(title="HairstyleHub AI")

# 3. CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST", "GET", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# 4. MODEL INITIALIZATION
try:
    detector = FaceMeshDetector()
except Exception as e:
    print(f"Critical Error: Failed to load FaceMeshDetector: {e}")
    detector = None

# 5. ROUTES
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/")
def read_root():
    """Handles Render health checks and browser pings."""
    return {"status": "online", "message": "HairstyleHub AI Backend is active"}

@app.post("/analyze-face")
async def analyze(file: UploadFile = File(...)):
    # Validation: Ensure it's an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    if detector is None:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    try:
        # Efficient Image Loading
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        # AI Processing
        analysis = detector.classify_shape(img)
        
        # Cleanup immediately to save RAM
        del img
        del nparr
        gc.collect()

        # Handle case where no face is found or pose is invalid
        if analysis is None:
            return {"status": "error", "message": "No face detected. Please try again."}
        
        if "error" in analysis:
            return {"status": "error", "message": analysis["error"]}

        # 6. RETURN STRUCTURE (Matches Team Requirements)
        return {
            "status": "success",
            "detected_shape": analysis["detected_shape"],
            "confidence": analysis["confidence"],
            "metric_ratio": analysis["metric_ratio"],
            "debug_metrics": analysis["debug"],
            "message": "Analysis complete"
        }

    except Exception as e:
        gc.collect()
        print(f"Server Error: {str(e)}")
        return {"status": "error", "message": "An internal server error occurred."}

# 7. SERVER START
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # 'workers=1' is mandatory for 512MB RAM limits on Render Free Tier
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)