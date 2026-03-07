import sys
import asyncio

# 1. WINDOWS STABILITY FIX (Must be at the top)
# This prevents the 'WinError 10054' and connection reset logs on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import cv2
import numpy as np
import gc
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from detector import FaceMeshDetector

# 2. APP INITIALIZATION
app = FastAPI(title="HairstyleHub AI")

# 3. CORS CONFIGURATION
# Added GET and OPTIONS for better browser compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# 4. MODEL INITIALIZATION
# Initialized at the module level for Phase 1
try:
    detector = FaceMeshDetector()
except Exception as e:
    print(f"Critical Error: Failed to load FaceMeshDetector: {e}")
    detector = None

# 5. ROUTES
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Stops the 404 logs from browsers looking for an icon."""
    return Response(status_code=204)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Backend is running!"}

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
            raise HTTPException(status_code=400, detail="Invalid image data. Could not decode.")

        # AI Processing
        # The detector handles the 6-shape logic and pose validation
        # AI Processing
        result = detector.classify_shape(img)
        
        # Cleanup
        del img
        del nparr
        gc.collect()

        return {
            "status": "success",
            "detected_shape": result["shape"],
            "metric_ratio": result["ratio"],
            "debug_metrics": result["debug"], # Now you'll see pixels in console
            "message": "Analysis complete"
        }

    except Exception as e:
        gc.collect()
        return {"status": "error", "message": f"Server Error: {str(e)}"}

# 6. SERVER START
if __name__ == "__main__":
    # Render uses the PORT environment variable; local defaults to 8000
    port = int(os.environ.get("PORT", 8000))
    # 'workers=1' is critical for staying under 512MB RAM
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)