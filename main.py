import sys
import asyncio
import cv2
import numpy as np
import gc
import os
import uvicorn
import httpx
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

# 5. DATA COLLECTION CONFIG
# Backend URL — set BACKEND_API_URL env var on Render to override
_BACKEND_URL = os.environ.get(
    "BACKEND_API_URL",
    "https://hairstyle-hub-backend.onrender.com/api/face-analysis/save-face-data"
)
# AI model version — bump this in Render env vars when you retrain
_MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")


async def _store_analysis(analysis: dict) -> None:
    """
    Fire-and-forget: POST analysis result to the backend database.

    Rules:
    - Never raises — a backend failure must never affect the AI response.
    - 4 s timeout — Render free tier can be slow waking from sleep, but
      we do not want to block the response coroutine indefinitely.
    - Only called on clean successful classifications, not on errors or
      Indeterminate results (those have no ground-truth value yet).
    - Parses confidence "72.4%" → 72.4 (float) to match Mongoose Number type.
    """
    try:
        debug   = analysis["debug"]
        conf_str = analysis["confidence"]                      # e.g. "72.4%"
        conf_num = float(conf_str.rstrip("%"))                 # → 72.4

        payload = {
            "detected_shape" : analysis["detected_shape"],
            "confidence"     : conf_num,
            "lw_ratio"       : debug["lw_ratio"],
            "fc_ratio"       : debug["fc_ratio"],
            "jc_ratio"       : debug["jc_ratio"],
            "fj_ratio"       : debug["fj_ratio"],
            "version"        : _MODEL_VERSION,
        }

        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(_BACKEND_URL, json=payload)

    except Exception as e:
        # Log but swallow — data collection is best-effort only
        print(f"[data-collect] store failed (non-critical): {e}")


#7. ROUTES

# ✅ Correct FastAPI syntax
@app.get("/ping")
def health_check():
    return {"status": "online", "message": "Backend is warm"}

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
        nparr    = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

        # Skip storage for Indeterminate — no reliable label to store
        if analysis["detected_shape"] == "Indeterminate":
            return {
                "status"        : "success",
                "detected_shape": analysis["detected_shape"],
                "confidence"    : analysis["confidence"],
                "metric_ratio"  : analysis["metric_ratio"],
                "debug_metrics" : analysis["debug"],
                "message"       : "Analysis complete",
            }

        # Store to database — fire-and-forget, does not block response
        asyncio.create_task(_store_analysis(analysis))

        # 7. RETURN STRUCTURE (Matches Team Requirements)
        return {
            "status"        : "success",
            "detected_shape": analysis["detected_shape"],
            "confidence"    : analysis["confidence"],
            "metric_ratio"  : analysis["metric_ratio"],
            "debug_metrics" : analysis["debug"],
            "message"       : "Analysis complete",
        }

    except Exception as e:
        gc.collect()
        print(f"Server Error: {str(e)}")
        return {"status": "error", "message": "An internal server error occurred."}


# 8. SERVER START
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # Bind to 0.0.0.0 so it's accessible externally
    uvicorn.run(app, host="0.0.0.0", port=port)