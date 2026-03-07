import cv2
import numpy as np
import os
import gc

class FaceShapeDetector:
    def __init__(self, model_path, cascade_path):
        # 1. Load the Face Detector (Haar Cascade)
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 2. Initialize the Facemark Kazemi API
        self.facemark = cv2.face.createFacemarkKazemi()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.facemark.loadModel(model_path)

    def _get_landmarks(self, image):
        """
        Processes image and extracts 68 facial points.
        Memory optimization: Downscaling and Grayscale conversion.
        """
        # A. Memory-Safe Resizing
        max_dim = 800
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            # Use INTER_AREA for better quality when shrinking
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # B. Convert to Grayscale & Equalize
        # We reuse the variable 'gray' to overwrite memory
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # C. Face Detection
        # minNeighbors=5 provides fewer false positives in production
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None
        
        # D. Fit landmarks
        # Kazemi fit requires the image. 
        # Note: landmarks[0] is a list of arrays (one per face)
        ok, landmarks = self.facemark.fit(image, faces)
        
        # Explicitly clear temporary image data from RAM
        del gray
        
        if not ok or len(landmarks) == 0:
            return None
            
        # Return the 68 points for the primary face (index 0)
        return landmarks[0][0]

    def _dist(self, p1, p2):
        """Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def classify_shape(self, image):
        """
        Analyzes geometric ratios of the face to determine shape.
        """
        points = self._get_landmarks(image)
        
        if points is None:
            return "No face detected", 0.0

        try:
            # Feature Extraction (Based on standard 68-point facial landmarks)
            # Jaw Width (Point 4 to 12)
            jaw_w = self._dist(points[4], points[12])
            
            # Cheekbone Width (Point 2 to 14)
            cheek_w = self._dist(points[2], points[14])
            
            # Forehead Width (Point 19 to 24)
            forehead_w = self._dist(points[19], points[24])
            
            # Face Length (Mid-brow 27 to Chin 8)
            face_len = self._dist(points[27], points[8])

            # Ratio: Length / Width
            ratio_lw = face_len / cheek_w
            
            # --- Geometric Classification Logic ---
            if ratio_lw > 1.25:
                shape = "Oblong"
            elif (jaw_w / cheek_w) > 0.9:
                shape = "Square"
            elif (forehead_w / cheek_w) > 0.85 and (jaw_w / cheek_w) < 0.8:
                shape = "Heart"
            elif 0.9 <= ratio_lw <= 1.05:
                shape = "Round"
            else:
                shape = "Oval"

            # Clean up points from memory
            del points
            gc.collect()

            return shape, round(float(ratio_lw), 2)

        except Exception as e:
            print(f"Error in classification math: {e}")
            return "Error", 0.0