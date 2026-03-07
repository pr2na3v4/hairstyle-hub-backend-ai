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
        points = self._get_landmarks(image)
        if points is None:
            return "No face detected", 0.0

        try:
            def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

            # 1. MEASUREMENTS
            # Jaw Width: Point 4 to 12
            jaw_w = dist(points[4], points[12])
            # Cheekbone Width: Point 2 to 14
            cheek_w = dist(points[2], points[14])
            # Forehead Width: Point 19 to 24 (Approximate brow width)
            forehead_w = dist(points[19], points[24])
            # Face Length: Point 27 (top of nose) to 8 (bottom of chin)
            # Note: We multiply by 1.3 to estimate full height from hairline
            face_len = dist(points[27], points[8]) * 1.3 

            # 2. RATIOS (The Secret Sauce)
            ratio_lw = face_len / cheek_w
            jaw_to_cheek = jaw_w / cheek_w
            forehead_to_cheek = forehead_w / cheek_w

            # 3. REFINED CLASSIFICATION
            # Oblong: Face is significantly longer than it is wide
            if ratio_lw > 1.4:
                shape = "Oblong"
            # Square: Wide jaw, jaw width is almost equal to cheek width
            elif jaw_to_cheek > 0.88:
                shape = "Square"
            # Heart: Forehead is wider than jaw, chin is pointed
            elif forehead_to_cheek > 0.8 and jaw_to_cheek < 0.75:
                shape = "Heart"
            # Round: Length and width are nearly equal, jaw is softer
            elif 0.9 <= ratio_lw <= 1.1 and jaw_to_cheek < 0.85:
                shape = "Round"
            # Oval: The balanced middle ground
            else:
                shape = "Oval"

            # Clean up points from memory
            del points
            gc.collect()

            return shape, round(float(ratio_lw), 2)

        except Exception as e:
            print(f"Error in classification math: {e}")
            return "Error", 0.0