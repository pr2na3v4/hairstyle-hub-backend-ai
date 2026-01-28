import cv2
import numpy as np
import os

class FaceShapeDetector:
    def __init__(self, model_path, cascade_path):
        # 1. Load the Face Detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 2. Initialize the Facemark Kazemi API
        # This is compatible with your face_landmark_model.dat
        self.facemark = cv2.face.createFacemarkKazemi()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        self.facemark.loadModel(model_path)

    def _get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) # Normalize lighting for production
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Fit landmarks for the detected faces
        # Note: fit() returns a success flag and the landmarks
        ok, landmarks = self.facemark.fit(image, faces)
        
        if not ok or len(landmarks) == 0:
            return None
            
        # landmarks[0][0] contains the 68 points for the first face
        return landmarks[0][0]

    def _dist(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def classify_shape(self, image):
        points = self._get_landmarks(image)
        if points is None:
            return "No face detected", 0.0

        # indices are 0-based:
        # Jaw Width: points[4] to points[12]
        # Cheekbone Width: points[2] to points[14]
        # Forehead Width: points[19] to points[24]
        # Face Length: points[27] to points[8]
        
        jaw_w = self._dist(points[4], points[12])
        cheek_w = self._dist(points[2], points[14])
        forehead_w = self._dist(points[19], points[24])
        face_len = self._dist(points[27], points[8])

        ratio_lw = face_len / cheek_w
        
        # Classification Logic
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

        return shape, round(float(ratio_lw), 2)

# --- Production Usage ---
# detector = FaceShapeDetector("models/face_landmark_model.dat", "models/haarcascade_frontalface_default.xml")