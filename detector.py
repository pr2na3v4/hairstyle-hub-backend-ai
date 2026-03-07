import cv2
import mediapipe as mp
import numpy as np
import gc

class FaceMeshDetector:
    def __init__(self):
        # Initialize MediaPipe with settings for low-memory environments
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def _get_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _dist(self, p1, p2, w, h):
        # Normalized coordinates to pixel distance
        return np.sqrt(((p2.x - p1.x) * w)**2 + ((p2.y - p1.y) * h)**2)

    def validate_pose(self, landmarks, w, h):
        """Phase 1 Gatekeeper: Rejects head tilts to ensure accuracy."""
        # Calculate distance from nose tip (1) to left eye (33) and right eye (263)
        left_dist = self._dist(landmarks[1], landmarks[33], w, h)
        right_dist = self._dist(landmarks[1], landmarks[263], w, h)
        
        # If one side is 25% larger than the other, the face is turned
        ratio = max(left_dist, right_dist) / min(left_dist, right_dist)
        return ratio < 1.25

    def classify_shape(self, image):
        h, w, _ = image.shape
        landmarks = self._get_landmarks(image)
        
        if not landmarks:
            return "No face detected", 0.0

        if not self.validate_pose(landmarks, w, h):
            return "Face tilted - Please look straight", 0.0

        # --- EXTRACT GEOMETRY ---
        # Vertical: Hairline(10) to Chin(152)
        face_len = self._dist(landmarks[10], landmarks[152], w, h)
        # Horizontal: Left Temple(103) to Right Temple(332)
        forehead_w = self._dist(landmarks[103], landmarks[332], w, h)
        # Horizontal: Left Cheek(234) to Right Cheek(454)
        cheek_w = self._dist(landmarks[234], landmarks[454], w, h)
        # Horizontal: Left Jaw(132) to Right Jaw(361)
        jaw_w = self._dist(landmarks[132], landmarks[361], w, h)

        # --- RATIO CALCULATION ---
        ratio_lw = face_len / cheek_w
        ratio_fj = forehead_w / jaw_w
        ratio_jc = jaw_w / cheek_w

        # --- 6-SHAPE DECISION TREE ---
       # --- 6-SHAPE DECISION TREE ---
        if ratio_lw > 1.45:
            shape = "Oblong"
        elif ratio_lw < 1.15:
            shape = "Square" if ratio_jc > 0.88 else "Round"
        elif ratio_fj > 1.25:
            shape = "Heart"
        elif forehead_w < cheek_w and jaw_w < cheek_w:
            shape = "Diamond"
        else:
            shape = "Oval"

        # Explicit RAM cleanup
        del landmarks
        gc.collect()

        # ADDED: detailed_metrics for your console logs
        return {
            "shape": shape,
            "ratio": round(float(ratio_lw), 2),
            "debug": {
                "forehead_px": round(forehead_w, 1),
                "cheek_px": round(cheek_w, 1),
                "jaw_px": round(jaw_w, 1),
                "f_to_j_ratio": round(ratio_fj, 2)
            }
        }
    