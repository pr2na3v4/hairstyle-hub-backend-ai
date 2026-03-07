import cv2
import mediapipe as mp
import numpy as np
import gc

class FaceMeshDetector:
    def __init__(self):
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
        return np.sqrt(((p2.x - p1.x) * w)**2 + ((p2.y - p1.y) * h)**2)

    def validate_pose(self, landmarks, w, h):
        """Ensures the user is looking straight to maintain measurement accuracy."""
        left_dist = self._dist(landmarks[1], landmarks[33], w, h)
        right_dist = self._dist(landmarks[1], landmarks[263], w, h)
        ratio = max(left_dist, right_dist) / min(left_dist, right_dist)
        return ratio < 1.25

    def calculate_confidence(self, value, target, margin=0.15):
        """Calculates confidence based on proximity to shape thresholds."""
        diff = abs(value - target)
        conf = max(0.5, 1.0 - (diff / margin))
        return round(conf * 100, 1)

    def classify_shape(self, image):
        h, w, _ = image.shape
        landmarks = self._get_landmarks(image)
        
        if not landmarks:
            return None

        if not self.validate_pose(landmarks, w, h):
            return {"error": "Face tilted - Please look straight"}

        # --- EXTRACT GEOMETRY ---
        face_len = self._dist(landmarks[10], landmarks[152], w, h)
        forehead_w = self._dist(landmarks[103], landmarks[332], w, h)
        cheek_w = self._dist(landmarks[234], landmarks[454], w, h)
        jaw_w = self._dist(landmarks[132], landmarks[361], w, h)

        # --- RATIO CALCULATION ---
        ratio_lw = face_len / cheek_w   # Length to Width
        ratio_fj = forehead_w / jaw_w   # Forehead to Jaw
        ratio_jc = jaw_w / cheek_w      # Jaw to Cheek

        # --- REFINED DECISION TREE WITH CONFIDENCE ---
        shape = "Oval"
        conf_score = 85.0 # Baseline for Oval

        if ratio_lw > 1.45:
            shape = "Oblong"
            conf_score = self.calculate_confidence(ratio_lw, 1.55)
        elif ratio_lw < 1.15:
            if ratio_jc > 0.88:
                shape = "Square"
                conf_score = self.calculate_confidence(ratio_jc, 0.95)
            else:
                shape = "Round"
                conf_score = self.calculate_confidence(ratio_jc, 0.75)
        elif ratio_fj > 1.25:
            shape = "Heart"
            conf_score = self.calculate_confidence(ratio_fj, 1.35)
        elif forehead_w < cheek_w and jaw_w < cheek_w:
            # Diamond refinement: Cheekbones must be the clear widest point
            shape = "Diamond"
            conf_score = self.calculate_confidence(ratio_jc, 0.75)
        else:
            shape = "Oval"
            conf_score = 90.0 if 1.2 < ratio_lw < 1.4 else 75.0

        # Memory Cleanup
        del landmarks
        gc.collect()

        return {
            "detected_shape": shape,
            "confidence": f"{conf_score}%",
            "metric_ratio": round(float(ratio_lw), 2),
            "debug": {
                "forehead_px": round(forehead_w, 1),
                "cheek_px": round(cheek_w, 1),
                "jaw_px": round(jaw_w, 1),
                "f_to_j_ratio": round(ratio_fj, 2),
                "j_to_c_ratio": round(ratio_jc, 2)
            }
        }