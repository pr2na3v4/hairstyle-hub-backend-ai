import cv2
import mediapipe as mp
import numpy as np
import gc


# =============================================================================
#  MEDIAPIPE 468-LANDMARK MAP  (key points used in this detector)
# =============================================================================
#
#   FACE HEIGHT
#     [10]  → Top of forehead (hairline midpoint)
#    [152]  → Chin tip (bottom of face)
#
#   FOREHEAD WIDTH  (true lateral forehead, above eyebrows)
#    [103]  → Left  temporal / lateral forehead   ← KEPT (acceptable)
#    [332]  → Right temporal / lateral forehead   ← KEPT (acceptable)
#   NOTE: replaced inner-eyebrow points with outer temporal points for
#         a wider, more stable forehead reading.
#
#   CHEEKBONE WIDTH  (widest zygomatic arch — most reliable width)
#    [234]  → Left  zygomatic (cheekbone)
#    [454]  → Right zygomatic (cheekbone)
#
#   JAW WIDTH  (true gonial angle / jaw-corner points)
#     OLD: [132] / [361]  — mid-jaw, not jaw angle
#     NEW: [172] / [397]  — gonial angle (jaw corners) — more accurate
#
#   JAW TIP
#    [152]  → Chin (shared with face height)
#
#   POSE / SYMMETRY CHECK  (outer eye corners for left-right balance)
#     OLD: [33]  / [263]  — inner eye corners
#     NEW: [130] / [359]  — outer eye corners → wider baseline = better tilt signal
#
#   NOSE TIP (for pose reference point)
#     [1]   → Nose tip
#
# =============================================================================


class FaceMeshDetector:

    # ------------------------------------------------------------------ init
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    # -------------------------------------------------------- landmark fetch
    def _get_landmarks(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    # ------------------------------------------------------- Euclidean dist
    def _dist(self, p1, p2, w, h):
        """Pixel-space Euclidean distance between two normalised landmarks."""
        dx = (p2.x - p1.x) * w
        dy = (p2.y - p1.y) * h
        return float(np.sqrt(dx * dx + dy * dy))

    # --------------------------------------------------------- pose check
    def _validate_pose(self, lm, w, h):
        """
        Symmetry check: distance from nose tip to each outer eye corner.
        If the face is rotated left/right these distances diverge.
        Outer corners [130] / [359] give a wider, more sensitive baseline
        than the old inner-corner pair [33] / [263].

        Threshold 1.20  →  allows ~±10° yaw before rejection.
        """
        left_dist  = self._dist(lm[1], lm[130], w, h)
        right_dist = self._dist(lm[1], lm[359], w, h)
        if min(left_dist, right_dist) == 0:
            return False
        ratio = max(left_dist, right_dist) / min(left_dist, right_dist)
        return ratio < 1.20

    # ------------------------------------------------------- confidence calc
    @staticmethod
    def _confidence(value, ideal, tolerance):
        """
        Gaussian-style confidence that peaks at 1.0 when value == ideal and
        falls to ~0.50 at ±1 tolerance from the ideal.

        Using a Gaussian (exp) instead of linear avoids the hard cliff-edge
        behaviour of the old approach and scales naturally across different
        ratio ranges.
        """
        diff = abs(value - ideal)
        sigma = tolerance / 1.4427          # tune: 1σ ≈ tolerance
        conf  = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        return round(max(0.50, conf) * 100, 1)

    # ====================================================== classify_shape ==
    def classify_shape(self, image):
        """
        Returns a dict with:
          detected_shape  – one of 7 shape labels
          confidence      – string "XX.X%"
          metric_ratio    – primary ratio used for classification
          debug           – all four raw ratios for inspection
        """
        h, w, _ = image.shape
        lm = self._get_landmarks(image)

        if lm is None:
            return {"error": "No face detected"}

        if not self._validate_pose(lm, w, h):
            return {"error": "Face tilted – please look straight at the camera"}

        # ----------------------------------------------------------------
        # STEP 1  –  MEASURE KEY DIMENSIONS  (all in pixels)
        # ----------------------------------------------------------------
        face_len    = self._dist(lm[10],  lm[152], w, h)   # hairline → chin
        forehead_w  = self._dist(lm[103], lm[332], w, h)   # lateral forehead
        cheek_w     = self._dist(lm[234], lm[454], w, h)   # zygomatic arch
        jaw_w       = self._dist(lm[172], lm[397], w, h)   # gonial angle corners

        # Safety guard — avoid division by zero on degenerate detections
        if cheek_w < 1 or jaw_w < 1 or forehead_w < 1:
            return {"error": "Landmark geometry is invalid – try a clearer image"}

        # ----------------------------------------------------------------
        # STEP 2  –  DERIVED RATIOS
        # ----------------------------------------------------------------
        #  ratio_lw : face Length / cheek Width
        #             Tells us how elongated the face is.
        #             Oval ≈ 1.3–1.5 | Oblong > 1.5 | Round/Square < 1.25
        #
        #  ratio_fc : Forehead / Cheek
        #             < 0.82 → forehead narrower than cheekbones (Diamond)
        #             > 0.92 → forehead as wide as cheeks (Square / Heart)
        #
        #  ratio_jc : Jaw / Cheek
        #             > 0.88 → jaw nearly as wide as cheeks (Square / Oblong)
        #             < 0.75 → jaw notably narrower than cheeks (Heart / Diamond)
        #
        #  ratio_fj : Forehead / Jaw
        #             > 1.20 → forehead wider than jaw (Heart)
        #             < 0.85 → jaw wider than forehead (Triangle / Pear)
        # ----------------------------------------------------------------
        ratio_lw = face_len   / cheek_w
        ratio_fc = forehead_w / cheek_w
        ratio_jc = jaw_w      / cheek_w
        ratio_fj = forehead_w / jaw_w

        debug = {
            "face_len_px"  : round(face_len,   1),
            "forehead_px"  : round(forehead_w, 1),
            "cheek_px"     : round(cheek_w,    1),
            "jaw_px"       : round(jaw_w,      1),
            "l_w_ratio"    : round(ratio_lw,   3),
            "f_c_ratio"    : round(ratio_fc,   3),
            "j_c_ratio"    : round(ratio_jc,   3),
            "f_j_ratio"    : round(ratio_fj,   3),
        }

        # ----------------------------------------------------------------
        # STEP 3  –  STRICT DECISION TREE
        #
        #  Priority order matters — each branch is evaluated only if all
        #  previous branches were False.  Branches are ordered from the
        #  most geometrically distinct shapes to the most "default" one.
        # ----------------------------------------------------------------

        # --- 1. OBLONG ---------------------------------------------------
        # Clearly elongated face; jaw and forehead both relatively wide
        # so it isn't just a pointed oval.
        #   Primary signal : ratio_lw > 1.55  (strict elongation)
        #   Secondary guard: ratio_jc > 0.80  (jaw is present, not heart-shaped)
        if ratio_lw > 1.55 and ratio_jc > 0.80:
            shape = "Oblong"
            conf  = self._confidence(ratio_lw, 1.65, 0.12)

        # --- 2. SQUARE ---------------------------------------------------
        # Face is nearly as wide as it is long AND the jaw is strong.
        #   ratio_lw ≤ 1.22 → width close to length (boxy proportions)
        #   ratio_jc ≥ 0.90 → jaw nearly as wide as cheekbones
        #   ratio_fc ≥ 0.85 → forehead also wide (all three rows similar)
        elif ratio_lw <= 1.22 and ratio_jc >= 0.90 and ratio_fc >= 0.85:
            shape = "Square"
            conf  = self._confidence(ratio_jc, 0.93, 0.05)

        # --- 3. ROUND ----------------------------------------------------
        # Face is nearly as wide as it is long BUT jaw tapers more than Square.
        #   ratio_lw ≤ 1.25 → still compact
        #   ratio_jc < 0.90 → softer, rounded jaw (distinguishes from Square)
        #   ratio_fc ≥ 0.80 → forehead width similar to cheeks (full round look)
        elif ratio_lw <= 1.25 and ratio_jc < 0.90 and ratio_fc >= 0.80:
            shape = "Round"
            conf  = self._confidence(ratio_lw, 1.13, 0.10)

        # --- 4. HEART ----------------------------------------------------
        # Wide forehead that tapers steeply to a narrow, pointed jaw.
        #   ratio_fj > 1.25 → forehead clearly wider than jaw
        #   ratio_jc < 0.75 → jaw narrow relative to cheekbones
        #   ratio_fc ≥ 0.82 → forehead is at least as wide as cheekbones
        #                      (distinguishes from Diamond where forehead is also narrow)
        elif ratio_fj > 1.25 and ratio_jc < 0.75 and ratio_fc >= 0.82:
            shape = "Heart"
            conf  = self._confidence(ratio_fj, 1.35, 0.10)

        # --- 5. TRIANGLE (Pear) ------------------------------------------
        # Wide jaw that is broader than the forehead — opposite of Heart.
        #   ratio_fj < 0.85 → jaw notably wider than forehead
        #   ratio_jc > 0.88 → jaw width relative to cheekbones is large
        elif ratio_fj < 0.85 and ratio_jc > 0.88:
            shape = "Triangle"
            conf  = self._confidence(ratio_fj, 0.78, 0.08)

        # --- 6. DIAMOND --------------------------------------------------
        # Cheekbones are the WIDEST feature; both forehead and jaw are
        # noticeably narrower.
        #   ratio_fc < 0.82 → forehead narrower than cheekbones
        #   ratio_jc < 0.82 → jaw narrower than cheekbones
        #   ratio_lw > 1.25 → some elongation (diamond isn't as short as round)
        #   ratio_fj: forehead and jaw may be similar size (not heart, not triangle)
        elif ratio_fc < 0.82 and ratio_jc < 0.82 and ratio_lw > 1.25:
            shape = "Diamond"
            # Confidence driven by how symmetrically narrow both ends are
            narrowness = (ratio_fc + ratio_jc) / 2
            conf  = self._confidence(narrowness, 0.74, 0.08)

        # --- 7. OVAL (default) -------------------------------------------
        # Gently elongated, balanced proportions — the catch-all for faces
        # that don't fit a more extreme shape.
        #   Ideal LW ≈ 1.35–1.45; forehead slightly narrower than cheeks;
        #   jaw gently tapers.
        else:
            shape = "Oval"
            conf  = self._confidence(ratio_lw, 1.40, 0.18)

        # ----------------------------------------------------------------
        # STEP 4  –  RETURN
        # ----------------------------------------------------------------
        gc.collect()

        return {
            "detected_shape" : shape,
            "confidence"     : f"{min(98.0, conf)}%",
            "metric_ratio"   : round(ratio_lw, 3),
            "debug"          : debug,
        }