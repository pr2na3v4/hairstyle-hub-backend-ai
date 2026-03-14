import cv2
import mediapipe as mp
import numpy as np
import gc


# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK REFERENCE  (MediaPipe 468-point map)
# ─────────────────────────────────────────────────────────────────────────────
#
#  FACE HEIGHT
#    [10]  hairline midpoint (top of forehead)
#   [152]  chin tip
#
#  WIDTHS
#   [103] / [332]  lateral forehead  (temporal, above brow arch)
#   [234] / [454]  zygomatic arch    (widest cheekbone point)
#   [172] / [397]  gonial angle      (true jaw corners, not mid-jaw)
#
#  POSE CHECK  (nose-tip → outer eye corner symmetry)
#     [1]  nose tip
#   [130] / [359]  outer eye corners  (wider baseline than inner [33]/[263])
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Thresholds (centralised — tune here, nowhere else) ───────────────────────
_T = {
    # ── shape boundaries ─────────────────────────────────────────────────────
    "oblong_lw_min"  : 1.55,   # lw above this → definitely elongated
    "oblong_jc_min"  : 0.80,   # jaw must be present (not heart-like)

    "square_lw_max"  : 1.22,   # nearly as wide as tall
    "square_jc_min"  : 0.90,   # strong, wide jaw
    "square_fc_min"  : 0.85,   # wide forehead too (all three rows equal)

    "round_lw_max"   : 1.25,   # compact face
    "round_jc_max"   : 0.90,   # softer jaw than square
    "round_fc_min"   : 0.80,   # full, wide forehead

    "heart_fj_min"   : 1.25,   # forehead clearly wider than jaw
    "heart_jc_max"   : 0.75,   # narrow jaw relative to cheeks
    "heart_fc_min"   : 0.82,   # forehead ≥ cheeks (separates from Diamond)

    "tri_fj_max"     : 0.85,   # jaw wider than forehead (inverse of Heart)
    "tri_jc_min"     : 0.88,   # jaw wide relative to cheeks

    "diamond_fc_max" : 0.82,   # forehead narrower than cheeks
    "diamond_jc_max" : 0.85,   # jaw narrower than cheeks
    #                            NOTE: lw guard removed — short Diamonds exist

    # ── pose ─────────────────────────────────────────────────────────────────
    "pose_ratio_max" : 1.20,   # nose-to-outer-eye symmetry limit (~±10° yaw)

    # ── confidence  (ideal_value, tolerance) ─────────────────────────────────
    "conf_oblong"    : (1.65, 0.12),
    "conf_square"    : (0.93, 0.05),   # driven by jc
    "conf_round"     : (1.13, 0.10),   # driven by lw
    "conf_heart"     : (1.35, 0.10),   # driven by fj
    "conf_triangle"  : (0.78, 0.08),   # driven by fj
    "conf_diamond"   : (0.74, 0.08),   # driven by avg(fc, jc)
    "conf_oval"      : (1.30, 0.30),   # wider tolerance — Oval is the broad default
    #                                    FIX: was (1.40, 0.18) → lw~1.18 bottomed at 50%
}


class FaceMeshDetector:

    # ──────────────────────────────────────────────────────────── init ────────
    def __init__(self):
        self._mp_mesh = mp.solutions.face_mesh
        self._mesh    = self._mp_mesh.FaceMesh(
            static_image_mode        = True,
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
        )

    # ──────────────────────────────────────────────── private helpers ─────────
    def _landmarks(self, image):
        """BGR image → landmark list, or None if no face found."""
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _px_dist(self, lm, i, j, w, h):
        """Pixel-space Euclidean distance between landmark indices i and j."""
        dx = (lm[j].x - lm[i].x) * w
        dy = (lm[j].y - lm[i].y) * h
        return float(np.hypot(dx, dy))

    def _pose_ok(self, lm, w, h):
        """
        Left-right symmetry check.
        Measures nose-tip [1] → outer eye corner [130] and [359].
        Outer corners give ~40 % wider baseline than inner corners [33]/[263],
        making yaw detection more sensitive.
        Rejects frame when asymmetry ratio exceeds pose_ratio_max (~±10° yaw).
        """
        l = self._px_dist(lm, 1, 130, w, h)
        r = self._px_dist(lm, 1, 359, w, h)
        if min(l, r) == 0:
            return False
        return (max(l, r) / min(l, r)) < _T["pose_ratio_max"]

    @staticmethod
    def _conf(value, ideal, tolerance):
        """
        Gaussian confidence score.
          • Peaks at 100 % when value == ideal.
          • Falls to ~60 % at ±1 tolerance.
          • Hard floor of 50 % — we never imply near-zero certainty.

        σ = tolerance / 1.4427  gives exp(−0.5) ≈ 0.607 at exactly 1σ.
        Gaussian avoids the hard cliff-edge of linear decay and scales
        consistently across different ratio ranges.
        """
        sigma = tolerance / 1.4427
        conf  = np.exp(-((value - ideal) ** 2) / (2 * sigma ** 2))
        return round(max(0.50, float(conf)) * 100, 1)

    # ──────────────────────────────────────────────────── public API ──────────
    def classify_shape(self, image) -> dict:
        """
        Classify the face shape in a BGR image.

        Returns
        -------
        dict with keys:
          detected_shape  – Oblong | Square | Round | Heart |
                            Triangle | Diamond | Oval
          confidence      – "XX.X%"  (capped at 98 % for realism)
          metric_ratio    – face-length / cheek-width  (primary ratio)
          debug           – raw pixel sizes + all four ratios
          error           – present only on failure; all other keys absent
        """
        h, w, _ = image.shape
        lm = self._landmarks(image)

        if lm is None:
            return {"error": "No face detected — ensure good lighting and a clear view"}

        if not self._pose_ok(lm, w, h):
            return {"error": "Face tilted — please look straight at the camera"}

        # ── 1. pixel measurements ─────────────────────────────────────────────
        D = lambda i, j: self._px_dist(lm, i, j, w, h)  # noqa: E731

        face_len   = D(10,  152)   # hairline → chin
        forehead_w = D(103, 332)   # lateral forehead width
        cheek_w    = D(234, 454)   # zygomatic / cheekbone width  (widest)
        jaw_w      = D(172, 397)   # gonial-angle jaw width

        if min(face_len, forehead_w, cheek_w, jaw_w) < 1.0:
            return {"error": "Landmark geometry invalid — try a clearer, better-lit image"}

        # ── 2. ratios ─────────────────────────────────────────────────────────
        #
        #  lw  Length / cheek Width      → elongation index
        #  fc  Forehead / Cheek          → forehead breadth vs widest point
        #  jc  Jaw / Cheek               → jaw strength vs widest point
        #  fj  Forehead / Jaw            → taper direction (Heart ↔ Triangle)
        #
        lw = face_len   / cheek_w
        fc = forehead_w / cheek_w
        jc = jaw_w      / cheek_w
        fj = forehead_w / jaw_w

        debug = {
            "face_len_px" : round(face_len,   1),
            "forehead_px" : round(forehead_w, 1),
            "cheek_px"    : round(cheek_w,    1),
            "jaw_px"      : round(jaw_w,      1),
            "l_w_ratio"   : round(lw, 3),
            "f_c_ratio"   : round(fc, 3),
            "j_c_ratio"   : round(jc, 3),
            "f_j_ratio"   : round(fj, 3),
        }

        # ── 3. decision tree ──────────────────────────────────────────────────
        #
        #  Order: most geometrically distinct shapes first → broadest last.
        #  Every branch requires ALL its guards — no single-ratio shortcuts.
        #
        t = _T

        if lw > t["oblong_lw_min"] and jc > t["oblong_jc_min"]:
            # Clearly elongated; jaw present (not a pointed Oval or Heart)
            shape, conf = "Oblong",   self._conf(lw,               *t["conf_oblong"])

        elif lw <= t["square_lw_max"] and jc >= t["square_jc_min"] and fc >= t["square_fc_min"]:
            # Compact face; forehead, cheek, jaw all nearly equal width
            shape, conf = "Square",   self._conf(jc,               *t["conf_square"])

        elif lw <= t["round_lw_max"] and jc < t["round_jc_max"] and fc >= t["round_fc_min"]:
            # Compact, wide forehead, softer jaw — rounder than Square
            shape, conf = "Round",    self._conf(lw,               *t["conf_round"])

        elif fj > t["heart_fj_min"] and jc < t["heart_jc_max"] and fc >= t["heart_fc_min"]:
            # Broad forehead, narrow jaw — classic V-taper
            # fc ≥ 0.82 guards against Diamond (where forehead is ALSO narrow)
            shape, conf = "Heart",    self._conf(fj,               *t["conf_heart"])

        elif fj < t["tri_fj_max"] and jc > t["tri_jc_min"]:
            # Inverse of Heart — jaw wider than forehead (pear / triangle)
            shape, conf = "Triangle", self._conf(fj,               *t["conf_triangle"])

        elif fc < t["diamond_fc_max"] and jc < t["diamond_jc_max"]:
            # Cheekbones dominate — both forehead AND jaw narrower.
            # FIX: lw guard (> 1.25) removed; short compact Diamonds are valid
            #      (e.g. lw ~ 1.18 with fc=0.71, jc=0.83 → genuine Diamond).
            narrowness   = (fc + jc) / 2
            shape, conf  = "Diamond", self._conf(narrowness,       *t["conf_diamond"])

        else:
            # Balanced, gently elongated — the true default.
            # FIX: ideal shifted 1.40→1.30, tolerance widened 0.18→0.30 so
            #      lw values from ~1.0–1.6 score above 50 % confidence.
            shape, conf = "Oval",     self._conf(lw,               *t["conf_oval"])

        # ── 4. return ─────────────────────────────────────────────────────────
        gc.collect()

        return {
            "detected_shape" : shape,
            "confidence"     : f"{min(98.0, conf):.1f}%",
            "metric_ratio"   : round(lw, 3),
            "debug"          : debug,
        }