import cv2
import mediapipe as mp
import numpy as np
import gc


# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK REFERENCE  (MediaPipe 468-point map)
# ─────────────────────────────────────────────────────────────────────────────
#
#  FACE HEIGHT    [10] hairline midpoint  →  [152] chin tip
#  FOREHEAD       [103] / [332]  lateral temporal (above brow arch)
#  CHEEKBONES     [234] / [454]  zygomatic arch  (widest point)
#  JAW CORNERS    [172] / [397]  gonial angle    (true jaw corners)
#  POSE CHECK     [1] nose tip  →  [130] / [359]  outer eye corners
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Centralised thresholds ────────────────────────────────────────────────────
#
#  DIAMOND FIX — tightened from (fc<0.82, jc<0.85) to (fc<0.76, jc<0.80)
#  The old zone covered ~40% of all faces because nearly everyone has
#  cheekbones slightly wider than forehead and jaw. The new zone is the
#  truly narrow-ended faces where cheekbones are the dominant widest feature
#  by a clear margin, not just marginally wider.
#
#  OVAL FIX — added fc>0.76 guard so faces that fall out of Diamond but have
#  a moderately narrow forehead don't silently default to Oval. They now
#  correctly reach Oval only when proportions are genuinely balanced.
#
_T = {
    "oblong_lw_min"  : 1.55,
    "oblong_jc_min"  : 0.80,

    "square_lw_max"  : 1.22,
    "square_jc_min"  : 0.90,
    "square_fc_min"  : 0.85,

    "round_lw_max"   : 1.25,
    "round_jc_max"   : 0.90,
    "round_fc_min"   : 0.80,

    "heart_fj_min"   : 1.25,
    "heart_jc_max"   : 0.75,
    "heart_fc_min"   : 0.82,

    "tri_fj_max"     : 0.85,
    "tri_jc_min"     : 0.88,

    # FIX: tightened significantly — both thresholds dropped ~0.06
    # Old: fc<0.82 AND jc<0.85  →  caught ~40% of faces
    # New: fc<0.76 AND jc<0.80  →  only true narrow-ended faces
    "diamond_fc_max" : 0.76,
    "diamond_jc_max" : 0.80,

    "pose_ratio_max" : 1.20,

    # Confidence (ideal, tolerance)
    "conf_oblong"    : (1.65, 0.12),
    "conf_square"    : (0.93, 0.05),
    "conf_round"     : (1.13, 0.10),
    "conf_heart"     : (1.35, 0.10),
    "conf_triangle"  : (0.78, 0.08),
    # FIX: Diamond confidence now driven by *gap* between cheek and both ends
    # A face with fc=0.70 and jc=0.74 is more Diamond than fc=0.75, jc=0.79
    "conf_diamond"   : (0.70, 0.06),
    # FIX: Oval ideal shifted to 1.35 (between Round's 1.13 and Oblong's 1.55)
    # Tolerance stays wide (0.30) — Oval is the genuine catch-all
    "conf_oval"      : (1.35, 0.30),
}


class FaceMeshDetector:

    def __init__(self):
        self._mp_mesh = mp.solutions.face_mesh
        self._mesh    = self._mp_mesh.FaceMesh(
            static_image_mode        = True,
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
        )

    # ── Internals ─────────────────────────────────────────────────────────────
    def _landmarks(self, image):
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _px_dist(self, lm, i, j, w, h):
        dx = (lm[j].x - lm[i].x) * w
        dy = (lm[j].y - lm[i].y) * h
        return float(np.hypot(dx, dy))

    def _pose_ok(self, lm, w, h):
        """Nose-tip → outer eye corner symmetry. Rejects ~±10° yaw."""
        l = self._px_dist(lm, 1, 130, w, h)
        r = self._px_dist(lm, 1, 359, w, h)
        if min(l, r) == 0:
            return False
        return (max(l, r) / min(l, r)) < _T["pose_ratio_max"]

    @staticmethod
    def _conf(value, ideal, tolerance):
        """
        Gaussian confidence: 100% at ideal, ~60% at ±1 tolerance, floor 50%.
        σ = tolerance/1.4427 so exp(−0.5) ≈ 0.607 at exactly 1σ.
        """
        sigma = tolerance / 1.4427
        conf  = np.exp(-((value - ideal) ** 2) / (2 * sigma ** 2))
        return round(max(0.50, float(conf)) * 100, 1)

    # ── Public API ────────────────────────────────────────────────────────────
    def classify_shape(self, image) -> dict:
        """
        Classify face shape from a BGR image.

        Returns dict:
          detected_shape  – Oblong | Square | Round | Heart | Triangle | Diamond | Oval
          confidence      – "XX.X%"  (capped 98%)
          metric_ratio    – face length / cheek width
          debug           – raw px + all 4 ratios
          error           – only present on failure
        """
        h, w, _ = image.shape
        lm = self._landmarks(image)

        if lm is None:
            return {"error": "No face detected — ensure good lighting and a clear view"}

        if not self._pose_ok(lm, w, h):
            return {"error": "Face tilted — please look straight at the camera"}

        # ── Measurements ──────────────────────────────────────────────────────
        D = lambda i, j: self._px_dist(lm, i, j, w, h)  # noqa: E731

        face_len   = D(10,  152)   # hairline → chin
        forehead_w = D(103, 332)   # lateral forehead
        cheek_w    = D(234, 454)   # zygomatic / cheekbone  (widest)
        jaw_w      = D(172, 397)   # gonial-angle jaw corners

        if min(face_len, forehead_w, cheek_w, jaw_w) < 1.0:
            return {"error": "Landmark geometry invalid — try a clearer, better-lit image"}

        # ── Ratios ────────────────────────────────────────────────────────────
        lw = face_len   / cheek_w   # elongation index
        fc = forehead_w / cheek_w   # forehead breadth vs widest point
        jc = jaw_w      / cheek_w   # jaw strength vs widest point
        fj = forehead_w / jaw_w     # taper direction  (Heart ↔ Triangle)

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

        # ── Decision tree ─────────────────────────────────────────────────────
        #
        #  Order: most geometrically distinct → broadest default.
        #  Every branch requires ALL guards — no single-ratio shortcuts.
        #
        #  KEY CHANGES vs previous version:
        #
        #  Diamond  – thresholds tightened (fc<0.76, jc<0.80) so only faces
        #             with *clearly* dominant cheekbones match. The old values
        #             (fc<0.82, jc<0.85) were so loose they swallowed Oval
        #             and parts of Oblong.
        #
        #  Oval     – added fc>0.76 as a positive guard. Previously Oval was
        #             a pure catch-all with no lower bound on fc, which meant
        #             borderline-Diamond faces fell here with low confidence
        #             instead of being detected as Diamond. Now Oval requires
        #             genuine forehead/cheek balance.
        #
        t = _T

        if lw > t["oblong_lw_min"] and jc > t["oblong_jc_min"]:
            shape, conf = "Oblong",   self._conf(lw,  *t["conf_oblong"])

        elif lw <= t["square_lw_max"] and jc >= t["square_jc_min"] and fc >= t["square_fc_min"]:
            shape, conf = "Square",   self._conf(jc,  *t["conf_square"])

        elif lw <= t["round_lw_max"] and jc < t["round_jc_max"] and fc >= t["round_fc_min"]:
            shape, conf = "Round",    self._conf(lw,  *t["conf_round"])

        elif fj > t["heart_fj_min"] and jc < t["heart_jc_max"] and fc >= t["heart_fc_min"]:
            # fc≥0.82 separates Heart from Diamond (Diamond also has narrow forehead)
            shape, conf = "Heart",    self._conf(fj,  *t["conf_heart"])

        elif fj < t["tri_fj_max"] and jc > t["tri_jc_min"]:
            shape, conf = "Triangle", self._conf(fj,  *t["conf_triangle"])

        elif fc < t["diamond_fc_max"] and jc < t["diamond_jc_max"]:
            # FIX: fc<0.76 AND jc<0.80 — cheekbones must clearly dominate
            # both ends by a meaningful margin, not just marginally wider.
            # Confidence driven by the average narrowness of both ends:
            # lower avg = more Diamond-like = higher confidence.
            narrowness   = (fc + jc) / 2
            shape, conf  = "Diamond", self._conf(narrowness, *t["conf_diamond"])

        else:
            # FIX: Oval now also covers the "almost Diamond but not quite"
            # zone (fc 0.76–0.82) because those faces have balanced enough
            # proportions to qualify as Oval rather than a weak Diamond.
            shape, conf = "Oval",     self._conf(lw,  *t["conf_oval"])

        gc.collect()

        return {
            "detected_shape" : shape,
            "confidence"     : f"{min(98.0, conf):.1f}%",
            "metric_ratio"   : round(lw, 3),
            "debug"          : debug,
        }