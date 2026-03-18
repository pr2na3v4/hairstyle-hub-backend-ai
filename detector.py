import cv2
import mediapipe as mp
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK REFERENCE  (MediaPipe 468-point canonical map)
# ─────────────────────────────────────────────────────────────────────────────
#
#  FACE HEIGHT
#    [10]  top-of-forehead  (nearest point to hairline in the 468-map;
#           unreliable on bald / receding hair, but no better option exists)
#   [152]  chin tip  (menton)
#
#  WIDTHS — measured at their anatomically correct vertical level
#    [67] / [297]  upper forehead width
#                  AUDIT CONSENSUS (3 rounds):
#                  [54]/[284]   → eyelid / brow zone — too low, causes fc to
#                                 read small on every face → Diamond flooding
#                  [103]/[332]  → lower-lateral brow — better but still below
#                                 mid-forehead, under-reads by ~8%
#                  [67]/[297]   → highest reliable bilateral forehead points,
#                                 closest to true hairline width — USE THESE
#
#   [234] / [454]  zygomatic arch / cheekbone
#                  Widest lateral face point. Correct across all audit rounds.
#
#   [132] / [361]  gonial angle / jaw corner  (mandibular ramus near angle)
#                  Multiple audits attempted to restore [172]/[397] claiming
#                  they are "correct gonial angle" — factually wrong.
#                  [172]/[397] are mid-mandible BODY points and under-read
#                  true jaw width by 10-15%.  [132]/[361] stay.
#
#  POSE — bilateral symmetry around the midsagittal plane
#    [6]         glabella  (nose bridge — stable midline, unaffected by gaze)
#   [234]/[454]  cheekbone points  (wide baseline, rigid)
#                Ratio of glabella→left-cheek vs glabella→right-cheek detects
#                yaw.  Original nose-tip→eye-corner was noisy because eye
#                corners shift with gaze and eye-openness.
#
#  PITCH — forward / back head tilt
#    [1]   nose tip
#    [4]   nose base (columella base)
#   [152]  chin tip
#           Perpendicular distance from nose-tip [1] to the chin→nose-base
#           LINE (cross-product formula).
#           Previous version computed axis_x = (nose_base_x + chin_x) / 2
#           which is a midpoint (a dot), not a line distance. Fixed.
#
#  VISIBILITY GATING
#   MediaPipe FaceMesh in static_image_mode=True frequently returns
#   visibility=0.0 for all landmarks — the model only populates visibility
#   reliably in video/stream mode where temporal context is available.
#   The gate therefore checks whether scores are populated at all before
#   enforcing them.  If max(visibility) < 0.1 across key points, static
#   mode is assumed and the gate is skipped entirely.  When scores ARE
#   populated (video mode or high-quality detection), the hard threshold
#   of 0.5 is enforced.  Perplexity's suggestion of dist * min(vi, vj)
#   was rejected — it silently shrinks distances and corrupts all ratios.
#
# ─────────────────────────────────────────────────────────────────────────────


# ── Key landmarks that must all pass visibility check ────────────────────────
_KEY_LANDMARKS = [10, 67, 297, 234, 454, 132, 361, 152, 6, 1, 4]


# ── Thresholds (single source of truth — tune here only) ─────────────────────
_T = {

    # ── Oblong ────────────────────────────────────────────────────────────────
    "oblong_lw_min"    : 1.50,   # face_len / cheek_w — elongation
    "oblong_jc_min"    : 0.78,   # jaw present (not pointy Heart/Diamond)
    "oblong_fj_max"    : 1.30,   # forehead not dramatically wider than jaw
                                  # (separates Oblong from elongated Heart)

    # ── Square ────────────────────────────────────────────────────────────────
    "square_lw_max"    : 1.22,   # compact height
    "square_jc_min"    : 0.90,   # strong wide jaw
    "square_fc_min"    : 0.85,   # wide forehead
    "square_eq_max"    : 0.15,   # |fc − jc| uniformity guard
                                  # Loosened 0.10 → 0.15: camera angle +
                                  # landmark noise produce |fc−jc| up to 0.12
                                  # on genuine square faces.

    # ── Round ─────────────────────────────────────────────────────────────────
    "round_lw_max"     : 1.30,   # compact
    "round_jc_max"     : 0.89,   # softer jaw than Square
    "round_fc_min"     : 0.82,   # wide forehead

    # ── Heart ─────────────────────────────────────────────────────────────────
    "heart_fj_min"     : 1.22,   # forehead clearly wider than jaw
    "heart_jc_max"     : 0.76,   # narrow jaw vs cheeks
    "heart_fc_min"     : 0.82,   # forehead ≥ cheeks (separates from Diamond)
    "heart_lw_max"     : 1.55,   # not excessively elongated

    # ── Triangle (pear) ───────────────────────────────────────────────────────
    "tri_fj_max"       : 0.86,   # jaw clearly wider than forehead
    "tri_jc_min"       : 0.88,   # jaw wide relative to cheeks
    "tri_lw_max"       : 1.55,   # not excessively elongated

    # ── Diamond ───────────────────────────────────────────────────────────────
    "diamond_fc_max"   : 0.81,   # narrow forehead vs cheeks
    "diamond_jc_max"   : 0.80,   # narrow jaw vs cheeks
    "diamond_fj_min"   : 0.83,   # lower fj — symmetric taper
    "diamond_fj_max"   : 1.10,   # upper fj — symmetric taper
                                  # Tightened 1.20 → 1.10: creates a 0.12-wide
                                  # buffer between Diamond (≤1.10) and
                                  # Heart (>1.22). Old 0.02 gap was too narrow.

    # ── Oval ──────────────────────────────────────────────────────────────────
    "oval_lw_min"      : 1.10,   # longer than Square/Round
                                  # Lowered 1.22 → 1.10: a face with lw=1.15
                                  # and balanced proportions is a real face type
                                  # and should not fall to Indeterminate.
    "oval_lw_max"      : 1.55,   # not as elongated as Oblong
    "oval_fc_min"      : 0.72,   # forehead present
    "oval_jc_min"      : 0.70,   # jaw present

    # ── Pose / quality ────────────────────────────────────────────────────────
    "pose_yaw_max"     : 1.18,   # glabella→cheek symmetry ratio (~±12° yaw)
    "pose_pitch_max"   : 0.08,   # perpendicular nose deviation / face_width
                                  # Raised 0.04 → 0.08: natural asymmetry +
                                  # MediaPipe jitter produce 2-5% offsets on
                                  # perfectly straight faces.
    "min_face_frac"    : 0.15,   # face_len must be ≥ 15% of image height
    "vis_threshold"    : 0.50,   # minimum landmark visibility when populated
    "vis_populated_min": 0.10,   # if max(visibility) < this, MediaPipe did not
                                  # populate scores (static_image_mode) → skip gate

    # ── Confidence  (ideal_value, tolerance) ─────────────────────────────────
    # Gaussian:  conf = exp(−(v − ideal)² / (2 · tolerance²))
    # σ = tolerance  →  exp(−0.5) ≈ 60.7% at ±1 tolerance from ideal.
    # Floor: 35% — a 50% floor falsely implies half-certainty on bad fits.
    # All pairs live in _T so every tuneable value is in one place.
    "conf_oblong"      : (1.55, 0.12),   # driven by lw
    "conf_square_jc"   : (0.93, 0.04),   # driven by jc
    "conf_square_fc"   : (0.90, 0.04),   # driven by fc
    "conf_round"       : (1.18, 0.09),   # driven by lw
    "conf_heart"       : (1.35, 0.10),   # driven by fj
    "conf_triangle"    : (0.80, 0.07),   # driven by fj
    "conf_diamond"     : (0.76, 0.06),   # driven by avg(fc, jc)
    "conf_diamond_fj"  : (1.02, 0.10),   # driven by fj — symmetric taper
    "conf_oval"        : (1.35, 0.14),   # driven by lw
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

    # ─────────────────────────────────────────── private helpers ─────────────

    def _landmarks(self, image):
        """BGR image → landmark list, or None if no face detected."""
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

    def _vis_ok(self, lm):
        """
        Visibility gate with static-image-mode awareness.

        MediaPipe FaceMesh returns visibility=0.0 for all landmarks when
        running in static_image_mode=True.  The model only populates
        visibility scores reliably in video/stream mode where it has
        temporal context to estimate occlusion.

        Strategy:
          1. Sample visibility across all key landmarks.
          2. If the maximum score is below vis_populated_min (0.10), the
             model has not populated scores at all — skip the gate entirely
             rather than incorrectly rejecting every static-mode image.
          3. If scores ARE populated (video mode / high-quality detection),
             enforce the hard threshold of vis_threshold (0.50).

        WHY hard rejection and not weighted scaling:
          Perplexity's P0 suggestion of  dist * min(vi, vj)  silently
          returns a shorter distance, e.g. visibility=0.6 → 60% of true
          jaw width.  That corrupts every ratio and produces wrong shape
          labels.  The only correct response to a confirmed occlusion is
          to reject the frame and ask the user to retry.
        """
        scores = [lm[i].visibility for i in _KEY_LANDMARKS]

        # Scores unpopulated → static_image_mode, skip gate
        if max(scores) < _T["vis_populated_min"]:
            return True

        # Scores populated → enforce threshold
        return all(s >= _T["vis_threshold"] for s in scores)

    def _pose_ok(self, lm, w, h):
        """
        Two-axis pose quality check.

        YAW (left-right rotation)
        ─────────────────────────
        Glabella [6] → left cheekbone [234] vs right cheekbone [454].
        Ratio > pose_yaw_max means head is turned too far; cheek and jaw
        widths on the near side will be foreshortened.

        PITCH (forward / back tilt)
        ─────────────────────────────
        Perpendicular distance from nose-tip [1] to the line defined by
        chin [152] → nose-base [4], expressed as a fraction of face width.

        Cross-product perpendicular distance formula:
            Line P1=(x1,y1) → P2=(x2,y2), point P0=(x0,y0):
            d = |(y2−y1)·x0 − (x2−x1)·y0 + x2·y1 − y2·x1|
                ─────────────────────────────────────────────
                          sqrt((y2−y1)² + (x2−x1)²)

        Previous version used axis_x = (nose_base_x + chin_x) / 2,
        which is the mean x of two points — a dot, not a line.
        """
        # ── yaw ──────────────────────────────────────────────────────────────
        l = self._px_dist(lm, 6, 234, w, h)
        r = self._px_dist(lm, 6, 454, w, h)
        if min(l, r) < 1.0:
            return False
        if (max(l, r) / min(l, r)) > _T["pose_yaw_max"]:
            return False

        # ── pitch ─────────────────────────────────────────────────────────────
        x1, y1 = lm[152].x * w, lm[152].y * h   # chin       (P1)
        x2, y2 = lm[4].x   * w, lm[4].y   * h   # nose base  (P2)
        x0, y0 = lm[1].x   * w, lm[1].y   * h   # nose tip   (P0)

        line_len = float(np.hypot(y2 - y1, x2 - x1))
        if line_len < 1.0:
            return False

        perp = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        perp /= line_len

        face_w = abs(lm[234].x - lm[454].x) * w
        if face_w < 1.0:
            return False

        return (perp / face_w) <= _T["pose_pitch_max"]

    @staticmethod
    def _conf(value, ideal, tolerance):
        """
        Gaussian confidence score.

          conf = exp(−(value − ideal)² / (2 · tolerance²))

        At value = ideal ± tolerance  →  exp(−0.5) ≈ 60.7%
        At value = ideal              →  100%
        Floor: 35% to avoid falsely implying certainty on marginal fits.
        """
        conf = np.exp(-((value - ideal) ** 2) / (2 * tolerance ** 2))
        return round(max(0.35, float(conf)) * 100, 1)

    @staticmethod
    def _multi_conf(pairs):
        """
        Composite confidence from multiple (value, ideal, tolerance) triples.

        Geometric mean of individual Gaussian scores — every dimension must
        fit well for the total to be high.  A single outlier pulls the whole
        score down, which is the desired behaviour for shapes that require
        several simultaneous conditions to hold.

        All (ideal, tolerance) pairs are sourced from _T.
        No hardcoded constants in this method.
        """
        scores = [
            float(np.exp(-((v - ideal) ** 2) / (2 * tol ** 2)))
            for v, ideal, tol in pairs
        ]
        combined = float(np.prod(scores) ** (1.0 / len(scores)))
        return round(max(0.35, combined) * 100, 1)

    # ──────────────────────────────────────────────────── public API ──────────

    def classify_shape(self, image: np.ndarray) -> dict:
        """
        Classify the face shape in a BGR image.

        Returns
        -------
        dict with keys:
          detected_shape  – Oblong | Square | Round | Heart |
                            Triangle | Diamond | Oval | Indeterminate
          confidence      – "XX.X%"  (capped at 98% for realism)
          metric_ratio    – face_len / cheek_w  (primary elongation ratio)
          debug           – raw pixel measurements + all four ratios
          error           – present only on failure; all other keys absent

        Shape geometry contracts
        ────────────────────────────────────────────────────────────────────
        Oblong      lw ≥ 1.50,  jaw present,  forehead not >> jaw
        Square      lw ≤ 1.22,  forehead ≈ cheek ≈ jaw  (all within 0.15)
        Round       lw ≤ 1.30,  wide forehead,  soft jaw
        Heart       fj > 1.22,  lw ≤ 1.55
        Triangle    fj < 0.86,  jaw widest point,  lw ≤ 1.55
        Diamond     cheeks dominate;  fj ∈ [0.83, 1.10]
        Oval        lw ∈ [1.10, 1.55],  balanced widths
        Indeterminate  no archetype fits — retry or review image
        """
        h, w, _ = image.shape
        lm = self._landmarks(image)

        if lm is None:
            return {"error": "No face detected — ensure good lighting and a clear, unobstructed view"}

        # ── minimum face size ─────────────────────────────────────────────────
        face_len_raw = abs(lm[152].y - lm[10].y) * h
        if face_len_raw < _T["min_face_frac"] * h:
            return {"error": "Face too small in frame — move closer to the camera"}

        # ── visibility gate ───────────────────────────────────────────────────
        if not self._vis_ok(lm):
            return {"error": "Key landmarks obscured — remove glasses or move hair away from face"}

        # ── pose gate ─────────────────────────────────────────────────────────
        if not self._pose_ok(lm, w, h):
            return {"error": "Face not straight — look directly at the camera (no tilt or turn)"}

        # ── 1. pixel measurements ─────────────────────────────────────────────
        D = lambda i, j: self._px_dist(lm, i, j, w, h)  # noqa: E731

        face_len   = D(10,  152)  # top-of-forehead → chin tip
        forehead_w = D(67,  297)  # upper forehead — audit-validated best option
        cheek_w    = D(234, 454)  # zygomatic arch — widest face point
        jaw_w      = D(132, 361)  # gonial angle / jaw corner — audit-validated

        if min(face_len, forehead_w, cheek_w, jaw_w) < 1.0:
            return {"error": "Landmark geometry invalid — try a clearer, better-lit image"}

        # ── 2. ratios ─────────────────────────────────────────────────────────
        #
        #  lw   face_Len / cheek_Width   → elongation index
        #  fc   Forehead / Cheek         → forehead breadth vs widest point
        #  jc   Jaw / Cheek              → jaw strength vs widest point
        #  fj   Forehead / Jaw           → taper direction  (Heart ↔ Triangle)
        #
        lw = face_len   / cheek_w
        fc = forehead_w / cheek_w
        jc = jaw_w      / cheek_w
        fj = forehead_w / jaw_w

        debug = {
            "face_len_px"  : round(face_len,   1),
            "forehead_px"  : round(forehead_w, 1),
            "cheek_px"     : round(cheek_w,    1),
            "jaw_px"       : round(jaw_w,      1),
            "lw_ratio"     : round(lw, 3),
            "fc_ratio"     : round(fc, 3),
            "jc_ratio"     : round(jc, 3),
            "fj_ratio"     : round(fj, 3),
        }

        # ── 3. decision tree ──────────────────────────────────────────────────
        #
        #  Order: most geometrically constrained first, broadest last.
        #  Every branch requires ALL guards — no shape identified by one
        #  ratio alone.
        #
        t = _T

        # ── Oblong ────────────────────────────────────────────────────────────
        # G1: very elongated face
        # G2: jaw present — not pointy Heart/Diamond
        # G3: forehead not dramatically wider than jaw — not elongated Heart
        if (lw  >  t["oblong_lw_min"]
                and jc >  t["oblong_jc_min"]
                and fj <  t["oblong_fj_max"]):
            shape = "Oblong"
            conf  = self._conf(lw, *t["conf_oblong"])

        # ── Square ────────────────────────────────────────────────────────────
        # G1: compact face
        # G2: strong wide jaw
        # G3: wide forehead
        # G4: all three rows nearly equal width
        elif (lw  <= t["square_lw_max"]
                and jc >= t["square_jc_min"]
                and fc >= t["square_fc_min"]
                and abs(fc - jc) <= t["square_eq_max"]):
            shape = "Square"
            conf  = self._multi_conf([
                (jc, *t["conf_square_jc"]),
                (fc, *t["conf_square_fc"]),
            ])

        # ── Round ─────────────────────────────────────────────────────────────
        # G1: compact face
        # G2: wide forehead
        # G3: soft jaw (less angular than Square)
        elif (lw  <= t["round_lw_max"]
                and fc >= t["round_fc_min"]
                and jc <  t["round_jc_max"]):
            shape = "Round"
            conf  = self._conf(lw, *t["conf_round"])

        # ── Heart ─────────────────────────────────────────────────────────────
        # G1: forehead clearly wider than jaw
        # G2: narrow jaw vs cheeks
        # G3: forehead ≥ cheeks (separates Heart from Diamond)
        # G4: not excessively elongated
        elif (fj >  t["heart_fj_min"]
                and jc <  t["heart_jc_max"]
                and fc >= t["heart_fc_min"]
                and lw <= t["heart_lw_max"]):
            shape = "Heart"
            conf  = self._conf(fj, *t["conf_heart"])

        # ── Triangle (pear) ───────────────────────────────────────────────────
        # G1: jaw wider than forehead (inverse of Heart)
        # G2: jaw wide relative to cheeks
        # G3: jaw is the absolute widest point on the face
        # G4: not excessively elongated
        elif (fj <  t["tri_fj_max"]
                and jc >  t["tri_jc_min"]
                and jc >  fc
                and lw <= t["tri_lw_max"]):
            shape = "Triangle"
            conf  = self._conf(fj, *t["conf_triangle"])

        # ── Diamond ───────────────────────────────────────────────────────────
        # G1: narrow forehead vs cheeks
        # G2: narrow jaw vs cheeks
        # G3: forehead ≈ jaw — symmetric taper; fj ∈ [0.83, 1.10]
        #     asymmetric tapers belong in Heart (fj > 1.22) or Triangle (fj < 0.86)
        elif (fc  <  t["diamond_fc_max"]
                and jc <  t["diamond_jc_max"]
                and t["diamond_fj_min"] <= fj <= t["diamond_fj_max"]):
            shape      = "Diamond"
            narrowness = (fc + jc) / 2.0
            conf       = self._multi_conf([
                (narrowness, *t["conf_diamond"]),
                (fj,         *t["conf_diamond_fj"]),
            ])

        # ── Oval ──────────────────────────────────────────────────────────────
        # NOT a catch-all — requires explicit range guards.
        # G1: gentle elongation (longer than Square/Round)
        # G2: not as elongated as Oblong
        # G3 & G4: balanced proportions — neither width severely narrow
        elif (t["oval_lw_min"] <= lw <= t["oval_lw_max"]
                and fc >= t["oval_fc_min"]
                and jc >= t["oval_jc_min"]):
            shape = "Oval"
            conf  = self._conf(lw, *t["conf_oval"])

        # ── Indeterminate ─────────────────────────────────────────────────────
        # No archetype fits cleanly. Returning this rather than forcing Oval
        # preserves honesty and signals the caller to retry or review.
        else:
            return {
                "detected_shape" : "Indeterminate",
                "confidence"     : "N/A",
                "metric_ratio"   : round(lw, 3),
                "debug"          : debug,
                "note"           : (
                    "Face proportions do not fit any standard archetype. "
                    "This may indicate an unusual ratio combination, a partially "
                    "obscured face, or an image quality issue."
                ),
            }

        # ── 4. return ─────────────────────────────────────────────────────────
        return {
            "detected_shape" : shape,
            "confidence"     : f"{min(98.0, conf):.1f}%",
            "metric_ratio"   : round(lw, 3),
            "debug"          : debug,
        }
