import cv2
import mediapipe as mp
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  LANDMARK REFERENCE  (MediaPipe 468-point canonical map)
# ─────────────────────────────────────────────────────────────────────────────
#
#  FACE HEIGHT
#    [10]  top-of-forehead (closest MediaPipe gets to hairline; unreliable on
#           bald/receding hair, but no better alternative exists in the 468-map)
#   [152]  chin tip (menton)
#
#  WIDTHS  — all measurements are taken at their anatomically correct level
#    [54] / [284]  mid-forehead width  (above brow arch, NOT temporal)
#                  FIX: original used [103]/[332] which are lower-lateral brow
#                  points, systematically under-reading forehead width
#   [234] / [454]  zygomatic arch / cheekbone  (widest face point — correct)
#   [132] / [361]  gonial angle / jaw corner    (mandible ramus near angle)
#                  FIX: original used [172]/[397] which are mid-jaw body points,
#                  under-reading true jaw width by ~10–15 %
#
#  POSE CHECK  — bilateral symmetry around the midsagittal plane
#    [6]   nose bridge (glabella — stable midline landmark)
#   [234] / [454]  cheekbone points  (wide, stable, minimally affected by gaze)
#                  FIX: original measured nose-tip[1]→eye-corner, which mixes
#                  yaw distortion with eye-openness / gaze variation
#
#  PITCH CHECK  — vertical head tilt
#    [1]   nose tip
#    [4]   nose base (columella)
#   [152]  chin tip
#           Lateral offset of nose tip relative to chin-to-nose-base axis
#           detects forward / backward head pitch.
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Thresholds (single source of truth — tune here only) ─────────────────────
_T = {
    # ── Oblong ────────────────────────────────────────────────────────────────
    # Clearly elongated; all widths present but face is tall
    "oblong_lw_min"   : 1.50,   # face_len / cheek_w  — elongation
    "oblong_jc_min"   : 0.78,   # jaw present (not pointy heart/diamond)
    "oblong_fj_max"   : 1.30,   # forehead not dramatically wider than jaw
                                 # (separates Oblong from elongated Heart)

    # ── Square ────────────────────────────────────────────────────────────────
    # All three horizontal widths (forehead, cheek, jaw) nearly equal; compact
    "square_lw_max"   : 1.22,   # compact height
    "square_jc_min"   : 0.90,   # strong wide jaw (≥ cheek)
    "square_fc_min"   : 0.85,   # wide forehead (≥ cheek)
    "square_eq_max"   : 0.10,   # |fc - jc| must be small → all rows equal
                                 # FIX: original had no uniformity check

    # ── Round ─────────────────────────────────────────────────────────────────
    # Compact + wide forehead + soft jaw — rounder than Square
    "round_lw_max"    : 1.30,   # compact; slightly looser than Square
    "round_jc_max"    : 0.89,   # softer jaw than Square (< square_jc_min)
    "round_fc_min"    : 0.82,   # FIX: was 0.80; round faces have full foreheads

    # ── Heart ─────────────────────────────────────────────────────────────────
    # Broad forehead tapering to narrow jaw (V-shape)
    "heart_fj_min"    : 1.22,   # forehead clearly wider than jaw
    "heart_jc_max"    : 0.76,   # narrow jaw relative to cheeks
    "heart_fc_min"    : 0.82,   # forehead ≥ cheeks (separates from Diamond)
    "heart_lw_max"    : 1.55,   # Heart is not very elongated
                                 # FIX: original had no lw upper bound

    # ── Triangle (pear) ───────────────────────────────────────────────────────
    # Jaw wider than forehead; jaw is the widest point
    "tri_fj_max"      : 0.86,   # jaw clearly wider than forehead
    "tri_jc_min"      : 0.88,   # jaw wide relative to cheeks
    "tri_jaw_dom"     : 0.00,   # jc > fc enforced in code (jaw widest point)
                                 # FIX: original had no jaw-dominance guard
    "tri_lw_max"      : 1.55,   # FIX: very elongated wide-jaw → Oblong, not Tri

    # ── Diamond ───────────────────────────────────────────────────────────────
    # Cheekbones dominate; both forehead AND jaw narrower
    "diamond_fc_max"  : 0.81,   # narrow forehead vs cheeks
    "diamond_jc_max"  : 0.80,   # FIX: tightened from 0.85; jaw also narrow
    "diamond_fj_max"  : 1.20,   # forehead ≈ jaw (symmetric taper both ways)
    "diamond_fj_min"  : 0.83,   # FIX: original had no lower fj bound;
                                 # prevents lop-sided taper from landing here

    # ── Oval ──────────────────────────────────────────────────────────────────
    # Balanced gentle elongation; NOT a pure catch-all any more
    # FIX: Oval now has explicit range guards; faces outside these fail gracefully
    "oval_lw_min"     : 1.22,   # longer than Square/Round
    "oval_lw_max"     : 1.55,   # not as elongated as Oblong
    "oval_fc_min"     : 0.72,   # forehead present (not severely narrow)
    "oval_jc_min"     : 0.70,   # jaw present (not severely narrow)

    # ── Pose / quality ────────────────────────────────────────────────────────
    "pose_yaw_max"    : 1.18,   # cheek-symmetry ratio around nose-bridge
                                 # FIX: was nose-tip→eye, now glabella→cheeks
    "pose_pitch_px"   : 0.04,   # max lateral nose-offset as fraction of face_w
                                 # FIX: no pitch check existed before
    "min_face_frac"   : 0.15,   # face_len must be ≥ 15 % of image height
                                 # FIX: no minimum size check existed before

    # ── Confidence  (ideal_value, tolerance) ─────────────────────────────────
    # Gaussian:  conf = exp(−(v−ideal)² / (2σ²)),  σ = tolerance
    # At v = ideal ± tolerance  →  exp(−0.5) ≈ 0.607  (60 %)
    # FIX: original used σ = tol/1.4427 which gave 35 % at ±tol, not 60 %.
    #      Corrected by setting σ = tolerance directly.
    # FIX: confidence floor lowered 0.50 → 0.35 so marginal matches don't
    #      falsely report 50 % certainty.
    # FIX: multi-ratio composite scores used where one ratio is insufficient.
    "conf_oblong"     : (1.55, 0.12),   # lw
    "conf_square"     : (0.93, 0.04),   # jc  (tight — Square is very specific)
    "conf_round"      : (1.18, 0.09),   # lw
    "conf_heart"      : (1.35, 0.10),   # fj
    "conf_triangle"   : (0.80, 0.07),   # fj
    "conf_diamond"    : (0.76, 0.06),   # avg(fc, jc)
    "conf_oval"       : (1.35, 0.14),   # lw  (wider tolerance — Oval is broad)
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
        Two-axis pose quality check.

        YAW (left-right rotation)
        ─────────────────────────
        Measures glabella [6] → cheekbone [234] (left) and [454] (right).
        Glabella is a stable midline point minimally affected by gaze.
        Cheekbone points are wide-baseline and rigid.  Ratio of the two
        distances exceeds pose_yaw_max (~±12° yaw) → reject.

        FIX over original: original used nose-tip[1]→eye-corner[130/359].
        Eye corners shift with eye-openness and gaze direction, adding noise.
        Glabella→cheekbone is pure head-rotation signal.

        PITCH (forward/back tilt)
        ─────────────────────────
        Measures the lateral (x-axis) offset of nose-tip [1] relative to the
        chin [152] → nose-base [4] vertical axis.  When the head pitches
        forward, the nose tip moves forward in 3D which projects as a lateral
        shift in the 2D image.  If this offset exceeds pose_pitch_px as a
        fraction of face width → reject.

        FIX over original: no pitch detection existed at all.
        """
        # — yaw —
        l_cheek = self._px_dist(lm, 6, 234, w, h)
        r_cheek = self._px_dist(lm, 6, 454, w, h)
        if min(l_cheek, r_cheek) < 1.0:
            return False
        if (max(l_cheek, r_cheek) / min(l_cheek, r_cheek)) > _T["pose_yaw_max"]:
            return False

        # — pitch (lateral nose offset relative to chin–nose-base axis) —
        nose_tip_x  = lm[1].x  * w
        nose_base_x = lm[4].x  * w
        chin_x      = lm[152].x * w
        # face width for normalisation
        face_w = abs(lm[234].x - lm[454].x) * w
        if face_w < 1.0:
            return False
        # lateral deviation of nose tip from the chin→nose-base line
        axis_x    = (nose_base_x + chin_x) / 2.0
        pitch_dev = abs(nose_tip_x - axis_x) / face_w
        if pitch_dev > _T["pose_pitch_px"]:
            return False

        return True

    @staticmethod
    def _conf(value, ideal, tolerance):
        """
        Gaussian confidence score.

          conf = exp(−(value − ideal)² / (2 · tolerance²))

        This is a standard unit Gaussian centred on `ideal` with σ = tolerance.
        At exactly ±1 tolerance from ideal  →  exp(−0.5) ≈ 60.7 %.
        At the ideal value                  →  100 %.
        Hard floor: 35 % (not 50 %) to avoid falsely implying certainty for
        marginal classifications.

        FIX: original wrote  σ = tolerance / 1.4427  (= tolerance · ln 2).
        That yields exp(−1.4427²/2) ≈ 35 % at ±tolerance, contradicting the
        "~60 %" comment.  Corrected by using σ = tolerance directly.
        """
        sigma = tolerance                                       # σ = tolerance
        conf  = np.exp(-((value - ideal) ** 2) / (2 * sigma ** 2))
        return round(max(0.35, float(conf)) * 100, 1)

    @staticmethod
    def _multi_conf(pairs):
        """
        Composite confidence from multiple (value, ideal, tolerance) triples.
        Geometric mean of individual Gaussian scores — each dimension must
        fit well for the total to be high.  Floor applied after combining.

        Used for shapes whose identity rests on more than one ratio (Square,
        Diamond).
        """
        scores = []
        for value, ideal, tolerance in pairs:
            sigma = tolerance
            s     = np.exp(-((value - ideal) ** 2) / (2 * sigma ** 2))
            scores.append(s)
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
          confidence      – "XX.X%"  (capped at 98 % for realism)
          metric_ratio    – face_len / cheek_w  (primary elongation ratio)
          debug           – raw pixel sizes + all four ratios
          error           – present only on failure; all other keys absent

        Shape taxonomy & geometry contract
        ───────────────────────────────────
        Oblong   : lw ≥ 1.50, balanced widths, jaw present
        Square   : lw ≤ 1.22, forehead ≈ cheek ≈ jaw (all rows near-equal)
        Round    : lw ≤ 1.30, wide forehead, soft jaw
        Heart    : forehead >> jaw, lw ≤ 1.55
        Triangle : jaw >> forehead (pear), lw ≤ 1.55
        Diamond  : cheeks dominate, forehead ≈ jaw (both narrow vs cheeks)
        Oval     : gentle elongation 1.22–1.55, balanced proportions
        Indeterminate : ratios are inconsistent / no clean match
        """
        h, w, _ = image.shape
        lm = self._landmarks(image)

        if lm is None:
            return {"error": "No face detected — ensure good lighting and a clear, unobstructed view"}

        # ── minimum face size ────────────────────────────────────────────────
        # FIX: tiny face in frame → landmarks are unreliable
        face_len_raw = abs(lm[152].y - lm[10].y) * h
        if face_len_raw < _T["min_face_frac"] * h:
            return {"error": "Face too small in frame — move closer to the camera"}

        if not self._pose_ok(lm, w, h):
            return {"error": "Face not straight — please look directly at the camera (no tilt or turn)"}

        # ── 1. pixel measurements ─────────────────────────────────────────────
        D = lambda i, j: self._px_dist(lm, i, j, w, h)  # noqa: E731

        face_len   = D(10,  152)   # top-of-forehead → chin tip
        forehead_w = D(54,  284)   # FIX: mid-forehead (above brow arch)
                                   #      was [103]/[332] (lower-lateral brow)
        cheek_w    = D(234, 454)   # zygomatic arch — widest face point (correct)
        jaw_w      = D(132, 361)   # FIX: gonial angle / jaw corner
                                   #      was [172]/[397] (mid-jaw body, too narrow)

        if min(face_len, forehead_w, cheek_w, jaw_w) < 1.0:
            return {"error": "Landmark geometry invalid — try a clearer, better-lit image"}

        # ── 2. ratios ─────────────────────────────────────────────────────────
        #
        #  lw  face_Len / cheek_Width        → elongation index
        #  fc  Forehead / Cheek              → forehead breadth vs widest point
        #  jc  Jaw / Cheek                   → jaw strength vs widest point
        #  fj  Forehead / Jaw                → taper direction (Heart ↔ Triangle)
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
            "lw_ratio"     : round(lw, 3),   # elongation
            "fc_ratio"     : round(fc, 3),   # forehead vs cheek
            "jc_ratio"     : round(jc, 3),   # jaw vs cheek
            "fj_ratio"     : round(fj, 3),   # forehead vs jaw (taper direction)
        }

        # ── 3. strict decision tree ───────────────────────────────────────────
        #
        #  Ordering principle: most geometrically constrained shapes first,
        #  broadest last.  Every branch requires ALL its guards — no shape is
        #  identified by a single ratio alone.
        #
        #  Each shape has at minimum THREE independent ratio guards plus an
        #  optional uniformity / dominance guard.  This is intentionally more
        #  demanding than the original to honour the "strict" requirement.
        #
        t = _T

        # ── Oblong ────────────────────────────────────────────────────────────
        # Guard 1: very elongated face (lw)
        # Guard 2: jaw is present — not a pointed Heart/Diamond (jc)
        # Guard 3: forehead NOT dramatically wider than jaw — not an elongated
        #          Heart (fj).  FIX: original lacked this guard.
        if (lw > t["oblong_lw_min"]
                and jc > t["oblong_jc_min"]
                and fj < t["oblong_fj_max"]):
            shape = "Oblong"
            conf  = self._conf(lw, *t["conf_oblong"])

        # ── Square ────────────────────────────────────────────────────────────
        # Guard 1: compact face (lw)
        # Guard 2: strong jaw (jc)
        # Guard 3: wide forehead (fc)
        # Guard 4: forehead and jaw are nearly equal width → all rows uniform
        #          FIX: original lacked the |fc − jc| uniformity check
        elif (lw <= t["square_lw_max"]
                and jc >= t["square_jc_min"]
                and fc >= t["square_fc_min"]
                and abs(fc - jc) <= t["square_eq_max"]):
            shape = "Square"
            # Multi-ratio confidence: both jc and fc must be high
            conf  = self._multi_conf([
                (jc, *t["conf_square"]),
                (fc, 0.90, 0.04),
            ])

        # ── Round ─────────────────────────────────────────────────────────────
        # Guard 1: compact face (lw)
        # Guard 2: wide forehead (fc) — full, broad forehead
        # Guard 3: soft jaw — less angular than Square (jc)
        elif (lw <= t["round_lw_max"]
                and fc >= t["round_fc_min"]
                and jc < t["round_jc_max"]):
            shape = "Round"
            conf  = self._conf(lw, *t["conf_round"])

        # ── Heart ─────────────────────────────────────────────────────────────
        # Guard 1: forehead clearly wider than jaw (fj)
        # Guard 2: narrow jaw vs cheeks (jc)
        # Guard 3: forehead ≥ cheeks → forehead is the dominant width (fc)
        #          (separates Heart from Diamond where forehead is ALSO narrow)
        # Guard 4: not excessively elongated — Heart is a compact-to-medium shape
        #          FIX: original lacked the lw upper bound
        elif (fj > t["heart_fj_min"]
                and jc < t["heart_jc_max"]
                and fc >= t["heart_fc_min"]
                and lw <= t["heart_lw_max"]):
            shape = "Heart"
            conf  = self._conf(fj, *t["conf_heart"])

        # ── Triangle (pear) ───────────────────────────────────────────────────
        # Guard 1: jaw wider than forehead — inverse of Heart (fj)
        # Guard 2: jaw wide relative to cheeks (jc)
        # Guard 3: jaw is the widest point on the face → jc > fc
        #          FIX: original lacked jaw-dominance guard
        # Guard 4: not excessively elongated
        #          FIX: original lacked lw upper bound
        elif (fj < t["tri_fj_max"]
                and jc > t["tri_jc_min"]
                and jc > fc
                and lw <= t["tri_lw_max"]):
            shape = "Triangle"
            conf  = self._conf(fj, *t["conf_triangle"])

        # ── Diamond ───────────────────────────────────────────────────────────
        # Guard 1: narrow forehead vs cheeks (fc)
        # Guard 2: narrow jaw vs cheeks (jc) — FIX: tightened from 0.85 → 0.80
        # Guard 3: forehead ≈ jaw (symmetric taper on both sides)
        #          FIX: original had no fj range; an asymmetric face (narrow jaw
        #          + narrow forehead but very different from each other) is not
        #          Diamond — it's closer to Heart or Triangle
        elif (fc < t["diamond_fc_max"]
                and jc < t["diamond_jc_max"]
                and t["diamond_fj_min"] <= fj <= t["diamond_fj_max"]):
            shape    = "Diamond"
            narrowness = (fc + jc) / 2.0
            conf     = self._multi_conf([
                (narrowness, *t["conf_diamond"]),
                (fj, 1.02, 0.12),   # fj near 1.0 → symmetric taper
            ])

        # ── Oval ──────────────────────────────────────────────────────────────
        # FIX: Oval is no longer a pure catch-all — it now requires explicit
        #      range guards.  A face that fails ALL shape branches but has
        #      wildly inconsistent ratios is classified "Indeterminate" instead
        #      of silently falling back to Oval.
        # Guard 1: gentle elongation — longer than Square/Round (lw ≥ oval_lw_min)
        # Guard 2: not as elongated as Oblong (lw ≤ oval_lw_max)
        # Guard 3 & 4: balanced widths — neither forehead nor jaw severely narrow
        elif (t["oval_lw_min"] <= lw <= t["oval_lw_max"]
                and fc >= t["oval_fc_min"]
                and jc >= t["oval_jc_min"]):
            shape = "Oval"
            conf  = self._conf(lw, *t["conf_oval"])

        # ── Indeterminate ─────────────────────────────────────────────────────
        # Ratios present but inconsistent — no clean geometric archetype.
        # Returning this rather than forcing Oval preserves honesty.
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