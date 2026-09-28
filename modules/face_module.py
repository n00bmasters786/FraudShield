"""
Module A — Face liveness & deepfake forensics on a live screen capture.

The banking officer shares their screen. `FaceStream` scans every captured
frame for the customer's face inside the video-call window, tracks it with
MediaPipe Face Mesh (478 landmarks) at native screen resolution and records
per-frame measurements. `analyze_window()` re-scores the most recent ~20 s
with independent checks, each aimed at a different attack:

  check                      catches
  -------------------------  ---------------------------------------------
  0. Presence / 2nd face     face-swap dropouts, coached / assisted sessions
  1. Blink behaviour         photos, masks, face-swaps (rate, depth,
                             left/right eye symmetry)
  2. Expression dynamics     printed photo / frozen frame / looped video
  3. 3-D parallax            photo held up and moved: a flat picture's
                             landmarks obey a single homography, a real
                             3-D head doesn't
  4. Landmark jitter         frame-by-frame generated faces flicker in
                             ways rigid head motion can't explain
  5. Remote pulse (rPPG)     live skin shows a heartbeat in the colour of
                             forehead/cheeks (POS algorithm); synthetic
                             faces and replays usually don't
  6. Face-swap blend seam    swapped faces are smoother (re-rendered and
                             up-scaled) than the untouched neck skin
  7. Screen recapture        moiré peaks + glare from a phone/monitor held
                             up to the customer's camera
  8. AI deepfake detector    trained MS-EffGCViT ensemble (FaceForensics++ +
                             Celeb-DF) on face crops — face swaps and
                             reenactment that pass every liveness check

Checks 0-3, 5 and 7 are *liveness* evidence (is a live person there?);
4, 6 and 8 are *synthesis* evidence (is the face generated?). The face score
is the worse of the two groups — see scoring.group_score. Within synthesis
the trained detector dominates: on the evaluation sample (tools/evaluate.py)
the hand-made jitter / seam checks were near chance at telling deepfakes from
real faces, so they are kept only as minor, non-decisive evidence.

`score_video(path)` runs the same pipeline over a video file (offline tests).
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as sps

from . import calibration
from .scoring import Signal, clip01, group_score, ramp

try:  # legacy "solutions" API — present in the pinned mediapipe 0.10.14
    import mediapipe as mp
    _FACE_MESH = mp.solutions.face_mesh
    _FACE_DET = mp.solutions.face_detection
except Exception:  # pragma: no cover - depends on install
    _FACE_MESH = _FACE_DET = None

# --------------------------------------------------------------------------
# Landmark index sets (MediaPipe Face Mesh topology)
# --------------------------------------------------------------------------
EYE_R = [33, 160, 158, 133, 153, 144]     # p1..p6 for EAR, image-left eye
EYE_L = [362, 385, 387, 263, 373, 380]    # image-right eye
IOD_PAIR = (33, 263)                       # outer eye corners -> scale
POSE_IDX = [1, 152, 33, 263, 61, 291]      # nose, chin, eye corners, mouth corners
POSE_3D = np.array([                       # generic head model, camera frame
    [0.0, 0.0, 0.0],                       # (y down, z away from camera)
    [0.0, 330.0, 65.0],
    [-225.0, -170.0, 135.0],
    [225.0, -170.0, 135.0],
    [-150.0, 150.0, 125.0],
    [150.0, 150.0, 125.0],
], dtype=np.float64)
RIGID_IDX = [1, 4, 5, 6, 9, 10, 151, 168, 195, 197, 33, 133, 362, 263, 127, 356]
PARALLAX_IDX = [1, 4, 6, 10, 152, 33, 133, 362, 263, 61, 291, 234, 454, 168, 199, 50, 280, 127, 356]
MOUTH = (13, 14, 61, 291)                  # upper lip, lower lip, corners
BROWS = (105, 334)                         # mid-brow points
FOREHEAD_C, CHEEK_R_C, CHEEK_L_C, CHIN = 151, 50, 280, 152

# drawn on the officer's live preview
CONTOURS = {
    "oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "eye_l": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "eye_r": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "brow_l": [336, 296, 334, 293, 300],
    "brow_r": [107, 66, 105, 63, 70],
    "nose": [168, 6, 197, 195, 5, 4, 1],
}
MESH_DOTS = list(range(0, 468, 7))

# --------------------------------------------------------------------------
# Live-tracking parameters
# --------------------------------------------------------------------------
WINDOW_S = 20.0          # analysis window
KEEP_S = 30.0            # history kept in memory
GAP_S = 1.5              # face missing this long = a new tracking segment
MIN_FACE_PX = 36         # ignore faces smaller than this on screen (thumbnails, avatars)
SCAN_EVERY_S = 0.35      # full-screen search while no face is locked
IDLE_SCAN_EVERY_S = 1.0  # ...slower once nothing has been on screen for IDLE_AFTER_S
IDLE_AFTER_S = 10.0
RESCAN_EVERY_S = 2.5     # full-screen check for a bigger face / a second person while tracking
SLOW_EVERY_S = 2.0       # 3-D parallax, pulse and texture checks are refreshed at this cadence
CROP_PAD = 1.25          # region streamed while tracking = Face Mesh crop × this (covers neck + detector margin)
PARALLAX_PAIRS = 600     # frame pairs sampled for the 3-D test
LOST_AFTER_S = 0.8       # mesh lost this long -> drop the lock and search again
TEX_EVERY_S = 0.5        # texture checks (blend seam, moiré) cadence
MESH_INPUT = 480         # crop fed to Face Mesh is resized to this side
MODEL_EVERY_S = 0.25     # trained detector cadence (4 crops/s)


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------
def _area(b):
    return b[2] * b[3]


def _center(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / (_area(a) + _area(b) - inter + 1e-9), inter / (min(_area(a), _area(b)) + 1e-9)


def _expand(b, fx, fy=None, W=None, H=None):
    fy = fx if fy is None else fy
    cx, cy = _center(b)
    w, h = b[2] * fx, b[3] * fy
    x0, y0 = cx - w / 2, cy - h / 2
    if W is not None:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        w, h = min(W - x0, w), min(H - y0, h)
    return (x0, y0, w, h)


def _inside(pt, b):
    return b[0] <= pt[0] <= b[0] + b[2] and b[1] <= pt[1] <= b[1] + b[3]


def _bbox(pts):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def _norm_box(b, W, H):
    return [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)]


# --------------------------------------------------------------------------
# Face finding on a whole screen
# --------------------------------------------------------------------------
class _ScreenFaceFinder:
    """Finds faces anywhere on a large screen frame.

    BlazeFace works on a 192×192 input, so a 150 px face on a 1920 px screen
    would shrink to ~15 px and be missed. The region is therefore scanned
    whole *and* as overlapping tiles, and the detections merged.
    """

    def __init__(self):
        self.det = None
        if _FACE_DET is not None:
            try:
                self.det = _FACE_DET.FaceDetection(model_selection=1, min_detection_confidence=0.55)
            except Exception:
                self.det = None
        self.cc = None if self.det else cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.name = "MediaPipe BlazeFace (tiled)" if self.det else "OpenCV Haar cascade"

    def _detect(self, img, ox, oy):
        h, w = img.shape[:2]
        if h < 32 or w < 32:
            return []
        s = min(1.0, 640.0 / max(h, w))
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
        out = []
        if self.det is not None:
            res = self.det.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                rb = d.location_data.relative_bounding_box
                x, y = max(0.0, rb.xmin) * w, max(0.0, rb.ymin) * h
                bw, bh = min(rb.width * w, w - x), min(rb.height * h, h - y)
                if bw > 4 and bh > 4:
                    out.append((ox + x, oy + y, bw, bh, float(d.score[0])))
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            for (x, y, bw, bh) in self.cc.detectMultiScale(gray, 1.15, 5, minSize=(24, 24)):
                out.append((ox + x / s, oy + y / s, bw / s, bh / s, 0.6))
        return out

    @staticmethod
    def _tiles(w, h):
        side = int(min(w, h) * 0.62)
        if side < 240:
            return []
        xs = np.linspace(0, w - side, max(1, math.ceil((w - side) / (0.72 * side)) + 1)).astype(int)
        ys = np.linspace(0, h - side, max(1, math.ceil((h - side) / (0.72 * side)) + 1)).astype(int)
        return [(int(x), int(y), side) for y in ys for x in xs]

    def find(self, frame, region=None, tiles=True):
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = region or (0, 0, W, H)
        sub = frame[y0:y1, x0:x1]
        boxes = self._detect(sub, x0, y0)
        if tiles:
            for tx, ty, side in self._tiles(x1 - x0, y1 - y0):
                boxes += self._detect(sub[ty:ty + side, tx:tx + side], x0 + tx, y0 + ty)
        # merge: biggest first, drop anything mostly covered by a kept box (tile-edge partials)
        kept = []
        for b in sorted(boxes, key=_area, reverse=True):
            if all(max(_iou(b, k)) < 0.5 for k in kept):
                kept.append(b)
        return [k[:4] for k in kept]

    def close(self):
        if self.det is not None:
            self.det.close()


class _Mesh:
    name = "MediaPipe Face Mesh · 478 landmarks"

    def __init__(self):
        self.m = _FACE_MESH.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __call__(self, rgb):
        res = self.m.process(rgb)
        if not res.multi_face_landmarks:
            return None
        return np.array([(p.x, p.y) for p in res.multi_face_landmarks[0].landmark], dtype=np.float32)

    def close(self):
        self.m.close()


# --------------------------------------------------------------------------
# Per-frame measurements
# --------------------------------------------------------------------------
def _ear(pts, idx):
    p = pts[idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    hdist = np.linalg.norm(p[0] - p[3]) + 1e-6
    return float(v / (2.0 * hdist))


def _roi_circles(lm, iod):
    """Forehead, cheeks, neck as (cx, cy, r)."""
    chin = lm[CHIN]
    return {
        "forehead": (*lm[FOREHEAD_C], 0.22 * iod),
        "cheek_r": (*lm[CHEEK_R_C], 0.16 * iod),
        "cheek_l": (*lm[CHEEK_L_C], 0.16 * iod),
        "neck": (chin[0], chin[1] + 0.45 * iod, 0.17 * iod),
    }


def _circle_patch(img, c, min_inside=0.6):
    """Pixels of `img` inside circle c=(cx,cy,r); None if mostly off-frame."""
    cx, cy, r = c
    h, w = img.shape[:2]
    r = max(2.0, r)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if mask.sum() < min_inside * math.pi * r * r:
        return None
    return img[y0:y1, x0:x1][mask]


def _noise_sigma(gray_patch_pixels):
    """Robust std of a high-pass residual (pixels already residual values)."""
    if gray_patch_pixels is None or gray_patch_pixels.size < 30:
        return None
    med = np.median(gray_patch_pixels)
    return float(1.4826 * np.median(np.abs(gray_patch_pixels - med)) + 1e-3)


def _moire_score(gray_face):
    """Spectral peak index: 99.9th-percentile z-score of mid/high-frequency
    bins against the radial median — screens add isolated moiré peaks.

    Computed at native resolution (no up-scaling) and ignoring the 8x8
    codec-block lattice so JPEG/H.264 blocking isn't mistaken for a screen.
    """
    n = min(gray_face.shape[:2])
    n = int(min(256, n - n % 8))
    if n < 64:
        return None
    cy, cx = gray_face.shape[0] // 2, gray_face.shape[1] // 2
    g = gray_face[cy - n // 2: cy - n // 2 + n, cx - n // 2: cx - n // 2 + n].astype(np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
    yy, xx = np.mgrid[:n, :n]
    u, v = xx - n // 2, yy - n // 2
    rad = np.sqrt(u ** 2 + v ** 2).astype(int)
    maxr = n // 2
    med = np.zeros(maxr + 1)
    mad = np.ones(maxr + 1)
    for r in range(1, maxr + 1):
        ring = mag[rad == r]
        if ring.size:
            med[r] = np.median(ring)
            mad[r] = 1.4826 * np.median(np.abs(ring - med[r])) + 1e-3
    rr = np.clip(rad, 0, maxr)
    z = (mag - med[rr]) / mad[rr]
    band = (rad > 0.2 * maxr) & (rad < 0.95 * maxr)
    band &= (np.abs(u) > 2) & (np.abs(v) > 2)                     # codec/axis lines
    step = n // 8
    lattice = (np.abs(((u + step // 2) % step) - step // 2) <= 1) & \
              (np.abs(((v + step // 2) % step) - step // 2) <= 1)
    band &= ~lattice
    if band.sum() == 0:
        return None
    return float(np.percentile(z[band], 99.9))


def _glare_fraction(bgr_face):
    hsv = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[..., 2] > 240) & (hsv[..., 1] < 25)))


@dataclass
class FrameRec:
    t: float
    lm: Optional[np.ndarray] = None          # (478, 2) screen pixels
    n_faces: int = 0                         # faces around the tracked one
    iod: float = float("nan")
    mouth: float = float("nan")              # lip gap / IOD, for lip-sync
    rgb: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    blend: Optional[float] = None
    moire: Optional[float] = None
    glare: Optional[float] = None
    # computed once per frame at ingest, instead of for the whole window every second
    ear_r: float = float("nan")
    ear_l: float = float("nan")
    brow: float = float("nan")
    yaw: float = float("nan")
    pitch: float = float("nan")
    jit: float = float("nan")                # landmark residual vs. the previous frame, % of IOD


# --------------------------------------------------------------------------
# Live stream
# --------------------------------------------------------------------------
class FaceStream:
    """Consumes screen frames, locks onto the customer's face, records measurements.

    `process()` is called from one worker thread; `analyze_window()` and
    `mouth_series()` may be called concurrently from another.
    """

    def __init__(self, keep_s: float = KEEP_S, model: Optional[str] = "async"):
        """model: "async" (live — scored on a background thread), "sync" (offline, every
        sampled frame scored inline) or None (forensic checks only)."""
        self.keep_s = keep_s
        self.lock = threading.Lock()
        self.records: deque = deque()
        self.model_recs: deque = deque()          # (t, P(fake) per checkpoint, face px)
        self.model_mode = model
        self._scorer = None
        self._clf = None
        if model == "async":
            from .detectors.face_classifier import AsyncFaceScorer
            self._scorer = AsyncFaceScorer(self._on_model)
        elif model == "sync":
            from .detectors.face_classifier import FaceDeepfakeClassifier
            self._clf = FaceDeepfakeClassifier.get()
        self.roi: Optional[Tuple[float, float, float, float]] = None   # normalised, officer-selected
        self._finder = _ScreenFaceFinder()
        self._mesh = _Mesh() if _FACE_MESH is not None else None
        self.backend = f"{_Mesh.name if self._mesh else 'no landmark model'} · {self._finder.name}"
        self.frame_size: Optional[Tuple[int, int]] = None
        self.faces_on_screen = 0
        self._slow = None                          # cached slow-check results (see analyze_window)
        self.events: List[Tuple[str, str]] = []   # (kind, message) drained by the session
        self._reset_tracking()

    def _reset_tracking(self):
        self._target = None
        self._crop = None
        self._lost_since = None
        self._last_scan = -1e9
        self._last_seen = None                      # last time any face was found (search back-off)
        self._prev_lm = None
        self._cam = None                            # running virtual camera (focal, cx, cy) for head pose
        self._last_tex = -1e9
        self._last_model_t = -1e9
        self._n_near = 1

    def reset(self):
        with self.lock:
            self.records.clear()
            self.model_recs.clear()
        self._reset_tracking()

    def _on_model(self, t, probs, meta):
        with self.lock:
            self.model_recs.append((t, np.asarray(probs, np.float32), meta["face_px"]))
            while self.model_recs and self.model_recs[0][0] < t - self.keep_s:
                self.model_recs.popleft()

    @property
    def model_names(self):
        from .detectors.face_classifier import FaceDeepfakeClassifier
        clf = FaceDeepfakeClassifier._instance
        return clf.names if clf is not None else None

    @property
    def model_status(self) -> str:
        from .detectors.face_classifier import FaceDeepfakeClassifier
        return "off" if self.model_mode is None else FaceDeepfakeClassifier.status

    def set_roi(self, roi):
        roi = tuple(float(v) for v in roi) if roi else None
        if roi != self.roi:
            self.roi = roi
            self._target = None          # re-acquire inside the new region
            self._crop = None
            self._last_scan = -1e9

    def close(self):
        if self._scorer is not None:
            self._scorer.close()
        self._finder.close()
        if self._mesh is not None:
            self._mesh.close()

    # ---------------------------------------------------------------- frames
    def _roi_px(self, W, H):
        if not self.roi:
            return None
        x, y, w, h = self.roi
        x0, y0 = int(max(0, x * W)), int(max(0, y * H))
        x1, y1 = int(min(W, (x + w) * W)), int(min(H, (y + h) * H))
        return (x0, y0, x1, y1) if x1 - x0 > 32 and y1 - y0 > 32 else None

    def _acquire(self, box, why):
        self._target = box
        self._crop = None
        self._lost_since = None
        if why == "switch":
            with self.lock:
                self.records.clear()     # a different person — start a fresh window
                self.model_recs.clear()
            self.events.append(("switch", "Switched to a larger face on screen — analysis restarted"))
        else:
            self.events.append(("acquire", "Customer face locked"))

    def _update_crop(self, W, H):
        """Square crop around the face with hysteresis, so Face Mesh tracking stays stable."""
        x, y, w, h = self._target
        cx, cy = x + w / 2, y + h / 2 + 0.12 * h
        side = 2.1 * max(w, h)
        if self._crop is not None:
            c0, c1, c2, c3 = self._crop
            cs = max(c2 - c0, c3 - c1)
            ok_center = abs(cx - (c0 + c2) / 2) < 0.2 * cs and abs(cy - (c1 + c3) / 2) < 0.2 * cs
            ok_size = 0.38 < max(w, h) / cs < 0.62
            if ok_center and ok_size:
                return
        x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        self._crop = (x0, y0, x1, y1)

    def process(self, t: float, frame: np.ndarray, origin=(0, 0), full_size=None) -> dict:
        """Ingest one BGR frame captured at time t (seconds). Returns overlay info + what to send next.

        The browser sends the whole screen only while searching (and every RESCAN_EVERY_S while
        tracking); otherwise just the region around the customer's face, at native resolution.
        `origin` is that region's top-left corner in full-screen pixels and `full_size` = (W, H)
        of the full screen. All tracking state is kept in full-screen coordinates.
        """
        ox, oy = int(origin[0]), int(origin[1])
        fh, fw = frame.shape[:2]
        W, H = full_size if full_size else (fw, fh)
        is_full = ox == 0 and oy == 0 and fw >= W - 1 and fh >= H - 1
        self.frame_size = (H, W)
        roi_px = self._roi_px(W, H)

        # ---- 1. locate the customer's face (full frames only)
        if is_full:
            idle = self._last_seen is None or t - self._last_seen > IDLE_AFTER_S
            scan_every = RESCAN_EVERY_S if self._target is not None else (IDLE_SCAN_EVERY_S if idle else SCAN_EVERY_S)
            if t - self._last_scan >= scan_every:
                self._last_scan = t
                faces = [b for b in self._finder.find(frame, roi_px) if min(b[2], b[3]) >= MIN_FACE_PX]
                self.faces_on_screen = len(faces)
                biggest = max(faces, key=_area, default=None)
                if biggest is not None:
                    self._last_seen = t
                if self._target is None:
                    if biggest is not None:
                        self._acquire(biggest, "acquire")
                else:
                    if biggest is not None and _area(biggest) > 2.5 * _area(self._target) \
                            and max(_iou(biggest, self._target)) < 0.1:
                        self._acquire(biggest, "switch")
                    # ---- 2. is a second person sitting next to the customer?
                    nb = _expand(self._target, 3.0, 2.0)
                    self._n_near = max(1, sum(1 for b in faces if _inside(_center(b), nb)
                                              and min(b[2], b[3]) >= 0.45 * min(self._target[2], self._target[3])))

        # ---- 3. landmarks on a native-resolution crop
        lm = None
        stale = False                                 # this frame doesn't cover the face region
        if self._target is not None and self._mesh is not None:
            self._update_crop(W, H)
            x0, y0, x1, y1 = self._crop
            lx0, ly0 = max(0, x0 - ox), max(0, y0 - oy)
            lx1, ly1 = min(fw, x1 - ox), min(fh, y1 - oy)
            if (lx1 - lx0) * (ly1 - ly0) < 0.6 * (x1 - x0) * (y1 - y0):
                stale = True
            else:
                crop = frame[ly0:ly1, lx0:lx1]
                s = MESH_INPUT / max(lx1 - lx0, ly1 - ly0)
                small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                nl = self._mesh(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                if nl is not None:
                    pts = np.empty_like(nl)
                    pts[:, 0] = ox + lx0 + nl[:, 0] * (lx1 - lx0)
                    pts[:, 1] = oy + ly0 + nl[:, 1] * (ly1 - ly0)
                    box = _bbox(pts[:468])
                    if _inside(_center(box), _expand(self._target, 1.6)) and min(box[2], box[3]) >= MIN_FACE_PX * 0.8:
                        lm = pts
                        self._target = box
                        self._lost_since = None
                        self._last_seen = t
            if lm is None and not stale:
                if self._lost_since is None:
                    self._lost_since = t
                elif t - self._lost_since > LOST_AFTER_S:
                    self._reset_tracking()
                    self._last_scan = t - SCAN_EVERY_S       # search again right away
                    self._last_seen = t
                    self.events.append(("lost", "Customer face lost from the screen"))

        # ---- 4. measurements (pixel work in this frame's local coordinates)
        rec = FrameRec(t=t, lm=lm, n_faces=self._n_near if lm is not None else 0)
        if lm is not None:
            local = lm - np.array([ox, oy], np.float32)
            iod = float(np.linalg.norm(lm[IOD_PAIR[0]] - lm[IOD_PAIR[1]]))
            rec.iod = iod
            rec.mouth = float(np.linalg.norm(lm[MOUTH[0]] - lm[MOUTH[1]]) / (iod + 1e-6))
            self._geometry(lm, iod, rec)
            rois = _roi_circles(local, iod)
            for k in ("forehead", "cheek_r", "cheek_l"):
                px = _circle_patch(frame, rois[k])
                if px is not None and len(px) >= 10:
                    b, g, r = px.reshape(-1, 3).mean(0)
                    rec.rgb[k] = (float(r), float(g), float(b))
            if t - self._last_tex >= TEX_EVERY_S:
                self._last_tex = t
                self._texture(frame, local, rois, rec)
            self._score_face(t, frame, local)
        self._prev_lm = lm

        if not stale:
            with self.lock:
                self.records.append(rec)
                while self.records and self.records[0].t < t - self.keep_s:
                    self.records.popleft()

        # ---- 5. overlay for the officer's preview + what the browser should send next
        ov = {"target": None, "contours": None, "dots": None, "crop": None,
              "faces_on_screen": self.faces_on_screen, "tracking": lm is not None,
              "want": self._want(t, W, H)}
        if self._target is not None:
            ov["target"] = _norm_box(self._target, W, H)
        if self._crop is not None:
            c = self._crop
            ov["crop"] = _norm_box((c[0], c[1], c[2] - c[0], c[3] - c[1]), W, H)
        if lm is not None:
            n = lm / np.array([W, H], np.float32)
            ov["contours"] = {k: np.round(n[idx], 4).ravel().tolist() for k, idx in CONTOURS.items()}
            ov["dots"] = np.round(n[MESH_DOTS], 4).ravel().tolist()
            ov["iod"] = round(rec.iod, 1)
        return ov

    def _want(self, t, W, H):
        """Tell the browser what to capture next: the whole screen (searching, periodic re-scan)
        or just the face region, and how often."""
        if self._target is None or self._crop is None:
            idle = self._last_seen is None or t - self._last_seen > IDLE_AFTER_S
            return {"full": True, "fps": 1.0 / (IDLE_SCAN_EVERY_S if idle else SCAN_EVERY_S)}
        x0, y0, x1, y1 = self._crop
        r = _expand((x0, y0, x1 - x0, y1 - y0), CROP_PAD, CROP_PAD, W, H)
        return {"full": t - self._last_scan >= RESCAN_EVERY_S - 0.07, "fps": 15,
                "rect": [round(r[0] / W, 4), round(r[1] / H, 4), round(r[2] / W, 4), round(r[3] / H, 4)]}

    def _geometry(self, lm, iod, rec):
        """Per-frame eye openness, brow height, head pose and landmark jitter (cached on the record)."""
        rec.ear_r, rec.ear_l = _ear(lm, EYE_R), _ear(lm, EYE_L)
        rec.brow = float((np.linalg.norm(lm[BROWS[0]] - lm[33]) + np.linalg.norm(lm[BROWS[1]] - lm[263])) / (2 * iod))
        cam = (6.0 * iod, float(lm[1][0]), float(lm[1][1]))
        # slowly-moving virtual camera ≈ the window-median camera the analysis used to rebuild every second
        self._cam = cam if self._cam is None else tuple(0.95 * a + 0.05 * b for a, b in zip(self._cam, cam))
        rec.yaw, rec.pitch, _ = head_pose(lm, self._cam)
        if self._prev_lm is not None:
            r, _ = _similarity_residual(self._prev_lm[RIGID_IDX], lm[RIGID_IDX])
            rec.jit = 100.0 * r / iod if np.isfinite(r) else float("nan")

    def _score_face(self, t, frame, lm):
        """Hand a native-resolution face crop to the trained detector."""
        if self._scorer is None and self._clf is None:
            return
        if t - self._last_model_t < MODEL_EVERY_S:
            return
        self._last_model_t = t
        from .detectors.face_classifier import effective_scale, face_crop
        box = _bbox(lm[:468])
        crop = face_crop(frame, box)
        if crop is None:
            return
        # pixels of real detail across the face, not its (possibly upscaled) size on screen
        meta = {"face_px": float(min(box[2], box[3])) * effective_scale(crop)}
        if self._scorer is not None:
            self._scorer.submit(t, crop, meta)
        else:
            self._on_model(t, self._clf.predict([crop])[0], meta)

    @staticmethod
    def _texture(frame, lm, rois, rec):
        """Blend-seam, moiré and glare checks on a native-resolution region around the face."""
        H, W = frame.shape[:2]
        fb = _bbox(lm[:468])
        nk = rois["neck"]
        x0 = int(max(0, fb[0] - 0.1 * fb[2]))
        y0 = int(max(0, fb[1] - 0.1 * fb[3]))
        x1 = int(min(W, fb[0] + 1.1 * fb[2]))
        y1 = int(min(H, max(fb[1] + 1.1 * fb[3], nk[1] + nk[2] + 2)))
        if x1 - x0 < 48 or y1 - y0 < 48:
            return
        sub = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
        shift = {k: (c[0] - x0, c[1] - y0, c[2]) for k, c in rois.items()}
        cheeks = [_noise_sigma(_circle_patch(resid, shift[k])) for k in ("cheek_r", "cheek_l")]
        neck_px = _circle_patch(sub, shift["neck"])
        neck_sig = _noise_sigma(_circle_patch(resid, shift["neck"]))
        if neck_px is not None and neck_sig is not None and neck_px.mean() > 40 and all(c is not None for c in cheeks):
            rec.blend = float(np.mean(cheeks) / neck_sig)
        fx0 = int(max(0, fb[0] + 0.15 * fb[2])) - x0
        fy0 = int(max(0, fb[1] + 0.2 * fb[3])) - y0
        fx1 = int(min(W, fb[0] + 0.85 * fb[2])) - x0
        fy1 = int(min(H, fb[1] + 0.9 * fb[3])) - y0
        if fx1 - fx0 > 48 and fy1 - fy0 > 48:
            face = sub[fy0:fy1, fx0:fx1]
            rec.moire = _moire_score(gray[fy0:fy1, fx0:fx1])
            rec.glare = _glare_fraction(face)

    # -------------------------------------------------------------- analysis
    def _segment(self, window_s):
        """Records of the current face-tracking segment inside the window."""
        with self.lock:
            recs = list(self.records)
        if not recs:
            return [], None
        t_end = recs[-1].t
        recs = [r for r in recs if r.t >= t_end - window_s]
        seen = [r.t for r in recs if r.lm is not None]
        if not seen or t_end - seen[-1] > GAP_S:
            return [], t_end
        start = seen[0]
        for a, b in zip(seen[-2::-1], seen[:0:-1]):     # walk back through consecutive sightings
            if b - a > GAP_S:
                start = b
                break
        return [r for r in recs if r.t >= start], t_end

    def mouth_series(self, window_s: float = 12.0):
        with self.lock:
            recs = list(self.records)
        if not recs:
            return np.array([]), np.array([])
        t_end = recs[-1].t
        recs = [r for r in recs if r.t >= t_end - window_s]
        return np.array([r.t for r in recs]), np.array([r.mouth for r in recs], float)

    def analyze_window(self, window_s: float = WINDOW_S) -> dict:
        seg, t_end = self._segment(window_s)
        out = {"status": "searching", "score": None, "confidence": 0.0, "signals": [], "reasons": [],
               "details": {"backend": self.backend, "faces_on_screen": self.faces_on_screen}}
        if not seg:
            return out
        dur = seg[-1].t - seg[0].t
        n = len(seg)
        if dur < 2.5 or n < 15:
            out["status"] = "warming"
            out["details"]["tracked_s"] = round(dur, 1)
            return out
        fps = (n - 1) / dur
        ts = np.array([r.t for r in seg])
        lms = [r.lm for r in seg]
        iods = np.array([r.iod for r in seg], float)
        present = np.array([r.lm is not None for r in seg])
        nf = np.array([r.n_faces for r in seg])
        signals: List[Signal] = []

        # 0. presence / second person
        pres = float(present.mean())
        multi = float(np.mean(nf[present] >= 2)) if present.any() else 0.0
        if pres < 0.7:
            s = Signal("presence", "Face tracking", f"{100 * pres:.0f}%", 0.55, 1.0, 1.0,
                       f"Face tracking dropped in {100 * (1 - pres):.0f}% of frames — occlusion, or a "
                       f"face-swap failing to render")
        elif multi > 0.2:
            s = Signal("presence", "Face tracking", f"{100 * pres:.0f}%", 0.55, 1.0, 1.0,
                       f"A second face is next to the customer in {100 * multi:.0f}% of frames — possible "
                       f"coached / assisted session")
        else:
            s = Signal("presence", "Face tracking", f"{100 * pres:.0f}%", 0.02, 1.0, 1.0,
                       f"Single face tracked steadily ({100 * pres:.0f}% of frames)")
        signals.append(s)

        med_iod = float(np.nanmedian(iods))
        size_rel = clip01((med_iod - 22.0) / 38.0)    # small faces on screen = noisier landmarks

        # per-frame quantities were computed once at ingest
        arr = lambda k: np.array([getattr(r, k) for r in seg], float)
        ear_r, ear_l = arr("ear_r"), arr("ear_l")
        pre = {"yaw": arr("yaw"), "pitch": arr("pitch"), "jit": arr("jit"), "mouth": arr("mouth"),
               "brow": arr("brow"), "ear": (ear_r + ear_l) / 2}

        # 1. blinks
        s_blink, b = analyze_blinks(ear_r, ear_l, fps)
        s_blink.reliability *= size_rel
        signals.append(s_blink)

        # 2-4 geometry, 5 pulse, 6-7 texture: the slower checks are refreshed every SLOW_EVERY_S
        # (they summarise 20 s of video, so a 1 s refresh added nothing but CPU)
        slow = self._slow
        if slow is None or slow["start"] != seg[0].t or t_end - slow["t"] >= SLOW_EVERY_S:
            geo_sigs, g = analyze_geometry(lms, iods, fps, pre)
            fps_u = float(np.clip(fps, 8.0, 30.0))
            tg = np.arange(ts[0], ts[-1], 1.0 / fps_u)
            rgb_u = {}
            for k in ("forehead", "cheek_r", "cheek_l"):
                a = np.array([r.rgb.get(k, (np.nan,) * 3) for r in seg], float)
                ok = np.all(np.isfinite(a), 1)
                if ok.mean() >= 0.8 and ok.sum() >= 3 * fps:
                    rgb_u[k] = np.stack([np.interp(tg, ts[ok], a[ok, c]) for c in range(3)], 1)
            s_pulse, p = analyze_pulse(rgb_u, fps_u, g.get("head_motion", 0.0) or 0.0)
            s_blend, bl = analyze_blend([r.blend for r in seg if r.blend is not None])
            s_rec, rc = analyze_recapture([r.moire for r in seg if r.moire is not None],
                                          [r.glare for r in seg if r.glare is not None])
            slow = self._slow = {"t": t_end, "start": seg[0].t, "geo": geo_sigs, "g": g, "pulse": s_pulse,
                                 "p": p, "fps_u": fps_u, "blend": s_blend, "bl": bl, "rec": s_rec, "rc": rc}
        g, p, fps_u, bl, rc = slow["g"], slow["p"], slow["fps_u"], slow["bl"], slow["rc"]
        for sg in [*slow["geo"], slow["pulse"], slow["blend"], slow["rec"]]:
            sg = Signal(**{k: v for k, v in sg.__dict__.items()})      # fresh copy: weights are adjusted below
            if sg.key in ("dynamics", "parallax", "jitter", "pulse"):
                sg.reliability *= size_rel
            signals.append(sg)

        # 8. trained deepfake detector
        with self.lock:
            mrecs = [m for m in self.model_recs if m[0] >= ts[0] - 0.5]
        s_model, md = analyze_model(mrecs, self.model_status, self.model_names)
        signals.append(s_model)

        for sg in signals:
            if sg.key in SYNTHESIS_KEYS:
                sg.group = "synthesis"
            if sg.key in MINOR_SYNTHESIS:          # evaluated near chance vs. real deepfakes
                sg.weight *= 0.45
                sg.decisive = False
        score, conf, reasons, groups = group_score(signals)

        # compact traces for the dashboard (time relative to now, seconds)
        ear = pre["ear"]
        k = max(1, n // 240)
        rel = np.round(ts - t_end, 2)
        det = {
            "backend": self.backend, "faces_on_screen": self.faces_on_screen,
            "fps": round(fps, 1), "frames": n, "tracked_s": round(dur, 1), "face_px": round(med_iod, 1),
            "blink": {"rate": b.get("rate"), "count": len(b.get("events", [])),
                      "events": [round(e[0] + ts[0] - t_end, 2) for e in b.get("events", [])],
                      "close_thr": b.get("close_thr"),
                      "ear": [[float(x), (None if not np.isfinite(y) else round(float(y), 4))]
                              for x, y in zip(rel[::k], ear[::k])]},
            "pulse": {"bpm": p.get("bpm"), "snr": p.get("snr"),
                      "wave": [round(float(v), 3) for v in (p.get("wave") or [])[-int(8 * fps_u):]],
                      "fps": fps_u},
            "geometry": {kk: g.get(kk) for kk in ("jitter", "spikes", "expression", "head_motion", "parallax_ratio")},
            "texture": {**bl, **rc},
            "model": {**{k: v for k, v in md.items() if k != "frames"},
                      "trace": [[round(float(t - t_end), 2), round(float(p), 3)] for t, p in md.get("frames", [])]},
            "groups": groups,
        }
        return {"status": "tracking", "score": score, "confidence": conf,
                "signals": [s.to_dict() for s in signals], "reasons": reasons, "details": det}


# --------------------------------------------------------------------------
# Analyses (pure functions over the tracked arrays — unit-testable)
# --------------------------------------------------------------------------
def analyze_blinks(ear_r, ear_l, fps):
    ear_r, ear_l = np.asarray(ear_r, float), np.asarray(ear_l, float)
    ear = (ear_r + ear_l) / 2.0
    valid = np.isfinite(ear)
    T = valid.sum() / fps
    out = {"events": [], "rate": None, "T": T, "base": None, "close_thr": None}
    if valid.sum() < max(10, 2 * fps):
        return Signal("blink", "Blink behaviour", "—", 0.5, 1.0, 0.0,
                      "Face tracked too briefly to measure blinking"), out

    e = ear.copy()
    if fps >= 20:
        e_f = e.copy()
        e_f[~valid] = np.nanmedian(e)
        e_f = sps.medfilt(e_f, 3)
        e[valid] = e_f[valid]
    base = float(np.nanpercentile(e[valid], 80))
    close_thr, open_thr = 0.75 * base, 0.88 * base
    out.update(base=base, close_thr=close_thr)

    events, in_blink, long_closed, start = [], False, False, 0
    for i in range(len(e)):
        if not valid[i]:
            in_blink = long_closed = False
            continue
        if long_closed:
            if e[i] > open_thr:
                long_closed = False
            continue
        if not in_blink and e[i] < close_thr:
            in_blink, start = True, i
        elif in_blink:
            if e[i] > open_thr:
                if (i - start) / fps <= 0.7:
                    events.append((start, i))
                in_blink = False
            elif (i - start) / fps > 0.7:
                in_blink, long_closed = False, True

    n = len(events)
    rate = 60.0 * n / T if T > 0 else 0.0
    out.update(events=[(s / fps, t / fps) for s, t in events], rate=rate)

    asym, depth = [], []
    bl, br = np.nanpercentile(ear_l, 80), np.nanpercentile(ear_r, 80)
    for s, t in events:
        a, b = max(0, s - 1), min(len(e), t + 2)
        dl = 1 - np.nanmin(ear_l[a:b]) / bl
        dr = 1 - np.nanmin(ear_r[a:b]) / br
        depth.append((dl + dr) / 2)
        asym.append(abs(dl - dr) / max(dl, dr, 1e-3))
    out.update(depth=float(np.median(depth)) if depth else None,
               asym=float(np.median(asym)) if asym else None)

    rel = clip01(T / 8.0)
    if n == 0:
        risk = 0.35 + 0.55 * clip01((T - 4.0) / 10.0)
        msg = f"No blinks in {T:.1f}s of tracked face — people blink 8–30×/min; photos and some deepfakes don't"
    elif rate < 4:
        risk, msg = 0.6, f"Very low blink rate ({rate:.1f}/min, {n} in {T:.1f}s) — below natural range"
    elif rate < 8:
        risk, msg = 0.3, f"Slightly low blink rate ({rate:.1f}/min) — borderline, common when reading a screen"
    elif rate <= 35:
        risk, msg = 0.05, f"Natural blink rate {rate:.1f}/min ({n} blinks in {T:.1f}s)"
    elif rate <= 50:
        risk, msg = 0.35, f"Unusually high blink rate ({rate:.1f}/min) — possible eye-region flicker"
    else:
        risk, msg = 0.6, f"Implausible blink rate ({rate:.1f}/min) — eye region is flickering, a deepfake artifact"

    if out["asym"] is not None and out["asym"] > 0.55 and n >= 2:
        risk = max(risk, 0.55)
        msg += f"; eyes close asymmetrically (asymmetry {out['asym']:.2f}) — eyes are rendered independently"
    elif out["depth"] is not None and out["depth"] < 0.3 and n >= 2:
        risk = max(risk, 0.4)
        msg += f"; blinks are shallow (depth {out['depth']:.2f}) — partial blinks are a face-swap artifact"
    return Signal("blink", "Blink behaviour", f"{rate:.1f} /min", risk, 1.0, rel, msg), out


def _similarity_residual(a, b):
    """RMS residual after best similarity transform a->b (both Nx2)."""
    M, _ = cv2.estimateAffinePartial2D(a.astype(np.float32), b.astype(np.float32), method=cv2.LMEDS)
    if M is None:
        return np.nan, None
    pred = a @ M[:, :2].T + M[:, 2]
    return float(np.sqrt(np.mean(np.sum((pred - b) ** 2, axis=1)))), M


def head_pose(lm, cam):
    """(yaw, pitch, roll) in degrees. cam = (focal, cx, cy) of the customer's (virtual) camera."""
    f, cx, cy = cam
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(POSE_3D, lm[POSE_IDX].astype(np.float64), K, np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return (np.nan, np.nan, np.nan)
    R, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(R)
    pitch, yaw, roll = angles
    return (yaw, pitch, roll)


def analyze_geometry(lms, iods, fps, pre):
    """Expression dynamics, 3-D parallax and landmark jitter from landmark tracks.

    pre: per-frame arrays computed at ingest — yaw, pitch (degrees), jit (landmark residual vs. the
    previous frame, % of IOD), mouth, brow, ear — NaN where no face was tracked.
    """
    idx = [i for i, l in enumerate(lms) if l is not None]
    out = {}
    na = lambda k, lab, why: Signal(k, lab, "—", 0.5, 1.0, 0.0, why)
    if len(idx) < max(10, fps):
        why = "Not enough frames with a tracked face"
        return [na("dynamics", "Expression dynamics", why), na("parallax", "3-D structure", why),
                na("jitter", "Landmark stability", why)], out
    yaw, pitch = pre["yaw"], pre["pitch"]

    # ---- landmark jitter (non-rigid residual between consecutive frames)
    present = np.array([l is not None for l in lms])
    consecutive = present & np.concatenate([[False], present[:-1]])
    res = pre["jit"][consecutive]
    res = res[np.isfinite(res)]
    res = res if len(res) else np.array([np.nan])
    jit = float(np.nanmedian(res))
    spikes = float(np.mean(res > max(3 * jit, 2.5))) if np.isfinite(jit) else 0.0
    out.update(jitter=jit, spikes=spikes)
    r_j = ramp(jit, 1.0, 3.0, 0.05, 0.85)
    r_j = max(r_j, ramp(spikes, 0.03, 0.15, 0.0, 0.7))
    if r_j < 0.35:
        msg_j = f"Landmarks move coherently with the head (residual jitter {jit:.2f}% of eye distance)"
    else:
        msg_j = (f"Facial landmarks flicker independently of head motion (jitter {jit:.2f}% IOD, "
                 f"{100 * spikes:.0f}% spike frames) — frame-by-frame synthesis artifact")
    s_jit = Signal("jitter", "Landmark stability", f"{jit:.2f}% IOD", r_j, 0.8,
                   clip01(len(res) / (3 * fps)), msg_j)

    # ---- expression dynamics (non-rigid facial motion)
    L = np.stack([lms[i] for i in idx])
    I = np.array([iods[i] for i in idx])
    mouth, brow, ear = pre["mouth"][idx], pre["brow"][idx], pre["ear"][idx]
    expr = float(100 * (np.std(mouth) + np.std(brow)) + 10 * np.std(ear))
    head_move = float(np.nanstd(yaw) + np.nanstd(pitch))
    out.update(expression=expr, head_motion=head_move)
    if expr < 0.35 and head_move < 0.6:
        r_d, msg_d = 0.9, (f"Face is completely static — no expression change or head movement "
                           f"(expression index {expr:.2f}) — consistent with a photo or frozen frame")
    elif expr < 0.35:
        r_d, msg_d = 0.7, (f"Head moves but the face never changes expression (index {expr:.2f}) "
                           f"— consistent with a printed photo being moved")
    elif expr < 0.7:
        r_d, msg_d = 0.3, f"Very little facial expression change (index {expr:.2f})"
    else:
        r_d, msg_d = 0.05, f"Natural non-rigid facial motion (expression index {expr:.2f}, head motion {head_move:.1f}°)"
    s_dyn = Signal("dynamics", "Expression dynamics", f"{expr:.2f}", r_d, 1.0, clip01(len(idx) / (4 * fps)), msg_d)

    # ---- 3-D parallax: is the face consistent with a flat plane?
    rng = np.random.default_rng(0)
    big, small = [], []
    ang = np.stack([yaw[idx], pitch[idx]], 1)
    cand = [(a, b) for a in range(0, len(idx), 2) for b in range(a + 2, len(idx), 3)]
    if len(cand) > PARALLAX_PAIRS:
        cand = [cand[k] for k in rng.choice(len(cand), PARALLAX_PAIRS, replace=False)]
    for a, b in cand:
        d = np.nanmax(np.abs(ang[a] - ang[b]))
        if not np.isfinite(d):
            continue
        if d >= 5.0 or d <= 1.0:
            src, dst = L[a][PARALLAX_IDX], L[b][PARALLAX_IDX]
            H, _ = cv2.findHomography(src, dst, 0)
            if H is None:
                continue
            proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
            e = np.sqrt(np.mean(np.sum((proj - dst) ** 2, 1))) / I[b]
            (big if d >= 5.0 else small).append(e)
    if len(big) < 3:
        max_rot = float(np.nanmax(np.nanmax(ang, 0) - np.nanmin(ang, 0))) if len(ang) else 0.0
        s_par = Signal("parallax", "3-D structure", "—", 0.4, 0.9, 0.0,
                       f"Head rotated only {max_rot:.1f}° — not enough to test 3-D depth "
                       f"(ask the customer to turn their head slightly)")
    else:
        raw_noise = float(np.median(small)) if small else 0.004
        noise = max(raw_noise, 0.004)
        ratio = float(np.median(big)) / noise
        out["parallax_ratio"] = ratio
        r_p = ramp(ratio, 1.3, 2.4, 0.85, 0.05)
        # a flickering landmark track makes the depth test meaningless
        noise_rel = ramp(100 * raw_noise, 1.0, 2.0, 1.0, 0.0)
        if r_p >= 0.5:
            msg_p = (f"When the head turns, landmarks move like a flat picture (parallax ratio {ratio:.2f}) "
                     f"— printed photo or screen held to the camera")
        else:
            msg_p = f"Nose/ear parallax confirms a real 3-D head (parallax ratio {ratio:.2f})"
        if noise_rel < 0.5:
            msg_p = "Landmarks too unstable to test 3-D depth reliably (see landmark stability)"
        s_par = Signal("parallax", "3-D structure", f"{ratio:.2f}×", r_p, 0.9,
                       clip01(len(big) / 12) * noise_rel, msg_p)

    return [s_dyn, s_par, s_jit], out


def _pos(rgb, fps):
    """Plane-Orthogonal-to-Skin rPPG (Wang et al., 2017), all sliding windows at once."""
    rgb = np.asarray(rgb, float)
    n = len(rgb)
    l = max(4, int(round(1.6 * fps)))
    H = np.zeros(n)
    if n < l:
        return H
    C = np.lib.stride_tricks.sliding_window_view(rgb, l, axis=0)       # (m, 3, l)
    mu = C.mean(2, keepdims=True)
    ok = np.all(mu[:, :, 0] > 0, 1)
    Cn = C / np.where(mu > 0, mu, 1.0)
    s0 = Cn[:, 1] - Cn[:, 2]                                             # P = [[0, 1, -1], [-2, 1, 1]]
    s1 = -2 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]
    h = s0 + (s0.std(1) / (s1.std(1) + 1e-9))[:, None] * s1
    h = (h - h.mean(1, keepdims=True)) * ok[:, None]
    m = len(h)
    for j in range(l):                                                   # overlap-add
        H[j:j + m] += h[:, j]
    return H


def _pulse_spectrum(x, fps):
    nfft = int(2 ** math.ceil(math.log2(max(len(x), 1) * 4)))
    nfft = max(nfft, 1024)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fps)
    return f, spec


def analyze_pulse(rgb_by_roi: Dict[str, np.ndarray], fps, motion_deg=0.0):
    """rgb_by_roi: uniformly-sampled (n, 3) RGB means per skin region."""
    out = {}
    series = {k: np.asarray(v, float) for k, v in rgb_by_roi.items() if len(v)}
    T = (len(next(iter(series.values()))) / fps) if series else 0.0
    if not series or T < 5 or fps < 8:
        return Signal("pulse", "Remote pulse (rPPG)", "—", 0.5, 0.7, 0.0,
                      "Needs ≥5 s of steadily tracked skin to look for a heartbeat"), out

    hi = min(3.0, 0.45 * fps)
    b, a = sps.butter(3, [0.7 / (fps / 2), hi / (fps / 2)], btype="band")

    def one(rgb):
        h = _pos(rgb, fps)
        if len(h) > 3 * max(len(a), len(b)):
            h = sps.filtfilt(b, a, sps.detrend(h))
        f, p = _pulse_spectrum(h, fps)
        band = (f >= 0.7) & (f <= hi)
        if not band.any():
            return None
        fpk = f[band][np.argmax(p[band])]
        wide = (f >= 0.7) & (f <= min(4.0, 0.49 * fps))
        sig = wide & ((np.abs(f - fpk) <= 0.1) | (np.abs(f - 2 * fpk) <= 0.2))
        snr = 10 * np.log10(p[sig].sum() / (p[wide & ~sig].sum() + 1e-12) + 1e-12)
        return {"bpm": 60 * fpk, "snr": float(snr), "wave": h}

    combo = sum(series.values()) / len(series)
    main = one(combo)
    per = {k: one(v) for k, v in series.items()}
    per = {k: v for k, v in per.items() if v is not None}
    if main is None:
        return Signal("pulse", "Remote pulse (rPPG)", "—", 0.5, 0.7, 0.0, "Pulse band not measurable"), out

    bpms = [v["bpm"] for v in per.values()]
    agree = (max(bpms) - min(bpms) <= 12) if len(bpms) >= 2 else True
    snr, bpm = main["snr"], main["bpm"]
    wave = main["wave"]
    wave = (wave - wave.mean()) / (wave.std() + 1e-9)
    out.update(bpm=float(bpm), snr=float(snr), agree=agree, wave=wave.tolist())

    rel = clip01((T - 4) / 6) * clip01(1.3 - motion_deg / 25.0)
    if snr > -1.0 and agree and 45 <= bpm <= 150:
        risk, msg = 0.05, f"Heartbeat detected in facial skin: {bpm:.0f} bpm (SNR {snr:.1f} dB, consistent across regions)"
    elif snr > -4.0:
        risk = 0.3 if agree else 0.45
        msg = f"Weak pulse signal ({bpm:.0f} bpm, SNR {snr:.1f} dB)" + ("" if agree else " — forehead and cheeks disagree")
    else:
        risk = 0.6 if agree else 0.7
        msg = (f"No coherent heartbeat in skin colour (SNR {snr:.1f} dB) — typical of synthetic faces and "
               f"replays, but heavy call compression can also hide it")
    # call compression can wash out the pulse, so rPPG alone never sets the floor
    return Signal("pulse", "Remote pulse (rPPG)", f"{bpm:.0f} bpm · {snr:.1f} dB", risk, 0.7, rel, msg,
                  decisive=False), out


def analyze_blend(ratios):
    if len(ratios) < 3:
        return Signal("blend", "Face-swap seam", "—", 0.4, 0.8, 0.0,
                      "Neck not visible — face-swap seam check skipped"), {}
    r = float(np.median(ratios))
    risk = ramp(r, 0.75, 0.4, 0.05, 0.8)
    if risk >= 0.5:
        msg = (f"Cheek skin is much smoother than neck skin (noise ratio {r:.2f}) — "
               f"swapped faces are re-rendered and blended in")
    else:
        msg = f"Face and neck share the same sensor noise (ratio {r:.2f}) — no blending seam"
    return Signal("blend", "Face-swap seam", f"{r:.2f}", risk, 0.8, clip01(len(ratios) / 8), msg), {"blend_ratio": r}


def analyze_recapture(moire, glare):
    if len(moire) < 3:
        return Signal("recapture", "Screen recapture", "—", 0.3, 0.6, 0.0,
                      "Face too small on screen for the recapture texture check"), {}
    m = float(np.median(moire))
    g = float(np.median(glare)) if glare else 0.0
    r_m = ramp(m, 3.2, 5.0, 0.05, 0.85)   # real faces ~2.2-2.8, recaptured screens 3.5-6.5
    r_g = ramp(g, 0.03, 0.12, 0.0, 0.6)
    risk = max(r_m, r_g)
    if r_m >= 0.5:
        msg = f"Periodic moiré peaks in the face texture (peak index {m:.1f}) — a screen is being held up to the customer's camera"
    elif r_g >= 0.4:
        msg = f"Large specular glare patches on the face ({100 * g:.1f}%) — possible screen or glossy print"
    else:
        msg = f"No screen-replay texture (moiré peak index {m:.1f}, glare {100 * g:.1f}%)"
    return Signal("recapture", "Screen recapture", f"{m:.1f} peak", risk, 0.6, clip01(len(moire) / 6), msg), \
        {"moire": m, "glare": g}


SYNTHESIS_KEYS = {"jitter", "blend", "model"}
MINOR_SYNTHESIS = {"jitter", "blend"}


def _conf_aggregate(p):
    """DeepGuard's clip aggregation: trust a clear fake majority / near-unanimous real, else mean."""
    p = np.asarray(p, float)
    fakes = p[p > 0.8]
    if len(fakes) > len(p) // 2.5:
        return float(fakes.mean())
    if np.count_nonzero(p < 0.2) > 0.9 * len(p):
        return float(p[p < 0.2].mean())
    return float(p.mean())


def analyze_model(results, status="ready", names=None):
    """results: [(t, P(fake) per checkpoint, face px)] from the trained detector."""
    lab = "AI deepfake detector"
    if not results:
        if status == "off":
            why = "Trained detector disabled"
        elif status == "loading":
            why = "Trained deepfake detector is loading…"
        elif str(status).startswith("unavailable"):
            why = "Trained deepfake detector not installed — run: python -m tools.download_models"
        else:
            why = "Waiting for face crops for the trained detector"
        return Signal("model", lab, "—", 0.5, 2.5, 0.0, why, group="synthesis"), {"status": status}
    P = np.stack([r[1] for r in results])                 # (n, checkpoints)
    per = [_conf_aggregate(P[:, k]) for k in range(P.shape[1])]   # clip-level P(fake) per checkpoint
    risk = calibration.risk("face_model", per, names)    # stacked + calibrated → fake score
    face_px = float(np.median([r[2] for r in results]))
    # the models were trained on ~100+ px faces; tiny faces are upscaled blur, so trust them less
    rel = clip01(len(results) / 8) * clip01((face_px - 40) / 60)
    labels = names or [f"#{i}" for i in range(len(per))]
    split = ", ".join(f"{labels[i]} {v:.2f}" for i, v in enumerate(per))
    if risk >= 0.65:
        msg = (f"Trained deepfake detector flags this face: fake score {risk:.2f} over {len(results)} frames "
               f"({split}) — face-swap / reenactment artifacts")
    elif risk >= 0.35:
        msg = f"Trained detector is unsure: fake score {risk:.2f} ({split})"
    else:
        msg = f"Trained detector sees a natural face: fake score {risk:.2f} over {len(results)} frames ({split})"
    if face_px < 80:
        msg += (f" — the face carries only ~{face_px:.0f} px of real detail (small tile or low-bandwidth call); "
                f"ask the customer to move closer or improve their connection for a surer result")
    return Signal("model", lab, f"{risk:.2f}", risk, 2.5, rel, msg, group="synthesis"), \
        {"status": status, "p": risk, "per_model": per, "n": len(results), "face_px": face_px, "risk": risk,
         "names": labels, "frames": [(r[0], calibration.risk("face_model", r[1], names)) for r in results]}


# --------------------------------------------------------------------------
# Offline entry point (video file) — same pipeline as the live stream
# --------------------------------------------------------------------------
def score_video(path: str, max_seconds: float = 20.0, target_fps: float = 15.0, model: Optional[str] = "sync",
                degrade=None) -> Tuple[float, List[str], dict]:
    """model: see FaceStream. degrade: optional fn(frame) -> frame, e.g. to simulate a compressed call."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps < 2 or src_fps > 240:
        src_fps = 30.0
    step = max(1, int(round(src_fps / target_fps)))
    fs = FaceStream(keep_s=max_seconds + 5, model=model)
    i = 0
    try:
        while i < max_seconds * src_fps:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                fs.process(i / src_fps, degrade(frame) if degrade else frame)
            i += 1
        res = fs.analyze_window(window_s=max_seconds)
    finally:
        fs.close()
        cap.release()
    if res["score"] is None:
        return 50.0, ["No face tracked long enough in this clip — defaulted to medium risk"], res
    return res["score"], res["reasons"], res
