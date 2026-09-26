"""
Frame-level deepfake classifier on face crops.

Ensemble of two MS-EffGCViT-B0 checkpoints (EfficientNet-B0 backbone + global-
context ViT branches, 8.7 M params each) from DeepGuard:

  ff++          FaceForensics++ c23 — Deepfakes, FaceSwap, Face2Face, FaceShifter,
                NeuralTextures (AUC 0.997 in-domain)
  celeb_df_v2   Celeb-DF v2 — higher-quality face swaps

Each checkpoint is strong on its own domain and weaker across domains
(FF++ → Celeb-DF AUC 0.70 on the model card), which is why both are run and
stacked (see modules/calibration.py). Preprocessing reproduces the authors'
test pipeline exactly: face box + 20 % margin, longest side resized to the
model size, zero-padded square, ImageNet normalisation. Output: P(fake) per
checkpoint.

The set is configurable with FRAUDSHIELD_FACE_MODELS, e.g. adding the larger
"b5-ff++" (50 M params, 384 px — use a GPU) for more robustness to compression.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import MODELS_DIR, torch_device

log = logging.getLogger("fraudshield.face_model")

CHECKPOINTS = {       # name → (Hugging Face repo, variant)
    "b0-ff++": ("KoreaPeter/ms-eff-gcvit-deepfake-b0-ff-plus-plus", "b0"),
    "b0-celeb": ("KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2", "b0"),
    "b5-ff++": ("KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus", "b5"),
    "b5-celeb": ("KoreaPeter/ms-eff-gcvit-deepfake-b5-celeb-df-v2", "b5"),
}
DEFAULT_SET = "b0-ff++,b0-celeb"
VARIANTS = {          # architecture kwargs from DeepGuard's ms_eff_gcvit_b0 / _b5 builders
    "b0": dict(
        model_name="tf_efficientnet_b0.ns_jft_in1k", img_size=[224, 224],
        l_dim=24, h_dim=256, l_depths=[2, 2, 4, 2], h_depths=[4],
        l_windows=[7, 7, 14, 7], h_windows=[7], l_heads=[1, 2, 4, 8], h_heads=[4],
        l_ratio=[4, 4, 4, 4], h_ratio=[4], h_drop=0.05, l_attn_drop=0.05,
        l_drop_path=0.1, h_drop_path=0.05),
    "b5": dict(
        model_name="tf_efficientnet_b5.ns_jft_in1k", img_size=[384, 384],
        l_dim=48, h_dim=512, l_depths=[2, 2, 6, 2], h_depths=[6],
        l_windows=[12, 12, 24, 12], h_windows=[12], l_heads=[2, 4, 8, 16], h_heads=[16],
        l_ratio=[3, 3, 3, 3], h_ratio=[3], h_drop=0.1, l_attn_drop=0.1,
        l_drop_path=0.15, h_drop_path=0.1),
}
MARGIN = 0.2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def face_crop(frame_bgr: np.ndarray, box, margin: float = MARGIN) -> Optional[np.ndarray]:
    """RGB crop of box=(x, y, w, h) plus `margin` on every side, clipped to the frame."""
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    pw, ph = int(w * margin), int(h * margin)
    x0, y0 = max(int(x - pw), 0), max(int(y - ph), 0)
    x1, y1 = min(int(x + w + pw), W), min(int(y + h + ph), H)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def effective_scale(crop_rgb: np.ndarray) -> float:
    """Rough share of the crop's pixels that carry real detail (0.25..1).

    Measures how much edge energy survives a 2× down/up-scale: a sharp, native face
    loses about half, an upscaled low-bandwidth call stream loses little because it
    had no fine detail to begin with. Heuristic — used only to scale reliability.
    """
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    if min(h, w) < 24:
        return 1.0
    energy = lambda im: float(np.mean(np.abs(cv2.Laplacian(im, cv2.CV_32F)))) + 1e-6
    half = cv2.resize(g, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
    kept = energy(cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR)) / energy(g)
    return float(np.clip((1.0 - kept) / 0.45, 0.25, 1.0))


def preprocess(crop_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """LongestMaxSize(size) → centre PadIfNeeded(size, 0) → ImageNet normalise, CHW float32."""
    h, w = crop_rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    x = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


class FaceDeepfakeClassifier:
    """Process-wide singleton; `predict` is thread-safe (inference is serialised)."""

    _instance: Optional["FaceDeepfakeClassifier"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    @classmethod
    def get(cls, wait: bool = True) -> Optional["FaceDeepfakeClassifier"]:
        """The loaded classifier, or None if it can't be loaded (status says why)."""
        if cls._instance is not None:
            return cls._instance
        if not wait and cls._init_lock.locked():
            return None
        with cls._init_lock:
            if cls._instance is None and not cls.status.startswith("unavailable"):
                try:
                    cls.status = "loading"
                    cls._instance = cls()
                    cls.status = "ready"
                except Exception as e:  # torch / weights missing, no network, ...
                    cls.status = f"unavailable: {e}"
                    log.warning("face deepfake model unavailable: %s", e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        threading.Thread(target=cls.get, name="fs-face-model-load", daemon=True).start()

    def __init__(self, names: Optional[List[str]] = None):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from .deepguard import MultiScaleEffGCViT

        self.torch = torch
        self.device = torch_device()
        if self.device == "cpu":
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        names = names or [n.strip() for n in os.environ.get("FRAUDSHIELD_FACE_MODELS", DEFAULT_SET).split(",") if n.strip()]
        self.models: Dict[str, "torch.nn.Module"] = {}
        self.sizes: Dict[str, int] = {}
        for name in names:
            repo, variant = CHECKPOINTS[name]
            path = hf_hub_download(repo, "model.safetensors", local_dir=MODELS_DIR / repo.split("/")[1])
            state = {k.removeprefix("model."): v for k, v in load_file(path).items()}
            net = MultiScaleEffGCViT(**VARIANTS[variant])
            missing, unexpected = net.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{name}: checkpoint mismatch ({len(missing)} missing, {len(unexpected)} unexpected keys)")
            self.models[name] = net.eval().to(self.device)
            self.sizes[name] = VARIANTS[variant]["img_size"][0]
        self._lock = threading.Lock()
        self.names = list(self.models)
        self.name = f"MS-EffGCViT ensemble ({' + '.join(self.names)}) · {self.device}"
        self.predict([np.zeros((64, 64, 3), np.uint8)])       # compile kernels once

    def predict(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        """(n_crops, n_models) array of P(fake)."""
        if not crops_rgb:
            return np.zeros((0, len(self.models)), np.float32)
        torch = self.torch
        batches = {sz: torch.from_numpy(np.stack([preprocess(c, sz) for c in crops_rgb])).to(self.device)
                   for sz in set(self.sizes.values())}
        out = []
        with self._lock, torch.inference_mode():
            for name, net in self.models.items():
                out.append(torch.sigmoid(net(batches[self.sizes[name]])).float().cpu().numpy()[:, 0])
        return np.stack(out, 1)


class AsyncFaceScorer:
    """Per-session background scorer: keeps only the newest crop, never blocks the frame thread."""

    def __init__(self, on_result):
        self.on_result = on_result
        self._slot = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="fs-face-model", daemon=True)
        self._thread.start()

    def submit(self, t: float, crop_rgb: np.ndarray, meta: dict):
        with self._cv:
            self._slot = (t, crop_rgb, meta)
            self._cv.notify()

    def _run(self):
        clf = FaceDeepfakeClassifier.get()
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                t, crop, meta = self._slot
                self._slot = None
            if clf is None:
                continue
            t0 = time.perf_counter()
            p = clf.predict([crop])[0]
            self.on_result(t, p, {**meta, "ms": 1000 * (time.perf_counter() - t0)})

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()
