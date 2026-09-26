"""
Calibration of the trained detectors' outputs into risk.

A raw classifier probability is not a calibrated risk: an FF++-trained model
seeing a webcam face through a video-call codec is systematically shifted,
and the two face checkpoints are good at different things (FF++ catches the
DFD / FaceForensics family, Celeb-DF is near-perfect on real faces). So the
clip-level probabilities of a detector's checkpoints are combined by a small
logistic stacker

    risk = sigmoid( Σ wᵢ · logit(pᵢ) + b )

fitted on labelled clips (`python -m tools.evaluate --calibrate ...`). The
defaults below were fitted on the public DFD + Celeb-DF sample with call-like
degradation (see README); re-fit them on your own KYC recordings and they are
picked up from models/calibration.json without code changes.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Sequence

from .detectors import CALIBRATION_FILE

DEFAULTS: Dict[str, dict] = {
    # fitted by tools/evaluate.py on 197 clip-runs: 66 DFD + Celeb-DF v2 clips (real and fake),
    # each at full resolution, 720p-call and 360p-call quality
    "face_model": {"w": [0.394, 0.323], "b": 0.609},
    "voice_model": {"w": [1.0], "b": 0.0},
}
DEFAULT_FACE_SET = ["b0-ff++", "b0-celeb"]
_cache = None


def key(name: str, models=None) -> str:
    """Calibration entry for a detector *and* its checkpoint set, e.g. face_model[b0-ff++,b0-celeb,b5-ff++].
    The default set uses the plain name, so re-fitting another set never overwrites it."""
    if not models or (name == "face_model" and list(models) == DEFAULT_FACE_SET):
        return name
    return f"{name}[{','.join(models)}]"


def params() -> Dict[str, dict]:
    global _cache
    if _cache is None:
        _cache = {k: dict(v) for k, v in DEFAULTS.items()}
        try:
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k.split("[")[0] in DEFAULTS and "w" in v and "b" in v:
                    _cache[k] = {**v, "w": [float(x) for x in v["w"]], "b": float(v["b"])}
        except (OSError, ValueError):
            pass
    return _cache


def reload():
    global _cache
    _cache = None
    return params()


def logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1 - p))


def risk(name: str, probs: Sequence[float], models=None) -> float:
    """Calibrated risk 0..1 from one probability per checkpoint (`models` names the checkpoint set)."""
    c = params().get(key(name, models))
    if c is None or len(c["w"]) != len(probs):   # uncalibrated checkpoint set: plain logit mean
        c = {"w": [1.0 / len(probs)] * len(probs), "b": 0.0}
    z = sum(w * logit(p) for w, p in zip(c["w"], probs)) + c["b"]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
