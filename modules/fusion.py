"""
Module D — Real-time fusion engine.

Every analysis tick produces three module readings (face, voice, lip-sync),
each a 0-100 risk score with a 0-1 confidence. They are fused into one
deepfake-risk score:

  * confidence-aware weighted average — a module that has seen little
    evidence counts less, a module with nothing to say is left out;
  * analyst rule — one confident module in the red can't be averaged away;
  * asymmetric smoothing — risk rises quickly (τ≈2.5 s) and decays slowly
    (τ≈8 s), so a burst of artifacts isn't forgotten a second later;
  * verdict with hysteresis, and a calibrating state until there is enough
    evidence to say anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MODULE_WEIGHTS = {"face": 0.50, "voice": 0.30, "sync": 0.20}

GENUINE_BELOW = 35.0
DEEPFAKE_FROM = 65.0
HYSTERESIS = 4.0
MIN_CONFIDENCE = 0.18
MIN_EVIDENCE_S = 5.0

VERDICTS = {
    "idle": ("Idle", "Share your screen to start monitoring the video KYC call."),
    "searching": ("Looking for a face", "Bring the customer's video tile into view — FraudShield is scanning the screen."),
    "calibrating": ("Calibrating", "Collecting evidence — keep the customer on camera and speaking."),
    "genuine": ("Likely genuine", "No deepfake indicators. Continue the KYC."),
    "suspicious": ("Suspicious", "Run a liveness challenge before proceeding."),
    "deepfake": ("Likely deepfake", "Stop the KYC and escalate to the fraud team."),
}


@dataclass
class Reading:
    score: Optional[float]      # 0..100, None = module has nothing to say
    confidence: float           # 0..1


def fuse(readings: Dict[str, Reading]) -> Tuple[Optional[float], float]:
    """(raw fused score 0..100 or None, overall confidence 0..1)."""
    active = {k: r for k, r in readings.items() if r.score is not None and r.confidence > 0.05}
    if not active:
        return None, 0.0
    w = {k: MODULE_WEIGHTS[k] * (0.35 + 0.65 * r.confidence) for k, r in active.items()}
    score = sum(w[k] * r.score for k, r in active.items()) / sum(w.values())
    for r in active.values():
        if r.score >= DEEPFAKE_FROM and r.confidence >= 0.5:
            score = max(score, 0.85 * r.score)
    conf = sum(MODULE_WEIGHTS[k] * r.confidence for k, r in active.items()) / sum(MODULE_WEIGHTS.values())
    return float(min(100.0, max(0.0, score))), float(min(1.0, conf))


class LiveFusion:
    def __init__(self, tau_up: float = 2.5, tau_down: float = 8.0):
        self.tau_up, self.tau_down = tau_up, tau_down
        self.reset()

    def reset(self):
        self.value: Optional[float] = None
        self.verdict = "idle"
        self._t: Optional[float] = None
        self.evidence_s = 0.0

    def update(self, t: float, raw: Optional[float], conf: float, face_present: bool) -> dict:
        dt = 1.0 if self._t is None else max(0.05, min(5.0, t - self._t))
        self._t = t
        if raw is not None and conf >= MIN_CONFIDENCE:
            self.evidence_s += dt
            if self.value is None:
                self.value = raw
            else:
                tau = self.tau_up if raw > self.value else self.tau_down
                self.value += (1 - math.exp(-dt / tau)) * (raw - self.value)

        if self.value is None or self.evidence_s < MIN_EVIDENCE_S:
            self.verdict = "calibrating" if face_present or raw is not None else "searching"
        else:
            v, prev = self.value, self.verdict
            lo = GENUINE_BELOW + (HYSTERESIS if prev == "genuine" else -HYSTERESIS if prev == "suspicious" else 0)
            hi = DEEPFAKE_FROM + (-HYSTERESIS if prev == "deepfake" else HYSTERESIS if prev == "suspicious" else 0)
            self.verdict = "genuine" if v < lo else "deepfake" if v >= hi else "suspicious"

        label, action = VERDICTS[self.verdict]
        return {"score": None if self.value is None else round(self.value, 1),
                "raw": None if raw is None else round(raw, 1),
                "confidence": round(conf, 2), "verdict": self.verdict, "label": label, "action": action,
                "evidence_s": round(self.evidence_s, 1)}
