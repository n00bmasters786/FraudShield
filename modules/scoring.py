"""
Shared evidence-aggregation helpers for the face and voice modules.

Each forensic check produces a `Signal`:
    risk         0..1   how suspicious the measurement is
    weight       >0     how much this check matters relative to the others
    reliability  0..1   how much we trust the measurement on THIS clip
                        (short clip, too little motion, noisy audio -> lower)

The module score is a reliability-weighted average of the risks, plus two
rules that make it behave like an analyst rather than a plain average:
  * a single strong, reliable red flag from a *decisive* check sets a
    floor (it can't be diluted away by many "looks fine" checks), and
  * several independent red flags compound.

Signals belong to a group. "liveness" checks answer *is a live person in
front of the camera* (photos, screens, replays); "synthesis" checks answer
*is the face / voice itself generated*. A deepfake passes liveness — it
blinks, turns and has a pulse-like texture — so the two groups are scored
separately and the module takes the worse of the two (`group_score`),
instead of letting a pile of clear liveness checks average a deepfake away.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple


@dataclass
class Signal:
    key: str
    label: str
    value: str            # short human-readable measurement, e.g. "16.4 /min"
    risk: float           # 0..1
    weight: float         # relative importance
    reliability: float    # 0..1
    message: str          # one-line explanation shown in the reasons panel
    decisive: bool = True # may a single strong reading of this check set a risk floor?
    group: str = "liveness"   # "liveness" or "synthesis"
    extra: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.reliability < 0.2:
            return "n/a"
        if self.risk >= 0.65:
            return "alert"
        if self.risk >= 0.35:
            return "watch"
        return "clear"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def ramp(x: float, x0: float, x1: float, r0: float, r1: float) -> float:
    """Linear interpolation of risk between (x0, r0) and (x1, r1), clamped."""
    if x1 == x0:
        return r1
    t = clip01((x - x0) / (x1 - x0))
    return float(r0 + t * (r1 - r0))


def aggregate(signals: List[Signal]) -> Tuple[float, float, List[str]]:
    """Return (score 0..100, confidence 0..1, ordered reasons)."""
    num = sum(s.weight * s.reliability * s.risk for s in signals)
    den = sum(s.weight * s.reliability for s in signals)
    total_w = sum(s.weight for s in signals) or 1.0

    if den < 1e-6:
        return 50.0, 0.0, ["Not enough usable evidence in this clip — defaulted to medium risk"]

    score = num / den

    strong = [s for s in signals if s.reliability >= 0.5 and s.risk >= 0.65]
    decisive = [s for s in strong if s.decisive]
    if decisive:
        # one reliable, decisive red flag => at least medium risk
        top = max(s.risk * min(1.0, s.reliability + 0.2) for s in decisive)
        score = max(score, 0.8 * top)
    if len(strong) >= 2:
        score += 0.08 * (len(strong) - 1)

    score = clip01(score)
    confidence = clip01(den / total_w)

    ordered = sorted(
        signals,
        key=lambda s: (s.reliability >= 0.2, s.weight * s.reliability * s.risk),
        reverse=True,
    )
    reasons = [s.message for s in ordered]
    return round(100.0 * score, 1), round(confidence, 2), reasons


def group_score(signals: List[Signal]) -> Tuple[Optional[float], float, List[str], dict]:
    """Score liveness and synthesis evidence separately; the module score is the worse group.

    Returns (score 0..100 or None if no usable evidence, confidence, reasons, per-group detail).
    """
    groups = {}
    for g in ("liveness", "synthesis"):
        sig = [s for s in signals if s.group == g]
        usable = sum(s.weight * s.reliability for s in sig)
        if sig and usable > 1e-6:
            sc, conf, _ = aggregate(sig)
            groups[g] = {"score": sc, "confidence": conf}
    _, _, reasons = aggregate(signals) if signals else (0, 0, [])
    if not groups:
        return None, 0.0, reasons, groups
    worst = max(groups, key=lambda g: groups[g]["score"])
    conf = max(groups[worst]["confidence"], max(v["confidence"] for v in groups.values()) * 0.8)
    return groups[worst]["score"], round(conf, 2), reasons, {**groups, "driver": worst}
