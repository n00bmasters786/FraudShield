"""
Module C — Audio-visual lip-sync (cross-modal).

Compares the customer's lip opening (from the face tracker) with the loudness
envelope of the call audio. In a genuine call the two move together, give or
take a small network offset. They drift apart when:

  * the voice comes from somewhere else (voice changer / cloned voice played
    over a real or pre-recorded face, a second person speaking off-camera),
  * the video is a replay or a frozen / looped frame while someone talks,
  * a face-swap or lip-sync model renders the mouth late or smeared.

The measure is the peak normalised cross-correlation between the two signals
over lags of ±400 ms, computed on a common 25 Hz clock.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .scoring import Signal, aggregate, clip01, ramp
from .voice_module import _resample, _vad, SR

GRID_DT = 0.04          # 25 Hz common clock
MAX_LAG_S = 0.4
WINDOW_S = 10.0


def _envelope(y, sr, t_end):
    """(t, amplitude envelope, speech mask) of the audio at 100 Hz."""
    y = _resample(y.astype(np.float32), sr)
    sos = sps.butter(4, [250, 3500], btype="band", fs=SR, output="sos")
    y = sps.sosfilt(sos, y)
    hop, win = SR // 100, SR // 50
    n = 1 + max(0, len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, 1)) + 1e-9
    db = 20 * np.log10(rms)
    speech, *_ = _vad(db)
    t = t_end - (len(y) - (hop * np.arange(n) + win / 2)) / SR
    return t, np.sqrt(rms), speech


def analyze_sync(face_t, mouth, audio, sr, audio_t_end, window_s: float = WINDOW_S) -> dict:
    out = {"status": "waiting", "score": None, "confidence": 0.0, "signals": [], "reasons": [], "details": {}}
    na = lambda why: {**out, "signals": [Signal("sync", "Lip-sync", "—", 0.3, 0.9, 0.0, why).to_dict()],
                      "reasons": [why]}
    if audio is None or not sr or len(audio) < sr * 2 or audio_t_end is None:
        return na("Needs the call's system audio to compare lips with voice")
    if len(face_t) < 20 or np.isfinite(mouth).sum() < 20:
        return na("Needs a tracked face to compare lips with voice")

    at, env, speech = _envelope(audio, sr, audio_t_end)
    t0 = max(face_t[0], at[0], max(face_t[-1], at[-1]) - window_s)
    t1 = min(face_t[-1], at[-1])
    if t1 - t0 < 3.0:
        return na("Waiting for overlapping audio and video")
    grid = np.arange(t0, t1, GRID_DT)

    ok = np.isfinite(mouth)
    in_win = (face_t >= t0) & (face_t <= t1)
    valid_frac = float(ok[in_win].mean()) if in_win.any() else 0.0
    if valid_frac < 0.6:
        return na("Face tracking too patchy for a lip-sync check")
    m = np.interp(grid, face_t[ok], mouth[ok])
    e = np.interp(grid, at, env)
    sp = np.interp(grid, at, speech.astype(float)) > 0.5
    speech_s = float(sp.sum() * GRID_DT)
    out["details"] = {"speech_s": round(speech_s, 1)}
    if speech_s < 2.5:
        out["status"] = "listening"
        return na("Waiting for the customer to speak")

    mouth_move = float(np.std(m[sp]))
    rel = clip01((speech_s - 2.0) / 5.0) * clip01((valid_frac - 0.5) / 0.4)
    if mouth_move < 0.012:
        s = Signal("sync", "Lip-sync", f"lips {100 * mouth_move:.1f}%", 0.8, 0.9, rel,
                   f"Voice is heard for {speech_s:.0f}s but the lips barely move — the audio is not coming "
                   f"from this face (voice-over, replay or frozen video)")
        lag_ms, best, curve = None, None, []
    else:
        mz = (m - m.mean()) / (m.std() + 1e-9)
        ez = (e - e.mean()) / (e.std() + 1e-9)
        L = int(round(MAX_LAG_S / GRID_DT))
        n = len(grid)
        curve = []
        for k in range(-L, L + 1):          # k > 0: audio lags the lips
            a = mz[max(0, -k): n - max(0, k)]
            b = ez[max(0, k): n - max(0, -k)]
            curve.append(float(np.mean(a * b)) if len(a) > 10 else 0.0)
        kbest = int(np.argmax(curve))
        best, lag_ms = curve[kbest], (kbest - L) * GRID_DT * 1000
        risk = ramp(best, 0.10, 0.30, 0.7, 0.05)
        if risk >= 0.5:
            msg = (f"Lip movement doesn't follow the voice (correlation {best:.2f}) — audio and face may come "
                   f"from different sources")
        elif risk >= 0.3:
            msg = f"Weak lip-voice coupling (correlation {best:.2f}, offset {lag_ms:+.0f} ms)"
        else:
            msg = f"Lips move in step with the voice (correlation {best:.2f}, offset {lag_ms:+.0f} ms)"
        s = Signal("sync", "Lip-sync", f"r {best:.2f}", risk, 0.9, rel, msg, decisive=False)

    score, conf, reasons = aggregate([s])
    out.update(status="analyzing", score=score, confidence=conf, reasons=reasons, signals=[s.to_dict()])
    out["details"].update(corr=best, lag_ms=lag_ms, mouth_move=round(mouth_move, 4),
                          curve=[round(c, 3) for c in curve])
    return out
