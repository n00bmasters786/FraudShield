"""
Module B — Voice liveness & anti-spoofing.

Pure NumPy/SciPy signal forensics (librosa is only used, when present, to
decode compressed formats). Live, `VoiceStream` keeps a rolling buffer of the
screen-share's system audio (i.e. the customer's side of the video call) and
`analyze(..., channel="call")` re-scores the last ~12 s. Eight checks:

  check                      catches
  -------------------------  ---------------------------------------------
  1. Digital silence         TTS / generated files: pauses are exact
                             zeros, no room or microphone noise floor
  2. Pitch expressiveness    monotone TTS (semitone spread of F0)
  3. Pitch micro-jitter      vocoder over-smooth F0 contours (humans have
                             cycle-to-cycle irregularity)
  4. Voice periodicity (HNR) "too clean" synthetic voicing
  5. Rhythm & pauses         machine-regular syllable timing, speech with
                             no natural pauses
  6. Bandwidth cut-off       audio generated at 16/22 kHz then up-sampled
                             (sharp brick-wall spectrum edge)
  7. Missing fundamental     replay through a phone / laptop speaker, which
                             can't reproduce the voice's lowest harmonic
  8. Breathing               no breath intakes between phrases

Call audio has already been through the conferencing app's noise gate, codec
and noise suppression, which fake exactly what checks 1, 6 and 8 look for
(digital silence, a band-limit wall, no breaths). With channel="call" those
checks are reported as not applicable instead of flagging every genuine
customer, and the periodicity / replay checks are softened.

Offline:  score_audio(path) -> (score_0_100, reasons, details)
"""

from __future__ import annotations

import threading
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

from . import calibration
from .scoring import Signal, clip01, group_score, ramp

SR = 16000
HOP = 160          # 10 ms
WIN = 400          # 25 ms
MODEL_SEG_S = 6.0  # trained detector window (shorter tails ≥ 2 s are used too)
MODEL_EVERY_S = 2.0


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_audio(path: str, max_seconds: float = 30.0):
    """Return (mono float32 at native rate, native sr)."""
    y, sr = None, None
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(1)
    except Exception:
        pass
    if y is None:
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, mono=True)
        except Exception:
            pass
    if y is None:
        sr, y = wavfile.read(path)
        if y.dtype.kind == "i":
            y = y.astype(np.float32) / float(np.iinfo(y.dtype).max)
        elif y.dtype.kind == "u":
            y = (y.astype(np.float32) - 128.0) / 128.0
        y = y.astype(np.float32)
        if y.ndim == 2:
            y = y.mean(1)
    y = np.asarray(y, np.float32)[: int(max_seconds * sr)]
    return y, int(sr)


def _resample(y, sr, target=SR):
    if sr == target:
        return y
    from math import gcd
    g = gcd(sr, target)
    return sps.resample_poly(y, target // g, sr // g).astype(np.float32)


def _frames(x, length, hop):
    if len(x) < length:
        x = np.pad(x, (0, length - len(x)))
    n = 1 + (len(x) - length) // hop
    return np.lib.stride_tricks.as_strided(x, (n, length), (x.strides[0] * hop, x.strides[0]))


# --------------------------------------------------------------------------
# Core features
# --------------------------------------------------------------------------
def _rms_db(y):
    fr = _frames(y, WIN, HOP)
    return 20 * np.log10(np.sqrt(np.mean(fr.astype(np.float64) ** 2, 1)) + 1e-10)


def _vad(db):
    floor = np.percentile(db, 5)
    peak = np.percentile(db, 98)
    thr = max(floor + 10.0, peak - 35.0)
    sp = db > thr
    # fill short gaps (<120 ms) and drop blips (<50 ms)
    sp = sps.medfilt(sp.astype(float), 5) > 0.5
    runs = _runs(sp)
    for s, e, v in runs:
        if not v and (e - s) < 12 and s > 0 and e < len(sp):
            sp[s:e] = True
    for s, e, v in _runs(sp):
        if v and (e - s) < 5:
            sp[s:e] = False
    return sp, thr, floor, peak


def _runs(mask):
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((start, i, bool(mask[start])))
            start = i
    return out


def yin(y, sr=SR, fmin=65.0, fmax=420.0, w=WIN, hop=HOP):
    """Vectorised YIN. Returns (f0 Hz or nan, aperiodicity 0..1) per hop."""
    tau_max = int(sr / fmin)
    tau_min = int(sr / fmax)
    L = w + tau_max
    fr = _frames(np.pad(y, (0, L)), L, hop).astype(np.float64)
    n = fr.shape[0]
    nfft = 1 << int(np.ceil(np.log2(L + w)))
    A = np.fft.rfft(fr[:, :w], nfft)
    B = np.fft.rfft(fr, nfft)
    r = np.fft.irfft(np.conj(A) * B, nfft)[:, : tau_max + 1]
    sq = np.cumsum(fr ** 2, 1)
    sq = np.concatenate([np.zeros((n, 1)), sq], 1)
    E0 = sq[:, w][:, None]
    tau = np.arange(tau_max + 1)
    Et = sq[:, tau + w] - sq[:, tau]
    d = np.maximum(E0 + Et - 2 * r, 0)
    cm = np.cumsum(d[:, 1:], 1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] * tau[1:] / (cm + 1e-12)

    f0 = np.full(n, np.nan)
    ap = np.ones(n)
    seg = dn[:, tau_min: tau_max]
    silent = E0[:, 0] < 1e-8 * w          # digital silence: YIN is 0/0 there
    for i in range(n):
        if silent[i]:
            continue
        s = seg[i]
        below = np.where(s < 0.15)[0]
        if below.size:
            k = below[0]
            while k + 1 < len(s) and s[k + 1] < s[k]:
                k += 1
        else:
            k = int(np.argmin(s))
        if k == 0 or k >= len(s) - 1:     # pinned to the search edge -> unreliable
            continue
        ap[i] = s[k]
        if 0 < k < len(s) - 1:   # parabolic refinement
            a, b, c = s[k - 1], s[k], s[k + 1]
            den = a - 2 * b + c
            off = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
        else:
            off = 0.0
        f0[i] = sr / (k + tau_min + off)
    return f0, ap


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def analyze(y_native, sr_native, channel: str = "file"):
    """channel: "file" for a recorded clip, "call" for conferencing-app audio."""
    call = channel == "call"
    y = _resample(y_native, sr_native)
    y = y - np.mean(y)
    dur = len(y) / SR
    signals: List[Signal] = []
    details = {"duration_s": round(dur, 2), "sr_native": sr_native}
    if dur < 1.0 or np.max(np.abs(y)) < 1e-4:
        s = Signal("input", "Input quality", f"{dur:.1f}s", 0.5, 1.0, 1.0,
                   "Clip is empty or shorter than 1 s — cannot assess voice liveness")
        return [s], details

    db = _rms_db(y)
    t = np.arange(len(db)) * HOP / SR
    sp, thr, floor, peak = _vad(db)
    details.update(t=t.round(3).tolist(), energy_db=np.round(db, 1).tolist(), speech=sp.tolist(),
                   vad_thr=float(thr), floor_db=float(floor), peak_db=float(peak))
    speech_s = sp.sum() * HOP / SR
    snr = peak - floor
    clip_frac = float(np.mean(np.abs(y_native) > 0.995))
    qual_rel = clip01((snr - 8) / 12) * clip01(1 - 20 * clip_frac)
    details.update(speech_s=round(float(speech_s), 2), snr_db=round(float(snr), 1), clipping=clip_frac)

    if speech_s < 1.5:
        signals.append(Signal("input", "Input quality", f"{speech_s:.1f}s speech", 0.5, 1.0, 1.0,
                              f"Only {speech_s:.1f}s of speech detected — need ≥3 s for a confident verdict"))
        return signals, details
    msg = f"{speech_s:.1f}s of speech, {snr:.0f} dB above the noise floor"
    if qual_rel < 0.6:
        msg += " — noisy/clipped recording lowers confidence"
    signals.append(Signal("input", "Input quality", f"{snr:.0f} dB SNR", 0.05, 0.4, 1.0, msg))

    # ---------- 1. digital silence / noise floor
    pauses = ~sp
    raw_fr = _frames(y_native, max(1, int(0.025 * sr_native)), max(1, int(0.010 * sr_native)))
    raw_db = 20 * np.log10(np.sqrt(np.mean(raw_fr.astype(np.float64) ** 2, 1)) + 1e-12)
    m = min(len(raw_db), len(pauses))
    pause_db = raw_db[:m][pauses[:m]]
    if call:
        signals.append(Signal("silence", "Noise floor", "—", 0.1, 1.0, 0.0,
                              "Call apps noise-gate pauses, so digital silence is normal on call audio — check disabled"))
    elif pause_db.size >= 20:
        dig = float(np.mean(pause_db < -85))
        nf = float(np.median(pause_db))
        details["pause_floor_db"] = nf
        risk = ramp(dig, 0.15, 0.6, 0.05, 0.85)
        if risk >= 0.5:
            msg = (f"{100 * dig:.0f}% of pause frames are pure digital silence (below −85 dBFS) — "
                   f"no microphone/room noise, typical of a generated audio file")
        else:
            msg = f"Natural microphone noise floor in pauses ({nf:.0f} dBFS)"
        signals.append(Signal("silence", "Noise floor", f"{nf:.0f} dBFS", risk, 1.0, 1.0, msg))
    else:
        signals.append(Signal("silence", "Noise floor", "—", 0.3, 1.0, 0.0,
                              "No pauses long enough to measure the noise floor"))

    # ---------- pitch
    f0, ap = yin(y)
    n = min(len(f0), len(sp))
    f0, ap = f0[:n], ap[:n]
    voiced = sp[:n] & (ap < 0.25) & np.isfinite(f0)
    f0v = np.where(voiced, f0, np.nan)
    # octave-jump cleanup against a running median
    med = np.nanmedian(f0v) if voiced.any() else np.nan
    if np.isfinite(med):
        for k in range(n):
            if voiced[k]:
                r = f0v[k] / med
                if r > 1.8:
                    f0v[k] /= 2
                elif r < 0.55:
                    f0v[k] *= 2
    details["f0"] = [None if not np.isfinite(v) else round(float(v), 1) for v in f0v]
    nv = int(voiced.sum())
    details["voiced_s"] = round(nv * HOP / SR, 2)

    if nv < 60:
        for key, lab in (("pitch", "Pitch expressiveness"), ("jitter", "Pitch micro-jitter"),
                         ("hnr", "Voice periodicity")):
            signals.append(Signal(key, lab, "—", 0.4, 1.0, 0.0, f"{lab}: too little voiced speech"))
    else:
        st = 12 * np.log2(f0v[voiced] / med)
        q25, q75 = np.percentile(st, [25, 75])
        spread = float((q75 - q25) / 1.349)          # robust std (outlier-proof)
        details["f0_median"] = float(med)
        details["pitch_spread_st"] = spread
        r_p = ramp(spread, 0.9, 2.0, 0.8, 0.05)
        if r_p >= 0.5:
            msg = f"Monotone delivery: pitch varies only ±{spread:.1f} semitones — typical of basic TTS"
        elif r_p >= 0.3:
            msg = f"Somewhat flat intonation (±{spread:.1f} semitones)"
        else:
            msg = f"Natural intonation: pitch moves ±{spread:.1f} semitones around {med:.0f} Hz"
        signals.append(Signal("pitch", "Pitch expressiveness", f"±{spread:.1f} st", r_p, 0.9,
                              clip01(nv / 200) * qual_rel, msg, decisive=False))

        # micro-jitter: frame-to-frame F0 change inside voiced runs, minus trend
        cents = []
        for s, e, v in _runs(voiced):
            if v and e - s >= 6:
                c = 1200 * np.log2(f0v[s:e])
                resid = c - sps.savgol_filter(c, min(7, (e - s) // 2 * 2 - 1), 2)
                cents.extend(np.abs(np.diff(resid)))
        jit = float(np.median(cents)) if cents else np.nan
        details["micro_jitter_cents"] = jit
        if np.isfinite(jit):
            r_j = ramp(jit, 1.0, 3.0, 0.75, 0.05)
            if r_j >= 0.5:
                msg = f"Pitch contour is unnaturally smooth (micro-jitter {jit:.1f} cents) — vocoder-generated voices lack vocal-fold irregularity"
            else:
                msg = f"Natural vocal-fold micro-jitter ({jit:.1f} cents per 10 ms)"
            signals.append(Signal("jitter", "Pitch micro-jitter", f"{jit:.1f} ¢", r_j, 0.6,
                                  clip01(len(cents) / 150) * qual_rel, msg, decisive=False))

        a = np.clip(ap[voiced], 1e-4, 0.999)
        hnr = float(np.median(10 * np.log10((1 - a) / a)))
        details["hnr_db"] = hnr
        # call noise-suppression cleans voices up, so "too clean" starts higher there
        r_h = ramp(hnr, 26.0, 34.0, 0.05, 0.6) if call else ramp(hnr, 22.0, 30.0, 0.05, 0.6)
        if r_h >= 0.4:
            msg = f"Voicing is unusually clean (HNR {hnr:.0f} dB) — no breathiness or aperiodicity"
        else:
            msg = f"Natural voice periodicity (HNR {hnr:.0f} dB)"
        signals.append(Signal("hnr", "Voice periodicity", f"{hnr:.0f} dB", r_h, 0.3 if call else 0.5,
                              clip01(nv / 200) * qual_rel, msg, decisive=False))

    # ---------- 5. rhythm & pauses
    first = np.argmax(sp) if sp.any() else 0
    last = len(sp) - np.argmax(sp[::-1]) if sp.any() else 0
    inner = sp[first:last]
    runs = _runs(inner) if len(inner) else []
    gaps = [(e - s) * HOP / SR for s, e, v in runs if not v and (e - s) * HOP / SR >= 0.15]
    speech_ratio = float(inner.mean()) if len(inner) else 0.0
    env = sps.savgol_filter(10 ** (db / 20), 9, 2)
    pk, _ = sps.find_peaks(np.where(sp, env, 0), distance=10, prominence=0.15 * np.max(env))
    ioi = np.diff(pk) * HOP / SR
    ioi = ioi[(ioi > 0.08) & (ioi < 0.6)]
    cv = float(np.std(ioi) / np.mean(ioi)) if len(ioi) >= 6 else np.nan
    rate = len(pk) / max(speech_s, 1e-3)
    details.update(pauses=len(gaps), speech_ratio=speech_ratio, rhythm_cv=cv, syll_rate=rate,
                   peaks_t=(pk * HOP / SR).round(3).tolist())
    rel_r = clip01((speech_s - 2) / 6) * qual_rel
    risks, notes = [], []
    if np.isfinite(cv):
        rr = ramp(cv, 0.15, 0.32, 0.7, 0.05)
        risks.append(rr)
        if rr >= 0.4:
            notes.append(f"syllable timing is machine-regular (CV {cv:.2f})")
    if speech_s > 6 and len(gaps) == 0:
        risks.append(0.6)
        notes.append(f"{speech_s:.0f}s of speech with no pauses at all")
    elif speech_ratio > 0.95 and speech_s > 6:
        risks.append(0.45)
        notes.append(f"speech occupies {100 * speech_ratio:.0f}% of the utterance")
    if len(gaps) >= 3:
        gcv = float(np.std(gaps) / np.mean(gaps))
        details["pause_cv"] = gcv
        if gcv < 0.2:
            risks.append(0.5)
            notes.append(f"pauses are all the same length (CV {gcv:.2f})")
    r_r = max(risks) if risks else 0.3
    if notes:
        msg = "Rhythm: " + "; ".join(notes) + " — synthetic speech pattern"
    else:
        msg = (f"Natural speech rhythm ({len(gaps)} pauses, timing CV "
               f"{cv:.2f})" if np.isfinite(cv) else f"Natural pausing ({len(gaps)} pauses)")
    signals.append(Signal("rhythm", "Rhythm & pauses", f"CV {cv:.2f}" if np.isfinite(cv) else f"{len(gaps)} pauses",
                          r_r, 0.7, rel_r if risks else 0.0, msg, decisive=False))

    # ---------- 6. bandwidth cut-off (needs native rate > 16 kHz)
    if call:
        signals.append(Signal("bandwidth", "Bandwidth cut-off", "—", 0.2, 0.6, 0.0,
                              "Call codecs band-limit audio on purpose — up-sampling test disabled on call audio"))
    elif sr_native >= 22050:
        nfft = 2048
        Y = _frames(y_native, nfft, nfft // 2) * np.hanning(nfft)
        spec_db = 10 * np.log10(np.mean(np.abs(np.fft.rfft(Y, axis=1)) ** 2, 0) + 1e-14)
        f = np.fft.rfftfreq(nfft, 1 / sr_native)
        sm = np.convolve(spec_db, np.ones(5) / 5, "same")
        ref = np.percentile(sm[(f > 300) & (f < 3000)], 90)
        cutoff, sharp = None, 0.0
        for fc in np.arange(5000, min(15000, 0.45 * sr_native - 2500), 100):
            below = np.median(sm[(f >= fc - 1000) & (f < fc)])
            above = np.median(sm[(f >= fc + 300) & (f < fc + 2300)])
            # a wall = real content up to fc, then a >30 dB cliff that stays down
            if below > ref - 40 and below - above > 30:
                cutoff, sharp = float(fc), float(below - above)
                break
        details["cutoff_hz"] = cutoff
        if cutoff is not None:
            r_b = 0.5 if 7500 <= cutoff <= 12000 else 0.3
            msg = (f"Spectrum ends abruptly at {cutoff / 1000:.1f} kHz ({sharp:.0f} dB wall) although the file is "
                   f"{sr_native / 1000:.1f} kHz — generated at a lower rate and up-sampled (or passed through a narrow-band codec)")
            signals.append(Signal("bandwidth", "Bandwidth cut-off", f"{cutoff / 1000:.1f} kHz", r_b, 0.6, 0.9, msg))
        else:
            signals.append(Signal("bandwidth", "Bandwidth cut-off", "full band", 0.05, 0.6, 0.9,
                                  f"Full-band spectrum up to {sr_native / 2000:.0f} kHz — no up-sampling wall"))
    else:
        signals.append(Signal("bandwidth", "Bandwidth cut-off", "—", 0.2, 0.6, 0.0,
                              f"File is {sr_native / 1000:.0f} kHz — bandwidth test needs ≥22 kHz audio"))

    # ---------- 7. missing fundamental (loudspeaker replay)
    if nv >= 60 and np.isfinite(med) and med < 260:
        nfft = 2048
        win = np.hanning(nfft)
        h12 = []
        centers = np.where(voiced)[0][::3]
        for c in centers:
            s0 = c * HOP
            seg = y[s0: s0 + nfft]
            if len(seg) < nfft:
                continue
            S = np.abs(np.fft.rfft(seg * win))
            f0c = f0v[c]
            b1, b2 = int(round(f0c * nfft / SR)), int(round(2 * f0c * nfft / SR))
            if b1 < 3:
                continue
            h1 = S[b1 - 1: b1 + 2].max()
            h2 = S[b2 - 1: b2 + 2].max()
            h12.append(20 * np.log10((h1 + 1e-9) / (h2 + 1e-9)))
        if len(h12) >= 15:
            h = float(np.median(h12))
            details["h1_h2_db"] = h
            r_m = ramp(h, -8.0, -16.0, 0.05, 0.75)
            if r_m >= 0.5:
                msg = (f"Fundamental harmonic is suppressed (H1−H2 {h:.0f} dB) — voice is probably being "
                       f"played through a small loudspeaker (replay attack)")
            else:
                msg = f"Low-frequency voice harmonics intact (H1−H2 {h:+.0f} dB) — live microphone capture"
            signals.append(Signal("replay", "Loudspeaker replay", f"{h:+.0f} dB", r_m, 0.7,
                                  clip01(len(h12) / 60) * qual_rel * (0.6 if call else 1.0), msg))
    if not any(s.key == "replay" for s in signals):
        signals.append(Signal("replay", "Loudspeaker replay", "—", 0.3, 0.7, 0.0,
                              "Replay check needs ≥1 s of voiced speech with a low/mid-pitched voice"))

    # ---------- 8. breathing
    long_gaps = [(s + first, e + first) for s, e, v in runs if not v and (e - s) >= 20]

    def _hissy(a0, a1):
        x = y[a0 * HOP:a1 * HOP]
        if len(x) < 5 * HOP:
            return False
        P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + 1e-12
        ff = np.fft.rfftfreq(len(x), 1 / SR)
        band = ff > 100
        flat = np.exp(np.mean(np.log(P[band]))) / np.mean(P[band])
        cent = np.sum(ff * P) / np.sum(P)
        return flat > 0.05 and cent > 1000               # noisy, hissy, not voiced

    breaths = 0
    # (a) faint breath left inside a pause
    for s, e in long_gaps:
        active = np.where(db[s:e] > floor + 4)[0]
        if len(active) >= 5 and _hissy(s + active[0], s + active[-1] + 1):
            breaths += 1
    # (b) louder breath the VAD kept as a short "speech" segment with no voicing
    vmask = np.zeros(len(sp), bool)
    vmask[:len(voiced)] = voiced
    for s, e, v in _runs(sp):
        if not v:
            continue
        if 10 <= e - s <= 70 and vmask[s:e].mean() < 0.15 and _hissy(s, e):
            breaths += 1
            continue
        # (c) quiet unvoiced lead-in fused onto the start of a phrase
        vk = np.where(vmask[s:e])[0]
        lead = vk[0] if vk.size else 0
        if lead >= 8 and db[s:s + lead].max() < peak - 20 and _hissy(s, s + lead):
            breaths += 1
    breaths = min(breaths, max(len(long_gaps), 0))
    details["breaths"] = breaths
    if call:
        signals.append(Signal("breath", "Breathing", f"{breaths}/{len(long_gaps)}", 0.2, 0.4, 0.0,
                              "Call noise suppression strips breath sounds — check disabled on call audio"))
    elif speech_s >= 8 and len(long_gaps) >= 2:
        if breaths == 0:
            r_b, msg = 0.35, f"No breath intakes in {len(long_gaps)} phrase pauses — speakers normally breathe between phrases"
        else:
            r_b, msg = 0.05, f"Breath intakes detected in {breaths} of {len(long_gaps)} phrase pauses"
        signals.append(Signal("breath", "Breathing", f"{breaths}/{len(long_gaps)}", r_b, 0.4, qual_rel, msg,
                              decisive=False))
    else:
        signals.append(Signal("breath", "Breathing", "—", 0.2, 0.4, 0.0,
                              "Clip too short for breath analysis (needs ≥8 s with phrase pauses)"))
    return signals, details


def _voice_model():
    from .detectors.voice_classifier import VoiceDeepfakeClassifier
    return VoiceDeepfakeClassifier


def speech_segments(y16, speech_mask, seg_s=MODEL_SEG_S, max_n=None, min_s=2.0):
    """Non-overlapping windows of up to seg_s (newest first, ≥ min_s long) that are ≥ 40 % speech."""
    n = int(seg_s * SR)
    out = []
    end = len(y16)
    while end >= int(min_s * SR) and (max_n is None or len(out) < max_n):
        start = max(0, end - n)
        m = speech_mask[start // HOP: end // HOP] if len(speech_mask) else []
        if len(m) and np.mean(m) >= 0.4:
            out.append(y16[start:end])
        end = start
    return out


def model_signal(probs, status="ready", quality=1.0) -> Tuple[Signal, dict]:
    """Clip-level signal from the trained detector's per-segment P(fake)."""
    lab = "AI voice-clone detector"
    if not len(probs):
        if str(status).startswith("unavailable"):
            why = "Trained voice detector not installed — run: python -m tools.download_models"
        elif status == "loading":
            why = "Trained voice detector is loading…"
        else:
            why = "Waiting for a few seconds of speech for the trained voice detector"
        return Signal("voice_model", lab, "—", 0.5, 2.0, 0.0, why, group="synthesis"), {"status": status}
    p = float(np.mean(probs))
    risk = calibration.risk("voice_model", [p])
    rel = clip01(len(probs) / 3) * quality
    # high precision, low recall: on modern commercial TTS it caught 3/18 clones with no false alarms,
    # so a "fake" verdict is strong evidence but a "clear" one is weak
    if risk < 0.35:
        rel *= 0.3
    if risk >= 0.65:
        msg = f"Trained detector hears synthetic / cloned speech: P(fake) {p:.2f} over {len(probs)} segments"
    elif risk >= 0.35:
        msg = f"Trained voice detector is unsure: P(fake) {p:.2f}"
    else:
        msg = (f"Trained detector hears no synthesis artifacts: P(fake) {p:.2f} over {len(probs)} segments "
               f"(it misses many modern voice clones, so this is weak evidence)")
    return Signal("voice_model", lab, f"P {p:.2f}", risk, 2.0, rel, msg, group="synthesis"), \
        {"status": status, "p": p, "n": len(probs), "risk": risk}


def score_audio(path: str, model: bool = True) -> Tuple[float, List[str], dict]:
    y, sr = load_audio(path)
    signals, details = analyze(y, sr)
    if model and details.get("speech"):
        clf = _voice_model().get()
        segs = speech_segments(_resample(y, sr), np.array(details["speech"]), max_n=6)
        probs = clf.predict(segs) if clf is not None and segs else []
        s_m, md = model_signal(probs, _voice_model().status)
        signals.append(s_m)
        details["model"] = md
    score, conf, reasons, groups = group_score(signals)
    details["confidence"] = conf
    details["groups"] = groups
    details["signals"] = [s.to_dict() for s in signals]
    return (50.0 if score is None else score), reasons, details


# --------------------------------------------------------------------------
# Live stream
# --------------------------------------------------------------------------
class VoiceStream:
    """Ring buffer of the shared system audio at its native sample rate.

    `push()` is called from the network loop, `snapshot()` / `analyze_window()`
    from the analysis thread.
    """

    def __init__(self, keep_s: float = 20.0, window_s: float = 12.0, model: bool = True):
        self.keep_s, self.window_s = keep_s, window_s
        self.use_model = model
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.sr: Optional[int] = None
            self.buf = np.zeros(0, np.float32)
            self.pos = self.filled = 0
            self.t_end: Optional[float] = None
            self.received_s = 0.0
            self.model_hist: deque = deque(maxlen=12)   # (t_end, P(fake))
            self._last_model_t = -1e9

    def push(self, t_end: float, sr: int, pcm: np.ndarray):
        """Append a chunk whose last sample was captured at t_end (seconds)."""
        n = len(pcm)
        if n == 0 or sr <= 0:
            return
        with self.lock:
            expected = None if self.t_end is None else self.t_end + n / sr
            # new rate, or a hole in the stream (tab throttled, track restarted) -> start over
            if sr != self.sr or (expected is not None and abs(t_end - expected) > 0.75):
                self.sr = sr
                self.buf = np.zeros(int(self.keep_s * sr), np.float32)
                self.pos = self.filled = 0
            cap = len(self.buf)
            pcm = pcm[-cap:]
            n = len(pcm)
            end = self.pos + n
            if end <= cap:
                self.buf[self.pos:end] = pcm
            else:
                k = cap - self.pos
                self.buf[self.pos:] = pcm[:k]
                self.buf[:n - k] = pcm[k:]
            self.pos = end % cap
            self.filled = min(cap, self.filled + n)
            self.t_end = t_end
            self.received_s += n / sr

    def snapshot(self, seconds: float):
        """(samples, sr, t_end) for the most recent `seconds` of audio."""
        with self.lock:
            if not self.sr or not self.filled:
                return np.zeros(0, np.float32), self.sr, self.t_end
            n = min(self.filled, int(seconds * self.sr))
            idx = (self.pos - n + np.arange(n)) % len(self.buf)
            return self.buf[idx].copy(), self.sr, self.t_end

    def _model_check(self, y, sr, det, t_end):
        """Score the newest 4 s of speech with the trained detector (at most every 2 s)."""
        cls = _voice_model()
        clf = cls.get(wait=False)                      # never stall the analysis loop on loading
        if clf is None and cls.status == "not loaded":
            cls.warmup_async()
        if clf is not None and t_end is not None and t_end - self._last_model_t >= MODEL_EVERY_S:
            segs = speech_segments(_resample(y, sr), np.array(det.get("speech") or []), max_n=1)
            if segs:
                self._last_model_t = t_end
                self.model_hist.append((t_end, float(clf.predict(segs)[0])))
        recent = [p for t, p in self.model_hist if t_end is None or t >= t_end - self.window_s]
        quality = clip01((det.get("snr_db", 0) - 8) / 12)
        return model_signal(recent, cls.status, quality)

    def analyze_window(self, y=None, sr=None, t_end=None) -> dict:
        if y is None:
            y, sr, t_end = self.snapshot(self.window_s)
        out = {"status": "no_audio", "score": None, "confidence": 0.0, "signals": [], "reasons": [], "details": {}}
        if not sr or len(y) < sr:
            return out
        if np.max(np.abs(y)) < 1e-4:
            out["status"] = "silent"
            return out
        signals, det = analyze(y, sr, channel="call")
        speech_s = det.get("speech_s", 0.0) or 0.0
        k = 3
        compact = {
            "speech_s": speech_s, "duration_s": det.get("duration_s"), "snr_db": det.get("snr_db"),
            "f0_median": det.get("f0_median"), "pitch_spread_st": det.get("pitch_spread_st"),
            "micro_jitter_cents": det.get("micro_jitter_cents"), "hnr_db": det.get("hnr_db"),
            "sr": sr, "step_s": HOP * k / SR,
            "f0": (det.get("f0") or [])[::k],
            "energy": [round(v, 1) for v in (det.get("energy_db") or [])[::k]],
        }
        out["details"] = compact
        if speech_s < 1.5:
            out["status"] = "listening"
            return out
        if self.use_model:
            s_m, md = self._model_check(y, sr, det, t_end)
            signals.append(s_m)
            compact["model"] = md
        score, conf, reasons, groups = group_score(signals)
        compact["groups"] = groups
        out.update(status="analyzing", score=score, confidence=conf, reasons=reasons,
                   signals=[s.to_dict() for s in signals])
        return out
