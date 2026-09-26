"""
LiveSession — one monitored video-KYC call.

Receives screen frames and system-audio chunks from the officer's browser,
feeds them to the face / voice streams, and once a second fuses face, voice
and lip-sync into the dashboard state (score, verdict, evidence, events and
suggested liveness challenges).
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .avsync import analyze_sync
from .face_module import FaceStream
from .fusion import LiveFusion, Reading, fuse
from .voice_module import VoiceStream

EVENT_COOLDOWN_S = 20.0
SENTENCES = [
    "The name on my ID card is my full legal name",
    "I am opening this account for myself",
    "Today I am completing my video KYC",
]


def clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def _top_finding(mod: dict) -> str:
    sigs = [s for s in mod.get("signals", []) if s["status"] != "n/a"]
    for want in ("alert", "watch"):
        hits = sorted((s for s in sigs if s["status"] == want), key=lambda s: -s["risk"] * s["weight"])
        if hits:
            return hits[0]["message"]
    return mod["reasons"][0] if mod.get("reasons") else ""


class LiveSession:
    def __init__(self):
        self.face = FaceStream()
        self.voice = VoiceStream()
        self.fusion = LiveFusion()
        self.lock = threading.Lock()
        self.has_audio: Optional[bool] = None
        self._new_challenge()
        self._reset_counters()

    # ------------------------------------------------------------- control
    def _new_challenge(self):
        digits = " ".join(str(random.randint(0, 9)) for _ in range(4))
        self.challenge = f"“{random.choice(SENTENCES)} — reference {digits}”"

    def _reset_counters(self):
        self.t0: Optional[float] = None
        self.t_media: Optional[float] = None
        self.frames = 0
        self.proc_ms = deque(maxlen=30)
        self.frame_ts = deque(maxlen=40)
        self.pending_events = []
        self._sig_status = {}
        self._last_alert = {}
        self._verdict = None
        self._face_status = None
        self._voice_status = None

    def configure(self, msg: dict):
        if "roi" in msg:
            self.face.set_roi(msg["roi"])
        if "has_audio" in msg:
            self.has_audio = bool(msg["has_audio"])

    def reset(self):
        self.face.reset()
        self.voice.reset()
        self.fusion.reset()
        self._new_challenge()
        with self.lock:
            self._reset_counters()

    def close(self):
        self.face.close()

    # --------------------------------------------------------------- media
    def _clock(self, ts_ms: float) -> float:
        t = ts_ms / 1000.0
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            self.t_media = t if self.t_media is None else max(self.t_media, t)
        return t

    def process_frame(self, ts_ms: float, jpeg: bytes) -> dict:
        t = self._clock(ts_ms)
        t_start = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"type": "ack", "error": "bad frame"}
        ov = self.face.process(t, frame)
        ms = 1000 * (time.perf_counter() - t_start)
        with self.lock:
            self.frames += 1
            self.proc_ms.append(ms)
            self.frame_ts.append(t)
            for kind, text in self.face.events:
                self._event("face", "info" if kind == "acquire" else "warn", text)
            self.face.events.clear()
        return clean({"type": "ack", "t": ts_ms, "proc_ms": round(ms, 1),
                      "size": [frame.shape[1], frame.shape[0]], **ov})

    def add_audio(self, ts_ms: float, sr: int, pcm16: bytes):
        t = self._clock(ts_ms)
        pcm = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.voice.push(t, int(sr), pcm)

    # -------------------------------------------------------------- events
    def _event(self, module, level, text, t=None):
        t = self.t_media if t is None else t
        rel = 0.0 if t is None or self.t0 is None else t - self.t0
        self.pending_events.append({"t": round(rel, 1), "module": module, "level": level, "text": text,
                                    "at": time.strftime("%H:%M:%S")})

    def _track_signals(self, module, mod):
        now = self.t_media or 0.0
        for s in mod.get("signals", []):
            key = f"{module}.{s['key']}"
            prev = self._sig_status.get(key)
            self._sig_status[key] = s["status"]
            if s["status"] == "alert" and prev != "alert" and s["reliability"] >= 0.5 \
                    and now - self._last_alert.get(key, -1e9) > EVENT_COOLDOWN_S:
                self._last_alert[key] = now
                self._event(module, "alert", f"{s['label']}: {s['message']}")

    # ------------------------------------------------------------ analysis
    def analyze(self) -> Optional[dict]:
        if self.t_media is None:
            return None
        face = self.face.analyze_window()
        y, sr, a_end = self.voice.snapshot(self.voice.window_s)
        voice = self.voice.analyze_window(y, sr)
        if self.has_audio is False:
            voice["status"] = "no_track"
        mt, mouth = self.face.mouth_series(12.0)
        sync = analyze_sync(mt, mouth, y, sr, a_end)
        mods = {"face": face, "voice": voice, "sync": sync}

        readings = {k: Reading(m["score"], m["confidence"]) for k, m in mods.items()}
        raw, conf = fuse(readings)

        with self.lock:
            t_rel = self.t_media - self.t0
            overall = self.fusion.update(self.t_media, raw, conf, face["status"] in ("tracking", "warming"))

            if face["status"] != self._face_status and face["status"] == "searching" and self._face_status:
                self._event("face", "warn", "No customer face on screen")
            self._face_status = face["status"]
            if voice["status"] == "analyzing" and self._voice_status not in (None, "analyzing"):
                self._event("voice", "info", "Customer speech detected — voice analysis running")
            self._voice_status = voice["status"]
            for k, m in mods.items():
                self._track_signals(k, m)
            if overall["verdict"] != self._verdict and overall["verdict"] in ("genuine", "suspicious", "deepfake"):
                lvl = {"genuine": "ok", "suspicious": "warn", "deepfake": "alert"}[overall["verdict"]]
                self._event("fusion", lvl, f"{overall['label']} (risk {overall['score']:.0f})")
            self._verdict = overall["verdict"]
            events, self.pending_events = self.pending_events, []

            ft = list(self.frame_ts)
            fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else 0.0
            stats = {"fps": round(fps, 1), "proc_ms": round(float(np.mean(self.proc_ms)), 1) if self.proc_ms else None,
                     "frames": self.frames, "audio_s": round(self.voice.received_s, 1),
                     "audio_sr": self.voice.sr}

        for k, m in mods.items():
            m["finding"] = _top_finding(m)
            m.pop("reasons", None)

        return clean({
            "type": "state", "t": round(t_rel, 2),
            "overall": overall,
            "modules": mods,
            "point": [round(t_rel, 2), overall["score"], face["score"], voice["score"], sync["score"]],
            "events": events,
            "prompts": self._prompts(mods, overall),
            "stats": stats,
        })

    def _prompts(self, mods, overall):
        """Liveness challenges the officer can ask for, based on what evidence is missing or suspicious."""
        face, voice = mods["face"], mods["voice"]
        sig = {s["key"]: s for m in mods.values() for s in m.get("signals", [])}
        out = []
        if face["status"] == "searching":
            out.append(("screen", "Bring the video call into view",
                        "No face on the shared screen. Un-minimise the call or maximise the customer's video."))
        if voice["status"] == "no_track":
            out.append(("audio", "Enable system audio",
                        "Stop and share again with “Also share system audio” ticked to analyse the voice."))
        elif voice["status"] in ("listening", "silent", "no_audio") and face["status"] == "tracking":
            out.append(("speak", "Ask the customer to read aloud", self.challenge))
        if face["status"] == "tracking":
            par = sig.get("parallax")
            if par and par["reliability"] < 0.3:
                out.append(("turn", "Ask them to turn their head slowly left and right",
                            "Needed for the 3-D depth test — flat photos and screens fail it."))
            pulse = sig.get("pulse")
            if pulse and pulse["reliability"] < 0.2 and face["details"].get("tracked_s", 0) > 8:
                out.append(("still", "Ask them to hold still facing the light for 10 s",
                            "Lets the remote-pulse check find a heartbeat in the skin."))
        if overall["verdict"] in ("suspicious", "deepfake"):
            out.append(("hand", "Ask them to wave a hand slowly across their face",
                        "Real-time face swaps tear and flicker when the face is covered."))
            out.append(("profile", "Ask for a full side profile, then back",
                        "Face-swap models are trained on frontal faces and break at 90°."))
        return [{"icon": i, "title": t, "text": x} for i, t, x in out[:4]]
