# FraudShield Live — real-time deepfake detection for video KYC

The banking officer shares their screen with FraudShield and runs the video
KYC call as usual, in any conferencing app. FraudShield finds the customer's
face in the call window and listens to the call audio. It checks face, voice
and lip-sync, and updates a deepfake-risk score every second on a live
dashboard.

```
 Officer's browser                          FraudShield engine (Python, localhost)
 ─────────────────                          ──────────────────────────────────────
 getDisplayMedia (entire screen             FaceStream   find customer face on screen (tiled BlazeFace)
   + system audio)                            │          → Face Mesh 478 landmarks on native pixels
   │                                          │          → 8 face checks over a rolling 20 s window
 capture-worker.js ── JPEG frames 15 fps ──▶  │
   (Web Worker: keeps running when the     VoiceStream  rolling 12 s of call audio → 8 voice checks
    dashboard is behind the call window)      │          (call-codec profile)
 pcm-worklet.js ───── PCM audio ──────────▶  avsync      lip opening ↔ voice envelope correlation
                                              │
 dashboard  ◀──── state JSON every 1 s ───── fusion      confidence-weighted score, smoothing,
   gauge · overlay · timeline · evidence                 verdict with hysteresis, events, challenges
```

## Setup

```bash
py -3.11 -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python app.py            # opens http://localhost:8000
```

Use **Chrome or Edge**. Screen capture only works on a secure page, so open
the dashboard as `http://localhost:8000` on the officer's own machine (not
via a LAN IP). To reach it from another machine, put the server behind HTTPS.

## Using it

1. Click **Share screen**. Choose **Entire screen** and tick **Also share
   system audio**. The system audio is the customer's voice from the call.
2. Start or continue the video call. FraudShield locks onto the largest face
   on the screen. The officer's own small self-view is ignored. If several
   people are on screen, click **Select region** and drag a box around the
   customer's video tile.
3. Put the dashboard on a second monitor or beside the call. It keeps
   analysing even when the call window is in front. **Pop-out HUD** opens a
   small always-on-top window with the score, which you can float over the
   call.
4. Follow the **Suggested challenges** when evidence is missing or suspicious.
   Examples: read a sentence aloud, turn the head, wave a hand across the
   face, show a side profile.
5. **Report** downloads a JSON evidence report: verdict, every check, the
   risk timeline and the event log. **New customer** resets the session.

Verdicts: **Calibrating** (not enough evidence yet), **Likely genuine**
(< 35), **Suspicious** (35–65, run a liveness challenge), **Likely deepfake**
(≥ 65, stop and escalate).

## What is checked

**Face** — `modules/face_module.py`

| Check | Catches |
|---|---|
| Face tracking / second face | face-swap dropouts, coached / assisted sessions |
| Blink behaviour (rate, depth, left/right symmetry) | photos, replays, face-swaps |
| Expression dynamics | photo, frozen or looped frame |
| 3-D parallax (landmarks vs. a single homography) | flat photo / screen held to the camera |
| Landmark jitter (residual after rigid alignment) | frame-by-frame synthesised faces |
| Remote pulse / rPPG (POS on forehead + cheeks) | synthetic faces, replays |
| Face-swap seam (cheek vs. neck sensor noise) | swapped faces blended onto a real head |
| Screen recapture (moiré peaks + glare) | a phone or monitor held up to the customer's camera |

**Voice** — `modules/voice_module.py`, on the call audio: intonation,
pitch micro-jitter, periodicity (HNR), rhythm & pauses, loudspeaker replay.
Conferencing apps noise-gate pauses, band-limit the audio and strip breaths.
That would make the digital-silence, bandwidth and breathing checks fire on
every genuine customer, so they are reported as *not applicable* on call audio.

**Lip-sync** — `modules/avsync.py`: peak cross-correlation (±400 ms)
between lip opening and the voice envelope. There is also a hard alarm when a
voice is heard but the lips don't move. This catches voice-overs, cloned
voices played over a real or recorded face, and replays.

Each check reports a risk, a weight and a **reliability** for the current
window. For example, the 3-D test is n/a until the head turns, and landmark
checks are down-weighted when the face is small on screen. Module scores come
from `modules/scoring.py`. Fusion is in `modules/fusion.py`: a
confidence-weighted average in which one confident red module can't be
averaged away. Risk rises fast and decays slowly.

## Privacy

Frames and audio go only to the engine on `localhost`. They are analysed in
memory (last ~30 s) and never written to disk.

## Honest limits

- The detectors are forensic signal checks, not a trained deepfake
  classifier. They reliably catch held-up photos and screens, replays,
  voice-overs and crude face-swaps. High-end real-time face-swaps may only
  show up as flicker or a missing pulse, and the liveness challenges exist
  to break them. Upgrade path: add a pretrained FaceForensics++ / ASVspoof
  (AASIST) model as one more signal in the same `Signal` / fusion framework.
- Heavy call compression and small faces lower confidence rather than raising
  risk, but thresholds should be tuned on real KYC recordings.

## Repo layout

```
app.py                  FastAPI server: static dashboard + /ws streaming endpoint
web/
  index.html styles.css app.js   dashboard (frosted glass, black)
  capture-worker.js     WebSocket + frame encoder (Web Worker)
  pcm-worklet.js        system-audio capture (AudioWorklet)
modules/
  live.py               LiveSession: per-call orchestration, events, challenges
  face_module.py        screen face finder + face forensics (score_video for files)
  voice_module.py       voice forensics + VoiceStream (score_audio for files)
  avsync.py             lip-sync check
  fusion.py             real-time fusion and verdict
  scoring.py            shared Signal / evidence aggregation
```
