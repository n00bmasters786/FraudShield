# FraudShield Live — real-time deepfake detection for video KYC

The banking officer shares their screen with FraudShield and runs the video
KYC call as usual, in any conferencing app. FraudShield finds the customer's
face in the call window and listens to the call audio. A trained deepfake
detector, liveness checks, a voice-clone detector and a lip-sync check feed a
deepfake-risk score that updates every second on a live dashboard.

```
 Officer's browser                          FraudShield engine (Python, localhost)
 ─────────────────                          ──────────────────────────────────────
 getDisplayMedia (entire screen             FaceStream   find customer face on screen (tiled BlazeFace)
   + system audio)                            │          → Face Mesh 478 landmarks on native pixels
   │                                          │          → trained deepfake detector (FF++ + Celeb-DF)
   │                                          │          → liveness checks over a rolling 20 s window
 capture-worker.js ── JPEG frames 15 fps ──▶  │
   (Web Worker: keeps running when the     VoiceStream  rolling 12 s of call audio → voice-clone
    dashboard is behind the call window)      │          detector + call-tuned voice checks
 pcm-worklet.js ───── PCM audio ──────────▶  avsync      lip opening ↔ voice envelope correlation
                                              │
 dashboard  ◀──── state JSON every 1 s ───── fusion      confidence-weighted score, smoothing,
   gauge · overlay · timeline · evidence                 verdict with hysteresis, events, challenges
```

## Setup

```bash
py -3.11 -m venv venv
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python -m tools.download_models   # ~430 MB of detector weights into ./models
venv\Scripts\python app.py                     # opens http://localhost:8000
```

With an NVIDIA GPU, install the CUDA builds instead; the detectors use the GPU
automatically. Install torch and torchvision **together from the same index**,
because a torchvision built for a different torch breaks both detectors
(`operator torchvision::nms does not exist`):

```bash
venv\Scripts\pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

If torch or the weights are missing, the app still runs on the forensic checks
alone, and the dashboard's **AI models** pill says so. Hover over the pill to
see why.

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

Evidence is split into two groups, scored separately (`modules/scoring.py`):

- **Synthesis**: is the face or voice itself generated?
- **Liveness**: is a live person in front of the camera, rather than a photo,
  a screen or a replay?

A module's score is the **worse** of the two groups. This matters: a deepfake
blinks, turns its head and moves its lips like a real person, so it passes
every liveness check. Averaging those "clear" results together with the
synthesis evidence is what used to let deepfakes through as genuine.

**Face** (`modules/face_module.py`)

| Check | Group | Catches |
|---|---|---|
| **AI deepfake detector**: MS-EffGCViT-B0 × 2 (FaceForensics++ and Celeb-DF v2 checkpoints), 4 face crops/s | synthesis | face swaps, reenactment, neural textures |
| Landmark jitter (residual after rigid alignment) | synthesis (minor) | frame-by-frame synthesised faces |
| Face-swap seam (cheek vs. neck sensor noise) | synthesis (minor) | swapped faces blended onto a real head |
| Face tracking / second face | liveness | face-swap dropouts, coached / assisted sessions |
| Blink behaviour (rate, depth, left/right symmetry) | liveness | photos, replays |
| Expression dynamics | liveness | photo, frozen or looped frame |
| 3-D parallax (landmarks vs. a single homography) | liveness | flat photo / screen held to the camera |
| Remote pulse / rPPG (POS on forehead + cheeks) | liveness | photos, replays |
| Screen recapture (moiré peaks + glare) | liveness | a phone or monitor held up to the customer's camera |

The two face checkpoints are good at different things. The FF++ checkpoint
catches the FaceForensics / DFD family; the Celeb-DF checkpoint is
near-perfect on genuine faces. Their clip-level probabilities are combined by
a small logistic stacker in `modules/calibration.py`. The detector's
reliability falls with the face's *effective* resolution: how much real detail
the crop carries, not its size on screen. A blurry, upscaled low-bandwidth
stream therefore counts for less.

**Voice** (`modules/voice_module.py`), on the call audio: a trained
**voice-clone detector** (wav2vec2-base fine-tuned on the In-the-Wild corpus;
synthesis group) plus intonation, pitch micro-jitter, periodicity (HNR),
rhythm & pauses and loudspeaker replay. Conferencing apps noise-gate pauses,
band-limit the audio and strip breaths. That would make the digital-silence,
bandwidth and breathing checks fire on every genuine customer, so they are
reported as *not applicable* on call audio.

**Lip-sync** (`modules/avsync.py`): peak cross-correlation (±400 ms) between
lip opening and the voice envelope, plus a hard alarm when a voice is heard
but the lips don't move. This catches voice-overs, cloned voices played over
a real or recorded face, and replays.

Each check reports a risk, a weight and a **reliability** for the current
window. For example, the 3-D test is n/a until the head turns. Fusion is in
`modules/fusion.py`: a confidence-weighted average in which one confident red
module can't be averaged away. Risk rises fast and decays slowly.

## Measured accuracy

`tools/evaluate.py` runs labelled clips through exactly the live pipeline.
The sample is 69 public clips (66 with a trackable face): 35 from Google's DeepFakeDetection (DFD) set
and 34 from Celeb-DF v2, about half real and half fake. Each clip is run at
full resolution, as a decent 720p call (1280 px, JPEG q70), and as a poor
360p call (640 px, JPEG q40, upscaled back onto the screen).

Face module with the shipped calibration. The calibration was fitted on these
same clips, so these results are *in-sample*:

| Video quality | Face score AUC | Fakes flagged *Likely deepfake* (≥ 65) | Genuine flagged ≥ 65 | Genuine flagged *Suspicious* (≥ 35) |
|---|---|---|---|---|
| Full resolution | 0.95 | 25 / 33 | 0 / 32 | 7 / 32 |
| 720p call | 0.92 | 22 / 34 | 1 / 32 | 12 / 32 |
| 360p call | 0.82 | 11 / 34 | 0 / 32 | 8 / 32 |

For comparison, the previous forensic-only version let most of these
deepfakes through as genuine.

Out-of-sample check (calibrate on one dataset, test on the other, full
resolution):

| Tested on | Detector AUC | Fakes flagged ≥ 65 | Genuine flagged ≥ 65 |
|---|---|---|---|
| Celeb-DF (calibrated on DFD) | 0.98 | 14 / 18 | 1 / 16 |
| DFD (calibrated on Celeb-DF) | 0.89 | 3 / 15 | 0 / 16 |

What this shows:

- **The trained detector is what catches deepfakes.** The forensic checks
  were near chance (AUC ≈ 0.5) at telling a deepfake from a real face. They
  still matter for photo, screen and replay attacks, which is what they are
  for.
- **It degrades with bandwidth.** At 360p there is too little real detail in
  the face. The detector's reliability drops accordingly, and the dashboard
  asks the officer to have the customer move closer or fix their connection.
- **Calibration is domain-specific.** Calibrated on Celeb-DF only, DFD fakes
  were under-scored even though the detector still ranked them well.
  Calibrate on recordings from your own KYC channel (next section).
- **These numbers are a sanity check, not a certification.** 66 clips give
  wide error bars. The Celeb-DF checkpoint may also have seen some of these
  Celeb-DF clips in training, so its Celeb-DF numbers are optimistic.

**Voice**, using the same method: 18 modern commercial TTS / clone clips
(ElevenLabs, Polly, Hume and others) against 25 genuine clips. The clone
detector raised no false alarms, but flagged only 4 of 18 clones at ≥ 65.
Open-source clone detectors lag behind commercial voice cloning. A "clear"
from this detector is therefore treated as weak evidence, and the lip-sync
check plus the *read this sentence aloud* challenge remain the main voice
defences.

## Calibrating on your own KYC recordings

Put genuine and deepfake recordings from your channel in folders and run:

```bash
venv\Scripts\python -m tools.evaluate --real kyc\real --fake kyc\fake --degrade none --calibrate
```

It prints each check's AUC and the catch / false-alarm counts at the
dashboard thresholds. It then fits the detector calibration and writes
`models/calibration.json`, which the engine loads at start-up. Add `--cross`
to see how a calibration fitted on some folders holds up on another. For
voice, use `--real-audio` / `--fake-audio`.

**More robust face model (GPU recommended).** The larger B5 FF++ checkpoint
held up better under compression on DFD: AUC 0.86 vs 0.79 at 720p, and 0.71
vs 0.54 at 360p. It costs ~355 ms per face on CPU, against ~100 ms for both
B0 checkpoints together:

```bash
set FRAUDSHIELD_FACE_MODELS=b0-ff++,b0-celeb,b5-ff++
venv\Scripts\python -m tools.evaluate --real ... --fake ... --degrade call --calibrate
venv\Scripts\python app.py
```

Each checkpoint set keeps its own calibration entry, and an uncalibrated set
falls back to a plain average.

## Performance

FraudShield has to run next to the video call without slowing it down. The
figures below were measured with a live session (dashboard visible, a deepfake
clip playing in the shared call tab) and are % of one CPU core:

| | Before | Now |
|---|---|---|
| Chrome (dashboard + capture) | ~252 % | ~97 % |
| Python engine | ~99 % | ~30 % |
| Frames analysed per second | 11 | 15 |

Detection results are unchanged (same verdicts and fake scores). How it stays
light:

- **Face-region streaming.** The browser sends the whole screen only while
  searching for the customer (≤ 3 fps, 1 fps when nothing is on screen) and
  every 2.5 s to re-check the screen. Otherwise it sends just the region around
  the face, at native resolution, so encoding and decoding cost a fraction of a
  full 1080p frame.
- **Work done once.** Eye openness, head pose and landmark jitter are computed
  when each frame arrives, not recomputed over the 20 s window every second.
  The slower checks (3-D depth, pulse, texture) refresh every 2 s and voice
  every 2 s; the pulse algorithm is vectorised.
- **GPU without CPU spin.** The face detector replays as a single CUDA graph
  (~6 ms CPU per face instead of ~490 ms of per-kernel launches), and GPU
  results are awaited without busy-waiting.
- **No thread storms.** Numeric libraries are capped at 2 threads, and OpenMP
  workers wait passively (`OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=0`);
  their spin-waiting alone was costing ~70 % of a core.
- **A dashboard that rests.** Screen capture is capped at 15 fps. Canvases
  redraw only when their data changes (at most 15–30 fps), bars and rings
  animate on the compositor, and nothing loops during a session. Any looping
  animation makes the browser redraw the whole page at 60 fps. The glass keeps
  its look with a static tint instead of a live blur.

Heavy software running at the same time (DJ or streaming software, many
browser tabs) competes for the same CPU and GPU. If the call still stutters,
close those first.

## Privacy

Frames and audio go only to the engine on `localhost`. They are analysed in
memory (last ~30 s) and never written to disk.

## Honest limits

- **Deepfake generators move fast.** The face detector is trained on
  FaceForensics++ / Celeb-DF era face swaps and reenactment, so brand-new
  generators may evade it until it is re-trained. The liveness challenges
  (hand across the face, side profile) are the fallback that breaks most
  real-time swaps.
- **Voice-clone detection is the weakest part** (see above).
- **Poor video lowers detection power.** With a 360p stream or a small tile,
  the dashboard shows lower confidence instead of guessing.

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
  scoring.py            Signal / evidence aggregation, liveness vs synthesis groups
  calibration.py        detector output → risk (stacker; models/calibration.json overrides)
  detectors/
    face_classifier.py  MS-EffGCViT ensemble on face crops (background thread)
    voice_classifier.py wav2vec2 voice-clone detector
    deepguard/          vendored model code (MIT, see NOTICE.md)
tools/
  download_models.py    fetch detector weights into ./models
  evaluate.py           accuracy report + calibration on labelled recordings
models/                 downloaded weights, calibration.json (git-ignored)
```
