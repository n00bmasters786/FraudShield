"""
Trained deepfake detectors, used as extra evidence next to the forensic checks.

  face_classifier   MS-EffGCViT-B0 ensemble (FaceForensics++ + Celeb-DF-v2 checkpoints)
  voice_classifier  wav2vec2-base fine-tuned on the In-the-Wild deepfake-speech corpus

Weights are fetched from Hugging Face into ./models on first use
(`python -m tools.download_models` does it ahead of time). If PyTorch or the
weights are missing, the detectors report themselves unavailable and the
engine keeps running on the forensic checks alone.
"""

import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CALIBRATION_FILE = MODELS_DIR / "calibration.json"


def torch_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def wait_gpu(torch, device):
    """Sleep until queued CUDA work is done. Reading a CUDA tensor back (.cpu()) otherwise
    busy-waits a whole CPU core for the length of the inference."""
    if device != "cuda":
        return
    done = torch.cuda.Event()
    done.record()
    while not done.query():
        time.sleep(0.001)
