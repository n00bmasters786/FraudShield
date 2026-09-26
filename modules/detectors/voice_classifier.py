"""
Utterance-level deepfake-speech classifier.

wav2vec2-base (95 M params) fine-tuned on the In-the-Wild corpus — real and
deepfaked recordings of public figures taken from the internet, i.e. messy,
compressed, real-world audio rather than clean lab TTS. Input: 16 kHz mono,
scored in ~4 s windows of speech. Output: P(fake).
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from . import MODELS_DIR, torch_device

log = logging.getLogger("fraudshield.voice_model")

REPO = "abhishtagatya/wav2vec2-base-960h-itw-deepfake"
SR = 16000


class VoiceDeepfakeClassifier:
    _instance: Optional["VoiceDeepfakeClassifier"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    @classmethod
    def get(cls, wait: bool = True) -> Optional["VoiceDeepfakeClassifier"]:
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
                except Exception as e:
                    cls.status = f"unavailable: {e}"
                    log.warning("voice deepfake model unavailable: %s", e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        threading.Thread(target=cls.get, name="fs-voice-model-load", daemon=True).start()

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

        self.torch = torch
        self.device = torch_device()
        path = snapshot_download(REPO, local_dir=MODELS_DIR / REPO.split("/")[1],
                                 allow_patterns=["*.json", "*.safetensors"])
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(path).eval().to(self.device)
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        fake = [i for name, i in labels.items() if any(w in name for w in ("fake", "spoof"))]
        if len(fake) != 1:
            raise RuntimeError(f"can't tell which output is 'fake' in {self.model.config.id2label}")
        self.fake_idx = fake[0]
        self._lock = threading.Lock()
        self.name = f"wav2vec2-base · In-the-Wild · {self.device}"

    def predict(self, segments_16k: List[np.ndarray]) -> np.ndarray:
        """P(fake) for each 16 kHz float32 segment."""
        if not segments_16k:
            return np.zeros(0, np.float32)
        torch = self.torch
        inp = self.fe([s.astype(np.float32) for s in segments_16k], sampling_rate=SR,
                      return_tensors="pt", padding=True)
        with self._lock, torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in inp.items()}).logits
        return torch.softmax(logits.float(), -1)[:, self.fake_idx].cpu().numpy()
