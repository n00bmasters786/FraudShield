"""
Fetch the trained deepfake detectors into ./models ahead of time.

    venv\\Scripts\\python -m tools.download_models
"""

from modules.detectors.face_classifier import FaceDeepfakeClassifier
from modules.detectors.voice_classifier import VoiceDeepfakeClassifier


def main():
    for cls in (FaceDeepfakeClassifier, VoiceDeepfakeClassifier):
        clf = cls.get()
        print(f"{cls.__name__:<26} {cls.status}" + (f" — {clf.name}" if clf else ""))


if __name__ == "__main__":
    main()
