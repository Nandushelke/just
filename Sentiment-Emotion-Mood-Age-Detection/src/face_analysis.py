from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:  # Pillow might not be installed in some envs
    Image = None  # type: ignore


@dataclass
class FaceResult:
    box: Tuple[int, int, int, int]
    probability: float
    age: int
    age_confidence: float
    emotion: str
    emotion_confidence: float


class FaceAnalyzer:
    """
    Detects faces and provides placeholder age and emotion estimates.
    Uses MTCNN for detection if available.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._mtcnn: Optional[object] = None

    def _ensure_mtcnn(self) -> None:
        if self._mtcnn is not None:
            return
        try:
            from facenet_pytorch import MTCNN  # type: ignore
            self._mtcnn = MTCNN(keep_all=True, device=self.device)
        except Exception:
            self._mtcnn = None

    def _open_image(self, image_path: str):
        if Image is None:
            raise RuntimeError("Pillow is not available.")
        return Image.open(image_path).convert("RGB")

    def _placeholder_age_emotion(self) -> Tuple[int, float, str, float]:
        # Placeholder values; integrate a trained model later
        return 30, 0.35, "neutral", 0.5

    def analyze_image(self, image_path: str) -> Dict[str, List[Dict[str, float | int | str]]]:
        self._ensure_mtcnn()
        try:
            img = self._open_image(image_path)
        except Exception:
            return {"faces": []}

        faces: List[FaceResult] = []
        if self._mtcnn is not None:
            try:
                boxes, probs = self._mtcnn.detect(img)
                if boxes is None or probs is None:
                    boxes = []
                    probs = []
                for box, prob in zip(boxes, probs):
                    if box is None or prob is None:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    age, age_conf, emotion, emo_conf = self._placeholder_age_emotion()
                    faces.append(
                        FaceResult(
                            box=(x1, y1, x2, y2),
                            probability=float(prob),
                            age=age,
                            age_confidence=age_conf,
                            emotion=emotion,
                            emotion_confidence=emo_conf,
                        )
                    )
            except Exception:
                pass

        # If detection failed or dependency missing, return a single default face if image loads
        if not faces:
            age, age_conf, emotion, emo_conf = self._placeholder_age_emotion()
            faces.append(
                FaceResult(
                    box=(0, 0, 0, 0),
                    probability=0.0,
                    age=age,
                    age_confidence=age_conf,
                    emotion=emotion,
                    emotion_confidence=emo_conf,
                )
            )

        return {
            "faces": [
                {
                    "box": list(f.box),
                    "probability": f.probability,
                    "age": f.age,
                    "age_confidence": f.age_confidence,
                    "emotion": f.emotion,
                    "emotion_confidence": f.emotion_confidence,
                }
                for f in faces
            ]
        }