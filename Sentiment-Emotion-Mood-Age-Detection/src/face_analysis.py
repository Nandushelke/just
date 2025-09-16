from __future__ import annotations

"""
Face analysis stub for age estimation and facial emotion recognition.

Plan:
- Detect faces with a lightweight detector (e.g., MTCNN from facenet-pytorch)
- Run facial expression classifier (FER-2013 fine-tuned) -> canonical emotions
- Run age estimator (UTKFace/Adience fine-tuned regressor/classifier)

Functions below return placeholders for integration.
"""

from typing import Dict


def analyze_face_placeholder(image_path: str) -> Dict[str, object]:
    """Placeholder: return neutral emotion and age ~30."""
    return {
        "age": 30,
        "emotions": {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "fearful": 0.0,
            "disgust": 0.0,
            "surprise": 0.0,
            "neutral": 1.0,
        },
    }

