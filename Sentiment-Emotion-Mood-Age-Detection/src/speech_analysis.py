from __future__ import annotations

"""
Speech emotion analysis stub.

Plan:
- Extract features with librosa (MFCCs, chroma, spectral contrast)
- Train/use a classifier (e.g., CNN/RNN) on RAVDESS or Emo-DB
- Map output to canonical emotions

Functions below provide placeholder outputs for integration testing.
"""

from typing import Dict


def analyze_audio_placeholder(wav_path: str) -> Dict[str, float]:
    """Placeholder: return a neutral distribution for now."""
    return {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "fearful": 0.0,
        "disgust": 0.0,
        "surprise": 0.0,
        "neutral": 1.0,
    }

