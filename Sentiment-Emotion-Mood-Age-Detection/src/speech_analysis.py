from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ClassificationResult:
    label: str
    confidence: float


class SpeechEmotionAnalyzer:
    """
    Lightweight, heuristic speech emotion estimator using basic acoustic features.
    Intended as a placeholder until a trained model is integrated.
    """

    def __init__(self, target_sr: int = 16000) -> None:
        self.target_sr = target_sr

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        import librosa  # Lazy import

        waveform, sr = librosa.load(path, sr=self.target_sr, mono=True)
        return waveform, sr

    def _extract_features(self, waveform: np.ndarray, sr: int) -> Dict[str, float]:
        import librosa

        if waveform.size == 0:
            return {"rms": 0.0, "centroid": 0.0, "zcr": 0.0}

        rms = float(np.mean(librosa.feature.rms(y=waveform)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)))
        return {"rms": rms, "centroid": centroid, "zcr": zcr}

    def _heuristic_classify(self, feats: Dict[str, float]) -> ClassificationResult:
        # Simple thresholds tuned coarsely for placeholder behavior
        energy = feats.get("rms", 0.0)
        brightness = feats.get("centroid", 0.0)
        zcr = feats.get("zcr", 0.0)

        # Normalize rough ranges to derive a confidence
        energy_norm = min(1.0, energy / 0.1)
        bright_norm = min(1.0, brightness / 4000.0)
        zcr_norm = min(1.0, zcr / 0.2)

        if energy_norm < 0.2 and bright_norm < 0.2 and zcr_norm < 0.2:
            return ClassificationResult("calm", 0.55)
        if energy_norm > 0.6 and bright_norm > 0.5 and zcr_norm > 0.5:
            return ClassificationResult("angry", 0.6)
        if bright_norm > 0.6 and zcr_norm > 0.4 and energy_norm < 0.5:
            return ClassificationResult("fear", 0.55)
        if energy_norm > 0.5 and bright_norm > 0.4:
            return ClassificationResult("happy", 0.55)
        return ClassificationResult("neutral", 0.5)

    def analyze_file(self, audio_path: str) -> Dict[str, float | str]:
        try:
            waveform, sr = self._load_audio(audio_path)
            feats = self._extract_features(waveform, sr)
            pred = self._heuristic_classify(feats)
            return {"label": pred.label, "confidence": pred.confidence, **feats}
        except Exception:
            return {"label": "neutral", "confidence": 0.5}