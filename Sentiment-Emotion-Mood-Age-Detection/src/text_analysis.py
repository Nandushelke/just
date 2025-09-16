from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Lazy imports inside methods to avoid heavy downloads at import time


@dataclass
class ClassificationResult:
    label: str
    confidence: float


class TextAnalyzer:
    """
    Performs sentiment and emotion classification on text.
    - Sentiment: Uses NLTK VADER (lightweight, offline)
    - Emotion: Tries a transformers pipeline; falls back to a simple heuristic
    """

    def __init__(self) -> None:
        self._vader: Optional[object] = None
        self._emotion_pipe: Optional[object] = None

    def _ensure_vader(self) -> None:
        if self._vader is not None:
            return
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            # Ensure required resources are available; handle environments without network
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                try:
                    nltk.download("vader_lexicon", quiet=True)
                except Exception:
                    pass
            self._vader = SentimentIntensityAnalyzer()
        except Exception:
            self._vader = None

    def _ensure_emotion_pipe(self) -> None:
        if self._emotion_pipe is not None:
            return
        try:
            from transformers import pipeline  # type: ignore
            # A commonly used emotion model; downloads on first use
            model_id = "j-hartmann/emotion-english-distilroberta-base"
            self._emotion_pipe = pipeline("text-classification", model=model_id, top_k=None)
        except Exception:
            self._emotion_pipe = None

    def analyze_sentiment(self, text: str) -> ClassificationResult:
        self._ensure_vader()
        if self._vader is None:
            # Minimal fallback: classify by simple polarity keywords
            lowered = text.lower()
            positive_cues = ["good", "great", "love", "excellent", "awesome", "happy"]
            negative_cues = ["bad", "terrible", "hate", "awful", "worst", "sad"]
            pos_hits = sum(cue in lowered for cue in positive_cues)
            neg_hits = sum(cue in lowered for cue in negative_cues)
            if pos_hits > neg_hits:
                return ClassificationResult("positive", 0.55)
            if neg_hits > pos_hits:
                return ClassificationResult("negative", 0.55)
            return ClassificationResult("neutral", 0.5)

        scores = self._vader.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        if compound >= 0.05:
            return ClassificationResult("positive", float(min(1.0, max(0.5, compound))))
        if compound <= -0.05:
            return ClassificationResult("negative", float(min(1.0, max(0.5, -compound))))
        return ClassificationResult("neutral", float(1.0 - abs(compound)))

    def analyze_emotion(self, text: str) -> ClassificationResult:
        self._ensure_emotion_pipe()
        if self._emotion_pipe is None:
            # Heuristic fallback based on keywords
            lowered = text.lower()
            heuristics = [
                ("joy", ["happy", "joy", "glad", "excited", "love", "great"]),
                ("sadness", ["sad", "down", "unhappy", "depressed", "cry"]),
                ("anger", ["angry", "mad", "furious", "rage", "annoyed"]),
                ("fear", ["afraid", "scared", "fear", "anxious", "nervous"]),
                ("disgust", ["disgust", "gross", "repuls", "nausea"]),
                ("surprise", ["surprise", "shocked", "astonished", "wow"]),
            ]
            for label, cues in heuristics:
                if any(cue in lowered for cue in cues):
                    return ClassificationResult(label, 0.55)
            return ClassificationResult("neutral", 0.5)

        try:
            outputs = self._emotion_pipe(text, truncation=True)
            # pipeline with top_k=None returns list of dicts per item; handle both shapes
            if isinstance(outputs, list) and len(outputs) > 0:
                result = outputs[0] if isinstance(outputs[0], dict) else outputs[0][0]
            else:
                result = {"label": "neutral", "score": 0.5}
            label = str(result.get("label", "neutral")).lower()
            score = float(result.get("score", 0.5))
            return ClassificationResult(label, score)
        except Exception:
            return ClassificationResult("neutral", 0.5)


def analyze_text(text: str) -> Dict[str, Dict[str, float | str]]:
    analyzer = TextAnalyzer()
    sentiment = analyzer.analyze_sentiment(text)
    emotion = analyzer.analyze_emotion(text)
    return {
        "sentiment": {"label": sentiment.label, "confidence": sentiment.confidence},
        "emotion": {"label": emotion.label, "confidence": emotion.confidence},
    }