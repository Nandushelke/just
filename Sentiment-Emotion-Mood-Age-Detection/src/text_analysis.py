from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # optional heavy dependency
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
except Exception:  # pragma: no cover
    nltk = None  # type: ignore
    SentimentIntensityAnalyzer = None  # type: ignore


CANONICAL_EMOTIONS: List[str] = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprise",
    "neutral",
]

_sentiment_pipeline = None
_emotion_pipeline = None


def _get_device() -> int:
    if torch is None:
        return -1
    try:
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _lazy_load_sentiment_pipeline() -> None:
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    if pipeline is None:
        raise ImportError("transformers is not available")
    _sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=_get_device(),
        truncation=True,
    )


def _lazy_load_emotion_pipeline() -> None:
    global _emotion_pipeline
    if _emotion_pipeline is not None:
        return
    # Compact and widely used English emotion classifier
    model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
    if pipeline is None:
        raise ImportError("transformers is not available")
    _emotion_pipeline = pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        device=_get_device(),
        top_k=None,
        truncation=True,
        return_all_scores=True,
    )


def _normalize_sentiment_label(label: str, score: float) -> Tuple[str, float]:
    upper = label.upper()
    if upper in {"POS", "POSITIVE"}:
        norm_label = "positive"
        polarity = score
    elif upper in {"NEG", "NEGATIVE"}:
        norm_label = "negative"
        polarity = -score
    else:
        norm_label = "neutral"
        polarity = 0.0
    # Introduce neutral band when polarity is weak
    if abs(polarity) < 0.2:
        norm_label = "neutral"
        polarity = 0.0
    return norm_label, float(max(min(polarity, 1.0), -1.0))


def _vader_fallback(text: str) -> Dict[str, float]:
    if nltk is None or SentimentIntensityAnalyzer is None:
        return {"compound": 0.0}
    try:
        # Ensure lexicon available
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores
    except Exception:
        return {"compound": 0.0}


def analyze_text(
    text: str,
    include_raw: bool = False,
) -> Dict[str, object]:
    """Analyze text to produce sentiment and emotion distributions.

    Returns a dictionary with keys: sentiment, emotions, and optionally raw.
    - sentiment: {label, score, polarity}
    - emotions: dict of canonical emotions to probabilities (0..1)
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {
            "sentiment": {"label": "neutral", "score": 0.0, "polarity": 0.0},
            "emotions": {k: (1.0 if k == "neutral" else 0.0) for k in CANONICAL_EMOTIONS},
        }

    sentiment_result: Dict[str, object] = {
        "label": "neutral",
        "score": 0.0,
        "polarity": 0.0,
    }
    emotions_result: Dict[str, float] = {k: 0.0 for k in CANONICAL_EMOTIONS}
    raw: Dict[str, object] = {}

    # Sentiment via Transformers (with VADER fallback)
    try:
        _lazy_load_sentiment_pipeline()
        assert _sentiment_pipeline is not None
        sent_out = _sentiment_pipeline(cleaned)
        raw["sentiment"] = sent_out
        label = sent_out[0]["label"]
        score = float(sent_out[0]["score"])  # confidence of the predicted class
        norm_label, polarity = _normalize_sentiment_label(label, score)
        sentiment_result.update({"label": norm_label, "score": score, "polarity": polarity})
    except Exception:
        vader = _vader_fallback(cleaned)
        compound = float(vader.get("compound", 0.0))
        if compound > 0.2:
            label = "positive"
        elif compound < -0.2:
            label = "negative"
        else:
            label = "neutral"
        sentiment_result.update({"label": label, "score": abs(compound), "polarity": compound})
        raw["sentiment"] = {"vader_compound": compound}

    # Emotions via Transformers (best-effort mapping to canonical set)
    try:
        _lazy_load_emotion_pipeline()
        assert _emotion_pipeline is not None
        emo_out = _emotion_pipeline(cleaned)
        # emo_out is List[List[{label, score}]]
        scores = {item["label"].lower(): float(item["score"]) for item in emo_out[0]}
        # Source model labels: sadness, joy, love, anger, fear, surprise
        emotions_result["happy"] = float(scores.get("joy", 0.0) + scores.get("love", 0.0))
        emotions_result["sad"] = float(scores.get("sadness", 0.0))
        emotions_result["angry"] = float(scores.get("anger", 0.0))
        emotions_result["fearful"] = float(scores.get("fear", 0.0))
        emotions_result["surprise"] = float(scores.get("surprise", 0.0))
        # Disgust not present in this model; leave as 0.0
        # Neutral: infer a small amount when sentiment is neutral
        neutral_hint = 0.15 if sentiment_result["label"] == "neutral" else 0.0
        emotions_result["neutral"] = neutral_hint
        # Normalize to sum to 1.0
        s = sum(emotions_result.values())
        if s > 0:
            for k in list(emotions_result.keys()):
                emotions_result[k] = emotions_result[k] / s
        raw["emotions"] = emo_out
    except Exception:
        # Fallback: derive a coarse emotion from sentiment only
        if sentiment_result["label"] == "positive":
            emotions_result.update({"happy": 0.8, "neutral": 0.2})
        elif sentiment_result["label"] == "negative":
            emotions_result.update({"sad": 0.5, "angry": 0.2, "fearful": 0.1, "neutral": 0.2})
        else:
            emotions_result.update({"neutral": 1.0})
        raw["emotions"] = {"fallback": True}

    result = {"sentiment": sentiment_result, "emotions": emotions_result}
    if include_raw:
        result["raw"] = raw
    return result


if __name__ == "__main__":  # manual quick test
    sample = "I am thrilled but a little nervous about the interview!"
    print(json.dumps(analyze_text(sample, include_raw=False), indent=2))

