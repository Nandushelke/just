from __future__ import annotations

from typing import Dict, Tuple


MOOD_ORDER = ["depressed", "stressed", "neutral", "calm", "energetic"]
MOOD_TO_NUMERIC = {
    "depressed": -1.0,
    "stressed": -0.5,
    "neutral": 0.0,
    "calm": 0.5,
    "energetic": 1.0,
}


def _compute_valence_arousal(emotions: Dict[str, float], polarity: float) -> Tuple[float, float]:
    """Approximate circumplex mapping from emotions to valence and arousal.

    - Valence ~ pleasantness (negative .. positive)
    - Arousal ~ activation (low .. high)
    """
    happy = emotions.get("happy", 0.0)
    sad = emotions.get("sad", 0.0)
    angry = emotions.get("angry", 0.0)
    fearful = emotions.get("fearful", 0.0)
    disgust = emotions.get("disgust", 0.0)
    surprise = emotions.get("surprise", 0.0)
    neutral = emotions.get("neutral", 0.0)

    negative_total = sad + angry + fearful + disgust
    positive_total = happy

    valence = positive_total - negative_total
    # Blend in sentiment polarity (range [-1, 1])
    valence = 0.6 * valence + 0.4 * float(polarity)

    # Arousal: high for anger/fear/surprise, medium for happy, low for sad/neutral
    arousal = 0.9 * (angry + fearful + surprise) + 0.5 * happy + 0.2 * neutral + 0.1 * sad

    # Clamp to [0, 1] for arousal, [-1, 1] for valence
    valence = max(min(valence, 1.0), -1.0)
    arousal = max(min(arousal, 1.0), 0.0)
    return valence, arousal


def infer_mood(sentiment: Dict[str, float], emotions: Dict[str, float]) -> Dict[str, float]:
    """Infer a coarse mood label from sentiment and emotion signals.

    Returns: {label, valence, arousal, score}
    - score is the numeric mapping of the label (for timelines)
    """
    polarity = float(sentiment.get("polarity", 0.0))
    valence, arousal = _compute_valence_arousal(emotions, polarity)

    # Decision rules based on valence/arousal quadrants
    if abs(valence) < 0.08 and arousal < 0.15:
        label = "neutral"
    elif valence >= 0.15 and arousal < 0.4:
        label = "calm"
    elif valence >= 0.15 and arousal >= 0.4:
        label = "energetic"
    elif valence < -0.15 and arousal >= 0.35:
        label = "stressed"
    else:
        label = "depressed"

    return {
        "label": label,
        "valence": valence,
        "arousal": arousal,
        "score": MOOD_TO_NUMERIC[label],
    }


def mood_label_to_numeric(label: str) -> float:
    return float(MOOD_TO_NUMERIC.get(label, 0.0))

