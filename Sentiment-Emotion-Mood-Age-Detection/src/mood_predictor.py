from __future__ import annotations

from typing import Dict, Optional


class MoodPredictor:
    """
    Combines multimodal predictions into a coarse mood estimate.
    Placeholder logic using simple scoring rules.
    """

    def __init__(self) -> None:
        pass

    def predict_mood(
        self,
        text_result: Optional[Dict[str, Dict[str, float | str]]] = None,
        speech_result: Optional[Dict[str, float | str]] = None,
        face_result: Optional[Dict[str, list[Dict[str, float | int | str]]]] = None,
    ) -> Dict[str, float | str]:
        score = 0.0
        confidence = 0.5

        # Text sentiment
        if text_result is not None:
            sentiment = text_result.get("sentiment", {})
            sent_label = str(sentiment.get("label", "neutral"))
            sent_conf = float(sentiment.get("confidence", 0.5))
            if sent_label == "positive":
                score += 1.0 * sent_conf
            elif sent_label == "negative":
                score -= 1.0 * sent_conf

            # Text emotion
            emotion = text_result.get("emotion", {})
            emo_label = str(emotion.get("label", "neutral")).lower()
            emo_conf = float(emotion.get("confidence", 0.5))
            if emo_label in {"joy", "happy", "happiness"}:
                score += 0.8 * emo_conf
            elif emo_label in {"sad", "sadness"}:
                score -= 0.8 * emo_conf
            elif emo_label in {"anger", "angry"}:
                score -= 0.9 * emo_conf
            elif emo_label in {"fear"}:
                score -= 0.6 * emo_conf

        # Speech emotion
        if speech_result is not None:
            sp_label = str(speech_result.get("label", "neutral")).lower()
            sp_conf = float(speech_result.get("confidence", 0.5))
            if sp_label in {"happy"}:
                score += 0.6 * sp_conf
            elif sp_label in {"angry"}:
                score -= 0.7 * sp_conf
            elif sp_label in {"fear"}:
                score -= 0.5 * sp_conf
            elif sp_label in {"calm"}:
                score += 0.2 * sp_conf

        # Face emotion (first face)
        if face_result is not None:
            faces = face_result.get("faces", [])
            if faces:
                face0 = faces[0]
                fe_label = str(face0.get("emotion", "neutral")).lower()
                fe_conf = float(face0.get("emotion_confidence", 0.5))
                if fe_label in {"happy", "joy"}:
                    score += 0.7 * fe_conf
                elif fe_label in {"sad", "sadness"}:
                    score -= 0.6 * fe_conf
                elif fe_label in {"angry", "anger"}:
                    score -= 0.7 * fe_conf
                elif fe_label in {"surprise"}:
                    score += 0.1 * fe_conf

        # Map score to mood label
        if score >= 0.8:
            mood = "energetic"
        elif score >= 0.3:
            mood = "calm"
        elif score <= -0.8:
            mood = "depressed"
        elif score <= -0.3:
            mood = "stressed"
        else:
            mood = "neutral"

        # Aggregate confidence as a sigmoid-like mapping
        confidence = float(min(0.99, max(0.5, 0.5 + abs(score) / 2.0)))

        return {"mood": mood, "confidence": confidence, "score": score}