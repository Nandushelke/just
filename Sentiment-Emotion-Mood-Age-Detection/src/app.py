from __future__ import annotations

import io
import os
import json
import tempfile
from datetime import datetime
from typing import Any, Dict

import streamlit as st

from text_analysis import analyze_text, TextAnalyzer
from speech_analysis import SpeechEmotionAnalyzer
from face_analysis import FaceAnalyzer
from mood_predictor import MoodPredictor
from recommender import Recommender


st.set_page_config(page_title="Multi-Modal Emotion & Age", layout="wide")

if "diary" not in st.session_state:
    st.session_state.diary = []  # list of entries


def add_diary_entry(entry: Dict[str, Any]) -> None:
    st.session_state.diary.append(entry)


st.title("Multi-Modal Sentiment, Emotion, Mood, and Age Detection")

with st.sidebar:
    st.header("Inputs")
    input_text = st.text_area("Text", placeholder="Type something to analyze…")
    audio_file = st.file_uploader("Speech (WAV/MP3)", type=["wav", "mp3", "ogg"])
    image_file = st.file_uploader("Face Image", type=["jpg", "jpeg", "png"])
    analyze_button = st.button("Analyze")

col1, col2, col3 = st.columns(3)

text_result: Dict[str, Dict[str, float | str]] | None = None
speech_result: Dict[str, float | str] | None = None
face_result: Dict[str, list[Dict[str, float | int | str]]] | None = None

if analyze_button:
    with st.spinner("Analyzing… this may download models on first run"):
        # Text analysis
        if input_text.strip():
            text_result = analyze_text(input_text.strip())
        # Speech analysis
        if audio_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix="_audio") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                speech_result = SpeechEmotionAnalyzer().analyze_file(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        # Face analysis
        if image_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(image_file.read())
                    tmp_path = tmp.name
                face_result = FaceAnalyzer().analyze_image(tmp_path)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    mood_pred = MoodPredictor().predict_mood(text_result, speech_result, face_result)

    # Determine primary emotion for recommendations
    primary_emotion = "neutral"
    if text_result is not None:
        primary_emotion = str(text_result.get("emotion", {}).get("label", "neutral"))
    if face_result is not None and face_result.get("faces"):
        primary_emotion = str(face_result["faces"][0].get("emotion", primary_emotion))

    recs = Recommender().recommend(mood_pred.get("mood", "neutral"), primary_emotion)

    # Display
    with col1:
        st.subheader("Text")
        st.write(text_result or {})
    with col2:
        st.subheader("Speech")
        st.write(speech_result or {})
    with col3:
        st.subheader("Face")
        st.write(face_result or {})

    st.subheader("Mood Prediction")
    st.write(mood_pred)

    st.subheader("Recommendations")
    st.json(recs)

    # Save to diary
    add_diary_entry(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "text": input_text,
            "text_result": text_result,
            "speech_result": speech_result,
            "face_result": face_result,
            "mood": mood_pred,
            "recommendations": recs,
        }
    )

st.markdown("---")

st.header("Personal Mood Diary")
if st.session_state.diary:
    st.write(f"Entries: {len(st.session_state.diary)}")
    st.dataframe(st.session_state.diary, use_container_width=True)
    diary_json = json.dumps(st.session_state.diary, indent=2)
    st.download_button(
        label="Download Diary (JSON)",
        data=io.BytesIO(diary_json.encode("utf-8")),
        file_name="mood_diary.json",
        mime="application/json",
    )
else:
    st.caption("No entries yet. Run an analysis to add one.")