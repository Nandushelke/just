from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from text_analysis import analyze_text
from mood_predictor import infer_mood
from recommender import recommend_for_mood
from storage import init_db, insert_entry, fetch_entries


st.set_page_config(page_title="Emotion & Mood Analyzer", layout="wide")


@st.cache_resource
def _init_once():
    init_db()
    return True


_init_once()


st.title("Multi-Modal Sentiment, Emotion, Mood, and Age Detection")
st.caption("Text-only demo. Audio/video coming soon.")


with st.sidebar:
    st.header("About")
    st.write("Analyze text to infer sentiment, emotions, and a mood label. Save to your diary and view trends over time.")
    st.write("Models load on first use and may take a moment.")


tab_analyze, tab_diary = st.tabs(["Analyze", "Diary & Insights"])


with tab_analyze:
    st.subheader("Text Analysis")
    user_text = st.text_area("Enter text", height=150, placeholder="How are you feeling today?")
    colA, colB = st.columns([1, 2])
    with colA:
        do_analyze = st.button("Analyze")
        save_diary = st.button("Save to Diary", disabled=True)
    results_container = st.empty()

    if do_analyze and user_text.strip():
        analysis = analyze_text(user_text, include_raw=False)
        sentiment = analysis["sentiment"]
        emotions = analysis["emotions"]
        mood = infer_mood(sentiment, emotions)

        results_container.success(
            f"Sentiment: {sentiment['label']} (polarity {sentiment['polarity']:.2f}) — Mood: {mood['label']}"
        )

        # Show emotion probabilities
        emo_df = pd.DataFrame({"emotion": list(emotions.keys()), "prob": list(emotions.values())})
        emo_fig = px.bar(emo_df, x="emotion", y="prob", title="Emotion probabilities", range_y=[0, 1])
        st.plotly_chart(emo_fig, use_container_width=True)

        # Recommendations
        st.subheader("Recommendations")
        recs = recommend_for_mood(mood["label"]) 
        cols = st.columns(4)
        for idx, key in enumerate(["music", "quotes", "relaxation", "tips"]):
            with cols[idx]:
                st.markdown(f"**{key.capitalize()}**")
                for item in recs[key]:
                    st.write(f"- {item}")

        # Enable save button with current results
        with colA:
            save = st.button("Save to Diary", key="save_enabled")
        if save:
            insert_entry(
                text=user_text,
                sentiment_label=str(sentiment["label"]),
                sentiment_polarity=float(sentiment["polarity"]),
                emotions=emotions,
                mood_label=str(mood["label"]),
                mood_score=float(mood["score"]),
            )
            st.toast("Entry saved.")


with tab_diary:
    st.subheader("Timeline")
    entries = fetch_entries()
    if not entries:
        st.info("No entries yet. Analyze some text and save it.")
    else:
        df = pd.DataFrame(entries)
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # type: ignore
        # Mood score over time
        fig = px.line(df, x="timestamp", y="mood_score", markers=True, title="Mood score over time (-1..1)")
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment polarity histogram
        hist = px.histogram(df, x="sentiment_polarity", nbins=20, title="Sentiment polarity distribution")
        st.plotly_chart(hist, use_container_width=True)

        # Table
        with st.expander("Entries table"):
            st.dataframe(df.drop(columns=["emotions"]))

