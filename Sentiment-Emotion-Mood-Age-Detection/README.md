Multi-Modal Sentiment, Emotion, Mood, and Age Detection System

Overview

This project analyzes text, speech, and facial expressions to infer sentiment, emotions, mood, and approximate age. It features a Streamlit app, a local mood diary, and a simple recommendation engine that suggests music, quotes, relaxation techniques, and study/work tips based on detected moods.

Current Status

- Text analysis (sentiment + emotion) implemented using Transformers
- Heuristic mood aggregator from text signals
- Recommendation system for music, quotes, relaxation, tips
- SQLite-based mood diary with a timeline dashboard in Streamlit
- Stubs for speech and face analysis (to be implemented)

Project Structure

Sentiment-Emotion-Mood-Age-Detection/
├── data/                  # datasets (raw + processed) and local SQLite DB
├── models/                # trained models (optional)
├── notebooks/             # Jupyter experiments
├── src/
│   ├── text_analysis.py   # sentiment & emotion from text (implemented)
│   ├── speech_analysis.py # emotion detection from audio (stub)
│   ├── face_analysis.py   # age + emotion from images (stub)
│   ├── mood_predictor.py  # combine multi-modal outputs (implemented for text)
│   ├── recommender.py     # personalized recommendations (implemented)
│   ├── storage.py         # SQLite mood diary utilities
│   └── app.py             # Streamlit main app
├── requirements.txt
└── README.md

Quickstart

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the Streamlit app

```bash
streamlit run src/app.py
```

4) Usage (Text-only workflow for now)

- Enter text in the input area
- Click Analyze to get sentiment, emotion, and mood
- Save to diary to persist the result
- View timeline charts and recommendations

Notes

- The first run will download transformer models automatically.
- Audio (speech) and image/video (face) analysis are scaffolded and can be implemented next.
- For librosa-based audio features, you may need system libs (e.g., libsndfile). If `pyaudio` installation fails, ensure PortAudio development headers are installed on your system.

License

For research and educational purposes. Review individual dataset and model licenses before production use.

