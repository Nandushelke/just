# Multi-Modal Sentiment, Emotion, Mood, and Age Detection System

An AI-driven system that analyzes text, speech, and facial expressions to detect sentiment, emotions, mood, and approximate age. Includes a visualization dashboard, context-aware tracking, recommendations, and a personal mood diary.

## Features
- Sentiment analysis (text)
- Emotion detection (text, speech, face)
- Mood prediction (multi-modal fusion)
- Age estimation (face)
- Context-aware emotion tracking over time
- Visualization dashboard (real-time charts)
- Recommendation system (music, quotes, relaxation, tips)
- Personal mood diary (secure storage + weekly/monthly insights)

## Tech Stack
- Python (primary), optional JS for frontend
- NLP: transformers, nltk, spacy
- DL: PyTorch / TensorFlow (using PyTorch by default)
- CV: OpenCV, facenet-pytorch
- Audio: librosa, pyaudio
- Web UI: Streamlit or Flask (Streamlit default)
- Viz: matplotlib, seaborn, plotly

## Project Structure
```
Sentiment-Emotion-Mood-Age-Detection/
├── data/                  # datasets (raw + processed)
├── models/                # trained models
├── notebooks/             # Jupyter experiments
├── src/
│   ├── text_analysis.py   # sentiment & emotion from text
│   ├── speech_analysis.py # emotion detection from audio
│   ├── face_analysis.py   # age + emotion from images
│   ├── mood_predictor.py  # combine multi-modal outputs
│   ├── recommender.py     # personalized recommendations
│   └── app.py             # Streamlit/Flask main app
├── requirements.txt
└── README.md
```

## Setup
1. Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt vader_lexicon stopwords
python -m spacy download en_core_web_sm
```

3. (Optional) Verify GPU support
```bash
python - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
PY
```

## Running the App
Streamlit (default):
```bash
streamlit run src/app.py
```

Flask (alternative, if you switch):
```bash
export FLASK_APP=src/app.py
flask run --reload
```

## Data
- Place datasets in `data/`. Add subfolders as needed, e.g., `data/text/`, `data/audio/`, `data/images/`.
- Trained models and weights go under `models/`.

## Notes
- First run will download transformer models and spaCy resources.
- Microphone access for speech features may require OS permissions.
- Webcam access for face features may require OS permissions.

## Future Extensions
- Real-time video emotion/age from webcam
- Cross-cultural adaptation
- Multi-language support (mBERT, XLM-R)
- Stress level detector (multimodal)