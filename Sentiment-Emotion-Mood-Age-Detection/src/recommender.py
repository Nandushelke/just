from __future__ import annotations

from typing import Dict, List


class Recommender:
    """
    Simple rule-based recommender that maps detected mood/emotion to suggestions.
    """

    def recommend(self, mood: str, emotion: str | None = None) -> Dict[str, List[str]]:
        mood = (mood or "").lower()
        emotion = (emotion or mood or "").lower()

        music = {
            "energetic": ["Upbeat Pop Mix", "EDM Booster", "Morning Run Beats"],
            "calm": ["Lo-Fi Chill", "Ambient Focus", "Acoustic Calm"],
            "stressed": ["Piano Relax", "Deep Focus", "Nature Sounds"],
            "depressed": ["Gentle Uplift", "Soft Indie", "Warm Acoustic"],
            "neutral": ["Daily Mix", "Chillhop Essentials", "Indie Discovery"],
        }

        quotes = {
            "energetic": [
                "The future depends on what you do today.",
                "Action is the foundational key to all success.",
            ],
            "calm": [
                "Peace comes from within.",
                "Almost everything will work again if you unplug it for a few minutes.",
            ],
            "stressed": [
                "You don’t have to control your thoughts. You just have to stop letting them control you.",
                "Simplicity is the ultimate sophistication.",
            ],
            "depressed": [
                "No dark night lasts forever.",
                "You are stronger than you think.",
            ],
            "neutral": [
                "Small steps every day.",
                "Do what you can, with what you have, where you are.",
            ],
        }

        relaxation = {
            "energetic": ["10-min stretch", "Box breathing 2 min"],
            "calm": ["Body scan 5 min", "4-7-8 breathing 3 min"],
            "stressed": ["Guided meditation 5 min", "Progressive muscle relaxation"],
            "depressed": ["Sunlight walk 10 min", "Gratitude journaling 3 prompts"],
            "neutral": ["Mindful tea break", "Light stretching 5 min"],
        }

        tips = {
            "energetic": ["Tackle deep work first", "Channel energy into a short sprint"],
            "calm": ["Good time for reading/study", "Batch shallow tasks"],
            "stressed": ["Timebox work in 25-min Pomodoros", "Limit notifications 1 hour"],
            "depressed": ["Start with one tiny task", "Pair with a friend for accountability"],
            "neutral": ["Plan next 3 priorities", "Declutter your workspace 5 min"],
        }

        selected_mood = mood if mood in music else "neutral"

        return {
            "music_playlists": music[selected_mood],
            "motivational_quotes": quotes[selected_mood],
            "relaxation": relaxation[selected_mood],
            "study_work_tips": tips[selected_mood],
        }