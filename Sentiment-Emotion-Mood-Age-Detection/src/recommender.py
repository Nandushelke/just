from __future__ import annotations

import random
from typing import Dict, List


_RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    "calm": {
        "music": [
            "Lo-fi beats to relax/study",
            "Peaceful Piano",
            "Ambient Chill",
        ],
        "quotes": [
            "Peace begins with a smile. – Mother Teresa",
            "The nearer a man comes to a calm mind, the closer he is to strength. – Marcus Aurelius",
            "Calm is a superpower.",
        ],
        "relaxation": [
            "10-minute mindful breathing",
            "Light stretching or yoga",
            "Take a short nature walk",
        ],
        "tips": [
            "Batch tasks and work in focused sprints",
            "Tidy your workspace for clarity",
            "Set a single clear priority for the next hour",
        ],
    },
    "energetic": {
        "music": [
            "Power Workout",
            "Uplifting Pop",
            "Driving Instrumentals",
        ],
        "quotes": [
            "Energy and persistence conquer all things. – Benjamin Franklin",
            "Act as if what you do makes a difference. It does. – William James",
            "The future depends on what you do today. – Gandhi",
        ],
        "relaxation": [
            "5-minute breath of fire (advanced)",
            "Quick jumping jacks or a brisk walk",
            "Channel energy into a short creative task",
        ],
        "tips": [
            "Tackle the hardest task first",
            "Time-block with short, intense intervals",
            "Avoid context switching; queue tasks",
        ],
    },
    "stressed": {
        "music": [
            "Deep Focus",
            "Gentle Classical",
            "Rain Sounds",
        ],
        "quotes": [
            "In the middle of difficulty lies opportunity. – Albert Einstein",
            "This too shall pass.",
            "You don’t have to control your thoughts; you just have to stop letting them control you. – Dan Millman",
        ],
        "relaxation": [
            "Box breathing: 4-4-4-4 cycles",
            "Progressive muscle relaxation",
            "Write down top worries and one next step",
        ],
        "tips": [
            "Reduce scope; define a minimum viable outcome",
            "Turn off notifications for 30 minutes",
            "Ask for help or delegate one item",
        ],
    },
    "depressed": {
        "music": [
            "Uplifting Acoustic",
            "Morning Motivation",
            "Soft Indie",
        ],
        "quotes": [
            "Once you choose hope, anything is possible. – Christopher Reeve",
            "If you’re going through hell, keep going. – Winston Churchill",
            "You are braver than you believe. – A. A. Milne",
        ],
        "relaxation": [
            "Gentle 5-minute walk and sunlight",
            "Warm beverage and slow breathing",
            "Call or message a supportive friend",
        ],
        "tips": [
            "Break tasks into tiny steps and start with one",
            "Do one self-care action (hydration, shower, stretch)",
            "Limit rumination; schedule a worry time",
        ],
    },
    "neutral": {
        "music": [
            "Study Lo-fi",
            "Instrumental Chill",
            "Calm Guitar",
        ],
        "quotes": [
            "What we think, we become. – Buddha",
            "Well begun is half done. – Aristotle",
            "Small deeds done are better than great deeds planned. – Peter Marshall",
        ],
        "relaxation": [
            "3-minute mindful pause",
            "Shoulder rolls and neck stretches",
            "Declutter one small area",
        ],
        "tips": [
            "Set an achievable goal for the next 25 minutes",
            "Review priorities and pick one task",
            "Plan a small reward for finishing",
        ],
    },
}


def recommend_for_mood(mood_label: str, seed: int | None = None, k: int = 3) -> dict:
    rng = random.Random(seed)
    bucket = _RECOMMENDATIONS.get(mood_label, _RECOMMENDATIONS["neutral"])
    def sample(items: List[str]) -> List[str]:
        if len(items) <= k:
            return items
        return rng.sample(items, k)
    return {
        "music": sample(bucket["music"]),
        "quotes": sample(bucket["quotes"]),
        "relaxation": sample(bucket["relaxation"]),
        "tips": sample(bucket["tips"]),
    }

