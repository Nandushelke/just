from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mood_diary.sqlite")


def ensure_dirs(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


@contextmanager
def db_conn(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    ensure_dirs(db_path)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    with db_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT,
                sentiment_label TEXT,
                sentiment_polarity REAL,
                emotion_json TEXT,
                mood_label TEXT,
                mood_score REAL
            )
            """
        )
        conn.commit()


def insert_entry(
    text: str,
    sentiment_label: str,
    sentiment_polarity: float,
    emotions: Dict[str, float],
    mood_label: str,
    mood_score: float,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    emotion_json = json_dumps_sorted(emotions)
    with db_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO entries (timestamp, text, sentiment_label, sentiment_polarity, emotion_json, mood_label, mood_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, text, sentiment_label, float(sentiment_polarity), emotion_json, mood_label, float(mood_score)),
        )
        conn.commit()


def fetch_entries(limit: Optional[int] = None, db_path: str = DEFAULT_DB_PATH) -> List[dict]:
    query = "SELECT timestamp, text, sentiment_label, sentiment_polarity, emotion_json, mood_label, mood_score FROM entries ORDER BY timestamp ASC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with db_conn(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(query).fetchall()
    results: List[dict] = []
    for ts, text, s_label, s_pol, e_json, m_label, m_score in rows:
        results.append(
            {
                "timestamp": ts,
                "text": text,
                "sentiment_label": s_label,
                "sentiment_polarity": float(s_pol),
                "emotions": json_loads_safe(e_json) or {},
                "mood_label": m_label,
                "mood_score": float(m_score),
            }
        )
    return results


def json_dumps_sorted(obj: dict) -> str:
    import json
    return json.dumps(obj, sort_keys=True)


def json_loads_safe(s: str) -> dict:
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}

