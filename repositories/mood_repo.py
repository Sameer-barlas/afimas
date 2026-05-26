"""
repositories/mood_repo.py — Mood Log Database Operations
---------------------------------------------------------
All database queries related to the 'mood_logs' table.

Functions:
    save_mood_log()          → Save a new mood scan result
    get_mood_logs_by_user()  → Get all mood logs for a user
    get_recent_moods()       → Get last N mood logs for a user
    get_mood_summary()       → Count of each emotion for a user
    get_total_scans()        → Total number of scans by a user
"""

import json
from database import get_db_connection


# ─────────────────────────────────────────────
#  CREATE
# ─────────────────────────────────────────────

def save_mood_log(user_id, dominant_emotion, confidence,
                  mood_category, all_scores_dict, image_path=None):
    """
    Saves a mood scan result into the 'mood_logs' table.

    Args:
        user_id           (int):   Logged-in user's ID
        dominant_emotion  (str):   e.g. "happy", "sad", "angry"
        confidence        (float): Confidence score (0.0 to 1.0)
        mood_category     (str):   e.g. "Positive", "Negative", "Neutral"
        all_scores_dict   (dict):  All 7 emotion scores as a dictionary
                                   e.g. {"happy": 0.9, "sad": 0.05, ...}
        image_path        (str):   Optional path to saved mood image

    Returns:
        bool: True if saved successfully, False on error
    """
    try:
        # Convert the scores dictionary → JSON string for DB storage
        all_scores_json = json.dumps(all_scores_dict)

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO mood_logs
                (user_id, dominant_emotion, confidence,
                 mood_category, all_scores_json, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, dominant_emotion, round(confidence, 4),
             mood_category, all_scores_json, image_path)
        )
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[mood_repo] Error saving mood log: {e}")
        return False


# ─────────────────────────────────────────────
#  READ
# ─────────────────────────────────────────────

def get_mood_logs_by_user(user_id):
    """
    Fetches all mood scan logs for a specific user.
    Results are ordered newest first.

    Args:
        user_id (int): The logged-in user's ID

    Returns:
        list[dict]: Each item contains all mood log fields,
                    with all_scores_json already parsed back to dict
    """
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, user_id, dominant_emotion, confidence,
               mood_category, all_scores_json, image_path, created_at
        FROM mood_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,)
    ).fetchall()
    conn.close()

    # Convert each sqlite3.Row → plain dict and parse JSON scores
    logs = []
    for row in rows:
        log = dict(row)
        # Convert JSON string back to dictionary
        if log["all_scores_json"]:
            log["all_scores"] = json.loads(log["all_scores_json"])
        else:
            log["all_scores"] = {}
        logs.append(log)

    return logs


def get_recent_moods(user_id, limit=5):
    """
    Fetches the most recent N mood logs for a user.
    Used on the dashboard for a quick overview.

    Args:
        user_id (int): The logged-in user's ID
        limit   (int): How many recent entries to return (default 5)

    Returns:
        list[dict]: Recent mood log entries
    """
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT dominant_emotion, confidence, mood_category, created_at
        FROM mood_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_mood_summary(user_id):
    """
    Returns the count of each dominant emotion for a user.
    Used for the analytics chart (e.g. pie chart / bar chart).

    Args:
        user_id (int): The logged-in user's ID

    Returns:
        dict: Emotion → count mapping
              e.g. {"happy": 10, "sad": 3, "neutral": 7}
    """
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT dominant_emotion, COUNT(*) as count
        FROM mood_logs
        WHERE user_id = ?
        GROUP BY dominant_emotion
        ORDER BY count DESC
        """,
        (user_id,)
    ).fetchall()
    conn.close()

    # Build a clean dictionary from the results
    summary = {}
    for row in rows:
        summary[row["dominant_emotion"]] = row["count"]

    return summary


def get_total_scans(user_id):
    """
    Returns total number of mood scans performed by a user.

    Args:
        user_id (int): The logged-in user's ID

    Returns:
        int: Total scan count
    """
    conn = get_db_connection()
    result = conn.execute(
        "SELECT COUNT(*) as total FROM mood_logs WHERE user_id = ?",
        (user_id,)
    ).fetchone()
    conn.close()

    return result["total"] if result else 0