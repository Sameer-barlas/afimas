"""
repositories/recognition_repo.py — Recognition Log Database Operations
-----------------------------------------------------------------------
All database queries related to the 'recognition_logs' table.

Functions:
    save_recognition_log()          → Log a face recognition attempt
    get_recognition_logs_by_user()  → Get all logs for a specific user
    get_recent_recognitions()       → Get last N recognition events
    get_recognition_summary()       → Count matched vs unknown attempts
"""

from database import get_db_connection


# ─────────────────────────────────────────────
#  CREATE
# ─────────────────────────────────────────────

def save_recognition_log(user_id=None, matched_user_id=None,
                         distance=None, confidence=None, status="unknown"):
    """
    Saves a face recognition attempt into 'recognition_logs'.

    Args:
        user_id         (int):   Who initiated the recognition (or None if unknown)
        matched_user_id (int):   Who was matched (or None if no match)
        distance        (float): Euclidean distance between encodings
                                 Lower = more similar faces
        confidence      (float): Confidence score derived from distance (0–100%)
        status          (str):   "matched", "unknown", or "error"

    Returns:
        bool: True if saved successfully, False on error
    """
    try:
        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO recognition_logs
                (user_id, matched_user_id, distance, confidence, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                matched_user_id,
                round(distance, 4) if distance is not None else None,
                round(confidence, 2) if confidence is not None else None,
                status
            )
        )
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[recognition_repo] Error saving log: {e}")
        return False


# ─────────────────────────────────────────────
#  READ
# ─────────────────────────────────────────────

def get_recognition_logs_by_user(user_id):
    """
    Fetches all recognition logs where this user was the initiator
    OR was the matched person.
    Results ordered newest first.

    Args:
        user_id (int): The logged-in user's ID

    Returns:
        list[dict]: Recognition log entries with matched user's name joined
    """
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT
            rl.id,
            rl.status,
            rl.distance,
            rl.confidence,
            rl.created_at,
            u.name AS matched_user_name
        FROM recognition_logs rl
        LEFT JOIN users u ON rl.matched_user_id = u.id
        WHERE rl.user_id = ? OR rl.matched_user_id = ?
        ORDER BY rl.created_at DESC
        """,
        (user_id, user_id)
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_recent_recognitions(user_id, limit=5):
    """
    Fetches the most recent N recognition events for a user.
    Used on the dashboard for a quick activity feed.

    Args:
        user_id (int): The logged-in user's ID
        limit   (int): Number of entries to return (default 5)

    Returns:
        list[dict]: Recent recognition log entries
    """
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT
            rl.status,
            rl.confidence,
            rl.created_at,
            u.name AS matched_user_name
        FROM recognition_logs rl
        LEFT JOIN users u ON rl.matched_user_id = u.id
        WHERE rl.user_id = ? OR rl.matched_user_id = ?
        ORDER BY rl.created_at DESC
        LIMIT ?
        """,
        (user_id, user_id, limit)
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_recognition_summary(user_id):
    """
    Returns a count of "matched" vs "unknown" recognition attempts
    for a user. Used for analytics charts.

    Args:
        user_id (int): The logged-in user's ID

    Returns:
        dict: e.g. {"matched": 15, "unknown": 4, "error": 1}
    """
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT status, COUNT(*) as count
        FROM recognition_logs
        WHERE user_id = ? OR matched_user_id = ?
        GROUP BY status
        """,
        (user_id, user_id)
    ).fetchall()
    conn.close()

    summary = {}
    for row in rows:
        summary[row["status"]] = row["count"]

    return summary