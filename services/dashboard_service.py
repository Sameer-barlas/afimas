"""
services/dashboard_service.py — Dashboard & Analytics Logic
------------------------------------------------------------
This replaces the old utils/analyticsutils.py.
Combines data from mood_repo and recognition_repo to build
the statistics and chart data needed by the dashboard
and analytics pages.

Functions:
    get_dashboard_summary()    → Stats cards data for dashboard
    get_analytics_data()       → Full chart data for analytics page
    get_mood_chart_data()      → Emotion distribution for pie/bar chart
    get_mood_trend_data()      → Mood over time for line chart
    get_recognition_chart_data()→ Matched vs unknown for doughnut chart
"""

import json
from repositories.mood_repo import (
    get_mood_logs_by_user,
    get_recent_moods,
    get_mood_summary,
    get_total_scans
)
from repositories.recognition_repo import (
    get_recent_recognitions,
    get_recognition_summary
)
from repositories.face_repo import has_face_registered


# ─────────────────────────────────────────────
#  DASHBOARD SUMMARY (Stats Cards)
# ─────────────────────────────────────────────

def get_dashboard_summary(user_id):
    """
    Builds a summary dictionary for the main dashboard page.
    Powers the stats cards shown at the top of the dashboard.

    Args:
        user_id (int): Logged-in user's ID

    Returns:
        dict:
            total_scans       (int):  Total mood scans done
            most_common_mood  (str):  Most frequent emotion detected
            face_registered   (bool): Whether face is registered
            recent_moods      (list): Last 5 mood logs
            recent_recognitions(list):Last 5 recognition events
    """
    # Total mood scans
    total_scans = get_total_scans(user_id)

    # Most common mood
    mood_summary = get_mood_summary(user_id)
    if mood_summary:
        # Get the emotion with the highest count
        most_common_mood = max(mood_summary, key=mood_summary.get)
    else:
        most_common_mood = "No scans yet"

    # Whether face is already registered
    face_registered = has_face_registered(user_id)

    # Recent mood & recognition activity
    recent_moods        = get_recent_moods(user_id, limit=5)
    recent_recognitions = get_recent_recognitions(user_id, limit=5)

    return {
        "total_scans":        total_scans,
        "most_common_mood":   most_common_mood,
        "face_registered":    face_registered,
        "recent_moods":       recent_moods,
        "recent_recognitions": recent_recognitions
    }


# ─────────────────────────────────────────────
#  MOOD CHART DATA
# ─────────────────────────────────────────────

def get_mood_chart_data(user_id):
    """
    Prepares emotion distribution data for Chart.js.
    Used in the pie chart / bar chart on the analytics page.

    Args:
        user_id (int): Logged-in user's ID

    Returns:
        dict:
            labels  (list[str]):   Emotion names  e.g. ["happy", "sad"]
            values  (list[int]):   Counts          e.g. [10, 3]
            colors  (list[str]):   Hex color codes for each emotion
    """
    mood_summary = get_mood_summary(user_id)

    # Color palette for each emotion (consistent across all charts)
    EMOTION_COLORS = {
        "happy":    "#FFD93D",  # Yellow
        "neutral":  "#6BCB77",  # Green
        "sad":      "#4D96FF",  # Blue
        "angry":    "#FF6B6B",  # Red
        "fear":     "#C77DFF",  # Purple
        "surprise": "#FF9F43",  # Orange
        "disgust":  "#A8A8A8"   # Grey
    }

    labels = list(mood_summary.keys())
    values = list(mood_summary.values())
    colors = [EMOTION_COLORS.get(emotion, "#CCCCCC") for emotion in labels]

    return {
        "labels": labels,
        "values": values,
        "colors": colors
    }


# ─────────────────────────────────────────────
#  MOOD TREND DATA (Line Chart)
# ─────────────────────────────────────────────

def get_mood_trend_data(user_id, limit=10):
    """
    Prepares the last N mood scans as time-series data
    for a line chart showing mood changes over time.

    Args:
        user_id (int): Logged-in user's ID
        limit   (int): Number of recent scans to include

    Returns:
        dict:
            dates      (list[str]):   Formatted date strings
            emotions   (list[str]):   Emotion at each point
            confidences(list[float]): Confidence at each point
    """
    logs = get_mood_logs_by_user(user_id)

    # Take only the most recent `limit` entries and reverse
    # so the chart goes left (oldest) → right (newest)
    recent_logs = logs[:limit][::-1]

    dates       = []
    emotions    = []
    confidences = []

    for log in recent_logs:
        # Format: "May 17" style date
        created_at = log["created_at"]
        # SQLite returns timestamps as strings: "2025-05-17 10:30:00"
        # We take just the date portion and reformat
        try:
            from datetime import datetime
            dt = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S")
            dates.append(dt.strftime("%b %d, %H:%M"))
        except Exception:
            dates.append(created_at[:10])

        emotions.append(log["dominant_emotion"].capitalize())
        confidences.append(round(log["confidence"] * 100, 1))

    return {
        "dates":       dates,
        "emotions":    emotions,
        "confidences": confidences
    }


# ─────────────────────────────────────────────
#  RECOGNITION CHART DATA (Doughnut Chart)
# ─────────────────────────────────────────────

def get_recognition_chart_data(user_id):
    """
    Prepares recognition attempt stats for a doughnut chart.
    Shows how many attempts were matched vs unknown.

    Args:
        user_id (int): Logged-in user's ID

    Returns:
        dict:
            labels (list[str]):  e.g. ["Matched", "Unknown", "Error"]
            values (list[int]):  Counts for each status
            colors (list[str]):  Color codes for each segment
    """
    summary = get_recognition_summary(user_id)

    STATUS_COLORS = {
        "matched": "#6BCB77",  # Green
        "unknown": "#FF6B6B",  # Red
        "error":   "#A8A8A8"   # Grey
    }

    labels = [key.capitalize() for key in summary.keys()]
    values = list(summary.values())
    colors = [STATUS_COLORS.get(key, "#CCCCCC") for key in summary.keys()]

    return {
        "labels": labels,
        "values": values,
        "colors": colors
    }


# ─────────────────────────────────────────────
#  FULL ANALYTICS PAGE DATA
# ─────────────────────────────────────────────

def get_analytics_data(user_id):
    """
    Combines all chart data into a single dictionary
    for the analytics page template.

    Args:
        user_id (int): Logged-in user's ID

    Returns:
        dict: All chart data + full mood log history
    """
    mood_summary = get_mood_summary(user_id)
    recognition_chart = get_recognition_chart_data(user_id)

    top_emotion = None
    if mood_summary:
        top_emotion = max(mood_summary, key=mood_summary.get)

    return {
        "mood_chart":        get_mood_chart_data(user_id),
        "mood_trend":        get_mood_trend_data(user_id, limit=10),
        "recognition_chart": recognition_chart,
        "full_mood_logs":    get_mood_logs_by_user(user_id),
        "total_scans":       get_total_scans(user_id),
        "mood_summary":      mood_summary,
        "top_emotion":       top_emotion,
        "recognition_total": sum(recognition_chart["values"])
    }
