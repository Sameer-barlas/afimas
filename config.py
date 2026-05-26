"""
config.py — Central Configuration File for AFIMAS
--------------------------------------------------
All project-wide settings are stored here.
Import this in any file that needs configuration values.
"""

import os

# ─────────────────────────────────────────────
#  BASE DIRECTORY
#  This gives us the absolute path of the project root
#  so all other paths are built correctly regardless of
#  where you run the app from.
# ─────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    """
    Main configuration class.
    All Flask app settings and custom paths live here.
    """

    # ── Flask Security ────────────────────────
    # SECRET_KEY is used to sign session cookies.
    # Change this to a long random string in production!
    SECRET_KEY = os.environ.get("SECRET_KEY", "afimas-secret-key-change-in-production")

    # ── Session Settings ──────────────────────
    # Sessions expire when the browser is closed
    SESSION_PERMANENT = False

    # ── Database ──────────────────────────────
    # SQLite database file path (stored in project root)
    DATABASE_PATH = os.path.join(BASE_DIR, "database.db")

    # ── Upload Folders ────────────────────────
    # Where face registration images are saved
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

    # Where mood scan capture images are saved
    MOOD_CAPTURE_FOLDER = os.path.join(BASE_DIR, "static", "mood_captures")

    # ── Allowed Image Extensions ──────────────
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    # ── Face Recognition ──────────────────────
    # Threshold for face match (lower = stricter match)
    # Euclidean distance below this value = match found
    FACE_MATCH_THRESHOLD = 0.50

    # ── Debug Mode ────────────────────────────
    # Set to False before deploying to production
    DEBUG = True


def create_upload_folders():
    """
    Creates upload folders if they don't already exist.
    Call this once when the app starts.
    """
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.MOOD_CAPTURE_FOLDER, exist_ok=True)