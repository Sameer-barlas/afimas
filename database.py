"""
database.py — Database Initialization for AFIMAS
-------------------------------------------------
This file:
  1. Creates a connection to the SQLite database
  2. Defines all tables (users, face_encodings, mood_logs, recognition_logs)
  3. Provides a reusable get_db_connection() function

Run this file directly once to initialize the database:
    python database.py
"""

import sqlite3
from config import Config


# ─────────────────────────────────────────────
#  DATABASE CONNECTION HELPER
# ─────────────────────────────────────────────

def get_db_connection():
    """
    Opens and returns a new SQLite database connection.

    - row_factory = sqlite3.Row lets us access columns by name
      instead of index, e.g. row["email"] instead of row[2]

    Usage:
        conn = get_db_connection()
        rows = conn.execute("SELECT * FROM users").fetchall()
        conn.close()
    """
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


# ─────────────────────────────────────────────
#  TABLE CREATION
# ─────────────────────────────────────────────

def initialize_database():
    """
    Creates all required tables if they do not already exist.
    Safe to run multiple times — uses IF NOT EXISTS.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # ── Table 1: users ────────────────────────
    # Stores registered user account information
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    NOT NULL UNIQUE,
            password_hash TEXT  NOT NULL,
            age         INTEGER,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Table 2: face_encodings ───────────────
    # Stores face encodings as binary BLOB (no .pkl file needed)
    # One user can have multiple face encodings (different angles)
    # encoding_blob: numpy array converted to bytes via pickle.dumps()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            encoding_blob BLOB    NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # ── Table 3: mood_logs ────────────────────
    # Stores each mood/emotion scan result for a user
    # all_scores_json: stores all 7 emotion scores as JSON string
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            dominant_emotion TEXT    NOT NULL,
            confidence       REAL    NOT NULL,
            mood_category    TEXT,
            all_scores_json  TEXT,
            image_path       TEXT,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # ── Table 4: recognition_logs ─────────────
    # Stores each face recognition attempt
    # matched_user_id: the user whose face was matched (or NULL if unknown)
    # status: "matched", "unknown", or "error"
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER,
            matched_user_id INTEGER,
            distance        REAL,
            confidence      REAL,
            status          TEXT    NOT NULL DEFAULT 'unknown',
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)         REFERENCES users(id),
            FOREIGN KEY (matched_user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()

    print("[database] Database initialized successfully.")
    print(f"[database] Location: {Config.DATABASE_PATH}")


# ─────────────────────────────────────────────
#  RUN DIRECTLY TO INITIALIZE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    initialize_database()
