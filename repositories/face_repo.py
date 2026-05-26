"""
repositories/face_repo.py — Face Encoding Database Operations
--------------------------------------------------------------
All database queries related to the 'face_encodings' table.

Key concept:
    Face encodings are numpy arrays (128 numbers per face).
    We convert them to bytes using pickle.dumps() before saving.
    We convert them back using pickle.loads() when reading.
    This replaces the old encodings.pkl file completely.

Functions:
    save_face_encoding()         → Store a face encoding for a user
    get_encodings_for_user()     → Get all encodings for one user
    get_all_encodings_with_ids() → Get all encodings from all users
    has_face_registered()        → Check if user has registered a face
    delete_encodings_for_user()  → Remove all encodings for a user
"""

import pickle
from database import get_db_connection


# ─────────────────────────────────────────────
#  CREATE
# ─────────────────────────────────────────────

def save_face_encoding(user_id, encoding_numpy_array):
    """
    Saves a face encoding into the database as a binary BLOB.

    How it works:
        numpy array → pickle.dumps() → bytes → stored in DB as BLOB

    Args:
        user_id              (int):          The user this encoding belongs to
        encoding_numpy_array (numpy.ndarray): The 128-number face encoding

    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Convert numpy array → bytes (binary format for DB storage)
        encoding_bytes = pickle.dumps(encoding_numpy_array)

        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO face_encodings (user_id, encoding_blob)
            VALUES (?, ?)
            """,
            (user_id, encoding_bytes)
        )
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[face_repo] Error saving encoding: {e}")
        return False


# ─────────────────────────────────────────────
#  READ
# ─────────────────────────────────────────────

def get_encodings_for_user(user_id):
    """
    Returns all face encodings belonging to a specific user.
    Used when you want to add more encodings or verify a specific user.

    Args:
        user_id (int): The user whose encodings to fetch

    Returns:
        list[numpy.ndarray]: List of face encoding arrays
                             Empty list if none found
    """
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT encoding_blob FROM face_encodings WHERE user_id = ?",
        (user_id,)
    ).fetchall()
    conn.close()

    # Convert each BLOB back to numpy array using pickle.loads()
    encodings = []
    for row in rows:
        encoding = pickle.loads(row["encoding_blob"])
        encodings.append(encoding)

    return encodings


def get_all_encodings_with_ids():
    """
    Returns all face encodings from ALL users in the database.
    Used during face recognition — we compare the uploaded face
    against every stored encoding to find a match.

    Returns:
        list[dict]: Each item has:
            - "user_id"  (int):          Who this encoding belongs to
            - "encoding" (numpy.ndarray): The 128-number face encoding
    """
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT user_id, encoding_blob FROM face_encodings"
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        encoding = pickle.loads(row["encoding_blob"])
        results.append({
            "user_id": row["user_id"],
            "encoding": encoding
        })

    return results


def has_face_registered(user_id):
    """
    Checks if the user has at least one face encoding stored.
    Used to show "Register Face" prompt if not done yet.

    Args:
        user_id (int): The user to check

    Returns:
        bool: True if at least one encoding exists
    """
    conn = get_db_connection()
    result = conn.execute(
        "SELECT id FROM face_encodings WHERE user_id = ? LIMIT 1",
        (user_id,)
    ).fetchone()
    conn.close()
    return result is not None


# ─────────────────────────────────────────────
#  DELETE
# ─────────────────────────────────────────────

def delete_encodings_for_user(user_id):
    """
    Deletes all face encodings for a user.
    Useful if the user wants to re-register their face.

    Args:
        user_id (int): The user whose encodings to delete

    Returns:
        bool: True if deleted successfully
    """
    try:
        conn = get_db_connection()
        conn.execute(
            "DELETE FROM face_encodings WHERE user_id = ?",
            (user_id,)
        )
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[face_repo] Error deleting encodings: {e}")
        return False