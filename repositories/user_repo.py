"""
repositories/user_repo.py — User Database Operations
------------------------------------------------------
All database queries related to the 'users' table live here.
No business logic — just raw DB read/write operations.

Functions:
    create_user()       → Insert a new user
    get_user_by_email() → Find user by email (used for login)
    get_user_by_id()    → Find user by their ID (used for session)
    email_exists()      → Check if email is already registered
"""

from database import get_db_connection


# ─────────────────────────────────────────────
#  CREATE
# ─────────────────────────────────────────────

def create_user(name, email, password_hash, age):
    """
    Inserts a new user into the 'users' table.

    Args:
        name          (str): Full name of the user
        email         (str): User's email address (must be unique)
        password_hash (str): Hashed password (from werkzeug)
        age           (int): User's age

    Returns:
        int: The newly created user's ID
        None: If insertion failed (e.g. duplicate email)
    """
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            """
            INSERT INTO users (name, email, password_hash, age)
            VALUES (?, ?, ?, ?)
            """,
            (name, email, password_hash, age)
        )
        conn.commit()
        new_user_id = cursor.lastrowid  # Get the ID of the inserted row
        return new_user_id

    except Exception as e:
        print(f"[user_repo] Error creating user: {e}")
        return None

    finally:
        conn.close()


# ─────────────────────────────────────────────
#  READ
# ─────────────────────────────────────────────

def get_user_by_email(email):
    """
    Fetches a user row by email address.
    Used during login to verify credentials.

    Args:
        email (str): The email to search for

    Returns:
        sqlite3.Row: User row (access like a dict, e.g. user["name"])
        None: If no user found with that email
    """
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()
    conn.close()
    return user


def get_user_by_id(user_id):
    """
    Fetches a user row by their ID.
    Used after login to load user info from session.

    Args:
        user_id (int): The user's primary key

    Returns:
        sqlite3.Row: User row
        None: If no user found with that ID
    """
    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    conn.close()
    return user


def email_exists(email):
    """
    Checks if a given email is already registered.
    Used during signup to prevent duplicate accounts.

    Args:
        email (str): The email to check

    Returns:
        bool: True if email already exists, False otherwise
    """
    conn = get_db_connection()
    result = conn.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,)
    ).fetchone()
    conn.close()
    return result is not None  # True if a row was found