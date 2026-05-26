"""
services/auth_service.py — Authentication Business Logic
---------------------------------------------------------
Handles signup, login, and session management logic.
This layer sits between routes and repositories —
routes call service functions, services call repo functions.

Functions:
    register_user()   → Validate + hash password + create account
    login_user()      → Verify credentials + return user data
    get_current_user()→ Load full user object from session ID
    login_required()  → Decorator to protect routes from guests
"""

from functools import wraps
from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

from repositories.user_repo import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    email_exists
)


# ─────────────────────────────────────────────
#  SIGNUP
# ─────────────────────────────────────────────

def register_user(name, email, password, age):
    """
    Validates signup data, hashes the password, and creates the user.

    Args:
        name     (str): Full name
        email    (str): Email address
        password (str): Plain text password (will be hashed here)
        age      (int): User's age

    Returns:
        dict with keys:
            "success" (bool): Whether registration worked
            "message" (str):  Human-readable result message
            "user_id" (int):  New user's ID (only if success=True)
    """

    # ── Validation ────────────────────────────

    # Check all fields are filled
    if not all([name, email, password, age]):
        return {"success": False, "message": "All fields are required."}

    # Basic email format check
    if "@" not in email or "." not in email:
        return {"success": False, "message": "Please enter a valid email address."}

    # Password length check
    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters."}

    # Age validation
    try:
        age = int(age)
        if age < 1 or age > 120:
            raise ValueError
    except ValueError:
        return {"success": False, "message": "Please enter a valid age."}

    # Check if email is already registered
    if email_exists(email):
        return {"success": False, "message": "An account with this email already exists."}

    # ── Password Hashing ──────────────────────
    # Never store plain passwords — always hash them!
    # werkzeug uses PBKDF2-SHA256 by default (industry standard)
    hashed_password = generate_password_hash(password)

    # ── Create User ───────────────────────────
    user_id = create_user(name, email, hashed_password, age)

    if user_id:
        return {
            "success": True,
            "message": "Account created successfully! Please log in.",
            "user_id": user_id
        }
    else:
        return {"success": False, "message": "Registration failed. Please try again."}


# ─────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────

def login_user(email, password):
    """
    Verifies login credentials.
    Does NOT set the session here — the route does that
    after receiving the user object.

    Args:
        email    (str): User's email
        password (str): Plain text password to verify

    Returns:
        dict with keys:
            "success" (bool): Whether login succeeded
            "message" (str):  Human-readable result
            "user"    (Row):  User DB row (only if success=True)
    """

    # Basic field check
    if not email or not password:
        return {"success": False, "message": "Email and password are required."}

    # Fetch user from DB
    user = get_user_by_email(email)

    # User not found
    if not user:
        return {"success": False, "message": "No account found with that email."}

    # Verify password against the stored hash
    if not check_password_hash(user["password_hash"], password):
        return {"success": False, "message": "Incorrect password. Please try again."}

    # ✅ Login successful
    return {
        "success": True,
        "message": f"Welcome back, {user['name']}!",
        "user": user
    }


# ─────────────────────────────────────────────
#  SESSION HELPERS
# ─────────────────────────────────────────────

def get_current_user():
    """
    Loads the full user object using the ID stored in the session.
    Call this in any route to get the logged-in user's data.

    Returns:
        sqlite3.Row: Full user row from DB
        None: If no user is logged in
    """
    user_id = session.get("user_id")  # Get ID stored during login
    if not user_id:
        return None
    return get_user_by_id(user_id)


def is_logged_in():
    """
    Simple check — is a user currently logged in?

    Returns:
        bool: True if session has a user_id
    """
    return "user_id" in session


# ─────────────────────────────────────────────
#  LOGIN REQUIRED DECORATOR
# ─────────────────────────────────────────────

def login_required(f):
    """
    A decorator that protects Flask routes from unauthenticated access.

    Usage (in any route file):
        @bp.route("/dashboard")
        @login_required
        def dashboard():
            ...

    If the user is NOT logged in, they are redirected to the login page
    with a friendly flash message.
    """
    @wraps(f)  # Preserves the original function name (important for Flask)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth.login"))  # auth = Blueprint name
        return f(*args, **kwargs)
    return decorated_function