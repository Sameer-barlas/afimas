"""
routes/auth_routes.py — Authentication Routes (Signup / Login / Logout)
------------------------------------------------------------------------
This Blueprint handles all URL routes related to user accounts.
Uses Flask Blueprints so routes stay modular and separate from app.py.

Routes:
    GET  /signup  → Show signup form
    POST /signup  → Process signup form submission
    GET  /login   → Show login form
    POST /login   → Process login form submission
    GET  /logout  → Clear session and redirect to login
"""

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session, flash
)
from services.auth_service import register_user, login_user

# ─────────────────────────────────────────────
#  CREATE BLUEPRINT
#  name="auth" → used in url_for("auth.login"), url_for("auth.signup")
#  url_prefix="/auth" → all routes here start with /auth/...
#  Change url_prefix="" if you want /login instead of /auth/login
# ─────────────────────────────────────────────
bp = Blueprint("auth", __name__, url_prefix="/auth")


# ─────────────────────────────────────────────
#  SIGNUP
# ─────────────────────────────────────────────

@bp.route("/signup", methods=["GET", "POST"])
def signup():
    """
    GET  → Render the signup form page
    POST → Process the submitted form data
    """

    # If already logged in, redirect to dashboard
    if "user_id" in session:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        # Read form fields from the submitted form
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        age      = request.form.get("age", "").strip()

        # Call the service layer to validate + create user
        result = register_user(name, email, password, age)

        if result["success"]:
            # Registration worked → redirect to login with success message
            flash(result["message"], "success")
            return redirect(url_for("auth.login"))
        else:
            # Something went wrong → show error on the same page
            flash(result["message"], "danger")
            return render_template("auth/signup.html",
                                   # Pass back values so form isn't cleared
                                   name=name, email=email, age=age)

    # GET request → just show the empty signup form
    return render_template("auth/signup.html")


# ─────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────

@bp.route("/login", methods=["GET", "POST"])
def login():
    """
    GET  → Render the login form page
    POST → Verify credentials and start session
    """

    # If already logged in, redirect to dashboard
    if "user_id" in session:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Call service layer to verify credentials
        result = login_user(email, password)

        if result["success"]:
            user = result["user"]

            # ── Start the Session ─────────────────
            # Store user ID in the session cookie
            # All protected routes check for this
            session["user_id"]   = user["id"]
            session["user_name"] = user["name"]

            flash(result["message"], "success")
            return redirect(url_for("dashboard.index"))

        else:
            flash(result["message"], "danger")
            return render_template("auth/login.html", email=email)

    # GET request → show empty login form
    return render_template("auth/login.html")


# ─────────────────────────────────────────────
#  LOGOUT
# ─────────────────────────────────────────────

@bp.route("/logout")
def logout():
    """
    Clears the session (logs the user out) and
    redirects to the login page.
    """
    user_name = session.get("user_name", "User")

    # Remove all session data
    session.clear()

    flash(f"You have been logged out, {user_name}. See you soon!", "info")
    return redirect(url_for("auth.login"))