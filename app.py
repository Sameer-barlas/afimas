"""
app.py — Main Flask Application Entry Point for AFIMAS
-------------------------------------------------------
This file:
    1. Creates and configures the Flask app
    2. Registers all Blueprints (routes)
    3. Sets up the root redirect
    4. Initializes the database on first run
    5. Creates required upload folders

How Blueprints connect:
    auth_routes      → /auth/signup, /auth/login, /auth/logout
    face_routes      → /face/register, /face/recognize
    mood_routes      → /mood/scan, /mood/history
    dashboard_routes → /dashboard/, /dashboard/analytics

Run the app:
    python app.py
"""

from flask import Flask, redirect, render_template, session, url_for

from config import Config, create_upload_folders
from database import initialize_database

# ── Import all route Blueprints ───────────────
from routes.auth_routes      import bp as auth_bp
from routes.face_routes      import bp as face_bp
from routes.mood_routes      import bp as mood_bp
from routes.age_routes       import bp as age_bp
from routes.dashboard_routes import bp as dashboard_bp


# ─────────────────────────────────────────────
#  APP FACTORY FUNCTION
#  Using a factory function is a clean pattern —
#  makes testing and future scaling easier.
# ─────────────────────────────────────────────

def create_app():
    """
    Creates and fully configures the Flask application.

    Returns:
        Flask: The configured app instance
    """
    app = Flask(__name__)

    # ── Load Configuration ────────────────────
    # Apply all settings from config.py
    app.secret_key            = Config.SECRET_KEY
    app.config["SESSION_PERMANENT"] = Config.SESSION_PERMANENT
    app.config["UPLOAD_FOLDER"]     = Config.UPLOAD_FOLDER
    app.config["DEBUG"]             = Config.DEBUG

    # ── Register Blueprints ───────────────────
    # Each blueprint handles its own group of routes
    app.register_blueprint(auth_bp)       # /auth/...
    app.register_blueprint(face_bp)       # /face/...
    app.register_blueprint(mood_bp)       # /mood/...
    app.register_blueprint(age_bp)        # /age/...
    app.register_blueprint(dashboard_bp)  # /dashboard/...

    # ── Root Route ────────────────────────────
    # Guests see the public homepage; signed-in users continue to the app.
    @app.route("/")
    def index():
        if session.get("user_id"):
            return redirect(url_for("dashboard.index"))
        return render_template("home.html", public_page=True)

    @app.route("/home")
    def home():
        if session.get("user_id"):
            return redirect(url_for("dashboard.index"))
        return render_template("home.html", public_page=True)

    # Convenience alias for users who type /login directly.
    @app.route("/login")
    def login_shortcut():
        return redirect(url_for("auth.login"))

    # Convenience alias for users who type /signup directly.
    @app.route("/signup")
    def signup_shortcut():
        return redirect(url_for("auth.signup"))

    # Convenience alias for users who type /logout directly.
    @app.route("/logout")
    def logout_shortcut():
        return redirect(url_for("auth.logout"))

    @app.route("/dashboard")
    def dashboard_shortcut():
        return redirect(url_for("dashboard.index"))

    # ── 404 Error Handler ─────────────────────
    @app.errorhandler(404)
    def page_not_found(e):
        return redirect(url_for("index"))

    return app


# ─────────────────────────────────────────────
#  STARTUP TASKS
# ─────────────────────────────────────────────

def initialize_app():
    """
    Runs once when the app starts:
        1. Creates the database and all tables (safe if already exists)
        2. Creates upload/mood_captures folders (safe if already exists)
    """
    print("─" * 45)
    print("  AFIMAS — AI Face & Mood Analysis System")
    print("─" * 45)

    # Initialize database tables
    initialize_database()

    # Create upload folders if missing
    create_upload_folders()
    print("✅ Upload folders ready.")
    print("─" * 45)


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    initialize_app()              # Setup DB + folders
    app = create_app()            # Build the Flask app
    app.run(debug=Config.DEBUG)   # Start the dev server
