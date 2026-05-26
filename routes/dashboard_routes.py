"""
routes/dashboard_routes.py — Dashboard & Analytics Routes
----------------------------------------------------------
Handles all URL routes for the dashboard and analytics pages.
All routes require login via @login_required decorator.

Routes:
    GET /dashboard/          → Main dashboard with stats summary
    GET /dashboard/analytics → Full analytics page with all charts
"""

from flask import (
    Blueprint, render_template, session
)

from services.auth_service       import login_required, get_current_user
from services.dashboard_service  import (
    get_dashboard_summary,
    get_analytics_data
)
from repositories.face_repo import has_face_registered

# ─────────────────────────────────────────────
#  BLUEPRINT SETUP
# ─────────────────────────────────────────────
bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")


# ─────────────────────────────────────────────
#  MAIN DASHBOARD
# ─────────────────────────────────────────────

@bp.route("/", methods=["GET"])
@login_required
def index():
    """
    Main dashboard page shown after login.

    Displays:
        - Welcome message with user's name
        - Stats cards (total scans, most common mood, etc.)
        - Face registration status with prompt if not done
        - Recent mood scan activity (last 5)
        - Recent recognition activity (last 5)
    """
    user_id = session["user_id"]
    user    = get_current_user()

    # Fetch all dashboard data from the service layer
    summary = get_dashboard_summary(user_id)

    return render_template(
        "dashboard/dashboard.html",
        user    = user,
        summary = summary
    )


# ─────────────────────────────────────────────
#  ANALYTICS PAGE
# ─────────────────────────────────────────────

@bp.route("/analytics", methods=["GET"])
@login_required
def analytics():
    """
    Full analytics page with detailed charts and history.

    Displays:
        - Emotion distribution pie chart
        - Mood confidence trend line chart
        - Recognition matched vs unknown doughnut chart
        - Complete mood log history table
    """
    user_id = session["user_id"]
    user    = get_current_user()

    # Get all chart-ready data from the service layer
    analytics = get_analytics_data(user_id)

    return render_template(
        "dashboard/analytics.html",
        user      = user,
        analytics = analytics
    )