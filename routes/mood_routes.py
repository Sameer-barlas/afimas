"""
routes/mood_routes.py — Mood Scan Routes
-----------------------------------------
Handles all URL routes related to emotion/mood detection.
All routes require login via the @login_required decorator.

Routes:
    GET  /mood/scan          → Show the mood scan page (webcam/upload)
    POST /mood/scan/upload   → Process uploaded image for emotion
    POST /mood/scan/webcam   → Process base64 webcam image for emotion
    GET  /mood/history       → Show full mood scan history for user
"""

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session, flash, jsonify
)

from services.auth_service    import login_required, get_current_user
from services.emotion_service import (
    analyze_emotion_from_path,
    analyze_emotion_from_base64
)
from services.face_service    import save_uploaded_image, allowed_file
from repositories.mood_repo   import (
    save_mood_log,
    get_mood_logs_by_user
)
from config import Config

# ─────────────────────────────────────────────
#  BLUEPRINT SETUP
# ─────────────────────────────────────────────
bp = Blueprint("mood", __name__, url_prefix="/mood")


# ─────────────────────────────────────────────
#  MOOD SCAN PAGE
# ─────────────────────────────────────────────

@bp.route("/scan", methods=["GET"])
@login_required
def scan():
    """
    Renders the mood scan page.
    User can either upload an image or use the webcam.
    """
    user = get_current_user()
    return render_template("mood/mood.html", user=user)


# ─────────────────────────────────────────────
#  PROCESS UPLOADED IMAGE
# ─────────────────────────────────────────────

@bp.route("/scan/upload", methods=["POST"])
@login_required
def scan_upload():
    """
    Handles mood scan via uploaded image file.
    Saves image → runs emotion analysis → saves log → shows result.
    """
    user_id = session["user_id"]

    # Validate file presence
    if "mood_image" not in request.files:
        flash("No image file provided.", "warning")
        return redirect(url_for("mood.scan"))

    file = request.files["mood_image"]

    if file.filename == "":
        flash("No file selected. Please choose an image.", "warning")
        return redirect(url_for("mood.scan"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a JPG or PNG image.", "danger")
        return redirect(url_for("mood.scan"))

    # Save image to mood_captures folder
    saved_path, relative_path = save_uploaded_image(
        file,
        folder=Config.MOOD_CAPTURE_FOLDER
    )

    if not saved_path:
        flash("Failed to save the image. Please try again.", "danger")
        return redirect(url_for("mood.scan"))

    # Run emotion analysis on the saved image
    result = analyze_emotion_from_path(saved_path)

    if not result["success"]:
        flash(result["message"], "danger")
        return redirect(url_for("mood.scan"))

    # Save the mood scan result to the database
    save_mood_log(
        user_id          = user_id,
        dominant_emotion = result["dominant_emotion"],
        confidence       = result["confidence"],
        mood_category    = result["mood_category"],
        all_scores_dict  = result["all_scores"],
        image_path       = relative_path
    )

    flash("Mood scan completed successfully!", "success")

    # Render the mood page again with the result displayed
    user = get_current_user()
    return render_template(
        "mood/mood.html",
        user        = user,
        result      = result,
        image_path  = relative_path
    )


# ─────────────────────────────────────────────
#  PROCESS WEBCAM CAPTURE (AJAX)
# ─────────────────────────────────────────────

@bp.route("/scan/webcam", methods=["POST"])
@login_required
def scan_webcam():
    """
    Handles mood scan from webcam capture.
    Receives base64 image data via AJAX POST from webcam.js.
    Returns JSON so JavaScript can update the page dynamically.
    """
    user_id = session["user_id"]

    # Read JSON body sent by webcam.js
    data       = request.get_json()
    image_data = data.get("image_data") if data else None

    if not image_data:
        return jsonify({
            "success": False,
            "message": "No image data received from webcam."
        }), 400

    # Decode base64, save image, run analysis (all in one function)
    result, saved_path, relative_path = analyze_emotion_from_base64(
        image_data,
        save_folder=Config.MOOD_CAPTURE_FOLDER
    )

    if not result["success"]:
        return jsonify(result), 422

    # Save to database
    save_mood_log(
        user_id          = user_id,
        dominant_emotion = result["dominant_emotion"],
        confidence       = result["confidence"],
        mood_category    = result["mood_category"],
        all_scores_dict  = result["all_scores"],
        image_path       = relative_path
    )

    # Add image path to response for JS to display preview
    result["image_path"] = relative_path
    return jsonify(result), 200


# ─────────────────────────────────────────────
#  MOOD HISTORY PAGE
# ─────────────────────────────────────────────

@bp.route("/history", methods=["GET"])
@login_required
def history():
    """
    Shows the full mood scan history for the logged-in user.
    Sorted newest first.
    """
    user_id  = session["user_id"]
    user     = get_current_user()
    mood_logs = get_mood_logs_by_user(user_id)

    return render_template(
        "mood/moodhistory.html",
        user      = user,
        mood_logs = mood_logs
    )
