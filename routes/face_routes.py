"""
routes/face_routes.py — Face Registration & Recognition Routes
--------------------------------------------------------------
Handles all URL routes related to face operations.
All routes here require the user to be logged in.

Routes:
    GET  /face/register        → Show face registration page
    POST /face/register        → Process face image + save encoding
    GET  /face/recognize       → Show recognition page
    POST /face/recognize       → Run recognition + show result
    POST /face/register/webcam → Handle base64 webcam capture for registration
    POST /face/recognize/webcam→ Handle base64 webcam capture for recognition
"""

from flask import (
    Blueprint, render_template, request,
    redirect, url_for, session, flash, jsonify
)

from services.auth_service   import login_required, get_current_user
from services.face_service   import (
    register_face_from_image,
    recognize_face_from_image,
    save_uploaded_image,
    save_base64_image,
    allowed_file
)
from repositories.face_repo         import has_face_registered
from repositories.recognition_repo  import save_recognition_log
from config import Config

# ─────────────────────────────────────────────
#  BLUEPRINT SETUP
# ─────────────────────────────────────────────
bp = Blueprint("face", __name__, url_prefix="/face")


# ─────────────────────────────────────────────
#  FACE REGISTRATION
# ─────────────────────────────────────────────

@bp.route("/register", methods=["GET", "POST"])
@login_required
def register_face():
    """
    GET  → Show the face registration form with webcam/upload options
    POST → Process uploaded image and save encoding to DB
    """
    user_id = session["user_id"]
    already_registered = has_face_registered(user_id)

    if request.method == "POST":
        # ── Handle File Upload ─────────────────
        if "face_image" in request.files:
            file = request.files["face_image"]

            if file.filename == "":
                flash("No file selected. Please choose an image.", "warning")
                return redirect(url_for("face.register_face"))

            if not allowed_file(file.filename):
                flash("Invalid file type. Please upload JPG or PNG.", "danger")
                return redirect(url_for("face.register_face"))

            # Save image to uploads folder
            saved_path, relative_path = save_uploaded_image(file)

            if not saved_path:
                flash("Failed to save the image. Please try again.", "danger")
                return redirect(url_for("face.register_face"))

            # Run face encoding and save to DB
            result = register_face_from_image(user_id, saved_path)

            if result["success"]:
                flash(result["message"], "success")
                return redirect(url_for("dashboard.index"))
            else:
                flash(result["message"], "danger")
                return redirect(url_for("face.register_face"))

    # GET request → render the registration page
    user = get_current_user()
    return render_template(
        "face/register_face.html",
        user=user,
        already_registered=already_registered
    )


@bp.route("/register/webcam", methods=["POST"])
@login_required
def register_face_webcam():
    """
    Handles webcam capture for face registration.
    Receives a base64 image from webcam.js via AJAX POST.
    Returns JSON response so JS can handle success/error.
    """
    user_id = session["user_id"]

    data       = request.get_json()
    image_data = data.get("image_data") if data else None

    if not image_data:
        return jsonify({"success": False, "message": "No image data received."}), 400

    # Save the base64 webcam image to uploads folder
    saved_path, relative_path = save_base64_image(
        image_data,
        folder=Config.UPLOAD_FOLDER
    )

    if not saved_path:
        return jsonify({"success": False, "message": "Failed to save webcam image."}), 500

    # Run face encoding and save to DB
    result = register_face_from_image(user_id, saved_path)

    return jsonify(result)


# ─────────────────────────────────────────────
#  FACE RECOGNITION
# ─────────────────────────────────────────────

@bp.route("/recognize", methods=["GET", "POST"])
@login_required
def recognize():
    """
    GET  → Show the face recognition page
    POST → Process uploaded image and run recognition
    """
    user_id = session["user_id"]

    if request.method == "POST":
        # ── Handle File Upload ─────────────────
        if "face_image" not in request.files:
            flash("No image file provided.", "warning")
            return redirect(url_for("face.recognize"))

        file = request.files["face_image"]

        if file.filename == "":
            flash("No file selected.", "warning")
            return redirect(url_for("face.recognize"))

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload JPG or PNG.", "danger")
            return redirect(url_for("face.recognize"))

        # Save the uploaded image
        saved_path, relative_path = save_uploaded_image(file)

        if not saved_path:
            flash("Failed to save the image.", "danger")
            return redirect(url_for("face.recognize"))

        # Run face recognition
        result = recognize_face_from_image(saved_path)

        # Log this recognition attempt to the database
        save_recognition_log(
            user_id         = user_id,
            matched_user_id = result.get("user_id"),
            distance        = result.get("distance"),
            confidence      = result.get("confidence"),
            status          = result.get("status", "error")
        )

        # Redirect to the result page
        return render_template(
            "face/recognize_success.html",
            result       = result,
            image_path   = relative_path,
            user         = get_current_user()
        )

    # GET → show recognition form
    user = get_current_user()
    return render_template("face/recognize.html", user=user)


@bp.route("/recognize/webcam", methods=["POST"])
@login_required
def recognize_webcam():
    """
    Handles webcam capture for face recognition.
    Receives base64 image from JS, runs recognition, returns JSON.
    """
    user_id = session["user_id"]

    data       = request.get_json()
    image_data = data.get("image_data") if data else None

    if not image_data:
        return jsonify({"success": False, "message": "No image data received."}), 400

    # Save base64 webcam image to uploads folder
    saved_path, relative_path = save_base64_image(
        image_data,
        folder=Config.UPLOAD_FOLDER
    )

    if not saved_path:
        return jsonify({"success": False, "message": "Failed to process image."}), 500

    # Run recognition
    result = recognize_face_from_image(saved_path)

    # Log the attempt
    save_recognition_log(
        user_id         = user_id,
        matched_user_id = result.get("user_id"),
        distance        = result.get("distance"),
        confidence      = result.get("confidence"),
        status          = result.get("status", "error")
    )

    result["image_path"] = relative_path
    return jsonify(result)