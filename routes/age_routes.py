"""
routes/age_routes.py - Age prediction routes
--------------------------------------------
Handles image upload and DeepFace age prediction.
"""

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for

from config import Config
from services.auth_service import get_current_user, login_required
from services.face_service import allowed_file, save_base64_image, save_uploaded_image
from services.age_service import predict_age_from_image


bp = Blueprint("age", __name__, url_prefix="/age")


@bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    user = get_current_user()
    result = None
    image_path = None

    if request.method == "POST":
        if "age_image" not in request.files:
            flash("No image file provided.", "warning")
            return redirect(url_for("age.predict"))

        file = request.files["age_image"]

        if file.filename == "":
            flash("No file selected. Please choose an image.", "warning")
            return redirect(url_for("age.predict"))

        if not allowed_file(file.filename):
            flash("Invalid file type. Please upload JPG or PNG.", "danger")
            return redirect(url_for("age.predict"))

        saved_path, image_path = save_uploaded_image(
            file,
            folder=Config.AGE_CAPTURE_FOLDER,
        )

        if not saved_path:
            flash("Failed to save the image. Please try again.", "danger")
            return redirect(url_for("age.predict"))

        result = predict_age_from_image(saved_path)

        if result["success"]:
            flash("Age prediction completed successfully.", "success")
        else:
            flash(result["message"], "danger")

    return render_template(
        "age/predict.html",
        user=user,
        result=result,
        image_path=image_path,
    )


@bp.route("/predict/webcam", methods=["POST"])
@login_required
def predict_webcam():
    data = request.get_json()
    image_data = data.get("image_data") if data else None

    if not image_data:
        return jsonify({"success": False, "message": "No image data received."}), 400

    saved_path, image_path = save_base64_image(
        image_data,
        folder=Config.AGE_CAPTURE_FOLDER,
    )

    if not saved_path:
        return jsonify({"success": False, "message": "Failed to save webcam image."}), 500

    result = predict_age_from_image(saved_path)
    result["image_path"] = image_path

    status_code = 200 if result.get("success") else 422
    return jsonify(result), status_code
