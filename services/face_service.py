"""
services/face_service.py — Face Registration & Recognition Logic
----------------------------------------------------------------
This replaces the old utils/faceutils.py.
Contains the actual AI logic for face encoding and matching.

Key difference from old code:
    OLD → encodings saved to encodings.pkl file
    NEW → encodings saved to database as BLOB via face_repo

Functions:
    register_face_from_image()  → Encode face + save to DB
    recognize_face_from_image() → Compare face against all DB encodings
    allowed_file()              → Validate uploaded file extension
    save_uploaded_image()       → Save image file to uploads folder
"""

import os
import uuid
import numpy as np
from werkzeug.utils import secure_filename

try:
    import face_recognition
except ImportError as e:
    face_recognition = None
    FACE_RECOGNITION_IMPORT_ERROR = e
else:
    FACE_RECOGNITION_IMPORT_ERROR = None

from config import Config
from repositories.face_repo import (
    save_face_encoding,
    get_all_encodings_with_ids,
    has_face_registered
)
from repositories.user_repo import get_user_by_id


# ─────────────────────────────────────────────
#  FILE HELPERS
# ─────────────────────────────────────────────

def allowed_file(filename):
    """
    Checks if the uploaded file has an allowed image extension.

    Args:
        filename (str): The original filename from the upload

    Returns:
        bool: True if extension is in ALLOWED_EXTENSIONS
    """
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def save_uploaded_image(file, folder=None):
    """
    Saves an uploaded image file to the uploads folder.
    Generates a unique filename to prevent overwrites.

    Args:
        file   (FileStorage): The uploaded file from Flask request
        folder (str):         Optional custom folder path
                              Defaults to Config.UPLOAD_FOLDER

    Returns:
        tuple:
            saved_path  (str): Full filesystem path to saved file
            relative_path (str): Path relative to static/ for DB storage
                                  e.g. "uploads/abc123.jpg"
        (None, None) on error
    """
    try:
        if folder is None:
            folder = Config.UPLOAD_FOLDER

        # Secure the original filename, then make it unique
        original_name = secure_filename(file.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{extension}"  # e.g. a1b2c3d4.jpg

        saved_path = os.path.join(folder, unique_name)
        file.save(saved_path)

        # Build a relative path for storing in DB
        # e.g. /static/uploads/a1b2c3d4.jpg → uploads/a1b2c3d4.jpg
        folder_name = os.path.basename(folder)
        relative_path = f"{folder_name}/{unique_name}"

        return saved_path, relative_path

    except Exception as e:
        print(f"[face_service] Error saving image: {e}")
        return None, None


def save_base64_image(base64_data, folder=None):
    """
    Saves a base64-encoded image (from webcam capture) to a folder.

    Args:
        base64_data (str): Base64 string, optionally with data URL prefix
                           e.g. "data:image/jpeg;base64,/9j/4AAQ..."
        folder      (str): Folder to save into (default: UPLOAD_FOLDER)

    Returns:
        tuple: (saved_path, relative_path) or (None, None) on error
    """
    import base64

    try:
        if folder is None:
            folder = Config.UPLOAD_FOLDER

        # Strip the data URL prefix if present
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        image_bytes = base64.b64decode(base64_data)
        unique_name = f"{uuid.uuid4().hex}.jpg"
        saved_path = os.path.join(folder, unique_name)

        with open(saved_path, "wb") as f:
            f.write(image_bytes)

        folder_name = os.path.basename(folder)
        relative_path = f"{folder_name}/{unique_name}"

        return saved_path, relative_path

    except Exception as e:
        print(f"[face_service] Error saving base64 image: {e}")
        return None, None


# ─────────────────────────────────────────────
#  FACE REGISTRATION
# ─────────────────────────────────────────────

def register_face_from_image(user_id, image_path):
    """
    Detects and encodes a face from an image, then saves it to the DB.

    Steps:
        1. Load image from disk
        2. Detect face locations in image
        3. Generate 128-number face encoding
        4. Save encoding to database as BLOB

    Args:
        user_id    (int): The user this face belongs to
        image_path (str): Full path to the saved image file

    Returns:
        dict:
            "success" (bool): Whether encoding was saved
            "message" (str):  Result description
    """
    if face_recognition is None:
        return {
            "success": False,
            "message": "Face recognition is not available. "
                       "Please install the face_recognition package."
        }

    try:
        # Load the image from disk into memory
        image = face_recognition.load_image_file(image_path)

        # Detect face locations in the image (returns list of tuples)
        face_locations = face_recognition.face_locations(image)

        # No face detected in the image
        if len(face_locations) == 0:
            return {
                "success": False,
                "message": "No face detected in the image. "
                           "Please try again with a clearer photo."
            }

        # More than one face in the image
        if len(face_locations) > 1:
            return {
                "success": False,
                "message": "Multiple faces detected. "
                           "Please upload a photo with only your face."
            }

        # Generate face encoding (128-dimensional numpy array)
        encodings = face_recognition.face_encodings(image, face_locations)
        face_encoding = encodings[0]

        # Save the encoding to the database
        saved = save_face_encoding(user_id, face_encoding)

        if saved:
            return {
                "success": True,
                "message": "Face registered successfully!"
            }
        else:
            return {
                "success": False,
                "message": "Face detected but could not be saved. Please try again."
            }

    except Exception as e:
        print(f"[face_service] Error during registration: {e}")
        return {
            "success": False,
            "message": f"An error occurred during face registration: {str(e)}"
        }


# ─────────────────────────────────────────────
#  FACE RECOGNITION
# ─────────────────────────────────────────────

def recognize_face_from_image(image_path):
    """
    Compares a face in the given image against ALL stored encodings.
    Uses Euclidean distance — lower distance = better match.

    Steps:
        1. Load and encode the face from image
        2. Load all stored encodings from DB
        3. Compare using Euclidean distance
        4. Return the best match under the threshold

    Args:
        image_path (str): Full path to the image for recognition

    Returns:
        dict:
            "status"     (str):   "matched" or "unknown"
            "user_id"    (int):   Matched user's ID (if matched)
            "user_name"  (str):   Matched user's name (if matched)
            "distance"   (float): Euclidean distance (lower = better)
            "confidence" (float): Confidence % (0–100)
            "message"    (str):   Result description
    """
    if face_recognition is None:
        return {
            "status": "error",
            "message": "Face recognition is not available. "
                       "Please install the face_recognition package.",
            "confidence": 0
        }

    try:
        # Step 1: Load and encode the uploaded image
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            return {
                "status": "error",
                "message": "No face detected in the uploaded image.",
                "confidence": 0
            }

        # Get encoding of the face to recognize
        unknown_encoding = face_recognition.face_encodings(image, face_locations)[0]

        # Step 2: Load all stored encodings from the database
        all_stored = get_all_encodings_with_ids()

        if not all_stored:
            return {
                "status": "error",
                "message": "No registered faces in the system yet.",
                "confidence": 0
            }

        # Step 3: Compare with Euclidean distance
        best_match_user_id = None
        best_distance = float("inf")  # Start with worst possible distance

        for entry in all_stored:
            stored_encoding = entry["encoding"]
            stored_user_id  = entry["user_id"]

            # Calculate Euclidean distance between two 128-d vectors
            distance = np.linalg.norm(unknown_encoding - stored_encoding)

            if distance < best_distance:
                best_distance = distance
                best_match_user_id = stored_user_id

        # Step 4: Check if best match is within the threshold
        threshold = Config.FACE_MATCH_THRESHOLD

        if best_distance <= threshold:
            # Calculate confidence: 0 distance = 100%, threshold distance = 0%
            confidence = max(0, (1 - best_distance / threshold)) * 100

            # Get the matched user's info
            matched_user = get_user_by_id(best_match_user_id)
            user_name = matched_user["name"] if matched_user else "Unknown"

            return {
                "status":     "matched",
                "user_id":    best_match_user_id,
                "user_name":  user_name,
                "distance":   round(best_distance, 4),
                "confidence": round(confidence, 2),
                "message":    f"Face matched: {user_name} ({confidence:.1f}% confidence)"
            }

        else:
            # Distance too high — no match found
            confidence = max(0, (1 - best_distance / threshold)) * 100
            return {
                "status":     "unknown",
                "user_id":    None,
                "user_name":  None,
                "distance":   round(best_distance, 4),
                "confidence": round(confidence, 2),
                "message":    "No matching face found in the system."
            }

    except Exception as e:
        print(f"[face_service] Error during recognition: {e}")
        return {
            "status":     "error",
            "message":    f"Recognition error: {str(e)}",
            "confidence": 0
        }
