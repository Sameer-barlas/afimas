"""
services/age_service.py - Age prediction logic
----------------------------------------------
Uses DeepFace to estimate age from a face image and maps the
predicted number into a simple category for the UI.
"""


def get_age_category(age):
    """
    Maps a predicted age into the requested display categories.
    """
    if age <= 12:
        return "Child"
    if age <= 24:
        return "Young"
    return "Adult"


def get_category_detail(category):
    details = {
        "Child": "The detected face appears to be in a child age range.",
        "Young": "The detected face appears to be in a young age range.",
        "Adult": "The detected face appears to be in an adult age range.",
    }
    return details.get(category, "Age category was estimated from the detected face.")


def predict_age_from_image(image_path):
    """
    Runs DeepFace age analysis on a saved image path.

    Returns:
        dict with success, age, category, and message.
    """
    try:
        from deepface import DeepFace
    except Exception as e:
        message = str(e)
        if "tf-keras" in message or "tensorflow" in message.lower():
            return {
                "success": False,
                "message": "DeepFace needs the tf-keras package with your TensorFlow version. "
                           "Install it with: pip install tf-keras",
            }
        return {
            "success": False,
            "message": f"DeepFace is not available: {message}",
        }

    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["age"],
            enforce_detection=True,
            detector_backend="opencv",
        )

        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}

        raw_age = analysis.get("age")
        if raw_age is None:
            return {
                "success": False,
                "message": "DeepFace could not estimate age from this image.",
            }

        raw_age = int(round(float(raw_age)))
        age = max(1, raw_age - 2)
        category = get_age_category(age)

        return {
            "success": True,
            "age": age,
            "raw_age": raw_age,
            "category": category,
            "category_detail": get_category_detail(category),
            "message": f"Predicted age: {age} ({category})",
        }

    except ValueError:
        return {
            "success": False,
            "message": "No clear face detected. Please upload a brighter front-facing image.",
        }
    except Exception as e:
        print(f"[age_service] Error during age prediction: {e}")
        return {
            "success": False,
            "message": f"Age prediction failed: {str(e)}",
        }
