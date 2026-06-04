"""
services/mood_transform_service.py - Local mood transform demo
--------------------------------------------------------------
This is the lightweight OpenCV/landmark-style demo version used inside AFIMAS.
The stronger image generation model is being checked separately on Kaggle, so
this file intentionally avoids local heavy image-generation models.
"""

import base64
import os
import time
import uuid

import numpy as np

try:
    import cv2
except ImportError as e:
    cv2 = None
    CV2_IMPORT_ERROR = e
else:
    CV2_IMPORT_ERROR = None

try:
    import face_recognition
except Exception as e:
    face_recognition = None
    FACE_RECOGNITION_IMPORT_ERROR = e
else:
    FACE_RECOGNITION_IMPORT_ERROR = None

from config import Config


TARGET_MOODS = {"happy", "sad", "angry"}
TRANSFORM_STRENGTH = 0.85


def get_target_moods():
    return [
        {
            "key": "happy",
            "label": "Happy",
            "icon": "bi-emoji-laughing",
            "detail": "OpenCV demo: lifts mouth corners and warms the face.",
        },
        {
            "key": "sad",
            "label": "Sad",
            "icon": "bi-emoji-frown",
            "detail": "OpenCV demo: lowers mouth corners and cools the face.",
        },
        {
            "key": "angry",
            "label": "Angry",
            "icon": "bi-emoji-angry",
            "detail": "OpenCV demo: pushes brow area and increases contrast.",
        },
    ]


def transform_mood_from_path(image_path, target_mood, output_folder=None):
    if output_folder is None:
        output_folder = Config.MOOD_TRANSFORM_FOLDER

    timestamp = _make_timestamp()
    target_mood = (target_mood or "").strip().lower()
    print(f"[mood_transform_service] selected mood: {target_mood}")
    print(f"[mood_transform_service] original image path: {image_path}")

    validation_error = _validate_runtime(target_mood)
    if validation_error:
        return validation_error

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return _error_result("Could not read the uploaded image.", target_mood, timestamp)

    result = _transform_image(image_bgr, target_mood, timestamp)
    if not result["success"]:
        return result

    transformed_path, transformed_relative_path = _save_transformed_image(
        result["image_bgr"],
        target_mood,
        output_folder,
        timestamp,
    )
    if not transformed_path:
        return _error_result("Expression was generated but could not be saved.", target_mood, timestamp)

    print(f"[mood_transform_service] transformed image path: {transformed_path}")
    return {
        "success": True,
        "target_mood": target_mood,
        "selected_mood": target_mood,
        "transformed_image_path": transformed_relative_path,
        "transformed_image": transformed_relative_path,
        "timestamp": timestamp,
        "landmarks_detected": result.get("landmarks_detected", False),
        "transformation_applied": True,
        "message": f"Demo facial mood transformed to {target_mood}.",
    }


def transform_mood_from_base64(base64_data, target_mood, output_folder=None):
    if output_folder is None:
        output_folder = Config.MOOD_TRANSFORM_FOLDER

    timestamp = _make_timestamp()
    target_mood = (target_mood or "").strip().lower()
    print(f"[mood_transform_service] selected mood: {target_mood}")

    validation_error = _validate_runtime(target_mood)
    if validation_error:
        return validation_error, None, None

    image_bgr = _decode_base64_image(base64_data)
    if image_bgr is None:
        return _error_result("Could not read the captured webcam image.", target_mood, timestamp), None, None

    os.makedirs(output_folder, exist_ok=True)
    original_name = f"{timestamp}_{uuid.uuid4().hex[:8]}_original.jpg"
    original_path = os.path.join(output_folder, original_name)
    if not cv2.imwrite(original_path, image_bgr):
        return _error_result("Could not save the captured webcam image.", target_mood, timestamp), None, None

    original_relative_path = _relative_static_path(output_folder, original_name)
    print(f"[mood_transform_service] original image path: {original_path}")

    result = _transform_image(image_bgr, target_mood, timestamp)
    if not result["success"]:
        return result, original_path, original_relative_path

    transformed_path, transformed_relative_path = _save_transformed_image(
        result["image_bgr"],
        target_mood,
        output_folder,
        timestamp,
    )
    if not transformed_path:
        return _error_result("Expression was generated but could not be saved.", target_mood, timestamp), original_path, original_relative_path

    result.pop("image_bgr", None)
    result.update({
        "success": True,
        "target_mood": target_mood,
        "selected_mood": target_mood,
        "original_image_path": original_relative_path,
        "original_image": original_relative_path,
        "transformed_image_path": transformed_relative_path,
        "transformed_image": transformed_relative_path,
        "timestamp": timestamp,
        "message": f"Demo facial mood transformed to {target_mood}.",
    })
    print(f"[mood_transform_service] transformed image path: {transformed_path}")
    return result, original_path, original_relative_path


def _transform_image(image_bgr, target_mood, timestamp):
    face_data = _detect_face_data(image_bgr)
    if face_data is None:
        print("[mood_transform_service] landmarks detected: False")
        styled = _apply_mood_filter(image_bgr, target_mood, None)
        return {
            "success": True,
            "target_mood": target_mood,
            "selected_mood": target_mood,
            "image_bgr": styled,
            "timestamp": timestamp,
            "landmarks_detected": False,
            "transformation_applied": True,
        }

    print("[mood_transform_service] landmarks detected: True")
    print(f"[mood_transform_service] bounding box used: {face_data['bbox']}")
    warped = _apply_local_expression_warp(image_bgr, face_data, target_mood)
    styled = _apply_mood_filter(warped, target_mood, face_data)

    return {
        "success": True,
        "target_mood": target_mood,
        "selected_mood": target_mood,
        "image_bgr": styled,
        "timestamp": timestamp,
        "landmarks_detected": True,
        "transformation_applied": True,
    }


def _detect_face_data(image_bgr):
    if face_recognition is not None:
        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb)
            locations = face_recognition.face_locations(rgb, model="hog")
            if landmarks_list and locations:
                top, right, bottom, left = locations[0]
                return {
                    "bbox": [left, top, right, bottom],
                    "landmarks": landmarks_list[0],
                    "source": "face_recognition",
                }
        except Exception as e:
            print(f"[mood_transform_service] face_recognition failed: {e}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return {
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "landmarks": None,
        "source": "opencv_fallback",
    }


def _apply_local_expression_warp(image_bgr, face_data, target_mood):
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = face_data["bbox"]
    face_w = max(1, x2 - x1)
    face_h = max(1, y2 - y1)
    landmarks = face_data.get("landmarks") or {}
    controls = []

    def point_from_landmark(group, index, fallback):
        points = landmarks.get(group) or []
        if len(points) > index:
            return np.float32(points[index])
        return np.float32(fallback)

    left_mouth = point_from_landmark("top_lip", 0, (x1 + face_w * 0.33, y1 + face_h * 0.70))
    right_mouth = point_from_landmark("top_lip", 6, (x1 + face_w * 0.67, y1 + face_h * 0.70))
    left_brow = point_from_landmark("left_eyebrow", 3, (x1 + face_w * 0.42, y1 + face_h * 0.32))
    right_brow = point_from_landmark("right_eyebrow", 1, (x1 + face_w * 0.58, y1 + face_h * 0.32))

    def add(point, dx, dy, radius):
        controls.append({
            "point": np.float32(point),
            "dx": float(dx * TRANSFORM_STRENGTH),
            "dy": float(dy * TRANSFORM_STRENGTH),
            "radius": float(radius),
        })

    mouth_radius = max(24, face_w * 0.16)
    brow_radius = max(20, face_w * 0.12)

    if target_mood == "happy":
        add(left_mouth, -0.035 * face_w, -0.105 * face_h, mouth_radius)
        add(right_mouth, 0.035 * face_w, -0.105 * face_h, mouth_radius)
    elif target_mood == "sad":
        add(left_mouth, 0.020 * face_w, 0.115 * face_h, mouth_radius)
        add(right_mouth, -0.020 * face_w, 0.115 * face_h, mouth_radius)
        add(left_brow, 0.020 * face_w, -0.060 * face_h, brow_radius)
        add(right_brow, -0.020 * face_w, -0.060 * face_h, brow_radius)
    elif target_mood == "angry":
        add(left_brow, 0.050 * face_w, 0.090 * face_h, brow_radius)
        add(right_brow, -0.050 * face_w, 0.090 * face_h, brow_radius)
        add(left_mouth, 0.018 * face_w, 0.030 * face_h, mouth_radius)
        add(right_mouth, -0.018 * face_w, 0.030 * face_h, mouth_radius)

    if not controls:
        return image_bgr.copy()

    pad_x = int(face_w * 0.28)
    pad_y = int(face_h * 0.28)
    roi_x1 = max(0, x1 - pad_x)
    roi_y1 = max(0, y1 - pad_y)
    roi_x2 = min(width, x2 + pad_x)
    roi_y2 = min(height, y2 + pad_y)
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    grid_y, grid_x = np.mgrid[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
    shift_x = np.zeros((roi_h, roi_w), dtype=np.float32)
    shift_y = np.zeros((roi_h, roi_w), dtype=np.float32)
    mask = np.zeros((roi_h, roi_w), dtype=np.float32)

    for control in controls:
        point = control["point"]
        radius = control["radius"]
        dist2 = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
        weight = np.exp(-dist2 / (2 * radius * radius)).astype(np.float32)
        shift_x += control["dx"] * weight
        shift_y += control["dy"] * weight
        mask = np.maximum(mask, weight)

    shift_x = np.clip(shift_x, -face_w * 0.12, face_w * 0.12)
    shift_y = np.clip(shift_y, -face_h * 0.14, face_h * 0.14)

    warped_roi = cv2.remap(
        image_bgr,
        grid_x - shift_x,
        grid_y - shift_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = cv2.GaussianBlur(np.clip(mask * 1.35, 0, 1), (0, 0), sigmaX=max(4, face_w * 0.025))
    mask = np.clip(mask, 0, 0.88)[:, :, None]

    result = image_bgr.copy().astype(np.float32)
    original_roi = result[roi_y1:roi_y2, roi_x1:roi_x2]
    result[roi_y1:roi_y2, roi_x1:roi_x2] = original_roi * (1 - mask) + warped_roi.astype(np.float32) * mask
    print(f"[mood_transform_service] local demo controls moved: {len(controls)}")
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_mood_filter(image_bgr, target_mood, face_data):
    mask = _build_face_mask(image_bgr.shape[:2], face_data)

    if target_mood == "happy":
        adjusted = cv2.convertScaleAbs(image_bgr, alpha=1.08, beta=8)
        warm = np.full_like(adjusted, (0, 12, 26))
        adjusted = cv2.addWeighted(adjusted, 0.96, warm, 0.04, 0)
    elif target_mood == "sad":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.88, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.88, 0, 255)
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        cool = np.full_like(adjusted, (18, 6, 0))
        adjusted = cv2.addWeighted(adjusted, 0.96, cool, 0.04, 0)
    else:
        adjusted = cv2.convertScaleAbs(image_bgr, alpha=1.22, beta=-22)
        blur = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=1.0)
        adjusted = cv2.addWeighted(adjusted, 1.32, blur, -0.32, 0)

    mask = np.clip(mask * 0.55, 0, 1)[:, :, None]
    blended = image_bgr.astype(np.float32) * (1 - mask) + adjusted.astype(np.float32) * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_face_mask(size, face_data):
    height, width = size
    mask = np.zeros((height, width), dtype=np.float32)

    if face_data is None:
        mask[:, :] = 1.0
        return mask

    x1, y1, x2, y2 = face_data["bbox"]
    face_w = x2 - x1
    face_h = y2 - y1
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2 + face_h * 0.06))
    axes = (max(20, int(face_w * 0.58)), max(28, int(face_h * 0.66)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8, face_w * 0.04))


def _validate_runtime(target_mood):
    if target_mood not in TARGET_MOODS:
        return _error_result("Choose a valid target mood: happy, sad, or angry.", target_mood)

    if cv2 is None:
        return _error_result(f"OpenCV is not available. Install opencv-python. Details: {CV2_IMPORT_ERROR}", target_mood)

    return None


def _decode_base64_image(base64_data):
    if not base64_data:
        return None

    try:
        if "," in base64_data:
            base64_data = base64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[mood_transform_service] Failed to decode webcam image: {e}")
        return None


def _save_transformed_image(image_bgr, target_mood, output_folder, timestamp):
    try:
        os.makedirs(output_folder, exist_ok=True)
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}_{target_mood}.jpg"
        output_path = os.path.join(output_folder, filename)
        saved = cv2.imwrite(output_path, image_bgr)
        if not saved:
            return None, None
        return output_path, _relative_static_path(output_folder, filename)
    except Exception as e:
        print(f"[mood_transform_service] Error saving transformed image: {e}")
        return None, None


def _relative_static_path(folder, filename):
    return f"{os.path.basename(folder)}/{filename}"


def _make_timestamp():
    return str(int(time.time() * 1000))


def _error_result(message, target_mood="", timestamp=None):
    return {
        "success": False,
        "target_mood": target_mood,
        "selected_mood": target_mood,
        "timestamp": timestamp or _make_timestamp(),
        "message": message,
        "landmarks_detected": False,
        "transformation_applied": False,
    }
