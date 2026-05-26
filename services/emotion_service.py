"""
services/emotion_service.py — Mood & Emotion Detection Logic
-------------------------------------------------------------
This replaces the old utils/emotionutils.py.
Uses the FER (Facial Emotion Recognition) library to detect
emotions from face images captured via webcam or upload.

Detects 7 emotions:
    happy, sad, angry, neutral, fear, surprise, disgust

Functions:
    analyze_emotion_from_path()  → Run FER on a saved image file
    analyze_emotion_from_base64()→ Decode base64 image then run FER
    get_mood_category()          → Map emotion → Positive/Negative/Neutral
    build_mood_result()          → Format full result dict for saving/display
"""

import numpy as np
import base64
import os
import uuid

try:
    import cv2
except ImportError as e:
    cv2 = None
    CV2_IMPORT_ERROR = e
else:
    CV2_IMPORT_ERROR = None

try:
    from fer.fer import FER
except ImportError:
    try:
        from fer import FER
    except ImportError as e:
        FER = None
        FER_IMPORT_ERROR = e
    else:
        FER_IMPORT_ERROR = None
else:
    FER_IMPORT_ERROR = None
from config import Config

# ─────────────────────────────────────────────
#  INITIALIZE FER MODEL (once at import time)
#  mtcnn=True uses deep learning face detector
#  mtcnn=False uses faster OpenCV Haar cascade
# ─────────────────────────────────────────────
try:
    detector = FER(mtcnn=False)  # Use False for faster, lighter detection
    print("✅ FER emotion detector loaded.")
except Exception as e:
    detector = None
    print(f"❌ FER failed to load: {e}")


# ─────────────────────────────────────────────
#  MOOD CATEGORY MAPPING
# ─────────────────────────────────────────────

# Maps each emotion label to a broader mood category
MOOD_CATEGORY_MAP = {
    "happy":    "Positive",
    "surprise": "Positive",
    "neutral":  "Neutral",
    "sad":      "Negative",
    "angry":    "Negative",
    "fear":     "Negative",
    "disgust":  "Negative"
}

# Emoji for display in templates
EMOTION_EMOJI_MAP = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "neutral":  "😐",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢"
}


def get_mood_category(emotion):
    """
    Returns the mood category for a given emotion label.

    Args:
        emotion (str): One of the 7 FER emotion labels

    Returns:
        str: "Positive", "Negative", or "Neutral"
    """
    return MOOD_CATEGORY_MAP.get(emotion.lower(), "Neutral")


def get_emotion_emoji(emotion):
    """
    Returns an emoji for a given emotion label.

    Args:
        emotion (str): Emotion label

    Returns:
        str: Corresponding emoji character
    """
    return EMOTION_EMOJI_MAP.get(emotion.lower(), "😐")


# ─────────────────────────────────────────────
#  CORE ANALYSIS FUNCTION
# ─────────────────────────────────────────────

# Guidance text shown after mood detection.
MOOD_GUIDANCE_MAP = {
    "happy": {
        "headline": "You look upbeat and energized.",
        "summary": "This is a good moment for creative work, collaboration, and tasks that need confidence.",
        "suggestions": [
            "Use this energy for a meaningful task you have been postponing.",
            "Schedule communication-heavy work such as calls, presentations, or brainstorming.",
            "Capture what is working in your routine today so you can repeat it later.",
            "Do a short walk or workout to keep the positive momentum steady."
        ]
    },
    "surprise": {
        "headline": "You look alert and mentally activated.",
        "summary": "Your attention may be high right now, so channel it into quick decisions and discovery.",
        "suggestions": [
            "Write down the trigger or new information that shifted your mood.",
            "Handle short research, planning, or idea-generation tasks while your curiosity is active.",
            "Avoid rushed commitments; give important decisions a second look.",
            "Take two calm breaths before moving into the next high-focus task."
        ]
    },
    "neutral": {
        "headline": "You look steady and composed.",
        "summary": "This is a useful state for focused, practical, and detail-oriented work.",
        "suggestions": [
            "Pick one priority task and work on it for 25 minutes without switching context.",
            "Use this balanced state for studying, organizing files, or reviewing analytics.",
            "Add a small break after the next work block to keep your energy from dipping.",
            "If you feel flat, use music, light movement, or fresh air to lift activation gently."
        ]
    },
    "sad": {
        "headline": "You look low or emotionally tired.",
        "summary": "A softer routine may help. Choose manageable tasks and avoid overloading yourself.",
        "suggestions": [
            "Start with one small task that can be finished in under ten minutes.",
            "Take a short walk, drink water, or step into brighter light before heavy work.",
            "Delay high-pressure decisions if possible and focus on simple maintenance tasks.",
            "Message someone supportive or write down what is weighing on you."
        ]
    },
    "angry": {
        "headline": "You look tense or frustrated.",
        "summary": "Your system may need a reset before difficult communication or precision work.",
        "suggestions": [
            "Pause for a few minutes before replying to messages or making decisions.",
            "Use physical reset actions: walk, stretch, or controlled breathing.",
            "Choose structured tasks like cleanup, sorting, or routine admin until intensity drops.",
            "Write the issue privately first, then convert it into one calm next action."
        ]
    },
    "fear": {
        "headline": "You look anxious or uncertain.",
        "summary": "Grounding and smaller next steps can make the situation feel more manageable.",
        "suggestions": [
            "Name the concern clearly and separate facts from assumptions.",
            "Break the next task into the smallest possible first step.",
            "Use a calm environment and avoid multitasking for the next work block.",
            "Try a simple breathing pattern: inhale four counts, exhale six counts, repeat five times."
        ]
    },
    "disgust": {
        "headline": "You look uncomfortable or mentally resistant.",
        "summary": "This can happen when a task feels unpleasant, messy, or misaligned.",
        "suggestions": [
            "Identify what exactly feels unpleasant: the task, environment, or expectation.",
            "Do a quick environment reset before returning to work.",
            "Choose cleanup, filtering, or review tasks where this critical mood can be useful.",
            "Avoid forcing creative work immediately; start with a practical action."
        ]
    }
}


def get_mood_guidance(emotion, mood_category):
    guidance = MOOD_GUIDANCE_MAP.get(
        emotion.lower(),
        {
            "headline": "Your mood was detected.",
            "summary": "Use this result as a quick check-in before choosing your next task.",
            "suggestions": [
                "Pause briefly and notice your current energy level.",
                "Choose a task that matches your focus and emotional capacity.",
                "Take a small reset break if the result feels intense."
            ]
        }
    )

    return {
        "category": mood_category,
        "headline": guidance["headline"],
        "summary": guidance["summary"],
        "suggestions": guidance["suggestions"]
    }


def _run_fer_on_image(image_bgr):
    """
    Internal helper — runs FER model on a BGR numpy image array.
    (OpenCV loads images in BGR format by default)

    Args:
        image_bgr (numpy.ndarray): BGR image array from cv2.imread()

    Returns:
        dict: Full mood result dictionary (see build_mood_result)
              or error dict if detection fails
    """
    if detector is None:
        return {
            "success": False,
            "message": "Emotion detector is not available."
        }

    if image_bgr is None:
        return {
            "success": False,
            "message": "Could not read the image file."
        }

    # Run FER detection — returns list of (face_box, emotion_scores)
    results = detector.detect_emotions(image_bgr)

    # No face detected in the image
    if not results:
        return {
            "success": False,
            "message": "No face detected for emotion analysis. "
                       "Please ensure your face is clearly visible."
        }

    # Take the first detected face's emotions
    emotions_dict = results[0]["emotions"]
    # emotions_dict looks like:
    # {"angry": 0.02, "disgust": 0.0, "fear": 0.05,
    #  "happy": 0.87, "sad": 0.03, "surprise": 0.01, "neutral": 0.02}

    # Find the dominant emotion (highest score)
    dominant_emotion = max(emotions_dict, key=emotions_dict.get)
    confidence       = emotions_dict[dominant_emotion]
    mood_category    = get_mood_category(dominant_emotion)
    emoji            = get_emotion_emoji(dominant_emotion)
    guidance         = get_mood_guidance(dominant_emotion, mood_category)

    return {
        "success":          True,
        "dominant_emotion": dominant_emotion,
        "confidence":       round(confidence, 4),
        "mood_category":    mood_category,
        "emoji":            emoji,
        "all_scores":       emotions_dict,  # Full dict for charts
        "guidance":         guidance,
        "message":          f"Emotion detected: {dominant_emotion.capitalize()} {emoji}"
    }


# ─────────────────────────────────────────────
#  PUBLIC FUNCTIONS
# ─────────────────────────────────────────────

def analyze_emotion_from_path(image_path):
    """
    Analyzes emotion from an image file saved on disk.
    Used when the user uploads an image file directly.

    Args:
        image_path (str): Full path to the image file

    Returns:
        dict: Emotion result with success, dominant_emotion,
              confidence, mood_category, all_scores, emoji, message
    """
    if cv2 is None:
        return {
            "success": False,
            "message": "OpenCV is not available. Please install opencv-python."
        }

    # Load image using OpenCV (returns BGR numpy array)
    image_bgr = cv2.imread(image_path)
    return _run_fer_on_image(image_bgr)


def analyze_emotion_from_base64(base64_data, save_folder=None):
    """
    Decodes a base64 image string (from webcam JS capture),
    saves it to disk, then runs emotion analysis.

    Args:
        base64_data (str): Base64 image string (with or without data URL prefix)
        save_folder (str): Where to save the captured image
                           Defaults to Config.MOOD_CAPTURE_FOLDER

    Returns:
        tuple:
            result      (dict): Emotion analysis result dictionary
            saved_path  (str):  Full path where image was saved (or None)
            relative_path(str): Relative path for DB storage (or None)
    """
    if cv2 is None:
        return {
            "success": False,
            "message": "OpenCV is not available. Please install opencv-python."
        }, None, None

    if save_folder is None:
        save_folder = Config.MOOD_CAPTURE_FOLDER

    saved_path    = None
    relative_path = None

    try:
        # Strip data URL prefix if present
        # e.g. "data:image/jpeg;base64,/9j/..." → "/9j/..."
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]

        # Decode base64 → bytes → numpy array → BGR image
        image_bytes   = base64.b64decode(base64_data)
        nparr         = np.frombuffer(image_bytes, np.uint8)
        image_bgr     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save the captured image to disk
        os.makedirs(save_folder, exist_ok=True)
        unique_name   = f"{uuid.uuid4().hex}.jpg"
        saved_path    = os.path.join(save_folder, unique_name)

        cv2.imwrite(saved_path, image_bgr)

        folder_name   = os.path.basename(save_folder)
        relative_path = f"{folder_name}/{unique_name}"

        # Run emotion analysis on the decoded image
        result = _run_fer_on_image(image_bgr)
        return result, saved_path, relative_path

    except Exception as e:
        print(f"[emotion_service] Error analyzing base64 image: {e}")
        return {
            "success": False,
            "message": f"Failed to process webcam image: {str(e)}"
        }, None, None
