# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

def load_config():
    """
    Loads configuration from environment variables and returns a dictionary.
    """
    config = {
        # General Settings
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "/output"),
        "VIDEO_SOURCE": os.getenv("VIDEO_SOURCE", "0"),

        # Model and Detection Settings
        "DETECTOR_MODEL_CHOICE": os.getenv("DETECTOR_MODEL_CHOICE", "yolo"),  # Only "yolo" supported for now
        "DETECTOR_MODEL_PATH": os.getenv("DETECTOR_MODEL_PATH", "models/best.onnx"),
        "CONFIDENCE_THRESHOLD_DETECTION": float(os.getenv("CONFIDENCE_THRESHOLD_DETECTION", 0.8)),
        "SAVE_THRESHOLD": float(os.getenv("SAVE_THRESHOLD", 0.8)),
        "MAX_FPS_DETECTION": float(os.getenv("MAX_FPS_DETECTION", 1.0)),

        # Model and Classifier Settings
        "CLASSIFIER_MODEL_PATH": os.getenv("CLASSIFIER_MODEL_PATH", "models/classifier_best.onnx"),
        "CLASSIFIER_CLASSES_PATH": os.getenv("CLASSIFIER_CLASSES_PATH", "models/classifier_classes.txt"),
        "CLASSIFIER_CONFIDENCE_THRESHOLD": float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", 0.7)),

        # Streaming Settings
        "STREAM_FPS": float(os.getenv("STREAM_FPS", 1)),
        "STREAM_WIDTH_OUTPUT_RESIZE": int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 640)),

        # Day and Night Capture Settings
        "DAY_AND_NIGHT_CAPTURE": os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower() == "true",
        "DAY_AND_NIGHT_CAPTURE_LOCATION": os.getenv("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin"),

        # CPU and Resource Management
        "CPU_LIMIT": int(float(os.getenv("CPU_LIMIT", 2))),

        # Telegram Notification Settings
        "TELEGRAM_COOLDOWN": float(os.getenv("TELEGRAM_COOLDOWN", 5)),  # seconds between alerts
    }
    return config

if __name__ == "__main__":
    # For testing purposes, print the configuration
    config = load_config()
    from pprint import pprint
    pprint(config)