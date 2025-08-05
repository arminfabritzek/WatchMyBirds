# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

def load_config():
    """
    Loads configuration from environment variables and returns a dictionary.
    """
    # Determine the image size based on the classifier model
    classifier_model = os.getenv("CLASSIFIER_MODEL", "efficientnet_b0")
    model_sizes = {
        "efficientnet_b0": 224,
        "efficientnet_b1": 240,
        "efficientnet_b2": 256,
        "efficientnet_b3": 288
    }
    classifier_image_size = model_sizes.get(classifier_model, 224)

    location_str = os.getenv("LOCATION_DATA", "52.516, 13.377")
    try:
        lat_str, lon_str = location_str.split(",")
        LOCATION_DATA = {"latitude": float(lat_str), "longitude": float(lon_str)}
    except Exception:
        # Fallback to defaults if parsing fails
        LOCATION_DATA = {"latitude": 52.516, "longitude": 13.377}

    config = {
        # General Settings
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "/output"),
        "VIDEO_SOURCE": os.getenv("VIDEO_SOURCE", "0"),

        # GPS Location
        "LOCATION_DATA": LOCATION_DATA,

        # Model and Detection Settings
        "DETECTOR_MODEL_CHOICE": os.getenv("DETECTOR_MODEL_CHOICE", "yolo"),
        "CONFIDENCE_THRESHOLD_DETECTION": float(os.getenv("CONFIDENCE_THRESHOLD_DETECTION", 0.55)),
        "SAVE_THRESHOLD": float(os.getenv("SAVE_THRESHOLD", 0.55)),
        "MAX_FPS_DETECTION": float(os.getenv("MAX_FPS_DETECTION", 0.5)),
        "MODEL_BASE_PATH": os.getenv("MODEL_BASE_PATH", "models"),

        # Model and Classifier Settings
        "CLASSIFIER_MODEL": classifier_model,
        "CLASSIFIER_IMAGE_SIZE": classifier_image_size,
        "CLASSIFIER_CONFIDENCE_THRESHOLD": float(os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", 0.55)),

        # Results Settings
        "FUSION_ALPHA": float(os.getenv("FUSION_ALPHA", 0.5)),

        # Streaming Settings
        "STREAM_FPS": float(os.getenv("STREAM_FPS", 1)),
        "STREAM_WIDTH_OUTPUT_RESIZE": int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 640)),

        # Day and Night Capture Settings
        "DAY_AND_NIGHT_CAPTURE": os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower() == "true",
        "DAY_AND_NIGHT_CAPTURE_LOCATION": os.getenv("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin"),

        # CPU and Resource Management
        "CPU_LIMIT": int(float(os.getenv("CPU_LIMIT", 1))),

        # Telegram Notification Settings
        "TELEGRAM_COOLDOWN": float(os.getenv("TELEGRAM_COOLDOWN", 5)),  # seconds between alerts

        # GA + Cookiebot
        "GA_MEASUREMENT_ID": os.getenv("GA_MEASUREMENT_ID", "G-REPLACE-ME-XXXXXX"),
        "COOKIEBOT_CBID": os.getenv("COOKIEBOT_CBID", None),

        "EDIT_PASSWORD": os.getenv("EDIT_PASSWORD", "SECRET_PASSWORD"),
    }
    return config


if __name__ == "__main__":
    # For testing purposes, print the configuration
    config = load_config()
    from pprint import pprint

    pprint(config)
