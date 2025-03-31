# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Dash Web Interface
# main.py
# ------------------------------------------------------------------------------
import time
from config import load_config
config = load_config()
from logging_config import get_logger
logger = get_logger(__name__)
import json
import os
from utils.telegram_notifier import send_telegram_message
from utils.cpu_limiter import restrict_to_cpus  # Import CPU limiter

# Apply CPU restriction before starting any threads for slow systems
restrict_to_cpus()

# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
# use the configuration values from the config dictionary.
_debug = config["DEBUG_MODE"]
output_dir = config["OUTPUT_DIR"]

logger.info(f"Debug mode is {'enabled' if _debug else 'disabled'}.")
logger.info(f"Configuration: {json.dumps(config, indent=2)}")

if _debug:
    send_telegram_message(text="🐦 Birdwatching has started in DEBUG mode!", photo_path="assets/the_birdwatcher_small.jpeg")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Start the Detection Manager
# -----------------------------
from detectors.detection_manager import DetectionManager

# Create a DetectionManager instance.
detection_manager = DetectionManager()

# Start the detection loop on a background thread.
detection_manager.start()

# Wait until video_capture is initialized
while detection_manager.video_capture is None:
    time.sleep(0.5)

# Register the cleanup function
import atexit
atexit.register(detection_manager.stop)

# -----------------------------
# Import and Run the Web Interface
# -----------------------------
from web.web_interface import create_web_interface

# Expose the Flask server as the WSGI app for Waitress.
interface = create_web_interface(detection_manager)
app = interface["server"]

if __name__ == '__main__':
    # Run the web interface
    try:
        interface["run"](debug=_debug, host='0.0.0.0', port=8050)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down detection manager...")
        detection_manager.stop()
