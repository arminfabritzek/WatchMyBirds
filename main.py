# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Web Interface
# main.py
# ------------------------------------------------------------------------------
import time
from config import get_config
config = get_config()
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
logger.debug(f"Configuration: {json.dumps(config, indent=2)}")

if _debug:
    send_telegram_message(text="🐦 Birdwatching has started in DEBUG mode!", photo_path="assets/debug.jpg")

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Start the Detection Manager
# -----------------------------
from detectors.detection_manager import DetectionManager
import threading

# Create a DetectionManager instance.
detection_manager = DetectionManager()

# Start detection asynchronously so the web UI can come up immediately.
threading.Thread(target=detection_manager.start, daemon=True).start()

# Register the cleanup function
import atexit
atexit.register(detection_manager.stop)

# -----------------------------
# Import and Run the Web Interface
# -----------------------------
from web.web_interface import create_web_interface

# Expose the Flask server as the WSGI app for Waitress.
app = create_web_interface(detection_manager)

if __name__ == '__main__':
    # Use Waitress instead of Werkzeug dev server.
    # Werkzeug delays accepting connections for ~8s after socket.bind(),
    # causing the UI to be unreachable immediately after startup.
    # Waitress responds instantly, providing consistent dev/prod behavior.
    from waitress import serve
    host = '0.0.0.0'
    port = 8050
    logger.info(f"Starting Waitress server on http://{host}:{port}")
    try:
        # threads=8 because /video_feed holds connections open indefinitely (streaming),
        # which can exhaust the default 4 threads and block normal page requests.
        serve(app, host=host, port=port, threads=8)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down detection manager...")
        detection_manager.stop()
