# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Streaming
# ------------------------------------------------------------------------------

# main.py
import random
from flask import Flask, Response
from dotenv import load_dotenv
load_dotenv()
import json
import os
import time
import csv
import cv2
import threading
import pytz
from datetime import datetime, timedelta
import numpy as np
import logging
from astral.geocoder import database, lookup
from astral.sun import sun
from camera.detector import Detector
from utils.telegram_notifier import send_telegram_message
from utils.cpu_limiter import restrict_to_cpus  # Import CPU limiter

# Apply CPU restriction before starting any threads for slow systems
restrict_to_cpus()

# Read the debug flag from the environment variable (default: False)
_debug = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if _debug else logging.INFO,  # Set level based on _debug
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

logging.getLogger().setLevel(logging.DEBUG if _debug else logging.INFO)

# Log the current debug mode
logger.info(f"Debug mode is {'enabled' if _debug else 'disabled'}.")

print(f"DEBUG_MODE environment variable: {os.getenv('DEBUG_MODE')}")
print(f"Debug mode in code: {_debug}")

if _debug:
    send_telegram_message(text="🐦 Birdwatching has started in DEBUG mode!", photo_path="assets/No_Signal.jpeg")

# Global lock for thread-safe frame access
frame_lock = threading.Lock()
latest_frame = None  # Global variable to store the latest frame for streaming

# Global Detector instance
detector_instance = None
detector_lock = threading.Lock()  # Lock to protect access to detector_instance

# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
# Load variables from environment variables or use default values
model_choice = os.getenv("MODEL_CHOICE", "pytorch_ssd")
class_filter = json.loads(os.getenv("CLASS_FILTER", '["bird"]'))  # Load JSON for array
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
save_threshold = float(os.getenv("SAVE_THRESHOLD", 0.5))
save_interval = int(os.getenv("SAVE_INTERVAL", 0))  # Seconds between saving # 0 ensures all detected frames are saved
max_fps_detection = float(os.getenv("MAX_FPS_DETECTION", 3))  # CPU/GPU usage limiter
STREAM_FPS = float(os.getenv("STREAM_FPS", 3))  # Max FPS for the output stream
output_resize_width = int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 800))  # Fixed output width after Docker start
day_and_night_capture = os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower() == "true"
day_and_night_capture_location = os.getenv("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin")
cpu_limit = int(float(os.getenv("CPU_LIMIT", 2)))  # Safe conversion

config = {
    "model_choice": model_choice,
    "class_filter": class_filter,
    "confidence_threshold": confidence_threshold,
    "save_threshold": save_threshold,
    "max_fps_detection": max_fps_detection,
    "STREAM_FPS": STREAM_FPS,
    "STREAM_WIDTH_OUTPUT_RESIZE": output_resize_width,
    "DAY_NIGHT_CAPTURE": day_and_night_capture,
    "DAY_NIGHT_CAPTURE_LOCATION": day_and_night_capture_location,
    "cpu_limit": cpu_limit
}
logger.info(f"Configuration: {json.dumps(config, indent=2)}")


output_dir = os.getenv("OUTPUT_DIR", "/output")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "all_bounding_boxes.csv")

# If the file doesn't exist yet, create it with a header row
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "class_name", "confidence", "x1", "y1", "x2", "y2"])

# Function to check if it's currently daytime
def is_daytime(city_name):
    """
    Returns True if it's currently daytime in the given city.
    """
    try:
        city = lookup(city_name, database())  # Get location info from database
        tz = pytz.timezone(city.timezone)
        now = datetime.now(tz)
        s = sun(city.observer, date=now)

        return s["sunrise"] < now < s["dusk"]
    except Exception as e:
        logger.error(f"Error determining daylight status: {e}")
        return True  # Default to True to avoid unintended blocking


# ------------------------------------------------------------------------------
# Global Video Source Retrieval and Processing
# ------------------------------------------------------------------------------
video_source_env = os.getenv("VIDEO_SOURCE", "0")  # Default to "0" if VIDEO_SOURCE isn't set

try:
    # Attempt to convert to integer (webcam index)
    video_source = int(video_source_env)
    logger.info("Video source is a webcam.")
except ValueError:
    # Treat as RTSP or HTTP stream URL
    video_source = video_source_env
    logger.info(f"Video source is a stream: {video_source}")


# ------------------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------------------
app = Flask(__name__)


def generate_frames():
    global latest_frame, detector_instance

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    timestamp_y = 50

    # Try loading the static placeholder image
    static_placeholder_path = "assets/static_placeholder.jpg"
    if os.path.exists(static_placeholder_path):
        static_placeholder = cv2.imread(static_placeholder_path)
        if static_placeholder is not None:
            # Compute aspect ratio from the original placeholder
            original_h, original_w = static_placeholder.shape[:2]
            ratio = original_h / float(original_w)

            # Derive placeholder height based on desired width and original ratio
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * ratio)

            # Resize the placeholder to keep its aspect ratio
            static_placeholder = cv2.resize(static_placeholder, (placeholder_w, placeholder_h))
            logger.info("Using static placeholder from file, adjusted to aspect ratio.")
        else:
            logger.error("Failed to load static placeholder image; falling back to black image.")
            # Fall back to a black image at the configured width, fixed ratio 16:9 (as an example)
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
    else:
        logger.error("Static placeholder image not found; falling back to black image.")
        # Fall back to a black image at the configured width, fixed ratio 16:9 (as an example)
        placeholder_w = output_resize_width
        placeholder_h = int(placeholder_w * 9 / 16)
        static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)

    while True:
        start_loop_time = time.time()
        try:
            # Retrieve the current resolution from the detector (if available)
            with detector_lock:
                if detector_instance:
                    resolution = detector_instance.resolution
                else:
                    resolution = None

            # Retrieve the latest frame
            with frame_lock:
                current_frame = latest_frame

            # If we have a frame and a resolution, resize the frame and send it
            if current_frame is not None and resolution:
                current_input_width, current_input_height = resolution

                # Calculate the new height while preserving the aspect ratio
                output_resize_height = int(current_input_height * placeholder_w / current_input_width)

                # Resize the frame
                resized_frame = cv2.resize(current_frame, (placeholder_w, output_resize_height))

                # Add timestamp to the resized frame
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    resized_frame,
                    timestamp,
                    (10, timestamp_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness
                )

                # Encode the frame to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 60% quality
                ret, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
                else:
                    logger.error("Failed to encode resized frame.")
            else:
                # No current frame/resolution -> use the placeholder
                placeholder_with_time = static_placeholder.copy()

                # Add random noise
                noise = np.random.randint(-100, 20, placeholder_with_time.shape, dtype=np.int16)
                placeholder_with_time = np.clip(placeholder_with_time.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                # Add timestamp to the placeholder
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    placeholder_with_time,
                    timestamp,
                    (10, timestamp_y),
                    font,
                    font_scale,
                    (0, 255, 0),
                    font_thickness
                )

                # Encode the placeholder frame
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 60% quality
                ret, buffer = cv2.imencode('.jpg', placeholder_with_time, encode_param)
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
                else:
                    logger.error("Failed to encode placeholder frame.")

        except Exception as e:
            logger.error(f"Error generating frames: {e}")
            time.sleep(0.1)  # Slight delay to prevent tight error loops

        # Enforce streaming FPS
        elapsed = time.time() - start_loop_time
        desired_frame_time = 1.0 / STREAM_FPS  # e.g. 0.2s for 5 FPS
        if elapsed < desired_frame_time:
            time.sleep(desired_frame_time - elapsed)

@app.route('/video_feed')
def video_feed():
    """
    Route to serve the MJPEG video feed.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """
    Simple HTML page displaying the video feed.
    """
    return (
        '''
        <html>
            <body>
                <h1>Live Stream</h1>
                <img src="/video_feed" style="max-width: 100%; height: auto;">
            </body>
        </html>
        '''
    )


def detection_loop():
    global latest_frame, detector_instance

    # Use the globally defined video_source
    source = video_source  # Already processed globally

    # Initialize Detector instance
    try:
        detector = Detector(source=source, model_choice=model_choice, debug=_debug)
        with detector_lock:
            detector_instance = detector  # Assign to global variable
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    print("Object detection started. Press 'Ctrl+C' to stop.")

    frame_count = 0
    last_save_time = 0

    retries = 0
    while True:
        try:
            with detector_lock:
                frame = detector_instance.get_frame()

            if frame is None:
                logger.debug("No frame available. Attempting to reinitialize detector...")
                detector_instance.release()
                time.sleep(min(2 ** retries, 30))  # Exponential backoff with a max of 30 seconds when reinitializing the detector fails.
                retries += 1
                try:
                    with detector_lock:
                        detector_instance = Detector(source=source, model_choice=model_choice, debug=_debug)
                    logger.debug("Detector reinitialized successfully.")

                    retries = 0  # Reset retries on success
                except Exception as e:
                    logger.debug(f"Failed to reinitialize detector: {e}")
                    time.sleep(1)  # Prevent rapid retries when the detector fails to initialize.
                    continue
                continue  # Restart loop after reinitializing

            if not day_and_night_capture and not is_daytime(day_and_night_capture_location):
                next_check = 300  # Default: Check every 5 minutes if an error occurs

                try:
                    city = lookup(day_and_night_capture_location, database())
                    now = datetime.now(pytz.timezone(city.timezone))
                    s = sun(city.observer, date=now)

                    logger.info(
                        f"🌙 Nighttime in {day_and_night_capture_location} - Current Time: {now}, Sunrise: {s['sunrise']}, Dusk: {s['dusk']}")

                    if now < s["sunrise"]:
                        next_check = (s["sunrise"] - now).seconds
                    elif now > s["dusk"]:  # If it's after sunset, wait until the next sunrise
                        tomorrow_sunrise = sun(city.observer, date=now + timedelta(days=1))["sunrise"]
                        next_check = (tomorrow_sunrise - now).seconds

                except Exception as e:
                    logger.error(f"⚠️ Could not determine next sunrise time: {e}")
                    next_check = 300  # Fallback: Check again in 5 minutes

                logger.info(
                    f"🌙 Nighttime detected in {day_and_night_capture_location}. Sleeping for {next_check} seconds before checking again.")
                time.sleep(next_check)  # Sleep until the next daylight check
                continue  # Restart loop after sleep

            start_time = time.time()
            # We now get 4 values back
            annotated_frame, object_detected, original_frame, detection_info_list = detector.detect_objects(
                frame,
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
                save_threshold=save_threshold
            )
            detection_time = time.time() - start_time

            logger.debug(f"Detection took {detection_time:.4f} seconds for one frame.")

            # Update the global frame for streaming
            with frame_lock:
                latest_frame = annotated_frame.copy()  # Make a writable copy

            # Save results if needed
            current_time = time.time()
            if object_detected and (save_interval == 0 or (current_time - last_save_time >= save_interval)):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                annotated_name = f"{timestamp}_frame_annotated.jpg"
                original_name = f"{timestamp}_frame_original.jpg"

                # Save annotated and original frames
                cv2.imwrite(os.path.join(output_dir, annotated_name), annotated_frame)
                cv2.imwrite(os.path.join(output_dir, original_name), original_frame)
                print(f"Saved frames: {annotated_name}, {original_name}")

                # Save bounding box details
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    for det in detection_info_list:
                        writer.writerow([
                            timestamp,
                            annotated_name,
                            det["class_name"],
                            f"{det['confidence']:.2f}",
                            det["x1"], det["y1"], det["x2"], det["y2"]
                        ])
                last_save_time = current_time

                # 🔔 ✅ Send Telegram alert *after* saving the image
                if detection_info_list:
                    detected_classes = ", ".join(
                        set(det["class_name"] for det in detection_info_list))  # Unique class names
                    detection_text = f"🔎 Detection Alert!\nDetected: {detected_classes}"

                    # Send the stored annotated image to Telegram
                    send_telegram_message(text=detection_text, photo_path=os.path.join(output_dir, annotated_name))
                    logger.info(f"📩 Telegram notification sent: {detection_text}")

            frame_count += 1

            detection_duration = time.time() - start_time
            target_duration = 1.0 / max_fps_detection
            if detection_duration < target_duration:
                time.sleep(target_duration - detection_duration)
        except Exception as e:
            print(f"Error in detection loop: {e}")
            time.sleep(1)  # Prevents rapid error handling loops.


# Register cleanup function to ensure resources are released on exit
import atexit

def cleanup():
    global detector_instance
    with detector_lock:
        if detector_instance:
            detector_instance.release()

atexit.register(cleanup)

# Start the detection loop in a separate thread
detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

if __name__ == '__main__':
    # Start the Flask server using Waitress
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)
