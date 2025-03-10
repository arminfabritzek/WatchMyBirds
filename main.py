# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Dash Web Interface
# main.py
# ------------------------------------------------------------------------------

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
import logging
from astral.geocoder import database, lookup
from astral.sun import sun
from camera.detector import Detector
from utils.telegram_notifier import send_telegram_message
from utils.cpu_limiter import restrict_to_cpus  # Import CPU limiter
# Initialize VideoCapture
from camera.video_capture import VideoCapture

# Apply CPU restriction before starting any threads for slow systems
restrict_to_cpus()

# Read the debug flag from the environment variable (default: False)
_debug = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if _debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG if _debug else logging.INFO)

logger.info(f"Debug mode is {'enabled' if _debug else 'disabled'}.")
print(f"DEBUG_MODE environment variable: {os.getenv('DEBUG_MODE')}")
print(f"Debug mode in code: {_debug}")

if _debug:
    send_telegram_message(text="🐦 Birdwatching has started in DEBUG mode!", photo_path="assets/the_birdwatcher_small.jpeg")

# -----------------------------
# Global Variables and Locks
# -----------------------------
frame_lock = threading.Lock()
latest_frame = None
detector_instance = None
detector_lock = threading.Lock()
video_capture_lock = threading.Lock()
video_capture = None
telegram_lock = threading.Lock()
latest_frame_timestamp = 0
detection_occurred = False
last_notification_time = time.time()  # using time.time() for wall-clock
detection_counter = 0  # Global variable to count detections between alerts
detection_classes_agg = set()  # Aggregated set of detected classes since last alert.

# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
model_choice = os.getenv("MODEL_CHOICE", "yolo")  # only "yolo" supported for now
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
save_threshold = float(os.getenv("SAVE_THRESHOLD", 0.8))
max_fps_detection = float(os.getenv("MAX_FPS_DETECTION", 1.0))  # Lowering this reduces the frame rate for inference, thus lowering CPU usage.
STREAM_FPS = float(os.getenv("STREAM_FPS", 1))  # Lowering this reduces the frame rate for streaming, conserving resources.
output_resize_width = int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 640))
day_and_night_capture = os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower() == "true"
day_and_night_capture_location = os.getenv("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin")
cpu_limit = int(float(os.getenv("CPU_LIMIT", 2)))
model_path = os.getenv("YOLO8N_MODEL_PATH", "models/best.pt")
telegram_cooldown = float(os.getenv("TELEGRAM_COOLDOWN", 5))  # seconds between telegram alerts

config = {
    "model_choice": model_choice,
    "confidence_threshold": confidence_threshold,
    "save_threshold": save_threshold,
    "max_fps_detection": max_fps_detection,
    "STREAM_FPS": STREAM_FPS,
    "STREAM_WIDTH_OUTPUT_RESIZE": output_resize_width,
    "DAY_NIGHT_CAPTURE": day_and_night_capture,
    "DAY_NIGHT_CAPTURE_LOCATION": day_and_night_capture_location,
    "cpu_limit": cpu_limit,
    "model_path": model_path,
    "telegram_cooldown": telegram_cooldown
}
logger.info(f"Configuration: {json.dumps(config, indent=2)}")

output_dir = os.getenv("OUTPUT_DIR", "/output")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "all_bounding_boxes.csv")
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "class_name", "confidence", "x1", "y1", "x2", "y2"])

# -----------------------------
# Helper Functions
# -----------------------------
def is_daytime(city_name):
    """
    Returns True if the current time is between dawn and dusk,
    meaning there is sufficient light (including post-sunset twilight).
    """
    try:
        city = lookup(city_name, database())
        tz = pytz.timezone(city.timezone)
        now = datetime.now(tz)
        # Use 12 for nautical dawn (or 18 for astronomical dawn, if desired)
        dawn_depression = 12
        s = sun(city.observer, date=now, tzinfo=tz, dawn_dusk_depression=dawn_depression)
        # Capture is active from dawn until dusk.
        return s["dawn"] < now < s["dusk"]
    except Exception as e:
        logger.error(f"Error determining daylight status: {e}")
        # Default to daytime to avoid halting capture on error.
        return True

def send_alert(detected_classes, annotated_path):
    detection_text = f"🔎 Detection Alert!\nDetected: {detected_classes}"
    logger.debug("Sending Telegram alert asynchronously.")
    try:
        send_telegram_message(text=detection_text, photo_path=annotated_path)
        logger.info(f"📩 Telegram notification sent: {detection_text}")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")

video_source_env = os.getenv("VIDEO_SOURCE", "0")
try:
    video_source = int(video_source_env)
    logger.info("Video source is a webcam.")
except ValueError:
    video_source = video_source_env
    logger.info(f"Video source is a stream.")

try:
    video_capture = VideoCapture(video_source, debug=_debug)
    logger.info("VideoCapture initial initialization.")
except Exception as e:
    logger.error(f"Failed to initialize video capture.")

# -----------------------------
# Detection Loop Function
# -----------------------------
def detection_loop():
    global latest_frame, latest_frame_timestamp, detector_instance, video_capture
    global last_notification_time, detection_occurred, detection_counter, detection_classes_agg

    # Keep trying to initialize video_capture and detector until successful.
    while video_capture is None:
        try:
            video_capture = VideoCapture(video_source, debug=_debug)
            logger.info("VideoCapture initialized successfully in detection loop.")
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}. Retrying in 5 seconds.")
            time.sleep(5)

    while detector_instance is None:
        try:
            detector_instance = Detector(model_choice=model_choice, debug=_debug)
            logger.info("Detector initialized successfully in detection loop.")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}. Retrying in 5 seconds.")
            time.sleep(5)

    print("Object detection started. Press 'Ctrl+C' to stop.")
    frame_count = 0

    while True:
        try:
            # Retrieve the latest frame from VideoCapture.
            with video_capture_lock:
                frame = video_capture.get_frame()
            if frame is None:
                logger.debug("No frame available from VideoCapture. Pausing detection.")
                time.sleep(5)  # Pause detection entirely for 5 seconds
                continue

            # Always update the video stream (latest_frame) regardless of detection.
            with frame_lock:
                latest_frame = frame.copy()
                latest_frame_timestamp = time.time()

            # Determine if we should run detection.
            # - If DAY_AND_NIGHT_CAPTURE is True: run detection regardless.
            # - If DAY_AND_NIGHT_CAPTURE is False: run detection only if it's daytime.
            run_detection = False
            if day_and_night_capture:
                run_detection = True
            else:
                if is_daytime(day_and_night_capture_location):
                    run_detection = True
                else:
                    # Calculate sleep duration until next dawn.
                    try:
                        city = lookup(day_and_night_capture_location, database())
                        tz = pytz.timezone(city.timezone)
                        now = datetime.now(tz)
                        dawn_depression = 12  # Use 12 for nautical dawn (or change to 18 for astronomical dawn)
                        s = sun(city.observer, date=now, tzinfo=tz, dawn_dusk_depression=dawn_depression)
                        logger.info(
                            f"🌙 Insufficient light in {day_and_night_capture_location} - Now: {now}, Dawn: {s['dawn']}, Dusk: {s['dusk']}")

                        # Determine sleep duration:
                        if now < s["dawn"]:
                            # Before dawn, sleep until dawn.
                            next_check = (s["dawn"] - now).seconds
                        elif now >= s["dusk"]:
                            # After dusk, sleep until the next dawn.
                            tomorrow = now + timedelta(days=1)
                            tomorrow_s = sun(city.observer, date=tomorrow, tzinfo=tz, dawn_dusk_depression=dawn_depression)
                            next_check = (tomorrow_s["dawn"] - now).seconds
                        else:
                            # Fallback: short sleep.
                            next_check = 60
                    except Exception as e:
                        logger.error(f"⚠️ Could not determine next dawn time: {e}")
                        next_check = 60

                    logger.info(
                        f"🌙 Light insufficient for capture in {day_and_night_capture_location}. Sleeping for {next_check} seconds.")
                    time.sleep(next_check)
                    continue

            # If we have decided to run detection on the captured frame.
            if run_detection:
                try:
                    start_time = time.time()
                    annotated_frame, object_detected, original_frame, detection_info_list = detector_instance.detect_objects(
                        frame,
                        confidence_threshold=confidence_threshold,
                        save_threshold=save_threshold
                    )
                except Exception as e:
                    logger.error(f"Inference error detected: {e}. Reinitializing detector...")
                    with detector_lock:
                        try:
                            detector_instance = Detector(model_choice=model_choice, debug=_debug)
                        except Exception as e2:
                            logger.error(f"Detector reinitialization failed: {e2}")
                    time.sleep(1)
                    continue

                detection_time = time.time() - start_time
                logger.debug(f"Detection took {detection_time:.4f} seconds for one frame.")

                with frame_lock:
                    latest_frame = annotated_frame.copy()
                    latest_frame_timestamp = time.time()  # update timestamp when a new frame arrives

                current_time = time.time()
                if object_detected:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    annotated_name = f"{timestamp}_frame_annotated.jpg"
                    original_name = f"{timestamp}_frame_original.jpg"
                    zoomed_name = f"{timestamp}_frame_zoomed.jpg"

                    cv2.imwrite(os.path.join(output_dir, annotated_name), annotated_frame)
                    cv2.imwrite(os.path.join(output_dir, original_name), original_frame)

                    # Generate the zoomed version based on the first detection's bounding box.
                    if detection_info_list:
                        # For simplicity, use the first detected object's bounding box.
                        first_det = detection_info_list[0]
                        x1, y1, x2, y2 = first_det["x1"], first_det["y1"], first_det["x2"], first_det["y2"]
                        # Optionally add some margin (e.g. 10 pixels) and ensure the values are within image bounds.
                        margin = 100
                        h, w = annotated_frame.shape[:2]
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        zoomed_frame = annotated_frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(output_dir, zoomed_name), zoomed_frame)
                    else:
                        # If no bounding box is available, fall back to a resized annotated frame.
                        cv2.imwrite(os.path.join(output_dir, zoomed_name), annotated_frame)

                    with open(csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        for det in detection_info_list:
                            writer.writerow([
                                timestamp, annotated_name, det["class_name"],
                                f"{det['confidence']:.2f}",
                                det["x1"], det["y1"], det["x2"], det["y2"]
                            ])

                    # Mark that a detection occurred and update the aggregated values.
                    detection_occurred = True
                    detection_counter += len(detection_info_list)
                    # Aggregate the classes from the current detection.
                    detection_classes_agg.update(det["class_name"] for det in detection_info_list)

                    # Send Telegram alert if detection information is available.
                    with telegram_lock:
                        if detection_occurred and (current_time - last_notification_time >= telegram_cooldown):
                            aggregated_classes = ", ".join(sorted(detection_classes_agg))
                            alert_text = (f"🔎 Detection Alert!\n"
                                          f"Detected classes: {aggregated_classes}\n"
                                          f"Total detections since last alert: {detection_counter}")
                            send_telegram_message(
                                text=alert_text,
                                photo_path=os.path.join(output_dir, annotated_name)
                            )
                            logger.info(f"📩 Telegram notification sent: {alert_text}")
                            last_notification_time = current_time
                            detection_occurred = False
                            detection_counter = 0  # Reset the counter.
                            detection_classes_agg = set()  # Reset the aggregated classes.
                        else:
                            logger.debug("Cooldown active, not sending alert.")

                frame_count += 1
                detection_duration = time.time() - start_time
                target_duration = 1.0 / max_fps_detection
                sleep_time = target_duration - detection_duration
                logger.debug(f"Detection duration: {detection_duration:.4f}s, sleeping for: {sleep_time:.4f}s")
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            # If any error occurs, reinitialize the detector.
            with detector_lock:
                try:
                    detector_instance = Detector(model_choice=model_choice, debug=_debug)
                except Exception as e2:
                    logger.error(f"Detector reinitialization failed: {e2}")
            time.sleep(1)

# -----------------------------
# Cleanup Function
# -----------------------------
import atexit
def cleanup():
    global video_capture
    with video_capture_lock:
        if video_capture:
            video_capture.release()
atexit.register(cleanup)


# -----------------------------
# Start Detection Loop in Background
# -----------------------------
detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()


# -----------------------------
# Import and Run the Web Interface
# -----------------------------
from web.web_interface import create_web_interface

# Prepare parameters to pass to the web interface module.
params = {
    "output_dir": output_dir,
    "video_capture": video_capture,
    "output_resize_width": output_resize_width,
    "STREAM_FPS": STREAM_FPS,
    "IMAGE_WIDTH": 150,
    "RECENT_IMAGES_COUNT": 3,
    "PAGE_SIZE": 20,
}

interface = create_web_interface(params)

# Expose the Flask server as the WSGI app for Waitress.
app = interface["server"]

if __name__ == '__main__':
    # Run the web interface
    interface["run"](debug=_debug, host='0.0.0.0', port=8050)
