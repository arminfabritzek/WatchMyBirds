# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Streaming
# ------------------------------------------------------------------------------

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

# Global lock for thread-safe frame access
frame_lock = threading.Lock()
latest_frame = None  # Global variable to store the latest frame for streaming

# Global Detector instance and lock
detector_instance = None
detector_lock = threading.Lock()

# Global Video Capture instance and lock
video_capture_lock = threading.Lock()
video_capture = None  # Global variable for the VideoCapture instance

# Global lock for telegram notifications
telegram_lock = threading.Lock()

# Timestamp of the last updated frame
latest_frame_timestamp = 0

# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
model_choice = os.getenv("MODEL_CHOICE", "yolo8n")  # only "yolo8n" supported now
class_filter = json.loads(os.getenv("CLASS_FILTER", '["bird", "person"]'))
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
save_threshold = float(os.getenv("SAVE_THRESHOLD", 0.8))
max_fps_detection = float(os.getenv("MAX_FPS_DETECTION", 3))
STREAM_FPS = float(os.getenv("STREAM_FPS", 3))
output_resize_width = int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 800))
day_and_night_capture = os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower() == "true"
day_and_night_capture_location = os.getenv("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin")
cpu_limit = int(float(os.getenv("CPU_LIMIT", 2)))
model_path = os.getenv("YOLO8N_MODEL_PATH", "models/yolov8n.pt")
telegram_cooldown = float(os.getenv("TELEGRAM_COOLDOWN", 5))  # seconds between telegram alerts

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

def is_daytime(city_name):
    try:
        city = lookup(city_name, database())
        tz = pytz.timezone(city.timezone)
        now = datetime.now(tz)
        s = sun(city.observer, date=now)
        return s["sunrise"] < now < s["dusk"]
    except Exception as e:
        logger.error(f"Error determining daylight status: {e}")
        return True

def send_alert(detected_classes, annotated_path):
    detection_text = f"🔎 Detection Alert!\nDetected: {detected_classes}"
    logger.debug("Sending Telegram alert asynchronously.")
    try:
        send_telegram_message(text=detection_text, photo_path=annotated_path)
        logger.info(f"📩 Telegram notification sent: {detection_text}")
    except Exception as e:
        logger.error(f"Error sending Telegram alert: {e}")

last_telegram_time = 0

video_source_env = os.getenv("VIDEO_SOURCE", "0")
try:
    video_source = int(video_source_env)
    logger.info("Video source is a webcam.")
except ValueError:
    video_source = video_source_env
    logger.info(f"Video source is a stream: {video_source}")

app = Flask(__name__)

def generate_frames():
    global latest_frame, latest_frame_timestamp, video_capture
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    timestamp_y = 50
    placeholder_threshold = 5  # seconds without a new frame before showing placeholder

    static_placeholder_path = "assets/static_placeholder.jpg"
    if os.path.exists(static_placeholder_path):
        static_placeholder = cv2.imread(static_placeholder_path)
        if static_placeholder is not None:
            original_h, original_w = static_placeholder.shape[:2]
            ratio = original_h / float(original_w)
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * ratio)
            static_placeholder = cv2.resize(static_placeholder, (placeholder_w, placeholder_h))
            logger.info("Using static placeholder from file, adjusted to aspect ratio.")
        else:
            logger.error("Failed to load static placeholder image; using black image.")
            placeholder_w = output_resize_width
            placeholder_h = int(placeholder_w * 9 / 16)
            static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)
    else:
        logger.error("Static placeholder not found; using black image.")
        placeholder_w = output_resize_width
        placeholder_h = int(placeholder_w * 9 / 16)
        static_placeholder = np.zeros((placeholder_h, placeholder_w, 3), dtype=np.uint8)

    while True:
        start_loop_time = time.time()
        try:
            # Retrieve resolution if video_capture is defined; otherwise, set to None.
            with video_capture_lock:
                if video_capture is None:
                    logger.debug("video_capture is not initialized; using placeholder.")
                    resolution = None
                else:
                    resolution = video_capture.resolution

            with frame_lock:
                current_frame = latest_frame
                last_ts = latest_frame_timestamp
            # If we have a fresh frame and a valid resolution, use it;
            # otherwise, show the placeholder.
            if current_frame is not None and resolution and (time.time() - last_ts < placeholder_threshold):
                current_input_width, current_input_height = resolution
                output_resize_height = int(current_input_height * placeholder_w / current_input_width)
                resized_frame = cv2.resize(current_frame, (placeholder_w, output_resize_height))
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(resized_frame, timestamp, (10, timestamp_y),
                            font, font_scale, (0, 255, 0), font_thickness)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode('.jpg', resized_frame, encode_param)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    logger.error("Failed to encode resized frame.")
            else:
                # Either no new frame, or video_capture is not defined—show the placeholder.
                placeholder_with_time = static_placeholder.copy()
                noise = np.random.randint(-100, 20, placeholder_with_time.shape, dtype=np.int16)
                placeholder_with_time = np.clip(placeholder_with_time.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(placeholder_with_time, timestamp, (10, timestamp_y),
                            font, font_scale, (0, 255, 0), font_thickness)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode('.jpg', placeholder_with_time, encode_param)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    logger.error("Failed to encode placeholder frame.")
        except Exception as e:
            logger.error(f"Error generating frames: {e}")
            time.sleep(0.1)
        elapsed = time.time() - start_loop_time
        desired_frame_time = 1.0 / STREAM_FPS
        if elapsed < desired_frame_time:
            time.sleep(desired_frame_time - elapsed)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
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
    global latest_frame, latest_frame_timestamp, detector_instance, last_telegram_time, video_capture

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

            if not day_and_night_capture and not is_daytime(day_and_night_capture_location):
                try:
                    city = lookup(day_and_night_capture_location, database())
                    now = datetime.now(pytz.timezone(city.timezone))
                    s = sun(city.observer, date=now)
                    logger.info(f"🌙 Nighttime in {day_and_night_capture_location} - Now: {now}, Sunrise: {s['sunrise']}, Dusk: {s['dusk']}")
                    if now < s["sunrise"]:
                        next_check = (s["sunrise"] - now).seconds
                    elif now > s["dusk"]:
                        tomorrow_sunrise = sun(city.observer, date=now + timedelta(days=1))["sunrise"]
                        next_check = (tomorrow_sunrise - now).seconds
                except Exception as e:
                    logger.error(f"⚠️ Could not determine next sunrise time: {e}")
                    next_check = 300
                logger.info(f"🌙 Nighttime detected in {day_and_night_capture_location}. Sleeping for {next_check} seconds.")
                time.sleep(next_check)
                continue

            # Run object detection on the captured frame.
            try:
                start_time = time.time()
                annotated_frame, object_detected, original_frame, detection_info_list = detector_instance.detect_objects(
                    frame, class_filter=class_filter,
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
                cv2.imwrite(os.path.join(output_dir, annotated_name), annotated_frame)
                cv2.imwrite(os.path.join(output_dir, original_name), original_frame)
                print(f"Saved frames: {annotated_name}, {original_name}")

                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    for det in detection_info_list:
                        writer.writerow([
                            timestamp, annotated_name, det["class_name"],
                            f"{det['confidence']:.2f}",
                            det["x1"], det["y1"], det["x2"], det["y2"]
                        ])

                # Send Telegram alert if detection information is available.
                if detection_info_list:
                    with telegram_lock:
                        if current_time - last_telegram_time >= telegram_cooldown:
                            detected_classes = ", ".join(set(det["class_name"] for det in detection_info_list))
                            alert_thread = threading.Thread(
                                target=send_alert,
                                args=(detected_classes, os.path.join(output_dir, annotated_name)),
                                daemon=True
                            )
                            alert_thread.start()
                            logger.info(f"📩 Telegram notification sent: 🔎 Detection Alert! Detected: {detected_classes}")
                            last_telegram_time = current_time

            frame_count += 1
            detection_duration = time.time() - start_time
            target_duration = 1.0 / max_fps_detection
            if detection_duration < target_duration:
                time.sleep(target_duration - detection_duration)
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
            # If any error occurs, reinitialize the detector.
            with detector_lock:
                try:
                    detector_instance = Detector(model_choice=model_choice, debug=_debug)
                except Exception as e2:
                    logger.error(f"Detector reinitialization failed: {e2}")
            time.sleep(1)

import atexit
def cleanup():
    global video_capture
    with video_capture_lock:
        if video_capture:
            video_capture.release()
atexit.register(cleanup)

detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)