# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Dash Web Interface
# ------------------------------------------------------------------------------
from flask import send_from_directory, Response
from dash import Dash, html, dcc, callback_context, ALL, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv()
import json
import os
import math
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
max_fps_detection = float(os.getenv("MAX_FPS_DETECTION", 3))  # Lowering this reduces the frame rate for inference, thus lowering CPU usage.
STREAM_FPS = float(os.getenv("STREAM_FPS", 3))  # Lowering this reduces the frame rate for streaming, conserving resources.
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

# Additional constants for the frontend
IMAGE_WIDTH = 150
RECENT_IMAGES_COUNT = 3
PAGE_SIZE = 20  # Number of images per page


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

def get_captured_images() -> list:
    """Returns a list of captured images sorted by date."""
    try:
        files = [f for f in os.listdir(output_dir) if f.endswith("_frame_annotated.jpg")]
        files.sort(reverse=True)  # Neueste zuerst
        return files
    except Exception as e:
        logger.error(f"Error retrieving captured images: {e}")
        return []

def derive_original_filename(annotated_filename: str,
                             annotated_suffix: str = "_frame_annotated",
                             original_suffix: str = "_frame_original") -> str:
    """Derives the original image filename from the annotated filename."""
    return annotated_filename.replace(annotated_suffix, original_suffix)

def derive_zoomed_filename(annotated_filename: str,
                           annotated_suffix: str = "_frame_annotated",
                           zoomed_suffix: str = "_frame_zoomed") -> str:
    """Derives the zoomed image filename from the annotated filename."""
    return annotated_filename.replace(annotated_suffix, zoomed_suffix)

def create_thumbnail(image_filename: str, index: int) -> html.Button:
    """
    Creates a thumbnail button that shows the zoomed version of the annotated image.
    When clicked, the modal will show the original image.
    """
    zoomed_filename = derive_zoomed_filename(image_filename)
    style = {
        "cursor": "pointer",
        "border": "none",
        "background": "none",
        "padding": "5px",
        "width": f"{IMAGE_WIDTH}px"
    }
    return html.Button(
        html.Img(
            src=f"/images/{zoomed_filename}",
            alt=f"Thumbnail of {zoomed_filename}",
            style=style
        ),
        id={'type': 'thumbnail', 'index': index},
        n_clicks=0,
        style={"border": "none", "background": "none", "padding": "0"}
    )

def create_image_modal(image_filename: str, index: int) -> dbc.Modal:
    """
    Creates a modal to display the original image when its thumbnail is clicked.
    The modal header will show the filename, and the body displays the full-size original image.
    """
    original_filename = image_filename
    return dbc.Modal(
        [
            # Disable the default close button by setting close_button=False.
            dbc.ModalHeader(dbc.ModalTitle(original_filename), close_button=False),
            dbc.ModalBody(
                html.Img(src=f"/images/{original_filename}", style={"width": "100%"})
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id={'type': 'close', 'index': index}, className="ml-auto", n_clicks=0)
            ),
        ],
        id={'type': 'modal', 'index': index},
        is_open=False,
        size="lg",
    )

def generate_recent_gallery() -> html.Div:
    """Generates the gallery of the last captured images including modals."""
    images = get_captured_images()
    recent_images = images[:RECENT_IMAGES_COUNT]
    thumbnails = [create_thumbnail(img, i) for i, img in enumerate(recent_images)]
    modals = [create_image_modal(img, i) for i, img in enumerate(recent_images)]
    return html.Div(thumbnails + modals, id="recent-gallery", style={"textAlign": "center"})

def generate_gallery() -> html.Div:
    """Generates a grid view of all captured images with modals for larger display."""
    images = get_captured_images()
    grid_items = []
    modals = []
    for i, img in enumerate(images):
        grid_items.append(
            html.Div(
                create_thumbnail(img, i),
                style={
                    "flex": "1 0 21%",  # Adjust to control items per row
                    "margin": "5px",
                    "maxWidth": "150px"
                }
            )
        )
        modals.append(create_image_modal(img, i))
    return html.Div(
        grid_items + modals,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "center"
        }
    )

video_source_env = os.getenv("VIDEO_SOURCE", "0")
try:
    video_source = int(video_source_env)
    logger.info("Video source is a webcam.")
except ValueError:
    video_source = video_source_env
    logger.info(f"Video source is a stream: {video_source}")


# -----------------------------
# Frame Generation Function
# -----------------------------
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


# ------------------------------------------------------------------------------
# 8. Flask routes: static files and video feed
# ------------------------------------------------------------------------------
def serve_image(filename):
    image_path = os.path.join(output_dir, filename)
    if not os.path.exists(image_path):
        return "Image not found", 404
    return send_from_directory(output_dir, filename)

# Registering the static route
from flask import Flask
server = Flask(__name__)
server.route("/images/<path:filename>")(serve_image)
server.route("/video_feed")(lambda: Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"))

# ------------------------------------------------------------------------------
# 9. Dash App Setup und Layouts
# ------------------------------------------------------------------------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

def stream_layout() -> html.Div:
    """Responsive layout for the main page with live stream and recent images gallery."""
    return dbc.Container([
        dbc.NavbarSimple(
            brand=html.Img(
                src="/assets/the_birdwatcher.jpeg",
                className="img-fluid",
                style={"width": "20vw", "minWidth": "100px", "maxWidth": "200px"}
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Image Gallery", href="/gallery", className="mx-auto"))
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center"
        ),
        dbc.Row([
            dbc.Col(html.H1("Real-Time Stream", className="text-center"), width=12, className="my-3")
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-video",
                    type="default",
                    children=html.Img(
                        id="video-feed",
                        src="/video_feed",
                        style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto"}
                    )
                ),
                width=12
            )
        ], className="my-3"),
        dbc.Row([
            dbc.Col(html.H2("Recent Detections", className="text-center"), width=12, className="mt-4")
        ]),
        dbc.Row([
            dbc.Col(generate_recent_gallery(), width=12)
        ], className="mb-5")
    ], fluid=True)


def gallery_layout() -> html.Div:
    return dbc.Container([
        dbc.NavbarSimple(
            brand=html.Img(
                src="/assets/the_birdwatcher.jpeg",
                className="img-fluid",
                style={"width": "20vw", "minWidth": "100px", "maxWidth": "200px"}
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Image Gallery", href="/gallery", className="mx-auto"))
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center"
        ),
        # Store to hold the current page index
        dcc.Store(id="page-store", data=0),
        dbc.Row([
            dbc.Col(html.H1("Image Gallery", className="text-center"), width=12, className="my-3")
        ]),
        dbc.Row([
            # Pagination controls
            dbc.Col([
                dbc.Button("Previous", id="prev-page", n_clicks=0, color="primary", className="me-2"),
                dbc.Button("Next", id="next-page", n_clicks=0, color="primary"),
            ], width=12, className="mb-2", style={"textAlign": "center"}),
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="gallery-content"), width=12)
        ], className="mb-5"),
    ], fluid=True)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

# ------------------------------------------------------------------------------
# 10. Callbacks
# ------------------------------------------------------------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/gallery":
        return gallery_layout()
    elif pathname == "/" or pathname == "":
        return stream_layout()
    else:
        return "404 Not Found"

# Callback for opening/closing image modals
@app.callback(
    Output({"type": "modal", "index": ALL}, "is_open"),
    [Input({'type': 'thumbnail', 'index': ALL}, 'n_clicks'),
     Input({'type': 'close', 'index': ALL}, 'n_clicks')],
    [State({"type": "modal", "index": ALL}, "is_open")]
)
def toggle_modal(thumbnail_clicks, close_clicks, current_states):
    ctx = callback_context
    if not ctx.triggered:
        return current_states

    triggered_prop = ctx.triggered[0]['prop_id']
    triggered_id = json.loads(triggered_prop.split('.')[0])
    new_states = [False] * len(current_states)
    # If a thumbnail is clicked, open that modal.
    if "thumbnail" in triggered_prop:
        new_states[triggered_id["index"]] = True
    # If a close button is clicked, ensure that modal is closed.
    elif "close" in triggered_prop:
        new_states[triggered_id["index"]] = False
    return new_states

# Callback to show/hide the spinner while the video is loading.
@app.callback(
    Output("loading-spinner", "style"),
    [Input("video-feed", "loading_state")],
    prevent_initial_call=True
)
def show_hide_spinner(loading_state):
    if loading_state and loading_state.get('is_loading'):
        return {"display": "block"}
    else:
        return {"display": "none"}

@app.callback(
    [Output("gallery-content", "children"),
     Output("page-store", "data")],
    [Input("prev-page", "n_clicks"),
     Input("next-page", "n_clicks"),
     Input("page-store", "data")]
)
def update_gallery_and_page(prev_clicks, next_clicks, current_page):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    images = get_captured_images()
    total_pages = math.ceil(len(images) / PAGE_SIZE) if images else 1

    # Update current_page based on the button clicked
    if triggered_id == "prev-page" and current_page > 0:
        current_page -= 1
    elif triggered_id == "next-page" and current_page < total_pages - 1:
        current_page += 1

    # Build the subset for the current page
    start_idx = current_page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(images))
    subset = images[start_idx:end_idx]

    # Create gallery using local (page-relative) indices
    grid_items = []
    modals = []
    for i, img in enumerate(subset):
        # Use 'i' instead of start_idx + i so that indices go from 0 to len(subset)-1
        grid_items.append(
            html.Div(
                create_thumbnail(img, i),
                style={"flex": "1 0 21%", "margin": "5px", "maxWidth": "150px"}
            )
        )
        modals.append(create_image_modal(img, i))

    gallery_div = html.Div(
        grid_items + modals,
        style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
    )

    return gallery_div, current_page

# -----------------------------
# Run the Dash App
# -----------------------------
if __name__ == '__main__':
    # Only run the development server when not in production
    if _debug:
        app.run_server(host='0.0.0.0', port=8050, debug=True)
