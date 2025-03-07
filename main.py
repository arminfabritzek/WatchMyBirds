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
last_telegram_time = 0

# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
model_choice = os.getenv("MODEL_CHOICE", "yolo")  # only "yolo" supported for now
class_filter = json.loads(os.getenv("CLASS_FILTER", '["bird", "person"]'))
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
save_threshold = float(os.getenv("SAVE_THRESHOLD", 0.8))
max_fps_detection = float(os.getenv("MAX_FPS_DETECTION", 3))
STREAM_FPS = float(os.getenv("STREAM_FPS", 3))
output_resize_width = int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 640))
day_and_night_capture = os.getenv("DAY_AND_NIGHT_CAPTURE", "True").lower() == "true"
day_and_night_capture_location = os.getenv("DAY_AND_NIGHT_CAPTURE_LOCATION", "Berlin")
cpu_limit = int(float(os.getenv("CPU_LIMIT", 2)))
model_path = os.getenv("YOLO8N_MODEL_PATH", "models/best.pt")
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

# Additional constants for the frontend
THUMBNAIL_WIDTH = 150
IMAGE_WIDTH = 150
RECENT_IMAGES_COUNT = 5
CAROUSEL_INTERVAL = 0


# -----------------------------
# Helper Functions
# -----------------------------
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

def get_captured_images() -> list:
    """Returns a list of captured images sorted by date."""
    try:
        files = [f for f in os.listdir(output_dir) if f.endswith("_frame_annotated.jpg")]
        files.sort(reverse=True)  # Neueste zuerst
        return files
    except Exception as e:
        logger.error(f"Error retrieving captured images: {e}")
        return []

def create_thumbnail(image_filename: str, index: int, carousel: bool = False) -> html.Button:
    """Creates a thumbnail button that can be clicked on for the image."""
    style = {
        "cursor": "pointer",
        "border": "none",
        "background": "none",
        "padding": "5px"
    }
    # Larger images are displayed in the stream.
    if not carousel:
        style["width"] = f"{IMAGE_WIDTH}px"
        id_type = 'thumbnail'
    else:
        style["width"] = f"{THUMBNAIL_WIDTH}px"
        id_type = 'carousel-thumbnail'

    return html.Button(
        html.Img(
            src=f"/images/{image_filename}",
            alt=f"Thumbnail of {image_filename}",
            style=style
        ),
        id={'type': id_type, 'index': index},
        n_clicks=0,
        style={"border": "none", "background": "none", "padding": "0"}
    )

def create_image_modal(image_filename: str) -> dbc.Modal:
    """Creates a modal to display the enlarged image."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(image_filename)),
            dbc.ModalBody(
                html.Img(src=f"/images/{image_filename}", style={"width": "100%"})
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id=f"close-{image_filename}", className="ml-auto", n_clicks=0)
            ),
        ],
        id=f"modal-{image_filename}",
        is_open=False,
        size="lg",
    )

def generate_recent_gallery() -> html.Div:
    """Generates the gallery of the last captured images including modals."""
    images = get_captured_images()
    recent_images = images[:RECENT_IMAGES_COUNT]
    thumbnails = [create_thumbnail(img, i) for i, img in enumerate(recent_images)]
    modals = [create_image_modal(img) for img in recent_images]
    return html.Div(thumbnails + modals, id="recent-gallery", style={"textAlign": "center"})

def generate_carousel() -> dbc.Carousel:
    """Generates the image carousel."""
    images = get_captured_images()
    items = [
        {"key": str(i), "src": f"/images/{image}", "caption": image, "alt": image}
        for i, image in enumerate(images)
    ]
    return dbc.Carousel(
        id="carousel-component",
        items=items,
        controls=True,
        indicators=True,
        interval=CAROUSEL_INTERVAL,
        ride="carousel",
        style={"width": "80%", "margin": "auto"}
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
            # Center the logo and nav items by using a custom class.
            brand=html.Img(
                src="/assets/the_birdwatcher.jpeg",
                className="img-fluid",  # Responsive logo image
                style={
                    "width": "20vw",      # 20% of the viewport width
                    "minWidth": "100px",  # Do not shrink below 100px
                    "maxWidth": "200px"   # Do not grow larger than 200px
                }
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Image Gallery", href="/carousel", className="mx-auto"))
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center"  # Centers nav items in the navbar
        ),
        dbc.Row([
            dbc.Col(html.H1("Real-Time Stream", className="text-center"), width=12, className="my-3")
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Loading(
                    id="loading-video",
                    type="default",  # Spinner type; can be "circle" or "dot"
                    children=html.Img(
                        id="video-feed",
                        src="/video_feed",
                        style={
                            "width": "100%",
                            "maxWidth": "800px",
                            "display": "block",
                            "margin": "0 auto"
                        }
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


def gallery_carousel_layout() -> html.Div:
    """Responsive layout for the carousel page with inline thumbnails beneath the main carousel."""
    images = get_captured_images()

    # Build carousel items from images.
    carousel_items = [
        {"key": str(i), "src": f"/images/{img}", "caption": img, "alt": img}
        for i, img in enumerate(images)
    ]

    carousel = dbc.Carousel(
        id="carousel-component",
        items=carousel_items,
        controls=True,
        indicators=True,
        interval=CAROUSEL_INTERVAL,
        ride=False,
        style={"width": "100%", "maxWidth": "800px", "margin": "auto"}
    )

    # Build a row of thumbnails.
    thumbnails = html.Div(
        [
            html.Img(
                src=f"/images/{img}",
                alt=f"Thumbnail of {img}",
                style={
                    "width": "100%",
                    "maxWidth": "150px",
                    "padding": "5px",
                    "cursor": "pointer",
                    "transition": "transform 0.2s"  # Hover effect
                },
                id={'type': 'carousel-thumbnail', 'index': i},
                n_clicks=0
            )
            for i, img in enumerate(images)
        ],
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "center",
            "marginTop": "20px"
        }
    )

    return dbc.Container([
        dbc.NavbarSimple(
            # Center the logo and navigation items.
            brand=html.Img(
                src="/assets/the_birdwatcher.jpeg",
                className="img-fluid",
                style={
                    "width": "20vw",
                    "minWidth": "100px",
                    "maxWidth": "200px"
                }
            ),
            brand_href="/",
            children=[
                dbc.NavItem(dbc.NavLink("Live Stream", href="/", className="mx-auto")),
                dbc.NavItem(dbc.NavLink("Image Gallery", href="/carousel", className="mx-auto"))
            ],
            color="primary",
            dark=True,
            fluid=True,
            className="justify-content-center"
        ),
        dbc.Row([
            dbc.Col(html.H1("Image Gallery", className="text-center"), width=12, className="my-3")
        ]),
        dbc.Row([
            dbc.Col(carousel, width=12)
        ]),
        dbc.Row([
            dbc.Col(thumbnails, width=12)
        ], className="mt-3")
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
    if pathname == "/carousel":
        return gallery_carousel_layout()
    elif pathname == "/" or pathname == "":
        return stream_layout()
    else:
        return "404 Not Found"

# callback for updating the carousel active index when clicking on thumbnails
@app.callback(
    Output("carousel-component", "active_index"),
    [Input({'type': 'carousel-thumbnail', 'index': ALL}, 'n_clicks')],
    [State({'type': 'carousel-thumbnail', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_carousel_index(n_clicks_list, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        clicked_id = json.loads(trigger_id)
        return int(clicked_id["index"])
    except Exception:
        return no_update

# Callback for opening/closing image modals
@app.callback(
    Output({"type": "modal", "index": ALL}, "is_open"),
    [Input({'type': 'thumbnail', 'index': ALL}, 'n_clicks'),
     Input({'type': 'close', 'index': ALL}, 'n_clicks')],
    [State({"type": "modal", "index": ALL}, "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n1, n2, is_open):
    # Simply toggling when a thumbnail or close button is clicked.
    return [not open_state if (n1 or n2) else open_state for open_state in is_open]

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


# -----------------------------
# Run the Dash App
# -----------------------------
if __name__ == '__main__':
    # Only run the development server when not in production
    if _debug:
        app.run_server(host='0.0.0.0', port=8050, debug=True)
