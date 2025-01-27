# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Streaming
# ------------------------------------------------------------------------------

# main.py
from flask import Flask, Response
from dotenv import load_dotenv
import json
import os
import time
import csv
import cv2
import threading
from camera.detector import Detector
import numpy as np
import logging

# Read the debug flag from the environment variable (default: False)
_debug = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if _debug else logging.INFO,  # Set level based on _debug
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log the current debug mode
logger.info(f"Debug mode is {'enabled' if _debug else 'disabled'}.")

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
load_dotenv()
model_choice = os.getenv("MODEL_CHOICE", "pytorch_ssd")
class_filter = json.loads(os.getenv("CLASS_FILTER", '["bird"]'))  # Load JSON for array
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
save_threshold = float(os.getenv("SAVE_THRESHOLD", 0.5))
save_interval = int(os.getenv("SAVE_INTERVAL", 1))  # Seconds between saving
input_fps = float(os.getenv("INPUT_FPS", 10))  # Default FPS is 10
process_time = float(os.getenv("PROCESS_TIME", 0.1))  # Average detection time per frame

logger.info(f"model_choice: {model_choice} ")
logger.info(f"class_filter: {class_filter} ")
logger.info(f"confidence_threshold: {confidence_threshold} ")
logger.info(f"save_threshold: {save_threshold} ")
logger.info(f"save_interval: {save_interval} ")
logger.info(f"input_fps: {input_fps} ")
logger.info(f"process_time: {process_time} ")

# Output resize width from environment variables with a default value
output_resize_width = int(os.getenv("STREAM_WIDTH_OUTPUT_RESIZE", 800))  # Fixed output width after Docker start
logger.info(f"Output Resize Width set to: {output_resize_width} pixels")

# Input stream dimensions initialized as None
input_stream_width = None
input_stream_height = None

output_dir = os.getenv("OUTPUT_DIR", "/output")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "all_bounding_boxes.csv")

# If the file doesn't exist yet, create it with a header row
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "class_name", "confidence", "x1", "y1", "x2", "y2"])

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
    global latest_frame, detector_instance, input_stream_width, input_stream_height

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Waiting for Stream"

    # Fixed font size adjustment variables
    font_scale = 0.7  # Suitable for output width of 1080
    font_thickness = 2  # Suitable thickness

    # Placeholder initialization flag and variables
    placeholder_initialized = False
    placeholder = None
    # Fixed output resolution (enforced for all frames)
    fixed_output_width = 640
    fixed_output_height = 360
    timestamp_y = 50  # Will be updated based on input stream height

    while True:
        try:
            # Retrieve current resolution
            with detector_lock:
                if detector_instance:
                    resolution = detector_instance.resolution  # Tuple (width, height)
                else:
                    resolution = None

            if resolution:
                current_input_width, current_input_height = resolution

                # Check if resolution has changed
                if (input_stream_width, input_stream_height) != (current_input_width, current_input_height):
                    input_stream_width, input_stream_height = current_input_width, current_input_height

                    # Calculate output_resize_height to maintain aspect ratio
                    output_resize_height = int(input_stream_height * output_resize_width / input_stream_width)

                    # Update timestamp Y-position based on input stream height
                    timestamp_y = int(input_stream_height * 0.03)  # 3% from the top

                    # Reinitialize the placeholder with new resized dimensions
                    placeholder = np.zeros((output_resize_height, output_resize_width, 3), dtype=np.uint8)

                    # Center the "Waiting for Stream" text on the placeholder
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = (output_resize_width - text_size[0]) // 2
                    text_y_center = (output_resize_height + text_size[1]) // 2
                    cv2.putText(
                        placeholder,
                        text,
                        (text_x, text_y_center),
                        font,
                        font_scale,
                        (0, 0, 255),  # Red text color
                        font_thickness
                    )
                    placeholder_initialized = True
                    logger.info("Placeholder initialized with 'Waiting for Stream' text.")
            elif not placeholder_initialized:
                # Initialize a default placeholder if resolution is not detected
                placeholder = np.zeros((fixed_output_height, fixed_output_width, 3), dtype=np.uint8)
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = (output_resize_width - text_size[0]) // 2
                text_y_center = (fixed_output_height + text_size[1]) // 2
                cv2.putText(
                    placeholder,
                    text,
                    (text_x, text_y_center),
                    font,
                    font_scale,
                    (0, 0, 255),  # Red text color
                    font_thickness
                )
                placeholder_initialized = True
                logger.info("Default placeholder initialized.")

            # Retrieve the latest frame
            with frame_lock:
                current_frame = latest_frame

            if current_frame is not None and resolution:
                # Calculate output_resize_height to maintain aspect ratio
                output_resize_height = int(input_stream_height * output_resize_width / input_stream_width)

                # Resize frame for output streaming while maintaining aspect ratio
                resized_frame = cv2.resize(current_frame, (output_resize_width, output_resize_height))

                # Add timestamp to the resized frame
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    resized_frame,
                    timestamp,
                    (10, timestamp_y),  # Fixed position relative to resized output
                    font,
                    font_scale,
                    (0, 255, 0),  # Green text color
                    font_thickness
                )

                # Encode the resized frame
                ret, buffer = cv2.imencode('.jpg', resized_frame)
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
                else:
                    logger.error("Failed to encode resized frame.")
            else:
                # Use the placeholder
                placeholder_with_time = placeholder.copy()

                # Add timestamp to the placeholder
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    placeholder_with_time,
                    timestamp,
                    (10, timestamp_y),  # Fixed position relative to resized output
                    font,
                    font_scale,
                    (0, 255, 0),  # Green text color
                    font_thickness
                )

                # Encode the placeholder frame
                ret, buffer = cv2.imencode('.jpg', placeholder_with_time)
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
                else:
                    logger.error("Failed to encode placeholder frame.")
        except Exception as e:
            logger.error(f"Error generating frames: {e}")
            time.sleep(0.1)  # Baseline sleep to prevent rapid looping on error

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

            # We now get 4 values back
            annotated_frame, should_save_interval, original_frame, detection_info_list = detector.detect_objects(
                frame,
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
                save_threshold=save_threshold
            )

            # Update the global frame for streaming
            with frame_lock:
                latest_frame = annotated_frame.copy()  # Make a writable copy

            # Save results if needed
            current_time = time.time()
            if should_save_interval and (current_time - last_save_time >= save_interval):
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

            frame_count += 1

            delay = max(1.0 / input_fps, process_time)
            time.sleep(delay)  # Frame limiter to avoid processing frames too frequently.
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
