# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Streaming
# ------------------------------------------------------------------------------

# main.py
from flask import Flask, Response
from dotenv import load_dotenv
import json
load_dotenv()
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
# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
# Load variables from environment variables or use default values
model_choice = os.getenv("MODEL_CHOICE", "pytorch_ssd")
class_filter = json.loads(os.getenv("CLASS_FILTER", '["bird"]'))  # Load JSON for array
confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
save_threshold = float(os.getenv("SAVE_THRESHOLD", 0.5))
save_interval = int(os.getenv("SAVE_INTERVAL", 1))  # Seconds between saving
input_fps = float(os.getenv("INPUT_FPS", 10))  # Default FPS is 10
process_time = float(os.getenv("PROCESS_TIME", 1))  # Average detection time per frame
stream_width = int(os.getenv("STREAM_WIDTH", 640))  # Default streaming width: 640
stream_height = int(os.getenv("STREAM_HEIGHT", 360))  # Default streaming height: 360

output_dir = os.getenv("OUTPUT_DIR", "/output")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "all_bounding_boxes.csv")

# If the file doesn't exist yet, create it with a header row
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "filename", "class_name", "confidence", "x1", "y1", "x2", "y2"])

# ------------------------------------------------------------------------------
# Flask App Setup
# ------------------------------------------------------------------------------
app = Flask(__name__)


def generate_frames():
    global latest_frame

    # Dynamically adjust font size and thickness based on input stream resolution
    input_stream_resolution = 1080  # Reference resolution for scaling (1080p)
    # Dynamic font size adjustment
    scale_factor = 1.5  # Custom scaling factor
    font_scale = (stream_height / input_stream_resolution) * scale_factor  # Scale font size based on stream height and scaler
    font_thickness = max(1, int(font_scale * 2))  # Ensure thickness is at least 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Waiting for Stream"

    # Create the placeholder frame
    placeholder = np.zeros((stream_height, stream_width, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (stream_width - text_size[0]) // 2
    text_y = (stream_height + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    # Dynamically calculate the timestamp position
    timestamp_y = int(font_scale * 50)  # Scaled Y offset for upper-left placement

    while True:
        try:
            with frame_lock:
                current_frame = latest_frame

            if current_frame is not None:
                # Resize frame for streaming
                resized_frame = cv2.resize(current_frame.copy(), (stream_width, stream_height))

                # Add timestamp to the resized frame
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    resized_frame,
                    timestamp,
                    (10, timestamp_y),  # Dynamically scaled upper-left placement
                    font,
                    font_scale,
                    (0, 255, 0),  # Green text color
                    font_thickness
                )

                # Encode the resized frame
                ret, buffer = cv2.imencode('.jpg', resized_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                raise ValueError("No live frame available, retrying...")
        except Exception as e:
            print(f"Error generating frames: {e}")
            placeholder_with_time = placeholder.copy()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                placeholder_with_time,
                timestamp,
                (10, timestamp_y),  # Dynamically scaled upper-left placement
                font,
                font_scale,
                (0, 255, 0),  # Green text color
                font_thickness
            )
            ret, buffer = cv2.imencode('.jpg', placeholder_with_time)
            if ret:
                yield (b'--frame\r\n'   
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)  # Prevents the generate_frames loop from running too fast when no new frames are available.



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
    global latest_frame

    # Load variables from .env file
    load_dotenv()
    video_source = os.getenv("VIDEO_SOURCE", 0)  # Default to 0 if VIDEO_SOURCE isn't set

    try:
        # Convert source to int if it's a webcam index, otherwise treat as a string
        source = int(video_source)
        print("Video source is a webcam.")
    except ValueError:
        source = video_source  # RTSP or HTTP stream
        print(f"Video source is a stream: {source}")

    # Initialize Detector instance
    try:
        detector = Detector(source=source, model_choice=model_choice, debug=_debug)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    print("Object detection started. Press 'Ctrl+C' to stop.")

    frame_count = 0
    last_save_time = 0

    retries = 0
    while True:
        try:
            frame = detector.get_frame()
            if frame is None:
                logger.debug("No frame available. Attempting to reinitialize detector...")
                detector.release()
                time.sleep(min(2 ** retries, 30))  # Exponential backoff with a max of 30 seconds when reinitializing the detector fails.
                retries += 1
                try:
                    detector = Detector(source=source, model_choice=model_choice, debug=_debug)
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


# Start the detection loop in a separate thread
detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

if __name__ == '__main__':
    # Start the Flask server using Waitress
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)
