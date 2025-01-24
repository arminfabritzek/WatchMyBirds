# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam and Flask Streaming
# ------------------------------------------------------------------------------
# main.py
from flask import Flask, Response
from dotenv import load_dotenv
import os
import time
import csv
import cv2
import threading
from camera.detector import Detector
import numpy as np

# Global lock for thread-safe frame access
frame_lock = threading.Lock()
latest_frame = None  # Global variable to store the latest frame for streaming

# --------------------------------------------------------------------------
# Configuration Parameters
# --------------------------------------------------------------------------
model_choice = "pytorch_ssd"  # pytorch_ssd or efficientdet_lite4 or ssd_mobilenet_v2
class_filter = ["bird"]
confidence_threshold = 0.3
save_threshold = 0.6
save_interval = 3  # Seconds between saving
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
    width, height = 1920, 1080
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Waiting for Stream"
    text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), font, 2, (0, 0, 255), 3)

    while True:
        try:
            with frame_lock:
                current_frame = latest_frame

            if current_frame is not None:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                frame_with_time = current_frame.copy()
                cv2.putText(frame_with_time, timestamp, (10, 30), font, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame_with_time)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                raise ValueError("No live frame available")
        except Exception as e:
            print(f"Error generating frames: {e}")
            placeholder_with_time = placeholder.copy()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(placeholder_with_time, timestamp, (10, 30), font, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder_with_time)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)



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
        detector = Detector(source=source, model_choice=model_choice)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    print("Object detection started. Press 'Ctrl+C' to stop.")

    frame_count = 0
    last_save_time = 0

    while True:
        try:
            frame = detector.get_frame()
            if frame is None:
                print("No frame available. Attempting to reinitialize detector...")
                detector.release()
                time.sleep(2)  # Short delay before retry
                detector = Detector(source=source, model_choice=model_choice)
                continue

            # We now get 4 values back
            annotated_frame, should_save_interval, original_frame, detection_info_list = detector.detect_objects(
                frame,
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
                save_threshold=save_threshold
            )

            # Update the global frame for streaming
            with frame_lock:
                latest_frame = annotated_frame

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

            time.sleep(0.03)  # Optional frame limiter
        except Exception as e:
            print(f"Error in detection loop: {e}")
            time.sleep(1)


# Start the detection loop in a separate thread
detection_thread = threading.Thread(target=detection_loop, daemon=True)
detection_thread.start()

if __name__ == '__main__':
    # Start the Flask server using Waitress
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)
