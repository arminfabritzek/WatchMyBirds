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
from camera.webcam_camera import WebcamCamera
import numpy as np

# Global lock for thread-safe frame access
frame_lock = threading.Lock()

app = Flask(__name__)
latest_frame = None  # Global variable to store the latest frame for streaming


def generate_frames():
    global latest_frame
    # Create simple "No Signal" placeholder
    width, height = 640, 480
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Waiting for Images"
    text_size = cv2.getTextSize(text, font, 2, 3)[0]
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
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            placeholder_with_time = placeholder.copy()
            cv2.putText(placeholder_with_time, timestamp, (10, 30), font, 1, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder_with_time)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        dynamic_sleep = 0.1 if current_frame is not None else 1.0
        time.sleep(dynamic_sleep)




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
    """
    Main function to run real-time object detection using a webcam.
    """
    global latest_frame

    # Load variables from .env file
    load_dotenv()
    video_source = os.getenv("VIDEO_SOURCE", 0)  # Default to 0 if VIDEO_SOURCE isn't set

    # Convert webcam_source to int if applicable
    try:
        source = int(video_source)
        backend = None  # Use default backend for webcams
        print("Video source is a WebCam.")
    except ValueError:
        source = video_source  # Use RTSP URL if webcam_source is not a valid integer
        # backend = cv2.CAP_GSTREAMER  # Specify GSTREAMER backend for RTSP
        backend = cv2.CAP_FFMPEG  # Specify GSTREAMER backend for RTSP
        print(f"Video source is RTSP Stream: {source}")

    # --------------------------------------------------------------------------
    # Configuration Parameters
    # --------------------------------------------------------------------------
    use_threaded = True  # Set to False to use non-threaded video capture
    model_choice = "pytorch_ssd"  # pytorch_ssd or efficientdet_lite4 or ssd_mobilenet_v2
    class_filter = ["bird"]
    confidence_threshold = 0.3
    save_threshold = 0.6
    save_interval = 3  # Seconds between saving

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------
    camera = WebcamCamera(source=source, backend=backend, model_choice=model_choice, use_threaded=use_threaded)

    # Get the output directory from the environment variable, with a default fallback
    output_dir = os.getenv("OUTPUT_DIR", "output")
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    last_save_time = 0

    # We'll store bounding boxes in "all_bounding_boxes.csv"
    csv_path = os.path.join(output_dir, "all_bounding_boxes.csv")

    # If the file doesn't exist yet, create it with a header row
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "filename", "class_name", "confidence", "x1", "y1", "x2", "y2"])

    print("Drücke 'q', um den Livestream zu beenden.")

    # --------------------------------------------------------------------------
    # Livestream and Object Detection Loop
    # --------------------------------------------------------------------------
    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                print("No frame available. Using placeholder.")
                with frame_lock:
                    latest_frame = None

                print("Attempting to reinitialize camera.")
                camera.release()

                # Reinitialize the camera
                camera = WebcamCamera(source=source, backend=backend, model_choice=model_choice, use_threaded=use_threaded)
                time.sleep(2)
                continue

            # We now get 4 values back
            annotated_frame, should_save_interval, original_frame, detection_info_list = camera.detect_objects(
                frame,
                class_filter=class_filter,
                confidence_threshold=confidence_threshold,
                save_threshold=save_threshold
            )

            # Update the latest frame for streaming
            with frame_lock:
                latest_frame = annotated_frame

            current_time = time.time()
            if should_save_interval and (current_time - last_save_time >= save_interval):
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                # 1) Save annotated frame
                annotated_name = f"{timestamp}_frame_annotated.jpg"
                annotated_path = os.path.join(output_dir, annotated_name)
                cv2.imwrite(annotated_path, annotated_frame)

                # 2) Save unannotated frame
                unannotated_name = f"{timestamp}_frame_original.jpg"
                original_path = os.path.join(output_dir, unannotated_name)
                cv2.imwrite(original_path, original_frame)

                print(f"Annotated frame saved: {annotated_path}")
                print(f"Original frame saved: {original_path}")

                # 3) Append bounding-box info to CSV (for the *annotated* image)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    for det in detection_info_list:
                        writer.writerow([
                            timestamp,
                            annotated_name,  # We link bounding boxes to the annotated file
                            det["class_name"],
                            f"{det['confidence']:.2f}",
                            det["x1"],
                            det["y1"],
                            det["x2"],
                            det["y2"]
                        ])

                last_save_time = current_time

            frame_count += 1

            # Optional frame limiter
            time.sleep(0.03)
        except Exception as e:
            print(f"Error in detection loop: {e}")
            time.sleep(2)  # Prevent busy looping during errors
# Start the detection loop in a separate thread
detection_thread = threading.Thread(target=detection_loop)
detection_thread.daemon = True
detection_thread.start()

'''
if __name__ == '__main__':

    # Start the Flask server
    app.run(host='0.0.0.0', port=5001, debug=True)
'''
if __name__ == '__main__':
    # Start the Flask server using Waitress
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001)
