# ------------------------------------------------------------------------------
# Main Script for Real-Time Object Detection with Webcam
# ------------------------------------------------------------------------------

import os
import time
import csv
import cv2
from camera.webcam_camera import WebcamCamera


def main():
    """
    Main function to run real-time object detection using a webcam.
    The script captures frames, performs object detection, displays annotated
    frames, and optionally saves detected objects and metadata.

    Parameters:
        None (hardcoded parameters can be modified within the script).

    Usage:
        - Press 'q' to quit the livestream.
        - Detected objects and metadata are saved to the output directory.
    """

    # --------------------------------------------------------------------------
    # Configuration Parameters
    # --------------------------------------------------------------------------
    # Hard-coded parameters
    model_choice = "pytorch_ssd"  # pytorch_ssd or efficientdet_lite4 or ssd_mobilenet_v2
    class_filter = ["bird"]
    confidence_threshold = 0.5
    save_threshold = 0.8
    save_interval = 5  # Seconds between saving

    # --------------------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------------------
    camera = WebcamCamera(model_choice=model_choice)
    output_dir = "output"
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
        frame = camera.get_frame()
        if frame is None:
            print("Kein Frame verfügbar. Überprüfe die Kamera.")
            break

        # We now get 4 values back
        annotated_frame, should_save_interval, original_frame, detection_info_list = camera.detect_objects(
            frame,
            class_filter=class_filter,
            confidence_threshold=confidence_threshold,
            save_threshold=save_threshold
        )

        # Display livestream
        cv2.imshow("Livestream with Object Detection", annotated_frame)

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

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional frame limiter
        time.sleep(0.03)

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------
    camera.release()
    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Run the Script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()