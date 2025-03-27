# ------------------------------------------------------------------------------
# Detection Manager Module for Object Detection
# detectors/detection_manager.py
# ------------------------------------------------------------------------------
import time
from config import load_config
config = load_config()
import re
import threading
from datetime import datetime, timedelta
import cv2
import pytz
from astral.geocoder import database, lookup
from astral.sun import sun
from detectors.detector import Detector
from detectors.classifier import ImageClassifier
from camera.video_capture import VideoCapture
from utils.telegram_notifier import send_telegram_message
from logging_config import get_logger
import os
import csv
import json

logger = get_logger(__name__)

class DetectionManager:
    def __init__(self):
        self.config = config
        self.model_choice = self.config["DETECTOR_MODEL_CHOICE"]
        self.video_source = self.config["VIDEO_SOURCE"]
        self.debug = self.config["DEBUG_MODE"]

        # Initialize the classifier.
        self.classifier = ImageClassifier()
        print("Classifier initialized.")

        # Locks for thread-safe operations.
        self.frame_lock = threading.Lock()      # Protects raw frame and timestamp.
        self.detector_lock = threading.Lock()   # Protects detector reinitialization.
        self.telegram_lock = threading.Lock()

        # Shared state.
        self.latest_raw_frame = None       # Updated continuously by the frame updater.
        self.latest_raw_timestamp = 0      # Timestamp for the raw frame.
        self.latest_detection_time = 0

        # Statistics and notifications.
        self.detection_occurred = False
        self.last_notification_time = time.time()
        self.detection_counter = 0
        self.detection_classes_agg = set()

        # Video capture and detector instances.
        self.video_capture = None
        self.detector_instance = None

        # Set up output directory
        self.output_dir = self.config["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)

        # For clean shutdown.
        self.stop_event = threading.Event()

        # Two threads: one for frame acquisition and one for detection.
        self.frame_thread = threading.Thread(target=self._frame_update_loop, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)

    def _is_daytime(self, city_name):
        try:
            city = lookup(city_name, database())
            tz = pytz.timezone(city.timezone)
            now = datetime.now(tz)
            dawn_depression = 12
            s = sun(city.observer, date=now, tzinfo=tz, dawn_dusk_depression=dawn_depression)
            return s["dawn"] < now < s["dusk"]
        except Exception as e:
            logger.error(f"Error determining daylight status: {e}")
            return True  # Default to daytime

    def _initialize_components(self):
        # Initialize video capture.
        while self.video_capture is None and not self.stop_event.is_set():
            try:
                self.video_capture = VideoCapture(self.video_source, debug=self.debug)
                logger.info("VideoCapture initialized in DetectionManager.")
            except Exception as e:
                logger.error(f"Failed to initialize video capture: {e}. Retrying in 5 seconds.")
                time.sleep(5)

        # Initialize detector.
        while self.detector_instance is None and not self.stop_event.is_set():
            try:
                self.detector_instance = Detector(model_choice=self.model_choice, debug=self.debug)
                logger.info("Detector initialized in DetectionManager.")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}. Retrying in 5 seconds.")
                time.sleep(5)

    def _frame_update_loop(self):
        """Continuously updates the latest raw frame from VideoCapture."""
        while not self.stop_event.is_set():
            # Wait until video_capture is initialized.
            if self.video_capture is None:
                logger.debug("Video capture not initialized yet. Waiting...")
                time.sleep(0.1)
                continue

            frame = self.video_capture.get_frame()
            if frame is not None:
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                    self.latest_raw_timestamp = time.time()
            else:
                logger.debug("No frame available from VideoCapture in frame updater.")
                time.sleep(0.1)

    def create_square_crop(self, image, bbox, margin_percent=0.2, pad_color=(0, 0, 0)):
        """
        Creates a square crop centered on the object defined by bbox, adding padding if necessary
        so that the output is always a full square with the object centered.

        Parameters:
            image (np.ndarray): The source image.
            bbox (tuple): Bounding box as (x1, y1, x2, y2).
            margin_percent (float): Extra margin percentage to add around the bbox.
            pad_color (tuple): Color for padding (default is black).

        Returns:
            np.ndarray: The square cropped image.
        """
        bx1, by1, bx2, by2 = bbox
        # Compute the center of the bounding box.
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2

        # Compute width and height of the bbox, and choose the larger side.
        bbox_width = bx2 - bx1
        bbox_height = by2 - by1
        bbox_side = max(bbox_width, bbox_height)

        # Apply margin to get the desired square side length.
        new_side = int(bbox_side * (1 + margin_percent))

        # Desired square coordinates centered on the bbox center.
        desired_x1 = int(cx - new_side / 2)
        desired_y1 = int(cy - new_side / 2)
        desired_x2 = desired_x1 + new_side
        desired_y2 = desired_y1 + new_side

        # Get original image dimensions.
        image_h, image_w = image.shape[:2]

        # Determine the part of the desired square that lies within the image.
        crop_x1 = max(0, desired_x1)
        crop_y1 = max(0, desired_y1)
        crop_x2 = min(image_w, desired_x2)
        crop_y2 = min(image_h, desired_y2)

        # Extract that region from the image.
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Calculate the padding needed on each side.
        pad_left = crop_x1 - desired_x1
        pad_top = crop_y1 - desired_y1
        pad_right = desired_x2 - crop_x2
        pad_bottom = desired_y2 - crop_y2

        # Use cv2.copyMakeBorder to add padding and form a full square.
        square_crop = cv2.copyMakeBorder(crop,
                                         pad_top,
                                         pad_bottom,
                                         pad_left,
                                         pad_right,
                                         borderType=cv2.BORDER_CONSTANT,
                                         value=pad_color)

        return square_crop

    def _detection_loop(self):
        """
        Continuously processes the latest frame for detection.
        This is decoupled from frame acquisition.
        """
        self._initialize_components()
        logger.info("Detection loop (worker) started.")
        while not self.stop_event.is_set():
            raw_frame = None
            # Grab the most recent frame.
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    raw_frame = self.latest_raw_frame.copy()
            if raw_frame is None:
                logger.debug("No raw frame available for detection. Sleeping briefly.")
                time.sleep(0.1)
                continue

            # Run detection if either DAY_AND_NIGHT_CAPTURE is True or it's daytime.
            if not (self.config["DAY_AND_NIGHT_CAPTURE"] or
                    self._is_daytime(self.config["DAY_AND_NIGHT_CAPTURE_LOCATION"])):
                logger.info("Not enough light for detection. Sleeping for 60 seconds.")
                time.sleep(60)
                continue

            # Run detection.
            start_time = time.time()
            try:
                object_detected, original_frame, detection_info_list = \
                    self.detector_instance.detect_objects(
                        raw_frame,
                        confidence_threshold=self.config["CONFIDENCE_THRESHOLD_DETECTION"],
                        save_threshold=self.config["SAVE_THRESHOLD"]
                    )
            except Exception as e:
                logger.error(f"Inference error detected: {e}. Reinitializing detector...")
                with self.detector_lock:
                    try:
                        self.detector_instance = Detector(model_choice=self.model_choice, debug=self.debug)
                    except Exception as e2:
                        logger.error(f"Detector reinitialization failed: {e2}")
                time.sleep(1)
                continue

            # Update the last frame time.
            with self.frame_lock:
                self.latest_detection_time = time.time()

            if object_detected:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Determine the subfolder for the current day, e.g., "20250318"
                day_folder = os.path.join(self.output_dir, timestamp[:8])
                os.makedirs(day_folder, exist_ok=True)
                csv_path = os.path.join(day_folder, "images.csv")

                # ------------------------------------------
                # 1) Figure out the best detection's class
                # ------------------------------------------
                best_det = max(detection_info_list, key=lambda d: d["confidence"])
                best_class = best_det["class_name"]  # e.g., "Eurasian Blue Tit" --> "Eurasian_Blue_Tit"
                best_class_sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', best_class)
                best_class_conf = best_det["confidence"]

                # ------------------------------------------
                # 2) Include the class name in your filenames
                # ------------------------------------------
                original_name = f"{timestamp}_{best_class_sanitized}_original.jpg"
                optimized_name = f"{timestamp}_{best_class_sanitized}_optimized.jpg"
                zoomed_name = f"{timestamp}_{best_class_sanitized}_zoomed.jpg"

                # Generate COCO formatted detection information
                image_id = int(timestamp.replace('_', ''))
                image_info = {
                    "id": image_id,
                    "file_name": original_name,
                    "width": original_frame.shape[1],
                    "height": original_frame.shape[0]
                }
                annotations = []
                for i, det in enumerate(detection_info_list, start=1):
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # COCO uses [x, y, width, height]
                    area = (x2 - x1) * (y2 - y1)
                    # Assign a dummy category_id: use 7 if the class matches best_class_sanitized, else 1
                    category_id = 7 if det["class_name"].replace(' ', '_') == best_class_sanitized else 1
                    annotations.append({
                        "id": image_id * 100 + i,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                categories = []
                unique_categories = {}
                for det in detection_info_list:
                    cat_name = det["class_name"].replace(' ', '_')
                    if cat_name not in unique_categories:
                        cat_id = 7 if cat_name == best_class_sanitized else 1
                        unique_categories[cat_name] = cat_id
                        categories.append({
                            "id": cat_id,
                            "name": cat_name
                        })
                coco_detection = {
                    "annotations": annotations,
                    "images": [image_info],
                    "categories": categories
                }
                coco_json = json.dumps(coco_detection)

                # 1. Save the original full-resolution image (for download).
                cv2.imwrite(os.path.join(day_folder, original_name), original_frame)

                # 2. Generate an optimized version of the original image for display.
                if original_frame.shape[1] > 800:
                    optimized = cv2.resize(original_frame,
                                           (800, int(original_frame.shape[0] * 800 / original_frame.shape[1])))
                    cv2.imwrite(os.path.join(day_folder, optimized_name), optimized,[int(cv2.IMWRITE_JPEG_QUALITY), 70])
                else:
                    cv2.imwrite(os.path.join(day_folder, optimized_name), original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                # Generate the zoomed version based on the detection with the highest confidence using create_square_crop()
                if detection_info_list:
                    # Extract bounding box from the best detection
                    bbox = (best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"])
                    zoomed_frame = self.create_square_crop(original_frame, bbox, margin_percent=0.2)

                    # Ensure the zoomed frame is resized to 224x224, if needed
                    zoomed_frame = cv2.resize(zoomed_frame, (224, 224))

                    # Perform classification directly on the zoomed_frame
                    zoomed_frame_rgb = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)
                    top_k_indices, top_k_confidences, top1_class_name, top1_confidence = self.classifier.predict_from_image(zoomed_frame_rgb)

                    # Save the zoomed image with classification result for download/viewing.
                    cv2.imwrite(os.path.join(day_folder, zoomed_name), zoomed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                else:
                    # If no detection exists, save the original frame and leave classification empty.
                    cv2.imwrite(os.path.join(day_folder, zoomed_name), original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    top1_class_name = ""
                    top1_confidence = ""

                logger.debug(f"Classification Restuls: {top1_class_name} - {top1_confidence}")

                # Write image metadata and classification result to CSV (append mode)
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if os.stat(csv_path).st_size == 0:
                        writer.writerow(["timestamp", "original_name", "optimized_name", "zoomed_name", "best_class", "best_class_conf", "top1_class_name", "top1_confidence", "coco_json"])
                    writer.writerow([timestamp, original_name, optimized_name, zoomed_name, best_class_sanitized, best_class_conf, top1_class_name, top1_confidence, coco_json])

                self.detection_occurred = True
                self.detection_counter += len(detection_info_list)
                self.detection_classes_agg.update(det["class_name"] for det in detection_info_list)

                current_time = time.time()
                with self.telegram_lock:
                    if self.detection_occurred and (current_time - self.last_notification_time >= self.config["TELEGRAM_COOLDOWN"]):
                        aggregated_classes = ", ".join(sorted(self.detection_classes_agg))
                        alert_text = (f"🔎 Detection Alert!\n"
                                      f"Detected classes: {aggregated_classes}\n"
                                      f"Total detections since last alert: {self.detection_counter}")
                        send_telegram_message(
                            text=alert_text,
                            photo_path=os.path.join(day_folder, zoomed_name)
                        )
                        logger.info(f"Telegram notification sent: {alert_text}")
                        self.last_notification_time = current_time
                        self.detection_occurred = False
                        self.detection_counter = 0
                        self.detection_classes_agg = set()
                    else:
                        logger.debug("Cooldown active, not sending alert.")

            detection_time = time.time() - start_time
            logger.info(f"Detection took {detection_time:.4f} seconds.")

            target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
            sleep_time = target_duration - detection_time
            print(f"Detection duration: {detection_time:.4f}s, sleeping for: {sleep_time:.4f}s")
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Detection loop stopped.")

    def get_display_frame(self):
        """
        Returns the most recent frame to be displayed.
        """
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            else:
                return None

    def start(self):
        self._initialize_components()  # Ensure video_capture and detector are initialized first.
        self.frame_thread.start()
        self.detection_thread.start()
        logger.info("DetectionManager started.")

    def stop(self):
        self.stop_event.set()
        self.frame_thread.join()
        self.detection_thread.join()
        if self.video_capture:
            self.video_capture.release()
        logger.info("DetectionManager stopped and video capture released.")