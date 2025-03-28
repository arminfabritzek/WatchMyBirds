# ------------------------------------------------------------------------------
# Detection Manager Module for Object Detection
# detectors/detection_manager.py
# ------------------------------------------------------------------------------
import time
from config import load_config
config = load_config()
import re
import threading
from datetime import datetime, timedelta, timezone
import cv2
import pytz
from astral.geocoder import database, lookup
from astral.sun import sun
import piexif
import piexif.helper
from detectors.detector import Detector
from detectors.classifier import ImageClassifier
from camera.video_capture import VideoCapture
from utils.telegram_notifier import send_telegram_message
from logging_config import get_logger
import os
import csv
import json

logger = get_logger(__name__)

# --- Helper function for GPS coordinates ---
def degrees_to_dms_rational(degrees_float):
    """Converts decimal degrees to DMS rational format for EXIF."""
    is_positive = degrees_float >= 0
    degrees_float = abs(degrees_float)
    degrees = int(degrees_float)
    minutes_float = (degrees_float - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    # Use rational representation (numerator, denominator)
    # Ensure seconds are non-negative before int conversion if very close to zero
    seconds_int = max(0, int(seconds_float * 1000))
    return [(degrees, 1), (minutes, 1), (seconds_int, 1000)]

def add_exif_metadata(image_path, capture_time, location_config=None):
    """
    Adds DateTimeOriginal and optional GPS EXIF data to an image file.

    Args:
        image_path (str): Path to the saved JPEG image.
        capture_time (datetime): The datetime object representing capture time.
        location_config (dict, optional): Dict with 'latitude' and 'longitude'.
    """
    try:
        # 1. Format DateTime for EXIF
        exif_dt_str = capture_time.strftime("%Y:%m:%d %H:%M:%S")

        # 2. Prepare EXIF dictionary structure
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}

        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = exif_dt_str
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = exif_dt_str

        # 3. Add GPS Data (if available AND is a dictionary)
        # Check if location_config is a dictionary and has the required keys
        if isinstance(location_config, dict) and \
           'latitude' in location_config and \
           'longitude' in location_config:
            try:
                lat = float(location_config['latitude'])
                lon = float(location_config['longitude'])

                gps_latitude = degrees_to_dms_rational(lat)
                gps_longitude = degrees_to_dms_rational(lon)

                exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = gps_latitude
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = gps_longitude

                utc_now = datetime.now(timezone.utc)
                exif_dict["GPS"][piexif.GPSIFD.GPSDateStamp] = utc_now.strftime("%Y:%m:%d")
                exif_dict["GPS"][piexif.GPSIFD.GPSTimeStamp] = [
                    (utc_now.hour, 1), (utc_now.minute, 1), (max(0, int(utc_now.second * 1000)), 1000)
                ]
            except (ValueError, TypeError) as gps_e:
                 logger.warning(f"EXIF Warning: Could not parse GPS data from location_config: {location_config}. Error: {gps_e}")
        elif location_config is not None:
            # Log a warning if location_config exists but isn't the expected format
            logger.warning(f"EXIF Warning: location_config is not a valid dictionary with lat/lon. Skipping GPS. Value: {location_config}")


        # 4. Dump EXIF data to bytes
        exif_bytes = piexif.dump(exif_dict)

        # 5. Insert EXIF data into the image file
        piexif.insert(exif_bytes, image_path)
        logger.debug(f"Successfully added EXIF data to {os.path.basename(image_path)}")

    except FileNotFoundError:
        logger.error(f"EXIF Error: Image file not found at {image_path}")
    except Exception as e:
        # Log the specific exception and traceback
        logger.error(f"EXIF Error: Failed to add EXIF data to {os.path.basename(image_path)}. Error: {e}", exc_info=True)


class DetectionManager:
    def __init__(self):
        self.config = config
        self.model_choice = self.config["DETECTOR_MODEL_CHOICE"]
        self.video_source = self.config["VIDEO_SOURCE"]
        self.location_config = self.config.get("LOCATION_DATA")
        self.debug = self.config["DEBUG_MODE"]
        self.CLASSIFIER_IMAGE_SIZE = self.config["CLASSIFIER_IMAGE_SIZE"]

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
            capture_time_precise = datetime.now() # <<< Get precise time early

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
                timestamp_str = capture_time_precise.strftime("%Y%m%d_%H%M%S")
                # Determine the subfolder for the current day, e.g., "20250318"
                day_folder = os.path.join(self.output_dir, timestamp_str[:8])
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
                original_name = f"{timestamp_str}_{best_class_sanitized}_original.jpg"
                optimized_name = f"{timestamp_str}_{best_class_sanitized}_optimized.jpg"
                zoomed_name = f"{timestamp_str}_{best_class_sanitized}_zoomed.jpg"

                original_path = os.path.join(day_folder, original_name)
                optimized_path = os.path.join(day_folder, optimized_name)
                zoomed_path = os.path.join(day_folder, zoomed_name)

                # Generate COCO formatted detection information
                image_id = int(timestamp_str.replace('_', ''))
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
                try:
                    save_success_original = cv2.imwrite(original_path, original_frame)
                    if save_success_original:
                        add_exif_metadata(original_path, capture_time_precise, self.location_config)
                    else:
                        logger.error(f"Failed to save original image (imwrite returned false): {original_path}")
                except Exception as e:
                    logger.error(f"Error during original image saving/EXIF: {e}")
                    save_success_original = False  # Ensure flag is False on exception

                # 2. Generate an optimized version of the original image for display.
                try:
                    if original_frame.shape[1] > 800:
                        optimized_frame = cv2.resize(original_frame,
                                                     (
                                                     800, int(original_frame.shape[0] * 800 / original_frame.shape[1])))
                        save_success_optimized = cv2.imwrite(optimized_path, optimized_frame,
                                                             [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    else:
                        # No need to copy 'optimized_frame = original_frame' if just saving
                        save_success_optimized = cv2.imwrite(optimized_path, original_frame,
                                                             [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Save directly

                    if save_success_optimized:
                        add_exif_metadata(optimized_path, capture_time_precise, self.location_config)
                    else:
                        logger.error(f"Failed to save optimized image: {optimized_path}")
                except cv2.error as e:
                    logger.error(f"OpenCV error during optimized image processing/saving: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during optimized image processing/saving: {e}")

                # Generate the zoomed version based on the detection with the highest confidence using create_square_crop()
                top1_class_name = ""
                top1_confidence = ""
                if detection_info_list: # Should always be true if object_detected is true, but check anyway
                    try:
                        bbox = (best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"])
                        # Assuming create_square_crop handles potential errors and returns None or valid frame
                        zoomed_frame_raw = self.create_square_crop(original_frame, bbox, margin_percent=0.2)

                        if zoomed_frame_raw is not None and zoomed_frame_raw.size > 0:
                            zoomed_frame = cv2.resize(zoomed_frame_raw,
                                                      (self.CLASSIFIER_IMAGE_SIZE, self.CLASSIFIER_IMAGE_SIZE))

                            # Perform classification
                            try:
                                zoomed_frame_rgb = cv2.cvtColor(zoomed_frame, cv2.COLOR_BGR2RGB)
                                _, _, top1_class_name, top1_confidence = self.classifier.predict_from_image(
                                    zoomed_frame_rgb)
                            except Exception as e:
                                logger.error(f"Classification failed: {e}")
                                top1_class_name = "CLASSIFICATION_ERROR"  # Mark error in CSV
                                top1_confidence = 0.0

                            # Save the zoomed image
                            save_success_zoomed = cv2.imwrite(zoomed_path, zoomed_frame,
                                                              [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                            if save_success_zoomed:
                                add_exif_metadata(zoomed_path, capture_time_precise, self.location_config)
                                actual_zoomed_path = zoomed_path  # Store path only if save succeeded
                            else:
                                logger.error(f"Failed to save zoomed image: {zoomed_path}")
                        else:
                            logger.warning("Zoomed frame generation failed or resulted in empty image. Skipping save.")
                            zoomed_name = ""  # Indicate zoomed was not created in CSV

                    except cv2.error as e:
                        logger.error(f"OpenCV error during zoomed image processing/saving: {e}")
                        zoomed_name = ""  # Indicate zoomed was not created in CSV
                    except Exception as e:
                        logger.error(f"Unexpected error during zoomed image processing/saving: {e}")
                        zoomed_name = ""  # Indicate zoomed was not created in CSV

                logger.debug(f"Classification Result: {top1_class_name} - {top1_confidence}")  # Log even if save failed

                # Write image metadata and classification result to CSV (append mode)
                try:
                    file_exists = os.path.exists(csv_path) and os.stat(csv_path).st_size > 0
                    with open(csv_path, mode="a", newline="", encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(
                                ["timestamp", "original_name", "optimized_name", "zoomed_name", "best_class",
                                 "best_class_conf", "top1_class_name", "top1_confidence", "coco_json"])
                        # Use potentially updated zoomed_name (empty if failed)
                        writer.writerow(
                            [timestamp_str, original_name, optimized_name, zoomed_name, best_class_sanitized,
                             best_class_conf, top1_class_name, top1_confidence, coco_json])
                except IOError as e:
                    logger.error(f"Error writing to CSV {csv_path}: {e}")

                self.detection_occurred = True
                self.detection_counter += len(detection_info_list)
                self.detection_classes_agg.update(det["class_name"] for det in detection_info_list)

                # Telegram Notification
                current_time = time.time()
                cooldown = self.config.get("TELEGRAM_COOLDOWN", 60)  # Use .get() for safety

                # Check if cooldown has passed *before* acquiring lock for efficiency
                if self.detection_occurred and (current_time - self.last_notification_time >= cooldown):
                    with self.telegram_lock:
                        # Double-check condition inside lock to prevent race condition
                        if self.detection_occurred and (current_time - self.last_notification_time >= cooldown):
                            aggregated_classes = ", ".join(sorted(self.detection_classes_agg))
                            alert_text = (f"🔎 Detection Alert!\n"
                                          f"Detected: {aggregated_classes}\n"  # Simplified text slightly
                                          f"Total ({len(detection_info_list)} new / {self.detection_counter} since last alert)")

                            # Determine best photo to send, check existence
                            photo_to_send = None
                            if actual_zoomed_path and os.path.exists(actual_zoomed_path):
                                photo_to_send = actual_zoomed_path
                            elif save_success_optimized and os.path.exists(optimized_path):  # Check flag and existence
                                photo_to_send = optimized_path
                            elif save_success_original and os.path.exists(original_path):  # Check flag and existence
                                photo_to_send = original_path

                            if photo_to_send:
                                try:
                                    send_telegram_message(text=alert_text, photo_path=photo_to_send) # Actual call
                                    logger.info(
                                        f"(Simulated) Telegram notification sent: {alert_text} with {os.path.basename(photo_to_send)}")

                                    # Reset state *after* successful send attempt
                                    self.last_notification_time = current_time
                                    self.detection_occurred = False
                                    self.detection_counter = 0
                                    self.detection_classes_agg = set()

                                except Exception as e:
                                    logger.error(f"Failed to send Telegram notification: {e}")
                                    # Decide if state should be reset even on failure. Usually not.
                            else:
                                logger.warning(
                                    "No suitable image found to send with Telegram alert. Sending text only.")
                                try:
                                    send_telegram_message(text=alert_text) # Send text only
                                    logger.info(f"(Simulated) Telegram notification sent (text only): {alert_text}")
                                    # Reset state *after* successful send attempt
                                    self.last_notification_time = current_time
                                    self.detection_occurred = False
                                    self.detection_counter = 0
                                    self.detection_classes_agg = set()
                                except Exception as e:
                                    logger.error(f"Failed to send Telegram text-only notification: {e}")

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