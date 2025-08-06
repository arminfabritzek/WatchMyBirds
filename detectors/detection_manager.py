# ------------------------------------------------------------------------------
# Detection Manager Module for Object Detection
# detectors/detection_manager.py
# ------------------------------------------------------------------------------

import time
from config import load_config

config = load_config()
import re
import threading
from datetime import datetime, timezone
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
import hashlib

"""
This module defines the DetectionManager class, which orchestrates video frame acquisition,
object detection, image classification, and saving of results, including EXIF metadata.
"""
logger = get_logger(__name__)


# >>> Helper Functions >>>
def degrees_to_dms_rational(degrees_float):
    """
    Converts decimal degrees to DMS rational format for EXIF.

    Args:
        degrees_float (float): Decimal degree value.

    Returns:
        list: DMS rational format [(deg,1),(min,1),(sec*1000,1000)].
    """
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
        capture_time (datetime): Datetime representing capture time.
        location_config (dict, optional): Dict with 'latitude' and 'longitude'.

    Returns:
        None
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
        if (
            isinstance(location_config, dict)
            and "latitude" in location_config
            and "longitude" in location_config
        ):
            try:
                lat = float(location_config["latitude"])
                lon = float(location_config["longitude"])
                gps_latitude = degrees_to_dms_rational(lat)
                gps_longitude = degrees_to_dms_rational(lon)
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = (
                    "N" if lat >= 0 else "S"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = gps_latitude
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = (
                    "E" if lon >= 0 else "W"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = gps_longitude
                utc_now = datetime.now(timezone.utc)
                exif_dict["GPS"][piexif.GPSIFD.GPSDateStamp] = utc_now.strftime(
                    "%Y:%m:%d"
                )
                exif_dict["GPS"][piexif.GPSIFD.GPSTimeStamp] = [
                    (utc_now.hour, 1),
                    (utc_now.minute, 1),
                    (max(0, int(utc_now.second * 1000)), 1000),
                ]
            except (ValueError, TypeError) as gps_e:
                logger.warning(
                    f"EXIF Warning: Could not parse GPS data from location_config: {location_config}. Error: {gps_e}"
                )
        elif location_config is not None:
            # Log a warning if location_config exists but isn't the expected format
            logger.warning(
                f"EXIF Warning: location_config is not a valid dictionary with lat/lon. Skipping GPS. Value: {location_config}"
            )
        # 4. Dump EXIF data to bytes
        exif_bytes = piexif.dump(exif_dict)
        # 5. Insert EXIF data into the image file
        piexif.insert(exif_bytes, image_path)
        logger.debug(f"Successfully added EXIF data to {os.path.basename(image_path)}")
    except FileNotFoundError:
        logger.error(f"EXIF Error: Image file not found at {image_path}")
    except Exception as e:
        # Log the specific exception and traceback
        logger.error(
            f"EXIF Error: Failed to add EXIF data to {os.path.basename(image_path)}. Error: {e}",
            exc_info=True,
        )


class DetectionManager:
    """
    Orchestrates the entire detection and classification pipeline.

    This class manages video frame acquisition, object detection, image classification,
    and result handling. It operates using two main threads: one for continuously
    capturing frames from a video source and another for processing these frames.

    Key Responsibilities:
    - Initializes and manages `VideoCapture`, `Detector`, and `ImageClassifier`.
    - Runs a frame acquisition loop to get the latest video frame.
    - Runs a detection loop that:
        - Performs object detection on frames.
        - Checks for daylight hours to operate.
        - Skips processing for identical consecutive frames.
        - If a detection meets the confidence threshold:
            - Saves original, optimized, and a square-cropped (zoomed) image of the detection.
            - Adds EXIF metadata (timestamp, GPS) to saved images.
            - Runs the `ImageClassifier` on the zoomed crop.
            - Records detection and classification results in a daily CSV log.
            - Sends a Telegram notification with an image, subject to a cooldown period.
    - Provides thread-safe mechanisms for accessing frames and managing components.
    - Handles graceful startup and shutdown of all components and threads.
    """

    def __init__(self):
        """
        Initializes the DetectionManager instance.

        Sets up configuration, initializes the classifier, creates thread locks for safe
        multithreaded operations, and prepares shared state variables. It also creates
        the necessary output directories and sets up the frame acquisition and detection
        threads without starting them.
        """
        self.config = config
        self.model_choice = self.config["DETECTOR_MODEL_CHOICE"]
        self.video_source = self.config["VIDEO_SOURCE"]
        self.location_config = self.config.get("LOCATION_DATA")
        self.debug = self.config["DEBUG_MODE"]
        self.SAVE_RESOLUTION_ZOOMED = (
            512  # Resolution for zoomed images, may be used for reclassification.
        )

        # Initializes the classifier.
        self.classifier = ImageClassifier()
        print("Classifier initialized.")

        # Locks for thread-safe operations.
        self.frame_lock = threading.Lock()  # Protects raw frame and timestamp.
        self.detector_lock = (
            threading.Lock()
        )  # Protects detector instance during reinitialization.
        self.telegram_lock = threading.Lock()

        # Shared state.
        self.latest_raw_frame = None  # Updated continuously by the frame updater.
        self.latest_raw_timestamp = 0  # Timestamp for the raw frame.
        self.latest_detection_time = 0
        self.previous_frame_hash = None
        self.consecutive_identical_frames = 0

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
        self.frame_thread = threading.Thread(
            target=self._frame_update_loop, daemon=True
        )
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )

    # >>> Daytime Check Function >>>
    def _is_daytime(self, city_name):
        """
        Determines if it is daytime in the given city.

        Args:
            city_name (str): Name of the city.

        Returns:
            bool: True if daytime, False otherwise.
        """
        try:
            city = lookup(city_name, database())
            tz = pytz.timezone(city.timezone)
            now = datetime.now(tz)
            dawn_depression = 12
            s = sun(
                city.observer, date=now, tzinfo=tz, dawn_dusk_depression=dawn_depression
            )
            return s["dawn"] < now < s["dusk"]
        except Exception as e:
            logger.error(f"Error determining daylight status: {e}")
            return True  # Defaults to daytime

    # >>> Initialize Components >>>
    def _initialize_components(self):
        """
        Initializes video capture and detector components with retries.

        Args:
            None

        Returns:
            None
        """
        while self.video_capture is None and not self.stop_event.is_set():
            try:
                self.video_capture = VideoCapture(self.video_source, debug=self.debug)
                self.video_capture.start()
                logger.info("VideoCapture initialized in DetectionManager.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize video capture: {e}. Retrying in 5 seconds."
                )
                time.sleep(5)

        while self.detector_instance is None and not self.stop_event.is_set():
            try:
                self.detector_instance = Detector(
                    model_choice=self.model_choice, debug=self.debug
                )
                logger.info("Detector initialized in DetectionManager.")
            except Exception as e:
                logger.error(
                    f"Failed to initialize detector: {e}. Retrying in 5 seconds."
                )
                time.sleep(5)

    # >>> Frame Update Loop >>>
    def _frame_update_loop(self):
        """
        Continuously updates the latest raw frame from VideoCapture.

        Args:
            None

        Returns:
            None
        """
        while not self.stop_event.is_set():
            # Waits until video_capture is initialized.
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
                # If no new frame for more than 5 seconds, mark it as unavailable
                if time.time() - self.latest_raw_timestamp > 5:
                    with self.frame_lock:
                        if self.latest_raw_frame is not None:
                            logger.warning(
                                "No new frames received for over 5 seconds. "
                                "Clearing latest_raw_frame to trigger placeholder display."
                            )
                        self.latest_raw_frame = None
                logger.debug("No frame available from VideoCapture in frame updater.")
                time.sleep(0.1)

    # >>> Create Square Crop Function >>>
    def create_square_crop(self, image, bbox, margin_percent=0.2, pad_color=(0, 0, 0)):
        """
        Creates a square crop centered on the object defined by bbox, adding padding if necessary
        so that the output is always a full square with the object centered.

        Args:
            image (np.ndarray): The source image.
            bbox (tuple): Bounding box as (x1, y1, x2, y2).
            margin_percent (float): Extra margin percentage to add around the bbox.
            pad_color (tuple): Color for padding (default is black).

        Returns:
            np.ndarray: The square cropped image.
        """
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        bbox_width = bx2 - bx1
        bbox_height = by2 - by1
        bbox_side = max(bbox_width, bbox_height)
        new_side = int(bbox_side * (1 + margin_percent))
        desired_x1 = int(cx - new_side / 2)
        desired_y1 = int(cy - new_side / 2)
        desired_x2 = desired_x1 + new_side
        desired_y2 = desired_y1 + new_side
        image_h, image_w = image.shape[:2]
        crop_x1 = max(0, desired_x1)
        crop_y1 = max(0, desired_y1)
        crop_x2 = min(image_w, desired_x2)
        crop_y2 = min(image_h, desired_y2)
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        pad_left = crop_x1 - desired_x1
        pad_top = crop_y1 - desired_y1
        pad_right = desired_x2 - crop_x2
        pad_bottom = desired_y2 - crop_y2
        square_crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )
        return square_crop

    # >>> Detection Loop Thread Function >>>
    def _detection_loop(self):
        """
        Continuously processes the latest frame for detection.
        This is decoupled from frame acquisition.

        Args:
            None

        Returns:
            None
        """
        self._initialize_components()
        logger.info("Detection loop (worker) started.")
        while not self.stop_event.is_set():
            raw_frame = None
            capture_time_precise = datetime.now()
            # Grabs the most recent frame.
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    raw_frame = self.latest_raw_frame.copy()
            if raw_frame is None:
                logger.debug("No raw frame available for detection. Sleeping briefly.")
                self.previous_frame_hash = None
                self.consecutive_identical_frames = 0
                time.sleep(0.1)
                continue

            # >>> Identical Frame Check >>>
            try:
                current_frame_hash = hashlib.md5(raw_frame.tobytes()).hexdigest()
                if (
                    self.previous_frame_hash is not None
                    and current_frame_hash == self.previous_frame_hash
                ):
                    self.consecutive_identical_frames += 1
                    logger.warning(
                        f"Identical frame detected. Consecutive count: {self.consecutive_identical_frames}."
                    )
                    self.previous_frame_hash = current_frame_hash
                    target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
                    time.sleep(max(0.01, target_duration))
                    continue
                else:
                    self.consecutive_identical_frames = 0
                    self.previous_frame_hash = current_frame_hash
            except Exception as hash_e:
                logger.error(f"Error during frame hashing: {hash_e}")
                self.previous_frame_hash = None
                self.consecutive_identical_frames = 0

            if not (
                self.config["DAY_AND_NIGHT_CAPTURE"]
                or self._is_daytime(self.config["DAY_AND_NIGHT_CAPTURE_LOCATION"])
            ):
                logger.info("Not enough light for detection. Sleeping for 60 seconds.")
                time.sleep(60)
                continue

            start_time = time.time()
            try:
                object_detected, original_frame, detection_info_list = (
                    self.detector_instance.detect_objects(
                        raw_frame,
                        confidence_threshold=self.config[
                            "CONFIDENCE_THRESHOLD_DETECTION"
                        ],
                        save_threshold=self.config["SAVE_THRESHOLD"],
                    )
                )
            except Exception as e:
                logger.error(
                    f"Inference error detected: {e}. Reinitializing detector..."
                )
                with self.detector_lock:
                    try:
                        self.detector_instance = Detector(
                            model_choice=self.model_choice, debug=self.debug
                        )
                    except Exception as e2:
                        logger.error(f"Detector reinitialization failed: {e2}")
                time.sleep(1)
                continue

            with self.frame_lock:
                self.latest_detection_time = time.time()

            if object_detected:
                timestamp_str = capture_time_precise.strftime("%Y%m%d_%H%M%S")
                # Determine the subfolder for the current day, e.g., "20250318"
                day_folder = os.path.join(self.output_dir, timestamp_str[:8])
                os.makedirs(day_folder, exist_ok=True)
                csv_path = os.path.join(day_folder, "images.csv")

                # Finds the best detection's class.
                best_det = max(detection_info_list, key=lambda d: d["confidence"])
                best_class = best_det[
                    "class_name"
                ]  # e.g., "Eurasian Blue Tit" --> "Eurasian_Blue_Tit"
                best_class_sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", best_class)
                best_class_conf = best_det["confidence"]

                # Includes the class name in filename
                original_name = f"{timestamp_str}_{best_class_sanitized}_original.jpg"
                optimized_name = f"{timestamp_str}_{best_class_sanitized}_optimized.jpg"
                zoomed_name = f"{timestamp_str}_{best_class_sanitized}_zoomed.jpg"

                original_path = os.path.join(day_folder, original_name)
                optimized_path = os.path.join(day_folder, optimized_name)
                zoomed_path = os.path.join(day_folder, zoomed_name)

                # Generate COCO formatted detection information.
                image_id = int(timestamp_str.replace("_", ""))
                image_info = {
                    "id": image_id,
                    "file_name": original_name,
                    "width": original_frame.shape[1],
                    "height": original_frame.shape[0],
                }
                annotations = []
                for i, det in enumerate(detection_info_list, start=1):
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height] # COCO
                    area = (x2 - x1) * (y2 - y1)
                    # Assign a dummy category_id: use 7 if the class matches best_class_sanitized, else 1.
                    category_id = (
                        7
                        if det["class_name"].replace(" ", "_") == best_class_sanitized
                        else 1
                    )
                    annotations.append(
                        {
                            "id": image_id * 100 + i,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                        }
                    )
                categories = []
                unique_categories = {}
                for det in detection_info_list:
                    cat_name = det["class_name"].replace(" ", "_")
                    if cat_name not in unique_categories:
                        cat_id = 7 if cat_name == best_class_sanitized else 1
                        unique_categories[cat_name] = cat_id
                        categories.append({"id": cat_id, "name": cat_name})
                coco_detection = {
                    "annotations": annotations,
                    "images": [image_info],
                    "categories": categories,
                }
                coco_json = json.dumps(coco_detection)

                # Saves the original full-resolution image (for download and analysis).
                try:
                    save_success_original = cv2.imwrite(original_path, original_frame)
                    if save_success_original:
                        add_exif_metadata(
                            original_path, capture_time_precise, self.location_config
                        )
                    else:
                        logger.error(
                            f"Failed to save original image (imwrite returned false): {original_path}"
                        )
                except Exception as e:
                    logger.error(f"Error during original image saving/EXIF: {e}")
                    save_success_original = False  # Ensure flag is False on exception

                # Generates an optimized version of the original image for display.
                try:
                    if original_frame.shape[1] > 800:
                        optimized_frame = cv2.resize(
                            original_frame,
                            (
                                800,
                                int(
                                    original_frame.shape[0]
                                    * 800
                                    / original_frame.shape[1]
                                ),
                            ),
                        )
                        save_success_optimized = cv2.imwrite(
                            optimized_path,
                            optimized_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                        )
                    else:
                        save_success_optimized = cv2.imwrite(
                            optimized_path,
                            original_frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                        )

                    if save_success_optimized:
                        add_exif_metadata(
                            optimized_path, capture_time_precise, self.location_config
                        )
                    else:
                        logger.error(
                            f"Failed to save optimized image: {optimized_path}"
                        )
                except cv2.error as e:
                    logger.error(
                        f"OpenCV error during optimized image processing/saving: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Unexpected error during optimized image processing/saving: {e}"
                    )

                # Generates the zoomed version based on the detection with the highest confidence using create_square_crop()
                top1_class_name = ""
                top1_confidence = ""
                if detection_info_list:
                    try:
                        bbox = (
                            best_det["x1"],
                            best_det["y1"],
                            best_det["x2"],
                            best_det["y2"],
                        )
                        zoomed_frame_raw = self.create_square_crop(
                            original_frame, bbox, margin_percent=0.1
                        )

                        if zoomed_frame_raw is not None and zoomed_frame_raw.size > 0:
                            zoomed_frame = cv2.resize(
                                zoomed_frame_raw,
                                (
                                    self.SAVE_RESOLUTION_ZOOMED,
                                    self.SAVE_RESOLUTION_ZOOMED,
                                ),
                            )

                            # Performs classification.
                            try:
                                zoomed_frame_rgb = cv2.cvtColor(
                                    zoomed_frame, cv2.COLOR_BGR2RGB
                                )
                                _, _, top1_class_name, top1_confidence = (
                                    self.classifier.predict_from_image(zoomed_frame_rgb)
                                )
                            except Exception as e:
                                logger.error(f"Classification failed: {e}")
                                top1_class_name = (
                                    "CLASSIFICATION_ERROR"  # Mark error in CSV
                                )
                                top1_confidence = 0.0

                            # Saves the zoomed image.
                            save_success_zoomed = cv2.imwrite(
                                zoomed_path,
                                zoomed_frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 70],
                            )
                            if save_success_zoomed:
                                add_exif_metadata(
                                    zoomed_path,
                                    capture_time_precise,
                                    self.location_config,
                                )
                                actual_zoomed_path = (
                                    zoomed_path  # Store path only if save succeeded
                                )
                            else:
                                logger.error(
                                    f"Failed to save zoomed image: {zoomed_path}"
                                )
                        else:
                            logger.warning(
                                "Zoomed frame generation failed or resulted in empty image. Skipping save."
                            )
                            zoomed_name = ""  # Indicate zoomed was not created in CSV

                    except cv2.error as e:
                        logger.error(
                            f"OpenCV error during zoomed image processing/saving: {e}"
                        )
                        zoomed_name = ""  # Indicate zoomed was not created in CSV
                    except Exception as e:
                        logger.error(
                            f"Unexpected error during zoomed image processing/saving: {e}"
                        )
                        zoomed_name = ""  # Indicate zoomed was not created in CSV

                logger.debug(
                    f"Classification Result: {top1_class_name} - {top1_confidence}"
                )

                # Writes image metadata and classification result to CSV (append mode).
                try:
                    file_exists = (
                        os.path.exists(csv_path) and os.stat(csv_path).st_size > 0
                    )
                    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(
                                [
                                    "timestamp",
                                    "original_name",
                                    "optimized_name",
                                    "zoomed_name",
                                    "best_class",
                                    "best_class_conf",
                                    "top1_class_name",
                                    "top1_confidence",
                                    "coco_json",
                                ]
                            )
                        # Use potentially updated zoomed_name (empty if failed)
                        writer.writerow(
                            [
                                timestamp_str,
                                original_name,
                                optimized_name,
                                zoomed_name,
                                best_class_sanitized,
                                best_class_conf,
                                top1_class_name,
                                top1_confidence,
                                coco_json,
                            ]
                        )
                except IOError as e:
                    logger.error(f"Error writing to CSV {csv_path}: {e}")

                self.detection_occurred = True
                self.detection_counter += len(detection_info_list)
                self.detection_classes_agg.update(
                    det["class_name"] for det in detection_info_list
                )

                # Telegram notification.
                current_time = time.time()
                cooldown = self.config.get(
                    "TELEGRAM_COOLDOWN", 60
                )  # Use .get() for safety

                # Checks if cooldown has passed before acquiring lock for efficiency
                if self.detection_occurred and (
                    current_time - self.last_notification_time >= cooldown
                ):
                    with self.telegram_lock:
                        # Double-checks condition inside lock to prevent race condition
                        if self.detection_occurred and (
                            current_time - self.last_notification_time >= cooldown
                        ):
                            aggregated_classes = ", ".join(
                                sorted(self.detection_classes_agg)
                            )
                            alert_text = (
                                f"🔎 Detection Alert!\n"
                                f"Detected: {aggregated_classes}\n"  # Simplified text slightly
                                f"Total ({len(detection_info_list)} new / {self.detection_counter} since last alert)"
                            )

                            # Determines best photo to send, checks existence.
                            photo_to_send = None
                            if actual_zoomed_path and os.path.exists(
                                actual_zoomed_path
                            ):
                                photo_to_send = actual_zoomed_path
                            elif save_success_optimized and os.path.exists(
                                optimized_path
                            ):  # Check flag and existence
                                photo_to_send = optimized_path
                            elif save_success_original and os.path.exists(
                                original_path
                            ):  # Check flag and existence
                                photo_to_send = original_path

                            if photo_to_send:
                                try:
                                    send_telegram_message(
                                        text=alert_text, photo_path=photo_to_send
                                    )  # Actual call
                                    logger.info(
                                        f"(Simulated) Telegram notification sent: {alert_text} with {os.path.basename(photo_to_send)}"
                                    )

                                    # Resets state after successful send attempt.
                                    self.last_notification_time = current_time
                                    self.detection_occurred = False
                                    self.detection_counter = 0
                                    self.detection_classes_agg = set()

                                except Exception as e:
                                    logger.error(
                                        f"Failed to send Telegram notification: {e}"
                                    )
                            else:
                                logger.warning(
                                    "No suitable image found to send with Telegram alert. Sending text only."
                                )
                                try:
                                    send_telegram_message(text=alert_text)
                                    logger.info(
                                        f"(Simulated) Telegram notification sent (text only): {alert_text}"
                                    )
                                    # Resets state after successful send attempt.
                                    self.last_notification_time = current_time
                                    self.detection_occurred = False
                                    self.detection_counter = 0
                                    self.detection_classes_agg = set()
                                except Exception as e:
                                    logger.error(
                                        f"Failed to send Telegram text-only notification: {e}"
                                    )

            detection_time = time.time() - start_time
            target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
            sleep_time = target_duration - detection_time
            # Ensures a minimal sleep time to reduce CPU usage even if detection takes longer than target
            if sleep_time <= 0:
                sleep_time = 0.01  # Minimal sleep duration in seconds
            logger.info(
                f"AI duration: {detection_time:.4f}s, sleeping for: {sleep_time:.4f}s"
            )
            time.sleep(sleep_time)

        logger.info("Detection loop stopped.")

    # >>> Get Display Frame Function >>>
    def get_display_frame(self):
        """
        Returns the most recent frame to be displayed.

        Args:
            None

        Returns:
            np.ndarray or None: The latest frame or None if not available.
        """
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            else:
                return None

    # >>> Start DetectionManager >>>
    def start(self):
        """
        Starts the DetectionManager by initializing components and starting threads.

        Args:
            None

        Returns:
            None
        """
        self._initialize_components()
        self.frame_thread.start()
        self.detection_thread.start()
        logger.info("DetectionManager started.")

    # >>> Stop DetectionManager >>>
    def stop(self):
        """
        Stops the DetectionManager and releases resources.

        Args:
            None

        Returns:
            None
        """
        self.stop_event.set()
        self.frame_thread.join()
        self.detection_thread.join()
        if self.video_capture:
            self.video_capture.stop()
        logger.info("DetectionManager stopped and video capture released.")
