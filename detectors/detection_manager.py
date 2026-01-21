# ------------------------------------------------------------------------------
# Detection Manager Module for Object Detection
# detectors/detection_manager.py
# ------------------------------------------------------------------------------

import time
from config import get_config

config = get_config()
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
import json
import hashlib
import queue
from utils.db import get_connection, insert_image, insert_detection, insert_classification, get_or_create_default_source
from utils.image_ops import create_square_crop
from utils.path_manager import get_path_manager

"""
This module defines the DetectionManager class, which orchestrates video frame acquisition,
object detection, image classification, and saving of results, including EXIF metadata.
"""
logger = get_logger(__name__)


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
            - Saves original, optimized, and a square-cropped (crop) image of the detection.
            - Adds EXIF metadata (timestamp, GPS) to saved images.
            - Runs the `ImageClassifier` on the detection crop.
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
        self.SAVE_RESOLUTION_CROP = (
            512  # Resolution for crop images, may be used for reclassification.
        )

        # Initializes the classifier.
        self.classifier = ImageClassifier()
        # self.classifier_model_id will be populated lazily during processing loop
        self.classifier_model_id = "" 
        logger.debug("Classifier created (model will be lazy-loaded on first use).")
        
        # Load common names for Telegram notifications
        common_names_file = os.path.join(os.getcwd(), "assets", "common_names_DE.json")
        try:
            with open(common_names_file, "r", encoding="utf-8") as f:
                self.common_names = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load common names: {e}")
            self.common_names = {}

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
        
        # Buffer for collecting species during cooldown
        # Dict: species_latin -> {"common": str, "score": float, "image_path": str}
        self.pending_species = {}
        self.pending_species_lock = threading.Lock()
        
        # Resource control flags
        self.paused = False
        self.last_detection_had_frame = True
        self._last_components_ready_state = True
        self._last_frame_was_stale = False
        self._no_frame_log_state = False
        self._inference_error_state = False

        # Video capture and detector instances.
        self.video_capture = None
        self.detector_instance = None
        self.detector_model_id = ""
        self.processing_queue = queue.Queue(maxsize=1)
        self.db_conn = get_connection()
        self.current_source_id = get_or_create_default_source(self.db_conn)

        # Set up output directory via PathManager
        self.output_dir = self.config["OUTPUT_DIR"]
        self.path_mgr = get_path_manager(self.output_dir)
        # Ensure base structure exists (though path_mgr does it lazily usually, strict init is good)
        # os.makedirs(self.output_dir, exist_ok=True) -> Handled by ensures

        # For clean shutdown.
        self.stop_event = threading.Event()

        # Daylight cache
        self._daytime_cache = {
            "city": None,
            "value": True,
            "ts": 0.0,
        }
        self._daytime_ttl = 300  # seconds

        # Two threads: one for frame acquisition and one for detection.
        self.frame_thread = threading.Thread(
            target=self._frame_update_loop, daemon=True
        )
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )

    def _is_daytime(self, city_name):
        """Determines daylight using Astral with a cached result (sun-position basis)."""
        try:
            now_ts = time.time()
            cache = self._daytime_cache
            if (
                cache["city"] == city_name
                and (now_ts - cache["ts"]) < self._daytime_ttl
            ):
                return cache["value"]

            city = lookup(city_name, database())
            tz = pytz.timezone(city.timezone)
            now = datetime.now(tz)
            dawn_depression = 12
            s = sun(
                city.observer, date=now, tzinfo=tz, dawn_dusk_depression=dawn_depression
            )
            value = s["dawn"] < now < s["dusk"]
            cache.update({"city": city_name, "value": value, "ts": now_ts})
            return value
        except Exception as e:
            logger.error(f"Error determining daylight status: {e}")
            return True  # Defaults to daytime (per existing behavior)

    def _initialize_components(self):
        """
        Initializes video capture and detector components without blocking startup.
        Safe to call repeatedly; returns True when both components are ready.
        """
        if self.stop_event.is_set():
            return False

        if self.video_capture is None:
            try:
                self.video_capture = VideoCapture(
                    self.video_source, debug=self.debug, auto_start=False
                )
                self.video_capture.start()
                logger.info("VideoCapture initialized in DetectionManager.")
            except Exception as e:
                logger.error(f"Failed to initialize video capture: {e}")
                self.video_capture = None

        if self.detector_instance is None:
            try:
                self.detector_instance = Detector(
                    model_choice=self.model_choice, debug=self.debug
                )
                model_id = getattr(self.detector_instance, "model_id", "") or ""
                if not model_id and hasattr(self.detector_instance, "model_path"):
                    model_id = os.path.basename(self.detector_instance.model_path)
                self.detector_model_id = model_id
                logger.info("Detector initialized in DetectionManager.")
            except Exception as e:
                logger.error(f"Failed to initialize detector: {e}")
                self.detector_instance = None

        return self.video_capture is not None and self.detector_instance is not None

    def _frame_update_loop(self):
        """
        Continuously updates the latest raw frame from VideoCapture.

        Args:
            None

        Returns:
            None
        """
        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(1)
                continue

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
                
                # Recovery Check
                if self._no_frame_log_state:
                    logger.debug("Frames received again (stream recovered).")
                    self._no_frame_log_state = False
            else:
                # If no new frame for more than 5 seconds, mark it as unavailable
                if time.time() - self.latest_raw_timestamp > 5:
                    with self.frame_lock:
                        if self.latest_raw_frame is not None:
                            logger.info("No new frames received for over 5 seconds. Clearing buffer.")
                        self.latest_raw_frame = None
                    
                    if not self._no_frame_log_state:
                        logger.warning(
                            "No new frames received for over 5 seconds. "
                            "Clearing latest_raw_frame to trigger placeholder display."
                        )
                        self._no_frame_log_state = True
                time.sleep(0.1)



    def _detection_loop(self):
        """
        Continuously processes the latest frame for detection.
        This is decoupled from frame acquisition.

        Args:
            None

        Returns:
            None
        """
        logger.info("Detection loop (worker) started.")
        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(1)
                continue

            if not self._initialize_components():
                if self._last_components_ready_state:
                    logger.debug("Components not ready yet (Entering wait state).")
                    self._last_components_ready_state = False
                time.sleep(1)
                continue
            
            self._last_components_ready_state = True
            raw_frame = None
            capture_time_precise = datetime.now()
            # Grabs the most recent frame.
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    raw_frame = self.latest_raw_frame.copy()
            if raw_frame is None:
                if self.last_detection_had_frame:
                    logger.debug("No raw frame available for detection (Entering wait state).")
                    self.last_detection_had_frame = False
                self.previous_frame_hash = None
                self.consecutive_identical_frames = 0
                time.sleep(0.1)
                continue
            
            self.last_detection_had_frame = True

            try:
                current_frame_hash = hashlib.md5(raw_frame.tobytes()).hexdigest()
                if (
                    self.previous_frame_hash is not None
                    and current_frame_hash == self.previous_frame_hash
                ):
                    self.consecutive_identical_frames += 1
                    if self.consecutive_identical_frames == 30: # ~3 seconds at 10fps
                        logger.warning(
                            f"Identical frames detected (30 consecutive). Processing might be stale."
                        )
                        self._last_frame_was_stale = True
                    self.previous_frame_hash = current_frame_hash
                    target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
                    time.sleep(max(0.01, target_duration))
                    continue
                else:
                    if self._last_frame_was_stale:
                        logger.debug("Stream active (new frames detected).")
                        self._last_frame_was_stale = False
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
                # State-Change Logging for Inference Errors
                if not self._inference_error_state:
                    logger.error(
                        f"Inference error detected: {e}. Reinitializing detector..."
                    )
                    self._inference_error_state = True
                
                with self.detector_lock:
                    try:
                        self.detector_instance = Detector(
                            model_choice=self.model_choice, debug=self.debug
                        )
                        model_id = getattr(self.detector_instance, "model_id", "") or ""
                        if not model_id and hasattr(self.detector_instance, "model_path"):
                            model_id = os.path.basename(self.detector_instance.model_path)
                        self.detector_model_id = model_id
                    except Exception as e2:
                        # Re-init errors might be frequent, but if the main loop error is once-per-state, 
                        # we can log this as separate error or rely on the main flag. 
                        # Let's log it only if it's a DIFFERENT error or we want detailed debug.
                        # For policy compliance, we stick to the main flag or Debug.
                        logger.debug(f"Detector reinitialization failed: {e2}")
                time.sleep(1)
                continue

            # Recovery Log
            if self._inference_error_state:
                logger.info("Inference recovered (object detection working again).")
                self._inference_error_state = False

            with self.frame_lock:
                self.latest_detection_time = time.time()

            detection_time = time.time() - start_time

            target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
            sleep_time = target_duration - detection_time
            # Ensures a minimal sleep time to reduce CPU usage even if detection takes longer than target
            if sleep_time <= 0:
                sleep_time = 0.01  # Minimal sleep duration in seconds

            if object_detected:
                self._enqueue_processing_job(
                    {
                        "capture_time_precise": capture_time_precise,
                        "original_frame": original_frame,
                        "detection_info_list": detection_info_list,
                        "detection_time_ms": int(detection_time * 1000),
                        "sleep_time_ms": int(sleep_time * 1000),
                    }
                )
            
            # Only log [DET] when no object was detected
            # When object IS detected, [DET+CLS] will be logged from processing loop
            # To be precise, we want to see [DET] duration even if object is found?
            # User wants: "ob es nur eine det durchgeführt wurde und die zeit dafür. wenn eine det+cls gemacht wurde das gleiche."
            
            # If object detected -> Processing loop handles logging final stats.
            # But the detection loop finishes here.
            # Let's log [DET] here IF no object.
            # If object, let's include the DET time in the job, and log the full [DET+CLS] later.
            
            if not object_detected:
                logger.info(
                    f"[DET] {int(detection_time * 1000)}ms | sleep {int(sleep_time * 1000)}ms"
                )
            time.sleep(sleep_time)

        logger.info("Detection loop stopped.")

    def _enqueue_processing_job(self, job):
        """Enqueues a processing job, dropping the oldest if the queue is full."""
        try:
            self.processing_queue.put_nowait(job)
        except queue.Full:
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.processing_queue.put_nowait(job)
            except queue.Full:
                logger.warning("Processing queue full; dropped latest job.")

    def _processing_loop(self):
        """Handles I/O heavy work off the detection loop."""
        logger.info("Processing loop (worker) started.")
        while not self.stop_event.is_set():
            try:
                job = self.processing_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            capture_time_precise = job.get("capture_time_precise")
            original_frame = job.get("original_frame")
            detection_info_list = job.get("detection_info_list", [])
            detection_time_ms = job.get("detection_time_ms", 0)
            sleep_time_ms = job.get("sleep_time_ms", 0)

            if original_frame is None or not detection_info_list:
                continue

            # Stable Filename Generation
            # Format: YYYYMMDD_HHMMSS_ffffff.jpg (microseconds for uniqueness)
            timestamp_str = capture_time_precise.strftime("%Y%m%d_%H%M%S_%f")
            base_filename = f"{timestamp_str}.jpg"
            date_str = capture_time_precise.strftime("%Y-%m-%d")

            # Ensure Directories
            self.path_mgr.ensure_date_structure(date_str)
            
            # Paths
            original_path = self.path_mgr.get_original_path(base_filename)
            optimized_path = self.path_mgr.get_derivative_path(base_filename, "optimized")

            # --- Save Original & Optimized ---
            try:
                # 1. Save Original
                if cv2.imwrite(str(original_path), original_frame):
                    add_exif_metadata(str(original_path), capture_time_precise, self.location_config)
                else:
                    logger.error(f"Failed to save original image: {original_path}")
                    continue # abort if original fails

                # 2. Save Optimized (WebP, resized if huge)
                # Using WebP for derivatives as per plan implication (or user request for optimized)
                # Plan said: "Differences ... Dateiformat".
                # path_mgr.get_derivative_path returns .webp extension by default for 'optimized'.
                # Resize logic:
                if original_frame.shape[1] > 1920:
                    scale = 1920 / original_frame.shape[1]
                    new_h = int(original_frame.shape[0] * scale)
                    optimized_frame = cv2.resize(original_frame, (1920, new_h))
                else:
                    optimized_frame = original_frame
                
                # Save as WebP
                cv2.imwrite(str(optimized_path), optimized_frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                # EXIF is lost in WebP usually or harder to handle, skipping for optimized/thumbs logic for now
                # (Original has it, that's what counts)

            except Exception as e:
                logger.error(f"Error saving images: {e}")
                continue

            # --- DB INSERT: Image Record ---
            image_persisted = False
            try:
                insert_image(
                    self.db_conn,
                    {
                        "filename": base_filename,
                        "timestamp": timestamp_str, # keeping precise ts
                        "coco_json": None, # Dropped legacy COCO
                        "downloaded_timestamp": "",
                        "detector_model_id": self.detector_model_id,
                        "classifier_model_id": getattr(self.classifier, "model_id", "") or self.classifier_model_id,
                        "source_id": self.current_source_id,
                        "content_hash": None, # Could calculate if needed
                    },
                )
                image_persisted = True
            except Exception as e:
                logger.error(f"Error persisting image record: {e}")
            
            if not image_persisted:
                continue

            # --- Process Detections ---
            img_h, img_w = original_frame.shape[:2]
            created_at_iso = datetime.now(timezone.utc).isoformat()
            
            enriched_detections = []
            
            # Detection Loop & Classification
            cls_start_time = time.time()
            for i, det in enumerate(detection_info_list, start=1):
                # Classification logic
                cls_name = ""
                cls_conf = 0.0
                
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                bbox_tuple = (x1, y1, x2, y2)

                try:
                    crop = create_square_crop(original_frame, bbox_tuple, margin_percent=0.1)
                    if crop is not None and crop.size > 0:
                         crop_rgb = cv2.cvtColor(cv2.resize(crop, (self.SAVE_RESOLUTION_CROP, self.SAVE_RESOLUTION_CROP)), cv2.COLOR_BGR2RGB)
                         _, _, cls_name, cls_conf = self.classifier.predict_from_image(crop_rgb)
                except Exception as e:
                    logger.error(f"Error classifying detection: {e}")

                # --- Generate Thumbnail (Deterministic Name) ---
                # Naming Convention: {basename_no_ext}_crop_{i}.webp
                # path_mgr doesn't officially support 'custom' suffixes easily in get_derivative_path without modification
                # BUT, I can construct the filename manually and ask path_mgr for the dir
                
                base_name_no_ext = os.path.splitext(base_filename)[0]
                thumb_filename = f"{base_name_no_ext}_crop_{i}.webp"
                
                # We need the dir. path_mgr.get_derivative_path uses filename to find date folder.
                # So we can pass the thumb_filename to it!
                # get_derivative_path extracts date from 'YYYYMMDD_...' part. 
                # Our thumb_filename starts with base_filename which has the date. Perfect.
                thumb_path = self.path_mgr.get_derivative_path(thumb_filename, "thumb")
                
                try:
                    # Crop Logic for Thumbnail (Edge-Shifted Square)
                    TARGET_THUMB_SIZE = 256
                    EXPANSION_PERCENT = 0.10
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    side = int(max(bbox_w, bbox_h) * (1 + EXPANSION_PERCENT))
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    sq_x1, sq_y1 = int(cx - side/2), int(cy - side/2)
                    sq_x2, sq_y2 = sq_x1 + side, sq_y1 + side
                    
                    # Clamp/Shift
                    if sq_x1 < 0: sq_x2 -= sq_x1; sq_x1 = 0
                    if sq_y1 < 0: sq_y2 -= sq_y1; sq_y1 = 0
                    if sq_x2 > img_w: sq_x1 -= (sq_x2 - img_w); sq_x2 = img_w
                    if sq_y2 > img_h: sq_y1 -= (sq_y2 - img_h); sq_y2 = img_h
                    
                    if sq_x2 > sq_x1 and sq_y2 > sq_y1:
                        thumb_crop = original_frame[sq_y1:sq_y2, sq_x1:sq_x2]
                        if thumb_crop.size > 0:
                            thumb_img = cv2.resize(thumb_crop, (TARGET_THUMB_SIZE, TARGET_THUMB_SIZE), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(str(thumb_path), thumb_img, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                except Exception as e:
                    logger.error(f"Error generating thumbnail {thumb_filename}: {e}")

                # Scores
                od_conf = det["confidence"]
                if cls_conf > 0:
                    score = 0.5 * od_conf + 0.5 * cls_conf
                    agreement = min(od_conf, cls_conf)
                else:
                    score = od_conf
                    agreement = od_conf
                
                enriched_detections.append({
                    "det": det,
                    "cls_name": cls_name,
                    "cls_conf": cls_conf,
                    "score": score,
                    "agreement": agreement,
                    "thumb_filename": thumb_filename
                })
            
            cls_duration_ms = int((time.time() - cls_start_time) * 1000)

            # --- DB INSERT: Detections ---
            for item in enriched_detections:
                det = item["det"]
                try:
                    det_id = insert_detection(
                        self.db_conn,
                        {
                            "image_filename": base_filename, # FK
                            "bbox_x": det["x1"] / img_w,
                            "bbox_y": det["y1"] / img_h,
                            "bbox_w": (det["x2"] - det["x1"]) / img_w,
                            "bbox_h": (det["y2"] - det["y1"]) / img_h,
                            "od_class_name": det["class_name"],
                            "od_confidence": det["confidence"],
                            "od_model_id": self.detector_model_id,
                            "created_at": created_at_iso,
                            "score": item["score"],
                            "agreement_score": item["agreement"],
                            "detector_model_name": self.config["DETECTOR_MODEL_CHOICE"],
                            "detector_model_version": self.detector_model_id,
                            "classifier_model_name": "classifier",
                            "classifier_model_version": self.classifier_model_id,
                            "thumbnail_path": item["thumb_filename"],
                        }
                    )
                    
                    if item["cls_name"]:
                         insert_classification(
                            self.db_conn,
                            {
                                "detection_id": det_id,
                                "cls_class_name": item["cls_name"],
                                "cls_confidence": item["cls_conf"],
                                "cls_model_id": self.classifier_model_id,
                                "created_at": created_at_iso,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error inserting detection: {e}")

            # Telegram Notification with collection during cooldown
            self.detection_occurred = True
            current_time = time.time()
            cooldown = self.config.get("TELEGRAM_COOLDOWN", 60)
            
            # Collect species info for this detection
            if self.config.get("TELEGRAM_ENABLED", True):
                best_det = enriched_detections[0]
                species_latin = best_det.get('cls_name') or 'Unknown'
                species_common = self.common_names.get(species_latin, species_latin.replace('_', ' '))
                score = best_det.get('score', 0.0)
                
                # Use thumbnail (crop) instead of full image
                thumb_filename = best_det.get('thumb_filename')
                if thumb_filename:
                    thumb_path = self.path_mgr.get_derivative_path(thumb_filename, "thumb")
                    image_for_telegram = str(thumb_path)
                else:
                    image_for_telegram = str(optimized_path)
                
                with self.pending_species_lock:
                    # Only update if this detection has a higher score for this species
                    if species_latin not in self.pending_species or score > self.pending_species[species_latin]['score']:
                        self.pending_species[species_latin] = {
                            'common': species_common,
                            'score': score,
                            'image_path': image_for_telegram
                        }
            
            # Check if cooldown has expired - send summary
            if (current_time - self.last_notification_time >= cooldown) and self.config.get("TELEGRAM_ENABLED", True):
                with self.telegram_lock:
                    # Double check locking
                    if (current_time - self.last_notification_time >= cooldown):
                        try:
                            with self.pending_species_lock:
                                if self.pending_species:
                                    # Build summary message - unified format for 1 or more species
                                    species_count = len(self.pending_species)
                                    
                                    sorted_species = sorted(
                                        self.pending_species.items(),
                                        key=lambda x: x[1]['score'],
                                        reverse=True
                                    )
                                    
                                    # Format: "🐦 X Art(en) erkannt:" + list with common + latin names
                                    art_text = "Art" if species_count == 1 else "Arten"
                                    species_lines = []
                                    for species_latin, info in sorted_species:
                                        latin_formatted = species_latin.replace('_', ' ')
                                        species_lines.append(f"• {info['common']} ({latin_formatted})")
                                    
                                    message = f"🐦 {species_count} {art_text} erkannt:\n" + "\n".join(species_lines)
                                    
                                    # Use image of highest scoring species
                                    image_path = sorted_species[0][1]['image_path']
                                    
                                    send_telegram_message(
                                        text=message,
                                        photo_path=image_path
                                    )
                                    
                                    # Clear pending buffer
                                    self.pending_species.clear()
                                    
                            self.last_notification_time = current_time
                        except Exception as e:
                            logger.error(f"Telegram failed: {e}")
            
            # LOGGING: [DET+CLS]
            # DET time is from detection loop. CLS time is measured above.
            total_pipeline_ms = detection_time_ms + cls_duration_ms
            logger.debug(f"[DET+CLS] Total={total_pipeline_ms}ms (DET={detection_time_ms}ms, CLS={cls_duration_ms}ms) | Objects={len(detection_info_list)} | sleep {sleep_time_ms}ms")

        logger.info("Processing loop stopped.")

    def start_user_ingest(self, folder_path="/ingest"):
        """
        Orchestrates the User Ingest process:
        1. Parse Ingest Source (User Import).
        2. Pauses live stream detection (exclusively).
        3. Runs Ingest with 'move_files' semantics.
        4. Resumes live stream.
        """
        try:
            logger.info("Initiating User Ingest. Pausing detection...")
            self.paused = True
            
            # Wait briefly for loops to sleep
            time.sleep(2) 
            
            # 1. Get Source ID
            from utils.db import get_or_create_user_import_source
            source_id = get_or_create_user_import_source(self.db_conn)
            
            # 2. Run Ingest
            from utils.ingest import ingest_folder
            if os.path.exists(folder_path):
                logger.info(f"Running ingest on {folder_path}...")
                ingest_folder(folder_path, source_id, move_files=True)
                logger.info("User Ingest complete.")
            else:
                logger.error(f"Ingest folder not found: {folder_path}")
                
        except Exception as e:
            logger.error(f"Error during User Ingest: {e}", exc_info=True)
        finally:
            logger.info("Resuming detection...")
            self.paused = False

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

    def start(self):
        """
        Starts the DetectionManager by initializing components and starting threads.

        Args:
            None

        Returns:
            None
        """
        self.frame_thread.start()
        self.detection_thread.start()
        self.processing_thread.start()
        logger.info("DetectionManager started.")

    def stop(self):
        """
        Stops the DetectionManager and releases resources.

        Args:
            None

        Returns:
            None
        """
        self.stop_event.set()
        if hasattr(self, "frame_thread") and self.frame_thread.is_alive():
            self.frame_thread.join()
        if hasattr(self, "detection_thread") and self.detection_thread.is_alive():
            self.detection_thread.join()
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            self.processing_thread.join()
        if self.video_capture:
            self.video_capture.stop()
        try:
            if self.db_conn:
                self.db_conn.close()
        except Exception:
            pass
        logger.info("DetectionManager stopped and video capture released.")
