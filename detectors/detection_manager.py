import threading
import time
from datetime import datetime, timedelta
import cv2
import pytz
from astral.geocoder import database, lookup
from astral.sun import sun
from detectors.detector import Detector
from camera.video_capture import VideoCapture
from utils.telegram_notifier import send_telegram_message
from logging_config import get_logger
import os
import csv

logger = get_logger(__name__)

class DetectionManager:
    def __init__(self, video_source, model_choice, config, debug=False):
        self.video_source = video_source
        self.model_choice = model_choice
        self.config = config
        self.debug = debug

        # Locks for thread-safe operations.
        self.frame_lock = threading.Lock()
        self.video_capture_lock = threading.Lock()
        self.detector_lock = threading.Lock()
        self.telegram_lock = threading.Lock()

        # Shared state.
        self.latest_frame = None
        self.latest_frame_timestamp = 0
        self.detection_occurred = False
        self.last_notification_time = time.time()
        self.detection_counter = 0
        self.detection_classes_agg = set()

        # Video capture and detector instances.
        self.video_capture = None
        self.detector_instance = None

        # Set up output directory and CSV file.
        self.output_dir = self.config["OUTPUT_DIR"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "all_bounding_boxes.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "filename", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        # For clean shutdown.
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)

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

    def _detection_loop(self):
        self._initialize_components()
        logger.info("Detection loop started.")
        while not self.stop_event.is_set():
            try:
                with self.video_capture_lock:
                    frame = self.video_capture.get_frame()
                if frame is None:
                    logger.debug("No frame available from VideoCapture. Pausing detection for 5 seconds.")
                    time.sleep(5)
                    continue

                # Update the latest frame.
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_timestamp = time.time()

                # Run detection if either DAY_AND_NIGHT_CAPTURE is True or it's daytime.
                if not (self.config["DAY_AND_NIGHT_CAPTURE"] or
                        self._is_daytime(self.config["DAY_AND_NIGHT_CAPTURE_LOCATION"])):
                    logger.info("Not enough light for detection. Sleeping for 60 seconds.")
                    time.sleep(60)
                    continue

                # Run detection.
                start_time = time.time()
                try:
                    annotated_frame, object_detected, original_frame, detection_info_list = \
                        self.detector_instance.detect_objects(
                            frame,
                            confidence_threshold=self.config["CONFIDENCE_THRESHOLD"],
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

                detection_time = time.time() - start_time
                logger.debug(f"Detection took {detection_time:.4f} seconds.")

                # Update latest frame with annotations.
                with self.frame_lock:
                    self.latest_frame = annotated_frame.copy()
                    self.latest_frame_timestamp = time.time()

                current_time = time.time()
                if object_detected:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    # Filenames for the three versions:
                    original_name = f"{timestamp}_frame_original.jpg"
                    annotated_name = f"{timestamp}_frame_annotated.jpg"
                    zoomed_name = f"{timestamp}_frame_zoomed.jpg"

                    # 1. Save the original full-resolution image (for download).
                    cv2.imwrite(os.path.join(self.output_dir, original_name), original_frame)

                    # 2. Generate an optimized version of annotated image for display.
                    if annotated_frame.shape[1] > 800:
                        optimized = cv2.resize(annotated_frame,
                                               (800, int(annotated_frame.shape[0] * 800 / annotated_frame.shape[1])))
                        cv2.imwrite(os.path.join(self.output_dir, annotated_name), optimized,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    else:
                        cv2.imwrite(os.path.join(self.output_dir, annotated_name), annotated_frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                    # Generate the zoomed version based on the detection with the highest confidence.
                    if detection_info_list:
                        best_det = max(detection_info_list, key=lambda d: d["confidence"])
                        x1, y1, x2, y2 = best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"]
                        margin = 100
                        h, w = annotated_frame.shape[:2]
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w, x2 + margin)
                        y2 = min(h, y2 + margin)
                        zoomed_frame = annotated_frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(self.output_dir, zoomed_name), zoomed_frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    else:
                        cv2.imwrite(os.path.join(self.output_dir, zoomed_name), annotated_frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 70])

                    with open(self.csv_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        for det in detection_info_list:
                            writer.writerow([
                                timestamp, original_name, det["class_name"],
                                f"{det['confidence']:.2f}",
                                det["x1"], det["y1"], det["x2"], det["y2"]
                            ])

                    self.detection_occurred = True
                    self.detection_counter += len(detection_info_list)
                    self.detection_classes_agg.update(det["class_name"] for det in detection_info_list)

                    with self.telegram_lock:
                        if self.detection_occurred and (current_time - self.last_notification_time >= self.config["TELEGRAM_COOLDOWN"]):
                            aggregated_classes = ", ".join(sorted(self.detection_classes_agg))
                            alert_text = (f"🔎 Detection Alert!\n"
                                          f"Detected classes: {aggregated_classes}\n"
                                          f"Total detections since last alert: {self.detection_counter}")
                            send_telegram_message(
                                text=alert_text,
                                photo_path=os.path.join(self.output_dir, zoomed_name)
                            )
                            logger.info(f"Telegram notification sent: {alert_text}")
                            self.last_notification_time = current_time
                            self.detection_occurred = False
                            self.detection_counter = 0
                            self.detection_classes_agg = set()
                        else:
                            logger.debug("Cooldown active, not sending alert.")

                target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
                sleep_time = target_duration - detection_time
                print(f"Detection duration: {detection_time:.4f}s, sleeping for: {sleep_time:.4f}s")
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                with self.detector_lock:
                    try:
                        self.detector_instance = Detector(model_choice=self.model_choice, debug=self.debug)
                    except Exception as e2:
                        logger.error(f"Detector reinitialization failed: {e2}")
                time.sleep(1)
        logger.info("Detection loop stopped.")

    def start(self):
        self.thread.start()
        logger.info("DetectionManager started.")

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        with self.video_capture_lock:
            if self.video_capture:
                self.video_capture.release()
        logger.info("DetectionManager stopped and video capture released.")