"""
Detection Manager V2 - Service-Oriented Orchestrator.

This is a THIN WRAPPER over the existing DetectionManager that:
1. Uses the same initialization and lifecycle as the original
2. Delegates detection/classification/persistence to Services where possible
3. Maintains 100% behavioral compatibility

The goal is incremental migration, not a complete rewrite.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime

from camera.video_capture import VideoCapture
from config import get_config
from detectors.classifier import ImageClassifier
from detectors.motion_detector import MotionDetector
from detectors.services import NotificationService, PersistenceService
from detectors.services.classification_service import ClassificationService
from detectors.services.crop_service import CropService
from detectors.services.detection_service import DetectionService
from logging_config import get_logger
from utils.db import get_connection, get_or_create_default_source
from utils.path_manager import get_path_manager

logger = get_logger(__name__)
config = get_config()


class DetectionManager:
    """
    Service-oriented detection manager.

    Uses NotificationService and PersistenceService for their respective tasks,
    while maintaining the same lifecycle and threading model as the original.
    """

    def __init__(self):
        """Initialize exactly like the original DetectionManager."""
        self.config = config
        self.model_choice = self.config["DETECTOR_MODEL_CHOICE"]
        self.video_source = self.config["VIDEO_SOURCE"]
        self.location_config = self.config.get("LOCATION_DATA")
        self.exif_gps_enabled = self.config.get("EXIF_GPS_ENABLED", True)
        self.debug = self.config["DEBUG_MODE"]
        self.SAVE_RESOLUTION_CROP = 512

        # Classifier (lazy-loaded)
        self.classifier = ImageClassifier()
        # Wrap classifier with ClassificationService for clean interface
        self.classification_service = ClassificationService(self.classifier)
        self.classifier_model_id = ""

        # Load common names
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        common_names_file = os.path.join(project_root, "assets", "common_names_DE.json")
        try:
            with open(common_names_file, encoding="utf-8") as f:
                self.common_names = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load common names: {e}")
            self.common_names = {}

        # Motion Detector
        self.motion_detector = MotionDetector(
            sensitivity=self.config.get("MOTION_SENSITIVITY", 500), debug=self.debug
        )

        # Thread locks
        self.frame_lock = threading.Lock()
        self.detector_lock = threading.Lock()
        self.telegram_lock = threading.Lock()

        # Shared state
        self.latest_raw_frame = None
        self.latest_raw_timestamp = 0
        self.latest_detection_time = 0
        self.previous_frame_hash = None
        self.consecutive_identical_frames = 0

        # Statistics
        self.detection_occurred = False
        self.last_notification_time = time.time()
        self.detection_counter = 0
        self.detection_classes_agg = set()

        # Pending species buffer (for notifications)
        self.pending_species = {}
        self.pending_species_lock = threading.Lock()

        # Control flags
        self.paused = False
        self.last_detection_had_frame = True
        self._last_components_ready_state = True
        self._last_frame_was_stale = False
        self._no_frame_log_state = False
        self._inference_error_state = False

        # Components (lazy-init)
        self.video_capture = None
        # DetectionService for object detection (lazy loading)
        self.detection_service = DetectionService(
            model_choice=self.model_choice,
            debug=self.debug,
        )
        self.detector_model_id = ""

        # Queue and DB
        self.processing_queue = queue.Queue(maxsize=1)
        self.db_conn = get_connection()
        self.current_source_id = get_or_create_default_source(self.db_conn)

        # Path manager
        self.output_dir = self.config["OUTPUT_DIR"]
        self.path_mgr = get_path_manager(self.output_dir)

        # Stop event
        self.stop_event = threading.Event()

        # Daylight cache
        self._daytime_cache = {"city": None, "value": True, "ts": 0.0}
        self._daytime_ttl = 300

        # Threads
        self.frame_thread = threading.Thread(
            target=self._frame_update_loop, daemon=True
        )
        self.detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )

        # Backoff
        self.initialization_retry_count = 0

        # --- NEW: Services ---
        self.notification_service = NotificationService(common_names=self.common_names)
        self.persistence_service = PersistenceService()
        self.crop_service = CropService()

        logger.info("DetectionManager V2 initialized (with Services)")

    # =========================================================================
    # LIFECYCLE - Exact copy from original
    # =========================================================================

    def start(self):
        """Starts the DetectionManager."""
        self.stop_event.clear()
        self.frame_thread.start()
        self.detection_thread.start()
        self.processing_thread.start()
        logger.info("DetectionManager V2 started.")

    def stop(self):
        """Stops the DetectionManager."""
        self.stop_event.set()

        for thread in [
            self.frame_thread,
            self.detection_thread,
            self.processing_thread,
        ]:
            if thread.is_alive():
                thread.join(timeout=2.0)

        if self.video_capture:
            try:
                self.video_capture.stop_event.set()
                if self.video_capture.cap:
                    self.video_capture.cap.release()
            except Exception as e:
                logger.error(f"Error releasing video capture: {e}")

        logger.info("DetectionManager V2 stopped.")

    # =========================================================================
    # COMPONENT INITIALIZATION - Exact copy from original
    # =========================================================================

    def _initialize_components(self):
        """Lazy-init video capture and detector."""
        if self.stop_event.is_set():
            return False

        if self.video_capture is None:
            try:
                self.video_capture = VideoCapture(
                    self.video_source, debug=self.debug, auto_start=False
                )
                self.video_capture.start()
                logger.info("VideoCapture initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize video capture: {e}")
                self.video_capture = None

        # Detector init via DetectionService (lazy)
        if not self.detection_service.is_ready():
            if self.detection_service._ensure_initialized():
                self.detector_model_id = self.detection_service.get_model_id()
                logger.info("Detector initialized via DetectionService.")
            else:
                logger.error("Failed to initialize detector via DetectionService")

        return self.video_capture is not None and self.detection_service.is_ready()

    # =========================================================================
    # FRAME LOOP - Exact copy from original
    # =========================================================================

    def _frame_update_loop(self):
        """Continuously updates latest_raw_frame from VideoCapture."""
        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(1)
                continue

            if self.video_capture is None:
                time.sleep(0.1)
                continue

            frame = self.video_capture.get_frame()
            if frame is not None:
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                    self.latest_raw_timestamp = time.time()

                if self._no_frame_log_state:
                    logger.debug("Frames received again.")
                    self._no_frame_log_state = False
            else:
                if time.time() - self.latest_raw_timestamp > 5:
                    with self.frame_lock:
                        if self.latest_raw_frame is not None:
                            logger.info("No frames for 5s. Clearing buffer.")
                        self.latest_raw_frame = None

                    if not self._no_frame_log_state:
                        logger.warning("No frames for 5s.")
                        self._no_frame_log_state = True
                time.sleep(0.1)

    # =========================================================================
    # DETECTION LOOP - Exact copy from original
    # =========================================================================

    def _detection_loop(self):
        """Detection loop - exact behavior as original."""
        logger.info("Detection loop started.")

        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(1)
                continue

            if not self._initialize_components():
                self.initialization_retry_count += 1
                backoff_time = min(60, 2**self.initialization_retry_count)

                if self._last_components_ready_state:
                    logger.warning(
                        f"Components not ready. Retrying in {backoff_time}s..."
                    )
                    self._last_components_ready_state = False

                if self.stop_event.wait(timeout=backoff_time):
                    break
                continue

            if self.initialization_retry_count > 0:
                logger.info("Components recovered.")
                self.initialization_retry_count = 0

            self._last_components_ready_state = True

            # Get frame
            raw_frame = None
            capture_time_precise = datetime.now()

            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    raw_frame = self.latest_raw_frame.copy()

            if raw_frame is None:
                if self.last_detection_had_frame:
                    logger.debug("No frame available.")
                    self.last_detection_had_frame = False
                time.sleep(0.1)
                continue

            self.last_detection_had_frame = True

            # Motion detection gate
            if self.config.get("MOTION_DETECTION_ENABLED", True):
                if not self.motion_detector.detect(raw_frame):
                    time.sleep(0.1)
                    continue

            # Run detection via DetectionService
            start_time = time.time()

            detection_result = self.detection_service.detect(
                frame=raw_frame,
                confidence_threshold=self.config["CONFIDENCE_THRESHOLD_DETECTION"],
                save_threshold=self.config["SAVE_THRESHOLD"],
            )

            # Extract results from DetectionResult
            object_detected = detection_result.detected
            original_frame = detection_result.original_frame
            detection_info_list = detection_result.detections

            # Update model ID for persistence (lazy loaded)
            if not self.detector_model_id and detection_result.model_id:
                self.detector_model_id = detection_result.model_id

            # Handle detection failures with reinit
            if original_frame is None and not object_detected:
                if not self._inference_error_state:
                    logger.error("Inference error detected. Reinitializing detector...")
                    self._inference_error_state = True

                with self.detector_lock:
                    if not self.detection_service.reinitialize():
                        logger.debug("Detector reinitialization failed")
                    else:
                        self.detector_model_id = self.detection_service.get_model_id()
                time.sleep(1)
                continue

            if self._inference_error_state:
                logger.info("Inference recovered.")
                self._inference_error_state = False

            with self.frame_lock:
                self.latest_detection_time = time.time()

            detection_time = time.time() - start_time
            target_duration = 1.0 / self.config["MAX_FPS_DETECTION"]
            sleep_time = max(0.01, target_duration - detection_time)

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
            else:
                logger.info(
                    f"[DET] {int(detection_time * 1000)}ms | sleep {int(sleep_time * 1000)}ms"
                )

            time.sleep(sleep_time)

        logger.info("Detection loop stopped.")

    def _enqueue_processing_job(self, job):
        """Enqueue job, drop oldest if full."""
        try:
            self.processing_queue.put_nowait(job)
        except queue.Full:
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                pass
            self.processing_queue.put_nowait(job)

    # =========================================================================
    # PROCESSING LOOP - Uses Services
    # =========================================================================

    def _processing_loop(self):
        """Process detections using Services."""
        logger.info("Processing loop started.")

        while not self.stop_event.is_set():
            try:
                job = self.processing_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            capture_time = job["capture_time_precise"]
            original_frame = job["original_frame"]
            detection_info_list = job["detection_info_list"]
            detection_time_ms = job["detection_time_ms"]

            cls_start = time.time()

            # --- Use PersistenceService for image saving ---
            try:
                img_result = self.persistence_service.save_image(
                    frame=original_frame,
                    capture_time=capture_time,
                    detector_model_id=self.detector_model_id,
                    classifier_model_id=self.classifier_model_id,
                    source_id=self.current_source_id,
                    location_config=self.location_config,
                    exif_gps_enabled=self.exif_gps_enabled,
                )

                if not img_result.success:
                    logger.error("Failed to save image")
                    continue

                base_filename = img_result.base_filename

            except Exception as e:
                logger.error(f"PersistenceService.save_image error: {e}")
                continue

            # --- Process each detection ---
            best_species = None
            best_score = 0.0
            best_thumb_path = None

            from detectors.interfaces.persistence import DetectionData

            for idx, det in enumerate(detection_info_list, start=1):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                od_conf = det["confidence"]
                bbox_tuple = (x1, y1, x2, y2)

                # Create crop for classification via CropService
                crop_rgb = self.crop_service.create_classification_crop(
                    frame=original_frame,
                    bbox=bbox_tuple,
                    size=self.SAVE_RESOLUTION_CROP,
                    margin_percent=0.1,
                    to_rgb=True,
                )
                if crop_rgb is None:
                    continue

                # Classify via ClassificationService
                cls_name = ""
                cls_conf = 0.0
                try:
                    if crop_rgb is not None:
                        cls_result = self.classification_service.classify(crop_rgb)
                        cls_name = cls_result.class_name
                        cls_conf = cls_result.confidence
                        if not self.classifier_model_id:
                            self.classifier_model_id = cls_result.model_id or ""
                except Exception as e:
                    logger.error(f"Classification error: {e}")

                # Calculate score
                if cls_conf > 0:
                    score = 0.5 * od_conf + 0.5 * cls_conf
                    agreement = min(od_conf, cls_conf)
                else:
                    score = od_conf
                    agreement = od_conf

                # Save detection via PersistenceService
                det_data = DetectionData(
                    bbox=(x1, y1, x2, y2),
                    confidence=od_conf,
                    class_name=det.get("class_name", "bird"),
                    cls_class_name=cls_name,
                    cls_confidence=cls_conf,
                    score=score,
                    agreement_score=agreement,
                )

                try:
                    det_result = self.persistence_service.save_detection(
                        image_filename=base_filename,
                        detection=det_data,
                        frame=original_frame,
                        detector_model_id=self.detector_model_id,
                        classifier_model_id=self.classifier_model_id,
                        crop_index=idx,
                    )

                    # Track best for notification
                    if score > best_score:
                        best_score = score
                        best_species = cls_name or "Unknown"
                        best_thumb_path = (
                            str(det_result.thumbnail_path)
                            if det_result.thumbnail_path
                            else None
                        )

                except Exception as e:
                    logger.error(f"PersistenceService.save_detection error: {e}")

            # --- Notification via NotificationService ---
            if best_species and best_thumb_path:
                species_info = self.notification_service.create_species_info(
                    latin_name=best_species,
                    score=best_score,
                    image_path=best_thumb_path,
                )
                self.notification_service.queue_detection(species_info)

                if self.notification_service.should_send():
                    self.notification_service.send_summary()

            # Log timing
            cls_duration_ms = int((time.time() - cls_start) * 1000)
            logger.info(
                f"[DET+CLS] Total={detection_time_ms + cls_duration_ms}ms "
                f"(DET={detection_time_ms}ms, CLS={cls_duration_ms}ms) | "
                f"Objects={len(detection_info_list)} | sleep {job['sleep_time_ms']}ms"
            )

        logger.info("Processing loop stopped.")

    # =========================================================================
    # PUBLIC INTERFACE - Same as original
    # =========================================================================

    def get_display_frame(self):
        """Returns the most recent frame for display."""
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return None

    def update_source(self, new_source):
        """Updates video source at runtime."""
        logger.info(f"Updating video source to: {new_source}")
        self.video_source = new_source
        self.config["VIDEO_SOURCE"] = new_source

        if self.motion_detector:
            self.motion_detector.reset()

        with self.frame_lock:
            self.latest_raw_frame = None
            self.latest_raw_timestamp = 0

        if self.video_capture:
            try:
                self.video_capture.stop_event.set()
                if self.video_capture.cap:
                    self.video_capture.cap.release()
            except Exception as e:
                logger.error(f"Error stopping video capture: {e}")
            self.video_capture = None

    def update_configuration(self, changes: dict):
        """Handles runtime config changes."""
        if "VIDEO_SOURCE" in changes:
            self.update_source(changes["VIDEO_SOURCE"])

        if "DEBUG_MODE" in changes:
            self.debug = changes["DEBUG_MODE"]
            self.config["DEBUG_MODE"] = self.debug

        if "LOCATION_DATA" in changes:
            self.location_config = changes["LOCATION_DATA"]
            self.config["LOCATION_DATA"] = self.location_config

        if "EXIF_GPS_ENABLED" in changes:
            self.exif_gps_enabled = changes["EXIF_GPS_ENABLED"]
            self.config["EXIF_GPS_ENABLED"] = self.exif_gps_enabled

    def start_user_ingest(self, folder_path=None):
        """Orchestrates User Ingest process."""
        from utils.db import get_or_create_user_import_source
        from utils.ingest import ingest_folder

        if folder_path is None:
            folder_path = self.config.get("INGEST_DIR", "")

        try:
            logger.info("Initiating User Ingest. Pausing detection...")
            self.paused = True
            time.sleep(2)

            source_id = get_or_create_user_import_source(self.db_conn)

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
