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
from collections import deque
from datetime import datetime

from camera.video_capture import VideoCapture
from config import get_config
from detectors.classifier import ImageClassifier
from detectors.interfaces.classification import DecisionState
from detectors.motion_detector import MotionDetector
from detectors.services import NotificationService, PersistenceService
from detectors.services.capability_registry import build_default_registry
from detectors.services.classification_service import ClassificationService
from detectors.services.crop_service import CropService
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.detection_service import DetectionService
from detectors.od_classes import is_bird_od_class
from detectors.services.scoring_pipeline import ScoringResult, compute_detection_signals
from detectors.services.temporal_decision_service import TemporalDecisionService
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

        # Detection / classification timing summary (logged periodically
        # instead of per-frame). Both lists are appended from different
        # threads (detect-loop / processing-loop); list.append() is
        # atomic under CPython's GIL so no lock is needed.
        self._det_times: list[int] = []
        self._cls_times: list[int] = []
        self._det_summary_interval = 15  # seconds
        self._det_summary_last = time.monotonic()

        # Statistics
        self.detection_occurred = False
        self.last_notification_time = time.time()
        self.detection_counter = 0
        self.detection_classes_agg = set()

        # Decision state session counters (P1-03 observability)
        self.decision_state_counts: dict[str, int] = {
            "confirmed": 0,
            "uncertain": 0,
            "unknown": 0,
            "rejected": 0,
        }

        # Burst-cap state (Filter B): timestamps of admitted detections
        # within the rolling window. Uses monotonic clock so wall-clock
        # adjustments don't move the window.
        #
        # The cap and window values themselves are read live from
        # self.config in _burst_admit() so Web-UI changes take effect on
        # the next detection — same live-reload semantics as
        # SAVE_THRESHOLD. The deque has no maxlen because the cap can
        # change at runtime; _burst_admit() trims the left end on every
        # call so memory stays bounded by the active cap.
        self._burst_timestamps: deque[float] = deque()
        self._burst_skipped_total = 0
        self._burst_skipped_last_log = time.monotonic()

        # Pending species buffer (for notifications)
        self.pending_species = {}
        self.pending_species_lock = threading.Lock()

        # Control flags
        self.paused = False
        self._deep_scan_active = False
        self._deep_scan_gate_count = 0
        self._paused_before_deep_scan = False
        self._deep_scan_lock = threading.Lock()
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

        # Service layer components
        self.notification_service = NotificationService(common_names=self.common_names)
        self.persistence_service = PersistenceService()
        self.crop_service = CropService()
        self.decision_policy_service = DecisionPolicyService()
        self.temporal_decision_service = TemporalDecisionService()
        self.capability_registry = build_default_registry()

        logger.info("DetectionManager V2 initialized (with Services)")

    # =========================================================================
    # SIGNAL DELEGATE — single-entry scoring pipeline for external callers
    # (e.g. analysis_service) without requiring cross-layer imports.
    # =========================================================================

    def compute_detection_signals(
        self,
        *,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
        od_conf: float,
        cls_conf: float,
        top_k_confidences: list[float] | None,
        species_key: str,
    ) -> "ScoringResult":
        """Delegate to :func:`scoring_pipeline.compute_detection_signals`."""
        return compute_detection_signals(
            bbox=bbox,
            frame_shape=frame_shape,
            od_conf=od_conf,
            cls_conf=cls_conf,
            top_k_confidences=top_k_confidences,
            decision_policy=self.decision_policy_service,
            temporal_service=self.temporal_decision_service,
            capability_registry=self.capability_registry,
            species_key=species_key,
        )

    def run_exhaustive_scan(self, frame):
        """Compatibility adapter for orphan deep-scan workflows."""
        detections = self.detection_service.exhaustive_detect(frame)
        if not self.detector_model_id:
            self.detector_model_id = self.detection_service.get_model_id()
        return detections

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self):
        """Starts the DetectionManager."""
        self.stop_event.clear()
        self.frame_thread.start()
        self.detection_thread.start()
        self.processing_thread.start()
        # Refresh HF model registries in the background so the Settings
        # UI reflects the current HF state (hf_latest_advertised,
        # hf_known_ids) without waiting for the first bird-detection
        # event to lazy-load the classifier. Non-blocking: failures are
        # logged but do not abort startup (guarded by its own thread).
        self._refresh_hf_registries_async()
        logger.info("DetectionManager V2 started.")

    def _refresh_hf_registries_async(self) -> None:
        """Fire-and-forget HF-registry refresh for both detector and
        classifier. Merges remote view into local latest_models.json so
        the UI ``Latest`` badge follows HF's current advertised latest —
        without downloading weights. The detector is typically refreshed
        anyway when its ONNX is loaded at service init; this call makes
        the classifier side consistent at boot instead of on-first-bird.
        """
        import threading

        def _refresh() -> None:
            try:
                from detectors.classifier import HF_BASE_URL as CLS_HF_BASE_URL
                from detectors.detector import HF_BASE_URL as DET_HF_BASE_URL
                from utils.model_downloader import fetch_latest_json

                cfg_base = self.config.get("MODEL_BASE_PATH", "models")
                det_dir = os.path.join(cfg_base, "object_detection")
                cls_dir = os.path.join(cfg_base, "classifier")

                # Detector side — refresh registry snapshot even though
                # the loader will re-fetch at first use. Cheap, idempotent.
                try:
                    fetch_latest_json(DET_HF_BASE_URL, det_dir)
                except Exception as exc:
                    logger.debug(f"HF detector registry refresh skipped: {exc}")

                # Classifier side — the actual reason this method exists.
                try:
                    fetch_latest_json(CLS_HF_BASE_URL, cls_dir)
                    logger.info("HF classifier registry refreshed at boot")
                except Exception as exc:
                    logger.debug(f"HF classifier registry refresh skipped: {exc}")
            except Exception as exc:
                logger.debug(f"HF registry refresh thread failed: {exc}")

        threading.Thread(
            target=_refresh,
            name="hf-registry-refresh",
            daemon=True,
        ).start()

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

    def enter_deep_scan_mode(self):
        """Pause live loops while a manual/nightly deep scan is running."""
        with self._deep_scan_lock:
            if self._deep_scan_gate_count == 0:
                self._paused_before_deep_scan = self.paused
                self.paused = True
                self._deep_scan_active = True
            self._deep_scan_gate_count += 1

    def exit_deep_scan_mode(self):
        """Restore live loops after deep scan completes."""
        with self._deep_scan_lock:
            if self._deep_scan_gate_count <= 0:
                self._deep_scan_gate_count = 0
                self._deep_scan_active = False
                return

            self._deep_scan_gate_count -= 1
            if self._deep_scan_gate_count == 0:
                self._deep_scan_active = False
                self.paused = self._paused_before_deep_scan

    def is_deep_scan_active(self) -> bool:
        """Whether deep-scan gating is currently active."""
        with self._deep_scan_lock:
            return self._deep_scan_active

    # =========================================================================
    # COMPONENT INITIALIZATION
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
    # FRAME LOOP
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
    # DETECTION LOOP
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

            # Detection floor is owned by the model (model_metadata.json
            # drives self._detector.conf_threshold_default). Save-threshold
            # is operator policy and may be auto-derived or manually set —
            # see config.effective_save_threshold() + SAVE_THRESHOLD_MODE.
            from config import effective_save_threshold

            detector_obj = getattr(self.detection_service, "_detector", None)
            underlying = getattr(detector_obj, "model", None) if detector_obj else None
            detector_conf = (
                getattr(underlying, "conf_threshold_default", None)
                if underlying is not None
                else None
            )
            save_thr = effective_save_threshold(self.config, detector_conf)
            detection_result = self.detection_service.detect(
                frame=raw_frame,
                save_threshold=save_thr,
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

            det_ms = int(detection_time * 1000)

            if object_detected:
                self._enqueue_processing_job(
                    {
                        "capture_time_precise": capture_time_precise,
                        "original_frame": original_frame,
                        "detection_info_list": detection_info_list,
                        "detection_time_ms": det_ms,
                        "sleep_time_ms": int(sleep_time * 1000),
                    }
                )

            # Collect timing and log a periodic summary
            self._det_times.append(det_ms)
            now_mono = time.monotonic()
            if now_mono - self._det_summary_last >= self._det_summary_interval:
                window_s = int(now_mono - self._det_summary_last)
                n_det = len(self._det_times)
                det_avg = sum(self._det_times) // n_det
                det_lo = min(self._det_times)
                det_hi = max(self._det_times)
                # Snapshot CLS samples (different thread writes them).
                # Slice-copy so we don't race with a concurrent append;
                # CLS may have zero samples in this window if no frames
                # had detections (Processing loop only fires on detects).
                cls_snapshot = self._cls_times[:]
                if cls_snapshot:
                    n_cls = len(cls_snapshot)
                    cls_avg = sum(cls_snapshot) // n_cls
                    cls_lo = min(cls_snapshot)
                    cls_hi = max(cls_snapshot)
                    logger.info(
                        "[DET+CLS] %ds summary: %d frames | "
                        "DET avg %dms (min %dms / max %dms) | "
                        "CLS %d samples avg %dms (min %dms / max %dms)",
                        window_s, n_det,
                        det_avg, det_lo, det_hi,
                        n_cls, cls_avg, cls_lo, cls_hi,
                    )
                else:
                    logger.info(
                        "[DET] %ds summary: %d frames | "
                        "avg %dms | min %dms | max %dms | CLS no samples",
                        window_s, n_det, det_avg, det_lo, det_hi,
                    )
                self._det_times.clear()
                self._cls_times.clear()
                self._det_summary_last = now_mono

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

    def _burst_admit(self) -> bool:
        """Sliding-window burst-cap gate (Filter B).

        Returns True and records the admission timestamp when the rolling
        window has capacity. Returns False (and increments a skip counter)
        when the cap is hit — the caller should skip persistence for that
        detection.

        Reads MAX_DETECTIONS_PER_BURST and BURST_WINDOW_SECONDS live from
        self.config so Web-UI changes apply on the next detection cycle.
        Disabled when MAX_DETECTIONS_PER_BURST <= 0.
        """
        try:
            max_admits = int(self.config.get("MAX_DETECTIONS_PER_BURST", 100))
        except (TypeError, ValueError):
            max_admits = 100
        try:
            window_seconds = float(self.config.get("BURST_WINDOW_SECONDS", 60.0))
        except (TypeError, ValueError):
            window_seconds = 60.0
        if window_seconds <= 0:
            window_seconds = 60.0

        if max_admits <= 0:
            return True

        now = time.monotonic()
        cutoff = now - window_seconds
        # Trim left until the oldest entry is inside the window. deque
        # keeps timestamps in monotonic order so a single while-loop
        # suffices.
        while self._burst_timestamps and self._burst_timestamps[0] < cutoff:
            self._burst_timestamps.popleft()
        # Also trim if the cap was lowered at runtime — keep only the
        # newest max_admits entries so a freshly-tightened cap takes
        # effect immediately rather than after the window expires.
        while len(self._burst_timestamps) > max_admits:
            self._burst_timestamps.popleft()

        if len(self._burst_timestamps) >= max_admits:
            self._burst_skipped_total += 1
            # Throttled log so a sustained flock doesn't spam.
            if now - self._burst_skipped_last_log >= 30.0:
                logger.warning(
                    "[BURST-CAP] skipped %d detections in last %ds "
                    "(cap=%d / window=%.0fs)",
                    self._burst_skipped_total,
                    int(now - self._burst_skipped_last_log),
                    max_admits,
                    window_seconds,
                )
                self._burst_skipped_total = 0
                self._burst_skipped_last_log = now
            return False

        self._burst_timestamps.append(now)
        return True

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

            # Resolve the active save threshold once per frame so Filter (A)
            # uses exactly the same value as the detect-loop gate at
            # detector.py:635 (any-above-threshold). Without this, a frame
            # admitted by ONE strong detection would also persist all the
            # weaker companion detections — the root cause of issue #32.
            from config import effective_save_threshold

            detector_obj = getattr(self.detection_service, "_detector", None)
            underlying = (
                getattr(detector_obj, "model", None) if detector_obj else None
            )
            detector_conf = (
                getattr(underlying, "conf_threshold_default", None)
                if underlying is not None
                else None
            )
            save_thr = effective_save_threshold(self.config, detector_conf)

            for idx, det in enumerate(detection_info_list, start=1):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                od_conf = det["confidence"]
                bbox_tuple = (x1, y1, x2, y2)
                od_class_name = det.get("class_name", "bird")
                is_bird = is_bird_od_class(od_class_name)

                # Filter (A): per-detection save-threshold gate.
                # The frame-level gate (detector.py:635) only checks whether
                # ANY detection clears the threshold. Re-apply per detection
                # here so weaker companions in a flock are not persisted.
                if od_conf < save_thr:
                    continue

                # Filter (B): sliding-window burst cap. When too many
                # detections fire in a short window (e.g. sparrow flock),
                # stop persisting until the burst subsides.
                if not self._burst_admit():
                    continue

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

                # Classify via ClassificationService — bird track only.
                # Non-bird OD classes (squirrel/cat/marten_mustelid/hedgehog)
                # skip CLS entirely; their species identity comes straight
                # from od_class_name.
                cls_name = ""
                cls_conf = 0.0
                cls_result = None
                if is_bird:
                    try:
                        if crop_rgb is not None:
                            cls_result = self.classification_service.classify(crop_rgb)
                            cls_name = cls_result.class_name
                            cls_conf = cls_result.confidence
                            if not self.classifier_model_id:
                                self.classifier_model_id = cls_result.model_id or ""
                    except Exception as e:
                        logger.error(f"Classification error: {e}")

                # Species key for temporal smoothing:
                # - bird track: CLS result, or "unknown" when CLS failed
                # - non-bird track: the OD class name itself (it IS the species)
                if is_bird:
                    species_key = cls_name or "unknown"
                else:
                    species_key = od_class_name

                # Centralised scoring pipeline (single source of truth)
                signals = compute_detection_signals(
                    bbox=bbox_tuple,
                    frame_shape=original_frame.shape,
                    od_conf=od_conf,
                    cls_conf=cls_conf,
                    top_k_confidences=(
                        cls_result.top_k_confidences
                        if cls_result is not None and cls_conf > 0
                        else None
                    ),
                    decision_policy=self.decision_policy_service,
                    temporal_service=self.temporal_decision_service,
                    capability_registry=self.capability_registry,
                    species_key=species_key,
                    od_class_name=od_class_name,
                    non_bird_confirm_threshold=self.config.get(
                        "SAVE_THRESHOLD", 0.65
                    ),
                )
                score = signals.score
                agreement = signals.agreement_score
                smoothed_state = signals.decision_state

                # Save detection via PersistenceService
                det_data = DetectionData(
                    bbox=(x1, y1, x2, y2),
                    confidence=od_conf,
                    class_name=det.get("class_name", "bird"),
                    cls_class_name=cls_name,
                    cls_confidence=cls_conf,
                    score=score,
                    agreement_score=agreement,
                    decision_state=smoothed_state,
                    bbox_quality=signals.bbox_quality,
                    unknown_score=signals.unknown_score,
                    decision_reasons=signals.decision_reasons_json,
                    policy_version=signals.policy_version,
                    top_k_predictions=list(
                        zip(
                            (getattr(cls_result, "top_k_classes", []) or [])[1:],
                            [
                                float(c)
                                for c in (
                                    getattr(cls_result, "top_k_confidences", []) or []
                                )[1:]
                            ],
                            strict=False,
                        )
                    )
                    if cls_conf > 0
                    else [],
                    decision_level=getattr(cls_result, "decision_level", None)
                    if cls_result is not None
                    else None,
                    raw_species_name=getattr(cls_result, "raw_species_name", None)
                    if cls_result is not None
                    else None,
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

                    # P1-03: session counter for operational monitoring
                    if smoothed_state and smoothed_state in self.decision_state_counts:
                        self.decision_state_counts[smoothed_state] += 1

                    # Track best detection for notification.
                    # Policy ON:  gate on smoothed decision state == CONFIRMED.
                    # Policy OFF: conservative legacy gate — cls_conf > 0
                    #             AND score >= SAVE_THRESHOLD.
                    notify_eligible = False
                    if smoothed_state is not None:
                        # Decision policy active → require CONFIRMED
                        notify_eligible = smoothed_state == DecisionState.CONFIRMED
                    else:
                        # Legacy-conservative fallback (no decision policy).
                        # Reuses save_thr already resolved at the top of the
                        # detection loop so auto and manual modes stay
                        # consistent across gates.
                        notify_eligible = cls_conf > 0 and score >= save_thr

                    if score > best_score and notify_eligible:
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

            # Log timing.
            # The detect-loop and processing-loop run on separate threads.
            # det_cycle = DETECTION_INTERVAL_SECONDS target = DET + det_idle
            # (the idle is what the *detect* loop sleeps after queuing this
            # job). CLS happens here in the processing loop and does NOT
            # add to det_cycle — it runs in parallel with the next detect.
            cls_duration_ms = int((time.time() - cls_start) * 1000)
            det_idle_ms = job["sleep_time_ms"]
            det_cycle_ms = detection_time_ms + det_idle_ms
            # Feed the CLS sample into the shared summary buffer; the
            # detect-loop reads it on its 15s-tick so the summary line
            # carries both DET and CLS aggregates.
            self._cls_times.append(cls_duration_ms)
            logger.info(
                f"[DET+CLS] pipeline={detection_time_ms + cls_duration_ms}ms "
                f"(DET={detection_time_ms}ms, CLS={cls_duration_ms}ms) | "
                f"Objects={len(detection_info_list)} | "
                f"det_cycle={det_cycle_ms}ms (idle {det_idle_ms}ms)"
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
