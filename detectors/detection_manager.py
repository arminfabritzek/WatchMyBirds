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
from datetime import UTC, datetime

import numpy as np

from camera.video_capture import VideoCapture
from config import get_config
from core.ptz_tracking_core import AutoPtzController
from detectors.classifier import ImageClassifier
from detectors.interfaces.classification import DecisionState
from detectors.motion_detector import MotionDetector
from detectors.od_classes import is_bird_od_class
from detectors.services import NotificationService, PersistenceService
from detectors.services.capability_registry import build_default_registry
from detectors.services.classification_service import ClassificationService
from detectors.services.crop_service import CropService
from detectors.services.decision_policy_service import DecisionPolicyService
from detectors.services.detection_service import DetectionService
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

    def __init__(self) -> None:
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

        # Same-bird burst suppression (Filter B2): rolling list of
        # recently-admitted (timestamp, bbox, species_key) tuples. A new
        # detection that overlaps a recent admission by IoU > threshold
        # AND shares the species_key is skipped. Window and IoU thresh
        # are read live from self.config so Web-UI changes take effect
        # immediately. species_key is the OD class name for non-birds
        # and the CLS top-1 species name for birds — set to "" when CLS
        # has not yet run, in which case any overlapping recent entry
        # (regardless of species) blocks the admission.
        self._recent_admissions: deque[tuple[float, tuple[int, int, int, int], str]] = (
            deque()
        )
        self._same_bird_skipped_total = 0
        self._same_bird_skipped_last_log = time.monotonic()

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

        # Daylight cache for the OD night-pause gate. Refreshed at
        # most once per `_daytime_ttl` seconds via _should_run_od_now().
        # When DAY_AND_NIGHT_CAPTURE is True (default), the gate is a
        # no-op — OD runs 24/7. When False, OD pauses outside the
        # operator-defined daytime window (see utils/sun_times.py).
        # `value` is the most recent is_daytime() result;
        # `next_transition` is the UTC timestamp at which the value
        # will flip — used by the status endpoint and the transition
        # log line. `ts` is the monotonic clock at last refresh.
        self._daytime_cache: dict = {
            "value": True,
            "next_transition": None,
            "ts": 0.0,
        }
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

        # Service layer components.
        # Order matters: auto_ptz_controller is constructed first so the
        # PersistenceService can hold a reference and record frame-time
        # PTZ state on each saved image (see images.ptz_* columns).
        self.auto_ptz_controller = AutoPtzController()
        # Register with the empirical-probe wizard backend so the wizard
        # can pause/resume Auto-PTZ for the duration of operator-attended
        # probe runs. Lazy import to keep detection_manager free of any
        # web-layer dependencies (H-02).
        try:
            from core.ptz_empirical_probe import register_auto_ptz_controller

            register_auto_ptz_controller(self.auto_ptz_controller)
        except Exception:  # noqa: BLE001 — wizard registration is optional
            logger.debug("Empirical-probe wizard registration skipped")
        self.notification_service = NotificationService(common_names=self.common_names)
        self.persistence_service = PersistenceService(
            ptz_controller=self.auto_ptz_controller
        )
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
        od_class_name: str | None = None,
    ) -> "ScoringResult":
        """Delegate to :func:`scoring_pipeline.compute_detection_signals`.

        ``od_class_name`` routes deep-review reanalysis through the same
        non-bird gate as live ingest. Omitting it preserves the legacy
        bird-track-only behaviour for callers that have no class info.

        Builds the same per-class resolver as the live `_processing_loop`
        so deep-review reanalysis uses the model's per-class floors when
        a v2-coco-shaped detector is loaded, and falls back to the scalar
        for 5-class models.
        """
        detection_service = getattr(self, "detection_service", None)
        detector_obj = getattr(detection_service, "_detector", None)
        underlying = getattr(detector_obj, "model", None) if detector_obj else None
        per_class_map: dict[str, float] = (
            getattr(underlying, "conf_per_class_name", {}) or {}
            if underlying is not None
            else {}
        )
        global_non_bird_floor = float(
            self.config.get("NON_BIRD_CONFIRM_THRESHOLD", 0.80)
        )

        def non_bird_floor_for(class_name: str) -> float:
            # See non_bird_floor_for() in run() for the rationale:
            # global NON_BIRD_CONFIRM_THRESHOLD is the minimum; per-class
            # entries from the detector YAML may only RAISE it.
            per_class = per_class_map.get(class_name)
            if per_class is None:
                return global_non_bird_floor
            return max(float(per_class), global_non_bird_floor)

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
            od_class_name=od_class_name,
            non_bird_confirm_threshold=global_non_bird_floor,
            non_bird_confirm_threshold_fn=non_bird_floor_for,
        )

    def run_exhaustive_scan(self, frame: np.ndarray) -> list:
        """Compatibility adapter for orphan deep-scan workflows."""
        detections = self.detection_service.exhaustive_detect(frame)
        if not self.detector_model_id:
            self.detector_model_id = self.detection_service.get_model_id()
        return detections

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:
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
        try:
            self.auto_ptz_controller.return_to_overview(allow_disabled=True)
        except Exception as exc:  # noqa: BLE001 — PTZ failures must not block startup
            logger.warning("PTZ return-to-overview at boot failed: %s", exc)
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

    def stop(self) -> None:
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

        self.auto_ptz_controller.stop()

        logger.info("DetectionManager V2 stopped.")

    def enter_deep_scan_mode(self) -> None:
        """Pause live loops while a manual/nightly deep scan is running."""
        with self._deep_scan_lock:
            if self._deep_scan_gate_count == 0:
                self._paused_before_deep_scan = self.paused
                self.paused = True
                self._deep_scan_active = True
            self._deep_scan_gate_count += 1

    def exit_deep_scan_mode(self) -> None:
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
    # OD NIGHT-PAUSE GATE
    # =========================================================================
    #
    # When DAY_AND_NIGHT_CAPTURE is True (default), OD runs 24/7
    # exactly as before — the gate is a no-op. When False, OD pauses
    # outside the operator-defined daytime window. The daytime window
    # is civil-twilight by default, widened by the offsets:
    #   * OD_NIGHT_START_OFFSET_MIN extends evening (default +30 min)
    #   * OD_NIGHT_END_OFFSET_MIN extends early morning (default -45 min)
    #
    # The cache refresh is rate-limited to _daytime_ttl seconds so
    # astral.sun() is not called on every detect tick. The single
    # log line on transitions makes pauses visible in app.log
    # without per-frame spam.

    def _should_run_od_now(self) -> bool:
        """Return True if OD should execute now, False if it should pause.

        Reads the daytime cache and refreshes it if the TTL is up.
        Logs a single info line on every day↔night transition.
        """
        master_switch = bool(self.config.get("DAY_AND_NIGHT_CAPTURE", True))
        if master_switch:
            return True

        now_mono = time.monotonic()
        if (now_mono - self._daytime_cache["ts"]) > self._daytime_ttl:
            self._refresh_daytime_cache()

        return bool(self._daytime_cache["value"])

    def _refresh_daytime_cache(self) -> None:
        """Recompute is_daytime() and update the cache. Logs transitions."""
        from datetime import datetime as _dt

        from utils.sun_times import is_daytime

        # Location resolution: prefer LOCATION_DATA (lat/lon, primary)
        # over DAY_AND_NIGHT_CAPTURE_LOCATION (city name, legacy).
        lat, lon = self._resolve_location()
        if lat is None or lon is None:
            # No location configured → can't compute twilight → fall
            # back to "always daytime" so OD never pauses silently.
            logger.warning(
                "OD night-pause: no location configured "
                "(LOCATION_DATA missing); pause disabled."
            )
            self._daytime_cache = {
                "value": True,
                "next_transition": None,
                "ts": time.monotonic(),
            }
            return

        try:
            start_off = int(self.config.get("OD_NIGHT_START_OFFSET_MIN", 30))
            end_off = int(self.config.get("OD_NIGHT_END_OFFSET_MIN", -45))
            tw_mode = str(self.config.get("OD_NIGHT_TWILIGHT_MODE", "civil"))
            now_utc = _dt.now(tz=UTC)
            is_day, next_transition = is_daytime(
                now_utc,
                lat=lat,
                lon=lon,
                start_offset_min=start_off,
                end_offset_min=end_off,
                twilight=tw_mode,  # type: ignore[arg-type]
            )
        except Exception:
            logger.exception(
                "OD night-pause: is_daytime() raised; defaulting to daytime."
            )
            self._daytime_cache = {
                "value": True,
                "next_transition": None,
                "ts": time.monotonic(),
            }
            return

        prev_value = self._daytime_cache.get("value")
        if prev_value is not None and prev_value != is_day:
            logger.info(
                "OD night-pause transition: %s → %s (next at %s, lat=%.4f, lon=%.4f, twilight=%s)",
                "daytime" if prev_value else "night",
                "daytime" if is_day else "night",
                next_transition.isoformat(),
                lat,
                lon,
                tw_mode,
            )
        self._daytime_cache = {
            "value": is_day,
            "next_transition": next_transition,
            "ts": time.monotonic(),
        }

    def _resolve_location(self) -> tuple[float | None, float | None]:
        """Return (lat, lon) from LOCATION_DATA. None if unset/(0,0)."""
        loc = self.config.get("LOCATION_DATA")
        if not loc:
            return (None, None)
        # LOCATION_DATA is stored as a dict {"latitude": .., "longitude": ..}
        # or a "lat,lon" string in some legacy configs. Handle both.
        try:
            if isinstance(loc, dict):
                lat = float(loc.get("latitude") or loc.get("lat") or 0.0)
                lon = float(loc.get("longitude") or loc.get("lon") or 0.0)
            elif isinstance(loc, str) and "," in loc:
                a, b = loc.split(",", 1)
                lat, lon = float(a), float(b)
            else:
                return (None, None)
        except (TypeError, ValueError):
            return (None, None)
        if lat == 0.0 and lon == 0.0:
            return (None, None)
        return (lat, lon)

    def get_od_status(self) -> dict:
        """Snapshot for the /api/v1/od/status endpoint and the UI pill.

        Returns a dict with the current OD activity state, the reason
        ("master-switch-on" / "daytime" / "night-paused" / "no-location"),
        the next transition timestamp if known, and the resolved
        location lat/lon (for debugging).
        """
        master_switch = bool(self.config.get("DAY_AND_NIGHT_CAPTURE", True))
        lat, lon = self._resolve_location()

        if master_switch:
            return {
                "od_active": True,
                "reason": "master-switch-on",
                "next_transition_utc": None,
                "lat": lat,
                "lon": lon,
                "twilight_mode": str(
                    self.config.get("OD_NIGHT_TWILIGHT_MODE", "civil")
                ),
            }

        if lat is None or lon is None:
            return {
                "od_active": True,
                "reason": "no-location",
                "next_transition_utc": None,
                "lat": None,
                "lon": None,
                "twilight_mode": str(
                    self.config.get("OD_NIGHT_TWILIGHT_MODE", "civil")
                ),
            }

        # Trigger a refresh if stale, then read.
        if (time.monotonic() - self._daytime_cache["ts"]) > self._daytime_ttl:
            self._refresh_daytime_cache()

        is_day = bool(self._daytime_cache["value"])
        nt = self._daytime_cache["next_transition"]
        return {
            "od_active": is_day,
            "reason": "daytime" if is_day else "night-paused",
            "next_transition_utc": nt.isoformat() if nt is not None else None,
            "lat": lat,
            "lon": lon,
            "twilight_mode": str(self.config.get("OD_NIGHT_TWILIGHT_MODE", "civil")),
        }

    # =========================================================================
    # COMPONENT INITIALIZATION
    # =========================================================================

    def _initialize_components(self) -> bool:
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

    def _frame_update_loop(self) -> None:
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

    def _detection_loop(self) -> None:
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

            # OD night-pause gate. Master switch DAY_AND_NIGHT_CAPTURE
            # is True by default → this is a no-op. When False, skip
            # OD entirely outside the operator-defined daytime window.
            # We sleep the same interval the live loop uses, so the
            # next tick respects DETECTION_INTERVAL_SECONDS regardless
            # of whether we ran detection or not.
            if not self._should_run_od_now():
                time.sleep(float(self.config.get("DETECTION_INTERVAL_SECONDS", 1.0)))
                continue

            # Motion detection gate
            if self.config.get("MOTION_DETECTION_ENABLED", True):
                if not self.motion_detector.detect(raw_frame):
                    try:
                        self.auto_ptz_controller.handle_no_detection()
                    except Exception:
                        logger.exception("Auto PTZ no-detection update failed")
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

            try:
                frame_for_ptz = (
                    original_frame if original_frame is not None else raw_frame
                )
                if object_detected:
                    # Pass the same save-threshold the gallery uses so the
                    # cam doesn't chase phantom detections between the
                    # detection floor and the save floor (leaves, shadows,
                    # low-confidence false positives). Detections under
                    # save_thr fall through to handle_no_detection().
                    self.auto_ptz_controller.handle_detections(
                        frame_shape=frame_for_ptz.shape,
                        detections=detection_info_list,
                        min_confidence=float(save_thr or 0.0),
                    )
                else:
                    self.auto_ptz_controller.handle_no_detection()
            except Exception:
                logger.exception("Auto PTZ update failed")

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
                        window_s,
                        n_det,
                        det_avg,
                        det_lo,
                        det_hi,
                        n_cls,
                        cls_avg,
                        cls_lo,
                        cls_hi,
                    )
                else:
                    logger.info(
                        "[DET] %ds summary: %d frames | "
                        "avg %dms | min %dms | max %dms | CLS no samples",
                        window_s,
                        n_det,
                        det_avg,
                        det_lo,
                        det_hi,
                    )
                self._det_times.clear()
                self._cls_times.clear()
                self._det_summary_last = now_mono

            time.sleep(sleep_time)

        logger.info("Detection loop stopped.")

    def _enqueue_processing_job(self, job: dict[str, object]) -> None:
        """Enqueue job, drop oldest if full."""
        try:
            self.processing_queue.put_nowait(job)
        except queue.Full:
            try:
                self.processing_queue.get_nowait()
            except queue.Empty:
                # Another consumer drained the queue after the Full check.
                pass
            self.processing_queue.put_nowait(job)

    @staticmethod
    def _bbox_iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two (x1, y1, x2, y2) bboxes.

        Returns 0.0 for non-overlapping or zero-area boxes. Pure stdlib
        so the method is testable without numpy stubs.
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _same_bird_burst_admit(
        self,
        bbox: tuple[int, int, int, int],
        species_key: str,
    ) -> bool:
        """Per-bbox burst-suppression gate (Filter B2).

        Returns True when no recent admission overlaps this bbox by
        IoU > SAME_BIRD_BURST_IOU within SAME_BIRD_BURST_WINDOW_SECONDS,
        AND records the new admission.

        Returns False (skip persistence) when an overlapping same-species
        admission was made recently. species_key="" matches any species
        in the recent window — used when CLS has not yet decided what
        species this is.

        Reads window + IoU thresholds live from self.config so Web-UI
        changes apply on the next detection cycle. Disabled when
        SAME_BIRD_BURST_WINDOW_SECONDS <= 0.
        """
        try:
            window_seconds = float(
                self.config.get("SAME_BIRD_BURST_WINDOW_SECONDS", 15.0)
            )
        except (TypeError, ValueError):
            window_seconds = 15.0
        if window_seconds <= 0:
            return True
        try:
            iou_thr = float(self.config.get("SAME_BIRD_BURST_IOU", 0.6))
        except (TypeError, ValueError):
            iou_thr = 0.6

        now = time.monotonic()
        cutoff = now - window_seconds
        # Trim expired entries from the left. Deque is sorted by
        # admission time so a single while-loop suffices.
        while self._recent_admissions and self._recent_admissions[0][0] < cutoff:
            self._recent_admissions.popleft()

        # Check overlap against every still-recent admission.
        for _ts, recent_bbox, recent_species in self._recent_admissions:
            if self._bbox_iou(bbox, recent_bbox) < iou_thr:
                continue
            # Empty species_key means "block on any species match" —
            # the caller doesn't yet know what species this is, so any
            # overlapping recent admission gates it.
            if not species_key or not recent_species:
                pass
            elif species_key != recent_species:
                continue
            self._same_bird_skipped_total += 1
            if now - self._same_bird_skipped_last_log >= 30.0:
                logger.info(
                    "[SAME-BIRD] skipped %d duplicate detections in last %ds "
                    "(iou_thr=%.2f, window=%.0fs)",
                    self._same_bird_skipped_total,
                    int(now - self._same_bird_skipped_last_log),
                    iou_thr,
                    window_seconds,
                )
                self._same_bird_skipped_total = 0
                self._same_bird_skipped_last_log = now
            return False

        self._recent_admissions.append((now, bbox, species_key))
        return True

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

    def _processing_loop(self) -> None:
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

            # --- Defer image save until we know the frame has a keeper ---
            #
            # If every detection in this frame ends up at
            # ``decision_level='reject'``, we do NOT write the
            # original/optimized image or an images-row. We only log a
            # metadata audit per reject detection. This keeps disk and DB
            # from filling up with static-background FPs (tree-branch,
            # fat-ball) that the OD classifies as bird but the classifier
            # rejects.
            #
            # We pre-compute the filename stamp that ``save_image`` would
            # have generated, so detection rows / crop files / reject-
            # audit rows can all reference the same frame identity even
            # when no image file exists on disk.
            timestamp_stamp = capture_time.strftime("%Y%m%d_%H%M%S_%f")
            base_filename = f"{timestamp_stamp}.jpg"
            img_result = None  # filled lazily on first keeper detection

            # --- Process each detection ---
            best_species = None
            best_score = 0.0
            best_thumb_path = None

            # Resolve the active save threshold once per frame so Filter (A)
            # uses exactly the same value as the detect-loop gate at
            # detector.py:635 (any-above-threshold). Without this, a frame
            # admitted by ONE strong detection would also persist all the
            # weaker companion detections — the root cause of issue #32.
            from config import effective_save_threshold
            from detectors.interfaces.persistence import DetectionData

            detector_obj = getattr(self.detection_service, "_detector", None)
            underlying = getattr(detector_obj, "model", None) if detector_obj else None
            detector_conf = (
                getattr(underlying, "conf_threshold_default", None)
                if underlying is not None
                else None
            )
            save_thr = effective_save_threshold(self.config, detector_conf)

            # Build the per-class non-bird floor resolver once per frame.
            # Reads the detector's per-class map (v2-coco and later);
            # falls back to the config scalar NON_BIRD_CONFIRM_THRESHOLD
            # for any class the model didn't ship a threshold for
            # (covers 5-class models entirely).
            per_class_map: dict[str, float] = (
                getattr(underlying, "conf_per_class_name", {}) or {}
                if underlying is not None
                else {}
            )
            global_non_bird_floor = float(
                self.config.get("NON_BIRD_CONFIRM_THRESHOLD", 0.80)
            )

            def non_bird_floor_for(
                class_name: str,
                _map: dict[str, float] = per_class_map,
                _floor: float = global_non_bird_floor,
            ) -> float:
                # The global ``NON_BIRD_CONFIRM_THRESHOLD`` is the
                # *minimum* floor — non-bird classes never run through
                # the bird classifier's sanity check, so we want at
                # least bird-track-equivalent confidence before
                # persisting. Per-class entries from the detector YAML
                # may RAISE the floor (e.g. squirrel 0.70 → 0.80;
                # cat 0.75 → 0.80) but must NOT lower it (e.g.
                # hedgehog 0.30 must NOT mean "persist all hedgehogs
                # above 0.30" — that would defeat
                # NON_BIRD_DROP_BELOW_CONFIRM).
                #
                # Previously this read ``_map.get(class_name, _floor)``
                # which silently dropped the global floor whenever the
                # detector shipped a per-class threshold below it —
                # letting static-background FPs of low-confidence
                # non-bird classes flood the gallery.
                per_class = _map.get(class_name)
                if per_class is None:
                    return _floor
                return max(float(per_class), _floor)

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

                # Filter (A2): non-bird pre-persist gate. Non-bird OD classes
                # ride OD confidence directly with no CLS sanity check, so a
                # weaker floor than the bird track is the wrong default. Drop
                # non-bird detections below NON_BIRD_CONFIRM_THRESHOLD entirely
                # — no crop, no DB row, no derivative files. The downstream
                # scoring pipeline would have gated them to UNCERTAIN anyway;
                # this just stops them from costing disk and compute on the
                # way there. Flip NON_BIRD_DROP_BELOW_CONFIRM=false to keep
                # them in the DB as UNCERTAIN (e.g. during a Phase-7
                # bbox-cluster collection window).
                if not is_bird and self.config.get("NON_BIRD_DROP_BELOW_CONFIRM", True):
                    if od_conf < non_bird_floor_for(od_class_name):
                        continue

                # Filter (B): sliding-window burst cap. When too many
                # detections fire in a short window (e.g. sparrow flock),
                # stop persisting until the burst subsides.
                if not self._burst_admit():
                    continue

                # Filter (B2): same-bird burst suppression. Skip
                # persisting when a near-identical bbox + species was
                # already admitted in the last few seconds — the
                # "bird sits 30s at feeder, detected every 2s, all 15
                # frames flood the review queue" pattern. species_key
                # is set to the OD class name for non-birds (CLS does
                # not run) and left "" for birds at this point (CLS
                # has not run yet); the empty key gates on bbox-overlap
                # alone, which is the right semantics: same bbox in a
                # short window is the same bird regardless of how CLS
                # may flip-flop between Parus/Cyanistes per frame.
                species_key_for_burst = "" if is_bird else od_class_name
                if not self._same_bird_burst_admit(bbox_tuple, species_key_for_burst):
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
                    non_bird_confirm_threshold=global_non_bird_floor,
                    non_bird_confirm_threshold_fn=non_bird_floor_for,
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

                # A1a routing — reject detections do NOT write image/
                # crop/detection rows. They land in ``reject_audit`` as
                # metadata only. Everything else (species/species_review/
                # genus/None for non-bird) goes through the regular
                # persistence path, lazy-initialising the image save on
                # the first keeper.
                detection_level = (
                    getattr(cls_result, "decision_level", None)
                    if cls_result is not None
                    else None
                )
                if (
                    detection_level is not None
                    and str(detection_level).lower() == "reject"
                ):
                    try:
                        h, w = original_frame.shape[:2]
                        # Normalise bbox to 0..1 so audit rows are
                        # resolution-agnostic and cluster-queryable
                        # regardless of the source stream size.
                        norm_x = max(0.0, min(1.0, x1 / w))
                        norm_y = max(0.0, min(1.0, y1 / h))
                        norm_w = max(0.0, min(1.0, (x2 - x1) / w))
                        norm_h = max(0.0, min(1.0, (y2 - y1) / h))
                        from utils.db import insert_reject_audit

                        # Top-1 prob — when the classifier ran, its
                        # top_k_confidences[0] IS the top-1 softmax. For
                        # non-bird (no cls) and classifier failures
                        # (cls_result is None) leave it NULL.
                        top1 = None
                        if cls_result is not None:
                            tk = getattr(cls_result, "top_k_confidences", None)
                            if tk:
                                top1 = float(tk[0])

                        # Best-effort: never let an audit failure block
                        # the detection pipeline. The audit row is for
                        # cluster diagnostics, not for correctness.
                        insert_reject_audit(
                            self.persistence_service._db_conn,
                            {
                                "frame_timestamp": timestamp_stamp,
                                "frame_width": w,
                                "frame_height": h,
                                "bbox_x": norm_x,
                                "bbox_y": norm_y,
                                "bbox_w": norm_w,
                                "bbox_h": norm_h,
                                "od_class_name": od_class_name,
                                "od_confidence": float(od_conf),
                                "raw_species_name": getattr(
                                    cls_result, "raw_species_name", None
                                )
                                if cls_result is not None
                                else None,
                                "top1_prob": top1,
                                "decision_state": str(smoothed_state)
                                if smoothed_state is not None
                                else None,
                                "decision_reasons": signals.decision_reasons_json,
                                "detector_model_id": self.detector_model_id,
                                "classifier_model_id": self.classifier_model_id,
                            },
                        )
                    except Exception as audit_exc:
                        logger.warning(
                            f"reject_audit insert failed for {timestamp_stamp}: "
                            f"{audit_exc}"
                        )
                    # Still bump the session counter so the operational
                    # dashboard counts reject as a real decision event,
                    # not just silent loss.
                    if smoothed_state and smoothed_state in self.decision_state_counts:
                        self.decision_state_counts[smoothed_state] += 1
                    continue

                # First keeper of this frame triggers the lazy image
                # save. Subsequent keepers reuse the same base_filename.
                if img_result is None:
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
                    except Exception as e:
                        logger.error(f"PersistenceService.save_image error: {e}")
                        img_result = None

                    if img_result is None or not img_result.success:
                        logger.error(
                            f"Failed to save image for keeper frame {timestamp_stamp}"
                        )
                        # Without an images row we cannot persist this
                        # detection's FK either — skip the rest of the
                        # frame.
                        break
                    # PathManager generates ``<stamp>.jpg``; keep our
                    # pre-computed stamp in sync with the persisted file.
                    base_filename = img_result.base_filename

                try:
                    det_result = self.persistence_service.save_detection(
                        image_filename=base_filename,
                        detection=det_data,
                        frame=original_frame,
                        detector_model_id=self.detector_model_id,
                        classifier_model_id=self.classifier_model_id,
                        crop_index=idx,
                    )

                    # Push to the live-event bus so the stream-page LED
                    # ticker can render this detection in real time. Wrapped
                    # so a bus-side bug never breaks the detector loop.
                    try:
                        from utils.live_event_bus import get_bus

                        _species_latin = cls_name or det_data.class_name
                        _species_common = self.common_names.get(
                            _species_latin,
                            _species_latin.replace("_", " "),
                        )
                        get_bus().publish(
                            {
                                "type": "detection",
                                "ts": time.time(),
                                "species_latin": _species_latin,
                                "species_common": _species_common,
                                "od_class": det_data.class_name,
                                "confidence": float(score),
                                "od_confidence": float(od_conf),
                                "cls_confidence": float(cls_conf),
                                "decision_state": (
                                    smoothed_state.value
                                    if smoothed_state is not None
                                    else None
                                ),
                            }
                        )
                    except Exception as bus_err:
                        logger.debug("live_event_bus publish failed: %s", bus_err)

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
            logger.debug(
                f"[DET+CLS] pipeline={detection_time_ms + cls_duration_ms}ms "
                f"(DET={detection_time_ms}ms, CLS={cls_duration_ms}ms) | "
                f"Objects={len(detection_info_list)} | "
                f"det_cycle={det_cycle_ms}ms (idle {det_idle_ms}ms)"
            )

        logger.info("Processing loop stopped.")

    # =========================================================================
    # PUBLIC INTERFACE - Same as original
    # =========================================================================

    def get_display_frame(self) -> np.ndarray | None:
        """Returns the most recent frame for display."""
        with self.frame_lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return None

    def update_source(self, new_source: str) -> None:
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

    def update_configuration(self, changes: dict) -> None:
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

    def start_user_ingest(self, folder_path: str | None = None) -> None:
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
