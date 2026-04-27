"""Tests for the per-detection save-threshold filter (A) and the
sliding-window burst-cap (B) in DetectionManager._processing_loop.

Background — issue #32 (sparrow-flock floods the review queue):
The frame-level save-threshold gate at detector.py:635 only checks
whether ANY detection in a frame clears the threshold. Once a frame
is admitted, ALL detections in it are persisted, including weaker
companions. For a flock of 1400 sparrows in a single event this means
1400 DB rows + 1400 thumbnails — the review UI cannot keep up.

Filter (A): per-detection re-application of save_threshold.
Filter (B): sliding-window cap on persisted detections per minute.

Both gates run in detection_manager._processing_loop right at the top
of the per-detection loop, before any classification/persistence work.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from detectors.detection_manager import DetectionManager
from detectors.interfaces.classification import ClassificationResult
from detectors.interfaces.persistence import (
    DetectionPersistenceResult,
    ImagePersistenceResult,
)


@pytest.fixture
def manager(monkeypatch, tmp_path):
    """Build a DetectionManager with mocked services and SAVE_THRESHOLD=0.65.

    Manual mode is forced so the test does not depend on a live detector
    (effective_save_threshold falls back to manual_val when no detector
    is wired up — but pinning the mode makes the intent explicit).
    """
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SAVE_THRESHOLD", "0.65")
    monkeypatch.setenv("SAVE_THRESHOLD_MODE", "manual")

    mgr = DetectionManager()

    mgr.persistence_service = MagicMock()
    mgr.notification_service = MagicMock()
    mgr.crop_service = MagicMock()
    mgr.classification_service = MagicMock()

    mgr.persistence_service.save_image.return_value = ImagePersistenceResult(
        success=True, base_filename="test.jpg"
    )
    mgr.persistence_service.save_detection.return_value = (
        DetectionPersistenceResult(success=True, thumbnail_path="thumb.webp")
    )
    mgr.crop_service.create_classification_crop.return_value = np.zeros(
        (224, 224, 3), dtype=np.uint8
    )
    # Provide a real ClassificationResult so cls_conf is a float, not a Mock.
    # Confidence is high enough that the CLS-side gates don't accidentally
    # filter the detection — these tests focus on the OD-confidence path.
    mgr.classification_service.classify.return_value = ClassificationResult(
        class_name="parus_major",
        confidence=0.8,
        model_id="test_model",
        top_k_confidences=[0.8, 0.05, 0.05, 0.05, 0.05],
    )

    return mgr


def _drain_one_job(mgr, job):
    """Run _processing_loop just long enough to drain `job` and exit."""
    mgr.processing_queue.put(job)

    def mock_is_set(_state={"called": False}):
        if _state["called"]:
            return True
        _state["called"] = True
        return False

    mgr.stop_event.is_set = mock_is_set
    mgr._processing_loop()


# ---------------------------------------------------------------------------
# Filter (A): per-detection save-threshold gate
# ---------------------------------------------------------------------------


def test_filter_a_skips_detections_below_save_threshold(manager):
    """A frame with one strong + two weak detections should persist
    only the strong one. This is the issue-#32 fix in a nutshell."""
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            # Strong detection — clears 0.65
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.90},
            # Weak companions — would have been persisted under the old
            # any-above-threshold logic
            {"x1": 300, "y1": 100, "x2": 400, "y2": 200, "confidence": 0.40},
            {"x1": 500, "y1": 100, "x2": 600, "y2": 200, "confidence": 0.55},
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }

    _drain_one_job(manager, job)

    assert manager.persistence_service.save_detection.call_count == 1
    persisted_conf = (
        manager.persistence_service.save_detection.call_args[1]["detection"].confidence
    )
    assert persisted_conf == pytest.approx(0.90)


def test_filter_a_persists_detection_at_exact_threshold(manager):
    """Boundary check: confidence == save_threshold should pass (>=, not >)."""
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.65},
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }
    _drain_one_job(manager, job)
    assert manager.persistence_service.save_detection.call_count == 1


def test_filter_a_persists_all_when_all_above_threshold(manager):
    """No regression for the legitimate flock case where every detection
    is genuinely confident — those should still all be saved."""
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.90},
            {"x1": 300, "y1": 100, "x2": 400, "y2": 200, "confidence": 0.85},
            {"x1": 500, "y1": 100, "x2": 600, "y2": 200, "confidence": 0.75},
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }
    _drain_one_job(manager, job)
    assert manager.persistence_service.save_detection.call_count == 3


# ---------------------------------------------------------------------------
# Filter (B): sliding-window burst cap
# ---------------------------------------------------------------------------


def test_filter_b_admits_up_to_cap(manager):
    """Within the rolling window, exactly cap detections are admitted."""
    manager.config["MAX_DETECTIONS_PER_BURST"] = 3
    manager.config["BURST_WINDOW_SECONDS"] = 60.0
    manager._burst_timestamps.clear()

    # 5 strong detections in one frame; cap=3 means 3 admitted, 2 rejected.
    dets = [
        {"x1": 10 * i, "y1": 0, "x2": 10 * i + 5, "y2": 5, "confidence": 0.9}
        for i in range(5)
    ]
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": dets,
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }
    _drain_one_job(manager, job)

    assert manager.persistence_service.save_detection.call_count == 3


def test_filter_b_disabled_when_cap_is_zero(manager):
    """MAX_DETECTIONS_PER_BURST=0 must be a hard disable — every detection
    above the save threshold passes through."""
    manager.config["MAX_DETECTIONS_PER_BURST"] = 0
    manager._burst_timestamps.clear()

    dets = [
        {"x1": 10 * i, "y1": 0, "x2": 10 * i + 5, "y2": 5, "confidence": 0.9}
        for i in range(50)
    ]
    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": dets,
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }
    _drain_one_job(manager, job)

    assert manager.persistence_service.save_detection.call_count == 50


def test_filter_b_window_expires_old_entries(manager, monkeypatch):
    """Once entries fall outside the window, capacity opens up again."""
    manager.config["MAX_DETECTIONS_PER_BURST"] = 2
    manager.config["BURST_WINDOW_SECONDS"] = 60.0
    manager._burst_timestamps.clear()

    # Fast-forward the monotonic clock between calls.
    fake_now = [1000.0]
    monkeypatch.setattr("detectors.detection_manager.time.monotonic", lambda: fake_now[0])

    # First two calls fill the window
    assert manager._burst_admit() is True
    assert manager._burst_admit() is True
    # Third is rejected
    assert manager._burst_admit() is False

    # Move time past the window
    fake_now[0] += 61.0

    # Now both old entries are expired; capacity is fresh
    assert manager._burst_admit() is True


def test_filter_b_runs_after_filter_a(manager):
    """Order matters: weak detections should not consume burst-cap slots.
    A frame with 3 weak + 3 strong detections, cap=2 → 2 strong admitted,
    1 strong rejected by burst-cap, all weak rejected by save-threshold
    (and never reach the burst-cap)."""
    manager.config["MAX_DETECTIONS_PER_BURST"] = 2
    manager._burst_timestamps.clear()

    job = {
        "capture_time_precise": datetime.now(),
        "original_frame": np.zeros((1080, 1920, 3), dtype=np.uint8),
        "detection_info_list": [
            {"x1": 0, "y1": 0, "x2": 5, "y2": 5, "confidence": 0.30},  # weak
            {"x1": 10, "y1": 0, "x2": 15, "y2": 5, "confidence": 0.90},  # strong
            {"x1": 20, "y1": 0, "x2": 25, "y2": 5, "confidence": 0.40},  # weak
            {"x1": 30, "y1": 0, "x2": 35, "y2": 5, "confidence": 0.85},  # strong
            {"x1": 40, "y1": 0, "x2": 45, "y2": 5, "confidence": 0.55},  # weak
            {"x1": 50, "y1": 0, "x2": 55, "y2": 5, "confidence": 0.80},  # strong
        ],
        "detection_time_ms": 100,
        "sleep_time_ms": 10,
    }
    _drain_one_job(manager, job)

    # If filter (A) ran first: 3 strong reach (B), cap=2 admits 2 → 2 saves.
    # If filter (B) ran first: weak detections would steal slots → fewer saves.
    assert manager.persistence_service.save_detection.call_count == 2


def test_filter_b_reads_cap_live_from_config(manager):
    """The burst cap must be read live from self.config on every call so
    Web-UI changes apply on the next detection — same semantics as
    SAVE_THRESHOLD. Cache-on-init would have required a restart."""
    manager.config["MAX_DETECTIONS_PER_BURST"] = 2
    manager._burst_timestamps.clear()

    assert manager._burst_admit() is True
    assert manager._burst_admit() is True
    assert manager._burst_admit() is False  # cap hit at 2

    # Operator raises the cap via Web UI mid-run
    manager.config["MAX_DETECTIONS_PER_BURST"] = 5
    assert manager._burst_admit() is True  # capacity restored
    assert manager._burst_admit() is True
    assert manager._burst_admit() is True

    # And tightens it again — entries beyond the new cap should be
    # trimmed immediately, not after the window expires.
    manager.config["MAX_DETECTIONS_PER_BURST"] = 1
    assert manager._burst_admit() is False


def test_filter_b_keys_in_runtime_keys():
    """Both burst-cap keys must be in RUNTIME_KEYS so the Web UI
    settings POST handler actually persists them. Without this,
    update_runtime_settings() silently drops the change."""
    from config import RUNTIME_KEYS

    assert "MAX_DETECTIONS_PER_BURST" in RUNTIME_KEYS
    assert "BURST_WINDOW_SECONDS" in RUNTIME_KEYS


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_loads_burst_cap_defaults():
    """Defaults match the documented values; both keys present after load."""
    from config import DEFAULTS, get_config

    assert DEFAULTS["MAX_DETECTIONS_PER_BURST"] == 100
    assert DEFAULTS["BURST_WINDOW_SECONDS"] == 60.0

    cfg = get_config()
    assert "MAX_DETECTIONS_PER_BURST" in cfg
    assert "BURST_WINDOW_SECONDS" in cfg
    assert cfg["MAX_DETECTIONS_PER_BURST"] >= 0
    assert cfg["BURST_WINDOW_SECONDS"] > 0


def test_config_validates_burst_cap_via_validator():
    """_validate_value rejects negatives and zeros where appropriate."""
    from config import _validate_value

    ok, val = _validate_value("MAX_DETECTIONS_PER_BURST", "0")
    assert ok is True and val == 0  # 0 disables the cap

    ok, val = _validate_value("MAX_DETECTIONS_PER_BURST", "250")
    assert ok is True and val == 250

    ok, _ = _validate_value("MAX_DETECTIONS_PER_BURST", "-1")
    assert ok is False

    ok, val = _validate_value("BURST_WINDOW_SECONDS", "30")
    assert ok is True and val == 30.0

    ok, _ = _validate_value("BURST_WINDOW_SECONDS", "0")
    assert ok is False  # window must be > 0

    ok, _ = _validate_value("BURST_WINDOW_SECONDS", "abc")
    assert ok is False
