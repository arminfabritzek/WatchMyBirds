"""DetectionManager wiring: reject vs keeper persistence (A1a).

Pins the contract that classifier-rejected detections do NOT trigger
``save_image`` / ``save_detection`` and instead land in
``reject_audit`` as metadata rows. Mixed frames (1 keeper + 1 reject)
must still write the image + the keeper's detection row.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from detectors.interfaces.classification import (
    ClassificationResult,
    DecisionState,
)


def _build_mgr_fixture():
    """Bare DetectionManager with mocked services for routing tests.

    Closely mirrors the existing
    test_detection_manager_non_bird_routing._build_detection_manager_fixture
    but adds the persistence service mock and pre-stubs a non-suppressed
    bird crop pipeline.
    """
    from detectors.detection_manager import DetectionManager

    mgr = DetectionManager.__new__(DetectionManager)
    mgr.config = {
        "SAVE_THRESHOLD": 0.30,
        "NON_BIRD_CONFIRM_THRESHOLD": 0.80,
        "NON_BIRD_DROP_BELOW_CONFIRM": True,
        "MAX_FPS_DETECTION": 30,
    }
    mgr.SAVE_RESOLUTION_CROP = 260

    mgr.crop_service = MagicMock()
    mgr.crop_service.create_classification_crop.return_value = np.zeros(
        (260, 260, 3), dtype=np.uint8
    )

    mgr.classification_service = MagicMock()
    mgr.persistence_service = MagicMock()
    mgr.persistence_service._db_conn = MagicMock()
    # save_image returns a successful result with the same base_filename
    # the lazy pre-compute would have produced
    fake_img_result = MagicMock()
    fake_img_result.success = True
    fake_img_result.base_filename = "20260519_211700_273402.jpg"
    mgr.persistence_service.save_image.return_value = fake_img_result
    fake_det_result = MagicMock()
    fake_det_result.thumbnail_path = "/fake/thumb.webp"
    mgr.persistence_service.save_detection.return_value = fake_det_result

    mgr.notification_service = MagicMock()

    # Real scoring services (cheap, isolated)
    from detectors.services.capability_registry import build_default_registry
    from detectors.services.decision_policy_service import DecisionPolicyService
    from detectors.services.temporal_decision_service import TemporalDecisionService

    mgr.capability_registry = build_default_registry()
    mgr.decision_policy_service = DecisionPolicyService()
    mgr.temporal_decision_service = TemporalDecisionService()

    mgr.classifier_model_id = "20260427_143835"
    mgr.detector_model_id = "yolox_s_v2_coco"
    mgr.current_source_id = 1
    mgr.location_config = None
    mgr.exif_gps_enabled = False
    mgr.decision_state_counts = dict.fromkeys(DecisionState, 0)

    # Detection-service plumbing so the non_bird_floor lookup finds
    # the empty per-class map (= falls back to global 0.80).
    fake_underlying = MagicMock()
    fake_underlying.conf_per_class_name = {}
    fake_underlying.conf_threshold_default = 0.30
    fake_detector_wrapper = MagicMock()
    fake_detector_wrapper.model = fake_underlying
    mgr.detection_service = MagicMock()
    mgr.detection_service._detector = fake_detector_wrapper

    # Burst gate stubbed open
    mgr._burst_admit = MagicMock(return_value=True)

    # State the production loop touches but the routing logic doesn't
    # care about. Stubbed to keep _processing_loop from raising on
    # missing private attrs.
    mgr._det_times = []
    mgr._cls_times = []
    mgr._inference_error_state = False

    return mgr


def _make_cls_result(level: str, conf: float, raw_species: str = "Parus_major"):
    """Build a ClassificationResult with a specific decision level."""
    return ClassificationResult(
        class_name="Parus_major" if level == "species" else "",
        confidence=conf,
        model_id="20260427_143835",
        top_k_classes=[raw_species, "Cyanistes_caeruleus", "Erithacus_rubecula"],
        top_k_confidences=[conf, 0.05, 0.02],
        decision_level=level,
        raw_species_name=raw_species,
    )


def _enqueue_one_frame(mgr, detections, cls_results):
    """Run one detection frame through the manager's processing path.

    We bypass the queue + threading and invoke the body of the loop
    directly by constructing a job dict and calling the private helper.
    """
    from datetime import datetime

    job = {
        "capture_time_precise": datetime(2026, 5, 19, 21, 17, 0, 273402),
        "original_frame": np.zeros((1920, 2560, 3), dtype=np.uint8),
        "detection_info_list": detections,
        "detection_time_ms": 800,
        "sleep_time_ms": 100,
    }
    # Have the classification_service return cls_results in order
    mgr.classification_service.classify.side_effect = cls_results
    # Use a queue-of-one to feed _run_processing_loop_once
    import queue

    mgr.processing_queue = queue.Queue()
    mgr.processing_queue.put(job)
    mgr.stop_event = MagicMock()
    # Loop body checks ``while not self.stop_event.is_set():`` plus a
    # second check after queue.get; let two reads return False (entry +
    # body) then True so the loop exits after one job.
    mgr.stop_event.is_set.side_effect = [False, False, True]
    mgr.frame_lock = MagicMock()
    mgr.frame_lock.__enter__ = MagicMock(return_value=None)
    mgr.frame_lock.__exit__ = MagicMock(return_value=False)
    mgr.latest_detection_time = 0

    # Patch the symbol where the production code imports it from. The
    # detection_manager does ``from utils.db import insert_reject_audit``
    # inside the reject branch, so the patch target is utils.db.
    with patch("utils.db.insert_reject_audit") as mock_audit:
        try:
            mgr._processing_loop()
        except Exception:
            # The loop exits cleanly when stop_event reads True; any
            # raise here is a real bug in the routing we want to surface.
            raise
        return mock_audit


# ---------------------------------------------------------------------------
# Reject-only frame: NO save_image, NO save_detection, audit rows only
# ---------------------------------------------------------------------------


def test_reject_only_frame_does_not_call_save_image():
    """Statik-FP scenario: 3 bird detections all rejected → no image saved."""
    mgr = _build_mgr_fixture()
    detections = [
        {"class_name": "bird", "confidence": 0.43, "x1": 1456, "y1": 288, "x2": 1712, "y2": 864},
        {"class_name": "bird", "confidence": 0.51, "x1": 1456, "y1": 290, "x2": 1712, "y2": 868},
        {"class_name": "bird", "confidence": 0.47, "x1": 1455, "y1": 287, "x2": 1715, "y2": 864},
    ]
    cls_results = [
        _make_cls_result("reject", 0.22, "Troglodytes_troglodytes"),
        _make_cls_result("reject", 0.18, "Troglodytes_troglodytes"),
        _make_cls_result("reject", 0.25, "Troglodytes_troglodytes"),
    ]
    mock_audit = _enqueue_one_frame(mgr, detections, cls_results)

    # No image, no detection row, three audit inserts
    mgr.persistence_service.save_image.assert_not_called()
    mgr.persistence_service.save_detection.assert_not_called()
    assert mock_audit.call_count == 3


def test_reject_audit_payload_carries_bbox_and_species():
    """Each audit insert gets normalised bbox + raw_species_name."""
    mgr = _build_mgr_fixture()
    detections = [
        {"class_name": "bird", "confidence": 0.43, "x1": 1280, "y1": 288, "x2": 1536, "y2": 864},
    ]
    cls_results = [_make_cls_result("reject", 0.22, "Troglodytes_troglodytes")]
    mock_audit = _enqueue_one_frame(mgr, detections, cls_results)

    assert mock_audit.call_count == 1
    payload = mock_audit.call_args.args[1]
    assert payload["od_class_name"] == "bird"
    assert payload["raw_species_name"] == "Troglodytes_troglodytes"
    assert payload["frame_timestamp"] == "20260519_211700_273402"
    # bbox normalised against 2560×1920
    assert abs(payload["bbox_x"] - (1280 / 2560)) < 1e-6
    assert abs(payload["bbox_y"] - (288 / 1920)) < 1e-6
    assert abs(payload["bbox_w"] - ((1536 - 1280) / 2560)) < 1e-6
    assert abs(payload["bbox_h"] - ((864 - 288) / 1920)) < 1e-6


# ---------------------------------------------------------------------------
# Mixed frame: keeper triggers image save, reject still goes to audit
# ---------------------------------------------------------------------------


def test_mixed_frame_saves_image_for_keeper_and_audits_reject():
    """1 species + 1 reject → save_image once, save_detection once, 1 audit."""
    mgr = _build_mgr_fixture()
    detections = [
        {"class_name": "bird", "confidence": 0.85, "x1": 100, "y1": 100, "x2": 400, "y2": 500},
        {"class_name": "bird", "confidence": 0.45, "x1": 1456, "y1": 288, "x2": 1712, "y2": 864},
    ]
    cls_results = [
        _make_cls_result("species", 0.95, "Parus_major"),
        _make_cls_result("reject", 0.22, "Troglodytes_troglodytes"),
    ]
    mock_audit = _enqueue_one_frame(mgr, detections, cls_results)

    # Image saved once (lazy: only on first keeper)
    assert mgr.persistence_service.save_image.call_count == 1
    # Only the keeper got a detection row
    assert mgr.persistence_service.save_detection.call_count == 1
    # The reject got audited
    assert mock_audit.call_count == 1


def test_two_keepers_share_a_single_image_save():
    """First keeper triggers save_image; second keeper reuses base_filename."""
    mgr = _build_mgr_fixture()
    detections = [
        {"class_name": "bird", "confidence": 0.85, "x1": 100, "y1": 100, "x2": 400, "y2": 500},
        {"class_name": "bird", "confidence": 0.80, "x1": 600, "y1": 100, "x2": 900, "y2": 500},
    ]
    cls_results = [
        _make_cls_result("species", 0.95, "Parus_major"),
        _make_cls_result("species", 0.90, "Cyanistes_caeruleus"),
    ]
    mock_audit = _enqueue_one_frame(mgr, detections, cls_results)

    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 2
    assert mock_audit.call_count == 0


# ---------------------------------------------------------------------------
# Non-bird and species_review still persist normally
# ---------------------------------------------------------------------------


def test_species_review_keeper_persists_normally():
    """species_review detections go in like species (Unclear-tab surface).

    A1a only carves out level='reject'. species_review continues to land
    in the detections table so the Unclear tab can show it.
    """
    mgr = _build_mgr_fixture()
    detections = [
        {"class_name": "bird", "confidence": 0.55, "x1": 100, "y1": 100, "x2": 400, "y2": 500},
    ]
    cls_results = [_make_cls_result("species_review", 0.50, "Sylvia_atricapilla")]
    mock_audit = _enqueue_one_frame(mgr, detections, cls_results)

    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 1
    assert mock_audit.call_count == 0


@pytest.mark.parametrize(
    "class_name", ["squirrel", "marten_mustelid", "cat", "hedgehog"]
)
def test_non_bird_high_conf_skips_classifier_and_persists(class_name):
    """Non-bird at OD-conf ≥ 0.80 persists normally (no audit, no classifier).

    Regression guard: A1a's reject-vs-keeper branch must NOT misfire on
    non-bird detections where cls_result is None.
    """
    mgr = _build_mgr_fixture()
    detections = [
        {"class_name": class_name, "confidence": 0.85, "x1": 100, "y1": 100, "x2": 400, "y2": 500},
    ]
    mock_audit = _enqueue_one_frame(mgr, detections, [])

    mgr.classification_service.classify.assert_not_called()
    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 1
    assert mock_audit.call_count == 0
