"""Rejected classifier results never persist photos or object records."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from detectors.detection_manager import DetectionManager

import numpy as np
import pytest

from detectors.interfaces.classification import (
    ClassificationResult,
    DecisionState,
)


def _build_mgr_fixture() -> DetectionManager:
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
    fake_img_result = MagicMock()
    fake_img_result.success = True
    fake_img_result.base_filename = "20260519_211700_273402.jpg"
    mgr.persistence_service.save_image.return_value = fake_img_result
    fake_det_result = MagicMock()
    fake_det_result.thumbnail_path = "/fake/thumb.webp"
    mgr.persistence_service.save_detection.return_value = fake_det_result

    mgr.notification_service = MagicMock()

    from detectors.services.capability_registry import build_default_registry
    from detectors.services.decision_policy_service import DecisionPolicyService

    mgr.capability_registry = build_default_registry()
    mgr.decision_policy_service = DecisionPolicyService()

    mgr.classifier_model_id = "20260427_143835"
    mgr.detector_model_id = "yolox_s_v2_coco"
    mgr.current_source_id = 1
    mgr.location_config = None
    mgr.exif_gps_enabled = False
    mgr.decision_state_counts = dict.fromkeys(DecisionState, 0)

    fake_underlying = MagicMock()
    fake_underlying.conf_per_class_name = {}
    fake_underlying.conf_threshold_default = 0.30
    fake_detector_wrapper = MagicMock()
    fake_detector_wrapper.model = fake_underlying
    mgr.detection_service = MagicMock()
    mgr.detection_service._detector = fake_detector_wrapper

    mgr._burst_admit = MagicMock(return_value=True)
    mgr._same_bird_burst_admit = MagicMock(return_value=True)
    mgr._burst_timestamps = deque()
    mgr._recent_admissions = deque()

    from detectors.pipeline_metrics import PipelineMetrics

    mgr.pipeline_metrics = PipelineMetrics()
    mgr._inference_error_state = False

    return mgr


def _make_cls_result(
    level: str, conf: float, raw_species: str = "Parus_major"
) -> ClassificationResult:
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


def _enqueue_one_frame(
    mgr: DetectionManager,
    detections: list[dict[str, Any]],
    cls_results: list[ClassificationResult],
) -> None:
    from datetime import datetime

    job = {
        "capture_time_precise": datetime(2026, 5, 19, 21, 17, 0, 273402),
        "original_frame": np.zeros((1920, 2560, 3), dtype=np.uint8),
        "detection_info_list": detections,
        "detection_time_ms": 800,
        "sleep_time_ms": 100,
    }
    mgr.classification_service.classify.side_effect = cls_results
    mgr.frame_lock = MagicMock()
    mgr.latest_detection_time = 0
    mgr._process_job(job)


def test_reject_only_frame_does_not_call_save_image() -> None:
    """Static false-positive scenario: 3 bird detections all rejected → no image saved."""
    mgr = _build_mgr_fixture()
    detections = [
        {
            "class_name": "bird",
            "confidence": 0.43,
            "x1": 1456,
            "y1": 288,
            "x2": 1712,
            "y2": 864,
        },
        {
            "class_name": "bird",
            "confidence": 0.51,
            "x1": 1456,
            "y1": 290,
            "x2": 1712,
            "y2": 868,
        },
        {
            "class_name": "bird",
            "confidence": 0.47,
            "x1": 1455,
            "y1": 287,
            "x2": 1715,
            "y2": 864,
        },
    ]
    cls_results = [
        _make_cls_result("reject", 0.22, "Troglodytes_troglodytes"),
        _make_cls_result("reject", 0.18, "Troglodytes_troglodytes"),
        _make_cls_result("reject", 0.25, "Troglodytes_troglodytes"),
    ]
    _enqueue_one_frame(mgr, detections, cls_results)

    mgr.persistence_service.save_image.assert_not_called()
    mgr.persistence_service.save_detection.assert_not_called()


def test_mixed_frame_saves_only_the_keeper() -> None:
    """1 species + 1 reject → save_image once, save_detection once, no reject records."""
    mgr = _build_mgr_fixture()
    detections = [
        {
            "class_name": "bird",
            "confidence": 0.85,
            "x1": 100,
            "y1": 100,
            "x2": 400,
            "y2": 500,
        },
        {
            "class_name": "bird",
            "confidence": 0.45,
            "x1": 1456,
            "y1": 288,
            "x2": 1712,
            "y2": 864,
        },
    ]
    cls_results = [
        _make_cls_result("species", 0.95, "Parus_major"),
        _make_cls_result("reject", 0.22, "Troglodytes_troglodytes"),
    ]
    _enqueue_one_frame(mgr, detections, cls_results)

    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 1


def test_two_keepers_share_a_single_image_save() -> None:
    """First keeper triggers save_image; second keeper reuses base_filename."""
    mgr = _build_mgr_fixture()
    detections = [
        {
            "class_name": "bird",
            "confidence": 0.85,
            "x1": 100,
            "y1": 100,
            "x2": 400,
            "y2": 500,
        },
        {
            "class_name": "bird",
            "confidence": 0.80,
            "x1": 600,
            "y1": 100,
            "x2": 900,
            "y2": 500,
        },
    ]
    cls_results = [
        _make_cls_result("species", 0.95, "Parus_major"),
        _make_cls_result("species", 0.90, "Cyanistes_caeruleus"),
    ]
    _enqueue_one_frame(mgr, detections, cls_results)

    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 2


def test_species_review_keeper_persists_normally() -> None:
    mgr = _build_mgr_fixture()
    detections = [
        {
            "class_name": "bird",
            "confidence": 0.55,
            "x1": 100,
            "y1": 100,
            "x2": 400,
            "y2": 500,
        },
    ]
    cls_results = [_make_cls_result("species_review", 0.50, "Sylvia_atricapilla")]
    _enqueue_one_frame(mgr, detections, cls_results)

    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 1


@pytest.mark.parametrize(
    "class_name", ["squirrel", "marten_mustelid", "cat", "hedgehog"]
)
def test_non_bird_high_conf_skips_classifier_and_persists(class_name: str) -> None:
    mgr = _build_mgr_fixture()
    detections = [
        {
            "class_name": class_name,
            "confidence": 0.85,
            "x1": 100,
            "y1": 100,
            "x2": 400,
            "y2": 500,
        },
    ]
    _enqueue_one_frame(mgr, detections, [])

    mgr.classification_service.classify.assert_not_called()
    assert mgr.persistence_service.save_image.call_count == 1
    assert mgr.persistence_service.save_detection.call_count == 1
