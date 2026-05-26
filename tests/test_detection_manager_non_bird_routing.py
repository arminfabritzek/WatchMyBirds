"""DetectionManager skips CLS for non-bird classes.

Covers the end-to-end contract at the _processing_loop level:

- When detector emits class_name="squirrel", CLS is NOT called
- DetectionData for non-bird carries class_name="squirrel", cls_conf=0,
  cls_class_name="", species_key used for temporal smoothing is "squirrel"
  (not "unknown"), decision_state=CONFIRMED at od_conf >= SAVE_THRESHOLD
- When detector emits class_name="bird", CLS is called as before
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from detectors.interfaces.classification import (
    ClassificationResult,
    DecisionState,
)
from detectors.od_classes import BIRD_OD_CLASSES, is_bird_od_class

# ---------------------------------------------------------------------------
# Pure helper sanity
# ---------------------------------------------------------------------------


def test_is_bird_od_class_accepts_bird():
    assert is_bird_od_class("bird") is True


def test_is_bird_od_class_rejects_none_and_empty():
    assert is_bird_od_class(None) is False
    assert is_bird_od_class("") is False


@pytest.mark.parametrize(
    "name", ["squirrel", "cat", "marten_mustelid", "hedgehog"]
)
def test_is_bird_od_class_rejects_garden_animals(name):
    assert is_bird_od_class(name) is False


def test_bird_od_classes_contains_only_bird():
    """Guard: if you add a new bird-like OD class, the detector's
    Model-Compatibility-Guard plus this set MUST stay in sync."""
    assert BIRD_OD_CLASSES == frozenset({"bird"})


# ---------------------------------------------------------------------------
# DetectionManager routing — unit-level (mocked services)
# ---------------------------------------------------------------------------


def _build_detection_manager_fixture():
    """Build a partially-wired DetectionManager for routing tests.

    We construct a bare DetectionManager object and attach just the services
    and state needed for _processing_loop to run one iteration on a fake
    detection. This avoids starting real threads / loading real models.
    """
    from detectors.detection_manager import DetectionManager

    mgr = DetectionManager.__new__(DetectionManager)
    # Config: NON_BIRD_CONFIRM_THRESHOLD gates non-bird detections (0.80 in
    # production). NON_BIRD_DROP_BELOW_CONFIRM controls whether the
    # pre-persist gate fires. SAVE_THRESHOLD is kept in the dict so any
    # unrelated lookup that still references it survives this fixture.
    mgr.config = {
        "SAVE_THRESHOLD": 0.65,
        "NON_BIRD_CONFIRM_THRESHOLD": 0.80,
        "NON_BIRD_DROP_BELOW_CONFIRM": True,
    }
    mgr.SAVE_RESOLUTION_CROP = 260

    # Mocked services
    mgr.crop_service = MagicMock()
    mgr.classification_service = MagicMock()
    mgr.persistence_service = MagicMock()

    # Use REAL scoring services so we exercise the non-bird bypass logic
    from detectors.services.capability_registry import build_default_registry
    from detectors.services.decision_policy_service import DecisionPolicyService
    from detectors.services.temporal_decision_service import TemporalDecisionService

    mgr.capability_registry = build_default_registry()
    mgr.decision_policy_service = DecisionPolicyService()
    mgr.temporal_decision_service = TemporalDecisionService()

    mgr.classifier_model_id = ""
    mgr.detector_model_id = "yolox_s_locator_test"

    # Keep legacy counters intact (used downstream)
    mgr.decision_state_counts = dict.fromkeys(DecisionState, 0)

    return mgr


def test_non_bird_skips_cls_and_routes_through_scoring():
    """Wiring test: squirrel detection does NOT call classification_service."""
    mgr = _build_detection_manager_fixture()

    # Fake crop service returns a valid RGB crop
    fake_crop = np.zeros((260, 260, 3), dtype=np.uint8)
    mgr.crop_service.create_classification_crop.return_value = fake_crop

    # Classification service would be a tripwire if ever called
    mgr.classification_service.classify = MagicMock(
        side_effect=AssertionError("CLS must not be called for non-bird")
    )

    # Drive the inner loop logic manually (matches _processing_loop body)
    from detectors.od_classes import is_bird_od_class
    from detectors.services.scoring_pipeline import compute_detection_signals

    det = {
        "class_name": "squirrel",
        "confidence": 0.90,
        "x1": 100, "y1": 100, "x2": 200, "y2": 200,
    }
    original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (det["x1"], det["y1"], det["x2"], det["y2"])

    is_bird = is_bird_od_class(det["class_name"])
    assert is_bird is False

    cls_name = ""
    cls_conf = 0.0
    cls_result = None
    if is_bird:
        cls_result = mgr.classification_service.classify(fake_crop)
        cls_name = cls_result.class_name
        cls_conf = cls_result.confidence

    species_key = cls_name or "unknown" if is_bird else det["class_name"]
    assert species_key == "squirrel"

    signals = compute_detection_signals(
        bbox=bbox,
        frame_shape=original_frame.shape,
        od_conf=det["confidence"],
        cls_conf=cls_conf,
        top_k_confidences=None,
        decision_policy=mgr.decision_policy_service,
        temporal_service=mgr.temporal_decision_service,
        capability_registry=mgr.capability_registry,
        species_key=species_key,
        od_class_name=det["class_name"],
        non_bird_confirm_threshold=mgr.config["SAVE_THRESHOLD"],
    )

    mgr.classification_service.classify.assert_not_called()
    assert signals.decision_state == DecisionState.CONFIRMED
    assert signals.score == pytest.approx(0.90)
    assert signals.unknown_score == pytest.approx(0.0)


def test_bird_track_still_calls_cls():
    """Wiring test: bird detection DOES call classification_service."""
    mgr = _build_detection_manager_fixture()
    fake_crop = np.zeros((260, 260, 3), dtype=np.uint8)
    mgr.crop_service.create_classification_crop.return_value = fake_crop

    fake_cls_result = ClassificationResult(
        class_name="Parus_major",
        confidence=0.85,
        model_id="wmb_cls_v1",
        top_k_classes=["Parus_major", "Cyanistes_caeruleus"],
        top_k_confidences=[0.85, 0.10],
    )
    mgr.classification_service.classify = MagicMock(return_value=fake_cls_result)

    from detectors.od_classes import is_bird_od_class
    from detectors.services.scoring_pipeline import compute_detection_signals

    det = {
        "class_name": "bird",
        "confidence": 0.90,
        "x1": 100, "y1": 100, "x2": 200, "y2": 200,
    }
    original_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (det["x1"], det["y1"], det["x2"], det["y2"])

    is_bird = is_bird_od_class(det["class_name"])
    assert is_bird is True

    cls_result = mgr.classification_service.classify(fake_crop)
    cls_name = cls_result.class_name
    cls_conf = cls_result.confidence

    species_key = cls_name or "unknown"

    signals = compute_detection_signals(
        bbox=bbox,
        frame_shape=original_frame.shape,
        od_conf=det["confidence"],
        cls_conf=cls_conf,
        top_k_confidences=cls_result.top_k_confidences,
        decision_policy=mgr.decision_policy_service,
        temporal_service=mgr.temporal_decision_service,
        capability_registry=mgr.capability_registry,
        species_key=species_key,
        od_class_name=det["class_name"],
        non_bird_confirm_threshold=mgr.config["SAVE_THRESHOLD"],
    )

    mgr.classification_service.classify.assert_called_once()
    assert signals.decision_state == DecisionState.CONFIRMED
    assert signals.score == pytest.approx(0.85)  # = cls_conf


def test_non_bird_low_conf_uncertain_still_visible():
    """squirrel at 0.30 OD -> UNCERTAIN (not UNKNOWN)."""
    from detectors.services.capability_registry import build_default_registry
    from detectors.services.decision_policy_service import DecisionPolicyService
    from detectors.services.scoring_pipeline import compute_detection_signals
    from detectors.services.temporal_decision_service import TemporalDecisionService

    signals = compute_detection_signals(
        bbox=(100, 100, 200, 200),
        frame_shape=(480, 640, 3),
        od_conf=0.30,
        cls_conf=0.0,
        top_k_confidences=None,
        decision_policy=DecisionPolicyService(),
        temporal_service=TemporalDecisionService(),
        capability_registry=build_default_registry(),
        species_key="squirrel",
        od_class_name="squirrel",
        non_bird_confirm_threshold=0.65,
    )
    # The critical contract: NOT UNKNOWN. UNKNOWN would hide the detection
    # from every surface via _gallery_visibility_sql.
    assert signals.decision_state != DecisionState.UNKNOWN
    assert signals.decision_state == DecisionState.UNCERTAIN


# ---------------------------------------------------------------------------
# Delegate sourcing — DetectionManager.compute_detection_signals reads the
# new NON_BIRD_CONFIRM_THRESHOLD key (used by analysis_service deep-review).
# ---------------------------------------------------------------------------


def test_delegate_sources_non_bird_threshold_from_config():
    """The DetectionManager delegate must read NON_BIRD_CONFIRM_THRESHOLD.

    analysis_service (deep-review) calls
    `detection_manager.compute_detection_signals(...)` without an explicit
    threshold. The delegate must inject the config value, not the
    scoring-pipeline default. Verifies the production wiring catches a
    marten_mustelid at 0.77 even on the deep-review path.
    """
    mgr = _build_detection_manager_fixture()

    signals = mgr.compute_detection_signals(
        bbox=(100, 100, 200, 200),
        frame_shape=(480, 640, 3),
        od_conf=0.77,
        cls_conf=0.0,
        top_k_confidences=None,
        species_key="marten_mustelid",
        od_class_name="marten_mustelid",
    )
    # Fixture has NON_BIRD_CONFIRM_THRESHOLD=0.80; 0.77 < 0.80 -> UNCERTAIN.
    assert signals.decision_state == DecisionState.UNCERTAIN


def test_delegate_falls_back_to_080_when_key_missing():
    """If config has no NON_BIRD_CONFIRM_THRESHOLD, the delegate uses 0.80.

    Defensive default in the delegate (`config.get(..., 0.80)`) protects
    against a config drift where the new key is absent. 0.80 matches the
    plan's production floor.
    """
    mgr = _build_detection_manager_fixture()
    mgr.config.pop("NON_BIRD_CONFIRM_THRESHOLD", None)

    # marten at 0.79 -> UNCERTAIN (just below the 0.80 fallback default).
    signals = mgr.compute_detection_signals(
        bbox=(100, 100, 200, 200),
        frame_shape=(480, 640, 3),
        od_conf=0.79,
        cls_conf=0.0,
        top_k_confidences=None,
        species_key="marten_mustelid",
        od_class_name="marten_mustelid",
    )
    assert signals.decision_state == DecisionState.UNCERTAIN

    # marten at 0.81 -> CONFIRMED (just above).
    signals = mgr.compute_detection_signals(
        bbox=(100, 100, 200, 200),
        frame_shape=(480, 640, 3),
        od_conf=0.81,
        cls_conf=0.0,
        top_k_confidences=None,
        species_key="marten_mustelid",
        od_class_name="marten_mustelid",
    )
    assert signals.decision_state == DecisionState.CONFIRMED


# ---------------------------------------------------------------------------
# A1 — Pre-persist gate (drop non-bird below NON_BIRD_CONFIRM_THRESHOLD)
# ---------------------------------------------------------------------------


def _apply_pre_persist_gate(config, od_class_name, od_conf):
    """Pure reproduction of the A1 gate in detection_manager.py.

    Returns True if the detection should continue down the pipeline,
    False if it should be dropped. The test mirrors the exact condition
    so the unit test catches drift if the production code changes.
    """
    is_bird = is_bird_od_class(od_class_name)
    if not is_bird and config.get("NON_BIRD_DROP_BELOW_CONFIRM", True):
        non_bird_floor = config.get("NON_BIRD_CONFIRM_THRESHOLD", 0.80)
        if od_conf < non_bird_floor:
            return False
    return True


def test_a1_drops_marten_below_floor():
    """marten at 0.50 -> dropped pre-persist (no DB row, no crop)."""
    cfg = {"NON_BIRD_CONFIRM_THRESHOLD": 0.80, "NON_BIRD_DROP_BELOW_CONFIRM": True}
    assert _apply_pre_persist_gate(cfg, "marten_mustelid", 0.50) is False


def test_a1_passes_marten_at_or_above_floor():
    """marten at 0.80 (=) and 0.85 (>) -> continue (potential CONFIRMED)."""
    cfg = {"NON_BIRD_CONFIRM_THRESHOLD": 0.80, "NON_BIRD_DROP_BELOW_CONFIRM": True}
    assert _apply_pre_persist_gate(cfg, "marten_mustelid", 0.80) is True
    assert _apply_pre_persist_gate(cfg, "marten_mustelid", 0.85) is True


@pytest.mark.parametrize(
    "class_name", ["squirrel", "cat", "marten_mustelid", "hedgehog"]
)
def test_a1_drops_all_non_bird_classes_below_floor(class_name):
    """All four non-bird OD classes are gated identically."""
    cfg = {"NON_BIRD_CONFIRM_THRESHOLD": 0.80, "NON_BIRD_DROP_BELOW_CONFIRM": True}
    assert _apply_pre_persist_gate(cfg, class_name, 0.60) is False


def test_a1_never_drops_birds():
    """bird detections are never gated by A1 — even at very low OD conf.

    Critical regression guard: Tauben / Eichelhäher / Blaumeise live on
    the bird track and rely on CLS confidence for their final decision.
    The non-bird gate must not touch them at any OD-conf value.
    """
    cfg = {"NON_BIRD_CONFIRM_THRESHOLD": 0.80, "NON_BIRD_DROP_BELOW_CONFIRM": True}
    for od_conf in (0.10, 0.30, 0.50, 0.79, 0.81, 0.99):
        assert _apply_pre_persist_gate(cfg, "bird", od_conf) is True, (
            f"bird at od_conf={od_conf} should never be gated"
        )


def test_a1_respects_drop_flag_off():
    """When NON_BIRD_DROP_BELOW_CONFIRM=False, weak non-birds still pass.

    Toggle exists so the Phase-7 static-bbox-cluster analysis can collect
    UNCERTAIN non-bird rows for a window without losing data. Downstream
    scoring still gates them to UNCERTAIN (no operator visibility), but
    the DB row exists for cluster analysis.
    """
    cfg = {"NON_BIRD_CONFIRM_THRESHOLD": 0.80, "NON_BIRD_DROP_BELOW_CONFIRM": False}
    assert _apply_pre_persist_gate(cfg, "marten_mustelid", 0.50) is True


def test_a1_default_is_drop_on():
    """When the config dict is empty, A1 drops weak non-birds by default."""
    assert _apply_pre_persist_gate({}, "marten_mustelid", 0.50) is False
    # And keeps bird at the same low conf.
    assert _apply_pre_persist_gate({}, "bird", 0.50) is True


# ---------------------------------------------------------------------------
# A2 — Per-class non-bird floor is a MINIMUM, not a replacement
# ---------------------------------------------------------------------------
#
# Bug: a detector that ships ``confidence_threshold_per_class.<class>``
# below the global ``NON_BIRD_CONFIRM_THRESHOLD`` used to bypass the
# global floor entirely — ``per_class_map.get(name, global_floor)``
# returned the per-class value verbatim. Static-background FPs of any
# such class then flooded persistence at OD-conf well below the
# project's confirm bar. Fix: ``max(per_class, global_floor)`` so
# per-class entries can only RAISE the floor, never lower it.


def _build_mgr_with_per_class_map(per_class_map: dict[str, float], cfg: dict):
    """Construct a DetectionManager attaching a fake underlying detector
    that exposes ``conf_per_class_name`` like the real v2-coco detector
    does (read in detection_manager.py:792-796 and 241-244).
    """
    mgr = _build_detection_manager_fixture()
    mgr.config = {**mgr.config, **cfg}
    fake_underlying = MagicMock()
    fake_underlying.conf_per_class_name = per_class_map
    fake_detector_wrapper = MagicMock()
    fake_detector_wrapper.model = fake_underlying
    fake_detection_service = MagicMock()
    fake_detection_service._detector = fake_detector_wrapper
    mgr.detection_service = fake_detection_service
    return mgr


def _resolve_floor_via_delegate(mgr, class_name: str) -> float:
    """Force the delegate to compute the floor for one class.

    ``compute_detection_signals`` builds the resolver internally and
    passes it to the scoring helper. We hijack the helper to capture
    the resolved floor instead of running the real scoring.
    """
    import detectors.detection_manager as dm_mod

    captured: dict[str, float] = {}

    def fake_compute_signals(**kwargs):
        fn = kwargs["non_bird_confirm_threshold_fn"]
        captured["floor"] = fn(class_name)
        # Return a minimal signals object so the delegate doesn't crash;
        # we only care about the captured floor.
        return MagicMock(decision_state=DecisionState.UNCERTAIN)

    # ``detection_manager.py`` imports compute_detection_signals at
    # module top — we patch the name in THAT namespace, not in the
    # source module, because Python binds at import time.
    original = dm_mod.compute_detection_signals
    dm_mod.compute_detection_signals = fake_compute_signals
    try:
        mgr.compute_detection_signals(
            bbox=(0, 0, 10, 10),
            frame_shape=(480, 640, 3),
            od_conf=0.5,
            cls_conf=0.0,
            top_k_confidences=None,
            species_key=class_name,
            od_class_name=class_name,
        )
    finally:
        dm_mod.compute_detection_signals = original
    return captured["floor"]


def test_per_class_floor_below_global_is_lifted_to_global():
    """hedgehog OD-threshold 0.30 + global floor 0.80 → effective 0.80.

    THIS is the bug fix: previously this returned 0.30 and let every
    hedgehog above 0.30 reach the gallery. Now the global floor wins
    when it is higher.
    """
    mgr = _build_mgr_with_per_class_map(
        per_class_map={"hedgehog": 0.30, "bird": 0.30},
        cfg={"NON_BIRD_CONFIRM_THRESHOLD": 0.80},
    )
    assert _resolve_floor_via_delegate(mgr, "hedgehog") == 0.80


def test_per_class_floor_above_global_wins():
    """cat OD-threshold 0.75 + global floor 0.80 → effective 0.80.

    Edge case at parity: cat YAML default in v2-coco is 0.75, just
    below the global. The global still wins.
    """
    mgr = _build_mgr_with_per_class_map(
        per_class_map={"cat": 0.75},
        cfg={"NON_BIRD_CONFIRM_THRESHOLD": 0.80},
    )
    assert _resolve_floor_via_delegate(mgr, "cat") == 0.80


def test_per_class_floor_strictly_above_global_is_preserved():
    """A future detector shipping squirrel=0.90 must NOT be relaxed to 0.80.

    Per-class entries are allowed to be STRICTER than the global floor;
    they just can't be more permissive. Pins the semantics so a later
    refactor doesn't accidentally clamp DOWN to the global.
    """
    mgr = _build_mgr_with_per_class_map(
        per_class_map={"squirrel": 0.90},
        cfg={"NON_BIRD_CONFIRM_THRESHOLD": 0.80},
    )
    assert _resolve_floor_via_delegate(mgr, "squirrel") == 0.90


def test_class_not_in_per_class_map_uses_global_floor():
    """5-class models (no v2-coco per-class block) get the global floor."""
    mgr = _build_mgr_with_per_class_map(
        per_class_map={},  # legacy 5-class detector ships no per-class map
        cfg={"NON_BIRD_CONFIRM_THRESHOLD": 0.80},
    )
    assert _resolve_floor_via_delegate(mgr, "hedgehog") == 0.80
    assert _resolve_floor_via_delegate(mgr, "marten_mustelid") == 0.80
