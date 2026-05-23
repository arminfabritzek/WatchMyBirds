"""Tests for Filter B2 — same-bird burst suppression.

Covers the per-bbox dedup gate in
:class:`detectors.detection_manager.DetectionManager`:

- :meth:`_bbox_iou` returns 0 for non-overlapping and 1 for identical.
- :meth:`_same_bird_burst_admit` blocks duplicates within the window,
  expires entries outside it, gates on species_key when both sides
  supply one, and gates on bbox-overlap alone when either side leaves
  species_key empty.
- The config-disable knob (``SAME_BIRD_BURST_WINDOW_SECONDS <= 0``)
  returns True unconditionally.
"""

import time
from collections import deque
from unittest.mock import MagicMock

from detectors.detection_manager import DetectionManager


def _make_stub_manager(
    window_seconds: float = 15.0,
    iou_thr: float = 0.6,
) -> DetectionManager:
    """Build a DetectionManager-like stub without running __init__.

    The methods under test only touch ``self.config``, the deque, and
    two counters — no service-layer wiring needed.
    """
    mgr = DetectionManager.__new__(DetectionManager)
    mgr.config = {
        "SAME_BIRD_BURST_WINDOW_SECONDS": window_seconds,
        "SAME_BIRD_BURST_IOU": iou_thr,
    }
    mgr._recent_admissions = deque()
    mgr._same_bird_skipped_total = 0
    mgr._same_bird_skipped_last_log = time.monotonic()
    return mgr


# --- _bbox_iou --------------------------------------------------------


def test_iou_identical_boxes_is_one() -> None:
    assert DetectionManager._bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero() -> None:
    assert DetectionManager._bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap_is_between_zero_and_one() -> None:
    # Two 10x10 boxes overlapping by 5x5 at the corner.
    # inter = 25, union = 100 + 100 - 25 = 175, IoU = 25/175 ≈ 0.143
    iou = DetectionManager._bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.14 < iou < 0.15


def test_iou_zero_area_box_is_zero() -> None:
    assert DetectionManager._bbox_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_iou_high_overlap_above_threshold() -> None:
    # 10x10 vs 11x11 shifted by (1, 1) — most of the box overlaps.
    iou = DetectionManager._bbox_iou((0, 0, 10, 10), (1, 1, 11, 11))
    assert iou > 0.6


# --- _same_bird_burst_admit: first admission always wins --------------


def test_first_admission_returns_true_and_records() -> None:
    mgr = _make_stub_manager()
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is True
    assert len(mgr._recent_admissions) == 1


# --- block paths ------------------------------------------------------


def test_same_bbox_same_species_blocks() -> None:
    mgr = _make_stub_manager()
    mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major")
    # Identical bbox, identical species — must block.
    assert (
        mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major") is False
    )
    assert mgr._same_bird_skipped_total == 1


def test_overlapping_bbox_same_species_blocks() -> None:
    mgr = _make_stub_manager(iou_thr=0.6)
    mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major")
    # Slight shift but >0.6 IoU.
    assert (
        mgr._same_bird_burst_admit((5, 5, 100, 100), "Parus_major") is False
    )


def test_overlapping_bbox_empty_species_blocks() -> None:
    """Empty species_key on EITHER side means 'block on bbox alone'.

    Matches the bird-track call site where CLS has not yet run and we
    want the dedup to gate purely on bbox overlap.
    """
    mgr = _make_stub_manager()
    mgr._same_bird_burst_admit((0, 0, 100, 100), "")
    assert mgr._same_bird_burst_admit((0, 0, 100, 100), "") is False
    assert mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major") is False


def test_empty_species_blocks_against_named_recent() -> None:
    mgr = _make_stub_manager()
    mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major")
    # CLS-empty new admission overlaps a named recent — should block.
    assert mgr._same_bird_burst_admit((0, 0, 100, 100), "") is False


# --- pass-through paths -----------------------------------------------


def test_overlapping_bbox_different_species_passes() -> None:
    """If both sides supply different species, the dedup does NOT apply.

    Real-world case: a Sperling and a Meise at the same feeder slot in
    sequence — same bbox, different species, both deserve to land in DB.
    """
    mgr = _make_stub_manager()
    mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major")
    assert (
        mgr._same_bird_burst_admit((0, 0, 100, 100), "Cyanistes_caeruleus")
        is True
    )


def test_disjoint_bbox_passes_regardless_of_species() -> None:
    mgr = _make_stub_manager()
    mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major")
    # Different region of the frame, same species — different bird.
    assert (
        mgr._same_bird_burst_admit((500, 500, 600, 600), "Parus_major") is True
    )


def test_iou_just_below_threshold_passes() -> None:
    mgr = _make_stub_manager(iou_thr=0.6)
    mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major")
    # Shifted enough that IoU < 0.6.
    assert (
        mgr._same_bird_burst_admit((5, 5, 15, 15), "Parus_major") is True
    )


# --- window expiry ----------------------------------------------------


def test_expired_admission_is_evicted(monkeypatch) -> None:
    mgr = _make_stub_manager(window_seconds=15.0)

    fake_clock = MagicMock()
    fake_clock.return_value = 1000.0
    monkeypatch.setattr(time, "monotonic", fake_clock)

    mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major")
    assert len(mgr._recent_admissions) == 1

    # Jump past the window — first admission must be evicted.
    fake_clock.return_value = 1020.0
    assert mgr._same_bird_burst_admit((0, 0, 100, 100), "Parus_major") is True
    # Old entry is gone, new one in.
    assert len(mgr._recent_admissions) == 1
    assert mgr._recent_admissions[0][0] == 1020.0


# --- disable knob -----------------------------------------------------


def test_disabled_when_window_zero_admits_everything() -> None:
    mgr = _make_stub_manager(window_seconds=0.0)
    # Every call is True, nothing gets recorded.
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is True
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is True
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is True
    assert len(mgr._recent_admissions) == 0


def test_disabled_when_window_negative_admits_everything() -> None:
    mgr = _make_stub_manager(window_seconds=-5.0)
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "x") is True
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "x") is True


# --- config malformed fall-back to defaults ---------------------------


def test_malformed_window_falls_back_to_default() -> None:
    mgr = _make_stub_manager()
    mgr.config["SAME_BIRD_BURST_WINDOW_SECONDS"] = "not_a_number"
    # Default = 15.0, so first call admits and second on same bbox blocks.
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is True
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is False


def test_malformed_iou_falls_back_to_default() -> None:
    mgr = _make_stub_manager()
    mgr.config["SAME_BIRD_BURST_IOU"] = "not_a_number"
    # Default = 0.6 — identical bboxes (IoU=1.0) still block.
    mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major")
    assert mgr._same_bird_burst_admit((0, 0, 10, 10), "Parus_major") is False
