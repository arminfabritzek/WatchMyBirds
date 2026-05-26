"""Regression tests: ``build_bird_events`` resolves species + epoch once per detection.

The plan that introduced these tests (PERFORMANCE list-route caching,
2026-05-27) measured a profile where ``_resolve_detection_species``
appeared in 29 858 calls and ``_ts_to_epoch`` in the inner builtin
dict.get loop with 1 643 136 hits. Those numbers came from a build
that re-resolved per inner loop; today's ``_collect_items`` caches
both resolutions onto the item dict so downstream code reads
``item["species_resolved"]`` / ``item["epoch"]`` instead of
recomputing.

These tests pin that property in place: if a future refactor moves a
``_resolve_detection_species(...)`` call into the per-event loop (or
calls ``_ts_to_epoch(timestamp)`` somewhere besides ``_collect_items``),
the call count exceeds the number of input detections and these tests
fail.
"""

from __future__ import annotations

from unittest.mock import patch

from core import events


def _detection(detection_id: int, timestamp: str, species_key: str | None = None) -> dict:
    return {
        "detection_id": detection_id,
        "timestamp": timestamp,
        "filename": f"img_{detection_id}.jpg",
        "manual_species_override": None,
        "species_source": "classifier" if species_key else None,
        "species_key": species_key,
        "cls_class_name": species_key,
        "od_class_name": None,
        "bbox_x": 0.5,
        "bbox_y": 0.5,
        "bbox_w": 0.1,
        "bbox_h": 0.1,
        "sibling_detection_count": 1,
        "source_id": "cam-1",
        "context_only": False,
    }


def test_resolve_detection_species_called_once_per_detection():
    # ids start at 1: _collect_items filters detection_id <= 0 as a
    # missing-ID sentinel, which would skew the call count by one.
    detections = [
        _detection(i, f"20260527_12{i:02d}00", species_key="Parus_major")
        for i in range(1, 11)
    ]

    with patch(
        "core.events._resolve_detection_species",
        wraps=events._resolve_detection_species,
    ) as spy:
        result = events.build_bird_events(detections)

    assert spy.call_count == len(detections), (
        f"expected exactly {len(detections)} calls (one per detection), "
        f"got {spy.call_count} — someone is re-resolving species in an inner loop"
    )
    # Smoke: should still build at least one event from the input
    assert result, "expected at least one event from valid detections"


def test_ts_to_epoch_called_once_per_detection():
    detections = [
        _detection(i, f"20260527_12{i:02d}00", species_key="Parus_major")
        for i in range(1, 11)
    ]

    with patch(
        "core.events._ts_to_epoch",
        wraps=events._ts_to_epoch,
    ) as spy:
        events.build_bird_events(detections)

    assert spy.call_count == len(detections), (
        f"expected exactly {len(detections)} calls (one per detection), "
        f"got {spy.call_count} — someone is re-parsing timestamps in an inner loop"
    )


def test_resolve_called_once_with_mixed_species_and_unknowns():
    """Same guarantee under the harder grouping path (mixed species,
    unknown detections that may attach to known groups)."""
    detections = []
    for i in range(1, 6):
        detections.append(_detection(i, f"20260527_1200{i:02d}", species_key="Parus_major"))
    for i in range(5):
        detections.append(_detection(100 + i, f"20260527_1201{i:02d}", species_key=None))
    for i in range(5):
        detections.append(
            _detection(200 + i, f"20260527_1202{i:02d}", species_key="Cyanistes_caeruleus")
        )

    with patch(
        "core.events._resolve_detection_species",
        wraps=events._resolve_detection_species,
    ) as resolve_spy, patch(
        "core.events._ts_to_epoch",
        wraps=events._ts_to_epoch,
    ) as epoch_spy:
        events.build_bird_events(detections)

    assert resolve_spy.call_count == len(detections)
    assert epoch_spy.call_count == len(detections)


def test_resolve_called_zero_times_on_empty_input():
    with patch(
        "core.events._resolve_detection_species",
        wraps=events._resolve_detection_species,
    ) as spy:
        result = events.build_bird_events([])

    assert spy.call_count == 0
    assert result == []
