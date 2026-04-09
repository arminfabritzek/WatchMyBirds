"""Tests for gallery_core observation grouping and summaries.

The legacy 60 s + bbox proximity merge was retired in favour of the
biological event window (30 min, same species, no bbox split). The
assertions in this file have been updated to reflect that policy. The
function under test, ``group_detections_into_observations``, is now a
thin shim around ``core.events.build_bird_events`` that preserves the
legacy dict shape for callers that have not been migrated yet.
"""

from core.gallery_core import (
    group_detections_into_observations,
    summarize_observations,
)


def _det(
    det_id: int,
    ts: str,
    species: str = "parus_major",
    score: float = 0.80,
    bbox: tuple[float, float, float, float] = (0.10, 0.10, 0.20, 0.20),
) -> dict:
    """Build a minimal detection dict."""
    return {
        "detection_id": det_id,
        "image_timestamp": ts,
        "cls_class_name": species,
        "od_class_name": species,
        "score": score,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
    }


def test_group_detections_empty():
    """No detections → empty list."""
    result = group_detections_into_observations([])
    assert result == []


def test_group_detections_single_detection():
    """One detection → one observation with photo_count=1."""
    dets = [_det(1, "20260101_120000")]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["photo_count"] == 1
    assert obs[0]["cover_detection_id"] == 1
    assert obs[0]["duration_sec"] == 0.0


def test_group_detections_basic_grouping():
    """Nearby detections (same species, close time) → 1 observation."""
    dets = [
        _det(1, "20260101_120000", bbox=(0.10, 0.10, 0.20, 0.20)),
        _det(2, "20260101_120030", bbox=(0.12, 0.11, 0.21, 0.21)),
        _det(3, "20260101_120050", bbox=(0.11, 0.10, 0.20, 0.20)),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["photo_count"] == 3
    assert obs[0]["duration_sec"] == 50.0
    assert set(obs[0]["detection_ids"]) == {1, 2, 3}


def test_group_detections_within_event_window_stay_together():
    """Same species, gap below 30 min → still one biological observation."""
    dets = [
        _det(1, "20260101_120000"),
        _det(2, "20260101_120200"),  # 2 min gap, well inside the 30 min window
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["photo_count"] == 2


def test_group_detections_gap_above_event_window_splits():
    """Same species, gap above 30 min → two separate observations."""
    dets = [
        _det(1, "20260101_120000"),
        _det(2, "20260101_124000"),  # 40 min gap
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 2
    assert all(o["photo_count"] == 1 for o in obs)


def test_group_detections_different_species():
    """Different species → separate observations even if close in time."""
    dets = [
        _det(1, "20260101_120000", species="parus_major"),
        _det(2, "20260101_120010", species="erithacus_rubecula"),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 2
    species_set = {o["species"] for o in obs}
    assert species_set == {"parus_major", "erithacus_rubecula"}


def test_cover_uses_most_recent_detection():
    """Cover detection follows the newest image in the observation."""
    dets = [
        _det(1, "20260101_120000", score=0.70),
        _det(2, "20260101_120020", score=0.95),
        _det(3, "20260101_120040", score=0.80),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["cover_detection_id"] == 3
    assert obs[0]["best_score"] == 0.95


def test_cover_tiebreak_by_recency():
    """Same score still keeps the most recent detection as cover."""
    dets = [
        _det(1, "20260101_120000", score=0.90),
        _det(2, "20260101_120020", score=0.90),
        _det(3, "20260101_120040", score=0.90),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["cover_detection_id"] == 3  # most recent


def test_observations_sorted_by_time_desc():
    """Observations are returned sorted by start_time descending."""
    dets = [
        _det(1, "20260101_100000"),
        _det(2, "20260101_130000"),  # 3 hours later, separate event
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 2
    # First observation should be the later one
    assert obs[0]["start_time"] == "20260101_130000"
    assert obs[1]["start_time"] == "20260101_100000"


def test_same_species_at_distinct_positions_merge_into_one_event():
    """Two birds of the same species in the same window collapse into one event.

    Bbox proximity is no longer a split criterion. The biological event
    is "same species, gap <= 30 min". A bbox jump still surfaces in the
    upstream BirdEvent ``fallback_reason`` for the Review surface, but
    the gallery observation shape stays one row.
    """
    dets = [
        _det(1, "20260101_120000", bbox=(0.10, 0.10, 0.16, 0.16)),
        _det(2, "20260101_120005", bbox=(0.70, 0.70, 0.16, 0.16)),
        _det(3, "20260101_120010", bbox=(0.11, 0.11, 0.16, 0.16)),
        _det(4, "20260101_120015", bbox=(0.69, 0.69, 0.16, 0.16)),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["photo_count"] == 4
    assert sorted(obs[0]["detection_ids"]) == [1, 2, 3, 4]


def test_bbox_scale_change_no_longer_splits_event():
    """Drastic bbox size change is no longer a split criterion."""
    dets = [
        _det(1, "20260101_120000", bbox=(0.10, 0.10, 0.30, 0.30)),
        _det(2, "20260101_120020", bbox=(0.12, 0.12, 0.05, 0.05)),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["photo_count"] == 2


def test_microsecond_timestamps_compute_duration():
    """Real DB timestamps (YYYYMMDD_HHMMSS_ffffff) must parse correctly."""
    dets = [
        _det(1, "20260225_084116_427121"),
        _det(2, "20260225_084118_430501"),
        _det(3, "20260225_084120_433865"),
        _det(4, "20260225_084126_443368"),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["photo_count"] == 4
    # 08:41:26 - 08:41:16 = 10 seconds
    assert obs[0]["duration_sec"] == 10.0


def test_summarize_observations_uses_species_key_and_threshold():
    """Summary must align with gallery species resolution and thresholding.

    All four detections in the fixture span 4 minutes, so under the 30
    min biological event window the three Columba_palumbus rows collapse
    into a single event. The Parus_major detection has score 0.05, below
    the 0.10 threshold, so its single-row event is filtered out at the
    observation-best-score stage.
    """
    dets = [
        {
            **_det(
                1,
                "20260101_120000",
                species="wrong_species",
                score=0.90,
            ),
            "species_key": "Columba_palumbus",
        },
        {
            **_det(
                2,
                "20260101_120030",
                species="wrong_species",
                score=0.80,
            ),
            "species_key": "Columba_palumbus",
        },
        {
            **_det(
                3,
                "20260101_120200",
                species="wrong_species",
                score=0.70,
            ),
            "species_key": "Columba_palumbus",
        },
        {
            **_det(
                4,
                "20260101_120400",
                species="parus_major",
                score=0.05,
            ),
            "species_key": "Parus_major",
        },
    ]

    result = summarize_observations(dets, min_score=0.10)

    assert result["summary"]["total_observations"] == 1
    assert result["summary"]["total_detections"] == 3
    assert result["summary"]["species_counts"] == {"Columba_palumbus": 1}
    assert result["summary"]["avg_score"] == 0.8
    assert {det["detection_id"] for det in result["detections"]} == {1, 2, 3}
    assert all(obs["species"] == "Columba_palumbus" for obs in result["observations"])


def test_summarize_observations_keeps_full_event_when_best_score_passes():
    """A single biological event keeps all member detections once it passes the score gate."""
    dets = [
        _det(1, "20260101_120000", score=0.90),
        _det(2, "20260101_120020", score=0.05),
        _det(3, "20260101_120200", score=0.09),
    ]

    result = summarize_observations(dets, min_score=0.10)

    assert result["summary"]["total_observations"] == 1
    assert result["summary"]["total_detections"] == 3
    assert result["summary"]["species_counts"] == {"parus_major": 1}
    # avg_score = (0.90 + 0.05 + 0.09) / 3 ≈ 0.35
    assert result["summary"]["avg_score"] == 0.35
    assert {det["detection_id"] for det in result["detections"]} == {1, 2, 3}
