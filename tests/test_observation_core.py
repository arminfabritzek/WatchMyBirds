"""Tests for gallery_core.group_detections_into_observations().

Tests the in-memory spatio-temporal clustering logic used by the
observation-based gallery view (Issue #12).
"""

from core.gallery_core import group_detections_into_observations


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
    """Nearby detections (same species, close time, close bbox) → 1 observation."""
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


def test_group_detections_time_gap_splits():
    """Large time gap (>60s) → 2 separate observations."""
    dets = [
        _det(1, "20260101_120000"),
        _det(2, "20260101_120200"),  # 2 min gap
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


def test_cover_picks_best_score():
    """Cover detection = highest score, not first or last."""
    dets = [
        _det(1, "20260101_120000", score=0.70),
        _det(2, "20260101_120020", score=0.95),
        _det(3, "20260101_120040", score=0.80),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 1
    assert obs[0]["cover_detection_id"] == 2
    assert obs[0]["best_score"] == 0.95


def test_cover_tiebreak_by_recency():
    """Same score → cover is the most recent detection."""
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
        _det(2, "20260101_130000"),  # later, separate visit
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 2
    # First observation should be the later one
    assert obs[0]["start_time"] == "20260101_130000"
    assert obs[1]["start_time"] == "20260101_100000"


def test_bbox_scale_change_splits_observation():
    """Drastically different bbox size → separate observations."""
    dets = [
        _det(1, "20260101_120000", bbox=(0.10, 0.10, 0.30, 0.30)),
        _det(2, "20260101_120020", bbox=(0.12, 0.12, 0.05, 0.05)),
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 2


def test_parallel_same_species_stays_separate():
    """Two birds of same species at different positions → 2 observations."""
    dets = [
        _det(1, "20260101_120000", bbox=(0.10, 0.10, 0.16, 0.16)),  # Bird A
        _det(2, "20260101_120005", bbox=(0.70, 0.70, 0.16, 0.16)),  # Bird B
        _det(3, "20260101_120010", bbox=(0.11, 0.11, 0.16, 0.16)),  # Bird A
        _det(4, "20260101_120015", bbox=(0.69, 0.69, 0.16, 0.16)),  # Bird B
    ]
    obs = group_detections_into_observations(dets)
    assert len(obs) == 2
    assert sorted(o["photo_count"] for o in obs) == [2, 2]


def test_microsecond_timestamps_compute_duration():
    """Real DB timestamps (YYYYMMDD_HHMMSS_ffffff) must parse correctly.

    Previous bug: `_ts_to_epoch` failed on the _ffffff suffix, returning 0.0
    for every detection, making all durations 0.
    """
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
