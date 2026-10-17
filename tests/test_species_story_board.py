"""Tests for the Best of Species story-board aggregation."""

import random

import pytest

from core.gallery_core import build_species_story_board


def _det(
    det_id: int,
    ts: str,
    species: str,
    *,
    score: float = 0.8,
    favorite: bool = False,
    bbox: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.2),
) -> dict:
    return {
        "detection_id": det_id,
        "image_timestamp": ts,
        "species_key": species,
        "cls_class_name": species,
        "od_class_name": species,
        "score": score,
        "is_favorite": 1 if favorite else 0,
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
    }


def test_story_board_ranks_by_visit_count_then_last_seen():
    detections = [
        _det(1, "20260102_120000", "Parus_major", score=0.7),
        _det(2, "20260102_120030", "Parus_major", score=0.9),
        _det(3, "20260103_120000", "Parus_major", score=0.8),
        _det(4, "20260104_120000", "Erithacus_rubecula", score=0.95),
        _det(5, "20260104_120090", "Erithacus_rubecula", score=0.85),
        _det(6, "20260105_120000", "Cyanistes_caeruleus", score=0.88),
    ]

    board = build_species_story_board(
        detections,
        since_timestamp="20260101_000000",
        total_limit=3,
        featured_count=1,
        excluded_species={"Unknown_species"},
        rng=random.Random(1),
    )

    species_order = [board["featured"][0]["species_key"]] + [
        item["species_key"] for item in board["grid"]
    ]
    assert species_order == [
        "Erithacus_rubecula",
        "Parus_major",
        "Cyanistes_caeruleus",
    ]
    assert board["featured"][0]["visit_count"] == 2


def test_story_board_excludes_unknown_and_old_rows():
    detections = [
        _det(1, "20251231_235959", "Parus_major"),
        _det(2, "20260101_120000", "Unknown_species"),
        _det(3, "20260102_120000", "Fringilla_coelebs"),
    ]

    board = build_species_story_board(
        detections,
        since_timestamp="20260101_000000",
        total_limit=4,
        featured_count=2,
        excluded_species={"Unknown_species"},
        rng=random.Random(2),
    )

    assert [item["species_key"] for item in board["featured"]] == ["Fringilla_coelebs"]
    assert board["grid"] == []


def test_story_board_merges_space_and_underscore_species_labels():
    detections = [
        _det(1, "20260103_120000", "Parus major", score=0.7),
        _det(2, "20260103_120030", "Parus_major", score=0.9),
    ]

    board = build_species_story_board(
        detections,
        total_limit=3,
        featured_count=1,
        excluded_species={"Unknown_species"},
        rng=random.Random(3),
    )

    assert board["featured"][0]["species_key"] == "Parus_major"
    assert board["featured"][0]["visit_count"] == 1


def test_story_board_primary_uses_favorites_from_same_observation():
    detections = [
        _det(1, "20260103_120000", "Parus_major", score=0.95, favorite=False),
        _det(2, "20260103_120002", "Parus_major", score=0.65, favorite=True),
        _det(3, "20260103_120004", "Parus_major", score=0.75, favorite=True),
        _det(4, "20260103_120006", "Parus_major", score=0.72, favorite=False),
    ]

    primary_ids = set()
    for seed in range(8):
        board = build_species_story_board(
            detections,
            since_timestamp="20260101_000000",
            total_limit=1,
            featured_count=1,
            excluded_species=set(),
            rng=random.Random(seed),
        )
        primary_ids.add(board["featured"][0]["primary_detection"]["detection_id"])

    assert primary_ids == {3}


def test_story_board_single_favorite_still_allows_rotation():
    detections = [
        _det(1, "20260103_120000", "Turdus_merula", score=0.95, favorite=True),
        _det(2, "20260103_120002", "Turdus_merula", score=0.93, favorite=False),
        _det(3, "20260103_120004", "Turdus_merula", score=0.91, favorite=False),
    ]

    primary_ids = []
    secondary_ids = set()
    for seed in range(12):
        board = build_species_story_board(
            detections,
            since_timestamp="20260101_000000",
            total_limit=1,
            featured_count=1,
            excluded_species=set(),
            rng=random.Random(seed),
        )
        primary_ids.append(board["featured"][0]["primary_detection"]["detection_id"])
        secondary_ids.update(
            row["detection_id"] for row in board["featured"][0]["story_detections"][1:]
        )

    assert set(primary_ids) == {1}
    assert secondary_ids == {2, 3}


@pytest.mark.parametrize(
    "available,expected", [(0, 0), (1, 1), (2, 2), (3, 2), (4, 4), (7, 4)]
)
def test_homepage_frame_budget_uses_complete_rows(
    available: int, expected: int
) -> None:
    from core.gallery_core import _choose_story_board_frames

    rows = [
        _det(i + 1, f"20260103_1200{i:02d}", "Parus_major") for i in range(available)
    ]
    primary, frames = _choose_story_board_frames(rows, rng=random.Random(1))
    assert len(frames) == expected
    assert primary == (frames[0] if frames else None)


def test_homepage_frames_do_not_repeat_a_source_image() -> None:
    from core.gallery_core import _choose_story_board_frames

    rows = [
        _det(i, f"20260103_1200{i:02d}", "Parus_major", favorite=i == 2)
        for i in range(1, 7)
    ]
    for row in rows:
        row["original_name"] = f"frame-{(row['detection_id'] - 1) // 2}.jpg"
    primary, frames = _choose_story_board_frames(rows, rng=random.Random(1))
    assert primary["detection_id"] == 2
    assert len(frames) == 2
    assert len({row["original_name"] for row in frames}) == len(frames)


def test_homepage_secondary_species_receive_four_frames() -> None:
    rows = [_det(i, f"20260103_1200{i:02d}", "Parus_major") for i in range(1, 5)]
    board = build_species_story_board(rows, featured_count=0, rng=random.Random(1))
    assert len(board["grid"][0]["story_detections"]) == 4


def test_story_pool_does_not_spend_its_budget_on_duplicate_detections() -> None:
    from core.gallery_core import _build_story_board_candidate_pool

    rows = [_det(i, f"20260103_1200{i:02d}", "Parus_major") for i in range(1, 21)]
    for row in rows:
        row["original_name"] = (
            "shared.jpg" if row["detection_id"] < 17 else f"{row['detection_id']}.jpg"
        )
    pool = _build_story_board_candidate_pool(rows)
    assert len(pool) == 5
    assert len({row["original_name"] for row in pool}) == 5
