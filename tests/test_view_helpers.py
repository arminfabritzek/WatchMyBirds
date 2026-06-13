from __future__ import annotations

from web import view_helpers


def test_common_names_refresh_mutates_in_place():
    view_helpers.init_common_names("DE")
    live = view_helpers.COMMON_NAMES
    de_len = len(live)
    assert de_len > 0
    view_helpers.refresh_common_names("NO")
    # same dict object, contents swapped
    assert view_helpers.COMMON_NAMES is live
    assert len(view_helpers.COMMON_NAMES) > 0


def test_get_species_key_local_prefers_explicit_then_override():
    assert (
        view_helpers.get_species_key({"species_key": "Erithacus_rubecula"})
        == "Erithacus_rubecula"
    )
    assert (
        view_helpers.get_species_key({"manual_species_override": "Parus_major"})
        == "Parus_major"
    )
    assert view_helpers.get_species_key(None) == view_helpers.UNKNOWN_SPECIES_KEY


def test_compute_auto_rating_buckets():
    # high visual score -> 4
    assert view_helpers.compute_auto_rating(0.9, 0.9, 0.3, 0.3) == 4
    # tiny bbox penalty pushes low
    assert view_helpers.compute_auto_rating(0.1, 0.1, 0.01, 0.01) == 1


def test_bbox_touches_edge():
    assert (
        view_helpers.bbox_touches_edge(
            {"bbox_x": 0.0, "bbox_y": 0.5, "bbox_w": 0.1, "bbox_h": 0.1}
        )
        is True
    )
    assert (
        view_helpers.bbox_touches_edge(
            {"bbox_x": 0.4, "bbox_y": 0.4, "bbox_w": 0.1, "bbox_h": 0.1}
        )
        is False
    )


def test_date_iso_from_timestamp():
    assert view_helpers.date_iso_from_timestamp("20260613_120000") == "2026-06-13"
    assert view_helpers.date_iso_from_timestamp("") == ""


# --- DB-read page-data helpers (promoted from the factory closure) -----------


class _FakeConn:
    """Context-manager stand-in for db_service.closing_connection()."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_get_detections_for_date_maps_rows_to_dicts(monkeypatch):
    sentinel_conn = _FakeConn()
    captured = {}

    monkeypatch.setattr(
        view_helpers.db_service, "closing_connection", lambda: sentinel_conn
    )

    def fake_fetch(conn, date_str_iso, order_by):
        captured["conn"] = conn
        captured["date"] = date_str_iso
        captured["order_by"] = order_by
        return [{"detection_id": 1}, {"detection_id": 2}]

    monkeypatch.setattr(
        view_helpers.db_service, "fetch_detections_for_gallery", fake_fetch
    )

    result = view_helpers.get_detections_for_date("2026-06-13")

    assert result == [{"detection_id": 1}, {"detection_id": 2}]
    assert captured["conn"] is sentinel_conn
    assert captured["date"] == "2026-06-13"
    assert captured["order_by"] == "time"


def test_delete_detections_delegates_to_reject(monkeypatch):
    sentinel_conn = _FakeConn()
    captured = {}

    monkeypatch.setattr(
        view_helpers.db_service, "closing_connection", lambda: sentinel_conn
    )
    monkeypatch.setattr(
        view_helpers.db_service,
        "reject_detections",
        lambda conn, ids: captured.update(conn=conn, ids=ids),
    )

    assert view_helpers.delete_detections([7, 8]) is True
    assert captured["conn"] is sentinel_conn
    assert captured["ids"] == [7, 8]


def test_get_all_detections_returns_empty_on_db_error(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(view_helpers.db_service, "closing_connection", boom)

    assert view_helpers.get_all_detections() == []


def test_get_daily_covers_passes_common_names_and_swallows_errors(monkeypatch):
    captured = {}

    def fake_covers(common_names):
        captured["common_names"] = common_names
        return {"2026-06-13": {"x": 1}}

    monkeypatch.setattr(
        view_helpers.gallery_service, "get_daily_covers", fake_covers
    )

    out = view_helpers.get_daily_covers()
    assert out == {"2026-06-13": {"x": 1}}
    # passes the live COMMON_NAMES module dict through
    assert captured["common_names"] is view_helpers.COMMON_NAMES

    def boom(common_names):
        raise RuntimeError("nope")

    monkeypatch.setattr(view_helpers.gallery_service, "get_daily_covers", boom)
    assert view_helpers.get_daily_covers() == {}


def test_get_daily_species_summary_maps_and_handles_errors(monkeypatch):
    sentinel_conn = _FakeConn()
    monkeypatch.setattr(
        view_helpers.db_service, "closing_connection", lambda: sentinel_conn
    )
    monkeypatch.setattr(
        view_helpers.db_service,
        "fetch_detection_species_summary",
        lambda conn, date_iso: [
            {"species": "Parus_major", "count": 3},
            {"species": "", "count": 9},  # blank species skipped
        ],
    )
    view_helpers.refresh_common_names("DE")
    summary = view_helpers.get_daily_species_summary("2026-06-13")
    assert summary == [
        {
            "species": "Parus_major",
            "common_name": view_helpers.COMMON_NAMES.get(
                "Parus_major", "Parus major"
            ),
            "count": 3,
        }
    ]

    def boom(conn, date_iso):
        raise RuntimeError("query failed")

    monkeypatch.setattr(
        view_helpers.db_service, "fetch_detection_species_summary", boom
    )
    assert view_helpers.get_daily_species_summary("2026-06-13") == []


# --- captured-detection page-data helpers ------------------------------------


def test_get_captured_detections_delegates_and_swallows_errors(monkeypatch):
    monkeypatch.setattr(
        view_helpers.gallery_service,
        "get_all_detections",
        lambda: [{"detection_id": 5}],
    )
    assert view_helpers.get_captured_detections() == [{"detection_id": 5}]

    def boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(view_helpers.gallery_service, "get_all_detections", boom)
    assert view_helpers.get_captured_detections() == []


def test_get_captured_detections_by_date_groups_and_skips_short_ts(monkeypatch):
    rows = [
        {"image_timestamp": "20260613_080000", "detection_id": 1},
        {"image_timestamp": "20260613_090000", "detection_id": 2},
        {"image_timestamp": "20260612_120000", "detection_id": 3},
        {"image_timestamp": "123", "detection_id": 4},  # too short -> skipped
    ]
    monkeypatch.setattr(
        view_helpers.gallery_service, "get_all_detections", lambda: rows
    )

    grouped = view_helpers.get_captured_detections_by_date()

    assert set(grouped.keys()) == {"2026-06-13", "2026-06-12"}
    assert [d["detection_id"] for d in grouped["2026-06-13"]] == [1, 2]
    assert [d["detection_id"] for d in grouped["2026-06-12"]] == [3]


# --- stream-media page-data helpers ------------------------------------------


def test_format_stream_timestamp():
    assert (
        view_helpers.format_stream_timestamp("20260613_142500")
        == "13.06.2026 14:25"
    )
    assert view_helpers.format_stream_timestamp("") == "Unknown"
    assert view_helpers.format_stream_timestamp("not-a-timestamp") == "Unknown"


def test_pick_cover_for_group_delegates_to_gallery_core(monkeypatch):
    import core.gallery_core as gallery_core

    candidates = [{"detection_id": 1}, {"detection_id": 2}]
    monkeypatch.setattr(
        gallery_core, "pick_cover_for_group", lambda c: c[-1]
    )
    assert view_helpers.pick_cover_for_group(candidates) == {"detection_id": 2}


def test_build_stream_media_payload_none_returns_empty_shape():
    assert view_helpers.build_stream_media_payload(None) == {
        "detection_id": None,
        "display_path": "",
        "gallery_date": "",
        "is_favorite": False,
        "is_gallery_eligible": False,
        "score": 0.0,
    }


def test_build_stream_media_payload_prefers_thumb_then_optimized():
    thumb = view_helpers.build_stream_media_payload(
        {
            "detection_id": 9,
            "thumbnail_path_virtual": "a/b.webp",
            "relative_path": "orig/x.webp",
            "image_timestamp": "20260613_120000",
            "is_favorite": 1,
            "is_gallery_eligible": 0,
            "score": 0.5,
        }
    )
    assert thumb["display_path"] == "/uploads/derivatives/thumbs/a/b.webp"
    assert thumb["gallery_date"] == "2026-06-13"
    assert thumb["is_favorite"] is True
    assert thumb["is_gallery_eligible"] is False
    assert thumb["score"] == 0.5

    optimized = view_helpers.build_stream_media_payload(
        {"detection_id": 10, "relative_path": "orig/x.webp"}
    )
    assert optimized["display_path"] == "/uploads/derivatives/optimized/orig/x.webp"


# --- best-species board cluster ----------------------------------------------


def test_enrich_species_board_shapes_items_and_assigns_colours(monkeypatch):
    monkeypatch.setattr(
        view_helpers, "assign_species_colours", lambda keys: {k: 7 for k in keys}
    )
    monkeypatch.setattr(view_helpers, "get_common_name", lambda k: f"name:{k}")

    board = {
        "featured": [
            {
                "species_key": "Parus_major",
                "visit_count": 4,
                "last_seen_timestamp": "20260613_120000",
                "is_favorite_available": True,
                "best_cover_score": 0.8,
                "primary_detection": {
                    "detection_id": 1,
                    "relative_path": "p/x.webp",
                    "image_timestamp": "20260613_120000",
                },
                "story_detections": [
                    {"detection_id": 2, "relative_path": "p/y.webp"},
                ],
            }
        ],
        "grid": [],
    }

    out = view_helpers.enrich_species_board(board)

    item = out["featured"][0]
    assert item["species_key"] == "Parus_major"
    assert item["common_name"] == "name:Parus_major"
    assert item["last_seen_display"] == "13.06.2026 12:00"
    assert item["detection_id"] == 1
    assert item["species_colour"] == 7
    assert item["story_frames"][0]["detection_id"] == 2
    assert item["story_frames"][0]["species_colour"] == 7


def test_fetch_best_species_pools_groups_by_species_and_dedupes_modals(
    monkeypatch,
):
    monkeypatch.setattr(
        view_helpers.db_service, "closing_connection", lambda: _FakeConn()
    )
    rows = [
        {"species_key": "Parus_major", "detection_id": 1, "visit_count": 2},
        {"species_key": "Parus_major", "detection_id": 2, "visit_count": 2},
        {"species_key": "Erithacus_rubecula", "detection_id": 3, "visit_count": 1},
        {"species_key": "Parus_major", "detection_id": 1, "visit_count": 2},  # dupe id
    ]
    monkeypatch.setattr(
        view_helpers.db_service,
        "fetch_species_story_board_candidates",
        lambda conn, **kw: rows,
    )

    pools, modal_rows = view_helpers.fetch_best_species_pools()

    # ordered by first appearance
    assert [p["species_key"] for p in pools] == [
        "Parus_major",
        "Erithacus_rubecula",
    ]
    # Parus_major pool accumulated all its candidates (incl. dupe row)
    assert len(pools[0]["candidates"]) == 3
    # modal rows de-duplicated by detection_id
    assert sorted(r["detection_id"] for r in modal_rows) == [1, 2, 3]


def test_render_best_species_board_splits_featured_and_grid(monkeypatch):
    import core.gallery_core as gallery_core

    # deterministic frame chooser: primary = first candidate, no extra frames
    monkeypatch.setattr(
        gallery_core,
        "_choose_story_board_frames",
        lambda candidates, rng, frame_count: (
            candidates[0] if candidates else None,
            [],
        ),
    )

    pools = [
        {
            "species_key": f"S{i}",
            "visit_count": i,
            "last_seen_timestamp": "",
            "best_cover_score": 0.0,
            "is_favorite_available": False,
            "candidates": [{"detection_id": i}],
        }
        for i in range(5)
    ]

    board = view_helpers.render_best_species_board(
        pools, total_limit=4, featured_count=2
    )

    assert [it["species_key"] for it in board["featured"]] == ["S0", "S1"]
    assert [it["species_key"] for it in board["grid"]] == ["S2", "S3"]
    assert board["featured"][0]["primary_detection"] == {"detection_id": 0}
