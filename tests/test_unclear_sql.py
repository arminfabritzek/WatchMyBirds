"""
Tests for the Unclear bucket SQL layer.

The Unclear bucket targets exactly one row shape:
- ``detections.status='active'``
- ``lower(detections.decision_level)='reject'``
- ``images.review_status`` is NULL or ``'untagged'``

Everything else must be excluded — these tests pin that contract so a
later schema change doesn't silently let confirmed_bird or no_bird rows
leak into the Unclear surface.
"""

import sqlite3

from utils.db.unclear import (
    _is_iso_day,
    confirm_unclear_detections,
    fetch_unclear_days,
    fetch_unclear_detection_ids_for_day,
    fetch_unclear_total,
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            status TEXT DEFAULT 'active',
            decision_level TEXT,
            decision_state TEXT,
            raw_species_name TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            species_updated_at TEXT,
            thumbnail_path TEXT
        );
        """
    )
    return conn


def _insert_image(conn, filename, review_status=None):
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        (filename, filename[:15], review_status),
    )


def _insert_detection(
    conn,
    image_filename,
    *,
    status="active",
    decision_level="reject",
    decision_state="confirmed",
    raw_species_name="Parus_major",
    thumbnail_path=None,
):
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename, status, decision_level, decision_state,
            raw_species_name, thumbnail_path
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            image_filename,
            status,
            decision_level,
            decision_state,
            raw_species_name,
            thumbnail_path,
        ),
    )
    # sqlite3 always assigns a rowid on a successful INSERT into a table
    # with AUTOINCREMENT — the stub type is overly conservative.
    assert cur.lastrowid is not None
    return cur.lastrowid


def test_empty_db_returns_empty_results():
    conn = _make_conn()
    assert fetch_unclear_days(conn) == []
    assert fetch_unclear_total(conn) == 0


def test_includes_only_active_reject_untagged():
    """The Unclear surface must filter on three independent predicates."""
    conn = _make_conn()
    # IN bucket: active + reject + untagged
    _insert_image(conn, "20260519_100000_aaa.jpg", review_status="untagged")
    _insert_detection(conn, "20260519_100000_aaa.jpg")

    # NULL review_status counts as untagged too
    _insert_image(conn, "20260519_100001_bbb.jpg", review_status=None)
    _insert_detection(conn, "20260519_100001_bbb.jpg")

    # OUT: rejected detection (already in trash)
    _insert_image(conn, "20260519_100002_ccc.jpg", review_status="untagged")
    _insert_detection(conn, "20260519_100002_ccc.jpg", status="rejected")

    # OUT: decision_level != reject
    _insert_image(conn, "20260519_100003_ddd.jpg", review_status="untagged")
    _insert_detection(
        conn, "20260519_100003_ddd.jpg", decision_level="species"
    )

    # OUT: image in no_bird trash bucket
    _insert_image(conn, "20260519_100004_eee.jpg", review_status="no_bird")
    _insert_detection(conn, "20260519_100004_eee.jpg")

    # OUT: image already confirmed (don't override a human decision)
    _insert_image(
        conn, "20260519_100005_fff.jpg", review_status="confirmed_bird"
    )
    _insert_detection(conn, "20260519_100005_fff.jpg")

    assert fetch_unclear_total(conn) == 2


def test_days_grouped_newest_first_with_species_breakdown():
    conn = _make_conn()
    # Day 19 — two species
    _insert_image(conn, "20260519_100000_aa.jpg", review_status="untagged")
    _insert_detection(conn, "20260519_100000_aa.jpg", raw_species_name="Parus_major")
    _insert_image(conn, "20260519_100001_bb.jpg", review_status="untagged")
    _insert_detection(
        conn, "20260519_100001_bb.jpg", raw_species_name="Cyanistes_caeruleus"
    )
    _insert_image(conn, "20260519_100002_cc.jpg", review_status="untagged")
    _insert_detection(conn, "20260519_100002_cc.jpg", raw_species_name="Parus_major")

    # Day 18 — one detection
    _insert_image(conn, "20260518_080000_xx.jpg", review_status="untagged")
    _insert_detection(conn, "20260518_080000_xx.jpg", raw_species_name="Garrulus_glandarius")

    days = fetch_unclear_days(conn)
    assert [d["day"] for d in days] == ["2026-05-19", "2026-05-18"]

    # Day 19: total 3, two species, Parus dominant
    day19 = days[0]
    assert day19["total_count"] == 3
    species_map = {
        e["raw_species_name"]: e["count"] for e in day19["species_breakdown"]
    }
    assert species_map == {"Parus_major": 2, "Cyanistes_caeruleus": 1}
    assert day19["species_breakdown"][0]["count"] >= day19["species_breakdown"][1]["count"]

    # Day 18: single Garrulus
    day18 = days[1]
    assert day18["total_count"] == 1
    assert day18["species_breakdown"][0]["raw_species_name"] == "Garrulus_glandarius"


def test_sample_limit_caps_thumbnails_not_counts():
    """sample_limit affects only the preview list, never the counts."""
    conn = _make_conn()
    for i in range(15):
        fn = f"20260519_10000{i}_x.jpg"
        _insert_image(conn, fn, review_status="untagged")
        _insert_detection(conn, fn, thumbnail_path=f"thumb_{i}.webp")

    days = fetch_unclear_days(conn, sample_limit=5)
    assert days[0]["total_count"] == 15
    assert len(days[0]["samples"]) == 5


def test_sample_thumbnail_paths_include_date_subfolder():
    """thumbnail_path_virtual must include the ``YYYY-MM-DD/`` prefix.

    Files on disk live under
    ``derivatives/thumbs/2026-05-19/<filename>.webp`` — the static
    route ``/uploads/derivatives/thumbs/<path:filename>`` accepts the
    date-prefixed form directly. Serving just the bare filename would
    trigger the expensive ``regenerate_derivative`` fallback path for
    every tile.
    """
    conn = _make_conn()
    _insert_image(conn, "20260519_120000_aa.jpg", review_status="untagged")
    _insert_detection(
        conn,
        "20260519_120000_aa.jpg",
        thumbnail_path="20260519_120000_aa_crop_1.webp",
    )

    days = fetch_unclear_days(conn)
    sample = days[0]["samples"][0]
    assert sample["thumbnail_path_virtual"] == (
        "2026-05-19/20260519_120000_aa_crop_1.webp"
    )


def test_sample_thumbnail_paths_fallback_when_no_explicit_path():
    """When thumbnail_path is empty, derive from image filename + suffix."""
    conn = _make_conn()
    _insert_image(conn, "20260519_120000_bb.jpg", review_status="untagged")
    _insert_detection(
        conn,
        "20260519_120000_bb.jpg",
        thumbnail_path=None,
    )

    days = fetch_unclear_days(conn)
    sample = days[0]["samples"][0]
    # Falls back to <image_basename>_crop_1.webp, still date-prefixed.
    assert sample["thumbnail_path_virtual"] == (
        "2026-05-19/20260519_120000_bb_crop_1.webp"
    )


def test_fetch_ids_for_day_rejects_malformed_input():
    """Day parameter must be a strict YYYY-MM-DD string to block injection."""
    conn = _make_conn()
    _insert_image(conn, "20260519_100000_aa.jpg", review_status="untagged")
    _insert_detection(conn, "20260519_100000_aa.jpg")

    # Valid input works
    assert len(fetch_unclear_detection_ids_for_day(conn, "2026-05-19")) == 1

    # Malformed inputs must return [] without touching the DB
    for bad in ["", "abc", "2026/05/19", "2026-5-19", "%", "2026-05-1"]:
        assert fetch_unclear_detection_ids_for_day(conn, bad) == []


def test_confirm_promotes_raw_species_to_override_and_flips_decision():
    conn = _make_conn()
    _insert_image(conn, "20260519_100000_aa.jpg", review_status="untagged")
    det_id = _insert_detection(
        conn,
        "20260519_100000_aa.jpg",
        raw_species_name="Cyanistes_caeruleus",
    )

    affected = confirm_unclear_detections(conn, [det_id])
    assert affected == 1

    row = conn.execute(
        "SELECT manual_species_override, species_source, decision_state, decision_level "
        "FROM detections WHERE detection_id = ?",
        (det_id,),
    ).fetchone()
    assert row["manual_species_override"] == "Cyanistes_caeruleus"
    assert row["species_source"] == "manual_bulk_confirm"
    assert row["decision_state"] == "confirmed"
    assert row["decision_level"] == "species"

    # Idempotent: a second confirm finds 0 rows still in reject state
    assert confirm_unclear_detections(conn, [det_id]) == 0


def test_confirm_ignores_rejected_status_detections():
    """A detection that was already soft-deleted to Trash must not be revived."""
    conn = _make_conn()
    _insert_image(conn, "20260519_100000_aa.jpg", review_status="untagged")
    det_id = _insert_detection(
        conn, "20260519_100000_aa.jpg", status="rejected"
    )

    affected = confirm_unclear_detections(conn, [det_id])
    assert affected == 0
    row = conn.execute(
        "SELECT status FROM detections WHERE detection_id = ?", (det_id,)
    ).fetchone()
    assert row["status"] == "rejected"


def test_confirm_empty_list_is_noop():
    conn = _make_conn()
    assert confirm_unclear_detections(conn, []) == 0


def test_is_iso_day_strict():
    assert _is_iso_day("2026-05-19") is True
    assert _is_iso_day("2099-12-31") is True
    # length / separator checks
    assert _is_iso_day("2026-5-19") is False
    assert _is_iso_day("2026/05/19") is False
    assert _is_iso_day("2026-05-19T00") is False
    # range checks
    assert _is_iso_day("1999-12-31") is False
    assert _is_iso_day("2026-13-01") is False
    assert _is_iso_day("2026-00-15") is False
    assert _is_iso_day("2026-05-32") is False
    # type/empty
    assert _is_iso_day("") is False
    assert _is_iso_day(None) is False  # type: ignore[arg-type]
