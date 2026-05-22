"""
Tests for the user-groundtruth query layer.

Three buckets each have a contract — these tests pin every contract so
a future schema change cannot silently:

- cross-contaminate buckets (e.g. let species_review leak into
  confirmed_positives)
- include soft-deleted detections (Trash is explicitly excluded from
  Pipeline-Dev training data)
- include no-op manual overrides (relabel where user re-confirmed the
  same species the model picked — zero training signal)
- include hard-negatives that the user later un-rejected (review_status
  flipped back)

The fixture builds a maximally adversarial dataset: every row carries
columns that could match a *different* bucket, so a SQL slip would
show as a cross-bucket leak.
"""

from __future__ import annotations

import sqlite3

import pytest

from utils.db.user_groundtruth import (
    count_pending_by_bucket,
    fetch_confirmed_positives,
    fetch_favorites,
    fetch_hard_negatives,
    fetch_species_relabels,
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            review_status TEXT,
            review_updated_at TEXT
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT,
            status TEXT DEFAULT 'active',
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_confidence REAL,
            od_class_name TEXT,
            detector_model_version TEXT,
            classifier_model_version TEXT,
            frame_width INTEGER, frame_height INTEGER,
            decision_level TEXT,
            raw_species_name TEXT,
            manual_species_override TEXT,
            species_source TEXT,
            species_updated_at TEXT,
            is_favorite INTEGER DEFAULT 0,
            rating_source TEXT DEFAULT 'auto'
        );
        """
    )
    return conn


def _add_image(
    conn: sqlite3.Connection,
    filename: str,
    review_status: str | None = None,
    review_updated_at: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO images (filename, review_status, review_updated_at) "
        "VALUES (?, ?, ?)",
        (filename, review_status, review_updated_at),
    )


def _add_detection(
    conn: sqlite3.Connection,
    *,
    image_filename: str,
    detection_id: int | None = None,
    status: str = "active",
    decision_level: str | None = None,
    raw_species_name: str | None = None,
    manual_species_override: str | None = None,
    species_source: str | None = None,
    species_updated_at: str | None = None,
    od_confidence: float = 0.75,
    od_class_name: str = "bird",
    detector_model_version: str = "yolox_test_v1",
    classifier_model_version: str = "cls_test_v1",
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2),
    frame_wh: tuple[int, int] = (1920, 1080),
    is_favorite: int = 0,
    rating_source: str = "auto",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO detections (
            detection_id, image_filename, status,
            bbox_x, bbox_y, bbox_w, bbox_h,
            od_confidence, od_class_name,
            detector_model_version, classifier_model_version,
            frame_width, frame_height,
            decision_level, raw_species_name,
            manual_species_override, species_source, species_updated_at,
            is_favorite, rating_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            detection_id, image_filename, status,
            *bbox,
            od_confidence, od_class_name,
            detector_model_version, classifier_model_version,
            *frame_wh,
            decision_level, raw_species_name,
            manual_species_override, species_source, species_updated_at,
            is_favorite, rating_source,
        ),
    )
    new_id = cur.lastrowid
    assert new_id is not None  # INSERT always sets lastrowid; satisfies type checker
    return new_id


# ---------------------------------------------------------------------------
# fetch_hard_negatives
# ---------------------------------------------------------------------------


def test_hard_negatives_returns_active_detections_on_no_bird_images():
    conn = _make_conn()
    _add_image(conn, "img_hn.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    _add_detection(conn, image_filename="img_hn.jpg",
                   decision_level="species_review",
                   raw_species_name="Parus_major")

    rows = fetch_hard_negatives(conn)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "hard_negatives"
    assert rows[0]["od_class_name"] == "bird"
    assert rows[0]["user_action_at"] == "2026-05-22T10:00:00Z"


def test_hard_negatives_excludes_soft_deleted_detections():
    """Trash is out of scope — a deleted detection on a no_bird image
    must NOT be exported."""
    conn = _make_conn()
    _add_image(conn, "img.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    _add_detection(conn, image_filename="img.jpg", status="deleted")

    assert fetch_hard_negatives(conn) == []


def test_hard_negatives_excludes_images_without_no_bird_flag():
    conn = _make_conn()
    _add_image(conn, "img_untagged.jpg", review_status="untagged")
    _add_image(conn, "img_null.jpg", review_status=None)
    _add_image(conn, "img_confirmed.jpg", review_status="confirmed_species")
    for fn in ("img_untagged.jpg", "img_null.jpg", "img_confirmed.jpg"):
        _add_detection(conn, image_filename=fn)

    assert fetch_hard_negatives(conn) == []


def test_hard_negatives_returns_multiple_boxes_on_same_image():
    """A frame can carry several false-positive boxes (multi-bird FP).
    All of them must be exported individually so Pipeline-Dev sees the
    actual box geometry."""
    conn = _make_conn()
    _add_image(conn, "img.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    _add_detection(conn, image_filename="img.jpg", bbox=(0.1, 0.1, 0.2, 0.2))
    _add_detection(conn, image_filename="img.jpg", bbox=(0.5, 0.5, 0.1, 0.1))
    _add_detection(conn, image_filename="img.jpg", bbox=(0.7, 0.7, 0.2, 0.3))

    rows = fetch_hard_negatives(conn)
    assert len(rows) == 3
    assert {(r["bbox_x"], r["bbox_y"]) for r in rows} == {
        (0.1, 0.1), (0.5, 0.5), (0.7, 0.7),
    }


# ---------------------------------------------------------------------------
# fetch_confirmed_positives
# ---------------------------------------------------------------------------


def test_confirmed_positives_returns_user_authored_species_rows():
    """Only species rows the user explicitly authored qualify —
    species_source='manual' is the gate. A row with
    species_source='classifier' or 'model_top1' must NOT export."""
    conn = _make_conn()
    _add_image(conn, "img_user.jpg", review_status="untagged")
    _add_detection(conn, image_filename="img_user.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   species_source="manual",  # ← user-authored
                   species_updated_at="2026-05-22T11:00:00Z")
    # Negative control: pipeline-only row, must NOT appear in bucket
    _add_image(conn, "img_pipeline.jpg")
    _add_detection(conn, image_filename="img_pipeline.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   species_source="model_top1",  # pipeline-only
                   species_updated_at="2026-05-22T11:00:00Z")

    rows = fetch_confirmed_positives(conn)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "confirmed_positives"
    assert rows[0]["image_filename"] == "img_user.jpg"
    assert rows[0]["species"] == "Parus_major"
    assert rows[0]["species_source"] == "manual"


def test_confirmed_positives_excludes_pure_pipeline_predictions():
    """Regression guard for the original bug: a decision_level='species'
    row with no user touch (species_source='model_top1', no override)
    must NOT export. This was the confirmation-bias loop."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   species_source="model_top1",
                   species_updated_at="2026-05-22T11:00:00Z")

    assert fetch_confirmed_positives(conn) == []


def test_confirmed_positives_accepts_override_even_without_manual_source():
    """If a row carries a non-empty manual_species_override, that IS
    a user action (the picker write site stamps the override but may
    leave species_source untouched in some legacy paths). Accept it."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   manual_species_override="Cyanistes_caeruleus",
                   species_source="model_top1",  # source not flipped
                   species_updated_at="2026-05-22T11:00:00Z")

    rows = fetch_confirmed_positives(conn)
    assert len(rows) == 1
    assert rows[0]["species"] == "Cyanistes_caeruleus"


def test_confirmed_positives_excludes_unclear_levels():
    """species_review and reject are pipeline-uncertainty, not
    user-confirmation. They must not leak into the positive bucket."""
    conn = _make_conn()
    _add_image(conn, "img1.jpg")
    _add_image(conn, "img2.jpg")
    _add_image(conn, "img3.jpg")
    _add_detection(conn, image_filename="img1.jpg",
                   decision_level="species_review",
                   raw_species_name="Parus_major")
    _add_detection(conn, image_filename="img2.jpg",
                   decision_level="reject",
                   raw_species_name="Parus_major")
    _add_detection(conn, image_filename="img3.jpg",
                   decision_level="genus",
                   raw_species_name="Parus_major")

    assert fetch_confirmed_positives(conn) == []


def test_confirmed_positives_excludes_no_bird_images():
    """If the user later marked the frame no_bird, the species
    confirmation is invalidated — the box now contradicts reality."""
    conn = _make_conn()
    _add_image(conn, "img.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   species_updated_at="2026-05-22T09:00:00Z")

    assert fetch_confirmed_positives(conn) == []


def test_confirmed_positives_excludes_soft_deleted():
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   status="deleted",
                   decision_level="species",
                   raw_species_name="Parus_major")

    assert fetch_confirmed_positives(conn) == []


def test_confirmed_positives_prefers_manual_override_for_species_label():
    """When the user corrected the species AND the row is otherwise a
    valid positive (species-level, not no_bird), the exported label
    must be the user's correction, not the model's original guess.
    ``manual_species_override`` non-empty also counts as an explicit
    user action (no need for ``species_source='manual'`` alongside)."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   manual_species_override="Cyanistes_caeruleus",
                   species_source="manual_relabel")

    rows = fetch_confirmed_positives(conn)
    assert len(rows) == 1
    assert rows[0]["species"] == "Cyanistes_caeruleus"


# ---------------------------------------------------------------------------
# fetch_species_relabels
# ---------------------------------------------------------------------------


def test_species_relabels_returns_actual_corrections():
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="species_review",
                   raw_species_name="Parus_major",
                   manual_species_override="Cyanistes_caeruleus",
                   species_source="manual_relabel",
                   species_updated_at="2026-05-22T12:00:00Z")

    rows = fetch_species_relabels(conn)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "species_relabels"
    assert rows[0]["model_predicted_species"] == "Parus_major"
    assert rows[0]["user_corrected_species"] == "Cyanistes_caeruleus"


def test_species_relabels_excludes_no_op_overrides():
    """User clicked 'confirm' but picked the same species — no
    disagreement, no training signal, must not export."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   raw_species_name="Parus_major",
                   manual_species_override="Parus_major",
                   species_source="manual_bulk_confirm")

    assert fetch_species_relabels(conn) == []


def test_species_relabels_excludes_empty_string_override():
    """An empty string override is treated the same as NULL — no
    correction occurred."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   raw_species_name="Parus_major",
                   manual_species_override="")

    assert fetch_species_relabels(conn) == []


def test_species_relabels_excludes_soft_deleted():
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   status="deleted",
                   raw_species_name="Parus_major",
                   manual_species_override="Cyanistes_caeruleus")

    assert fetch_species_relabels(conn) == []


def test_species_relabels_includes_override_when_raw_is_null():
    """If the OD fired but classifier produced no species
    (raw_species_name NULL) and the user then *added* a species via
    moderation, that IS a relabel — the model guess was 'nothing
    specific'."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   raw_species_name=None,
                   manual_species_override="Erithacus_rubecula")

    rows = fetch_species_relabels(conn)
    assert len(rows) == 1
    assert rows[0]["user_corrected_species"] == "Erithacus_rubecula"
    assert rows[0]["model_predicted_species"] is None


# ---------------------------------------------------------------------------
# Cross-bucket isolation — adversarial dataset
# ---------------------------------------------------------------------------


def test_buckets_do_not_overlap_in_realistic_mixed_dataset():
    """Build a small but realistic dataset; every bucket query must
    return exactly its target rows, no row appears in two buckets."""
    conn = _make_conn()

    # img_hn: hard-negative frame, 2 boxes
    _add_image(conn, "img_hn.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    hn1 = _add_detection(conn, image_filename="img_hn.jpg",
                         decision_level="species_review",
                         raw_species_name="Parus_major")
    hn2 = _add_detection(conn, image_filename="img_hn.jpg",
                         decision_level="reject")

    # img_cp: confirmed positive (user-authored — species_source='manual')
    _add_image(conn, "img_cp.jpg", review_status="untagged")
    cp1 = _add_detection(conn, image_filename="img_cp.jpg",
                         decision_level="species",
                         raw_species_name="Cyanistes_caeruleus",
                         species_source="manual",
                         species_updated_at="2026-05-22T11:00:00Z")

    # img_rl: species relabel — note this is also species-level
    _add_image(conn, "img_rl.jpg")
    rl1 = _add_detection(conn, image_filename="img_rl.jpg",
                         decision_level="species",
                         raw_species_name="Parus_major",
                         manual_species_override="Poecile_palustris",
                         species_updated_at="2026-05-22T12:00:00Z")

    # img_trash: trash row, must appear in zero buckets
    _add_image(conn, "img_trash.jpg")
    _add_detection(conn, image_filename="img_trash.jpg",
                   status="deleted",
                   decision_level="species",
                   raw_species_name="Parus_major")

    hn_ids = {r["detection_id"] for r in fetch_hard_negatives(conn)}
    cp_ids = {r["detection_id"] for r in fetch_confirmed_positives(conn)}
    rl_ids = {r["detection_id"] for r in fetch_species_relabels(conn)}

    assert hn_ids == {hn1, hn2}
    # cp_ids includes the relabel row too — by design: a corrected
    # species at species-level IS a positive (with the corrected label).
    # The relabels bucket adds the model-vs-user-disagreement signal
    # on top; rows can legitimately appear in BOTH cp and rl.
    assert cp_ids == {cp1, rl1}
    assert rl_ids == {rl1}

    # Hard-negatives never overlap with positives or relabels:
    assert not (hn_ids & cp_ids)
    assert not (hn_ids & rl_ids)


# ---------------------------------------------------------------------------
# Time-window filtering
# ---------------------------------------------------------------------------


def test_time_window_since_is_inclusive_until_is_exclusive():
    """Half-open intervals so consecutive batches don't double-count:
    batch A: [t0, t1), batch B: [t1, t2). Row at exactly t1 belongs
    to batch B, not A."""
    conn = _make_conn()
    _add_image(conn, "img_early.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T09:59:59Z")
    _add_image(conn, "img_boundary.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    _add_image(conn, "img_late.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:01Z")
    for fn in ("img_early.jpg", "img_boundary.jpg", "img_late.jpg"):
        _add_detection(conn, image_filename=fn)

    rows = fetch_hard_negatives(
        conn,
        since="2026-05-22T10:00:00Z",
        until="2026-05-22T10:00:01Z",
    )
    fns = {r["image_filename"] for r in rows}
    assert fns == {"img_boundary.jpg"}


def test_time_window_only_since():
    conn = _make_conn()
    _add_image(conn, "img_old.jpg", review_status="no_bird",
               review_updated_at="2026-05-01T00:00:00Z")
    _add_image(conn, "img_new.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T00:00:00Z")
    for fn in ("img_old.jpg", "img_new.jpg"):
        _add_detection(conn, image_filename=fn)

    rows = fetch_hard_negatives(conn, since="2026-05-20T00:00:00Z")
    assert {r["image_filename"] for r in rows} == {"img_new.jpg"}


def test_time_window_only_until():
    conn = _make_conn()
    _add_image(conn, "img_old.jpg", review_status="no_bird",
               review_updated_at="2026-05-01T00:00:00Z")
    _add_image(conn, "img_new.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T00:00:00Z")
    for fn in ("img_old.jpg", "img_new.jpg"):
        _add_detection(conn, image_filename=fn)

    rows = fetch_hard_negatives(conn, until="2026-05-20T00:00:00Z")
    assert {r["image_filename"] for r in rows} == {"img_old.jpg"}


# ---------------------------------------------------------------------------
# count_pending_by_bucket must agree with fetch_*
# ---------------------------------------------------------------------------


def test_count_pending_by_bucket_matches_fetch_results():
    """The preview-page counts must equal the actual export counts —
    else the operator sees stale promises ('500 hard-negatives
    waiting!') that don't materialize in the ZIP."""
    conn = _make_conn()
    # 3 HN
    for i in range(3):
        fn = f"img_hn_{i}.jpg"
        _add_image(conn, fn, review_status="no_bird",
                   review_updated_at="2026-05-22T10:00:00Z")
        _add_detection(conn, image_filename=fn)
    # 2 CP (user-authored — species_source='manual')
    for i in range(2):
        fn = f"img_cp_{i}.jpg"
        _add_image(conn, fn)
        _add_detection(conn, image_filename=fn,
                       decision_level="species",
                       raw_species_name="Parus_major",
                       species_source="manual",
                       species_updated_at="2026-05-22T11:00:00Z")
    # 1 RL (also counts as CP)
    _add_image(conn, "img_rl.jpg")
    _add_detection(conn, image_filename="img_rl.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   manual_species_override="Cyanistes_caeruleus",
                   species_updated_at="2026-05-22T12:00:00Z")

    counts = count_pending_by_bucket(conn)
    assert counts == {
        "hard_negatives": 3,
        "confirmed_positives": 3,  # 2 plain + 1 relabel
        "species_relabels": 1,
        "favorites": 0,
    }
    # And the actual fetches produce the same cardinalities:
    assert len(fetch_hard_negatives(conn)) == counts["hard_negatives"]
    assert len(fetch_confirmed_positives(conn)) == counts["confirmed_positives"]
    assert len(fetch_species_relabels(conn)) == counts["species_relabels"]
    assert len(fetch_favorites(conn)) == counts["favorites"]


def test_count_pending_respects_time_window():
    conn = _make_conn()
    _add_image(conn, "img_old.jpg", review_status="no_bird",
               review_updated_at="2026-05-01T00:00:00Z")
    _add_image(conn, "img_new.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T00:00:00Z")
    for fn in ("img_old.jpg", "img_new.jpg"):
        _add_detection(conn, image_filename=fn)

    counts = count_pending_by_bucket(conn, since="2026-05-20T00:00:00Z")
    assert counts["hard_negatives"] == 1


# ---------------------------------------------------------------------------
# Empty-DB and edge cases
# ---------------------------------------------------------------------------


def test_empty_database_returns_empty_lists():
    conn = _make_conn()
    assert fetch_hard_negatives(conn) == []
    assert fetch_confirmed_positives(conn) == []
    assert fetch_species_relabels(conn) == []
    assert fetch_favorites(conn) == []
    assert count_pending_by_bucket(conn) == {
        "hard_negatives": 0,
        "confirmed_positives": 0,
        "species_relabels": 0,
        "favorites": 0,
    }


def test_decision_level_case_insensitivity():
    """The unclear.py module already normalizes via lower(); mirror
    that here so historical rows with mixed case still classify.
    Use a user-authored row so the explicit-user-action gate passes."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="SPECIES",  # uppercased
                   raw_species_name="Parus_major",
                   species_source="manual",
                   species_updated_at="2026-05-22T11:00:00Z")

    rows = fetch_confirmed_positives(conn)
    assert len(rows) == 1


@pytest.mark.parametrize("missing_field", [
    "od_confidence",
    "detector_model_version",
    "classifier_model_version",
])
def test_null_provenance_fields_propagate_as_none(missing_field):
    """Old detections from before a column existed have NULL values.
    The export should still ship them — Pipeline-Dev needs to know
    'no model version recorded' rather than have rows silently
    dropped."""
    conn = _make_conn()
    _add_image(conn, "img.jpg", review_status="no_bird",
               review_updated_at="2026-05-22T10:00:00Z")
    kwargs = {
        "image_filename": "img.jpg",
        "od_confidence": 0.5,
        "detector_model_version": "x",
        "classifier_model_version": "y",
    }
    kwargs[missing_field] = None
    _add_detection(conn, **kwargs)

    rows = fetch_hard_negatives(conn)
    assert len(rows) == 1
    assert rows[0][missing_field] is None


# ---------------------------------------------------------------------------
# fetch_favorites
# ---------------------------------------------------------------------------


def test_favorites_returns_heart_clicked_detections():
    """Only is_favorite=1 AND rating_source='manual' qualifies."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   decision_level="species",
                   raw_species_name="Parus_major",
                   is_favorite=1,
                   rating_source="manual",
                   species_updated_at="2026-05-22T11:00:00Z")

    rows = fetch_favorites(conn)
    assert len(rows) == 1
    assert rows[0]["bucket"] == "favorites"
    assert rows[0]["species"] == "Parus_major"


def test_favorites_excludes_auto_rating_source():
    """A row with is_favorite=1 but rating_source='auto' is a legacy
    backfill artefact (aesthetic tagger stamped it), NOT a user
    action — must be excluded."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   raw_species_name="Parus_major",
                   is_favorite=1,
                   rating_source="auto")  # ← legacy backfill

    assert fetch_favorites(conn) == []


def test_favorites_excludes_unfavorited_with_manual_source():
    """The mirror case: rating_source='manual' but is_favorite=0
    (e.g. user manually un-hearted). Should NOT export."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   raw_species_name="Parus_major",
                   is_favorite=0,
                   rating_source="manual")

    assert fetch_favorites(conn) == []


def test_favorites_excludes_soft_deleted():
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   status="deleted",
                   raw_species_name="Parus_major",
                   is_favorite=1,
                   rating_source="manual")

    assert fetch_favorites(conn) == []


def test_favorites_uses_override_species_when_present():
    """If the user both corrected the species AND hearted the
    detection, the favorites bucket carries the corrected species
    (same override-wins-over-raw semantics as confirmed_positives)."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    _add_detection(conn, image_filename="img.jpg",
                   raw_species_name="Parus_major",
                   manual_species_override="Cyanistes_caeruleus",
                   is_favorite=1,
                   rating_source="manual",
                   species_updated_at="2026-05-22T11:00:00Z")

    rows = fetch_favorites(conn)
    assert len(rows) == 1
    assert rows[0]["species"] == "Cyanistes_caeruleus"


def test_favorites_can_overlap_with_confirmed_positives():
    """A favorited confirmed-positive detection appears in both
    buckets — by design. The COCO de-dupe in the service layer
    handles the single-annotation guarantee; the manifests preserve
    every signal-source so Pipeline-Dev can weight them differently.
    Requires species_source='manual' to satisfy the explicit-user-
    action gate on the CP side."""
    conn = _make_conn()
    _add_image(conn, "img.jpg")
    det_id = _add_detection(conn, image_filename="img.jpg",
                            decision_level="species",
                            raw_species_name="Parus_major",
                            species_source="manual",
                            is_favorite=1,
                            rating_source="manual",
                            species_updated_at="2026-05-22T11:00:00Z")

    cp_ids = {r["detection_id"] for r in fetch_confirmed_positives(conn)}
    fav_ids = {r["detection_id"] for r in fetch_favorites(conn)}
    assert det_id in cp_ids
    assert det_id in fav_ids
