import sqlite3

from utils.db.review_queue import (
    fetch_recent_review_species,
    fetch_review_cluster_context,
    fetch_review_queue_count,
    fetch_review_queue_images,
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from utils.db.connection import _init_schema

    _init_schema(conn)
    return conn


def test_review_queue_uses_score_threshold_no_dead_zone():
    conn = _make_conn()
    try:
        # Orphan (no detections)
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("orphan.jpg", "20260210_000000", "untagged"),
        )

        # Low score but high OD confidence -> must appear in review queue (dead-zone fix)
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("low.jpg", "20260210_000001", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("low.jpg", 0.80, 0.50, "yolo"),
        )

        # High score -> not in review queue
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("high.jpg", "20260210_000002", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("high.jpg", 0.80, 0.90, "yolo"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        filenames = [row["filename"] for row in rows]
        assert "orphan.jpg" in filenames
        assert "low.jpg" in filenames
        assert "high.jpg" not in filenames

        count = fetch_review_queue_count(conn, gallery_threshold=0.7)
        assert count == 2
    finally:
        conn.close()


def test_review_queue_keeps_previous_scan_detections():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("ds.jpg", "20260210_000003", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("ds.jpg", 0.20, 0.20, "deep_scan_tiled"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        assert [row["filename"] for row in rows] == ["ds.jpg"]
    finally:
        conn.close()


def test_review_queue_exposes_top_bbox_for_low_score_items():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("low_bbox.jpg", "20260210_000010", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("low_bbox.jpg", 0.70, 0.40, "yolo"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("low_bbox.jpg", 0.65, 0.55, "yolo"),
        )
        conn.execute(
            """
            UPDATE detections
            SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?
            WHERE image_filename = ? AND score = ?
            """,
            (0.10, 0.20, 0.30, 0.40, "low_bbox.jpg", 0.55),
        )
        conn.execute(
            """
            UPDATE detections
            SET bbox_x = ?, bbox_y = ?, bbox_w = ?, bbox_h = ?
            WHERE image_filename = ? AND score = ?
            """,
            (0.50, 0.60, 0.10, 0.10, "low_bbox.jpg", 0.40),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        target = next(row for row in rows if row["filename"] == "low_bbox.jpg")
        assert target["review_reason"] == "low_score"
        assert target["bbox_x"] == 0.10
        assert target["bbox_y"] == 0.20
        assert target["bbox_w"] == 0.30
        assert target["bbox_h"] == 0.40
    finally:
        conn.close()


def test_review_queue_exposes_detection_context_and_bbox_review_state():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("ctx.jpg", "20260210_000020", "untagged"),
        )
        cur = conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, bbox_quality, unknown_score
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ctx.jpg", 0.75, 0.42, "yolo", 0.25, 0.68),
        )
        detection_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (detection_id, "Parus_major", 0.84, 1, "active"),
        )
        from core.human_label_core import (
            LabelProvenance,
            append_fact,
            ensure_object_subject,
        )

        conn.execute(
            "UPDATE detections SET bbox_x=0.1,bbox_y=0.1,bbox_w=0.3,bbox_h=0.3 WHERE detection_id=?",
            (detection_id,),
        )
        append_fact(
            conn,
            subject_id=ensure_object_subject(conn, detection_id),
            fact_type="bbox_quality",
            answer_value="unsuitable",
            provenance=LabelProvenance(
                "synthetic", "test", "normal_correction", "watchmybirds_ui", "test"
            ),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        target = next(row for row in rows if row["filename"] == "ctx.jpg")

        assert target["best_detection_id"] == detection_id
        assert target["species_key"] == "Parus_major"
        assert target["human_bbox_verdict"] == "unsuitable"
        assert target["cls_confidence"] == 0.84
        assert target["od_confidence"] == 0.75
    finally:
        conn.close()


def test_review_queue_returns_detection_items_per_unresolved_sibling():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("siblings.jpg", "20260210_000030", "untagged"),
        )
        first = conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, decision_state
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("siblings.jpg", 0.91, 0.91, "yolo", "uncertain"),
        ).lastrowid
        second = conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, decision_state
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            ("siblings.jpg", 0.66, 0.41, "yolo", "unknown"),
        ).lastrowid
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        sibling_rows = [row for row in rows if row["filename"] == "siblings.jpg"]

        assert len(sibling_rows) == 2
        assert {row["item_kind"] for row in sibling_rows} == {"detection"}
        assert {row["active_detection_id"] for row in sibling_rows} == {first, second}
        assert all(row["sibling_detection_count"] == 2 for row in sibling_rows)
    finally:
        conn.close()


def test_review_queue_ignores_retired_image_status():
    conn = _make_conn()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("trashed_import.jpg", "20260210_000040", "no_bird"),
        )
        conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id, decision_state, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("trashed_import.jpg", 0.62, 0.31, "yolo", "unknown", "active"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        filenames = [row["filename"] for row in rows]
        assert "trashed_import.jpg" in filenames

        count = fetch_review_queue_count(conn, gallery_threshold=0.7)
        assert count == 1
    finally:
        conn.close()


def _insert_confirmed_detection(
    conn,
    *,
    filename,
    timestamp,
    species,
    score=0.95,
):
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        (filename, timestamp, "confirmed_bird"),
    )
    cur = conn.execute(
        """
        INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
        VALUES (?, ?, ?, ?)
        """,
        (filename, 0.9, score, "yolo"),
    )
    conn.execute(
        """
        INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (cur.lastrowid, species, 0.9, 1, "active"),
    )

    conn.execute(
        "UPDATE detections SET decision_state='confirmed',decision_level='species' WHERE detection_id=?",
        (cur.lastrowid,),
    )


def test_review_cluster_context_returns_empty_for_empty_range():
    conn = _make_conn()
    try:
        rows, truncated = fetch_review_cluster_context(conn, untagged_time_range=None)
        assert rows == []
        assert truncated is False
    finally:
        conn.close()


def test_review_cluster_context_returns_neighbour_inside_window():
    conn = _make_conn()
    try:
        _insert_confirmed_detection(
            conn,
            filename="confirmed_close.jpg",
            timestamp="20260210_120500",
            species="Pica_pica",
        )
        conn.commit()
        # Untagged window 12:30 → ±30 min covers 12:00..13:00
        rows, truncated = fetch_review_cluster_context(
            conn,
            untagged_time_range=("20260210_123000", "20260210_123500"),
        )
        assert truncated is False
        assert len(rows) == 1
        row = rows[0]
        assert row["context_only"] is True
        assert row["filename"] == "confirmed_close.jpg"
        assert row["species_key"] == "Pica_pica"
        assert row["item_kind"] == "detection"
        assert row["review_reason"] == "context"
    finally:
        conn.close()


def test_review_cluster_context_excludes_neighbour_outside_window():
    conn = _make_conn()
    try:
        _insert_confirmed_detection(
            conn,
            filename="confirmed_far.jpg",
            timestamp="20260210_100000",
            species="Pica_pica",
        )
        conn.commit()
        rows, truncated = fetch_review_cluster_context(
            conn,
            untagged_time_range=("20260210_123000", "20260210_123500"),
        )
        assert rows == []
        assert truncated is False
    finally:
        conn.close()


def test_review_cluster_context_truncates_at_safety_cap():
    conn = _make_conn()
    try:
        for index in range(7):
            _insert_confirmed_detection(
                conn,
                filename=f"confirmed_{index}.jpg",
                timestamp=f"20260210_1230{index:02d}",
                species="Pica_pica",
            )
        conn.commit()
        rows, truncated = fetch_review_cluster_context(
            conn,
            untagged_time_range=("20260210_123000", "20260210_123500"),
            max_context_rows=3,
        )
        assert len(rows) == 3
        assert truncated is True
    finally:
        conn.close()


def test_review_cluster_context_excludes_unconfirmed_neighbours():
    conn = _make_conn()
    try:
        # Untagged neighbour — must be ignored, only confirmed_bird counts.
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("untagged_neighbour.jpg", "20260210_123100", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
            VALUES (?, ?, ?, ?)
            """,
            ("untagged_neighbour.jpg", 0.6, 0.6, "yolo"),
        )
        conn.commit()
        rows, _ = fetch_review_cluster_context(
            conn,
            untagged_time_range=("20260210_123000", "20260210_123500"),
        )
        assert rows == []
    finally:
        conn.close()


def test_recent_review_species_returns_recent_frequent_species():
    conn = _make_conn()
    try:
        conn.executemany(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            [
                ("a.jpg", "20990110_000001", "untagged"),
                ("b.jpg", "20990110_000002", "untagged"),
                ("c.jpg", "20990109_230000", "untagged"),
            ],
        )
        detections = []
        for filename, species in [
            ("a.jpg", "Parus_major"),
            ("b.jpg", "Parus_major"),
            ("c.jpg", "Erithacus_rubecula"),
        ]:
            cur = conn.execute(
                """
                INSERT INTO detections (image_filename, od_confidence, score, od_model_id)
                VALUES (?, ?, ?, ?)
                """,
                (filename, 0.8, 0.9, "yolo"),
            )
            detections.append((cur.lastrowid, species))

        conn.executemany(
            """
            INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(det_id, species, 0.9, 1, "active") for det_id, species in detections],
        )
        conn.commit()

        rows = fetch_recent_review_species(conn, limit=8, lookback_days=3650)
        species = [row["species_key"] for row in rows]
        assert species[:2] == ["Parus_major", "Erithacus_rubecula"]
    finally:
        conn.close()


def _make_conn_with_decision_level() -> sqlite3.Connection:
    """Schema variant that includes the live ``decision_level`` column.

    The base ``_make_conn`` fixture omits it because most tests pre-date
    that column. Tests that exercise the ex-Unclear routing (where
    Smoother=confirmed but Classifier=reject) need it present.
    """
    conn = _make_conn()
    conn.commit()
    return conn


def test_review_queue_admits_classifier_rejected_confirmed_detections():
    """Ex-Unclear: Smoother says confirmed, Classifier rejected the label.

    These detections used to live on a separate /unclear surface. Per
    P-01 (Single User-Decision-Point) they now route to Review.
    """
    conn = _make_conn_with_decision_level()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("ex_unclear.jpg", "20260525_120000", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id,
                decision_state, decision_level
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ex_unclear.jpg", 0.85, 0.95, "yolo", "confirmed", "reject"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        reasons = {row["filename"]: row["review_reason"] for row in rows}
        assert reasons.get("ex_unclear.jpg") == "classifier_reject"

        count = fetch_review_queue_count(conn, gallery_threshold=0.7)
        assert count == 1
    finally:
        conn.close()


def test_review_queue_admits_species_review_confirmed_detections():
    """Two-stage gate "soft reject" (species_review) also routes to Review."""
    conn = _make_conn_with_decision_level()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("soft.jpg", "20260525_120001", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id,
                decision_state, decision_level
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("soft.jpg", 0.85, 0.95, "yolo", "confirmed", "species_review"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        reasons = {row["filename"]: row["review_reason"] for row in rows}
        assert reasons.get("soft.jpg") == "classifier_species_review"
    finally:
        conn.close()


def test_review_queue_does_not_admit_gallery_eligible_confirmed_detections():
    """Smoother=confirmed AND Classifier=species (label landed) stays out of Review."""
    conn = _make_conn_with_decision_level()
    try:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
            ("gallery.jpg", "20260525_120002", "untagged"),
        )
        conn.execute(
            """
            INSERT INTO detections (
                image_filename, od_confidence, score, od_model_id,
                decision_state, decision_level
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("gallery.jpg", 0.85, 0.95, "yolo", "confirmed", "species"),
        )
        conn.commit()

        rows = fetch_review_queue_images(conn, gallery_threshold=0.7)
        assert "gallery.jpg" not in [row["filename"] for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lean summary path — feeds the "Move Review Queue to Trash" preview/run.
# These functions MUST select the exact same queue membership as
# fetch_review_queue_images (the rendered Review page), only cheaper.
# ---------------------------------------------------------------------------


def _seed_mixed_review_db(conn):
    """Orphan untagged image + queued low-score detection + excluded rows."""
    # Orphan (no detections) -> image item
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        ("orphan.jpg", "20260210_000000", "untagged"),
    )
    # Queued low-score detection -> detection item
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        ("low.jpg", "20260210_000100", "untagged"),
    )
    conn.execute(
        "INSERT INTO detections (image_filename, od_confidence, score, od_model_id, decision_state) "
        "VALUES (?, ?, ?, ?, ?)",
        ("low.jpg", 0.80, 0.40, "yolo", "uncertain"),
    )
    det_id = conn.execute(
        "SELECT detection_id FROM detections WHERE image_filename = 'low.jpg'"
    ).fetchone()["detection_id"]
    conn.execute(
        "INSERT INTO classifications (detection_id, cls_class_name, cls_confidence, rank, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (det_id, "Passer_domesticus", 0.5, 1, "active"),
    )
    # Excluded: confirmed high-score detection (gallery, not queue)
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        ("high.jpg", "20260210_000200", "untagged"),
    )
    conn.execute(
        "INSERT INTO detections (image_filename, od_confidence, score, od_model_id, decision_state) "
        "VALUES (?, ?, ?, ?, ?)",
        ("high.jpg", 0.90, 0.95, "yolo", "confirmed"),
    )
    # Excluded: confirmed_bird image (resolved)
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        ("done.jpg", "20260210_000300", "confirmed_bird"),
    )
    conn.commit()


def test_summary_rows_membership_equals_full_queue():
    from utils.db.review_queue import fetch_review_queue_summary_rows

    conn = _make_conn()
    try:
        _seed_mixed_review_db(conn)

        full = fetch_review_queue_images(conn, gallery_threshold=0.7)
        lean = fetch_review_queue_summary_rows(conn, gallery_threshold=0.7)

        full_keys = {(r["item_kind"], str(r["item_id"])) for r in full}
        lean_keys = {(r["item_kind"], str(r["item_id"])) for r in lean}
        assert lean_keys == full_keys
    finally:
        conn.close()


def test_summary_rows_carry_species_and_favorite():
    from utils.db.review_queue import fetch_review_queue_summary_rows

    conn = _make_conn()
    try:
        _seed_mixed_review_db(conn)
        lean = fetch_review_queue_summary_rows(conn, gallery_threshold=0.7)

        det = next(r for r in lean if r["item_kind"] == "detection")
        # species resolved from the rank-1 classification
        assert det["species_key"] == "Passer_domesticus"
        # favorite column present and defaulted
        assert det["is_favorite"] in (0, 1)
        # clustering needs a timestamp and the source filename
        assert det["filename"] == "low.jpg"
        assert det["timestamp"] == "20260210_000100"
    finally:
        conn.close()


def test_summary_counts_match_full_queue_partition():
    from utils.db.review_queue import fetch_review_queue_summary

    conn = _make_conn()
    try:
        _seed_mixed_review_db(conn)

        full = fetch_review_queue_images(conn, gallery_threshold=0.7)
        n_images = sum(1 for r in full if r["item_kind"] == "image")
        n_dets = sum(1 for r in full if r["item_kind"] == "detection")
        n_fav = sum(
            1 for r in full if r["item_kind"] == "detection" and r["is_favorite"]
        )

        summary = fetch_review_queue_summary(conn, gallery_threshold=0.7)
        assert summary["images"] == n_images
        assert summary["detections"] == n_dets
        assert summary["favorites"] == n_fav
    finally:
        conn.close()
