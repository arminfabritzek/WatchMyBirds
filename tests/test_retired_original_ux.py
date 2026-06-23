"""UX hardening for retention-retired originals.

When an original has been retired (``images.original_present=0``) the gallery/
review/detail payloads must carry that fact so the modal action bar can hide
the dead "Download Original" button and offer the optimized preview instead.

This is presentation only: the retention policy, the deletion logic, and the
410 backend guard on ``/api/image/download`` are all unchanged.
"""

import sqlite3
from pathlib import Path

from flask import Flask, render_template_string

from utils.db.detections import fetch_detections_for_gallery
from utils.db.review_queue import fetch_review_queue_images


def _read(rel: str) -> str:
    return (Path(__file__).resolve().parent.parent / rel).read_text(encoding="utf-8")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            review_status TEXT DEFAULT 'confirmed_bird',
            downloaded_timestamp TEXT,
            ptz_origin TEXT,
            original_present INTEGER DEFAULT 1
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_class_name TEXT, od_confidence REAL, score REAL,
            thumbnail_path TEXT,
            manual_species_override TEXT, species_source TEXT,
            raw_species_name TEXT,
            decision_state TEXT DEFAULT 'confirmed',
            decision_level TEXT,
            bbox_quality REAL, unknown_score REAL, decision_reasons TEXT,
            rating INTEGER, rating_source TEXT, is_favorite INTEGER DEFAULT 0,
            is_gallery_eligible INTEGER DEFAULT 1, aesthetic_score REAL,
            quality_gallery_ok INTEGER DEFAULT 1
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER, cls_class_name TEXT, cls_confidence REAL,
            rank INTEGER DEFAULT 1, status TEXT DEFAULT 'active'
        );
        """
    )
    return conn


def _seed(conn, filename, *, original_present):
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status, original_present) "
        "VALUES (?, ?, 'confirmed_bird', ?)",
        (filename, filename[:15], original_present),
    )
    conn.execute(
        "INSERT INTO detections (image_filename, status, created_at, score, "
        "decision_state, od_confidence, is_gallery_eligible) "
        "VALUES (?, 'active', ?, 0.9, 'confirmed', 0.9, 1)",
        (filename, filename[:15]),
    )
    conn.commit()


def test_gallery_payload_carries_original_present_when_present():
    conn = _conn()
    _seed(conn, "20260101_120000_a.jpg", original_present=1)
    rows = list(fetch_detections_for_gallery(conn, order_by="time"))
    assert len(rows) == 1
    assert rows[0]["original_present"] == 1


def test_gallery_payload_carries_original_present_when_retired():
    conn = _conn()
    _seed(conn, "20260101_120000_b.jpg", original_present=0)
    rows = list(fetch_detections_for_gallery(conn, order_by="time"))
    assert len(rows) == 1
    assert rows[0]["original_present"] == 0


# --- modal action bar gating ---------------------------------------------


def test_action_bar_accepts_retired_and_optimized_args():
    content = _read("templates/components/modal_action_bar.html")
    assert "original_retired=false" in content
    assert "optimized_url=none" in content


def test_action_bar_hides_download_original_when_retired():
    content = _read("templates/components/modal_action_bar.html")
    # Retired frames take the retired branch; the original-download link only
    # renders in the elif (i.e. when NOT retired).
    assert "if show_download and original_retired" in content
    assert "elif show_download and (detection_id or original_url)" in content
    assert (
        "/api/image/download/" in content
    )  # original download still exists for present


def test_action_bar_shows_retired_label_and_preview_download():
    content = _read("templates/components/modal_action_bar.html")
    assert "Original retired" in content
    assert "Download preview" in content


def test_detection_modal_passes_retired_and_optimized_to_action_bar():
    content = _read("templates/components/detection_modal.html")
    assert "original_retired=" in content
    assert "det.original_present" in content
    assert "optimized_url=" in content


def test_detection_view_dict_preserves_retired_original_state():
    from web import view_helpers

    det = view_helpers.build_detection_view_dict(
        {"detection_id": 3795, "original_present": 0},
        species_key="Poecile_palustris",
        common_name="Marsh Tit",
    )

    assert det["original_present"] == 0


def test_detection_view_dict_defaults_original_present_for_legacy_payloads():
    from web import view_helpers

    det = view_helpers.build_detection_view_dict(
        {"detection_id": 3796},
        species_key="Poecile_palustris",
        common_name="Marsh Tit",
    )

    assert det["original_present"] == 1


def test_detection_modal_renders_preview_download_for_retired_original():
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / "templates"))
    app.secret_key = "test"
    app.jinja_env.globals["wikipedia_species_url"] = lambda *_args: None
    det = {
        "detection_id": 3795,
        "species_key": "Poecile_palustris",
        "common_name": "Marsh Tit",
        "od_class_name": "bird",
        "od_confidence": 0.89,
        "cls_class_name": "Poecile_palustris",
        "cls_confidence": 0.99,
        "score": 0.89,
        "formatted_date": "07.05.2026",
        "formatted_time": "16:01:30",
        "gallery_date": "2026-05-07",
        "siblings": [],
        "sibling_count": 1,
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_w": 0.2,
        "bbox_h": 0.2,
        "is_favorite": False,
        "image_filename": "20260507_160130_108499.jpg",
        "original_path": "/uploads/originals/2026-05-07/20260507_160130_108499.jpg",
        "full_path": "/uploads/derivatives/optimized/2026-05-07/20260507_160130_108499.webp",
        "original_present": 0,
        "decision_state": "confirmed",
    }

    with app.test_request_context("/gallery/2026-05-07"):
        rendered = render_template_string(
            """
            {% from "components/detection_modal.html" import render_modal %}
            {{ render_modal(det, "species_overview") }}
            """,
            det=det,
        )

    assert "Original retired" in rendered
    assert "Download preview" in rendered
    assert "/api/image/download/3795" not in rendered


# --- review queue threading ----------------------------------------------


def test_review_queue_row_carries_original_present():
    from utils.db.review_queue import fetch_review_queue_images

    conn = _conn()
    # An untagged, low-score (uncertain) detection -> in the review queue;
    # its original has been retired.
    conn.execute(
        "UPDATE images SET review_status='untagged', original_present=0 "
        "WHERE filename IS NOT NULL"
    )
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status, original_present) "
        "VALUES ('20260101_120000_r.jpg', '20260101_120000', 'untagged', 0)"
    )
    conn.execute(
        "INSERT INTO detections (image_filename, status, created_at, score, "
        "decision_state, od_confidence) VALUES "
        "('20260101_120000_r.jpg', 'active', '20260101_120000', 0.2, 'uncertain', 0.2)"
    )
    conn.commit()
    rows = fetch_review_queue_images(conn)
    detrows = [r for r in rows if r["item_kind"] == "detection"]
    assert detrows
    assert detrows[0]["original_present"] == 0


def test_review_modal_detection_carries_retired_state_and_optimized():
    from web.blueprints.review import _build_review_modal_detection

    conn = _conn()
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status, original_present) "
        "VALUES ('20260101_120000_m.jpg', '20260101_120000', 'untagged', 0)"
    )
    conn.execute(
        "INSERT INTO detections (image_filename, status, created_at, score, "
        "decision_state, od_confidence) VALUES "
        "('20260101_120000_m.jpg', 'active', '20260101_120000', 0.2, 'uncertain', 0.2)"
    )
    conn.commit()
    row = next(
        r for r in fetch_review_queue_images(conn) if r["item_kind"] == "detection"
    )
    det = _build_review_modal_detection(
        row,
        filename="20260101_120000_m.jpg",
        full_url="/uploads/originals/2026-01-01/20260101_120000_m.jpg",
        thumb_url="/api/review-thumb/20260101_120000_m.jpg",
        optimized_url="/uploads/derivatives/optimized/2026-01-01/20260101_120000_m.webp",
        selected_species=None,
        selected_species_common=None,
        current_species=None,
        current_species_common=None,
        common_names={},
        conn=conn,
        siblings=[],
    )
    assert det is not None
    assert det["original_present"] == 0
    # The preview-download path is the optimized WebP, NOT the gone original.
    assert det["optimized_path"].endswith(".webp")
    assert "originals" not in det["optimized_path"]


def test_detection_modal_uses_optimized_path_with_full_path_fallback():
    content = _read("templates/components/detection_modal.html")
    assert "det.optimized_path" in content
    assert "det.full_path" in content  # gallery fallback (full_path == optimized)


def test_orphan_modal_gates_action_bar_on_retired():
    content = _read("templates/components/orphan_modal.html")
    assert "original_retired=" in content
