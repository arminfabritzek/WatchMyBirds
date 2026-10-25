"""Training protection requires current answers with originals or pending exports."""

import pytest

from config import get_config
from core import training_retention
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from utils.db.connection import closing_connection
from utils.db.detections import insert_detection
from utils.path_manager import PathManager


@pytest.fixture(autouse=True)
def wipe_schema_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def _add_image(conn, filename, review_status="confirmed_bird"):
    conn.execute(
        "INSERT INTO images (filename, timestamp, review_status) VALUES (?, ?, ?)",
        (filename, filename[:15], review_status),
    )

    path = PathManager(get_config()["OUTPUT_DIR"]).get_original_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original")

    if review_status == "no_bird":
        record_human_answer(
            conn,
            HumanAnswer(image_filename=filename, image_bird_presence="absent"),
            LabelProvenance(
                "synthetic", "test", "normal_correction", "watchmybirds_ui"
            ),
        )


def _add_detection(conn, filename, *, is_favorite=0, rating_source=None, **overrides):
    # insert_detection handles the original CREATE-TABLE columns + the
    # decision/species columns; status/is_favorite/rating_source are
    # migration-added columns set via UPDATE (mirrors production write sites).
    row = {
        "image_filename": filename,
        "bbox_x": 0.1,
        "bbox_y": 0.1,
        "bbox_w": 0.2,
        "bbox_h": 0.2,
        "od_class_name": "bird",
        "od_confidence": 0.9,
    }
    row.update(overrides)
    det_id = insert_detection(conn, row)
    conn.execute(
        "UPDATE detections SET status='active', is_favorite=?, rating_source=? "
        "WHERE detection_id=?",
        (is_favorite, rating_source, det_id),
    )
    if overrides.get("manual_species_override"):
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename=filename,
                detection_id=det_id,
                species_identity="corrected",
                species_key=overrides["manual_species_override"],
            ),
            LabelProvenance(
                "synthetic", "test", "normal_correction", "watchmybirds_ui"
            ),
        )
    return det_id


def test_plain_unreviewed_detection_is_not_export_relevant():
    fn = "20260101_120000_a.jpg"
    with closing_connection() as conn:
        _add_image(conn, fn, review_status="untagged")
        _add_detection(conn, fn)
        conn.commit()
        result = training_retention.is_export_relevant_any(conn, [fn])
    assert result == set()


def test_hard_negative_is_export_relevant():
    fn = "20260101_120001_b.jpg"
    with closing_connection() as conn:
        _add_image(conn, fn, review_status="no_bird")
        _add_detection(conn, fn)
        conn.commit()
        result = training_retention.is_export_relevant_any(conn, [fn])
    assert result == {fn}


def test_favorite_is_not_training_evidence():
    fn = "20260101_120002_c.jpg"
    with closing_connection() as conn:
        _add_image(conn, fn)
        _add_detection(conn, fn, is_favorite=1, rating_source="manual")
        conn.commit()
        result = training_retention.is_export_relevant_any(conn, [fn])
    assert result == set()


def test_species_relabel_is_export_relevant():
    fn = "20260101_120003_d.jpg"
    with closing_connection() as conn:
        _add_image(conn, fn)
        _add_detection(
            conn,
            fn,
            raw_species_name="Parus_major",
            manual_species_override="Cyanistes_caeruleus",
            species_source="manual",
        )
        conn.commit()
        result = training_retention.is_export_relevant_any(conn, [fn])
    assert result == {fn}


def test_filter_only_returns_requested_filenames():
    keep = "20260101_120004_e.jpg"
    other = "20260101_120005_f.jpg"
    with closing_connection() as conn:
        _add_image(conn, keep, review_status="no_bird")
        _add_detection(conn, keep)
        _add_image(conn, other, review_status="no_bird")
        _add_detection(conn, other)
        conn.commit()
        # Only ask about `keep`; `other` is export-relevant but not requested.
        result = training_retention.is_export_relevant_any(conn, [keep])
    assert result == {keep}


def test_empty_filenames_returns_empty_set():
    with closing_connection() as conn:
        result = training_retention.is_export_relevant_any(conn, [])
    assert result == set()
