"""Retention × daily report — graceful degradation.

The daily report reads the full-resolution ORIGINAL (it is one of the few
surfaces that does). When retention has removed an original, the report must
degrade gracefully — skip that photo without raising — never crash the
report build. This pins the existing os.path.isfile + continue guard so a
future change can't silently break it.
"""

import pytest

from utils.daily_report import _fetch_species_best_photos
from utils.db.connection import closing_connection
from utils.db.detections import insert_detection


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def test_best_photo_skipped_when_original_deleted_no_exception(tmp_path, monkeypatch):
    output_dir = tmp_path / "output"
    cfg = {"OUTPUT_DIR": str(output_dir), "SPECIES_COMMON_NAME_LOCALE": "DE"}
    monkeypatch.setattr("config.get_config", lambda: cfg)

    fn = "20260101_120000_a.jpg"
    # Confirmed detection, but the original file was retention-deleted: the
    # DB row stays, the file is absent on disk.
    with closing_connection() as conn:
        conn.execute(
            "INSERT INTO images (filename, timestamp, review_status, "
            "original_present, original_deleted_at) "
            "VALUES (?, ?, 'confirmed_bird', 0, '2026-05-01T00:00:00+00:00')",
            (fn, fn[:15]),
        )
        det = insert_detection(
            conn,
            {
                "image_filename": fn,
                "bbox_x": 0.1, "bbox_y": 0.1, "bbox_w": 0.2, "bbox_h": 0.2,
                "od_class_name": "bird", "od_confidence": 0.95,
                "raw_species_name": "Parus_major",
                "decision_state": "confirmed",
            },
        )
        conn.execute(
            "UPDATE detections SET status='active' WHERE detection_id=?", (det,)
        )
        conn.commit()

        # Must not raise even though the original is gone.
        result = _fetch_species_best_photos(conn, "2026-01-01")

    # The species is dropped from the report (graceful degradation), not an error.
    assert all(r.get("best_photo_path") for r in result)
    assert not any(r.get("scientific_name") == "Parus_major" for r in result)
