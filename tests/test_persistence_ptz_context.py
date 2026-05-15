"""Verify PersistenceService records PTZ context on image rows.

Integration-level: builds a real PersistenceService, injects a fake
controller, calls save_image, then reads the images table back and
asserts the PTZ columns were populated.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pytest

from detectors.services.persistence_service import PersistenceService


class FakePtzController:
    """Minimal controller stub — duck-typed against the snapshot interface.

    The real AutoPtzController exposes snapshot_for_image_persistence().
    PersistenceService only needs that one method, so this stub is enough.
    """

    def __init__(self, snapshot: dict):
        self._snapshot = snapshot

    def snapshot_for_image_persistence(self) -> dict:
        return dict(self._snapshot)


class BrokenPtzController:
    def snapshot_for_image_persistence(self) -> dict:
        raise RuntimeError("simulated PTZ failure")


@pytest.fixture(autouse=True)
def isolate_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def _ensure_default_source(service: PersistenceService) -> int:
    row = service._db_conn.execute(
        "SELECT source_id FROM sources LIMIT 1"
    ).fetchone()
    return int(row["source_id"])


def test_save_image_with_preset_controller_records_ptz_columns():
    controller = FakePtzController(
        {
            "ptz_origin": "preset",
            "ptz_preset_token": "Preset002",
            "ptz_zone": "center",
            "ptz_state": "tracking",
            "ptz_camera_id": 0,
            "ptz_pan": None,
            "ptz_tilt": None,
            "ptz_zoom": None,
            "ptz_position_at": None,
        }
    )
    service = PersistenceService(ptz_controller=controller)
    source_id = _ensure_default_source(service)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    capture_time = datetime(2026, 5, 15, 12, 0, 0)

    # Patch cv2.imwrite and the EXIF helper to avoid touching the
    # filesystem with a real JPEG/WebP encode.
    with (
        patch("detectors.services.persistence_service.cv2.imwrite", return_value=True),
        patch("detectors.services.persistence_service.add_exif_metadata"),
    ):
        result = service.save_image(
            frame=frame,
            capture_time=capture_time,
            detector_model_id="yolox-tiny",
            classifier_model_id="classifier-v1",
            source_id=source_id,
        )

    assert result.success
    row = service._db_conn.execute(
        "SELECT ptz_origin, ptz_preset_token, ptz_zone, ptz_state, "
        "ptz_camera_id FROM images WHERE filename = ?;",
        (result.base_filename,),
    ).fetchone()

    assert row["ptz_origin"] == "preset"
    assert row["ptz_preset_token"] == "Preset002"
    assert row["ptz_zone"] == "center"
    assert row["ptz_state"] == "tracking"
    assert row["ptz_camera_id"] == 0


def test_save_image_without_controller_records_null_ptz():
    service = PersistenceService(ptz_controller=None)
    source_id = _ensure_default_source(service)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    capture_time = datetime(2026, 5, 15, 12, 0, 1)

    with (
        patch("detectors.services.persistence_service.cv2.imwrite", return_value=True),
        patch("detectors.services.persistence_service.add_exif_metadata"),
    ):
        result = service.save_image(
            frame=frame,
            capture_time=capture_time,
            detector_model_id="yolox-tiny",
            classifier_model_id="classifier-v1",
            source_id=source_id,
        )

    assert result.success
    row = service._db_conn.execute(
        "SELECT ptz_origin, ptz_preset_token, ptz_zone, ptz_state, "
        "ptz_camera_id FROM images WHERE filename = ?;",
        (result.base_filename,),
    ).fetchone()

    assert row["ptz_origin"] is None
    assert row["ptz_preset_token"] is None
    assert row["ptz_zone"] is None
    assert row["ptz_state"] is None
    assert row["ptz_camera_id"] is None


def test_save_image_with_broken_controller_falls_back_to_null():
    """A buggy controller must not block image persistence."""
    service = PersistenceService(ptz_controller=BrokenPtzController())
    source_id = _ensure_default_source(service)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    capture_time = datetime(2026, 5, 15, 12, 0, 2)

    with (
        patch("detectors.services.persistence_service.cv2.imwrite", return_value=True),
        patch("detectors.services.persistence_service.add_exif_metadata"),
    ):
        result = service.save_image(
            frame=frame,
            capture_time=capture_time,
            detector_model_id="yolox-tiny",
            classifier_model_id="classifier-v1",
            source_id=source_id,
        )

    # Image still persists; PTZ columns NULL because the snapshot failed.
    assert result.success
    row = service._db_conn.execute(
        "SELECT ptz_origin FROM images WHERE filename = ?;",
        (result.base_filename,),
    ).fetchone()

    assert row["ptz_origin"] is None
