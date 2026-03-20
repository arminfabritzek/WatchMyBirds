import numpy as np
import pytest

from detectors.interfaces.persistence import DetectionData
from detectors.services.persistence_service import PersistenceService


@pytest.fixture(autouse=True)
def isolate_output_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def test_save_detection_persists_top5_predictions(monkeypatch):
    service = PersistenceService()
    monkeypatch.setattr(service, "generate_thumbnail", lambda *args, **kwargs: True)

    service._db_conn.execute(
        "INSERT INTO images (filename, timestamp) VALUES (?, ?)",
        ("test.jpg", "20260320_120000"),
    )
    service._db_conn.commit()

    detection = DetectionData(
        bbox=(10, 10, 110, 110),
        confidence=0.92,
        class_name="bird",
        cls_class_name="Parus_major",
        cls_confidence=0.85,
        score=0.88,
        top_k_predictions=[
            ("Cyanistes_caeruleus", 0.07),
            ("Sitta_europaea", 0.03),
            ("Erithacus_rubecula", 0.02),
            ("Picus_canus", 0.01),
        ],
    )

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = service.save_detection(
        "test.jpg",
        detection,
        frame,
        detector_model_id="detector-v1",
        classifier_model_id="classifier-v1",
        crop_index=1,
    )

    rows = service._db_conn.execute(
        """
        SELECT cls_class_name, cls_confidence, rank
        FROM classifications
        WHERE detection_id = ?
        ORDER BY rank ASC
        """,
        (result.detection_id,),
    ).fetchall()

    assert [(row["cls_class_name"], row["rank"]) for row in rows] == [
        ("Parus_major", 1),
        ("Cyanistes_caeruleus", 2),
        ("Sitta_europaea", 3),
        ("Erithacus_rubecula", 4),
        ("Picus_canus", 5),
    ]
    assert rows[0]["cls_confidence"] == pytest.approx(0.85)
    assert rows[4]["cls_confidence"] == pytest.approx(0.01)

    service.close()
