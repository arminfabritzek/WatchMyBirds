"""Regression contracts across correction, export, restoration and file cleanup."""

import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from flask.testing import FlaskClient

from config import get_config
from core.gallery_core import regenerate_derivative
from scripts.fiftyone_import_canonical import (
    build_sample_specs,
    import_into_fiftyone,
    prepare_bundle,
)
from tests.labeling_helpers import _seed, post
from tests.test_fiftyone_canonical_import import _fake_fiftyone
from tests.test_human_labels_blueprint import labeling_case as _labeling_case
from utils.db import connection as db_connection
from utils.db.detections import (
    fetch_detections_for_gallery,
    fetch_sibling_detections,
    fetch_sibling_detections_batch,
)
from utils.file_gc import hard_delete_detections
from utils.path_manager import PathManager
from web.services.canonical_dataset_service import (
    build_bundle,
    render_canonical_bundle_to_path,
)

labeling_case = _labeling_case
LabelingCase = tuple[FlaskClient, str, int]


def setup_image(filename: str) -> PathManager:
    pm = PathManager(str(get_config()["OUTPUT_DIR"]))
    original = pm.get_original_path(filename)
    original.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    frame[:, :500] = (255, 0, 0)
    frame[:, 500:] = (0, 255, 0)
    assert cv2.imwrite(str(original), frame)
    return pm


def add_manual_and_reject_proposal(
    client: FlaskClient, filename: str, detection_id: int
) -> PathManager:
    pm = setup_image(filename)
    response = post(
        client,
        "/api/manual-objects",
        dict(
            filename=filename,
            bbox=dict(x=0.65, y=0.2, w=0.2, h=0.2),
            species_key="Passer_domesticus",
            request_id="review-manual-001",
        ),
    )
    assert response.status_code == 200, response.get_json()
    response = post(
        client,
        "/api/labels/answer",
        dict(
            filename=filename, detection_id=detection_id, object_bird_presence="absent"
        ),
    )
    assert response.status_code == 200, response.get_json()
    return pm


def test_purging_rejected_proposal_preserves_active_manual_bird(
    labeling_case: LabelingCase,
) -> None:
    client, filename, detection_id = labeling_case
    pm = add_manual_and_reject_proposal(client, filename, detection_id)
    assert (
        post(client, "/api/trash/move", dict(detection_ids=[detection_id])).status_code
        == 200
    )
    optimized = pm.get_derivative_path(filename, "optimized")
    optimized.parent.mkdir(parents=True, exist_ok=True)
    optimized.write_bytes(b"retained-preview")
    thumbnail = pm.get_derivative_path(
        filename.replace(".jpg", "_crop_1.webp"), "thumb"
    )
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.write_bytes(b"rejected-thumbnail")
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM manual_objects WHERE status='active'"
            ).fetchone()[0]
            == 1
        )
        result = hard_delete_detections(conn, [detection_id])
        assert (
            conn.execute("SELECT COUNT(*) FROM manual_object_revisions").fetchone()[0]
            == 1
        )
        actual = dict(
            original_exists=pm.get_original_path(filename).is_file(),
            image_count=conn.execute(
                "SELECT COUNT(*) FROM images WHERE filename=?", (filename,)
            ).fetchone()[0],
            manual_count=conn.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[
                0
            ],
        )
    assert actual == dict(original_exists=True, image_count=1, manual_count=1), (
        result,
        actual,
    )
    assert optimized.read_bytes() == b"retained-preview"
    assert not thumbnail.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_regeneration_keeps_corrected_crop(
    labeling_case: LabelingCase, legacy: bool
) -> None:
    client, filename, detection_id = labeling_case
    pm = setup_image(filename)
    response = post(
        client,
        "/api/labels/answer",
        dict(
            filename=filename,
            detection_id=detection_id,
            bbox_correction=dict(x=0.65, y=0.2, w=0.2, h=0.2),
            bbox_quality="suitable",
        ),
    )
    assert response.status_code == 200, response.get_json()
    thumb = pm.get_derivative_path(filename.replace(".jpg", "_crop_1.webp"), "thumb")
    corrected_bytes = thumb.read_bytes()
    original_hash = hashlib.sha256(
        pm.get_original_path(filename).read_bytes()
    ).hexdigest()
    if legacy:
        with db_connection.closing_connection() as conn:
            conn.execute(
                "UPDATE detections SET thumbnail_path=NULL WHERE detection_id=?",
                (detection_id,),
            )
    thumb.unlink()
    assert regenerate_derivative(str(get_config()["OUTPUT_DIR"]), thumb.name, "thumb")
    regenerated_pixel = cv2.imread(str(thumb))[128, 128].tolist()
    assert regenerated_pixel[1] > 200
    assert thumb.read_bytes() == corrected_bytes
    assert (
        hashlib.sha256(pm.get_original_path(filename).read_bytes()).hexdigest()
        == original_hash
    )


@pytest.mark.parametrize("projected_status", ["no_bird", "confirmed_bird", "untagged"])
def test_box_verdict_does_not_undo_full_image_absence(
    labeling_case: LabelingCase,
    projected_status: str,
) -> None:
    client, filename, detection_id = labeling_case
    with db_connection.closing_connection() as conn:
        conn.execute(
            "UPDATE detections SET decision_state='uncertain', decision_level='species_review', score=.2 WHERE detection_id=?",
            (detection_id,),
        )
    response = post(
        client,
        "/api/labels/answer",
        dict(filename=filename, image_bird_presence="absent"),
    )
    assert response.status_code == 200, response.get_json()
    with db_connection.closing_connection() as conn:
        conn.execute(
            "UPDATE images SET review_status=? WHERE filename=?",
            (projected_status, filename),
        )
    response = post(
        client,
        "/api/labels/answer",
        dict(filename=filename, detection_id=detection_id, bbox_quality="unsuitable"),
    )
    assert response.status_code == 200, response.get_json()
    with db_connection.closing_connection() as conn:
        status = conn.execute(
            "SELECT review_status FROM images WHERE filename=?", (filename,)
        ).fetchone()[0]
        fact = conn.execute(
            "SELECT answer_value FROM current_human_label_facts WHERE scope='image' AND fact_type='bird_presence'"
        ).fetchone()[0]
    assert status == projected_status
    assert fact == "absent"


def test_manual_bundle_is_accepted_by_fiftyone_importer(
    labeling_case: LabelingCase, tmp_path: Path
) -> None:
    client, filename, detection_id = labeling_case
    pm = add_manual_and_reject_proposal(client, filename, detection_id)
    with db_connection.closing_connection() as conn:
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        archive = render_canonical_bundle_to_path(
            bundle,
            path_resolver=pm.get_original_path,
            destination=tmp_path / "bundle.zip",
        )
    extracted = prepare_bundle(archive, tmp_path / "extracted")
    metadata, specs = build_sample_specs(extracted)
    dataset = import_into_fiftyone(metadata, specs, fo_module=_fake_fiftyone())
    sample = next(iter(dataset.samples.values()))
    for field in ("od_ground_truth", "cls_ground_truth"):
        labels = sample[field].detections
        assert len(labels) == 1
        assert labels[0].values["wmb_detection_id"] is None
        assert labels[0].values["wmb_object_key"].startswith("manual:")
        assert labels[0].values["wmb_manual_object_id"] is not None


def test_restore_retains_identity_history_and_independent_axes(
    labeling_case: LabelingCase,
) -> None:
    client, filename, detection_id = labeling_case
    pm = setup_image(filename)
    original = pm.get_original_path(filename).read_bytes()
    corrected = dict(x=0.65, y=0.2, w=0.2, h=0.2)
    response = post(
        client,
        "/api/labels/answer",
        dict(
            filename=filename,
            detection_id=detection_id,
            object_bird_presence="absent",
            bbox_quality="unsuitable",
            bbox_correction=corrected,
        ),
    )
    assert response.status_code == 200
    rejected_fact = next(
        f["fact_id"]
        for f in response.get_json()["facts"]
        if f["fact_type"] == "bird_presence"
    )
    assert (
        post(
            client,
            "/api/labels/answer",
            dict(filename=filename, image_bird_presence="absent"),
        ).status_code
        == 200
    )
    with db_connection.closing_connection() as conn:
        assert fetch_sibling_detections(conn, filename) == []
        siblings = fetch_sibling_detections(conn, filename, include_rejected=True)
        batch = fetch_sibling_detections_batch(conn, [filename], include_rejected=True)[
            filename
        ]
        assert [dict(r) for r in siblings] == [
            {k: v for k, v in dict(r).items() if k != "image_filename"} for r in batch
        ]
        assert siblings[0]["object_status"] == "active"
        proposal = tuple(
            conn.execute(
                "SELECT bbox_x,bbox_y,bbox_w,bbox_h FROM detections WHERE detection_id=?",
                (detection_id,),
            ).fetchone()
        )
    response = post(
        client, "/api/labels/answer", dict(filename=filename, retract="bird_presence")
    )
    assert response.status_code == 200, response.get_json()
    assert response.get_json()["review_state"] is None
    with db_connection.closing_connection() as conn:
        facts = conn.execute(
            "SELECT * FROM current_human_label_facts WHERE image_filename=?",
            (filename,),
        ).fetchall()
        assert not any(f["fact_type"] == "bird_presence" for f in facts)
        assert not any(f["fact_type"] == "species_identity" for f in facts)
        assert conn.execute(
            "SELECT 1 FROM human_label_facts WHERE fact_id=?", (rejected_fact,)
        ).fetchone()
        row = fetch_sibling_detections(conn, filename)[0]
        assert row["object_status"] == "active"
        assert tuple(row[f"bbox_{axis}"] for axis in ("x", "y", "w", "h")) == tuple(
            corrected.values()
        )
        assert (
            tuple(
                conn.execute(
                    "SELECT bbox_x,bbox_y,bbox_w,bbox_h FROM detections WHERE detection_id=?",
                    (detection_id,),
                ).fetchone()
            )
            == proposal
        )
        assert (
            conn.execute(
                "SELECT status FROM classifications WHERE detection_id=?",
                (detection_id,),
            ).fetchone()[0]
            == "active"
        )
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        assert not bundle.coco["annotations"]
        assert not bundle.classifier_ready
        count = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
    assert (
        post(
            client,
            "/api/labels/answer",
            dict(filename=filename, retract="bird_presence"),
        ).status_code
        == 200
    )
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
            == count
        )
    assert pm.get_original_path(filename).read_bytes() == original


@pytest.mark.parametrize(
    "surface",
    ["/", "/species", "/species/overview?species_key=Parus_major", "gallery", "review"],
)
def test_dismissed_box_is_in_detail_even_with_only_one_active_bird(
    labeling_case: LabelingCase, surface: str
) -> None:
    client, filename, active_id = labeling_case
    setup_image(filename)
    with db_connection.closing_connection() as conn:
        dismissed_id = _seed(
            conn, filename=filename, timestamp=filename[:15], species="Sitta_europaea"
        )
        conn.execute(
            "UPDATE detections SET decision_state='confirmed', decision_level='species' WHERE detection_id=?",
            (active_id,),
        )
    assert (
        post(
            client,
            "/api/labels/answer",
            dict(
                filename=filename,
                detection_id=dismissed_id,
                object_bird_presence="absent",
            ),
        ).status_code
        == 200
    )
    route = f"/gallery/{datetime.now():%Y-%m-%d}" if surface == "gallery" else surface
    with db_connection.closing_connection() as conn:
        rows = fetch_detections_for_gallery(conn, datetime.now().strftime("%Y-%m-%d"))
        assert {r["detection_id"] for r in rows} == {active_id}
        assert rows[0]["sibling_count"] == 1
        if surface == "review":
            conn.execute(
                "UPDATE images SET review_status='untagged' WHERE filename=?",
                (filename,),
            )
            conn.execute(
                "UPDATE detections SET decision_state='uncertain', decision_level='species_review' WHERE detection_id=?",
                (active_id,),
            )
            route = f"/api/review/modal/bird-event-000000000000/{active_id}"
    response = client.get(route)
    assert response.status_code == 200

    text = response.get_data(as_text=True)
    rows = [
        r
        for value in re.findall(r"data-siblings='([^']*)'", text)
        for r in json.loads(html.unescape(value))
    ]
    dismissed = [r for r in rows if r.get("detection_id") == dismissed_id]
    assert dismissed, route
    assert all(
        r["object_status"] == "active"
        and r["human_review_state"] == "reviewed_negative"
        for r in dismissed
    )
    assert 'data-editor-action="restore"' in text


def test_restore_rejects_cross_image_and_missing_auth_or_csrf(
    labeling_case: LabelingCase,
) -> None:
    client, filename, detection_id = labeling_case
    assert (
        post(
            client,
            "/api/labels/answer",
            dict(
                filename=filename,
                detection_id=detection_id,
                object_bird_presence="absent",
            ),
        ).status_code
        == 200
    )
    url = "/api/labels/answer"
    assert (
        post(
            client,
            url,
            dict(
                filename="other.jpg", detection_id=detection_id, retract="bird_presence"
            ),
        ).status_code
        == 409
    )
    assert post(client, url, {}).status_code == 409
    assert client.post(url, json=dict(filename=filename)).status_code == 403
    with client.session_transaction() as session:
        session.pop("authenticated", None)
    assert post(client, url, dict(filename=filename)).status_code in (302, 401)
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute(
                "SELECT status FROM detections WHERE detection_id=?", (detection_id,)
            ).fetchone()[0]
            == "active"
        )
