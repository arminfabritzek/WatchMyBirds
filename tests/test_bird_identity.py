"""Shared gallery/review identity and duplicate prevention contracts."""

from pathlib import Path

import pytest
from flask.testing import FlaskClient
from PIL import Image
from werkzeug.test import TestResponse

from config import get_config
from tests.labeling_helpers import _seed, post
from tests.test_human_labels_blueprint import labeling_case as _labeling_case
from utils.db import connection as db_connection
from utils.db.detections import fetch_detections_for_gallery, fetch_sibling_detections
from utils.db.review_queue import fetch_review_queue_images
from utils.path_manager import PathManager
from web.services.canonical_dataset_service import build_bundle

labeling_case = _labeling_case
LabelingCase = tuple[FlaskClient, str, int]


def original(filename: str) -> PathManager:
    pm = PathManager(str(get_config()["OUTPUT_DIR"]))
    path = pm.get_original_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1000, 800), "gray").save(path)
    return pm


def answer(
    client: FlaskClient, filename: str, detection_id: int, **changes: object
) -> TestResponse:
    response = post(
        client,
        "/api/labels/answer",
        dict(filename=filename, detection_id=detection_id, **changes),
    )
    assert response.status_code == 200, response.get_json()
    return response


@pytest.mark.parametrize("rejected", [False, True])
def test_matching_existing_bird_preserves_identity_and_history(
    labeling_case: LabelingCase, rejected: bool
) -> None:
    client, filename, detection_id = labeling_case
    pm = original(filename)
    before = pm.get_original_path(filename).read_bytes()
    if rejected:
        answer(client, filename, detection_id, object_bird_presence="absent")
    payload = dict(
        filename=filename,
        bbox=dict(x=0.21, y=0.21, w=0.28, h=0.28),
        species_key="Passer_domesticus",
        request_id="matching-existing-001",
    )
    conflict = post(client, "/api/manual-objects", payload)
    assert conflict.status_code == 409
    matches = conflict.get_json()
    assert matches["status"] == "match_required"
    assert [row["detection_id"] for row in matches["candidates"]] == [detection_id]
    with db_connection.closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 0
    payload.update(
        target_key=f"detection:{detection_id}", match_token=matches["match_token"]
    )
    saved = post(client, "/api/manual-objects", payload)
    assert saved.status_code == 200, saved.get_json()
    assert saved.get_json()["object"]["detection_id"] == detection_id
    with db_connection.closing_connection() as conn:
        fact_count = conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[
            0
        ]
        assert conn.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 0
        assert tuple(
            conn.execute(
                "SELECT bbox_x,bbox_y,bbox_w,bbox_h FROM detections WHERE detection_id=?",
                (detection_id,),
            ).fetchone()
        ) == (0.2, 0.2, 0.3, 0.3)
        assert fetch_review_queue_images(conn, filename=filename) == []
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        assert len(bundle.classifier_ready) == len(bundle.coco["annotations"]) == 1
        assert bundle.classifier_ready[0]["species_key"] == "Passer_domesticus"
        assert bundle.classifier_ready[0]["object_key"] == f"detection:{detection_id}"
    assert post(client, "/api/manual-objects", payload).status_code == 200
    with db_connection.closing_connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM human_label_facts").fetchone()[0]
            == fact_count
        )
    assert pm.get_original_path(filename).read_bytes() == before


def test_mixed_gallery_review_and_independent_training_views(
    labeling_case: LabelingCase,
) -> None:
    client, filename, tit = labeling_case
    pm = original(filename)
    with db_connection.closing_connection() as conn:
        sparrow = _seed(
            conn,
            filename=filename,
            timestamp="20261012_121500",
            species="Passer_domesticus",
        )
        conn.execute(
            "UPDATE detections SET bbox_x=.6, score=.2, decision_state='uncertain', decision_level='species_review' WHERE detection_id=?",
            (sparrow,),
        )
        conn.execute(
            "UPDATE detections SET score=.2, decision_state='uncertain', decision_level='species_review' WHERE detection_id=?",
            (tit,),
        )
        assert not fetch_detections_for_gallery(conn)
    answer(
        client,
        filename,
        tit,
        object_bird_presence="present",
        species_identity="confirmed",
        species_key="Parus_major",
        bbox_quality="suitable",
    )
    with db_connection.closing_connection() as conn:
        assert [r["detection_id"] for r in fetch_detections_for_gallery(conn)] == [tit]
        siblings = fetch_sibling_detections(conn, filename)
        assert len(siblings) == 2
        assert {r["detection_id"] for r in siblings if r["needs_review"]} == {sparrow}
        assert {
            r["active_detection_id"]
            for r in fetch_review_queue_images(conn, filename=filename)
        } == {sparrow}
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        assert [r["detection_id"] for r in bundle.classifier_ready] == [tit]
        assert not bundle.coco["images"]
    answer(
        client,
        filename,
        sparrow,
        object_bird_presence="present",
        species_identity="confirmed",
        species_key="Passer_domesticus",
        bbox_quality="suitable",
    )
    with db_connection.closing_connection() as conn:
        assert not fetch_review_queue_images(conn, filename=filename)
        assert len(fetch_sibling_detections(conn, filename)) == 2
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        assert len(bundle.coco["annotations"]) == len(bundle.classifier_ready) == 2


def test_new_manual_bird_reloads_and_exports_once(labeling_case: LabelingCase) -> None:
    client, filename, tit = labeling_case
    pm = original(filename)
    answer(
        client,
        filename,
        tit,
        object_bird_presence="present",
        species_identity="confirmed",
        species_key="Parus_major",
        bbox_quality="suitable",
    )
    payload = dict(
        filename=filename,
        bbox=dict(x=0.65, y=0.2, w=0.2, h=0.2),
        species_key="Passer_domesticus",
        request_id="genuinely-new-bird",
    )
    response = post(client, "/api/manual-objects", payload)
    assert response.status_code == 200, response.get_json()
    obj = response.get_json()["object"]
    assert post(client, "/api/manual-objects", payload).status_code == 200
    with db_connection.closing_connection() as conn:
        assert len(fetch_sibling_detections(conn, filename)) == 2
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        expected = {f"detection:{tit}", f"manual:{obj['manual_object_id']}"}
        assert {r["object_key"] for r in bundle.classifier_ready} == expected
        assert {r["object_key"] for r in bundle.coco["annotations"]} == expected
    conflict = post(
        client,
        "/api/manual-objects",
        {**payload, "request_id": "new-request-same-bird"},
    )
    assert conflict.status_code == 409
    assert (
        conflict.get_json()["candidates"][0]["manual_object_id"]
        == obj["manual_object_id"]
    )


def test_stale_match_cannot_overwrite_a_later_decision(
    labeling_case: LabelingCase,
) -> None:
    client, filename, detection_id = labeling_case
    original(filename)
    payload = dict(
        filename=filename,
        bbox=dict(x=0.2, y=0.2, w=0.3, h=0.3),
        species_key="Passer_domesticus",
        request_id="stale-match-request",
    )
    matches = post(client, "/api/manual-objects", payload).get_json()
    answer(client, filename, detection_id, object_bird_presence="absent")
    response = post(
        client,
        "/api/manual-objects",
        {
            **payload,
            "target_key": f"detection:{detection_id}",
            "match_token": matches["match_token"],
        },
    )
    assert response.status_code == 409
    assert response.get_json()["match_token"] != matches["match_token"]


def test_explicit_different_bird_keeps_two_overlapping_objects(
    labeling_case: LabelingCase,
) -> None:
    client, filename, detection_id = labeling_case
    original(filename)
    with db_connection.closing_connection() as conn:
        conn.execute(
            "UPDATE detections SET decision_state='uncertain', score=.2 WHERE detection_id=?",
            (detection_id,),
        )
    payload = dict(
        filename=filename,
        bbox=dict(x=0.2, y=0.2, w=0.3, h=0.3),
        species_key="Passer_domesticus",
        request_id="different-overlapping-bird",
    )
    matches = post(client, "/api/manual-objects", payload).get_json()
    saved = post(
        client,
        "/api/manual-objects",
        {**payload, "target_key": "new", "match_token": matches["match_token"]},
    )
    assert saved.status_code == 200
    assert saved.get_json()["object"]["manual_object_id"] is not None
    with db_connection.closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 1
        assert {
            r["active_detection_id"]
            for r in fetch_review_queue_images(conn, filename=filename)
        } == {detection_id}


def test_corrected_no_bird_is_not_exported_as_negative(
    labeling_case: LabelingCase, tmp_path: Path
) -> None:
    import json
    import zipfile

    from web.services.canonical_dataset_service import render_canonical_bundle_to_path

    client, filename, detection_id = labeling_case
    pm = original(filename)
    answer(
        client,
        filename,
        detection_id,
        image_bird_presence="absent",
        object_bird_presence="absent",
    )
    payload = dict(
        filename=filename,
        bbox=dict(x=0.2, y=0.2, w=0.3, h=0.3),
        species_key="Passer_domesticus",
        request_id="correct-previous-negative",
    )
    matches = post(client, "/api/manual-objects", payload).get_json()
    saved = post(
        client,
        "/api/manual-objects",
        {
            **payload,
            "target_key": f"detection:{detection_id}",
            "match_token": matches["match_token"],
        },
    )
    assert saved.status_code == 200
    with db_connection.closing_connection() as conn:
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        archive = render_canonical_bundle_to_path(
            bundle,
            path_resolver=pm.get_original_path,
            destination=tmp_path / "corrected.zip",
        )
    with zipfile.ZipFile(archive) as zipped:
        manifest = [
            json.loads(line) for line in zipped.read("manifest.jsonl").splitlines()
        ]
        assert not [
            r
            for r in manifest
            if r["view"] == "od_negative" and r["decision"] == "included"
        ]
        coco = json.loads(zipped.read("views/coco.json"))
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["object_key"] == f"detection:{detection_id}"
        facts = [json.loads(line) for line in zipped.read("facts.jsonl").splitlines()]
        assert any(r["answer_value"] == "absent" for r in facts)
        assert any(
            r["answer_value"] == "present" and r["supersedes_fact_id"] for r in facts
        )


def test_manual_only_frame_has_geometry_and_auditable_archive(
    labeling_case: LabelingCase, tmp_path: Path
) -> None:
    import json
    import zipfile

    from web.services.canonical_dataset_service import render_canonical_bundle_to_path

    client, filename, detection_id = labeling_case
    pm = original(filename)
    answer(client, filename, detection_id, object_bird_presence="absent")
    response = post(
        client,
        "/api/manual-objects",
        dict(
            filename=filename,
            bbox=dict(x=0.65, y=0.2, w=0.2, h=0.2),
            species_key="Passer_domesticus",
            request_id="manual-archive-only-bird",
        ),
    )
    obj = response.get_json()["object"]
    with db_connection.closing_connection() as conn:
        conn.execute("UPDATE detections SET frame_width=NULL,frame_height=NULL")
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        archive = render_canonical_bundle_to_path(
            bundle,
            path_resolver=pm.get_original_path,
            destination=tmp_path / "manual.zip",
        )
    with zipfile.ZipFile(archive) as zipped:
        coco = json.loads(zipped.read("views/coco.json"))
        assert coco["images"][0]["width"] == 1000
        assert coco["images"][0]["height"] == 800
        assert len(coco["annotations"]) == 1
        assert coco["annotations"][0]["bbox"] == [650, 160, 200, 160]
        cls = [
            json.loads(line)
            for line in zipped.read("views/classifier_ready.jsonl").splitlines()
        ]
        assert len(cls) == 1
        assert cls[0]["object_key"] == f"manual:{obj['manual_object_id']}"
        manual = [
            json.loads(line)
            for line in zipped.read("manual_objects.jsonl").splitlines()
        ]
        assert len(manual) == 1
        assert "od_confidence" not in manual[0]
        assert "classifier_model_version" not in manual[0]
        assert zipped.read("manual_object_revisions.jsonl")


def test_ambiguous_match_corrects_only_the_chosen_bird(
    labeling_case: LabelingCase,
) -> None:
    client, filename, first = labeling_case
    pm = original(filename)
    with db_connection.closing_connection() as conn:
        second = _seed(conn, filename=filename, timestamp="20261012_121500")
        conn.execute(
            "UPDATE detections SET decision_state='uncertain', score=.2 WHERE image_filename=?",
            (filename,),
        )
    payload = dict(
        filename=filename,
        bbox=dict(x=0.2, y=0.2, w=0.3, h=0.3),
        species_key="Passer_domesticus",
        request_id="ambiguous-bird-selection",
    )
    conflict = post(client, "/api/manual-objects", payload)
    assert conflict.status_code == 409
    matches = conflict.get_json()
    assert {c["detection_id"] for c in matches["candidates"]} == {first, second}
    no_choice = post(
        client,
        "/api/manual-objects",
        {**payload, "match_token": matches["match_token"]},
    )
    assert no_choice.status_code == 409
    saved = post(
        client,
        "/api/manual-objects",
        {
            **payload,
            "match_token": matches["match_token"],
            "target_key": f"detection:{second}",
        },
    )
    assert saved.status_code == 200
    with db_connection.closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 0
        assert {
            r["active_detection_id"]
            for r in fetch_review_queue_images(conn, filename=filename)
        } == {first}
        bundle = build_bundle(conn, path_resolver=pm.get_original_path)
        assert [r["detection_id"] for r in bundle.classifier_ready] == [second]
        assert not bundle.coco["annotations"]
