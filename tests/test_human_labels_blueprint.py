"""End-to-end contracts for canonical human-label APIs."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from config import get_config
from tests.labeling_helpers import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from utils.db.detections import fetch_sibling_detections, fetch_sibling_detections_batch
from web.web_interface import create_web_interface


@pytest.fixture
def labeling_case(monkeypatch, tmp_path):
    _reset_test_config(monkeypatch, tmp_path)
    detection_manager = MagicMock()
    detection_manager.frame_lock = nullcontext()
    detection_manager.latest_raw_timestamp = 0.0
    detection_manager.last_good_frame_timestamp = 0.0
    detection_manager._first_frame_received = False

    with (
        patch(
            "web.services.auth_service.should_require_password_setup",
            return_value=False,
        ),
        patch("web.services.auth_service.is_default_password", return_value=False),
    ):
        app = create_web_interface(detection_manager)
        app.config["TESTING"] = True
        with app.test_client() as client:
            with client.session_transaction() as session:
                session["authenticated"] = True
                session["_csrf_token"] = "test-csrf-token"
            today = datetime.now().strftime("%Y%m%d")
            filename = f"{today}_121500_labeling.jpg"
            with db_connection.closing_connection() as conn:
                detection_id = _seed(
                    conn,
                    filename=filename,
                    timestamp=f"{today}_121500",
                )
            yield client, filename, detection_id


def test_standalone_labeling_workspace_is_not_exposed(labeling_case) -> None:
    client, _, detection_id = labeling_case
    response = client.get(f"/admin/labeling?detection_id={detection_id}")

    assert response.status_code == 404
    with open("templates/partials/appbar.html", encoding="utf-8") as handle:
        appbar = handle.read()
    assert "/admin/labeling" not in appbar
    assert ">Labeling<" not in appbar.replace(" ", "").replace("\n", "")


def test_answer_api_records_independent_facts_and_projection(labeling_case) -> None:
    client, filename, detection_id = labeling_case
    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "image_bird_presence": "present",
            "object_bird_presence": "present",
            "bbox_quality": "suitable",
            "species_identity": "corrected",
            "species_key": "Cyanistes_caeruleus",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["status"] == "success"
    assert len(payload["fact_ids"]) == 4
    assert {fact["fact_type"] for fact in payload["facts"]} == {
        "bird_presence",
        "bbox_quality",
        "species_identity",
    }
    with db_connection.closing_connection() as conn:
        legacy = conn.execute(
            """
            SELECT i.review_status, d.manual_bbox_review,
                   d.manual_species_override, d.decision_level
            FROM images i
            JOIN detections d ON d.image_filename = i.filename
            WHERE d.detection_id = ?
            """,
            (detection_id,),
        ).fetchone()
    assert tuple(legacy) == (
        "confirmed_bird",
        "correct",
        "Cyanistes_caeruleus",
        "species",
    )


def test_object_reject_api_does_not_create_image_no_bird(labeling_case) -> None:
    client, filename, detection_id = labeling_case
    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "absent",
        },
    )

    assert response.status_code == 200
    with db_connection.closing_connection() as conn:
        image_status = conn.execute(
            "SELECT review_status FROM images WHERE filename = ?", (filename,)
        ).fetchone()[0]
        image_fact_count = conn.execute(
            """
            SELECT COUNT(*) FROM current_human_label_facts
            WHERE image_filename = ? AND scope = 'image'
            """,
            (filename,),
        ).fetchone()[0]
    assert image_status == "untagged"
    assert image_fact_count == 0


def test_answer_api_rejects_out_of_frame_bbox(labeling_case) -> None:
    client, filename, detection_id = labeling_case
    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "bbox_correction": {"x": 0.9, "y": 0.2, "w": 0.2, "h": 0.2},
        },
    )

    assert response.status_code == 409
    assert "frame" in response.get_json()["message"]


def test_direct_image_correction_preserves_proposal_and_records_usable_bbox(
    labeling_case,
) -> None:
    client, filename, detection_id = labeling_case
    corrected = {"x": 0.18, "y": 0.22, "w": 0.31, "h": 0.27}

    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "bbox_quality": "suitable",
            "bbox_correction": corrected,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    facts = response.get_json()["facts"]
    bbox_facts = {fact["fact_type"]: fact for fact in facts}
    assert bbox_facts["bbox_quality"]["answer_value"] == "suitable"
    assert {
        axis: bbox_facts["bbox_correction"][f"bbox_{axis}"]
        for axis in ("x", "y", "w", "h")
    } == corrected

    with db_connection.closing_connection() as conn:
        proposal = conn.execute(
            """
            SELECT proposal_bbox_x, proposal_bbox_y,
                   proposal_bbox_w, proposal_bbox_h
            FROM label_subjects
            WHERE detection_id = ?
            """,
            (detection_id,),
        ).fetchone()
        detection = conn.execute(
            """
            SELECT bbox_x, bbox_y, bbox_w, bbox_h, manual_bbox_review
            FROM detections WHERE detection_id = ?
            """,
            (detection_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE detections
            SET decision_state = 'confirmed', decision_level = 'species',
                quality_gallery_ok = 1, status = 'active'
            WHERE detection_id = ?
            """,
            (detection_id,),
        )
        rendered = fetch_sibling_detections(conn, filename)[0]

    assert tuple(proposal) == tuple(detection[:4])
    assert detection["manual_bbox_review"] == "correct"
    assert tuple(
        rendered[axis] for axis in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
    ) == pytest.approx(tuple(corrected[axis] for axis in ("x", "y", "w", "h")))


def test_label_state_exposes_active_object_readiness_and_partial_progress(
    labeling_case,
) -> None:
    client, filename, detection_id = labeling_case
    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "corrected",
            "species_key": "Cyanistes_caeruleus",
        },
    )
    assert response.status_code == 200

    state = client.get(
        "/api/labels/state",
        query_string={"filename": filename, "detection_id": detection_id},
    )

    assert state.status_code == 200
    payload = state.get_json()
    assert payload["readiness"]["od"] == {
        "ready": False,
        "reasons": ["bbox_quality_unknown"],
    }
    assert payload["readiness"]["cls"] == {"ready": True, "reasons": []}
    assert payload["object_progress"] == {
        "total": 1,
        "answered": 1,
        "unanswered": 0,
        "active_detection_id": detection_id,
        "active_fact_count": 2,
    }


def test_direct_bbox_correction_makes_the_object_od_ready(labeling_case) -> None:
    """Dragging a box states that a bird is there, so OD readiness follows.

    Species stays unanswered: correcting geometry is not an identification.
    """
    client, filename, detection_id = labeling_case

    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "bbox_quality": "suitable",
            "bbox_correction": {"x": 0.18, "y": 0.22, "w": 0.31, "h": 0.27},
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)

    with db_connection.closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT fact_type, answer_value
            FROM current_human_label_facts
            WHERE scope = 'object' AND detection_id = ?
            """,
            (detection_id,),
        ).fetchall()
    facts = {row["fact_type"]: row["answer_value"] for row in rows}

    assert facts.get("bird_presence") == "present"
    assert facts.get("bbox_quality") == "suitable"
    assert "species_identity" not in facts


def test_saving_a_corrected_box_rewrites_the_detection_crop(
    labeling_case, tmp_path
) -> None:
    """The crop must follow the box, or the person cannot see their own edit."""
    import cv2
    import numpy as np

    from utils.path_manager import PathManager

    client, filename, detection_id = labeling_case
    path_manager = PathManager(str(tmp_path / "output"))

    original = path_manager.get_original_path(filename)
    original.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 180, 255), -1)
    cv2.imwrite(str(original), frame)

    with db_connection.closing_connection() as conn:
        thumb_name = conn.execute(
            "SELECT thumbnail_path FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()[0]
    thumb = path_manager.get_derivative_path(thumb_name, "thumb")
    thumb.parent.mkdir(parents=True, exist_ok=True)
    thumb.write_bytes(b"stale-placeholder")

    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "bbox_quality": "suitable",
            "bbox_correction": {"x": 0.05, "y": 0.05, "w": 0.20, "h": 0.20},
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert thumb.read_bytes() != b"stale-placeholder"


def _make_original(filename: str) -> None:
    from utils.path_manager import PathManager

    original = PathManager(str(get_config()["OUTPUT_DIR"])).get_original_path(filename)
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"test-image")


def test_manual_object_create_is_idempotent_and_does_not_fabricate_detection(
    labeling_case,
) -> None:
    client, filename, detection_id = labeling_case
    _make_original(filename)
    payload = {
        "filename": filename,
        "bbox": {"x": 0.42, "y": 0.38, "w": 0.16, "h": 0.24},
        "species_key": "Cyanistes_caeruleus",
        "request_id": "manual-create-0001",
    }

    first = post(client, "/api/manual-objects", payload)
    second = post(client, "/api/manual-objects", payload)

    assert first.status_code == 200, first.get_data(as_text=True)
    assert first.get_json()["created"] is True
    assert second.status_code == 200
    assert second.get_json()["created"] is False
    assert second.get_json()["object"] == first.get_json()["object"]

    with db_connection.closing_connection() as conn:
        detection_count = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE image_filename = ?", (filename,)
        ).fetchone()[0]
        objects = conn.execute(
            "SELECT * FROM manual_objects WHERE image_filename = ?", (filename,)
        ).fetchall()
        revisions = conn.execute("SELECT * FROM manual_object_revisions").fetchall()
        event_count = conn.execute(
            "SELECT COUNT(*) FROM station_event_reviews"
        ).fetchone()[0]
        proposal = conn.execute(
            "SELECT bbox_x, bbox_y, bbox_w, bbox_h FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
        conn.execute(
            """
            UPDATE detections
            SET decision_state = 'confirmed', decision_level = 'species',
                quality_gallery_ok = 1, status = 'active'
            WHERE detection_id = ?
            """,
            (detection_id,),
        )
        siblings = fetch_sibling_detections(conn, filename)

    assert detection_count == 1
    assert len(objects) == 1
    assert len(revisions) == 1
    assert revisions[0]["asserted_facts"] == (
        '["bird_presence","bbox_geometry","species_identity"]'
    )
    assert event_count == 0
    assert tuple(proposal) == pytest.approx((0.2, 0.2, 0.3, 0.3))
    assert len(siblings) == 2
    manual = next(
        item for item in siblings if dict(item).get("manual_object_id") is not None
    )
    assert manual["detection_id"] is None
    assert manual["od_confidence"] is None
    assert manual["cls_confidence"] is None
    assert manual["species_key"] == "Cyanistes_caeruleus"
    assert manual["provenance"] == "manually_added"


def test_manual_object_unknown_species_and_axis_specific_updates(
    labeling_case,
) -> None:
    client, filename, detection_id = labeling_case
    _make_original(filename)
    created = post(
        client,
        "/api/manual-objects",
        {
            "filename": filename,
            "bbox": {"x": 0.4, "y": 0.3, "w": 0.2, "h": 0.25},
            "species_key": None,
            "request_id": "manual-create-unknown-0001",
        },
    )
    assert created.status_code == 200, created.get_data(as_text=True)
    obj = created.get_json()["object"]
    assert obj["species_state"] == "unknown"

    updated = client.patch(
        f"/api/manual-objects/{obj['manual_object_id']}",
        json={
            "filename": filename,
            "expected_revision": 1,
            "bbox": {"x": 0.41, "y": 0.31, "w": 0.18, "h": 0.23},
        },
        headers={"X-CSRF-Token": "test-csrf-token"},
    )
    assert updated.status_code == 200, updated.get_data(as_text=True)
    assert updated.get_json()["object"]["revision"] == 2
    assert updated.get_json()["object"]["species_state"] == "unknown"

    species_update = client.patch(
        f"/api/manual-objects/{obj['manual_object_id']}",
        json={
            "filename": filename,
            "expected_revision": 2,
            "species_key": "Parus_major",
        },
        headers={"X-CSRF-Token": "test-csrf-token"},
    )
    assert species_update.status_code == 200
    assert species_update.get_json()["object"]["species_key"] == "Parus_major"

    with db_connection.closing_connection() as conn:
        facts = conn.execute(
            """
            SELECT revision, asserted_facts
            FROM manual_object_revisions
            WHERE manual_object_id = ? ORDER BY revision
            """,
            (obj["manual_object_id"],),
        ).fetchall()
        detection_facts = conn.execute(
            """
            SELECT COUNT(*) FROM current_human_label_facts
            WHERE detection_id = ?
            """,
            (detection_id,),
        ).fetchone()[0]
    assert [row["asserted_facts"] for row in facts] == [
        '["bird_presence","bbox_geometry","species_identity"]',
        '["bbox_geometry"]',
        '["species_identity"]',
    ]
    assert detection_facts == 0


def test_manual_object_retraction_is_audited_and_removes_companion(
    labeling_case,
) -> None:
    client, filename, _ = labeling_case
    _make_original(filename)
    created = post(
        client,
        "/api/manual-objects",
        {
            "filename": filename,
            "bbox": {"x": 0.4, "y": 0.3, "w": 0.2, "h": 0.25},
            "species_key": "Parus_major",
            "request_id": "manual-retract-0001",
        },
    )
    obj = created.get_json()["object"]

    response = post(
        client,
        f"/api/manual-objects/{obj['manual_object_id']}/retract",
        {
            "filename": filename,
            "expected_revision": obj["revision"],
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.get_json()["object"]["status"] == "retracted"
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT status, revision FROM manual_objects WHERE manual_object_id = ?",
            (obj["manual_object_id"],),
        ).fetchone()
        revisions = conn.execute(
            """
            SELECT operation, asserted_facts FROM manual_object_revisions
            WHERE manual_object_id = ? ORDER BY revision
            """,
            (obj["manual_object_id"],),
        ).fetchall()
        siblings = fetch_sibling_detections(conn, filename)

    assert tuple(row) == ("retracted", 2)
    assert [(item["operation"], item["asserted_facts"]) for item in revisions] == [
        ("create", '["bird_presence","bbox_geometry","species_identity"]'),
        ("update", '["bird_presence"]'),
    ]
    assert all(dict(item).get("manual_object_id") is None for item in siblings)


def test_manual_bbox_update_requires_original_but_species_update_does_not(
    labeling_case,
) -> None:
    client, filename, _ = labeling_case
    _make_original(filename)
    created = post(
        client,
        "/api/manual-objects",
        {
            "filename": filename,
            "bbox": {"x": 0.4, "y": 0.3, "w": 0.2, "h": 0.25},
            "species_key": None,
            "request_id": "manual-retired-original-0001",
        },
    ).get_json()["object"]

    from utils.path_manager import PathManager

    PathManager(str(get_config()["OUTPUT_DIR"])).get_original_path(filename).unlink()
    geometry = client.patch(
        f"/api/manual-objects/{created['manual_object_id']}",
        json={
            "filename": filename,
            "expected_revision": 1,
            "bbox": {"x": 0.41, "y": 0.31, "w": 0.18, "h": 0.23},
        },
        headers={"X-CSRF-Token": "test-csrf-token"},
    )
    species = client.patch(
        f"/api/manual-objects/{created['manual_object_id']}",
        json={
            "filename": filename,
            "expected_revision": 1,
            "species_key": "Parus_major",
        },
        headers={"X-CSRF-Token": "test-csrf-token"},
    )

    assert geometry.status_code == 409
    assert "original is unavailable" in geometry.get_json()["message"]
    assert species.status_code == 200


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "bbox": {"x": 0.95, "y": 0.2, "w": 0.2, "h": 0.2},
                "species_key": None,
                "request_id": "manual-invalid-bbox",
            },
            "frame",
        ),
        (
            {
                "bbox": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
                "species_key": "Definitely_not_a_species",
                "request_id": "manual-invalid-species",
            },
            "catalog",
        ),
    ],
)
def test_manual_object_rejects_invalid_input(
    labeling_case, payload: dict, message: str
) -> None:
    client, filename, _ = labeling_case
    _make_original(filename)
    response = post(client, "/api/manual-objects", {"filename": filename, **payload})

    assert response.status_code == 409
    assert message in response.get_json()["message"]


def test_manual_object_requires_authentication(labeling_case) -> None:
    client, filename, _ = labeling_case
    _make_original(filename)
    with client.session_transaction() as session:
        session.pop("authenticated", None)

    response = post(
        client,
        "/api/manual-objects",
        {
            "filename": filename,
            "bbox": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
            "species_key": None,
            "request_id": "manual-auth-required",
        },
    )

    assert response.status_code in {302, 401}


def test_manual_object_reloads_in_shared_live_gallery_and_species_editor(
    labeling_case,
) -> None:
    client, filename, detection_id = labeling_case
    _make_original(filename)
    with db_connection.closing_connection() as conn:
        conn.execute(
            """
            UPDATE detections
            SET decision_state = 'confirmed', decision_level = 'species',
                quality_gallery_ok = 1, status = 'active'
            WHERE detection_id = ?
            """,
            (detection_id,),
        )

    created = post(
        client,
        "/api/manual-objects",
        {
            "filename": filename,
            "bbox": {"x": 0.52, "y": 0.35, "w": 0.15, "h": 0.22},
            "species_key": "Cyanistes_caeruleus",
            "request_id": "manual-cross-surface-0001",
        },
    )
    assert created.status_code == 200
    manual_id = created.get_json()["object"]["manual_object_id"]
    today_iso = datetime.now().strftime("%Y-%m-%d")

    pages = {
        "live": client.get("/"),
        "gallery": client.get(f"/gallery/{today_iso}"),
        "species": client.get("/species"),
        "species_overview": client.get("/species/overview?species_key=Parus_major"),
    }

    for surface, response in pages.items():
        assert response.status_code == 200, surface
        html = response.get_data(as_text=True)
        assert "data-bird-editor" in html, surface
        assert f"manual:{manual_id}" in html, surface
        assert "Add missing bird" in html, surface
        assert "Adjust box" in html, surface


def test_sibling_detections_report_each_detections_own_favorite_state(
    labeling_case,
) -> None:
    """A duplicate sibling row for the viewed detection must not overwrite a
    correct favorite flag with an unset one (regression: fetch_sibling_detections
    omitted is_favorite, so every sibling read back as unfavorited)."""
    _client, filename, favorited_id = labeling_case
    with db_connection.closing_connection() as conn:
        conn.execute(
            """
            UPDATE detections
            SET is_favorite = 1, decision_state = 'confirmed', decision_level = 'species'
            WHERE detection_id = ?
            """,
            (favorited_id,),
        )
        other_id = _seed(
            conn,
            filename=filename,
            timestamp="20260920_140800",
            species="Sitta_europaea",
        )
        conn.execute(
            """
            UPDATE detections
            SET decision_state = 'confirmed', decision_level = 'species'
            WHERE detection_id = ?
            """,
            (other_id,),
        )
        conn.commit()

        siblings = {
            row["detection_id"]: row for row in fetch_sibling_detections(conn, filename)
        }
        assert bool(siblings[favorited_id]["is_favorite"])
        assert not bool(siblings[other_id]["is_favorite"])

        batch = fetch_sibling_detections_batch(conn, [filename])
        batch_siblings = {row["detection_id"]: row for row in batch[filename]}
        assert bool(batch_siblings[favorited_id]["is_favorite"])
        assert not bool(batch_siblings[other_id]["is_favorite"])


@pytest.mark.parametrize(
    "species_key", ["cat", "squirrel", "marten_mustelid", "hedgehog"]
)
def test_manual_bird_rejects_nonbird_on_create_and_update(
    labeling_case, species_key: str
) -> None:
    client, filename, _ = labeling_case
    _make_original(filename)
    payload = {
        "filename": filename,
        "bbox": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2},
        "species_key": species_key,
        "request_id": "nonbird-create-test",
    }
    rejected = post(client, "/api/manual-objects", payload)
    assert rejected.status_code == 409
    with db_connection.closing_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM manual_objects").fetchone()[0] == 0
    created = post(
        client, "/api/manual-objects", {**payload, "species_key": None}
    ).get_json()["object"]
    rejected_update = client.patch(
        f"/api/manual-objects/{created['manual_object_id']}",
        json={
            "filename": filename,
            "expected_revision": created["revision"],
            "species_key": species_key,
        },
        headers={"X-CSRF-Token": "test-csrf-token"},
    )
    assert rejected_update.status_code == 409
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT species_key, revision FROM manual_objects WHERE manual_object_id = ?",
            (created["manual_object_id"],),
        ).fetchone()
        assert tuple(row) == (None, 1)
        assert (
            conn.execute("SELECT COUNT(*) FROM manual_object_revisions").fetchone()[0]
            == 1
        )
