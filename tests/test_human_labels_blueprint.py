"""End-to-end contracts for canonical human-label APIs."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from tests.labeling_helpers import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from utils.db.detections import fetch_sibling_detections
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
