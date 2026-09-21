"""The HTTP path behind taking back a species confirmation.

Confirming and un-confirming must be reachable from the same surface, so
this drives the real endpoints and then reads the columns every gallery
and species view depends on.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from tests.labeling_helpers import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from web.web_interface import create_web_interface

AI_SPECIES = "Cyanistes_caeruleus"


@pytest.fixture
def case(monkeypatch, tmp_path):
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
            filename = f"{today}_124151_undo.jpg"
            with db_connection.closing_connection() as conn:
                detection_id = _seed(
                    conn,
                    filename=filename,
                    timestamp=f"{today}_124151",
                    species=AI_SPECIES,
                )
            yield client, filename, detection_id


def _row(detection_id: int) -> dict:
    with db_connection.closing_connection() as conn:
        return dict(
            conn.execute(
                """
                SELECT manual_species_override, species_source,
                       decision_state, decision_level, raw_species_name
                FROM detections WHERE detection_id = ?
                """,
                (detection_id,),
            ).fetchone()
        )


def _confirm(client, filename, detection_id):
    return post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "confirmed",
            "species_key": AI_SPECIES,
        },
    )


def _undo(client, filename, detection_id):
    return post(
        client,
        "/api/labels/species/retract",
        {"filename": filename, "detection_id": detection_id},
    )


def test_confirm_then_undo_restores_the_ai_proposal(case):
    client, filename, detection_id = case
    assert _confirm(client, filename, detection_id).status_code == 200

    response = _undo(client, filename, detection_id)

    assert response.status_code == 200, response.get_data(as_text=True)
    row = _row(detection_id)
    assert not row["manual_species_override"]
    assert not row["species_source"]
    assert row["decision_state"] != "confirmed"
    assert row["raw_species_name"] == AI_SPECIES


def test_undo_is_idempotent(case):
    """A double click that races itself must not error out."""
    client, filename, detection_id = case
    _confirm(client, filename, detection_id)

    assert _undo(client, filename, detection_id).status_code == 200
    assert _undo(client, filename, detection_id).status_code == 200


def test_undo_requires_a_filename(case):
    client, _, detection_id = case
    response = post(
        client,
        "/api/labels/species/retract",
        {"filename": "", "detection_id": detection_id},
    )
    assert response.status_code == 400


def test_undo_rejects_a_mismatched_image(case):
    client, _, detection_id = case
    _confirm(client, "x", detection_id)
    response = post(
        client,
        "/api/labels/species/retract",
        {"filename": "20260101_000000_other.jpg", "detection_id": detection_id},
    )
    assert response.status_code == 400


def test_a_guest_cannot_undo(case):
    client, filename, detection_id = case
    _confirm(client, filename, detection_id)
    with client.session_transaction() as session:
        session["authenticated"] = False

    response = _undo(client, filename, detection_id)

    assert response.status_code in (302, 401, 403)
    assert _row(detection_id)["manual_species_override"] == AI_SPECIES


def _set_verdict(client, filename, detection_id, verdict):
    return post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "bbox_quality": verdict,
        },
    )


def _clear_verdict(client, filename, detection_id):
    return post(
        client,
        "/api/labels/bbox-quality/retract",
        {"filename": filename, "detection_id": detection_id},
    )


def _bbox_fact(detection_id: int):
    with db_connection.closing_connection() as conn:
        return conn.execute(
            """
            SELECT f.answer_value FROM current_human_label_facts f
            JOIN label_subjects s ON s.subject_id = f.subject_id
            WHERE s.detection_id = ? AND f.fact_type = 'bbox_quality'
            """,
            (detection_id,),
        ).fetchone()


def test_a_box_verdict_can_be_cleared_again(case):
    """Pressing the active verdict clears it, so a mis-click is undoable."""
    client, filename, detection_id = case
    assert _set_verdict(client, filename, detection_id, "suitable").status_code == 200
    assert _bbox_fact(detection_id)["answer_value"] == "suitable"

    assert _clear_verdict(client, filename, detection_id).status_code == 200

    assert _bbox_fact(detection_id) is None


def test_clearing_a_box_verdict_is_idempotent(case):
    client, filename, detection_id = case
    _set_verdict(client, filename, detection_id, "suitable")

    assert _clear_verdict(client, filename, detection_id).status_code == 200
    assert _clear_verdict(client, filename, detection_id).status_code == 200


def test_clearing_a_box_verdict_requires_a_filename(case):
    client, _, detection_id = case
    response = post(
        client,
        "/api/labels/bbox-quality/retract",
        {"filename": "", "detection_id": detection_id},
    )
    assert response.status_code == 400


def test_a_guest_cannot_clear_a_box_verdict(case):
    client, filename, detection_id = case
    _set_verdict(client, filename, detection_id, "suitable")
    with client.session_transaction() as session:
        session["authenticated"] = False

    response = _clear_verdict(client, filename, detection_id)

    assert response.status_code in (302, 401, 403)
    assert _bbox_fact(detection_id)["answer_value"] == "suitable"
