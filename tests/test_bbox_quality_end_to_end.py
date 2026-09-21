"""The editor's box answer, all the way to OD training readiness.

The point of asking the question is that a row becomes usable for
object-detection training. These drive the real HTTP endpoint the editor
calls and then read the readiness contract, so a change that stores the
fact but fails to move readiness cannot pass.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.human_label_core import object_training_readiness
from tests.labeling_helpers import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from web.web_interface import create_web_interface

AI_SPECIES = "Sitta_europaea"


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
            filename = f"{today}_101500_bbox.jpg"
            with db_connection.closing_connection() as conn:
                detection_id = _seed(
                    conn,
                    filename=filename,
                    timestamp=f"{today}_101500",
                    species=AI_SPECIES,
                )
            yield client, filename, detection_id


def _facts(detection_id: int) -> list[dict]:
    with db_connection.closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT f.fact_type, f.answer_value, f.species_key
            FROM current_human_label_facts f
            JOIN label_subjects s ON s.subject_id = f.subject_id
            WHERE s.detection_id = ?
            """,
            (detection_id,),
        ).fetchall()
    return [
        {
            "scope": "object",
            "fact_type": r["fact_type"],
            "answer_value": r["answer_value"],
            "species_key": r["species_key"],
        }
        for r in rows
    ]


def test_confirming_species_and_box_makes_the_row_od_ready(case):
    """The whole point: one save, both axes, usable for box training."""
    client, filename, detection_id = case

    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "corrected",
            "species_key": "Parus_major",
            "bbox_quality": "suitable",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)

    readiness = object_training_readiness(_facts(detection_id))
    assert readiness["od"]["ready"] is True, readiness["od"]["reasons"]
    assert readiness["cls"]["ready"] is True, readiness["cls"]["reasons"]


def test_a_species_only_answer_leaves_od_blocked_on_the_box(case):
    """Silence on the box axis must keep OD blocked, not quietly pass."""
    client, filename, detection_id = case

    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "corrected",
            "species_key": "Parus_major",
        },
    )

    readiness = object_training_readiness(_facts(detection_id))
    assert readiness["od"]["ready"] is False
    assert readiness["od"]["reasons"] == ["bbox_quality_unknown"]
    assert readiness["cls"]["ready"] is True


def test_an_unsuitable_box_is_recorded_and_blocks_od(case):
    client, filename, detection_id = case

    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "corrected",
            "species_key": "Parus_major",
            "bbox_quality": "unsuitable",
        },
    )

    readiness = object_training_readiness(_facts(detection_id))
    assert readiness["od"]["ready"] is False
    assert "bbox_unsuitable" in readiness["od"]["reasons"]
    assert readiness["cls"]["ready"] is True, (
        "a badly framed crop can still carry a usable species label"
    )


def test_an_unknown_species_with_a_good_box_is_od_ready_only(case):
    """The case this whole line of work started from."""
    client, filename, detection_id = case

    post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "unknown",
            "bbox_quality": "suitable",
        },
    )

    readiness = object_training_readiness(_facts(detection_id))
    assert readiness["od"]["ready"] is True, readiness["od"]["reasons"]
    assert readiness["cls"]["ready"] is False
    assert "species_unknown" in readiness["cls"]["reasons"]
