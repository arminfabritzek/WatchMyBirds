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


@pytest.mark.parametrize("verdict", ["suitable", "unsuitable"])
def test_saved_box_verdict_survives_gallery_reload_and_relabel(case, verdict):
    from utils.db.detections import fetch_sibling_detections_batch

    client, filename, detection_id = case
    base = {"filename": filename, "detection_id": detection_id}
    response = post(client, "/api/labels/answer", {**base, "bbox_quality": verdict})
    assert response.status_code == 200
    assert response.get_json()["review_state"]["bbox_verdict"] == verdict
    # A later species-only correction must retain, not reassert, the box fact.
    with db_connection.closing_connection() as conn:
        before = conn.execute(
            "SELECT COUNT(*) FROM human_label_facts WHERE fact_type='bbox_quality'"
        ).fetchone()[0]
    response = post(
        client,
        "/api/labels/answer",
        {**base, "species_identity": "corrected", "species_key": "Parus_major"},
    )
    assert response.status_code == 200
    with db_connection.closing_connection() as conn:
        after = conn.execute(
            "SELECT COUNT(*) FROM human_label_facts WHERE fact_type='bbox_quality'"
        ).fetchone()[0]
        rows = fetch_sibling_detections_batch(conn, [filename])[filename]
    assert before == after
    assert dict(rows[0])["human_bbox_verdict"] == verdict
    date = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
    html = client.get(f"/gallery/{date}").get_data(as_text=True)
    assert f'"human_bbox_verdict": "{verdict}"' in html
    assert "Species 1/1" in html and "Boxes 1/1" in html
    # Species and box undo are atomic in one save.
    response = post(
        client,
        "/api/labels/answer",
        {
            **base,
            "retract_bbox_quality": True,
            "species_identity": "corrected",
            "species_key": "Cyanistes_caeruleus",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["review_state"]["bbox_verdict"] is None
    html = client.get(f"/gallery/{date}").get_data(as_text=True)
    assert '"human_bbox_verdict": null' in html
    assert "Species 1/1" in html and "Boxes 0/1" in html
    assert not any(f["fact_type"] == "bbox_quality" for f in _facts(detection_id))


def test_relabel_only_example_is_species_answered_box_unchecked(case):
    client, filename, detection_id = case
    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "corrected",
            "species_key": "Parus_major",
        },
    )
    assert response.status_code == 200
    assert response.get_json()["review_state"]["bbox_verdict"] is None
    readiness = object_training_readiness(_facts(detection_id))
    assert readiness["cls"]["ready"] is True
    assert "bbox_quality_unknown" in readiness["od"]["reasons"]


@pytest.mark.parametrize(
    "extra,status",
    [
        ({"retract_bbox_quality": "true"}, 400),
        ({"retract_bbox_quality": True, "bbox_quality": "suitable"}, 409),
        ({"retract_bbox_quality": True, "filename": "other.jpg"}, 409),
    ],
)
def test_invalid_retraction_never_changes_existing_fact(case, extra, status):
    client, filename, detection_id = case
    base = {"filename": filename, "detection_id": detection_id}
    assert (
        post(
            client, "/api/labels/answer", {**base, "bbox_quality": "suitable"}
        ).status_code
        == 200
    )
    before = _facts(detection_id)
    response = post(client, "/api/labels/answer", {**base, **extra})
    assert response.status_code == status
    assert _facts(detection_id) == before


def test_box_retraction_requires_authentication_and_csrf(case):
    client, filename, detection_id = case
    body = {
        "filename": filename,
        "detection_id": detection_id,
        "retract_bbox_quality": True,
    }
    assert client.post("/api/labels/answer", json=body).status_code == 403
    with client.session_transaction() as session:
        session.clear()
    assert post(client, "/api/labels/answer", body).status_code in {302, 401, 403}


def test_only_the_selected_bird_receives_a_box_assessment(case):
    from utils.db.detections import fetch_sibling_detections

    client, filename, detection_id = case
    with db_connection.closing_connection() as conn:
        other_id = _seed(conn, filename=filename, timestamp=filename[:15])
        conn.execute(
            "UPDATE detections SET decision_state='confirmed', decision_level='species'"
        )
    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "bbox_quality": "unsuitable",
        },
    )
    assert response.status_code == 200
    with db_connection.closing_connection() as conn:
        verdicts = {
            r["detection_id"]: r["human_bbox_verdict"]
            for r in fetch_sibling_detections(conn, filename)
        }
    assert verdicts[detection_id] == "unsuitable"
    assert verdicts[other_id] is None


def test_no_auth_required_to_read_saved_status_on_public_species_page(case):
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
    assert response.status_code == 200
    with client.session_transaction() as session:
        session.clear()
    html = client.get("/species").get_data(as_text=True)
    assert '"human_bbox_verdict": "suitable"' in html
