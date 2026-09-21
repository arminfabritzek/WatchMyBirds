"""The explicit-unknown lifecycle against a real database.

Reproduces the production sequence of detection 74759 on synthetic data: the
classifier proposes a species, a person answers "bird, species unknown", and
every surface must then agree that the species is unknown while the original
prediction survives as history.

Everything here writes to a throwaway SQLite file. No production record is
read for mutation and none is modified.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.human_label_core import DERIVED_FROM_SPECIES_ANSWER
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
            filename = f"{today}_094825_unknown.jpg"
            with db_connection.closing_connection() as conn:
                detection_id = _seed(
                    conn,
                    filename=filename,
                    timestamp=f"{today}_094825",
                    species=AI_SPECIES,
                )
            yield client, filename, detection_id


def _answer_unknown(client, filename: str, detection_id: int):
    return post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "species_identity": "unknown",
        },
    )


def _detection_row(detection_id: int) -> dict:
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT raw_species_name, manual_species_override, species_source,"
            " status, bbox_x, bbox_y, bbox_w, bbox_h"
            " FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
    return dict(row)


def _current_facts(detection_id: int) -> list[dict]:
    with db_connection.closing_connection() as conn:
        rows = conn.execute(
            "SELECT fact_type, answer_value, species_key, source_ref"
            " FROM current_human_label_facts WHERE detection_id = ?",
            (detection_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# --- what the answer actually stores ---------------------------------------


def test_unknown_answer_stores_one_species_fact_and_clears_the_override(case) -> None:
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    facts = _current_facts(detection_id)
    species_facts = [f for f in facts if f["fact_type"] == "species_identity"]
    assert len(species_facts) == 1
    assert species_facts[0]["answer_value"] == "unknown"
    assert not str(species_facts[0]["species_key"] or "").strip()

    row = _detection_row(detection_id)
    assert not str(row["manual_species_override"] or "").strip()
    assert row["species_source"] == "manual_unknown"


def test_unknown_answer_preserves_the_original_prediction_and_box(case) -> None:
    """The model's proposal is immutable history, not a value to overwrite."""
    client, filename, detection_id = case
    before = _detection_row(detection_id)
    assert _answer_unknown(client, filename, detection_id).status_code == 200
    after = _detection_row(detection_id)

    assert after["raw_species_name"] == before["raw_species_name"] == AI_SPECIES
    for axis in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
        assert after[axis] == before[axis]
    assert after["status"] == "active"


def test_unknown_answer_asserts_no_unrelated_axis(case) -> None:
    """Answering the species must not silently answer box quality or the image.

    ``bird_presence`` is the one exception and it is not silent: naming a
    species -- or declining to -- already says a bird is in the box, so the
    presence axis is filled from that answer and tagged as derived. Every
    other axis must stay untouched.
    """
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    types = {f["fact_type"] for f in _current_facts(detection_id)}
    assert types == {"species_identity", "bird_presence"}, (
        "the unknown answer asserted unrelated facts: " + repr(types)
    )


def test_derived_presence_is_marked_as_derived(case) -> None:
    """A derived fact must be auditable as such, not pass as a direct answer."""
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    facts = {f["fact_type"]: f for f in _current_facts(detection_id)}
    assert DERIVED_FROM_SPECIES_ANSWER in str(facts["bird_presence"]["source_ref"])
    assert DERIVED_FROM_SPECIES_ANSWER not in str(
        facts["species_identity"]["source_ref"] or ""
    ), "the species answer itself was given directly, not derived"


# --- how the surfaces then resolve the species -----------------------------


def test_species_resolution_reports_unknown_everywhere(case) -> None:
    """The shared SQL expression is what gallery/species/review all read."""
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    from utils.db.detections import effective_species_sql

    with db_connection.closing_connection() as conn:
        resolved = conn.execute(
            f"SELECT {effective_species_sql('d', conn)} AS s"
            " FROM detections d WHERE d.detection_id = ?",
            (detection_id,),
        ).fetchone()["s"]

    assert resolved != AI_SPECIES, "the withdrawn species came back"
    assert resolved == "Unknown_species"


def test_review_state_reports_reviewed_unknown(case) -> None:
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    from web.services import human_label_service

    with db_connection.closing_connection() as conn:
        states = human_label_service.fetch_detection_review_states(conn, [detection_id])

    assert states[detection_id]["state"] == "reviewed_unknown"
    assert states[detection_id]["reviewed"] is True
    assert states[detection_id]["species_key"] is None


def test_the_bird_stays_findable_and_is_not_deleted(case) -> None:
    """An unknown bird is still an active object a person can come back to."""
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    row = _detection_row(detection_id)
    assert row["status"] == "active"
    with db_connection.closing_connection() as conn:
        still_there = conn.execute(
            "SELECT COUNT(*) AS n FROM detections"
            " WHERE image_filename = ? AND COALESCE(status,'active') = 'active'",
            (filename,),
        ).fetchone()["n"]
    assert still_there == 1


# --- the species can be resolved later --------------------------------------


def test_a_later_species_answer_supersedes_the_unknown(case) -> None:
    """The unknown is a resting state, not a dead end."""
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "corrected",
            "species_key": "Parus_major",
        },
    )
    assert response.status_code == 200

    species_facts = [
        f for f in _current_facts(detection_id) if f["fact_type"] == "species_identity"
    ]
    assert len(species_facts) == 1, "the superseded unknown is still current"
    assert species_facts[0]["answer_value"] == "corrected"
    assert species_facts[0]["species_key"] == "Parus_major"

    row = _detection_row(detection_id)
    assert row["manual_species_override"] == "Parus_major"
    assert row["species_source"] != "manual_unknown"
    assert row["raw_species_name"] == AI_SPECIES, "model history was rewritten"


def test_resolving_the_species_restores_normal_resolution(case) -> None:
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200
    assert (
        post(
            client,
            "/api/labels/answer",
            {
                "filename": filename,
                "detection_id": detection_id,
                "object_bird_presence": "present",
                "species_identity": "corrected",
                "species_key": "Parus_major",
            },
        ).status_code
        == 200
    )

    from utils.db.detections import effective_species_sql

    with db_connection.closing_connection() as conn:
        resolved = conn.execute(
            f"SELECT {effective_species_sql('d', conn)} AS s"
            " FROM detections d WHERE d.detection_id = ?",
            (detection_id,),
        ).fetchone()["s"]
    assert resolved == "Parus_major"


# --- an ordinary confirm must not silently reinstate the AI species ---------


def test_confirming_after_unknown_requires_an_explicit_species(case) -> None:
    """A confirm carries the species it confirms; it is never implicit.

    This is the safety property behind "one Approve click must not turn a
    withdrawn Sitta europaea back into a positive label".
    """
    client, filename, detection_id = case
    assert _answer_unknown(client, filename, detection_id).status_code == 200

    response = post(
        client,
        "/api/labels/answer",
        {
            "filename": filename,
            "detection_id": detection_id,
            "object_bird_presence": "present",
            "species_identity": "confirmed",
        },
    )

    if response.status_code == 200:
        species_facts = [
            f
            for f in _current_facts(detection_id)
            if f["fact_type"] == "species_identity"
        ]
        assert species_facts[0]["species_key"] != AI_SPECIES, (
            "a confirm with no species key reinstated the withdrawn species"
        )
    else:
        assert response.status_code in (400, 409)

    row = _detection_row(detection_id)
    assert str(row["manual_species_override"] or "").strip() != AI_SPECIES, (
        "the withdrawn species was written back as a human-confirmed override"
    )
