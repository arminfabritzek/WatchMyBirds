"""Pin the five-axis outcome of every endpoint that records a human label.

One human statement ("this is a Blaumeise") must land on four coupled
columns of ``detections`` — ``decision_state``, ``decision_level``,
``species_source``, ``manual_species_override`` — plus
``images.review_status`` on the frame. Seven endpoints write related blocks
by hand today.

These tests run against a real SQLite database and assert the resulting
row state, not that some SQL was issued: the existing endpoint tests use
MagicMock connections, which cannot tell a correct write from a
forgotten column. They exist so the migration onto
``label_answer_service`` can be shown to preserve behaviour rather than
merely asserted to.

Covered endpoints:
- POST /api/review/quick-species
- POST /api/review/approve
- POST /api/review/event-approve
- POST /api/review/event-resolve
- POST /api/moderation/bulk/relabel
- POST /api/moderation/rescan-proposals/<id>/apply
- POST /api/detections/relabel
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from tests.labeling_helpers import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from web.web_interface import create_web_interface

SPECIES_AXES = (
    "decision_state",
    "decision_level",
    "species_source",
    "manual_species_override",
)

# Both are in the shipped picker list, so the endpoints' species
# validation accepts them.
TARGET_SPECIES = "Cyanistes_caeruleus"
PRIOR_SPECIES = "Parus_major"


def axes(detection_id: int) -> dict[str, str | None]:
    """Read the four detection-level species axes back from the DB."""
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            f"SELECT {', '.join(SPECIES_AXES)} FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
    return dict(zip(SPECIES_AXES, row, strict=True))


def review_status(filename: str) -> str | None:
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT review_status FROM images WHERE filename = ?", (filename,)
        ).fetchone()
    return row[0] if row else None


def current_facts(detection_id: int) -> dict[str, dict[str, str | None]]:
    """Read current canonical object facts keyed by semantic axis."""
    with db_connection.closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT fact_type, answer_value, species_key, source_kind, source_ref
            FROM current_human_label_facts
            WHERE scope = 'object' AND detection_id = ?
            ORDER BY fact_type
            """,
            (detection_id,),
        ).fetchall()
    return {
        row["fact_type"]: {
            "answer_value": row["answer_value"],
            "species_key": row["species_key"],
            "source_kind": row["source_kind"],
            "source_ref": row["source_ref"],
        }
        for row in rows
    }


def assert_confirmed_species(detection_id: int, species: str) -> None:
    """The invariant the eight hand-written blocks each try to uphold."""
    state = axes(detection_id)
    assert state["decision_state"] == "confirmed"
    assert state["decision_level"] == "species"
    assert state["species_source"] == "manual"
    assert state["manual_species_override"] == species


@pytest.fixture
def label_app(monkeypatch, tmp_path):
    output_dir = _reset_test_config(monkeypatch, tmp_path)

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
        yield app, output_dir


@pytest.fixture
def client(label_app):
    app, _ = label_app
    with app.test_client() as test_client:
        with test_client.session_transaction() as session:
            session["authenticated"] = True
            session["_csrf_token"] = "test-csrf-token"
        yield test_client


@pytest.fixture
def seeded(client):
    """One frame with two detections, plus a second single-detection frame."""
    today = datetime.now().strftime("%Y%m%d")
    frame_a = f"{today}_100000_a.jpg"
    frame_b = f"{today}_101000_b.jpg"

    with db_connection.closing_connection() as conn:
        det_a1 = _seed(conn, filename=frame_a, timestamp=f"{today}_100000")
        det_a2 = _seed(conn, filename=frame_a, timestamp=f"{today}_100000")
        det_b = _seed(conn, filename=frame_b, timestamp=f"{today}_101000")

    return {
        "client": client,
        "frame_a": frame_a,
        "frame_b": frame_b,
        "a1": det_a1,
        "a2": det_a2,
        "b": det_b,
    }


# --- 1. POST /api/review/quick-species -------------------------------------


def test_quick_species_sets_every_axis(seeded):
    response = post(
        seeded["client"],
        "/api/review/quick-species",
        {
            "filename": seeded["frame_b"],
            "detection_id": seeded["b"],
            "species": TARGET_SPECIES,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert_confirmed_species(seeded["b"], TARGET_SPECIES)
    assert current_facts(seeded["b"]) == {
        "species_identity": {
            "answer_value": "corrected",
            "species_key": TARGET_SPECIES,
            "source_kind": "watchmybirds_ui",
            "source_ref": "review:quick-species",
        }
    }


def test_quick_species_leaves_the_frame_review_status_alone(seeded):
    """Asymmetry worth pinning: quick-species writes the species axes but
    never touches ``images.review_status``, while ``/api/review/approve``
    refreshes it. Migrating this site onto a service that always sets
    ``confirmed_bird`` would silently change behaviour here."""
    post(
        seeded["client"],
        "/api/review/quick-species",
        {
            "filename": seeded["frame_b"],
            "detection_id": seeded["b"],
            "species": TARGET_SPECIES,
        },
    )

    assert review_status(seeded["frame_b"]) == "untagged"


def test_quick_species_leaves_other_detections_untouched(seeded):
    post(
        seeded["client"],
        "/api/review/quick-species",
        {
            "filename": seeded["frame_a"],
            "detection_id": seeded["a1"],
            "species": TARGET_SPECIES,
        },
    )

    assert axes(seeded["a2"])["decision_state"] is None


# --- 2. POST /api/review/approve -------------------------------------------


def test_bbox_review_clear_retracts_the_canonical_answer(seeded):
    saved = post(
        seeded["client"],
        "/api/review/bbox-review",
        {
            "filename": seeded["frame_b"],
            "detection_id": seeded["b"],
            "bbox_review": "correct",
        },
    )
    cleared = post(
        seeded["client"],
        "/api/review/bbox-review",
        {
            "filename": seeded["frame_b"],
            "detection_id": seeded["b"],
            "bbox_review": None,
        },
    )

    assert saved.status_code == 200
    assert cleared.status_code == 200
    assert "bbox_quality" not in current_facts(seeded["b"])
    with db_connection.closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT assertion_state
            FROM human_label_facts f
            JOIN label_subjects s ON s.subject_id = f.subject_id
            WHERE s.detection_id = ? AND f.fact_type = 'bbox_quality'
            ORDER BY f.fact_id
            """,
            (seeded["b"],),
        ).fetchall()
    assert [row[0] for row in rows] == ["asserted", "retracted"]


def test_approve_sets_every_axis(seeded):
    response = post(
        seeded["client"],
        "/api/review/approve",
        {
            "filename": seeded["frame_b"],
            "detection_id": seeded["b"],
            "species": TARGET_SPECIES,
            "bbox_review": "correct",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert_confirmed_species(seeded["b"], TARGET_SPECIES)
    facts = current_facts(seeded["b"])
    assert {
        fact_type: (fact["answer_value"], fact["species_key"])
        for fact_type, fact in facts.items()
    } == {
        "bbox_quality": ("suitable", None),
        "bird_presence": ("present", None),
        "species_identity": ("corrected", TARGET_SPECIES),
    }


def test_approve_scopes_the_write_to_the_named_frame(seeded):
    """approve carries an extra image_filename guard the others lack."""
    response = post(
        seeded["client"],
        "/api/review/approve",
        {
            "filename": seeded["frame_a"],
            "detection_id": seeded["b"],
            "species": TARGET_SPECIES,
            "bbox_review": "correct",
        },
    )

    # detection b lives on frame_b, so the frame-scoped UPDATE must miss it.
    if response.status_code == 200:
        assert axes(seeded["b"])["decision_state"] != "confirmed"


# --- 3. POST /api/review/event-approve -------------------------------------


def test_event_approve_sets_every_axis_for_the_whole_batch(seeded):
    response = post(
        seeded["client"],
        "/api/review/event-approve",
        {
            "detection_ids": [seeded["a1"], seeded["a2"]],
            "species": TARGET_SPECIES,
            "bbox_review": "correct",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    for detection_id in (seeded["a1"], seeded["a2"]):
        assert_confirmed_species(detection_id, TARGET_SPECIES)


def test_event_approve_without_bbox_answer_leaves_bbox_unknown(seeded):
    response = post(
        seeded["client"],
        "/api/review/event-approve",
        {
            "detection_ids": [seeded["a1"], seeded["a2"]],
            "species": TARGET_SPECIES,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    for detection_id in (seeded["a1"], seeded["a2"]):
        facts = current_facts(detection_id)
        assert facts["bird_presence"]["answer_value"] == "present"
        assert facts["species_identity"]["species_key"] == TARGET_SPECIES
        assert "bbox_quality" not in facts


# --- 4. POST /api/review/event-resolve -------------------------------------


def test_event_resolve_confirms_keeps_and_rejects_trash_together(seeded):
    response = post(
        seeded["client"],
        "/api/review/event-resolve",
        {
            "keep_detection_ids": [seeded["a1"]],
            "trash_detection_ids": [seeded["a2"]],
            "species": TARGET_SPECIES,
            "bbox_review": "correct",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert_confirmed_species(seeded["a1"], TARGET_SPECIES)

    with db_connection.closing_connection() as conn:
        status = conn.execute(
            "SELECT status FROM detections WHERE detection_id = ?",
            (seeded["a2"],),
        ).fetchone()[0]
    assert status == "rejected"
    assert current_facts(seeded["a1"])["bird_presence"]["answer_value"] == "present"
    assert "bird_presence" not in current_facts(seeded["a2"])


# --- 5. POST /api/moderation/bulk/relabel ----------------------------------


def test_bulk_relabel_sets_every_axis(seeded):
    response = post(
        seeded["client"],
        "/api/moderation/bulk/relabel",
        {"detection_ids": [seeded["a1"], seeded["b"]], "species": TARGET_SPECIES},
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    for detection_id in (seeded["a1"], seeded["b"]):
        assert_confirmed_species(detection_id, TARGET_SPECIES)
        fact = current_facts(detection_id)["species_identity"]
        assert fact["source_kind"] == "watchmybirds_bulk_moderation"
        assert fact["source_ref"] == "moderation:bulk-relabel"


# --- 7. POST /api/detections/relabel ---------------------------------------


def test_detections_relabel_sets_every_axis(seeded):
    response = post(
        seeded["client"],
        "/api/detections/relabel",
        {"detection_id": seeded["b"], "species": TARGET_SPECIES},
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert_confirmed_species(seeded["b"], TARGET_SPECIES)
    assert current_facts(seeded["b"])["species_identity"]["source_ref"] == (
        "detections:relabel"
    )


# --- 6. POST /api/moderation/rescan-proposals/<id>/apply -------------------


def test_rescan_proposal_apply_records_a_non_manual_provenance(seeded):
    """This site is deliberately NOT a human answer.

    Accepting a rescan proposal writes ``species_source='proposal_applied'``,
    not ``'manual'`` — the distinction says whether a person named the
    species or merely waved a model suggestion through. A migration that
    forces ``'manual'`` here would destroy that provenance.
    """
    with db_connection.closing_connection() as conn:
        conn.execute(
            """
            INSERT INTO rescan_proposals (
                proposal_id, job_id, target_detection_id, image_filename,
                suggested_species, status
            ) VALUES (1, 'job-test', ?, ?, ?, 'ready')
            """,
            (seeded["b"], seeded["frame_b"], TARGET_SPECIES),
        )
        conn.commit()

    response = post(seeded["client"], "/api/moderation/rescan-proposals/1/apply", {})

    assert response.status_code == 200, response.get_data(as_text=True)
    state = axes(seeded["b"])
    assert state["decision_state"] == "confirmed"
    assert state["decision_level"] == "species"
    assert state["manual_species_override"] == TARGET_SPECIES
    assert state["species_source"] == "proposal_applied"
    assert current_facts(seeded["b"]) == {}


def test_event_approve_does_not_overwrite_a_per_frame_relabel(seeded):
    """Per-frame relabel beats event-level species.

    If the operator relabelled one frame before approving the event, the
    species picked in the event rail must not overwrite that choice.
    """
    with db_connection.closing_connection() as conn:
        conn.execute(
            """
            UPDATE detections
            SET manual_species_override = ?, species_source = 'manual'
            WHERE detection_id = ?
            """,
            (PRIOR_SPECIES, seeded["a2"]),
        )
        conn.commit()

    response = post(
        seeded["client"],
        "/api/review/event-approve",
        {
            "detection_ids": [seeded["a1"], seeded["a2"]],
            "species": TARGET_SPECIES,
            "bbox_review": "correct",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    # a1 had no prior choice -> takes the event species.
    assert axes(seeded["a1"])["manual_species_override"] == TARGET_SPECIES
    # a2 was relabelled by hand -> keeps it, but still gets confirmed.
    assert axes(seeded["a2"])["manual_species_override"] == PRIOR_SPECIES
    assert axes(seeded["a2"])["decision_state"] == "confirmed"
    assert axes(seeded["a2"])["decision_level"] == "species"


def test_event_resolve_does_not_overwrite_a_per_frame_relabel(seeded):
    """Same rule on the mixed keep/trash path."""
    with db_connection.closing_connection() as conn:
        conn.execute(
            """
            UPDATE detections
            SET manual_species_override = ?, species_source = 'manual'
            WHERE detection_id = ?
            """,
            (PRIOR_SPECIES, seeded["a1"]),
        )
        conn.commit()

    response = post(
        seeded["client"],
        "/api/review/event-resolve",
        {
            "keep_detection_ids": [seeded["a1"]],
            "trash_detection_ids": [seeded["a2"]],
            "species": TARGET_SPECIES,
            "bbox_review": "correct",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert axes(seeded["a1"])["manual_species_override"] == PRIOR_SPECIES
    assert axes(seeded["a1"])["decision_state"] == "confirmed"
