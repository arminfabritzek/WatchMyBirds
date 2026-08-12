"""Ratified scope boundaries for human labels and export operations.

These tests describe the target meaning before production behavior changes.
They intentionally distinguish image-scoped answers, object-scoped answers,
unanswered facts, and export bookkeeping using a real SQLite database.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from tests.test_label_write_endpoints_contract import _reset_test_config, _seed, post
from utils.db import connection as db_connection
from web.web_interface import create_web_interface

LABEL_COLUMNS = (
    "decision_state",
    "decision_level",
    "species_source",
    "manual_species_override",
    "manual_bbox_review",
    "bbox_reviewed_at",
)


def _label_state(detection_id: int) -> dict[str, str | None]:
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            f"SELECT {', '.join(LABEL_COLUMNS)} "
            "FROM detections WHERE detection_id = ?",
            (detection_id,),
        ).fetchone()
    return dict(zip(LABEL_COLUMNS, row, strict=True))


def _review_status(filename: str) -> str | None:
    with db_connection.closing_connection() as conn:
        row = conn.execute(
            "SELECT review_status FROM images WHERE filename = ?", (filename,)
        ).fetchone()
    return row[0] if row else None


def _presence_facts(filename: str) -> list[tuple[str, int | None, str]]:
    with db_connection.closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT scope, detection_id, answer_value
            FROM current_human_label_facts
            WHERE image_filename = ? AND fact_type = 'bird_presence'
            ORDER BY scope, detection_id
            """,
            (filename,),
        ).fetchall()
    return [(row["scope"], row["detection_id"], row["answer_value"]) for row in rows]


@pytest.fixture
def semantic_client(monkeypatch, tmp_path):
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
            yield client


@pytest.fixture
def semantic_case(semantic_client):
    today = datetime.now().strftime("%Y%m%d")
    filename = f"{today}_110000_semantic.jpg"
    with db_connection.closing_connection() as conn:
        detection_id = _seed(
            conn,
            filename=filename,
            timestamp=f"{today}_110000",
        )
    return {
        "client": semantic_client,
        "filename": filename,
        "detection_id": detection_id,
    }


def test_rejecting_one_object_does_not_state_no_bird_for_image(semantic_case):
    response = post(
        semantic_case["client"],
        "/api/moderation/bulk/reject",
        {"detection_ids": [semantic_case["detection_id"]]},
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    with db_connection.closing_connection() as conn:
        detection_status = conn.execute(
            "SELECT status FROM detections WHERE detection_id = ?",
            (semantic_case["detection_id"],),
        ).fetchone()[0]
    assert detection_status == "rejected"
    assert _review_status(semantic_case["filename"]) == "untagged"
    assert _presence_facts(semantic_case["filename"]) == []


def test_whole_image_no_bird_requires_explicit_image_answer(semantic_case):
    response = post(
        semantic_case["client"],
        "/api/review/decision",
        {"filenames": [semantic_case["filename"]], "action": "no_bird"},
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert response.get_json()["action"] == "no_bird"
    assert _review_status(semantic_case["filename"]) == "no_bird"
    assert _presence_facts(semantic_case["filename"]) == [
        ("image", None, "absent")
    ]


def test_unanswered_object_bbox_and_species_facts_remain_unknown(semantic_case):
    response = post(
        semantic_case["client"],
        "/api/review/decision",
        {"filenames": [semantic_case["filename"]], "action": "confirm"},
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    assert _review_status(semantic_case["filename"]) == "confirmed_bird"
    assert _label_state(semantic_case["detection_id"]) == {
        column: None for column in LABEL_COLUMNS
    }
    assert _presence_facts(semantic_case["filename"]) == [
        ("image", None, "present")
    ]


def test_export_preview_does_not_create_or_change_label_facts(semantic_case):
    detection_id = semantic_case["detection_id"]
    before = _label_state(detection_id)
    response = semantic_case["client"].get("/api/groundtruth-export/preview")

    assert response.status_code == 200, response.get_data(as_text=True)
    assert _label_state(detection_id) == before


def test_groundtruth_export_page_remains_available(semantic_case):
    response = semantic_case["client"].get("/admin/groundtruth-export")

    assert response.status_code == 200, response.get_data(as_text=True)


def test_removed_training_export_endpoint_is_not_available(semantic_case):
    response = post(
        semantic_case["client"],
        "/api/training-export/add",
        {"detection_ids": [semantic_case["detection_id"]]},
    )

    assert response.status_code == 404
