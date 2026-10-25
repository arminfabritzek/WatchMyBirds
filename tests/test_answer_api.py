"""Atomic answers, withdrawal, and storage actions through their own endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from flask.testing import FlaskClient

from tests.labeling_helpers import _seed, post
from tests.test_human_labels_blueprint import labeling_case as labeling_case
from utils.db.connection import closing_connection

LabelingCase = tuple[FlaskClient, str, int]


def heads(conn: sqlite3.Connection, detection_id: int) -> dict[str, str]:
    return {
        row["fact_type"]: row["answer_value"]
        for row in conn.execute(
            "SELECT fact_type,answer_value FROM current_human_label_facts WHERE detection_id=?",
            (detection_id,),
        )
    }


@pytest.mark.parametrize(
    "answer,axis",
    [
        (
            {"species_identity": "corrected", "species_key": "Pica_pica"},
            "species_identity",
        ),
        (
            {"species_identity": "confirmed", "species_key": "Parus_major"},
            "species_identity",
        ),
        ({"species_identity": "unknown"}, "species_identity"),
        ({"species_identity": "wrong"}, "species_identity"),
        ({"object_bird_presence": "present"}, "bird_presence"),
        ({"object_bird_presence": "absent"}, "bird_presence"),
        ({"bbox_quality": "suitable"}, "bbox_quality"),
        ({"bbox_quality": "unsuitable"}, "bbox_quality"),
        (
            {"bbox_correction": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
            "bbox_correction",
        ),
    ],
)
def test_each_object_answer_and_withdrawal_leaves_automation_and_legacy_values_unchanged(
    labeling_case: LabelingCase, answer: dict, axis: str
) -> None:
    client, filename, did = labeling_case
    query = "SELECT decision_state, decision_level, status, manual_species_override, species_source, manual_bbox_review FROM detections WHERE detection_id=?"
    with closing_connection() as conn:
        before = tuple(conn.execute(query, (did,)).fetchone())
    response = post(client, "/api/labels/answer", {"detection_ids": [did], **answer})
    assert response.status_code == 200, response.json
    with closing_connection() as conn:
        assert axis in heads(conn, did)
        assert tuple(conn.execute(query, (did,)).fetchone()) == before
    response = post(
        client,
        "/api/labels/answer",
        {"filename": filename, "detection_id": did, "retract": axis},
    )
    assert response.status_code == 200, response.json
    with closing_connection() as conn:
        assert heads(conn, did) == {}
        assert tuple(conn.execute(query, (did,)).fetchone()) == before


@pytest.mark.parametrize(
    "answer,axis",
    [
        ({"detector_miss": True}, "detector_miss"),
        ({"image_bird_presence": "present"}, "bird_presence"),
    ],
)
def test_image_training_answers_are_withdrawable(
    labeling_case: LabelingCase, answer: dict, axis: str
) -> None:
    client, filename, _ = labeling_case
    assert (
        post(client, "/api/labels/answer", {"filename": filename, **answer}).status_code
        == 200
    )
    assert (
        post(
            client, "/api/labels/answer", {"filename": filename, "retract": axis}
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM current_human_label_facts WHERE image_filename=?",
                (filename,),
            ).fetchone()[0]
            == 0
        )


def test_batch_answer_is_one_click_and_rolls_back_unknown_targets(
    labeling_case: LabelingCase,
) -> None:
    client, filename, did = labeling_case
    with closing_connection() as conn:
        second = _seed(conn, filename=filename, timestamp=filename[:15])
    answer = {
        "detection_ids": [did, second],
        "species_identity": "corrected",
        "species_key": "Pica_pica",
        "bbox_quality": "suitable",
    }
    assert (
        post(
            client, "/api/labels/answer", {**answer, "detection_ids": [did, 987654321]}
        ).status_code
        == 409
    )
    with closing_connection() as conn:
        assert heads(conn, did) == {}
    assert post(client, "/api/labels/answer", answer).status_code == 200
    assert (
        post(
            client,
            "/api/labels/answer",
            {"detection_id": did, "retract": "species_identity"},
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        assert heads(conn, did) == heads(conn, second) == {}


def test_withdrawing_a_later_box_answer_keeps_earlier_species(
    labeling_case: LabelingCase,
) -> None:
    client, _, did = labeling_case
    assert (
        post(
            client,
            "/api/labels/answer",
            {
                "detection_id": did,
                "species_identity": "corrected",
                "species_key": "Pica_pica",
            },
        ).status_code
        == 200
    )
    assert (
        post(
            client,
            "/api/labels/answer",
            {"detection_id": did, "bbox_quality": "suitable"},
        ).status_code
        == 200
    )
    assert (
        post(
            client,
            "/api/labels/answer",
            {"detection_id": did, "retract": "bbox_quality"},
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        assert heads(conn, did) == {
            "bird_presence": "present",
            "species_identity": "corrected",
        }


def test_full_image_absence_is_atomic_and_reversible(
    labeling_case: LabelingCase,
) -> None:
    client, filename, did = labeling_case
    with closing_connection() as conn:
        second = _seed(conn, filename=filename, timestamp=filename[:15])
    assert (
        post(
            client,
            "/api/labels/answer",
            {"filename": filename, "image_bird_presence": "absent"},
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        assert heads(conn, did) == heads(conn, second) == {"bird_presence": "absent"}
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM detections WHERE status='rejected'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM classifications WHERE status='rejected'"
            ).fetchone()[0]
            == 0
        )
    assert (
        post(
            client,
            "/api/labels/answer",
            {"filename": filename, "retract": "bird_presence"},
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        assert heads(conn, did) == heads(conn, second) == {}


def test_full_image_and_manual_object_exclude_each_other(
    labeling_case: LabelingCase, tmp_path: Path
) -> None:
    from core.human_label_core import BBox, HumanLabelError, LabelProvenance
    from core.manual_object_core import ManualObjectDraft, create_manual_object

    client, filename, _ = labeling_case
    original = tmp_path / "original.jpg"
    original.write_bytes(b"synthetic original")
    draft = ManualObjectDraft(
        filename, BBox(0.6, 0.6, 0.2, 0.2), "Parus_major", "synthetic-request"
    )
    prov = LabelProvenance(
        "synthetic", "test", "normal_correction", "watchmybirds_ui", "test"
    )
    assert (
        post(
            client,
            "/api/labels/answer",
            {"filename": filename, "image_bird_presence": "absent"},
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        with pytest.raises(HumanLabelError, match="Withdraw No birds"):
            create_manual_object(conn, draft, prov, original_path=original, locale="DE")
    assert (
        post(
            client,
            "/api/labels/answer",
            {"filename": filename, "retract": "bird_presence"},
        ).status_code
        == 200
    )
    with closing_connection() as conn:
        create_manual_object(conn, draft, prov, original_path=original, locale="DE")
    state = client.get("/api/labels/state", query_string={"filename": filename})
    assert (
        state.json["full_image_absence_blocker"]
        == "Remove the 1 bird you added by hand first"
    )
    response = post(
        client,
        "/api/labels/answer",
        {"filename": filename, "image_bird_presence": "absent"},
    )
    assert response.status_code == 409
    assert response.json["message"] == state.json["full_image_absence_blocker"]
    with closing_connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM current_human_label_facts").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM manual_objects WHERE status='active'"
            ).fetchone()[0]
            == 1
        )


def test_trash_is_separate_from_answers_and_restore_clears_date(
    labeling_case: LabelingCase,
) -> None:
    client, _, did = labeling_case
    assert post(client, "/api/trash/move", {"detection_ids": [did]}).status_code == 200
    with closing_connection() as conn:
        row = conn.execute(
            "SELECT status,trashed_at FROM detections WHERE detection_id=?", (did,)
        ).fetchone()
        assert row[0] == "rejected" and row[1]
        assert heads(conn, did) == {}
    assert (
        post(client, "/api/trash/restore", {"detection_ids": [did]}).status_code == 200
    )
    with closing_connection() as conn:
        assert tuple(
            conn.execute(
                "SELECT status,trashed_at FROM detections WHERE detection_id=?", (did,)
            ).fetchone()
        ) == ("active", None)


@pytest.mark.parametrize(
    "route",
    [
        "/api/review/answers",
        "/api/review/decision",
        "/api/labels/species/retract",
        "/api/labels/bbox-quality/retract",
        "/api/detections/1/restore",
        "/api/moderation/bulk/relabel",
        "/api/moderation/bulk/reject",
        "/api/detections/relabel",
        "/api/detections/reject",
    ],
)
def test_replaced_writers_are_not_reachable(
    labeling_case: LabelingCase, route: str
) -> None:
    client, _, _ = labeling_case
    assert post(client, route, {}).status_code == 404


@pytest.mark.parametrize(
    "field",
    [
        "species_identity",
        "object_bird_presence",
        "image_bird_presence",
        "bbox_quality",
        "retract",
    ],
)
def test_invalid_axis_types_do_not_write(
    labeling_case: LabelingCase, field: str
) -> None:
    client, filename, did = labeling_case
    with closing_connection() as conn:
        before = list(conn.iterdump())
    response = post(
        client,
        "/api/labels/answer",
        {"filename": filename, "detection_id": did, field: ["invalid"]},
    )
    assert response.status_code == 409
    with closing_connection() as conn:
        assert list(conn.iterdump()) == before


def test_withdrawal_refreshes_boxes_from_the_same_click(
    labeling_case: LabelingCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, filename, did = labeling_case
    refreshed = []
    monkeypatch.setattr(
        "web.services.human_label_service.refresh_answer_thumbnails",
        lambda conn, targets, paths: refreshed.extend(targets),
    )
    assert (
        post(
            client,
            "/api/labels/answer",
            dict(
                filename=filename,
                detection_id=did,
                species_identity="confirmed",
                species_key="Parus_major",
                bbox_correction=dict(x=0.1, y=0.1, w=0.2, h=0.2),
            ),
        ).status_code
        == 200
    )
    refreshed.clear()
    assert (
        post(
            client,
            "/api/labels/answer",
            dict(detection_ids=[did], retract="species_identity"),
        ).status_code
        == 200
    )
    assert refreshed == [(filename, did)]
    with closing_connection() as conn:
        assert heads(conn, did) == {}
