from __future__ import annotations

import re
import time
from pathlib import Path

from flask import Flask

from tests.test_station_observations import answer, seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection


def test_ordinary_review_keeps_answered_unknown_reachable(local_db_app: Flask) -> None:
    with closing_connection() as conn:
        filename = seed(conn, identifier=101, state="uncertain")
        answer(conn, 101, filename, state="unknown", species=None)
        seed(conn, identifier=102, state="uncertain")
    with local_db_app.test_client() as client:
        assert client.get("/admin/review").status_code == 302
        with client.session_transaction() as session:
            session["authenticated"] = True
        open_body = client.get("/admin/review").get_data(as_text=True)
        assert 'data-detection-id="102"' in open_body
        assert 'data-detection-id="101"' not in open_body
        assert "review-grid__evidence" not in open_body
        answered = client.get("/admin/review?answered=1").get_data(as_text=True)
        assert 'data-detection-id="101"' in answered
        assert "Bird · species unknown · Answered" in answered
        assert "modal-station_review-101" in answered
        unknown = client.get("/species/overview?species_key=Unknown_species").get_data(
            as_text=True
        )
        assert 'data-detection-id="101"' in unknown
        assert client.get("/admin/review?answered=invalid").status_code == 400


def test_report_retains_snapshot_until_explicit_refresh(
    local_db_app: Flask,
) -> None:
    with closing_connection() as conn:
        filename = seed(conn, state="uncertain")
        answer(conn, 101, filename, species="Pica_pica")
    try:
        with local_db_app.test_client() as client:
            first = client.get("/analytics")
            assert first.status_code == 200
            assert "Building the station history automatically" in first.get_data(
                as_text=True
            )
            deadline = time.monotonic() + 5
            while (
                client.get("/api/station-report").get_json()["state"] != "ready"
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            body = client.get("/analytics").get_data(as_text=True)
            assert "Pica pica" in body and "User confirmed" in body
            assert "data-current-report" in body
            assert "No sourced local species profiles" in body
            with closing_connection() as conn:
                answer(conn, 101, filename, species="Picus_viridis")
            stale = client.get("/analytics").get_data(as_text=True)
            assert "data-current-report" in stale and "Pica pica" in stale
            assert "as of" in stale
            assert client.post("/api/station-report").status_code == 403
            with client.session_transaction() as session:
                session["_csrf_token"] = "report-test"
            assert (
                client.post(
                    "/api/station-report", headers={"X-CSRF-Token": "report-test"}
                ).status_code
                == 200
            )
            while client.get("/api/station-report").get_json()["state"] != "ready":
                assert time.monotonic() < deadline
                time.sleep(0.01)
            refreshed = client.get("/analytics").get_data(as_text=True)
            assert "Picus viridis" in refreshed and "Pica pica" not in refreshed
            assert (
                client.get("/api/station-report").headers["Cache-Control"] == "no-store"
            )
            assert client.get("/analytics?start=2026-02-30").status_code == 400
    finally:
        local_db_app.extensions["station_report"].close()


def test_raw_model_proposal_is_separate_and_scoped_to_its_image(
    local_db_app: Flask,
) -> None:
    with closing_connection() as conn:
        filename = seed(conn, state="uncertain")
        conn.execute(
            "UPDATE detections SET raw_species_name='Himantopus_leucocephalus'"
        )
        answer(conn, 101, filename, species="Pica_pica")
    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        result = client.get(
            "/api/labels/state",
            query_string={"filename": filename, "detection_id": 101},
        ).get_json()
        assert result["model_proposal"]["raw_species_key"] == "Himantopus_leucocephalus"
        assert (
            next(f for f in result["facts"] if f["fact_type"] == "species_identity")[
                "species_key"
            ]
            == "Pica_pica"
        )
        other = client.get(
            "/api/labels/state",
            query_string={"filename": "another.jpg", "detection_id": 101},
        ).get_json()
        assert other["model_proposal"] is None


def test_review_groups_whole_visits_and_selects_days(local_db_app: Flask) -> None:
    with closing_connection() as conn:
        for identifier in range(1, 56):
            seed(conn, identifier=identifier, state="uncertain")
        seed(conn, identifier=60, state="uncertain")
        conn.execute(
            "UPDATE images SET timestamp='20261012_082211' WHERE filename LIKE '%_60.jpg'"
        )
    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        body = client.get("/admin/review").get_data(as_text=True)
        assert "55 objects on this day" in body
        assert 'data-detection-id="55"' in body
        assert 'data-detection-id="60"' not in body
        assert "review-grid__card" in body
        older = client.get("/admin/review?day=2026-10-12").get_data(as_text=True)
        assert 'data-detection-id="60"' in older
        assert 'data-detection-id="55"' not in older
        assert client.get("/admin/review?day=bad").status_code == 400


def test_event_answers_are_atomic_and_never_overwrite_completed_objects(
    local_db_app: Flask,
) -> None:
    from utils.db.detections import fetch_observation_page

    with closing_connection() as conn:
        first = seed(conn, identifier=10, species="Pica_pica", state="uncertain")
        seed(conn, identifier=11, species="Pica_pica", state="uncertain")
        seed(conn, identifier=12, species="Pica_pica", state="uncertain")
    with local_db_app.test_client() as client:
        assert client.post("/api/labels/answer", json={}).status_code in {
            302,
            401,
            403,
        }
        with client.session_transaction() as session:
            session["authenticated"] = True
            session["_csrf_token"] = "review-test"
        client.environ_base["HTTP_X_CSRF_TOKEN"] = "review-test"
        with closing_connection() as conn:
            answer(conn, 10, first, species="Picus_viridis")
        result = client.post(
            "/api/labels/answer",
            json={
                "detection_ids": [10, 11],
                "object_bird_presence": "absent",
                "review_only": True,
            },
        )
        assert result.status_code == 409
        with closing_connection() as conn:
            assert {
                row["detection_id"]
                for row in fetch_observation_page(conn, answered=False)[0]
            } == {11, 12}
        result = client.post(
            "/api/labels/answer",
            json={
                "detection_ids": [11],
                "species_identity": "unknown",
                "review_only": True,
            },
        )
        assert result.status_code == 200
        result = client.post(
            "/api/labels/answer",
            json={
                "detection_ids": [12],
                "species_identity": "confirmed",
                "species_key": "Pica_pica",
                "review_only": True,
            },
        )
        assert result.status_code == 200
        with closing_connection() as conn:
            assert fetch_observation_page(conn, answered=False)[1] == 0
        assert "data-review-grid-card" not in client.get("/admin/review").get_data(
            as_text=True
        )
        assert (
            client.post(
                "/api/labels/answer",
                json={
                    "detection_ids": [12],
                    "object_bird_presence": "absent",
                    "review_only": True,
                },
            ).status_code
            == 409
        )
        for body in [
            {},
            [],
            {
                "detection_ids": "12",
                "object_bird_presence": "absent",
                "review_only": True,
            },
            {
                "detection_ids": [True],
                "object_bird_presence": "absent",
                "review_only": True,
            },
            {"detection_ids": [999], "action": []},
        ]:
            assert client.post("/api/labels/answer", json=body).status_code in {
                400,
                409,
            }


def test_done_keeps_absence_above_old_species_and_lists_image_answers(
    local_db_app: Flask, tmp_path: Path
) -> None:
    from core.human_label_core import (
        BBox,
        HumanAnswer,
        append_fact,
        ensure_object_subject,
        record_human_answer,
    )
    from core.manual_object_core import ManualObjectDraft, create_manual_object
    from tests.test_station_observations import provenance

    with closing_connection() as conn:
        absent = seed(conn, identifier=1, species="Parus_major")
        answer(conn, 1, absent, species="Pica_pica")
        append_fact(
            conn,
            subject_id=ensure_object_subject(conn, 1),
            fact_type="bird_presence",
            answer_value="absent",
            provenance=provenance(),
        )
        wrong = seed(conn, identifier=2)
        answer(conn, 2, wrong, state="wrong", species=None)
        manual = seed(conn, identifier=3, state="uncertain")
        original = tmp_path / "manual.jpg"
        original.write_bytes(b"synthetic image")
        create_manual_object(
            conn,
            ManualObjectDraft(
                manual, BBox(0.5, 0.5, 0.2, 0.2), "Pica_pica", "manual-review-test"
            ),
            provenance(),
            original_path=original,
            locale="DE",
        )
        conn.execute(
            "INSERT INTO images(filename,timestamp,original_present) VALUES ('empty.jpg','20261012_100000',0)"
        )
        record_human_answer(
            conn,
            HumanAnswer(image_filename="empty.jpg", image_bird_presence="absent"),
            provenance(),
        )
        conn.execute(
            "INSERT INTO images(filename,timestamp,original_present) VALUES ('missed.jpg','20261013_120000',0)"
        )
        record_human_answer(
            conn,
            HumanAnswer(image_filename="missed.jpg", detector_miss=True),
            provenance(),
        )
        conn.execute("UPDATE images SET original_present=0")
    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        response = client.get(
            "/admin/review?answered=1&day=all&start=2026-10-12&end=2026-10-13"
        )
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        tile = re.search(
            r'<article[^>]*data-detection-id="1".*?</article>', body, re.S
        ).group()
        assert '<span class="wm-tile__name">No bird</span>' in tile
        assert 'class="wm-tile__latin"' not in tile
        assert 'class="station-media-placeholder"' in tile
        assert '<span class="wm-tile__name">Wrong species</span>' in body
        for label in (
            "Manually added bird",
            "No birds in full image",
            "Bird missed by the detector",
        ):
            assert label in body
        assert 'name="answered" value="1"' in body
        assert 'aria-current="page">Done</a>' in body
        older = client.get("/admin/review?answered=1&day=2026-10-12").get_data(
            as_text=True
        )
        assert (
            "No birds in full image" in older
            and "Bird missed by the detector" not in older
        )
        empty = client.get("/admin/review?answered=1&start=2026-10-14").get_data(
            as_text=True
        )
        assert 'class="wm-tile__name"' not in empty


def test_done_keeps_withdrawn_manual_object_visible(
    local_db_app: Flask, tmp_path: Path
) -> None:
    from core.human_label_core import BBox
    from core.manual_object_core import (
        ManualObjectDraft,
        create_manual_object,
        retract_manual_object,
    )
    from tests.test_station_observations import provenance

    with closing_connection() as conn:
        filename = seed(conn, identifier=1, state="uncertain")
        original = tmp_path / "original.jpg"
        original.write_bytes(b"synthetic image")
        obj, _ = create_manual_object(
            conn,
            ManualObjectDraft(
                filename, BBox(0.5, 0.5, 0.2, 0.2), "Pica_pica", "manual-withdraw-test"
            ),
            provenance(),
            original_path=original,
            locale="DE",
        )
        retract_manual_object(
            conn,
            manual_object_id=obj["manual_object_id"],
            image_filename=filename,
            expected_revision=obj["revision"],
            provenance=provenance(),
        )
    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
        response = client.get("/admin/review?answered=1")
        assert response.status_code == 200
        assert "Manually added bird withdrawn" in response.get_data(as_text=True)
