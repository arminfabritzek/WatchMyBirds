from __future__ import annotations

import time

from flask import Flask

from tests.test_station_observations import answer, seed
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils.db import closing_connection


def test_ordinary_review_keeps_answered_unknown_reachable(local_db_app: Flask) -> None:
    with closing_connection() as conn:
        filename = seed(conn, identifier=101)
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


def test_report_builds_automatically_and_stale_claims_are_hidden(
    local_db_app: Flask,
) -> None:
    with closing_connection() as conn:
        filename = seed(conn)
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
            assert "data-current-report" not in stale and "Pica pica" not in stale
            assert "Earlier results are hidden" in stale
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
        filename = seed(conn)
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
            seed(conn, identifier=identifier)
        seed(conn, identifier=60)
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
        first = seed(conn, identifier=10, species="Pica_pica")
        seed(conn, identifier=11, species="Pica_pica")
        seed(conn, identifier=12, species="Pica_pica")
    with local_db_app.test_client() as client:
        assert client.post("/api/review/answers", json={}).status_code in {
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
            "/api/review/answers", json={"detection_ids": [10, 11], "action": "no_bird"}
        )
        assert result.status_code == 409
        with closing_connection() as conn:
            assert {
                row["detection_id"]
                for row in fetch_observation_page(conn, answered=False)[0]
            } == {11, 12}
        result = client.post(
            "/api/review/answers", json={"detection_ids": [11], "action": "unknown"}
        )
        assert result.status_code == 200
        result = client.post(
            "/api/review/answers",
            json={"detection_ids": [12], "action": "species", "species": "Pica_pica"},
        )
        assert result.status_code == 200
        with closing_connection() as conn:
            assert fetch_observation_page(conn, answered=False)[1] == 0
        assert "data-review-grid-card" not in client.get("/admin/review").get_data(
            as_text=True
        )
        assert (
            client.post(
                "/api/review/answers", json={"detection_ids": [12], "action": "no_bird"}
            ).status_code
            == 409
        )
        for body in [
            {},
            [],
            {"detection_ids": "12", "action": "no_bird"},
            {"detection_ids": [True], "action": "no_bird"},
            {"detection_ids": [999], "action": []},
        ]:
            assert client.post("/api/review/answers", json=body).status_code in {
                400,
                409,
            }
