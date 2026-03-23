from unittest.mock import MagicMock

import pytest
from flask import Flask

from web.blueprints import analytics as analytics_module
from web.blueprints.analytics import _sort_species_activity_by_peak_hour, analytics_bp


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)

    with app.test_client() as client:
        yield client


def test_sort_species_activity_by_peak_hour_orders_by_peak_then_species():
    items = [
        {"species": "Phoenicurus_ochruros", "peak_hour_formatted": "08:00"},
        {"species": "Dendrocopos_major", "peak_hour_formatted": "05:00"},
        {"species": "Passer_domesticus", "peak_hour_formatted": "08:00"},
        {"species": "Erithacus_rubecula", "peak_hour": 7.5},
    ]

    sorted_items = _sort_species_activity_by_peak_hour(items)

    assert [item["species"] for item in sorted_items] == [
        "Dendrocopos_major",
        "Erithacus_rubecula",
        "Passer_domesticus",
        "Phoenicurus_ochruros",
    ]


def test_species_activity_api_sorts_by_peak_hour(monkeypatch, client):
    mock_conn = MagicMock()
    rows = [
        {"species": "Phoenicurus_ochruros", "image_timestamp": "20260311_081500"},
        {"species": "Phoenicurus_ochruros", "image_timestamp": "20260311_082500"},
        {"species": "Phoenicurus_ochruros", "image_timestamp": "20260311_084500"},
        {"species": "Dendrocopos_major", "image_timestamp": "20260311_050500"},
        {"species": "Dendrocopos_major", "image_timestamp": "20260311_051000"},
        {"species": "Dendrocopos_major", "image_timestamp": "20260311_053500"},
        {"species": "Erithacus_rubecula", "image_timestamp": "20260311_071000"},
        {"species": "Erithacus_rubecula", "image_timestamp": "20260311_072000"},
        {"species": "Erithacus_rubecula", "image_timestamp": "20260311_075500"},
        {"species": "Passer_domesticus", "image_timestamp": "20260311_080500"},
        {"species": "Passer_domesticus", "image_timestamp": "20260311_081000"},
        {"species": "Passer_domesticus", "image_timestamp": "20260311_085500"},
    ]

    monkeypatch.setattr(
        analytics_module.db_service, "get_connection", lambda: mock_conn
    )
    monkeypatch.setattr(
        analytics_module.db_service, "fetch_species_timestamps", lambda conn, **kw: rows
    )

    response = client.get("/api/analytics/species-activity")

    assert response.status_code == 200
    assert [item["species"] for item in response.get_json()] == [
        "Dendrocopos_major",
        "Erithacus_rubecula",
        "Passer_domesticus",
        "Phoenicurus_ochruros",
    ]
    mock_conn.close.assert_called_once()
