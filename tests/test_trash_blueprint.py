"""Tests for trash blueprint relabel/rate/species-list APIs."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.trash import trash_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(trash_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


def test_species_list_returns_sorted_species(client):
    payload = {"Parus_major": "Great Tit"}
    with patch("builtins.open", mock_open(read_data="{}")):
        with patch("json.load", return_value=payload):
            response = client.get("/api/species-list")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["species"] == [{"scientific": "Parus_major", "common": "Great Tit"}]


def test_relabel_requires_detection_and_species(client):
    response = client.post("/api/detections/relabel", json={})
    assert response.status_code == 400
    assert "required" in response.get_json()["error"]


def test_relabel_updates_detection_and_classification(client):
    mock_conn = MagicMock()
    with patch(
        "web.blueprints.trash.db_service.get_connection", return_value=mock_conn
    ):
        response = client.post(
            "/api/detections/relabel",
            json={"detection_id": 7, "species": "False_Positive"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["new_species"] == "False_Positive"
    assert mock_conn.execute.call_count == 2
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


def test_rate_rejects_out_of_range_values(client):
    response = client.post(
        "/api/detections/rate",
        json={"detection_id": 7, "rating": 6},
    )
    assert response.status_code == 400
    assert "1-5" in response.get_json()["error"]


def test_rate_rejects_zero_rating(client):
    mock_conn = MagicMock()
    with patch(
        "web.blueprints.trash.db_service.get_connection", return_value=mock_conn
    ):
        response = client.post(
            "/api/detections/rate",
            json={"detection_id": 7, "rating": 0},
        )

    assert response.status_code == 400
    data = response.get_json()
    assert "1-5" in data["error"]
    mock_conn.commit.assert_not_called()
