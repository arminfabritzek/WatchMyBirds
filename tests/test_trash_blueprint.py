"""Tests for trash blueprint relabel/favorite/species-list APIs."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient


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
    names = {"Parus_major": "Great Tit"}
    with patch("utils.species_names.load_common_names", return_value=names):
        with patch(
            "config.get_config",
            return_value={"SPECIES_COMMON_NAME_LOCALE": "DE"},
        ):
            response = client.get("/api/species-list")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    # Unknown_species is always pinned at position 0
    assert data["species"][0]["scientific"] == "Unknown_species"
    assert data["species"][0]["common"] == "Unknown species"
    assert data["species"][0]["source"] == "model"
    assert any(
        sp["scientific"] == "Parus_major"
        and sp["common"] == "Great Tit"
        and sp["source"] == "model"
        for sp in data["species"]
    )


def test_species_list_deduplicates_unknown_species(client):
    """If common_names already contains Unknown_species, it must appear exactly once."""
    names = {"Unknown_species": "Unknown species", "Parus_major": "Great Tit"}
    with patch("utils.species_names.load_common_names", return_value=names):
        with patch(
            "config.get_config",
            return_value={"SPECIES_COMMON_NAME_LOCALE": "DE"},
        ):
            response = client.get("/api/species-list")

    data = response.get_json()
    unknown_entries = [
        s for s in data["species"] if s["scientific"] == "Unknown_species"
    ]
    assert len(unknown_entries) == 1
    assert data["species"][0]["scientific"] == "Unknown_species"


def test_species_list_uses_locale_no(client):
    """species-list with NO locale uses Norwegian common names."""
    names = {"Parus_major": "Kjøttmeis", "Unknown_species": "Unknown species"}
    with patch("utils.species_names.load_common_names", return_value=names):
        with patch(
            "config.get_config",
            return_value={"SPECIES_COMMON_NAME_LOCALE": "NO"},
        ):
            response = client.get("/api/species-list")

    data = response.get_json()
    assert data["species"][0]["scientific"] == "Unknown_species"
    assert any(
        sp["scientific"] == "Parus_major"
        and sp["common"] == "Kjøttmeis"
        and sp["source"] == "model"
        for sp in data["species"]
    )


def test_species_list_uses_common_nb_for_extended_species(
    client, tmp_path, monkeypatch
):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()

    (assets_dir / "common_names_DE.json").write_text(
        json.dumps({"Unknown_species": "Unknown species"}),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps(
            [
                {
                    "scientific": "Picus_canus",
                    "common_de": "Grauspecht",
                    "common_en": "Grey-headed Woodpecker",
                    "common_nb": "Gråspett",
                },
                {
                    "scientific": "Corvus_corax",
                    "common_de": "Kolkrabe",
                    "common_en": "Common Raven",
                    "common_nb": "",
                },
            ]
        ),
        encoding="utf-8",
    )

    from utils import species_names

    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    with (
        patch("web.blueprints.trash.db_service.get_connection", return_value=mock_conn),
        patch(
            "web.blueprints.trash.get_config",
            return_value={"SPECIES_COMMON_NAME_LOCALE": "NO"},
        ),
    ):
        response = client.get("/api/species-list")

    assert response.status_code == 200
    data = response.get_json()
    species_by_key = {row["scientific"]: row for row in data["species"]}
    assert species_by_key["Picus_canus"]["common"] == "Gråspett"
    assert species_by_key["Corvus_corax"]["common"] == "Common Raven"
    assert species_by_key["Picus_canus"]["source"] == "extended"


def test_species_list_with_detection_id_includes_predictions_first(client):
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {"cls_class_name": "Parus_major", "cls_confidence": 0.85, "rank": 1},
        {
            "cls_class_name": "Cyanistes_caeruleus",
            "cls_confidence": 0.07,
            "rank": 2,
        },
    ]
    names = {
        "Unknown_species": "Unknown species",
        "Parus_major": "Great Tit",
        "Cyanistes_caeruleus": "Blue Tit",
    }
    extended = [{"scientific": "Picus_canus", "common": "Grey-headed Woodpecker"}]

    with (
        patch("web.blueprints.trash.db_service.get_connection", return_value=mock_conn),
        patch("utils.species_names.load_common_names", return_value=names),
        patch("utils.species_names.load_extended_species", return_value=extended),
        patch(
            "config.get_config",
            return_value={"SPECIES_COMMON_NAME_LOCALE": "DE"},
        ),
    ):
        response = client.get("/api/species-list?detection_id=42")

    assert response.status_code == 200
    data = response.get_json()
    assert data["species"][0]["scientific"] == "Parus_major"
    assert data["species"][0]["source"] == "prediction"
    assert data["species"][0]["score"] == 0.85
    assert data["species"][1]["scientific"] == "Cyanistes_caeruleus"
    assert data["species"][1]["source"] == "prediction"
    assert any(
        sp["scientific"] == "Picus_canus" and sp["source"] == "extended"
        for sp in data["species"]
    )


def test_removed_rating_endpoint_returns_not_found(client: FlaskClient) -> None:
    response = client.post(
        "/api/detections/rate", json={"detection_id": 7, "rating": 5}
    )
    assert response.status_code == 404


def test_favorite_stamps_manual_source_and_action_time(client):
    mock_conn = MagicMock()
    select_result = MagicMock()
    select_result.fetchone.return_value = {"is_favorite": 0}
    mock_conn.execute.return_value = select_result

    with patch(
        "web.blueprints.trash.db_service.get_connection", return_value=mock_conn
    ):
        response = client.post(
            "/api/detections/favorite",
            json={"detection_id": 7},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["is_favorite"] is True

    assert mock_conn.execute.call_args_list[0].args == ("BEGIN IMMEDIATE",)
    update_sql, update_params = mock_conn.execute.call_args_list[2].args
    assert "rating_source = 'manual'" in update_sql
    assert "species_updated_at" in update_sql
    assert update_params[0] == 1
    assert update_params[2] == 7
    action_at = datetime.fromisoformat(update_params[1])
    assert action_at.tzinfo is not None
    mock_conn.commit.assert_called_once()


# ── Unauthenticated access tests ──


@pytest.fixture
def unauth_client(app):
    """Client without session authentication."""
    with app.test_client() as client:
        yield client


def test_unauth_reject_redirects_to_login(unauth_client):
    response = unauth_client.post(
        "/api/trash/move",
        json={"detection_ids": [1]},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_unauth_favorite_redirects_to_login(unauth_client):
    response = unauth_client.post(
        "/api/detections/favorite",
        json={"detection_id": 1},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
