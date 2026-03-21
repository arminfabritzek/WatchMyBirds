"""Tests for trash blueprint relabel/rate/species-list APIs."""

import json
from unittest.mock import MagicMock, patch

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


def test_species_list_uses_common_nb_for_extended_species(client, tmp_path, monkeypatch):
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
    with patch(
        "web.blueprints.trash.db_service.get_connection", return_value=mock_conn
    ), patch(
        "web.blueprints.trash.get_config",
        return_value={"SPECIES_COMMON_NAME_LOCALE": "NO"},
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

    with patch(
        "web.blueprints.trash.db_service.get_connection", return_value=mock_conn
    ), patch(
        "utils.species_names.load_common_names", return_value=names
    ), patch(
        "utils.species_names.load_extended_species", return_value=extended
    ), patch(
        "config.get_config",
        return_value={"SPECIES_COMMON_NAME_LOCALE": "DE"},
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


def test_relabel_requires_detection_and_species(client):
    response = client.post("/api/detections/relabel", json={})
    assert response.status_code == 400
    assert "required" in response.get_json()["error"]


def test_relabel_updates_detection_and_classification(client):
    mock_conn = MagicMock()
    with patch(
        "web.blueprints.trash.db_service.get_connection", return_value=mock_conn
    ), patch(
        "web.blueprints.trash.build_species_picker_entries",
        return_value=[
            {
                "scientific": "False_Positive",
                "common": "False Positive",
                "source": "extended",
                "score": None,
                "rank": None,
            }
        ],
    ):
        response = client.post(
            "/api/detections/relabel",
            json={"detection_id": 7, "species": "False_Positive"},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["new_species"] == "False_Positive"
    assert mock_conn.execute.call_count == 1
    assert "manual_species_override" in mock_conn.execute.call_args[0][0]
    assert "UPDATE classifications" not in mock_conn.execute.call_args[0][0]
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


# ── Unauthenticated access tests ──


@pytest.fixture
def unauth_client(app):
    """Client without session authentication."""
    with app.test_client() as client:
        yield client


def test_unauth_reject_redirects_to_login(unauth_client):
    response = unauth_client.post(
        "/api/detections/reject",
        json={"ids": [1]},
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_unauth_relabel_redirects_to_login(unauth_client):
    response = unauth_client.post(
        "/api/detections/relabel",
        json={"detection_id": 1, "species": "Parus_major"},
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
