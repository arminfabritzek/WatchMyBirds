"""Tests for the moderation blueprint and filter service.

Covers:
- resolve-selection (explicit + all_filtered)
- bulk/relabel
- bulk/reject
- Auth-expiry on all endpoints
- FilterContext validation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    from web.blueprints.auth import auth_bp
    from web.blueprints.moderation import moderation_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(moderation_bp)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as c:
        with c.session_transaction() as sess:
            sess["authenticated"] = True
        yield c


@pytest.fixture
def unauth_client(app):
    """Client without session authentication."""
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# resolve-selection
# ---------------------------------------------------------------------------


class TestResolveSelection:
    def test_explicit_mode_returns_given_ids(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "explicit", "ids": [1, 2, 3]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["detection_ids"] == [1, 2, 3]
        assert data["total_count"] == 3

    def test_explicit_mode_with_filenames(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "explicit", "filenames": ["a.jpg", "b.jpg"]},
        )
        data = resp.get_json()
        assert data["total_count"] == 2
        assert data["image_filenames"] == ["a.jpg", "b.jpg"]

    def test_all_filtered_requires_filter_context(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "all_filtered"},
        )
        assert resp.status_code == 400
        assert "filter_context" in resp.get_json()["message"]

    @patch("web.blueprints.moderation.resolve_filtered_ids")
    def test_all_filtered_delegates_to_resolver(self, mock_resolve, client):
        from web.services.filter_service import ResolvedSelection

        mock_resolve.return_value = ResolvedSelection(
            detection_ids=[10, 20, 30], total_count=3
        )
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={
                "mode": "all_filtered",
                "filter_context": {"surface": "gallery", "date": "2026-03-06"},
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["detection_ids"] == [10, 20, 30]
        assert data["total_count"] == 3
        mock_resolve.assert_called_once()

    @patch("web.blueprints.moderation.db_service")
    def test_date_range_resolves_active_detection_ids(self, mock_db, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_trash_candidate_selection_in_date_range.return_value = {
            "detection_ids": [4, 5, 6],
            "image_filenames": ["a.jpg", "b.jpg"],
            "image_count": 2,
        }

        resp = client.post(
            "/api/moderation/resolve-selection",
            json={
                "mode": "date_range",
                "from_date": "2026-03-01",
                "to_date": "2026-03-03",
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["detection_ids"] == [4, 5, 6]
        assert data["image_filenames"] == ["a.jpg", "b.jpg"]
        assert data["total_count"] == 3
        assert data["image_count"] == 2
        mock_db.fetch_trash_candidate_selection_in_date_range.assert_called_once_with(
            mock_conn, "2026-03-01", "2026-03-03"
        )

    @patch("web.blueprints.moderation.db_service")
    def test_date_range_returns_zero_matches(self, mock_db, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_trash_candidate_selection_in_date_range.return_value = {
            "detection_ids": [],
            "image_filenames": [],
            "image_count": 0,
        }

        resp = client.post(
            "/api/moderation/resolve-selection",
            json={
                "mode": "date_range",
                "from_date": "2026-03-01",
                "to_date": "2026-03-01",
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["detection_ids"] == []
        assert data["image_filenames"] == []
        assert data["total_count"] == 0
        assert data["image_count"] == 0

    @patch("web.blueprints.moderation.db_service")
    def test_date_range_counts_orphan_images_in_total(self, mock_db, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_trash_candidate_selection_in_date_range.return_value = {
            "detection_ids": [],
            "image_filenames": ["orphan-a.jpg", "orphan-b.jpg"],
            "image_count": 2,
            "orphan_count": 2,
        }

        resp = client.post(
            "/api/moderation/resolve-selection",
            json={
                "mode": "date_range",
                "from_date": "2026-03-01",
                "to_date": "2026-03-03",
            },
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["detection_ids"] == []
        assert data["image_filenames"] == ["orphan-a.jpg", "orphan-b.jpg"]
        assert data["orphan_count"] == 2
        assert data["image_count"] == 2
        assert data["total_count"] == 2

    def test_date_range_requires_both_dates(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "date_range", "from_date": "2026-03-01"},
        )

        # Loosened after the response stopped leaking the exception
        # text — the API contract is "400 + validation message that
        # mentions one of the date fields", not the literal verbatim
        # "to_date required" string.
        assert resp.status_code == 400
        message = resp.get_json()["message"].lower()
        assert "to_date" in message or "from_date" in message or "invalid" in message

    def test_date_range_rejects_invalid_ordering(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={
                "mode": "date_range",
                "from_date": "2026-03-05",
                "to_date": "2026-03-01",
            },
        )

        assert resp.status_code == 400
        assert "from_date must be on or before to_date" in resp.get_json()["message"]

    def test_logical_filter_is_no_longer_supported(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "logical_filter"},
        )

        assert resp.status_code == 400
        assert "Unknown mode" in resp.get_json()["message"]

    def test_unknown_mode_returns_400(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "foobar"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# bulk/relabel
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# bulk/reject
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Auth-expiry: all endpoints require login
# ---------------------------------------------------------------------------


class TestAuthExpiry:
    ENDPOINTS = [
        ("/api/moderation/resolve-selection", "POST"),
    ]

    @pytest.mark.parametrize("url,method", ENDPOINTS)
    def test_unauth_redirects_to_login(self, unauth_client, url, method):
        if method == "POST":
            resp = unauth_client.post(url, json={}, follow_redirects=False)
        else:
            resp = unauth_client.get(url, follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


# ---------------------------------------------------------------------------
# FilterContext unit tests
# ---------------------------------------------------------------------------


class TestFilterContext:
    def test_from_dict_defaults(self):
        from web.services.filter_service import FilterContext

        ctx = FilterContext.from_dict({"surface": "gallery"})
        assert ctx.surface == "gallery"
        assert ctx.min_score == 0.0
        assert ctx.min_score_explicit is False
        assert ctx.date is None

    def test_from_dict_full(self):
        from web.services.filter_service import FilterContext

        ctx = FilterContext.from_dict(
            {
                "surface": "species_overview",
                "species_key": "Parus_major",
                "min_score": 0.5,
            }
        )
        assert ctx.surface == "species_overview"
        assert ctx.species_key == "Parus_major"
        assert ctx.min_score == 0.5
        assert ctx.min_score_explicit is True

    def test_from_dict_normalizes_legacy_unknown_aliases(self):
        from web.services.filter_service import FilterContext

        ctx = FilterContext.from_dict(
            {
                "surface": "edit",
                "species_key": "Unknown",
            }
        )
        assert ctx.species_key == "Unknown_species"

    def test_frozen_immutable(self):
        from web.services.filter_service import FilterContext

        ctx = FilterContext(surface="gallery")
        with pytest.raises((AttributeError, TypeError)):
            ctx.surface = "edit"


class TestFilterResolverThresholds:
    def test_species_overview_uses_history_independent_of_photo_threshold(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from config import get_config
        from tests.test_station_observations import answer, seed
        from utils.db import closing_connection
        from utils.db.detections import fetch_detections_for_gallery
        from web.services.filter_service import FilterContext, resolve_filtered_ids

        monkeypatch.setitem(get_config(), "OUTPUT_DIR", str(tmp_path))
        with closing_connection() as conn:
            filename = seed(conn, identifier=11, species="Parus_major", quality=0)
            answer(conn, 11, filename, species="Parus_major")
            assert [r["detection_id"] for r in fetch_detections_for_gallery(conn)] == [
                11
            ]
        ctx = FilterContext.from_dict(
            {
                "surface": "species_overview",
                "species_key": "Parus_major",
                "min_score": 1.0,
                "favorites_only": True,
            }
        )
        result = resolve_filtered_ids(ctx)
        assert result.detection_ids == [11]
        assert result.total_count == 1

    @patch("web.services.filter_service.db_service")
    @patch("web.services.filter_service.get_config")
    def test_gallery_has_no_second_default_score_filter(self, mock_get_config, mock_db):
        from web.services.filter_service import FilterContext, resolve_filtered_ids

        mock_get_config.return_value = {"GALLERY_DISPLAY_THRESHOLD": 0.5}
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_detections_for_gallery.return_value = [
            {
                "detection_id": 21,
                "score": 0.2,
                "image_timestamp": "20260306_120000",
                "bbox_x": 0.10,
                "bbox_y": 0.10,
                "bbox_w": 0.20,
                "bbox_h": 0.20,
            },
            {
                "detection_id": 22,
                "score": 0.8,
                "image_timestamp": "20260306_130000",
                "bbox_x": 0.70,
                "bbox_y": 0.70,
                "bbox_w": 0.20,
                "bbox_h": 0.20,
            },
        ]

        ctx = FilterContext.from_dict(
            {
                "surface": "gallery",
                "date": "2026-03-06",
            }
        )

        result = resolve_filtered_ids(ctx)

        assert result.detection_ids == [22, 21]
        assert result.total_count == 2

    @patch("web.services.filter_service.db_service")
    @patch("web.services.filter_service.get_config")
    def test_gallery_returns_all_detection_ids_for_visible_observations(
        self, mock_get_config, mock_db
    ):
        from web.services.filter_service import FilterContext, resolve_filtered_ids

        mock_get_config.return_value = {"GALLERY_DISPLAY_THRESHOLD": 0.5}
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_detections_for_gallery.return_value = [
            {
                "detection_id": 41,
                "cls_class_name": "Parus_major",
                "od_class_name": "Parus_major",
                "score": 0.9,
                "image_timestamp": "20260306_120000",
                "bbox_x": 0.10,
                "bbox_y": 0.10,
                "bbox_w": 0.20,
                "bbox_h": 0.20,
            },
            {
                "detection_id": 42,
                "cls_class_name": "Parus_major",
                "od_class_name": "Parus_major",
                "score": 0.1,
                "image_timestamp": "20260306_120005",
                "bbox_x": 0.11,
                "bbox_y": 0.11,
                "bbox_w": 0.20,
                "bbox_h": 0.20,
            },
            {
                "detection_id": 43,
                "cls_class_name": "Parus_major",
                "od_class_name": "Parus_major",
                "score": 0.4,
                "image_timestamp": "20260306_130000",
                "bbox_x": 0.60,
                "bbox_y": 0.60,
                "bbox_w": 0.20,
                "bbox_h": 0.20,
            },
        ]

        ctx = FilterContext.from_dict(
            {
                "surface": "gallery",
                "date": "2026-03-06",
                "min_score": 0.5,
            }
        )

        result = resolve_filtered_ids(ctx)

        assert result.detection_ids == [41, 42]
        assert result.total_count == 2

    @patch("web.services.filter_service.db_service")
    def test_edit_unknown_species_filter_matches_unknown_species_key(self, mock_db):
        from web.services.filter_service import FilterContext, resolve_filtered_ids

        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_detections_for_gallery.return_value = [
            {
                "detection_id": 31,
                "cls_class_name": None,
                "od_class_name": None,
                "downloaded_timestamp": None,
                "od_confidence": 0.0,
                "cls_confidence": 0.0,
                "score": 0.0,
            }
        ]

        ctx = FilterContext.from_dict(
            {
                "surface": "edit",
                "date": "2026-03-06",
                "species_key": "Unknown_species",
            }
        )

        result = resolve_filtered_ids(ctx)

        assert result.detection_ids == [31]
        assert result.total_count == 1


@pytest.mark.parametrize(
    "url,method",
    [
        ("/api/moderation/bulk/rescan", "POST"),
        ("/api/moderation/rescan-jobs/abc123/status", "GET"),
        ("/api/moderation/rescan-proposals/1/apply", "POST"),
    ],
)
def test_removed_rescan_routes_return_404(
    client: FlaskClient, url: str, method: str
) -> None:
    assert client.open(url, method=method).status_code == 404
