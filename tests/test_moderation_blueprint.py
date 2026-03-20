"""Tests for the moderation blueprint and filter service.

Covers:
- resolve-selection (explicit + all_filtered)
- bulk/relabel
- bulk/reject
- Auth-expiry on all endpoints
- FilterContext validation
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

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
        mock_db.fetch_active_detection_ids_in_date_range.return_value = [4, 5, 6]

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
        assert data["image_filenames"] == []
        assert data["total_count"] == 3
        mock_db.fetch_active_detection_ids_in_date_range.assert_called_once_with(
            mock_conn, "2026-03-01", "2026-03-03"
        )

    @patch("web.blueprints.moderation.db_service")
    def test_date_range_returns_zero_matches(self, mock_db, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_active_detection_ids_in_date_range.return_value = []

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
        assert data["total_count"] == 0

    def test_date_range_requires_both_dates(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "date_range", "from_date": "2026-03-01"},
        )

        assert resp.status_code == 400
        assert "to_date required" in resp.get_json()["message"]

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

    @patch("web.blueprints.moderation.db_service")
    def test_logical_filter_resolves_user_import_selection(self, mock_db, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_active_detection_selection_by_source_type.return_value = {
            "detection_ids": [9, 10],
            "image_count": 1,
        }

        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "logical_filter", "source_type": "folder_upload"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["detection_ids"] == [9, 10]
        assert data["image_filenames"] == []
        assert data["total_count"] == 2
        assert data["image_count"] == 1
        mock_db.fetch_active_detection_selection_by_source_type.assert_called_once_with(
            mock_conn, "folder_upload"
        )

    @patch("web.blueprints.moderation.db_service")
    def test_logical_filter_returns_zero_matches(self, mock_db, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetch_active_detection_selection_by_source_type.return_value = {
            "detection_ids": [],
            "image_count": 0,
        }

        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "logical_filter", "source_type": "folder_upload"},
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["detection_ids"] == []
        assert data["total_count"] == 0
        assert data["image_count"] == 0

    def test_logical_filter_requires_source_type(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "logical_filter"},
        )

        assert resp.status_code == 400
        assert "source_type required" in resp.get_json()["message"]

    def test_unknown_mode_returns_400(self, client):
        resp = client.post(
            "/api/moderation/resolve-selection",
            json={"mode": "foobar"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# bulk/relabel
# ---------------------------------------------------------------------------


class TestBulkRelabel:
    def test_requires_detection_ids(self, client):
        resp = client.post(
            "/api/moderation/bulk/relabel",
            json={"species": "Parus_major"},
        )
        assert resp.status_code == 400
        assert "detection_ids" in resp.get_json()["message"]

    def test_requires_species(self, client):
        resp = client.post(
            "/api/moderation/bulk/relabel",
            json={"detection_ids": [1, 2]},
        )
        assert resp.status_code == 400
        assert "species" in resp.get_json()["message"]

    @patch("web.blueprints.moderation.gallery_service")
    @patch("web.blueprints.moderation.db_service")
    def test_relabel_succeeds(self, mock_db, mock_gallery, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.post(
            "/api/moderation/bulk/relabel",
            json={"detection_ids": [1, 2, 3], "species": "Parus_major"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["relabeled"] == 3
        assert data["new_species"] == "Parus_major"
        # 2 SQL calls per detection (detections + classifications)
        assert mock_conn.execute.call_count == 6
        mock_gallery.invalidate_cache.assert_called_once()


# ---------------------------------------------------------------------------
# bulk/reject
# ---------------------------------------------------------------------------


class TestBulkReject:
    def test_requires_targets(self, client):
        resp = client.post(
            "/api/moderation/bulk/reject",
            json={},
        )
        assert resp.status_code == 400

    @patch("web.blueprints.moderation.gallery_service")
    @patch("web.blueprints.moderation.db_service")
    def test_reject_detections(self, mock_db, mock_gallery, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)

        resp = client.post(
            "/api/moderation/bulk/reject",
            json={"detection_ids": [5, 6, 7]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["rejected_detections"] == 3
        mock_db.reject_detections.assert_called_once_with(mock_conn, [5, 6, 7])

    @patch("web.blueprints.moderation.gallery_service")
    @patch("web.blueprints.moderation.db_service")
    def test_reject_review_images(self, mock_db, mock_gallery, client):
        mock_conn = MagicMock()
        mock_db.closing_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_db.closing_connection.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.update_review_status.return_value = 2

        resp = client.post(
            "/api/moderation/bulk/reject",
            json={"image_filenames": ["orphan1.jpg", "orphan2.jpg"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["rejected_images"] == 2


# ---------------------------------------------------------------------------
# Auth-expiry: all endpoints require login
# ---------------------------------------------------------------------------


class TestAuthExpiry:
    ENDPOINTS = [
        ("/api/moderation/resolve-selection", "POST"),
        ("/api/moderation/bulk/relabel", "POST"),
        ("/api/moderation/bulk/reject", "POST"),
        ("/api/moderation/bulk/rescan", "POST"),
        ("/api/moderation/rescan-jobs/abc123/status", "GET"),
        ("/api/moderation/rescan-proposals/1/apply", "POST"),
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
    @patch("web.services.filter_service.db_service")
    @patch("web.services.filter_service.get_config")
    def test_species_overview_explicit_zero_keeps_zero_threshold(
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
                "detection_id": 11,
                "cls_class_name": "Parus_major",
                "od_class_name": "Parus_major",
                "score": 0.2,
            }
        ]

        ctx = FilterContext.from_dict(
            {
                "surface": "species_overview",
                "species_key": "Parus_major",
                "min_score": 0.0,
            }
        )

        result = resolve_filtered_ids(ctx)

        assert result.detection_ids == [11]
        assert result.total_count == 1

    @patch("web.services.filter_service.db_service")
    @patch("web.services.filter_service.get_config")
    def test_gallery_missing_min_score_falls_back_to_config_threshold(
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

        assert result.detection_ids == [22]
        assert result.total_count == 1

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
