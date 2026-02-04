"""
API v1 Blueprint Tests.

These tests verify that the /api/v1 endpoints return the expected structure
as documented in docs/API.md.
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def app():
    """Create a minimal Flask app with API v1 and auth blueprints."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"

    # Create mock detection manager
    mock_dm = MagicMock()
    mock_dm.paused = False

    # Register auth blueprint (needed for login_required to work)
    from web.blueprints.auth import auth_bp

    app.register_blueprint(auth_bp)

    # Register API v1 blueprint
    from web.blueprints.api_v1 import api_v1

    api_v1.detection_manager = mock_dm
    app.register_blueprint(api_v1)

    return app


@pytest.fixture
def client(app):
    """Create authenticated test client."""
    with app.test_client() as client:
        # Set authenticated session
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


class TestApiV1Status:
    """Test /api/v1/status endpoint."""

    def test_status_returns_expected_fields(self, client):
        """GET /api/v1/status returns detection state fields."""
        with patch("web.blueprints.api_v1.backup_restore_service") as mock_backup:
            mock_backup.is_restart_required.return_value = False

            response = client.get("/api/v1/status")

            assert response.status_code == 200
            data = response.get_json()

            # Per docs/API.md:
            assert "detection_paused" in data
            assert "detection_running" in data
            assert "restart_required" in data

            # Types must be boolean
            assert isinstance(data["detection_paused"], bool)
            assert isinstance(data["detection_running"], bool)
            assert isinstance(data["restart_required"], bool)


class TestApiV1Settings:
    """Test /api/v1/settings endpoints."""

    def test_settings_get_returns_json(self, client):
        """GET /api/v1/settings returns settings as JSON."""
        with patch("config.get_settings_payload") as mock_payload:
            mock_payload.return_value = {
                "VIDEO_SOURCE": "0",
                "DEBUG_MODE": False,
            }

            response = client.get("/api/v1/settings")

            assert response.status_code == 200
            data = response.get_json()

            assert "VIDEO_SOURCE" in data
            assert data["VIDEO_SOURCE"] == "0"

    def test_settings_post_returns_success_structure(self, client):
        """POST /api/v1/settings returns success response."""
        with patch("config.validate_runtime_updates") as mock_validate:
            with patch("config.update_runtime_settings"):
                mock_validate.return_value = ({"DEBUG_MODE": True}, [])

                response = client.post("/api/v1/settings", json={"DEBUG_MODE": True})

                assert response.status_code == 200
                data = response.get_json()

                # Per docs/API.md:
                assert data["status"] == "success"


class TestApiV1Onvif:
    """Test /api/v1/onvif endpoints."""

    def test_onvif_discover_returns_cameras_list(self, client):
        """GET /api/v1/onvif/discover returns camera list structure."""
        with patch("web.blueprints.api_v1.onvif_service") as mock_service:
            mock_service.discover_cameras.return_value = [
                {"ip": "192.168.1.100", "port": 80, "name": "Test Camera"}
            ]

            response = client.get("/api/v1/onvif/discover")

            assert response.status_code == 200
            data = response.get_json()

            # Per docs/API.md:
            assert data["status"] == "success"
            assert "cameras" in data
            assert isinstance(data["cameras"], list)
            assert len(data["cameras"]) == 1
            assert data["cameras"][0]["ip"] == "192.168.1.100"

    def test_onvif_discover_empty_returns_empty_list(self, client):
        """GET /api/v1/onvif/discover with no cameras returns empty list."""
        with patch("web.blueprints.api_v1.onvif_service") as mock_service:
            mock_service.discover_cameras.return_value = []

            response = client.get("/api/v1/onvif/discover")

            assert response.status_code == 200
            data = response.get_json()

            # Per docs/API.md: empty result still returns success
            assert data["status"] == "success"
            assert data["cameras"] == []


class TestApiV1Analytics:
    """Test /api/v1/analytics endpoints."""

    def test_analytics_summary_returns_structure(self, client):
        """GET /api/v1/analytics/summary returns expected structure."""
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=MagicMock())
        mock_context.__exit__ = MagicMock(return_value=False)

        with patch("web.blueprints.api_v1.db_service") as mock_db:
            mock_db.get_connection.return_value = mock_context
            mock_db.fetch_analytics_summary.return_value = {
                "total_detections": 100,
                "total_species": 5,
                "total_days": 10,
            }

            response = client.get("/api/v1/analytics/summary")

            assert response.status_code == 200
            data = response.get_json()

            # Per docs/API.md:
            assert "total_detections" in data
            assert "total_species" in data
            assert "total_days" in data


class TestApiV1DetectionControl:
    """Test detection pause/resume endpoints."""

    def test_detection_pause_success(self, app, client):
        """POST /api/v1/detection/pause pauses detection."""
        # Get the mock from blueprint
        from web.blueprints.api_v1 import api_v1

        api_v1.detection_manager.paused = False

        response = client.post("/api/v1/detection/pause")

        assert response.status_code == 200
        data = response.get_json()

        # Per docs/API.md:
        assert data["status"] == "success"
        assert "message" in data

    def test_detection_resume_success(self, app, client):
        """POST /api/v1/detection/resume resumes detection."""
        from web.blueprints.api_v1 import api_v1

        api_v1.detection_manager.paused = True

        response = client.post("/api/v1/detection/resume")

        assert response.status_code == 200
        data = response.get_json()

        # Per docs/API.md:
        assert data["status"] == "success"
        assert "message" in data


class TestApiV1SystemVitals:
    """Test /api/v1/system/vitals endpoint."""

    def test_vitals_without_monitor_returns_fallback(self, client):
        """GET /api/v1/system/vitals returns fallback when monitor not set."""
        from web.blueprints.api_v1 import api_v1

        # Ensure system_monitor is not set
        api_v1.system_monitor = None

        response = client.get("/api/v1/system/vitals")

        assert response.status_code == 200
        data = response.get_json()

        # Response structure
        assert data["status"] == "success"
        assert data["monitor_active"] is False
        assert "vitals" in data

        # Vitals shape
        vitals = data["vitals"]
        assert "ts" in vitals
        assert "cpu_percent" in vitals
        assert "ram_percent" in vitals
        assert "cpu_temp_c" in vitals
        assert "throttled" in vitals

        # Types
        assert isinstance(vitals["cpu_percent"], (int, float))
        assert isinstance(vitals["ram_percent"], (int, float))

    def test_vitals_with_monitor_returns_monitor_data(self, client):
        """GET /api/v1/system/vitals returns data from SystemMonitor."""
        from web.blueprints.api_v1 import api_v1

        # Create mock SystemMonitor
        mock_monitor = MagicMock()
        mock_monitor.get_current_vitals.return_value = {
            "ts": "2026-02-04T23:00:00",
            "cpu_percent": 42.5,
            "ram_percent": 65.2,
            "cpu_temp_c": 55.0,
            "throttled": None,
            "core_voltage": 1.35,
        }
        api_v1.system_monitor = mock_monitor

        response = client.get("/api/v1/system/vitals")

        assert response.status_code == 200
        data = response.get_json()

        # Response structure
        assert data["status"] == "success"
        assert data["monitor_active"] is True
        assert "vitals" in data

        # Vitals from monitor
        vitals = data["vitals"]
        assert vitals["ts"] == "2026-02-04T23:00:00"
        assert vitals["cpu_percent"] == 42.5
        assert vitals["ram_percent"] == 65.2
        assert vitals["cpu_temp_c"] == 55.0
        assert vitals["core_voltage"] == 1.35

        # Verify monitor was called
        mock_monitor.get_current_vitals.assert_called_once()

        # Cleanup
        api_v1.system_monitor = None

    def test_vitals_returns_required_json_shape(self, client):
        """GET /api/v1/system/vitals returns stable JSON shape for frontend."""
        from web.blueprints.api_v1 import api_v1

        mock_monitor = MagicMock()
        mock_monitor.get_current_vitals.return_value = {
            "ts": "2026-02-04T23:00:00",
            "cpu_percent": 50.0,
            "ram_percent": 70.0,
            "cpu_temp_c": 60.0,
            "throttled": {"under_voltage_now": False},
        }
        api_v1.system_monitor = mock_monitor

        response = client.get("/api/v1/system/vitals")
        data = response.get_json()

        # Required top-level keys
        required_keys = ["status", "monitor_active", "vitals"]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

        # Required vitals keys (per Phase D spec)
        vitals = data["vitals"]
        required_vitals_keys = [
            "ts",
            "cpu_percent",
            "ram_percent",
            "cpu_temp_c",
            "throttled",
        ]
        for key in required_vitals_keys:
            assert key in vitals, f"Missing required vitals key: {key}"

        # Cleanup
        api_v1.system_monitor = None
