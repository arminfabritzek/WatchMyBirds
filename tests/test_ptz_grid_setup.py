"""Tests for the grid-mode setup wizard backend.

Covers:
  - storage round-trip for grid_shape and grid_cells
  - ptz_core.set_grid_shape / set_grid_cell_at_current_position / clear_grid_cell
  - get_grid_state computes missing cells correctly
  - HTTP routes accept / reject inputs as specified
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(monkeypatch):
    """Fresh CameraStorage with a temp yaml file."""
    from utils.camera_storage import CameraStorage

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "cameras.yaml"
        s = CameraStorage(storage_path=str(yaml_path))
        # Seed one camera so camera_id=0 is valid.
        s.add_camera({"name": "Test Cam", "ip": "10.0.0.1", "ptz": {}})
        yield s


class TestStorageGridRoundtrip:
    def test_set_grid_shape_persists(self, storage):
        assert storage.set_grid_shape(0, 3, 4) is True
        cam = storage.get_camera(0, include_password=False)
        assert cam is not None
        assert cam["ptz"]["grid_shape"] == [3, 4]

    def test_set_grid_cell_persists(self, storage):
        assert storage.set_grid_cell(0, "r1_c2", "Preset042") is True
        cam = storage.get_camera(0, include_password=False)
        assert cam is not None
        assert cam["ptz"]["grid_cells"] == {"r1_c2": "Preset042"}

    def test_multiple_cells_accumulate(self, storage):
        storage.set_grid_cell(0, "r0_c0", "Preset001")
        storage.set_grid_cell(0, "r0_c1", "Preset002")
        storage.set_grid_cell(0, "r1_c0", "Preset003")
        cam = storage.get_camera(0, include_password=False)
        assert cam is not None
        assert cam["ptz"]["grid_cells"] == {
            "r0_c0": "Preset001",
            "r0_c1": "Preset002",
            "r1_c0": "Preset003",
        }

    def test_set_grid_cell_overwrites(self, storage):
        storage.set_grid_cell(0, "r0_c0", "Preset001")
        storage.set_grid_cell(0, "r0_c0", "Preset099")  # operator redid this cell
        cam = storage.get_camera(0, include_password=False)
        assert cam is not None
        assert cam["ptz"]["grid_cells"] == {"r0_c0": "Preset099"}

    def test_delete_grid_cell_removes_entry(self, storage):
        storage.set_grid_cell(0, "r0_c0", "Preset001")
        storage.set_grid_cell(0, "r0_c1", "Preset002")
        assert storage.delete_grid_cell(0, "r0_c0") is True
        cam = storage.get_camera(0, include_password=False)
        assert cam is not None
        assert cam["ptz"]["grid_cells"] == {"r0_c1": "Preset002"}

    def test_delete_grid_cell_missing_is_noop(self, storage):
        assert storage.delete_grid_cell(0, "r9_c9") is True  # no-op, not failure

    def test_invalid_camera_id_returns_false(self, storage):
        assert storage.set_grid_shape(99, 3, 3) is False
        assert storage.set_grid_cell(99, "r0_c0", "tok") is False
        assert storage.delete_grid_cell(99, "r0_c0") is False


# ---------------------------------------------------------------------------
# ptz_core grid functions (with patched client)
# ---------------------------------------------------------------------------


@pytest.fixture
def core_storage(monkeypatch, storage):
    """Patch ptz_core.get_camera_storage() to return our tmp storage."""
    from core import ptz_core

    monkeypatch.setattr(ptz_core, "get_camera_storage", lambda: storage)
    return storage


class TestPtzCoreGridShape:
    def test_set_grid_shape_valid(self, core_storage):
        from core import ptz_core

        result = ptz_core.set_grid_shape(0, 3, 3)
        assert result == {"rows": 3, "cols": 3, "mode_auto_set": True}

    def test_set_grid_shape_rejects_invalid(self, core_storage):
        from core import ptz_core

        with pytest.raises(ValueError, match="not in allowed set"):
            ptz_core.set_grid_shape(0, 5, 5)

    def test_set_grid_shape_unknown_camera(self, core_storage):
        from core import ptz_core

        assert ptz_core.set_grid_shape(99, 3, 3) is None


class TestPtzCoreGridCell:
    def test_set_cell_creates_preset_and_mapping(self, core_storage):
        from core import ptz_core

        fake_client = MagicMock()
        fake_client.set_preset.return_value = "Preset042"
        with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
            result = ptz_core.set_grid_cell_at_current_position(0, 1, 2)

        assert result == {
            "cell_key": "r1_c2",
            "name": "grid_r1_c2",
            "preset_token": "Preset042",
            "mode_auto_set": True,
        }
        # Client was called with the canonical name and no pre-existing token.
        fake_client.set_preset.assert_called_once_with(
            name="grid_r1_c2", preset_token=None
        )

    def test_set_cell_reuses_existing_token_on_redo(self, core_storage):
        from core import ptz_core

        # Pre-populate one cell.
        core_storage.set_grid_cell(0, "r0_c0", "Preset007")

        fake_client = MagicMock()
        # Camera returns the same token (typical: SetPreset with token
        # updates the slot in place).
        fake_client.set_preset.return_value = "Preset007"
        with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
            ptz_core.set_grid_cell_at_current_position(0, 0, 0)

        # Existing token was passed so the camera knows to overwrite.
        fake_client.set_preset.assert_called_once_with(
            name="grid_r0_c0", preset_token="Preset007"
        )

    def test_clear_cell_removes_mapping(self, core_storage):
        from core import ptz_core

        core_storage.set_grid_cell(0, "r0_c0", "Preset007")
        assert ptz_core.clear_grid_cell(0, 0, 0) is True
        cam = core_storage.get_camera(0, include_password=False)
        assert cam is not None
        assert cam["ptz"].get("grid_cells", {}) == {}

    def test_clear_cell_unknown_camera(self, core_storage):
        from core import ptz_core

        assert ptz_core.clear_grid_cell(99, 0, 0) is False


class TestEnsureGridMode:
    """Wizard actions auto-flip ptz.mode to "grid".

    Regression guard for the "wizard works, auto-PTZ doesn't" failure
    mode: an operator who completes the wizard but never opens the
    modal to flip mode would persist grid_cells but never trigger the
    grid dispatch path. Each wizard entry point now sets mode="grid"
    idempotently and reports mode_auto_set in its return dict.
    """

    def test_shape_call_flips_preset_mode_to_grid(self, core_storage, storage):
        from core import ptz_core

        # Default mode after add_camera() is "preset".
        result = ptz_core.set_grid_shape(0, 3, 3)
        assert result == {"rows": 3, "cols": 3, "mode_auto_set": True}
        cam = storage.get_camera(0, include_password=False)
        assert cam["ptz"]["mode"] == "grid"

    def test_shape_call_idempotent_when_already_grid(
        self, core_storage, storage
    ):
        from core import ptz_core

        storage.update_ptz_config(0, {"mode": "grid"})
        result = ptz_core.set_grid_shape(0, 3, 3)
        assert result["mode_auto_set"] is False
        cam = storage.get_camera(0, include_password=False)
        assert cam["ptz"]["mode"] == "grid"

    def test_cell_at_current_position_flips_mode(self, core_storage, storage):
        from core import ptz_core

        fake_client = MagicMock()
        fake_client.set_preset.return_value = "Preset042"
        with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
            result = ptz_core.set_grid_cell_at_current_position(0, 1, 2)
        assert result["mode_auto_set"] is True
        cam = storage.get_camera(0, include_password=False)
        assert cam["ptz"]["mode"] == "grid"

    def test_link_to_existing_preset_flips_mode(self, core_storage, storage):
        from camera.ptz_client import PtzPreset
        from core import ptz_core

        fake_client = MagicMock()
        fake_client.list_presets.return_value = [
            PtzPreset(token="Preset007", name="garden"),
        ]
        with patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
            result = ptz_core.link_grid_cell_to_existing_preset(
                0, 0, 0, "Preset007"
            )
        assert result["mode_auto_set"] is True
        cam = storage.get_camera(0, include_password=False)
        assert cam["ptz"]["mode"] == "grid"


class TestPtzCoreGridState:
    def test_state_with_no_cells_set(self, core_storage):
        from core import ptz_core

        state = ptz_core.get_grid_state(0)
        assert state is not None
        assert state["shape"] == [3, 3]  # default
        assert state["total_required"] == 9
        assert state["total_set"] == 0
        assert len(state["missing"]) == 9
        assert state["mode_active"] is False

    def test_state_with_partial_cells(self, core_storage):
        from core import ptz_core

        core_storage.set_grid_shape(0, 3, 3)
        core_storage.set_grid_cell(0, "r0_c0", "P1")
        core_storage.set_grid_cell(0, "r1_c1", "P5")

        state = ptz_core.get_grid_state(0)
        assert state is not None
        assert state["total_set"] == 2
        assert state["total_required"] == 9
        assert state["cells"] == {"r0_c0": "P1", "r1_c1": "P5"}
        assert set(state["missing"]) == {
            "r0_c1", "r0_c2",
            "r1_c0",         "r1_c2",
            "r2_c0", "r2_c1", "r2_c2",
        }

    def test_state_unknown_camera(self, core_storage):
        from core import ptz_core

        assert ptz_core.get_grid_state(99) is None


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.secret_key = "test-secret-key"
    mock_dm = MagicMock()
    mock_dm.paused = False
    from web.blueprints.auth import auth_bp

    app.register_blueprint(auth_bp)
    from web.blueprints.api_v1 import api_v1

    api_v1.detection_manager = mock_dm
    app.register_blueprint(api_v1)
    return app


@pytest.fixture
def client(app):
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


class TestGridRoutes:
    def test_get_grid_state_route(self, client):
        from web.blueprints import api_v1 as api_v1_module

        with patch.object(api_v1_module, "ptz_service") as mock_ptz:
            mock_ptz.get_grid_state.return_value = {
                "shape": [3, 3],
                "rows": 3,
                "cols": 3,
                "cells": {},
                "missing": [f"r{r}_c{c}" for r in range(3) for c in range(3)],
                "total_required": 9,
                "total_set": 0,
                "mode_active": False,
            }
            response = client.get("/api/v1/cameras/0/ptz/grid/state")

        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "success"
        assert body["grid"]["total_required"] == 9

    def test_get_grid_state_unknown_camera(self, client):
        from web.blueprints import api_v1 as api_v1_module

        with patch.object(api_v1_module, "ptz_service") as mock_ptz:
            mock_ptz.get_grid_state.return_value = None
            response = client.get("/api/v1/cameras/99/ptz/grid/state")

        assert response.status_code == 404

    def test_put_grid_shape_valid(self, client):
        from web.blueprints import api_v1 as api_v1_module

        with patch.object(api_v1_module, "ptz_service") as mock_ptz:
            mock_ptz.set_grid_shape.return_value = {"rows": 3, "cols": 4}
            response = client.put(
                "/api/v1/cameras/0/ptz/grid/shape",
                json={"rows": 3, "cols": 4},
            )

        assert response.status_code == 200
        body = response.get_json()
        assert body["shape"] == {"rows": 3, "cols": 4}

    def test_put_grid_shape_invalid_dimensions(self, client):
        from web.blueprints import api_v1 as api_v1_module

        with patch.object(api_v1_module, "ptz_service") as mock_ptz:
            mock_ptz.set_grid_shape.side_effect = ValueError(
                "Grid shape (5, 5) not in allowed set"
            )
            response = client.put(
                "/api/v1/cameras/0/ptz/grid/shape",
                json={"rows": 5, "cols": 5},
            )

        assert response.status_code == 400

    def test_put_grid_shape_missing_fields(self, client):
        response = client.put("/api/v1/cameras/0/ptz/grid/shape", json={"rows": 3})
        assert response.status_code == 400

    def test_put_grid_cell_calls_service(self, client):
        from web.blueprints import api_v1 as api_v1_module

        with patch.object(api_v1_module, "ptz_service") as mock_ptz:
            mock_ptz.set_grid_cell_at_current_position.return_value = {
                "cell_key": "r1_c2",
                "name": "grid_r1_c2",
                "preset_token": "Preset042",
            }
            response = client.put("/api/v1/cameras/0/ptz/grid/cells/1/2")

        assert response.status_code == 200
        mock_ptz.set_grid_cell_at_current_position.assert_called_once_with(0, 1, 2)

    def test_delete_grid_cell(self, client):
        from web.blueprints import api_v1 as api_v1_module

        with patch.object(api_v1_module, "ptz_service") as mock_ptz:
            mock_ptz.clear_grid_cell.return_value = True
            response = client.delete("/api/v1/cameras/0/ptz/grid/cells/1/2")

        assert response.status_code == 200
        mock_ptz.clear_grid_cell.assert_called_once_with(0, 1, 2)

    def test_routes_require_authentication(self, app):
        with app.test_client() as unauth:
            for path in [
                "/api/v1/cameras/0/ptz/grid/state",
            ]:
                r = unauth.get(path)
                assert r.status_code != 200, f"{path} accessible without auth"
