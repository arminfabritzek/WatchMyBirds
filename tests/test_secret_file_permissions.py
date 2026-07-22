"""Credential-bearing YAML stores must not inherit a permissive umask.

``settings.yaml`` can hold EDIT_PASSWORD / TELEGRAM_BOT_TOKEN and
``cameras.yaml`` holds ONVIF credentials in clear text. Both are written
by the app at runtime, so the mode has to be asserted at write time
rather than relying on the deployment's directory permissions.
"""

from __future__ import annotations

import stat

from utils.camera_storage import CameraStorage
from utils.settings import save_settings_yaml


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_settings_yaml_is_owner_only(tmp_path):
    save_settings_yaml({"EDIT_PASSWORD": "hunter2"}, output_dir=str(tmp_path))

    written = tmp_path / "settings.yaml"
    assert written.is_file()
    assert _mode(written) == 0o600


def test_cameras_yaml_is_owner_only(tmp_path):
    store = CameraStorage(storage_path=str(tmp_path / "cameras.yaml"))
    store.add_camera(ip="10.0.0.2", username="u", password="p", name="cam")

    written = tmp_path / "cameras.yaml"
    assert written.is_file()
    assert _mode(written) == 0o600


def test_cameras_yaml_permission_is_repaired_on_rewrite(tmp_path):
    """A store created before this rule is tightened on the next write."""
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras: []\n")
    path.chmod(0o644)

    store = CameraStorage(storage_path=str(path))
    store.add_camera(ip="10.0.0.3", name="cam")

    assert _mode(path) == 0o600
