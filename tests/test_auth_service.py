"""Tests for auth_service password verification and secret-key hardening."""

import os
import stat
from unittest.mock import patch

from web.services import auth_service

# ---------------------------------------------------------------------------
# authenticate() — empty-password bypass + constant-time comparison
# ---------------------------------------------------------------------------


class TestAuthenticate:
    def _with_stored(self, stored):
        return patch.object(
            auth_service.settings_core, "get_setting", return_value=stored
        )

    def test_empty_stored_password_rejects_any_input(self):
        # An unconfigured (empty) password must never authenticate, even
        # when the caller also submits an empty string.
        with self._with_stored(""):
            assert auth_service.authenticate("") is False
            assert auth_service.authenticate("anything") is False

    def test_none_stored_password_rejects(self):
        with self._with_stored(None):
            assert auth_service.authenticate("") is False
            assert auth_service.authenticate("x") is False

    def test_correct_password_authenticates(self):
        with self._with_stored("birdhouse123"):
            assert auth_service.authenticate("birdhouse123") is True

    def test_wrong_password_rejected(self):
        with self._with_stored("birdhouse123"):
            assert auth_service.authenticate("nope") is False

    def test_default_password_still_works_when_explicitly_set(self):
        # We only block the *empty* case here; the known-default value is
        # surfaced separately via is_default_password()/warn banners and
        # the first-run setup flow, not by refusing login.
        with self._with_stored("watchmybirds"):
            assert auth_service.authenticate("watchmybirds") is True

    def test_non_ascii_password_does_not_raise(self):
        with self._with_stored("paßwörtД"):
            assert auth_service.authenticate("paßwörtД") is True
            assert auth_service.authenticate("other") is False

    def test_none_provided_password_is_safe(self):
        with self._with_stored("birdhouse123"):
            assert auth_service.authenticate(None) is False


# ---------------------------------------------------------------------------
# Secret-key file is created with mode 0o600 (no TOCTOU window)
# ---------------------------------------------------------------------------


def test_config_secret_key_created_with_600_perms(tmp_path):
    import config

    secret_file = tmp_path / "secret.key"
    key = config.get_or_create_secret_key({"OUTPUT_DIR": str(tmp_path)})

    assert len(key) >= 32
    assert secret_file.exists()
    mode = stat.S_IMODE(os.stat(secret_file).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"


def test_config_secret_key_is_persistent(tmp_path):
    import config

    first = config.get_or_create_secret_key({"OUTPUT_DIR": str(tmp_path)})
    second = config.get_or_create_secret_key({"OUTPUT_DIR": str(tmp_path)})
    assert first == second


def test_config_secret_key_reads_existing_from_concurrent_writer(tmp_path):
    import config

    # Simulate a key written by another process before we generate ours.
    secret_file = tmp_path / "secret.key"
    existing = "a" * 64
    secret_file.write_text(existing)

    key = config.get_or_create_secret_key({"OUTPUT_DIR": str(tmp_path)})
    assert key == existing
