"""Persistence regression: TELEGRAM_MODE must survive Docker-deploy cycles.

Before the fix, ``update_runtime_settings`` removed any RUNTIME_KEY from
``settings.yaml`` when its value equalled the default and no env override
existed. That looked clean ("settings.yaml only carries deviations") but
collided with the legacy migration in ``_load_config``, which re-derives
``TELEGRAM_MODE`` from a lingering ``TELEGRAM_ENABLED`` whenever the mode
key is absent from YAML. Net effect on the operator: every fresh Docker
image rebuild flipped ``TELEGRAM_MODE`` from the chosen "off" (or
"daily", etc.) back to "live" ("Instant" in the UI) — exactly the
opposite of what they had set.

The fix:
- Always persist explicit user choices to YAML, default-equal or not.
- The legacy migration only fires when ``TELEGRAM_ENABLED`` lives
  inside the YAML itself, not when it comes from env / config defaults.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_config(monkeypatch, tmp_path: Path):
    """Reload config.py with OUTPUT_DIR pointing at tmp_path."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    sys.modules.pop("config", None)
    import config
    importlib.reload(config)
    # Reset the singleton so get_config rebuilds from env in the new tmp_path.
    config._CONFIG = None
    yield config
    config._CONFIG = None


def test_setting_telegram_mode_off_persists_to_yaml(fresh_config, tmp_path):
    """User picks 'off' in Settings → it must land in settings.yaml."""
    from utils.settings import load_settings_yaml

    cfg = fresh_config.get_config()
    cfg["TELEGRAM_MODE"] = "live"  # simulate prior state
    fresh_config.update_runtime_settings({"TELEGRAM_MODE": "off"})

    yaml_settings = load_settings_yaml(str(tmp_path))
    assert yaml_settings.get("TELEGRAM_MODE") == "off"


def test_setting_telegram_mode_default_still_persists(fresh_config, tmp_path):
    """Even when the picked value matches the DEFAULTS, persist it.

    The previous "skip default values" optimisation produced the
    Docker-deploy regression — keep the value written so the next
    boot reads the explicit choice instead of falling back to legacy
    migration.
    """
    from utils.settings import load_settings_yaml

    fresh_config.update_runtime_settings({"TELEGRAM_MODE": "off"})

    yaml_settings = load_settings_yaml(str(tmp_path))
    assert "TELEGRAM_MODE" in yaml_settings
    assert yaml_settings["TELEGRAM_MODE"] == "off"


def test_legacy_migration_does_not_fire_when_env_provides_telegram_enabled(
    monkeypatch, tmp_path
):
    """Critical Docker-deploy scenario: settings.yaml has TELEGRAM_MODE=off,
    docker-compose.yml has TELEGRAM_ENABLED=True in env. The legacy
    migration must NOT promote the env-derived TELEGRAM_ENABLED to
    TELEGRAM_MODE=live — that was the bug.
    """
    # Pre-write a settings.yaml with explicit off mode but no
    # TELEGRAM_ENABLED row (operator chose "off" in the UI).
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text("TELEGRAM_MODE: off\n", encoding="utf-8")

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ENABLED", "True")  # docker-compose sets it
    sys.modules.pop("config", None)
    import config
    importlib.reload(config)
    config._CONFIG = None

    cfg = config.get_config()
    assert cfg["TELEGRAM_MODE"] == "off"


def test_legacy_migration_fires_when_yaml_has_telegram_enabled_only(
    monkeypatch, tmp_path
):
    """Real upgrade from a pre-TELEGRAM_MODE build: settings.yaml has
    only TELEGRAM_ENABLED=true and no mode. The migration must still
    promote that to TELEGRAM_MODE=live for backwards compat."""
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text("TELEGRAM_ENABLED: true\n", encoding="utf-8")

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_ENABLED", raising=False)
    sys.modules.pop("config", None)
    import config
    importlib.reload(config)
    config._CONFIG = None

    cfg = config.get_config()
    assert cfg["TELEGRAM_MODE"] == "live"


def test_explicit_telegram_mode_in_yaml_wins_over_telegram_enabled(
    monkeypatch, tmp_path
):
    """Belt-and-suspenders: when YAML has both keys, TELEGRAM_MODE
    is the source of truth and the legacy migration is skipped."""
    yaml_path = tmp_path / "settings.yaml"
    yaml_path.write_text(
        "TELEGRAM_MODE: daily\nTELEGRAM_ENABLED: true\n", encoding="utf-8"
    )

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    sys.modules.pop("config", None)
    import config
    importlib.reload(config)
    config._CONFIG = None

    cfg = config.get_config()
    assert cfg["TELEGRAM_MODE"] == "daily"
