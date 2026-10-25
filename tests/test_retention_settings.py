from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from config import get_config, update_runtime_settings, validate_runtime_updates
from tests.labeling_helpers import _reset_test_config
from utils.db import connection
from utils.settings import load_settings_yaml, save_settings_yaml


@pytest.mark.parametrize("existing", [False, True])
def test_installation_defaults_persist_across_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing: bool
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    if existing:
        sqlite3.connect(output / "images.db").close()
    save_settings_yaml(
        {
            "RETENTION_ENABLED": True,
            "RETENTION_POSTURE": "reclaim",
            "RETENTION_AUTO_ENABLED": True,
        },
        str(output),
    )
    with connection.closing_connection():
        pass
    cfg = get_config()
    assert cfg["AUTOMATIC_DELETION_ENABLED"] is (not existing)
    assert bool(cfg["AUTOMATIC_DELETION_ENABLED_AT"]) is (not existing)
    assert cfg["OLD_BIRD_PHOTOS"] == "keep_previews"
    before = load_settings_yaml(str(output))
    connection._schema_initialized_paths.clear()
    with connection.closing_connection():
        pass
    assert before == load_settings_yaml(str(output))


def test_reenable_gets_new_date_but_saving_enabled_does_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    with connection.closing_connection():
        pass
    update_runtime_settings({"AUTOMATIC_DELETION_ENABLED": False})
    update_runtime_settings(
        {"AUTOMATIC_DELETION_ENABLED": True, "OLD_BIRD_PHOTOS": "delete_all"}
    )
    first = load_settings_yaml(str(output))["AUTOMATIC_DELETION_ENABLED_AT"]
    update_runtime_settings({"AUTOMATIC_DELETION_ENABLED": True})
    assert load_settings_yaml(str(output))["AUTOMATIC_DELETION_ENABLED_AT"] == first
    update_runtime_settings({"AUTOMATIC_DELETION_ENABLED": False})
    update_runtime_settings({"AUTOMATIC_DELETION_ENABLED": True})
    saved = load_settings_yaml(str(output))
    assert datetime.fromisoformat(
        saved["AUTOMATIC_DELETION_ENABLED_AT"]
    ) > datetime.fromisoformat(first)
    assert saved["OLD_BIRD_PHOTOS"] == "delete_all"


def test_removed_keys_are_ignored_and_activation_date_is_not_user_editable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    old = {
        "RETENTION_ENABLED": True,
        "RETENTION_POSTURE": "reclaim",
        "RETENTION_AUTO_ENABLED": True,
        "RETENTION_PROTECT_UNREVIEWED": False,
        "RETENTION_PROTECT_FAVORITES": False,
        "RETENTION_DAYS": 1,
    }
    save_settings_yaml(old, str(output))
    import config

    loaded = config._load_config()
    assert not old.keys() & loaded.keys()
    assert "AUTOMATIC_DELETION_ENABLED_AT" not in config.RUNTIME_KEYS
    assert validate_runtime_updates({"OLD_BIRD_PHOTOS": "unknown"})[1]
    assert (
        "AUTOMATIC_DELETION_ENABLED_AT"
        not in validate_runtime_updates(
            {"AUTOMATIC_DELETION_ENABLED_AT": "2000-01-01"}
        )[0]
    )


def test_retention_polling_recomputes_both_panels_after_every_toggle() -> None:
    template = (Path(__file__).parents[1] / "templates/settings.html").read_text()
    assert "requestAnimationFrame(scheduledJobPanelsVisibilityCheck);" in template
    assert "_nightlyPollVisible || _retentionPollVisible" in template
