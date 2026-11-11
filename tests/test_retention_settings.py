from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from config import get_config, update_runtime_settings, validate_runtime_updates
from core.retention_core import settings_for_plan
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


@pytest.mark.parametrize(
    ("stored", "days", "preview"),
    [
        (None, 30, 180),
        (90, 90, 180),
        ("90", 90, 180),
        (365, 365, 365),
        ("abc", 30, 180),
        (0, 30, 180),
        (-5, 30, 180),
        (99999, 3650, 3650),
        ("90.0", 90, 180),
        (90.0, 90, 180),
        (True, 30, 180),
        (12.5, 30, 180),
    ],
)
def test_original_days_come_from_settings(
    stored: object, days: int, preview: int
) -> None:
    values = {} if stored is None else {"ORIGINAL_PHOTO_DAYS": stored}
    plan = settings_for_plan(values)
    assert (plan.original_days, plan.preview_days) == (days, preview)


def test_original_days_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    assert validate_runtime_updates({"ORIGINAL_PHOTO_DAYS": 90})[0] == {
        "ORIGINAL_PHOTO_DAYS": 90
    }
    assert validate_runtime_updates({"ORIGINAL_PHOTO_DAYS": "45"})[0] == {
        "ORIGINAL_PHOTO_DAYS": 45
    }
    for bad in (6, 3651, "x", None, 12.5, True):
        assert validate_runtime_updates({"ORIGINAL_PHOTO_DAYS": bad})[1], bad


def test_bad_stored_days_load_as_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    save_settings_yaml({"ORIGINAL_PHOTO_DAYS": "abc"}, str(output))
    import config

    assert config._load_config()["ORIGINAL_PHOTO_DAYS"] == 30


def test_deletion_defaults_show_configured_days(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    update_runtime_settings({"ORIGINAL_PHOTO_DAYS": 90})
    from web.services import storage_service

    assert storage_service.deletion_defaults()["original_days"] == 90


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_training_keep_comes_from_settings(stored: object, expected: bool) -> None:
    values = {} if stored is None else {"KEEP_ANSWERED_PHOTOS_FOR_TRAINING": stored}
    assert settings_for_plan(values).keep_for_training is expected


def test_training_keep_is_a_boolean_runtime_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    assert get_config()["KEEP_ANSWERED_PHOTOS_FOR_TRAINING"] is True
    assert validate_runtime_updates({"KEEP_ANSWERED_PHOTOS_FOR_TRAINING": "false"})[
        0
    ] == {"KEEP_ANSWERED_PHOTOS_FOR_TRAINING": False}


def test_unclear_training_keep_value_never_releases_photos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    for unclear in ("maybe", "", None, 0, 2):
        assert validate_runtime_updates({"KEEP_ANSWERED_PHOTOS_FOR_TRAINING": unclear})[
            1
        ], unclear
        assert settings_for_plan(
            {"KEEP_ANSWERED_PHOTOS_FOR_TRAINING": unclear}
        ).keep_for_training
    save_settings_yaml({"KEEP_ANSWERED_PHOTOS_FOR_TRAINING": "maybe"}, str(output))
    import config

    assert config._load_config()["KEEP_ANSWERED_PHOTOS_FOR_TRAINING"] is True


@pytest.mark.parametrize("existing", [False, True])
def test_training_keep_default_follows_database_existence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing: bool
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    if existing:
        sqlite3.connect(output / "images.db").close()
    with connection.closing_connection():
        pass
    key = "KEEP_ANSWERED_PHOTOS_FOR_TRAINING"
    assert load_settings_yaml(str(output))[key] is existing
    assert get_config()[key] is existing
    update_runtime_settings({key: not existing})
    connection._schema_initialized_paths.clear()
    with connection.closing_connection():
        pass
    assert load_settings_yaml(str(output))[key] is (not existing)


def test_training_keep_default_preserves_an_explicit_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _reset_test_config(monkeypatch, tmp_path)
    sqlite3.connect(output / "images.db").close()
    key = "KEEP_ANSWERED_PHOTOS_FOR_TRAINING"
    save_settings_yaml({key: False}, str(output))
    with connection.closing_connection():
        pass
    assert load_settings_yaml(str(output))[key] is False


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, 256 * 1024**2),
        (10, 10 * 1024**3),
        ("2.5", int(2.5 * 1024**3)),
        ("x", 256 * 1024**2),
        (0, 256 * 1024**2),
        (-1, 256 * 1024**2),
        (5000, 256 * 1024**2),
    ],
)
def test_reserve_comes_from_settings(stored: object, expected: int) -> None:
    values = {} if stored is None else {"MIN_FREE_SPACE_GB": stored}
    assert settings_for_plan(values).reserve_bytes == expected


def test_reserve_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    _reset_test_config(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "config.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=2000 * 1024**3, free=1500 * 1024**3),
    )
    assert get_config()["MIN_FREE_SPACE_GB"] == 0.25
    assert validate_runtime_updates({"MIN_FREE_SPACE_GB": "10"})[0] == {
        "MIN_FREE_SPACE_GB": 10.0
    }
    for bad in (0.1, 1001, "x", None, True):
        assert validate_runtime_updates({"MIN_FREE_SPACE_GB": bad})[1], bad


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, 50),
        (0, 0),
        ("50", 50),
        (1000, 1000),
        (-1, 50),
        (1001, 1000),
        (5000, 1000),
        ("x", 50),
    ],
)
def test_best_originals_come_from_settings(stored: object, expected: int) -> None:
    values = {} if stored is None else {"BEST_ORIGINALS_PER_SPECIES": stored}
    assert settings_for_plan(values).best_per_species == expected


def test_best_originals_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    assert get_config()["BEST_ORIGINALS_PER_SPECIES"] == 50
    assert validate_runtime_updates({"BEST_ORIGINALS_PER_SPECIES": "0"})[0] == {
        "BEST_ORIGINALS_PER_SPECIES": 0
    }
    for bad in (-1, 1001, "x", None, 2.5, True):
        assert validate_runtime_updates({"BEST_ORIGINALS_PER_SPECIES": bad})[1], bad


def _disk(monkeypatch: pytest.MonkeyPatch, *, free_gb: float) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "config.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=64 * 1024**3, free=int(free_gb * 1024**3)),
    )


def test_free_space_limit_cannot_be_raised_above_what_is_free_now(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    _disk(monkeypatch, free_gb=20)
    for fine in (10, 19):
        assert validate_runtime_updates({"MIN_FREE_SPACE_GB": fine})[0] == {
            "MIN_FREE_SPACE_GB": float(fine)
        }
    for too_much in (19.99, 20, 20.5, 30, 100):
        valid, errors = validate_runtime_updates({"MIN_FREE_SPACE_GB": too_much})
        assert "MIN_FREE_SPACE_GB" not in valid
        assert "Only 20 GB are free" in errors["MIN_FREE_SPACE_GB"]


def test_stored_free_space_limit_can_be_kept_or_lowered_on_a_fuller_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    update_runtime_settings({"MIN_FREE_SPACE_GB": 30})
    _disk(monkeypatch, free_gb=5)
    assert validate_runtime_updates({"MIN_FREE_SPACE_GB": "30"})[0] == {
        "MIN_FREE_SPACE_GB": 30.0
    }
    assert validate_runtime_updates({"MIN_FREE_SPACE_GB": 8})[0] == {
        "MIN_FREE_SPACE_GB": 8.0
    }
    assert validate_runtime_updates({"MIN_FREE_SPACE_GB": 31})[1]


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "1e999"])
@pytest.mark.parametrize(
    "key,default", [("ORIGINAL_PHOTO_DAYS", 30), ("BEST_ORIGINALS_PER_SPECIES", 50)]
)
def test_nonfinite_photo_rules_load_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
    default: int,
    value: object,
) -> None:
    import config

    output = _reset_test_config(monkeypatch, tmp_path)
    save_settings_yaml({key: value}, str(output))
    assert config._load_config()[key] == default
    assert (
        getattr(
            settings_for_plan({key: value}),
            "original_days" if key == "ORIGINAL_PHOTO_DAYS" else "best_per_species",
        )
        == default
    )


@pytest.mark.parametrize(
    "value,expected", [("false", False), ("off", False), ("true", True), (None, True)]
)
def test_training_default_keeps_configuration_normalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: object, expected: bool
) -> None:
    from config import get_settings_payload
    from tests.conftest import reset_config_in_place

    output = _reset_test_config(monkeypatch, tmp_path)
    key = "KEEP_ANSWERED_PHOTOS_FOR_TRAINING"
    save_settings_yaml({key: value}, str(output))
    reset_config_in_place()
    with connection.closing_connection():
        pass
    assert get_config()[key] is expected
    assert get_settings_payload()[key]["value"] is expected
    assert settings_for_plan(get_config()).keep_for_training is expected


def test_reserve_validation_leaves_recovery_headroom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from core.storage_pressure import StoragePressure

    output = _reset_test_config(monkeypatch, tmp_path)
    _disk(monkeypatch, free_gb=20)
    budget = 256 * 1024**2
    valid, errors = validate_runtime_updates(
        {"MIN_FREE_SPACE_GB": 19}, capture_budget_bytes=budget
    )
    assert not errors
    guard = StoragePressure(
        str(output), free_bytes=lambda: 20 * 1024**3, settings_provider=lambda: valid
    )
    try:
        assert guard.allow_save(budget)
        assert guard._worker is None
    finally:
        guard.close()
    assert validate_runtime_updates(
        {"MIN_FREE_SPACE_GB": 19.01}, capture_budget_bytes=budget
    )[1]


def test_reserve_increase_requires_known_free_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)

    def unavailable(_path: str) -> None:
        raise OSError("unavailable")

    monkeypatch.setattr("config.shutil.disk_usage", unavailable)
    valid, errors = validate_runtime_updates({"MIN_FREE_SPACE_GB": 10})
    assert not valid and "unavailable" in errors["MIN_FREE_SPACE_GB"]
    assert not validate_runtime_updates({"MIN_FREE_SPACE_GB": 0.25})[1]
