"""Retention settings wired into the runtime config system."""

import importlib

import pytest


@pytest.fixture
def fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    importlib.reload(config)
    config._CONFIG = None
    return config


def test_defaults_present_and_conservative(fresh_config):
    cfg = fresh_config.get_config()
    # Feature ships OFF; protections ON; sane default window.
    assert cfg["RETENTION_ENABLED"] is False
    assert cfg["RETENTION_DAYS"] == 90
    assert cfg["RETENTION_PROTECT_FAVORITES"] is True
    assert cfg["RETENTION_PROTECT_UNREVIEWED"] is True


def test_retention_keys_are_runtime_editable(fresh_config):
    for key in (
        "RETENTION_ENABLED",
        "RETENTION_DAYS",
        "RETENTION_PROTECT_FAVORITES",
        "RETENTION_PROTECT_UNREVIEWED",
    ):
        assert key in fresh_config.RUNTIME_KEYS


def test_retention_days_validation_accepts_in_range(fresh_config):
    ok, coerced = fresh_config._validate_value("RETENTION_DAYS", "120")
    assert ok is True
    assert coerced == 120


def test_retention_days_validation_rejects_zero_and_negative(fresh_config):
    assert fresh_config._validate_value("RETENTION_DAYS", 0)[0] is False
    assert fresh_config._validate_value("RETENTION_DAYS", -5)[0] is False


def test_retention_days_validation_rejects_above_max(fresh_config):
    assert fresh_config._validate_value("RETENTION_DAYS", 4000)[0] is False


def test_retention_days_validation_rejects_nonnumeric(fresh_config):
    assert fresh_config._validate_value("RETENTION_DAYS", "soon")[0] is False


def test_retention_bool_validation_coerces(fresh_config):
    ok, coerced = fresh_config._validate_value("RETENTION_ENABLED", "true")
    assert ok is True
    assert coerced is True
    ok, coerced = fresh_config._validate_value("RETENTION_PROTECT_FAVORITES", "false")
    assert ok is True
    assert coerced is False


def test_settings_round_trip_through_yaml(fresh_config):
    fresh_config.update_runtime_settings(
        {"RETENTION_ENABLED": True, "RETENTION_DAYS": 45}
    )
    yaml_settings = fresh_config.load_settings_yaml(
        str(fresh_config.get_config()["OUTPUT_DIR"])
    )
    assert yaml_settings["RETENTION_ENABLED"] is True
    assert yaml_settings["RETENTION_DAYS"] == 45
    # In-memory config reflects the change too.
    assert fresh_config.get_config()["RETENTION_DAYS"] == 45


# --- Posture (V2) ---------------------------------------------------------


def test_posture_default_is_conservative(fresh_config):
    assert fresh_config.get_config()["RETENTION_POSTURE"] == "conservative"


def test_posture_is_runtime_editable(fresh_config):
    assert "RETENTION_POSTURE" in fresh_config.RUNTIME_KEYS


def test_posture_validation_accepts_known_values(fresh_config):
    for val in ("off", "conservative", "reclaim"):
        ok, coerced = fresh_config._validate_value("RETENTION_POSTURE", val)
        assert ok is True
        assert coerced == val


def test_posture_validation_normalizes_case_and_whitespace(fresh_config):
    ok, coerced = fresh_config._validate_value("RETENTION_POSTURE", "  Reclaim ")
    assert ok is True
    assert coerced == "reclaim"


def test_posture_validation_rejects_unknown(fresh_config):
    assert fresh_config._validate_value("RETENTION_POSTURE", "aggressive")[0] is False
