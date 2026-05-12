"""CPU-friendliness env-var plumbing for the aesthetic tagger.

The scheduler reads AESTHETIC_TAGGER_NICE and AESTHETIC_TAGGER_TORCH_THREADS
from config and surfaces them as env vars on each run. The worker reads
those env vars in main_with_args() and applies os.nice() +
torch.set_num_threads() before any inference. These tests pin the
plumbing without invoking torch.
"""

from __future__ import annotations

import os

import config as config_module
from web.services import aesthetic_tag_scheduler as ats


def _clear_env():
    for key in ("WMB_AESTHETIC_NICE", "WMB_AESTHETIC_TORCH_THREADS"):
        os.environ.pop(key, None)


def test_apply_cpu_friendliness_exports_env_from_config(monkeypatch):
    _clear_env()

    def fake_config():
        return {
            "AESTHETIC_TAGGER_NICE": 12,
            "AESTHETIC_TAGGER_TORCH_THREADS": 1,
        }

    monkeypatch.setattr(config_module, "get_config", fake_config)
    ats._apply_cpu_friendliness_env()
    assert os.environ["WMB_AESTHETIC_NICE"] == "12"
    assert os.environ["WMB_AESTHETIC_TORCH_THREADS"] == "1"
    _clear_env()


def test_apply_cpu_friendliness_handles_missing_keys(monkeypatch):
    """When the config dict omits the keys (e.g. older settings.yaml),
    no env var is set — the worker then falls back to its own defaults."""
    _clear_env()

    def fake_config():
        return {}

    monkeypatch.setattr(config_module, "get_config", fake_config)
    ats._apply_cpu_friendliness_env()
    assert "WMB_AESTHETIC_NICE" not in os.environ
    assert "WMB_AESTHETIC_TORCH_THREADS" not in os.environ


def test_apply_cpu_friendliness_handles_config_failure(monkeypatch):
    """If get_config() raises (early boot, bad import), the helper must
    not propagate — it's only a best-effort prelude to running the
    worker."""
    _clear_env()

    def boom():
        raise RuntimeError("config not ready")

    monkeypatch.setattr(config_module, "get_config", boom)
    # Should not raise:
    ats._apply_cpu_friendliness_env()
    assert "WMB_AESTHETIC_NICE" not in os.environ


def test_config_defaults_are_conservative():
    """Defaults must not break a fresh install: nice in [0,19], threads in [0, 16]."""
    nice = int(config_module.DEFAULTS["AESTHETIC_TAGGER_NICE"])
    threads = int(config_module.DEFAULTS["AESTHETIC_TAGGER_TORCH_THREADS"])
    assert 0 <= nice <= 19
    assert 0 <= threads <= 16


def test_runtime_keys_include_new_knobs():
    """The Settings UI must be able to flip these live without restart."""
    assert "AESTHETIC_TAGGER_NICE" in config_module.RUNTIME_KEYS
    assert "AESTHETIC_TAGGER_TORCH_THREADS" in config_module.RUNTIME_KEYS
