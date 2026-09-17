"""Durable job snapshots and storage-path validation."""

import pytest

from core import nightly_job_state
from utils.path_manager import PathManager


@pytest.fixture(autouse=True)
def output_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        nightly_job_state, "get_config", lambda: {"OUTPUT_DIR": str(tmp_path)}
    )


def test_snapshots_are_independent_and_replaced(tmp_path):
    nightly_job_state.save_status("sharpness", {"last_result": "running"})
    nightly_job_state.save_status("aesthetic_tagger", {"last_result": "failed"})
    nightly_job_state.save_status("sharpness", {"last_result": "succeeded"})
    assert nightly_job_state.load_status("sharpness") == {"last_result": "succeeded"}
    assert nightly_job_state.load_status("aesthetic_tagger") == {
        "last_result": "failed"
    }
    assert len(list((tmp_path / "nightly_jobs").iterdir())) == 2


@pytest.mark.parametrize("content", ["{broken", "null", "[]"])
def test_damaged_snapshot_is_ignored(tmp_path, content):
    path = PathManager(str(tmp_path)).get_nightly_job_status_path("sharpness")
    path.parent.mkdir()
    path.write_text(content)
    assert nightly_job_state.load_status("sharpness") == {}


def test_missing_snapshot_is_empty():
    assert nightly_job_state.load_status("sharpness") == {}


@pytest.mark.parametrize("name", ["../escape", "", "/tmp/job", "a/b"])
def test_job_path_rejects_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        PathManager(str(tmp_path)).get_nightly_job_status_path(name)


def test_failed_replace_preserves_previous_snapshot(monkeypatch, tmp_path):
    nightly_job_state.save_status("sharpness", {"last_result": "succeeded"})

    def fail_replace(*args):
        raise OSError("Disk unavailable")

    monkeypatch.setattr(nightly_job_state.os, "replace", fail_replace)
    with pytest.raises(OSError):
        nightly_job_state.save_status("sharpness", {"last_result": "running"})
    assert nightly_job_state.load_status("sharpness") == {"last_result": "succeeded"}
    assert len(list((tmp_path / "nightly_jobs").iterdir())) == 1
