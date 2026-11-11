from __future__ import annotations

import threading

import pytest

from core.retention_preview import Progress, RetentionPreview


def test_preview_coalesces_requests_and_reuses_completed_result() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = []

    def build(settings: dict, progress: Progress) -> dict:
        calls.append(1)
        progress(dict(images_done=2, images_total=10))
        started.set()
        assert release.wait(timeout=2)
        return dict(due_files=3)

    preview = RetentionPreview(build, lambda: dict(OLD_BIRD_PHOTOS="keep_previews"))
    try:
        assert preview.request()["status"] == "running"
        assert started.wait(timeout=1)
        for _ in range(3):
            state = preview.request(refresh=True)
            assert state["status"] == "running"
            assert state["progress"]["images_done"] == 2
            assert state["result"] is None
        assert calls == [1]
    finally:
        release.set()
        preview.wait(timeout=2)
    completed = preview.request()
    assert completed["result"] == dict(due_files=3)
    completed["result"]["due_files"] = 99
    assert preview.request()["result"]["due_files"] == 3
    assert calls == [1]
    preview.request(refresh=True)
    preview.wait(timeout=2)
    assert calls == [1, 1]


def test_rule_changes_replace_inflight_result_before_publication() -> None:
    started = threading.Event()
    release = threading.Event()
    rules = ["keep_previews"]
    calls = []

    def build(settings: dict, progress: Progress) -> dict:
        rule = settings["OLD_BIRD_PHOTOS"]
        calls.append(rule)
        progress(dict(images_done=1))
        started.set()
        assert release.wait(timeout=2)
        return dict(rule=rule)

    preview = RetentionPreview(build, lambda: dict(OLD_BIRD_PHOTOS=rules[0]))
    try:
        preview.request()
        assert started.wait(timeout=1)
        rules[0] = "delete_all"
        state = preview.request()
        assert state["status"] == "running" and state["result"] is None
    finally:
        release.set()
        preview.wait(timeout=2)
    assert calls == ["keep_previews", "delete_all"]
    assert preview.request()["result"] == dict(rule="delete_all")


def test_preview_failure_requires_explicit_retry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fail = [True]
    calls = []

    def build(settings: dict, progress: Progress) -> dict:
        calls.append(1)
        if fail[0]:
            raise OSError("private diagnostic")
        return dict(due_files=0)

    preview = RetentionPreview(build, lambda: dict(OLD_BIRD_PHOTOS="keep_previews"))
    preview.request()
    preview.wait(timeout=2)
    state = preview.request()
    assert state["status"] == "failed" and state["result"] is None
    assert "private diagnostic" not in state["error"]
    assert "private diagnostic" in caplog.text
    assert calls == [1]
    fail[0] = False
    preview.request(refresh=True)
    preview.wait(timeout=2)
    assert preview.request()["status"] == "complete"
    assert calls == [1, 1]


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("KEEP_ANSWERED_PHOTOS_FOR_TRAINING", (True, False)),
        ("ORIGINAL_PHOTO_DAYS", (30, 90)),
        ("BEST_ORIGINALS_PER_SPECIES", (20, 0)),
    ],
)
def test_every_photo_rule_invalidates_a_completed_preview(
    name: str, values: tuple[object, object]
) -> None:
    current = {name: values[0]}
    seen = []

    def build(settings: dict, progress: Progress) -> dict:
        seen.append(settings[name])
        return {}

    preview = RetentionPreview(build, lambda: dict(current))
    preview.request()
    preview.wait(timeout=2)
    assert preview.request()["status"] == "complete"
    current[name] = values[1]
    assert preview.request()["status"] == "running"
    preview.wait(timeout=2)
    assert seen == list(values)


def test_reserve_change_invalidates_a_completed_preview() -> None:
    current = {"MIN_FREE_SPACE_GB": 0.25}
    seen = []

    def build(settings: dict, progress: Progress) -> dict:
        seen.append(settings["MIN_FREE_SPACE_GB"])
        return {}

    preview = RetentionPreview(build, lambda: dict(current))
    preview.request()
    preview.wait(timeout=2)
    current["MIN_FREE_SPACE_GB"] = 10
    assert preview.request()["status"] == "running"
    preview.wait(timeout=2)
    assert seen == [0.25, 10]
