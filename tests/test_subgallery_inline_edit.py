from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read(relative_path: str) -> str:
    return (_project_root() / relative_path).read_text(encoding="utf-8")


def test_subgallery_uses_inline_edit_toggle_and_filter_context():
    content = _read("templates/subgallery.html")

    assert 'class="btn btn--secondary inline-edit-toggle"' in content
    assert "toggleInlineEdit(this)" in content
    assert "window.INLINE_EDIT_FILTER_CONTEXT = {" in content
    assert "surface: 'gallery'" in content
    assert "{% include 'partials/inline_edit.html' %}" in content


def test_subgallery_targets_observation_covers_for_inline_edit():
    content = _read("templates/subgallery.html")

    assert (
        'data-detection-id="{{ obs.cover_detection.detection_id }}" '
        'data-batch-target="detection"'
    ) in content
    assert 'data-batch-ids="{{ obs.detection_ids | join(\',\') }}"' in content
    assert (
        '<div class="wm-tile fade-in-item" data-detection-id="{{ det.detection_id }}">'
    ) in content
    assert (
        '<div class="obs-filmstrip__item" data-detection-id="{{ det.detection_id }}">'
    ) in content


def test_batch_actions_support_expanded_batch_ids():
    content = _read("assets/js/batch_actions.js")

    assert "var rawBatchIds = cb.dataset.batchIds;" in content
    assert "rawBatchIds.split(',').forEach" in content
    assert "ids = Array.from(new Set(ids));" in content
