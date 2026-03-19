from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_trash_template_supports_date_range_preview_and_confirm():
    content = (_project_root() / "templates" / "trash.html").read_text(
        encoding="utf-8"
    )

    assert 'id="trash-range-from"' in content
    assert 'id="trash-range-to"' in content
    assert 'id="trash-range-preview-btn"' in content
    assert 'id="trash-range-confirm-btn"' in content
    assert "/assets/js/batch_actions.js" in content
    assert "mode: 'date_range'" in content
    assert "Date range changed. Preview again" in content
