"""The shared editor explains retired originals before offering edits."""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader


@pytest.mark.parametrize("original_present", [0, 1, None])
def test_shared_editor_notice_is_visible_only_for_retired_originals(
    original_present: int | None,
) -> None:
    root = Path(__file__).resolve().parents[1]
    env = Environment(loader=FileSystemLoader(root / "templates"), autoescape=True)
    template = env.get_template("components/bird_editor_toolbar.html")
    detection = {
        "detection_id": 1,
        "image_filename": "20260922_120000.jpg",
        "common_name": "Robin",
        "bbox_x": 0.1,
        "bbox_y": 0.2,
        "bbox_w": 0.3,
        "bbox_h": 0.4,
    }
    if original_present is not None:
        detection["original_present"] = original_present
    html = template.module.render_bird_editor_toolbar(detection, can_moderate=True)
    if original_present == 0:
        assert "Not available for training export" in html
        assert "but cannot currently be exported as training data" in html
        assert html.index('role="note"') < html.index('role="toolbar"')
        assert "hidden" not in html[html.index('role="note"') : html.index("</div>")]
    else:
        assert 'role="note"' not in html
    assert 'data-editor-action="adjust"' in html
    assert 'data-editor-action="save"' in html
