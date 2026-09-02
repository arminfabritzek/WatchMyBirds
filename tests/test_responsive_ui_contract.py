"""Regression checks for shared responsive UI contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN_SYSTEM = (ROOT / "assets" / "design-system.css").read_text(encoding="utf-8")
SETTINGS_TEMPLATE = (ROOT / "templates" / "settings.html").read_text(encoding="utf-8")


def _media_blocks(condition: str) -> list[str]:
    """Return balanced CSS blocks for an exact media-query condition."""
    marker = f"@media {condition}"
    blocks: list[str] = []
    cursor = 0
    while True:
        start = DESIGN_SYSTEM.find(marker, cursor)
        if start < 0:
            return blocks
        opening = DESIGN_SYSTEM.find("{", start)
        depth = 0
        for index in range(opening, len(DESIGN_SYSTEM)):
            char = DESIGN_SYSTEM[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(DESIGN_SYSTEM[opening + 1 : index])
                    cursor = index + 1
                    break
        else:
            raise AssertionError(f"Unclosed CSS media block: {marker}")


def test_authenticated_appbar_collapses_at_laptop_width() -> None:
    blocks = _media_blocks("(max-width: 1100px)")

    assert any(
        ".app-bar__toggle" in block
        and "display: inline-flex" in block
        and ".app-bar__nav" in block
        and "display: none" in block
        for block in blocks
    )


def test_page_controls_wrap_before_dense_content_collides() -> None:
    blocks = _media_blocks("(max-width: 1100px)")

    assert any(
        ".page-control-bar" in block
        and "flex-wrap: wrap" in block
        and ".page-control-bar__center" in block
        and "flex-basis: 100%" in block
        for block in blocks
    )


def test_phone_filters_and_action_rows_can_reflow() -> None:
    blocks = _media_blocks("(max-width: 720px)")

    assert any(
        ".page-control-bar__right" in block and "flex: 1 1 100%" in block
        for block in blocks
    )
    assert any(
        ".page-control-bar .filter-pill" in block
        and "flex-wrap: wrap" in block
        and ".filter-btn-mini" in block
        and "flex: 0 0 36px" in block
        for block in blocks
    )


def test_touch_media_badges_are_not_hover_only() -> None:
    blocks = _media_blocks("(hover: none)")

    assert any(
        ".cover-jump-badge" in block
        and ".source-link-badge" in block
        and "opacity: 1" in block
        for block in blocks
    )


def test_settings_long_values_and_selects_use_responsive_classes() -> None:
    assert 'id="runtimeSourceDisplay" class="settings-status__value"' in (
        SETTINGS_TEMPLATE
    )
    assert 'id="TELEGRAM_MODE" class="form-select settings-field__select"' in (
        SETTINGS_TEMPLATE
    )
    assert "overflow-wrap: anywhere" in SETTINGS_TEMPLATE
