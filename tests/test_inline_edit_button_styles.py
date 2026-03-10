from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_inline_edit_footer_uses_defined_outline_button_variants():
    project_root = _project_root()
    template = (project_root / "templates" / "partials" / "inline_edit.html").read_text(
        encoding="utf-8"
    )
    css = (project_root / "assets" / "design-system.css").read_text(encoding="utf-8")

    assert "btn btn--outline-danger" in template
    assert "btn btn--outline-primary" in template
    assert ".btn--outline-danger" in css
    assert ".btn--outline-primary" in css
