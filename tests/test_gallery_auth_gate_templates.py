"""Static template structure tests for auth-gate guards.

Verifies that moderation controls (relabel, delete, inline-edit)
are wrapped in authentication guards so they are never rendered
for unauthenticated users.
"""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


# ── Helper ──


def _read_template(relative_path: str) -> str:
    """Read a template file and return its content."""
    path = TEMPLATES_DIR / relative_path
    assert path.exists(), f"Template not found: {path}"
    return path.read_text(encoding="utf-8")


# ── Modal component guards ──


class TestModalDetectionInfoGuard:
    """Relabel/Delete buttons in detection info must require can_moderate."""

    def test_macro_signature_has_can_moderate(self):
        content = _read_template("components/modal_detection_info.html")
        assert "can_moderate=false" in content

    def test_relabel_delete_wrapped_in_can_moderate(self):
        content = _read_template("components/modal_detection_info.html")
        # Every relabelDetection and deleteDetection call must be preceded
        # by a {% if can_moderate %} guard somewhere before it.
        assert content.count("{% if can_moderate %}") >= 2, (
            "Expected at least 2 can_moderate guards "
            "(one for multi-detection, one for single-detection)"
        )


class TestModalActionBarGuard:
    """Trash button in action bar must require can_moderate."""

    def test_macro_signature_has_can_moderate(self):
        content = _read_template("components/modal_action_bar.html")
        assert "can_moderate=false" in content

    def test_trash_button_gated_on_can_moderate(self):
        content = _read_template("components/modal_action_bar.html")
        assert "show_trash and can_moderate" in content


class TestDetectionModalPassthrough:
    """detection_modal.html must set can_moderate from session and pass it."""

    def test_can_moderate_set_from_session(self):
        content = _read_template("components/detection_modal.html")
        assert "session.get('authenticated')" in content

    def test_can_moderate_passed_to_detection_info(self):
        content = _read_template("components/detection_modal.html")
        assert "can_moderate=can_moderate" in content

    def test_can_moderate_passed_to_action_bar(self):
        content = _read_template("components/detection_modal.html")
        # Find can_moderate=can_moderate in the render_action_bar call
        assert content.count("can_moderate=can_moderate") >= 2, (
            "can_moderate must be passed to both render_detection_info and render_action_bar"
        )


# ── Inline edit guards ──


class TestSubgalleryInlineEditGuard:
    """Inline edit toggle and partial must be auth-gated in subgallery."""

    def test_edit_button_gated(self):
        content = _read_template("subgallery.html")
        # The inline-edit-toggle must appear after an auth check
        toggle_pos = content.find("inline-edit-toggle")
        auth_before = content.rfind("session.get('authenticated')", 0, toggle_pos)
        assert auth_before != -1, "inline-edit-toggle must be inside an auth guard"

    def test_inline_edit_include_gated(self):
        content = _read_template("subgallery.html")
        include_pos = content.find("inline_edit.html")
        auth_before = content.rfind("session.get('authenticated')", 0, include_pos)
        assert auth_before != -1, (
            "inline_edit.html include must be inside an auth guard"
        )


class TestSpeciesOverviewInlineEditGuard:
    """Inline edit toggle and partial must be auth-gated in species overview."""

    def test_edit_button_gated(self):
        content = _read_template("species_overview.html")
        toggle_pos = content.find("inline-edit-toggle")
        auth_before = content.rfind("session.get('authenticated')", 0, toggle_pos)
        assert auth_before != -1, "inline-edit-toggle must be inside an auth guard"

    def test_inline_edit_include_gated(self):
        content = _read_template("species_overview.html")
        include_pos = content.find("inline_edit.html")
        auth_before = content.rfind("session.get('authenticated')", 0, include_pos)
        assert auth_before != -1, (
            "inline_edit.html include must be inside an auth guard"
        )
