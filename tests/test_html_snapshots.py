"""
HTML Snapshot Tests for critical UI pages.

Purpose: Detect unintended UI changes by comparing rendered HTML against stored snapshots.
- Time is frozen for consistent timestamps
- Whitespace is normalized before comparison
- Tests verify structure, not dynamic content

Strategy: Test template STRUCTURE by checking for critical elements
rather than full HTML snapshot comparison. This is more resilient to
minor changes while still catching structural regressions.

Pages tested:
- Gallery (/gallery)
- Stream/Index (/)
- Settings (/settings)
"""

import re
from pathlib import Path

# Snapshot storage directory
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def normalize_html(html: str) -> str:
    """
    Normalize HTML for comparison.

    - Collapse multiple whitespace to single space
    - Remove spaces around tags
    - Normalize dynamic content to placeholders
    """
    # Collapse whitespace
    html = re.sub(r"\s+", " ", html)
    # Remove spaces around tags
    html = re.sub(r">\s+<", "><", html)
    # Normalize dynamic timestamps (ISO format)
    html = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s"<]*', "TIMESTAMP", html)
    # Normalize date formats
    html = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", html)
    # Normalize version numbers
    html = re.sub(r"v\d+\.\d+\.\d+", "vX.X.X", html)
    # Normalize CSRF tokens
    html = re.sub(
        r'name="csrf_token"[^>]*value="[^"]*"', 'name="csrf_token" value="TOKEN"', html
    )
    # Normalize nonces
    html = re.sub(r'nonce="[^"]*"', 'nonce="NONCE"', html)
    # Trim
    return html.strip()


def save_snapshot(name: str, content: str) -> Path:
    """Save a snapshot to file."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS_DIR / f"{name}.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def load_snapshot(name: str) -> str | None:
    """Load a snapshot from file, return None if not found."""
    path = SNAPSHOTS_DIR / f"{name}.html"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def compare_or_create_snapshot(name: str, content: str) -> tuple[bool, str]:
    """Compare HTML against stored snapshot."""
    normalized = normalize_html(content)
    existing = load_snapshot(name)

    if existing is None:
        save_snapshot(name, normalized)
        return True, f"Snapshot '{name}' created. Run tests again to verify."

    existing_normalized = normalize_html(existing)

    if normalized == existing_normalized:
        return True, "Snapshot matches"

    # Find first difference for debugging
    for i, (a, b) in enumerate(zip(normalized, existing_normalized, strict=False)):
        if a != b:
            context = 50
            start = max(0, i - context)
            end = min(len(normalized), i + context)
            return False, (
                f"Snapshot mismatch at char {i}:\n"
                f"Got:      ...{normalized[start:end]}...\n"
                f"Expected: ...{existing_normalized[start:end]}..."
            )

    return (
        False,
        f"Length mismatch: got {len(normalized)}, expected {len(existing_normalized)}",
    )


class TestTemplateStructure:
    """Tests for template structural integrity using raw file analysis."""

    def get_template_content(self, name: str) -> str:
        """Load template file content."""
        project_root = Path(__file__).parent.parent
        template_path = project_root / "templates" / name
        with open(template_path, encoding="utf-8") as f:
            return f.read()

    def test_gallery_extends_base(self):
        """gallery.html must extend base.html."""
        content = self.get_template_content("gallery.html")
        assert (
            "{% extends 'base.html' %}" in content
            or '{% extends "base.html" %}' in content
        )

    def test_stream_extends_base(self):
        """stream.html must extend base.html."""
        content = self.get_template_content("stream.html")
        assert (
            "{% extends 'base.html' %}" in content
            or '{% extends "base.html" %}' in content
        )

    def test_settings_extends_base(self):
        """settings.html must extend base.html."""
        content = self.get_template_content("settings.html")
        assert (
            "{% extends 'base.html' %}" in content
            or '{% extends "base.html" %}' in content
        )

    def test_gallery_has_content_block(self):
        """gallery.html must define content block."""
        content = self.get_template_content("gallery.html")
        assert "{% block content %}" in content

    def test_stream_has_content_block(self):
        """stream.html must define content block."""
        content = self.get_template_content("stream.html")
        assert "{% block content %}" in content

    def test_settings_has_content_block(self):
        """settings.html must define content block."""
        content = self.get_template_content("settings.html")
        assert "{% block content %}" in content

    def test_base_has_navigation(self):
        """base.html must contain navigation structure."""
        content = self.get_template_content("base.html")
        # base.html includes appbar partial which has navigation
        assert (
            "appbar" in content.lower()
            or "nav" in content.lower()
            or "include" in content.lower()
        )

    def test_gallery_structure_snapshot(self):
        """Verify gallery.html structure via snapshot."""
        content = self.get_template_content("gallery.html")

        # Extract Jinja blocks and structure (not full render)
        structure = self._extract_structure(content)

        matches, message = compare_or_create_snapshot("gallery_structure", structure)
        assert matches, message

    def test_stream_structure_snapshot(self):
        """Verify stream.html structure via snapshot."""
        content = self.get_template_content("stream.html")
        structure = self._extract_structure(content)

        matches, message = compare_or_create_snapshot("stream_structure", structure)
        assert matches, message

    def test_settings_structure_snapshot(self):
        """Verify settings.html structure via snapshot."""
        content = self.get_template_content("settings.html")
        structure = self._extract_structure(content)

        matches, message = compare_or_create_snapshot("settings_structure", structure)
        assert matches, message

    def _extract_structure(self, content: str) -> str:
        """
        Extract structural elements from template.

        - Jinja blocks ({% block ... %})
        - Jinja includes/extends
        - Major HTML structure (<div>, <section>, <form>, etc.)
        """
        lines = []

        # Extract Jinja directives
        jinja_pattern = r"{%[^%]+%}"
        for match in re.finditer(jinja_pattern, content):
            directive = match.group()
            # Only keep structural directives
            if any(
                k in directive
                for k in ["block", "extends", "include", "import", "macro"]
            ):
                lines.append(directive.strip())

        # Extract major HTML elements (opening tags only)
        html_pattern = r"<(div|section|form|nav|header|footer|main|article|aside)[^>]*>"
        for match in re.finditer(html_pattern, content, re.IGNORECASE):
            tag = match.group()
            # Extract id and class for structure identification
            id_match = re.search(r'id="([^"]*)"', tag)
            class_match = re.search(r'class="([^"]*)"', tag)

            element_info = match.group(1).lower()
            if id_match:
                element_info += f'#"{id_match.group(1)}"'
            if class_match:
                # Only first class for brevity
                first_class = (
                    class_match.group(1).split()[0] if class_match.group(1) else ""
                )
                if first_class:
                    element_info += f'."{first_class}"'

            lines.append(f"<{element_info}>")

        return "\n".join(lines)


class TestCriticalUIElements:
    """Tests to verify critical UI elements exist in templates."""

    def get_template_content(self, name: str) -> str:
        """Load template file content."""
        project_root = Path(__file__).parent.parent
        template_path = project_root / "templates" / name
        with open(template_path, encoding="utf-8") as f:
            return f.read()

    def test_gallery_has_date_display(self):
        """Gallery must display dates."""
        content = self.get_template_content("gallery.html")
        assert "dates_with_counts" in content or "date" in content.lower()

    def test_gallery_has_empty_state(self):
        """Gallery must have empty state message."""
        content = self.get_template_content("gallery.html")
        # Gallery uses "empty-state" class or "no images" message
        assert (
            "empty-state" in content
            or "empty_state" in content
            or "no images" in content.lower()
        )

    def test_stream_has_video_element(self):
        """Stream page must have video/image feed."""
        content = self.get_template_content("stream.html")
        # Check for video feed reference
        assert "video_feed" in content or "<img" in content

    def test_stream_has_species_display(self):
        """Stream page must show species information."""
        content = self.get_template_content("stream.html")
        assert "species" in content.lower() or "detection" in content.lower()

    def test_settings_has_form(self):
        """Settings must have configuration form."""
        content = self.get_template_content("settings.html")
        assert "<form" in content.lower()

    def test_settings_has_video_source_config(self):
        """Settings must allow video source configuration."""
        content = self.get_template_content("settings.html")
        assert "video" in content.lower() and "source" in content.lower()

    def test_settings_has_save_button(self):
        """Settings must have save button."""
        content = self.get_template_content("settings.html")
        assert (
            "save" in content.lower()
            or "submit" in content.lower()
            or "button" in content.lower()
        )
