from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.finalize_changelog import finalize_changelog

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "finalize_changelog.py"


def _changelog(unreleased: str) -> str:
    return f"""\
# Changelog

## Unreleased

{unreleased}

## 0.5.4 - 2026-09-01

### Fixed

- Previous fix.
"""


def test_finalizes_unreleased_notes_under_release_version(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_changelog("### Added\n\n- New feature."))

    assert finalize_changelog(changelog, "v0.5.5", "2026-09-02") is True

    updated = changelog.read_text()
    assert "## Unreleased\n\n## 0.5.5 - 2026-09-02" in updated
    assert updated.index("- New feature.") < updated.index("## 0.5.4")


def test_existing_release_section_is_idempotent(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    original = _changelog("")
    changelog.write_text(original)

    assert finalize_changelog(changelog, "0.5.4", "2026-09-01") is False
    assert changelog.read_text() == original


def test_refuses_to_finalize_empty_unreleased_section(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_changelog(""))

    result = subprocess.run(
        ["python3", str(SCRIPT), "0.5.5", "2026-09-02", str(changelog)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Unreleased changelog section is empty" in result.stderr


def test_rejects_invalid_release_metadata(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(_changelog("### Fixed\n\n- New fix."))

    result = subprocess.run(
        ["python3", str(SCRIPT), "next", "today", str(changelog)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "expected semantic version X.Y.Z" in result.stderr


def test_release_workflow_uses_and_archives_unreleased_notes() -> None:
    workflow = (REPO_ROOT / ".github/workflows/build-release.yml").read_text()

    assert "using Unreleased notes" in workflow
    assert "Unreleased app_code/CHANGELOG.md" in workflow
    assert 'python3 scripts/finalize_changelog.py "$RELEASED"' in workflow
    assert "git add APP_VERSION CHANGELOG.md" in workflow
