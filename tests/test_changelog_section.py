from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "changelog_section.sh"

SAMPLE = """\
# Changelog

## Unreleased

## 0.5.0 - 2026-06-23

### Added

- New thing.
- Another thing.

## 0.2.0 - 2026-04-20

Older release body.

## 0.1.1 - 2026-04-04

Oldest release body.
"""


def _run(changelog: Path, version: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), version, str(changelog)],
        capture_output=True,
        text=True,
    )


def test_extracts_matching_section(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE)
    result = _run(changelog, "0.5.0")
    assert result.returncode == 0
    assert "New thing." in result.stdout
    assert "Another thing." in result.stdout
    # Stops at the next heading — must not leak the older section.
    assert "Older release body." not in result.stdout
    assert "## 0.2.0" not in result.stdout


def test_strips_v_prefix(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE)
    result = _run(changelog, "v0.5.0")
    assert result.returncode == 0
    assert "New thing." in result.stdout


def test_unknown_version_is_empty_and_succeeds(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE)
    result = _run(changelog, "9.9.9")
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_unreleased_section_is_empty(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE)
    result = _run(changelog, "Unreleased")
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_missing_file_errors(tmp_path: Path) -> None:
    result = _run(tmp_path / "nope.md", "0.5.0")
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_no_trailing_blank_lines(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(SAMPLE)
    result = _run(changelog, "0.2.0")
    assert result.returncode == 0
    assert result.stdout == "Older release body.\n"
