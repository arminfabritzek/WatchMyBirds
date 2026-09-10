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

    assert "python3 app_code/scripts/prepare_release_changelog.py" in workflow
    assert "cat release_changes.md" in workflow
    assert 'python3 scripts/finalize_changelog.py "$RELEASED"' in workflow
    assert "git add APP_VERSION CHANGELOG.md" in workflow


def test_release_notes_are_prepared_before_image_build(tmp_path: Path) -> None:
    import os
    import shutil

    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/build-release.yml").read_text()
    )
    steps = workflow["jobs"]["build-release"]["steps"]
    prepare_index = next(
        i for i, step in enumerate(steps) if step.get("id") == "version"
    )
    build_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Build Golden Base Image On Demand"
    )
    assert prepare_index < build_index
    app = tmp_path / "app_code"
    scripts = app / "scripts"
    scripts.mkdir(parents=True)
    for name in (
        "version_info.sh",
        "changelog_section.sh",
        "prepare_release_changelog.py",
    ):
        shutil.copy(REPO_ROOT / "scripts" / name, scripts / name)
    subprocess.run(["git", "init", str(app)], check=True, capture_output=True)
    (app / "APP_VERSION").write_text("0.5.6\n")
    changelog = app / "CHANGELOG.md"
    env = dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "outputs"))
    for notes, expected_success in (("", False), ("- Restore camera recovery.", True)):
        changelog.write_text(_changelog(notes))
        result = subprocess.run(
            ["bash", "-c", steps[prepare_index]["run"]],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
        )
        assert (result.returncode == 0) is expected_success, result.stderr
        if expected_success:
            assert (tmp_path / "release_changes.md").read_text().strip() == notes
        else:
            assert "Release changelog preparation failed" in result.stderr
            assert not (tmp_path / "release_changes.md").exists()

    changelog.write_text(_changelog(""))
    subprocess.run(["git", "-C", str(app), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(app),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fix: restore capture",
        ],
        check=True,
    )
    result = subprocess.run(
        ["bash", "-c", steps[prepare_index]["run"]],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    notes = (tmp_path / "release_changes.md").read_text().strip()
    assert "fix: restore capture" in notes
    assert notes in result.stdout
    finalize_changelog(changelog, "0.5.6", "2026-09-10")
    assert notes in changelog.read_text()
