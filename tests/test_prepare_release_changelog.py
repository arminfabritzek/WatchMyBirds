from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.finalize_changelog import finalize_changelog
from scripts.prepare_release_changelog import prepare_release_changelog

ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo: Path, subject: str) -> None:
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        subject,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts/changelog_section.sh", tmp_path / "scripts")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## Unreleased\n\n## 0.5.5 - 2026-09-03\n\n- Old change.\n"
    )
    commit(tmp_path, "feat: previous feature")
    git(tmp_path, "tag", "v0.5.5")
    return tmp_path


def test_generates_direct_and_squashed_changes_and_archives_same_notes(
    repo: Path,
) -> None:
    commit(repo, "chore: finalize release 0.5.5 [skip ci]")
    commit(repo, "fix: quiet stream logging (#131)")
    commit(repo, "fix: keep camera recovery retries alive")
    notes = prepare_release_changelog(repo, "0.5.6")
    assert "quiet stream logging (#131)" in notes
    assert "keep camera recovery retries alive" in notes
    assert "finalize release" not in notes
    assert "previous feature" not in notes
    assert notes in (repo / "CHANGELOG.md").read_text()
    assert prepare_release_changelog(repo, "0.5.6") == notes
    finalize_changelog(repo / "CHANGELOG.md", "0.5.6", "2026-09-10")
    assert prepare_release_changelog(repo, "0.5.6") == notes


@pytest.mark.parametrize("section", ["Unreleased", "0.5.6 - 2026-09-10"])
def test_preserves_curated_notes_without_git_history(repo: Path, section: str) -> None:
    path = repo / "CHANGELOG.md"
    original = f"# Changelog\n\n## {section}\n\n- Carefully written notes.\n"
    path.write_text(original)
    shutil.rmtree(repo / ".git")
    assert prepare_release_changelog(repo, "0.5.6") == "- Carefully written notes."
    assert path.read_text() == original


def test_versioned_notes_take_precedence(repo: Path) -> None:
    path = repo / "CHANGELOG.md"
    path.write_text(
        "## Unreleased\n\n- Future change.\n\n## 0.5.6\n\n- Release change.\n"
    )
    assert prepare_release_changelog(repo, "0.5.6") == "- Release change."


def test_empty_range_fails_without_mutating_changelog(repo: Path) -> None:
    path = repo / "CHANGELOG.md"
    original = path.read_text()
    with pytest.raises(ValueError, match="No release changes"):
        prepare_release_changelog(repo, "0.5.6")
    assert path.read_text() == original


def test_first_release_uses_history_without_tags(repo: Path) -> None:
    git(repo, "tag", "-d", "v0.5.5")
    assert "previous feature" in prepare_release_changelog(repo, "0.5.6")


def test_includes_commits_from_merged_branch(repo: Path) -> None:
    branch = git(repo, "branch", "--show-current")
    git(repo, "checkout", "-qb", "feature")
    commit(repo, "feat: new camera support")
    git(repo, "checkout", "-q", branch)
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "merge",
        "--no-ff",
        "-qm",
        "Merge pull request #1",
        "feature",
    )
    notes = prepare_release_changelog(repo, "0.5.6")
    assert notes.count("new camera support") == 1
    assert "Merge pull request" not in notes


def test_subjects_are_not_executed_or_rendered_as_html(repo: Path) -> None:
    commit(repo, "fix: <script> and `literal` $(touch unexpected)")
    notes = prepare_release_changelog(repo, "0.5.6")
    assert r"\<script\>" in notes
    assert r"\`literal\`" in notes
    assert not (repo / "unexpected").exists()


def test_shallow_history_refused_for_generated_notes(
    repo: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth=1", repo.as_uri(), str(destination)], check=True
    )
    with pytest.raises(ValueError, match="Full Git history"):
        prepare_release_changelog(destination, "0.5.6")
