from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_INFO = REPO_ROOT / "scripts" / "version_info.sh"


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _version(repo: Path, command: str, *args: str) -> str:
    completed = subprocess.run(
        ["bash", str(VERSION_INFO), command, str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repo(tmp_path: Path, app_version: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "APP_VERSION").write_text(f"{app_version}\n")
    (repo / "README.md").write_text("fixture\n")
    _git(repo, "add", "APP_VERSION", "README.md")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_current_release_prefers_latest_semver_tag(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "0.1.5")
    _git(repo, "tag", "v0.1.5")

    assert _version(repo, "current-release") == "0.1.5"
    assert _version(repo, "next-release") == "0.1.6"


def test_next_release_honors_prebumped_app_version(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "0.1.6")
    _git(repo, "tag", "v0.1.5")

    assert _version(repo, "current-release") == "0.1.5"
    assert _version(repo, "next-release") == "0.1.6"


def test_dev_version_uses_next_release_version(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "0.1.5")
    _git(repo, "tag", "v0.1.5")

    dev_version = _version(repo, "dev-version", "abcdef123456")

    assert dev_version == "0.1.6-dev.abcdef1"


def test_dev_suffix_app_version_resolves_current_and_next_release(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "0.1.7-dev.deadbee")
    _git(repo, "tag", "v0.1.6")

    assert _version(repo, "current-release") == "0.1.6"
    assert _version(repo, "next-release") == "0.1.7"
