#!/usr/bin/env python3
"""Prepare release notes, filling an empty Unreleased section from Git history."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, encoding="utf-8"
    ).strip()


def _section(repo: Path, version: str) -> str:
    return subprocess.check_output(
        [
            "bash",
            str(repo / "scripts/changelog_section.sh"),
            version,
            str(repo / "CHANGELOG.md"),
        ],
        text=True,
        encoding="utf-8",
    ).strip()


def prepare_release_changelog(repo: Path, version: str) -> str:
    """Preserve curated notes; persist generated notes for later archival."""
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", version):
        raise ValueError("Expected release version X.Y.Z")
    for section in (version, "Unreleased"):
        notes = _section(repo, section)
        if notes:
            return notes

    if _git(repo, "rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("Full Git history is required to generate release notes")
    tags = _git(repo, "tag", "--merged", "HEAD", "--sort=-version:refname").splitlines()
    base = next((tag for tag in tags if re.fullmatch(r"v?\d+\.\d+\.\d+", tag)), None)
    revision = f"{base}..HEAD" if base else "HEAD"
    commits = _git(
        repo, "log", "--reverse", "--no-merges", "--format=%h%x09%s", revision
    )
    entries = []
    for line in commits.splitlines():
        sha, subject = line.split("\t", 1)
        if re.fullmatch(
            r"chore: finalize release \d+\.\d+\.\d+(?: \[skip ci\])?", subject
        ):
            continue
        # Commit subjects are data, including Markdown/HTML metacharacters.
        subject = re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", subject)
        entries.append(f"- {subject} ({sha})")
    if not entries:
        raise ValueError("No release changes found to generate changelog entries")
    notes = "### Changes\n\n" + "\n".join(entries)
    path = repo / "CHANGELOG.md"
    original = path.read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## (.+)$", original, re.MULTILINE))
    unreleased = [heading for heading in headings if heading.group(1) == "Unreleased"]
    if len(unreleased) != 1:
        raise ValueError("CHANGELOG must contain exactly one '## Unreleased' heading")
    start = unreleased[0].end()
    end = next(
        (heading.start() for heading in headings if heading.start() > start),
        len(original),
    )
    path.write_text(
        original[:start] + "\n\n" + notes + "\n\n" + original[end:], encoding="utf-8"
    )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        notes = prepare_release_changelog(args.repo.resolve(), args.version)
        args.output.write_text(notes + "\n", encoding="utf-8")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Release changelog preparation failed: {exc}\n")
    print("Release changelog prepared:")
    print(notes)


if __name__ == "__main__":
    main()
