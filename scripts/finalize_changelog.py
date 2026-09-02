#!/usr/bin/env python3
"""Archive the Unreleased changelog section after a successful release."""

from __future__ import annotations

import argparse
import os
import re
import stat
import tempfile
from datetime import date
from pathlib import Path

VERSION_RE = re.compile(r"v?(\d+\.\d+\.\d+)")
HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def _version_core(raw: str) -> str:
    match = VERSION_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"expected semantic version X.Y.Z, got {raw!r}")
    return match.group(1)


def _release_date(raw: str) -> str:
    try:
        return date.fromisoformat(raw.strip()).isoformat()
    except ValueError as exc:
        raise ValueError(f"expected release date YYYY-MM-DD, got {raw!r}") from exc


def finalize_changelog(path: Path, version: str, released_on: str) -> bool:
    """Move Unreleased notes below a version heading; return whether it changed."""
    version = _version_core(version)
    released_on = _release_date(released_on)
    text = path.read_text(encoding="utf-8")
    headings = list(HEADING_RE.finditer(text))

    if any(
        heading.group(1).removeprefix("v").split(maxsplit=1)[0] == version
        for heading in headings
    ):
        return False

    unreleased = [heading for heading in headings if heading.group(1) == "Unreleased"]
    if len(unreleased) != 1:
        raise ValueError("CHANGELOG must contain exactly one '## Unreleased' heading")

    heading = unreleased[0]
    next_heading = next(
        (item for item in headings if item.start() > heading.start()), None
    )
    section_end = next_heading.start() if next_heading else len(text)
    if not text[heading.end() : section_end].strip():
        raise ValueError("Unreleased changelog section is empty")

    replacement = f"## Unreleased\n\n## {version} - {released_on}"
    updated = text[: heading.start()] + replacement + text[heading.end() :]

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(updated)
        temporary_path = Path(temporary.name)
    try:
        os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary_path, path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("date")
    parser.add_argument("changelog", nargs="?", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args()

    try:
        changed = finalize_changelog(args.changelog, args.version, args.date)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("CHANGELOG finalized." if changed else "CHANGELOG already finalized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
