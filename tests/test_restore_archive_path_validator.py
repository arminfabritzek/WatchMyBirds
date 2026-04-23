"""Unit tests for the restore-archive path validator.

Guards against CodeQL py/path-injection (#137): the /api/restore/analyze
and /api/restore/apply endpoints accept an ``archive_path`` field from
the authenticated operator. Before this fix the server opened whatever
path the client sent — ``/etc/passwd``, ``../config/secrets.yml``, or
a symlink pointing out of the restore tmp dir.

The validator must only return paths that:
1. Resolve to a descendant of the restore-tmp directory.
2. End in ``.tar.gz`` / ``.tgz``.
3. Are syntactically valid paths (no embedded nulls, etc.).

Anything else returns ``None`` so the caller can 400 the request.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from web.blueprints.backup import _safe_restore_archive_path


class _FakePathManager:
    def __init__(self, tmp_dir: Path):
        self._tmp_dir = tmp_dir

    def get_restore_tmp_dir(self) -> Path:
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        return self._tmp_dir


@pytest.fixture
def restore_sandbox(tmp_path):
    """Creates a restore-tmp dir, an outside file, and a trap symlink."""
    restore_tmp = tmp_path / "restore_tmp"
    restore_tmp.mkdir()

    good = restore_tmp / "backup_2026.tar.gz"
    good.write_bytes(b"\x1f\x8b\x08")

    outside = tmp_path / "outside.tar.gz"
    outside.write_bytes(b"\x1f\x8b\x08")

    escape_link = restore_tmp / "escape.tar.gz"
    os.symlink(outside, escape_link)

    with patch(
        "web.blueprints.backup.path_service.get_path_manager",
        return_value=_FakePathManager(restore_tmp),
    ):
        yield {
            "restore_tmp": restore_tmp,
            "good": good,
            "outside": outside,
            "escape_link": escape_link,
        }


def test_accepts_legit_archive_inside_restore_tmp(restore_sandbox):
    result = _safe_restore_archive_path(str(restore_sandbox["good"]))
    assert result is not None
    assert result == restore_sandbox["good"].resolve()


def test_rejects_empty_string(restore_sandbox):
    assert _safe_restore_archive_path("") is None


def test_rejects_none_safely(restore_sandbox):
    # The caller typically already 400s on falsy input, but the helper
    # must not crash if it ever gets called with None-coerced falsy.
    assert _safe_restore_archive_path(None) is None  # type: ignore[arg-type]


def test_rejects_absolute_path_outside_restore_tmp(restore_sandbox):
    assert _safe_restore_archive_path("/etc/passwd") is None


def test_rejects_absolute_tarball_outside_restore_tmp(restore_sandbox):
    assert _safe_restore_archive_path(str(restore_sandbox["outside"])) is None


def test_rejects_relative_traversal(restore_sandbox):
    # ../.. on a resolved path never lands back inside restore_tmp
    assert _safe_restore_archive_path("../../../etc/passwd") is None


def test_rejects_symlink_that_points_outside(restore_sandbox):
    # The symlink LIVES inside restore_tmp but RESOLVES outside — the
    # validator must follow the resolve() so this case is caught.
    assert _safe_restore_archive_path(str(restore_sandbox["escape_link"])) is None


def test_rejects_wrong_extension(restore_sandbox, tmp_path):
    inside = restore_sandbox["restore_tmp"] / "notes.txt"
    inside.write_text("hi")
    assert _safe_restore_archive_path(str(inside)) is None


def test_accepts_tgz_variant(restore_sandbox):
    tgz = restore_sandbox["restore_tmp"] / "snapshot.tgz"
    tgz.write_bytes(b"\x1f\x8b\x08")
    result = _safe_restore_archive_path(str(tgz))
    assert result is not None


def test_rejects_extension_case_mismatch_is_tolerated(restore_sandbox):
    # Extension check is case-insensitive — a user upload that came
    # back as Foo.TAR.GZ still resolves.
    mixed = restore_sandbox["restore_tmp"] / "Foo.TAR.GZ"
    mixed.write_bytes(b"\x1f\x8b\x08")
    result = _safe_restore_archive_path(str(mixed))
    assert result is not None
