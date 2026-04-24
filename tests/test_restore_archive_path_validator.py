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
    # Result is rebuilt from restore_tmp + secure_filename(basename),
    # so the parent must be the canonical restore_tmp.
    assert result.parent == restore_sandbox["restore_tmp"].resolve()
    assert result.name == "backup_2026.tar.gz"


def test_rejects_empty_string(restore_sandbox):
    assert _safe_restore_archive_path("") is None


def test_rejects_none_safely(restore_sandbox):
    # The caller typically already 400s on falsy input, but the helper
    # must not crash if it ever gets called with None-coerced falsy.
    assert _safe_restore_archive_path(None) is None  # type: ignore[arg-type]


def test_rejects_absolute_non_archive_path(restore_sandbox):
    # /etc/passwd has no .tar.gz / .tgz suffix — basename "passwd"
    # fails the extension guard.
    assert _safe_restore_archive_path("/etc/passwd") is None


def test_remaps_absolute_tarball_into_restore_tmp(restore_sandbox):
    # Even an absolute path pointing outside restore_tmp gets its
    # basename extracted and remapped into restore_tmp. The caller
    # then 404s when the file does not exist there. There is no path
    # by which the validator can return a Path that escapes the sandbox.
    out = restore_sandbox["outside"]
    result = _safe_restore_archive_path(str(out))
    assert result is not None
    assert result.parent == restore_sandbox["restore_tmp"].resolve()
    assert result.name == out.name


def test_rejects_relative_traversal(restore_sandbox):
    # ../../../etc/passwd has basename "passwd" — fails extension guard.
    assert _safe_restore_archive_path("../../../etc/passwd") is None


def test_remaps_traversal_in_archive_name(restore_sandbox):
    # Even ../escape.tar.gz cannot escape — secure_filename strips
    # path components, basename is just "escape.tar.gz", remapped
    # into restore_tmp.
    result = _safe_restore_archive_path("../../escape.tar.gz")
    assert result is not None
    assert result.parent == restore_sandbox["restore_tmp"].resolve()
    assert result.name == "escape.tar.gz"


def test_rejects_wrong_extension(restore_sandbox):
    inside = restore_sandbox["restore_tmp"] / "notes.txt"
    inside.write_text("hi")
    assert _safe_restore_archive_path(str(inside)) is None


def test_accepts_tgz_variant(restore_sandbox):
    tgz = restore_sandbox["restore_tmp"] / "snapshot.tgz"
    tgz.write_bytes(b"\x1f\x8b\x08")
    result = _safe_restore_archive_path(str(tgz))
    assert result is not None


def test_accepts_uppercase_extension(restore_sandbox):
    mixed = restore_sandbox["restore_tmp"] / "Foo.TAR.GZ"
    mixed.write_bytes(b"\x1f\x8b\x08")
    result = _safe_restore_archive_path(str(mixed))
    assert result is not None
