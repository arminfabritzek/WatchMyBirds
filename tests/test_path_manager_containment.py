"""Path-containment barrier on PathManager.

``contained_path`` is the non-web twin of
``web.view_helpers._contained_static_path``. It guards the derivative
regeneration path, which resolves a filename taken from an unauthenticated
media URL. These tests pin that a resolved path escaping its root is
rejected while a legitimate in-root path passes through.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from utils.path_manager import PathManager


def _pm() -> tuple[PathManager, Path]:
    d = tempfile.mkdtemp()
    pm = PathManager(d)
    pm.originals_dir.mkdir(parents=True, exist_ok=True)
    return pm, pm.originals_dir


def test_legit_in_root_path_passes():
    pm, root = _pm()
    candidate = root / "2026-01-01" / "20260101_120000_bird.jpg"
    assert pm.contained_path(candidate, root) is not None


def test_parent_traversal_is_rejected():
    pm, root = _pm()
    escape = root / ".." / ".." / "etc" / "passwd"
    assert pm.contained_path(escape, root) is None


def test_absolute_outside_path_is_rejected():
    pm, root = _pm()
    assert pm.contained_path(Path("/etc/passwd"), root) is None


def test_root_itself_is_contained():
    pm, root = _pm()
    assert pm.contained_path(root, root) is not None
