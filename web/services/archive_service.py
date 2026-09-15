"""Temporary archive files on the configured output volume."""

from __future__ import annotations

import tempfile
from typing import BinaryIO

from web.services import path_service


def temporary_archive() -> BinaryIO:
    """Return a seekable file that is removed on close, including after download."""
    directory = path_service.get_path_manager().backup_dir
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryFile(mode="w+b", dir=directory, prefix="download-")
