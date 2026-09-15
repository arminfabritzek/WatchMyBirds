"""Large selected downloads stay on disk and close files on failure."""

from __future__ import annotations

import os
import tracemalloc
import zipfile
from types import SimpleNamespace

import pytest

from web.services import (
    archive_service,
    image_download_service,
    user_groundtruth_export_service,
)


@pytest.fixture
def archive_directory(tmp_path, monkeypatch):
    directory = tmp_path / "backup"
    monkeypatch.setattr(
        archive_service.path_service,
        "get_path_manager",
        lambda: SimpleNamespace(backup_dir=directory),
    )
    return directory


def test_selected_image_zip_memory_is_bounded(tmp_path, archive_directory, monkeypatch):
    monkeypatch.setattr(image_download_service.mx, "burn_in_enabled", lambda: False)
    files = []
    for index in range(12):
        image = tmp_path / f"image-{index}.jpg"
        image.write_bytes(os.urandom(1024 * 1024))
        files.append((str(image), image.name, "20260915_120000"))
    tracemalloc.start()
    try:
        with image_download_service.build_zip(files) as archive:
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 3 * 1024 * 1024
            assert archive.fileno() >= 0
            with zipfile.ZipFile(archive) as zipped:
                assert len(zipped.namelist()) == 12
    finally:
        tracemalloc.stop()
    assert not list(archive_directory.iterdir())


@pytest.mark.parametrize(
    "service", [image_download_service, user_groundtruth_export_service]
)
def test_archive_build_failure_closes_file(archive_directory, monkeypatch, service):
    created = []

    def create():
        archive = archive_service.temporary_archive()
        created.append(archive)
        return archive

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service, "temporary_archive", create)
    monkeypatch.setattr(service.zipfile, "ZipFile", fail)
    monkeypatch.setattr(image_download_service.mx, "burn_in_enabled", lambda: False)
    with pytest.raises(OSError, match="disk full"):
        if service is image_download_service:
            service.build_zip([])
        else:
            service.stream_batch_zip(None)
    assert created[0].closed
    assert not list(archive_directory.iterdir())
