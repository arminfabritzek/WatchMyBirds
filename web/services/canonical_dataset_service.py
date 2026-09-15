"""Deterministic serialization and explicit transports for canonical bundles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from core.canonical_dataset import (
    MANIFEST_SCHEMA_VERSION,
    RULE_VERSION,
    CanonicalDataset,
    build_canonical_dataset,
)

PathResolver = Callable[[str], Path]

# Streaming write chunk size for media files copied into the archive.
_COPY_CHUNK_BYTES = 1024 * 1024


def build_bundle(conn, *, path_resolver: PathResolver) -> CanonicalDataset:
    """Build one snapshot using the same media eligibility for every transport."""
    return build_canonical_dataset(
        conn,
        media_exists=lambda filename: path_resolver(filename).is_file(),
    )


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _media_archive_path(filename: str) -> str:
    safe_name = Path(filename).name
    date = safe_name[:8]
    shard = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if date.isdigit() else "unknown_date"
    return f"media/{shard}/{safe_name}"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _media_zip_info(name: str, source: Path) -> zipfile.ZipInfo:
    info = _zip_info(name)
    info.file_size = source.stat().st_size
    info._compresslevel = 9
    return info


def _non_media_entries(bundle: CanonicalDataset) -> dict[str, bytes]:
    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "bundle_id": bundle.bundle_id,
        "snapshot": bundle.snapshot,
        "counts": bundle.counts,
    }
    entries: dict[str, bytes] = {
        "bundle.json": _json_bytes(metadata),
        "subjects.jsonl": _jsonl_bytes(bundle.subjects),
        "facts.jsonl": _jsonl_bytes(bundle.facts),
        "manifest.jsonl": _jsonl_bytes(bundle.manifest),
        "views/coco.json": _json_bytes(bundle.coco),
        "views/classifier_ready.jsonl": _jsonl_bytes(bundle.classifier_ready),
    }
    for view in ("od_positive", "od_negative", "cls_positive", "unresolved"):
        entries[f"views/{view}.jsonl"] = _jsonl_bytes(
            [
                row
                for row in bundle.manifest
                if row["view"] == view and row["decision"] == "included"
            ]
        )
    return entries


def _write_bundle_zip(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
    destination: Path,
) -> None:
    """Write the archive directly to disk without holding all bytes at once.

    Entry order and per-entry metadata (fixed timestamp, compression,
    Unix external attrs) match the previous in-memory implementation, so
    byte-for-byte determinism is preserved. Media files are streamed via
    fixed-size copies into writable ZIP entries.
    """
    non_media = _non_media_entries(bundle)
    media_names = {
        _media_archive_path(filename): filename for filename in bundle.media_filenames
    }
    all_names = sorted(set(non_media) | set(media_names))

    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in all_names:
                if name in non_media:
                    archive.writestr(_zip_info(name), non_media[name], compresslevel=9)
                else:
                    source = path_resolver(media_names[name])
                    with (
                        source.open("rb") as src_file,
                        archive.open(_media_zip_info(name, source), "w") as dest_entry,
                    ):
                        shutil.copyfileobj(
                            src_file, dest_entry, length=_COPY_CHUNK_BYTES
                        )
        tmp_path.replace(destination)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def render_canonical_bundle_to_path(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
    destination: Path,
) -> Path:
    """Write byte-stable ZIP content for a fixed snapshot and media set to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_bundle_zip(bundle, path_resolver=path_resolver, destination=destination)
    return destination


@contextmanager
def render_canonical_bundle_tempfile(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
) -> Iterator[Path]:
    """Render the bundle to a disk-backed temp file, cleaned up on exit.

    For callers (the CLI, tests) that can render and consume the file
    within one `with` block.
    """
    tmp_dir, archive_path = render_canonical_bundle_to_tempdir(
        bundle, path_resolver=path_resolver
    )
    try:
        yield archive_path
    finally:
        archive_path.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def render_canonical_bundle_to_tempdir(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
    temporary_root: Path | None = None,
) -> tuple[Path, Path]:
    """Render the bundle to a fresh temp directory; caller owns cleanup.

    Used by the download route, where the file must outlive this
    function call (it is streamed by the WSGI layer after we return)
    and cleanup has to happen only once the response is fully sent.
    Returns ``(tmp_dir, archive_path)``.
    """
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="canonical-bundle-", dir=temporary_root))
    archive_path = tmp_dir / "bundle.zip"
    try:
        _write_bundle_zip(bundle, path_resolver=path_resolver, destination=archive_path)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir, archive_path


def write_canonical_bundle(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
    destination: Path,
) -> Path:
    """Write the same bytes used by manual download to an explicit target."""
    return render_canonical_bundle_to_path(
        bundle, path_resolver=path_resolver, destination=destination
    )
