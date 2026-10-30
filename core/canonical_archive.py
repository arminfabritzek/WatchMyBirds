"""Deterministic, bounded-memory canonical ZIP serialization."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

from core.canonical_dataset import CanonicalDataset

PathResolver = Callable[[str], Path]
_COPY_CHUNK_BYTES = 64 * 1024


def _media_archive_path(filename: str) -> str:
    safe_name = Path(filename).name
    date = safe_name[:8]
    shard = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if date.isdigit() else "unknown_date"
    return f"media/{shard}/{safe_name}"


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 9
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _media_zip_info(name: str, source: Path) -> zipfile.ZipInfo:
    info = _zip_info(name)
    info.file_size = source.stat().st_size
    info._compresslevel = 9
    return info


def _metadata_entries(bundle: CanonicalDataset) -> dict[str, object]:
    entries = {
        "bundle.json": {
            "schema_version": bundle.coco["info"]["schema_version"],
            "rule_version": bundle.coco["info"]["rule_version"],
            "bundle_id": bundle.bundle_id,
            "snapshot": bundle.snapshot,
            "counts": bundle.counts,
        },
        "subjects.jsonl": bundle.subjects,
        "facts.jsonl": bundle.facts,
        "objects.jsonl": bundle.objects,
        "manual_objects.jsonl": bundle.manual_objects,
        "manual_object_revisions.jsonl": bundle.manual_revisions,
        "manifest.jsonl": bundle.manifest,
        "views/coco.json": bundle.coco,
        "views/classifier_ready.jsonl": bundle.classifier_ready,
    }
    for view in ("od_positive", "od_negative", "cls_positive", "unresolved"):
        entries[f"views/{view}.jsonl"] = filter(
            lambda row, expected=view: (
                row["view"] == expected and row["decision"] == "included"
            ),
            bundle.manifest,
        )
    return entries


def _metadata_chunks(value: object, *, jsonl: bool) -> Iterator[bytes]:
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"))
    buffer = bytearray()
    for row in value if jsonl else [value]:
        for fragment in encoder.iterencode(row):
            buffer.extend(fragment.encode())
            while len(buffer) >= _COPY_CHUNK_BYTES:
                yield bytes(buffer[:_COPY_CHUNK_BYTES])
                del buffer[:_COPY_CHUNK_BYTES]
        buffer.extend(b"\n")
    if buffer:
        yield bytes(buffer)


class _ZipOutput(io.RawIOBase):
    def __init__(self) -> None:
        self.position = 0
        self.chunks: list[bytes] = []

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def write(self, data: bytes) -> int:
        self.position += len(data)
        if data:
            self.chunks.append(bytes(data))
        return len(data)

    def drain(self) -> Iterator[bytes]:
        chunks, self.chunks = self.chunks, []
        yield from chunks


def stream_canonical_bundle(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
) -> Iterator[bytes]:
    output = _ZipOutput()
    non_media = _metadata_entries(bundle)
    media_names = {_media_archive_path(name): name for name in bundle.media_filenames}
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(set(non_media) | set(media_names)):
            if name in non_media:
                with archive.open(_zip_info(name), "w") as dest:
                    for data in _metadata_chunks(
                        non_media[name], jsonl=name.endswith(".jsonl")
                    ):
                        dest.write(data)
                        yield from output.drain()
                yield from output.drain()
                continue
            source = path_resolver(media_names[name])
            with (
                source.open("rb") as src,
                archive.open(_media_zip_info(name, source), "w") as dest,
            ):
                while data := src.read(_COPY_CHUNK_BYTES):
                    dest.write(data)
                    yield from output.drain()
            yield from output.drain()
    yield from output.drain()


def _write_bundle_zip(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
    destination: Path,
) -> None:
    fd, name = tempfile.mkstemp(
        dir=destination.parent, prefix=".canonical-", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            for chunk in stream_canonical_bundle(bundle, path_resolver=path_resolver):
                output.write(chunk)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
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
