"""Deterministic serialization and explicit transports for canonical bundles."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

from config import get_config
from core.canonical_dataset import (
    CanonicalDataset,
    build_canonical_dataset,
    read_media_size,
)
from core.human_label_core import get_or_create_labeling_installation_id
from core.training_export import (
    create_export,
    list_exports,
    load_export,
    original_resolver,
    receive_export,
    waiting_revisions,
)

__all__ = [
    "create_export",
    "list_exports",
    "load_export",
    "original_resolver",
    "receive_export",
    "waiting_revisions",
    "build_bundle",
    "get_or_create_labeling_installation_id",
]

PathResolver = Callable[[str], Path]

# Streaming write chunk size for media files copied into the archive.
_COPY_CHUNK_BYTES = 64 * 1024


def build_bundle(
    conn: sqlite3.Connection, *, path_resolver: PathResolver
) -> CanonicalDataset:
    """Build one snapshot using the same media eligibility for every transport."""
    return build_canonical_dataset(
        conn,
        labeling_installation_id=get_or_create_labeling_installation_id(
            str(get_config()["OUTPUT_DIR"])
        ),
        media_exists=lambda filename: path_resolver(filename).is_file(),
        media_size=lambda filename: read_media_size(path_resolver(filename)),
    )


def box_walkthrough_candidate_ids(bundle: CanonicalDataset) -> list[int]:
    """Detections with an answered species but no box verdict yet.

    Derived from the same manifest the page already renders, not a second
    query: a detection qualifies when its species is answered (the
    ``cls_positive`` view includes it) and ``od_positive`` names a missing
    ``bbox_quality`` fact among its blockers — not an unsuitable one, which
    is already answered and must not reappear. A sibling detection on the
    same frame still waiting on its own answer also marks this row
    ``frame_has_unresolved_objects``; that reason does not disqualify the
    row, since answering its own verdict is still the useful action.
    """
    cls_ready_ids = {
        row["detection_id"]
        for row in bundle.manifest
        if row["view"] == "cls_positive" and row["decision"] == "included"
    }
    allowed_reasons = {
        "bbox_quality_unknown",
        "frame_has_unresolved_objects",
        "frame_dimensions_missing",
    }
    return [
        int(row["detection_id"])
        for row in bundle.manifest
        if row["view"] == "od_positive"
        and row["detection_id"] is not None
        and row["decision"] == "excluded"
        and "bbox_quality_unknown" in row["reasons"]
        and set(row["reasons"]) <= allowed_reasons
        and row["detection_id"] in cls_ready_ids
    ]


def needs_box_verdict_count(bundle: CanonicalDataset) -> int:
    """Count of :func:`box_walkthrough_candidate_ids`."""
    return len(box_walkthrough_candidate_ids(bundle))


def missing_original_bird_count(bundle: CanonicalDataset) -> int:
    """Count labeled birds with missing media once across both training views."""
    presence_blockers = {"object_bird_presence_unknown", "object_bird_absent"}
    return len(
        {
            row.get("object_key", row["detection_id"])
            for row in bundle.manifest
            if row["view"] in {"cls_positive", "od_positive"}
            and row["decision"] == "excluded"
            and "media_missing" in row["reasons"]
            and not presence_blockers.intersection(row["reasons"])
        }
    )


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
