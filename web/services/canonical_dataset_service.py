"""Deterministic serialization and explicit transports for canonical bundles."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

from core.canonical_dataset import (
    MANIFEST_SCHEMA_VERSION,
    RULE_VERSION,
    CanonicalDataset,
    build_canonical_dataset,
)

PathResolver = Callable[[str], Path]


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


def render_canonical_bundle(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
) -> bytes:
    """Return byte-stable ZIP content for a fixed snapshot and media set."""
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
    for filename in bundle.media_filenames:
        path = path_resolver(filename)
        entries[_media_archive_path(filename)] = path.read_bytes()

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(entries):
            archive.writestr(_zip_info(name), entries[name], compresslevel=9)
    return output.getvalue()


def write_canonical_bundle(
    bundle: CanonicalDataset,
    *,
    path_resolver: PathResolver,
    destination: Path,
) -> Path:
    """Write the same bytes used by manual download to an explicit target."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        render_canonical_bundle(bundle, path_resolver=path_resolver)
    )
    return destination
