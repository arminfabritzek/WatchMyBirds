#!/usr/bin/env python3
"""Inspect a finished canonical ZIP without loading the app or changing its data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

SCHEMA = "watchmybirds.canonical-label-bundle.v3"
RULE = "training-readiness-v3"
VIEWS = ("od_positive", "od_negative", "cls_positive", "unresolved")
JSONL_FILES = (
    "subjects.jsonl",
    "facts.jsonl",
    "objects.jsonl",
    "manual_objects.jsonl",
    "manual_object_revisions.jsonl",
    "manifest.jsonl",
    "views/classifier_ready.jsonl",
    *(f"views/{view}.jsonl" for view in VIEWS),
)
METADATA_LIMIT = 64 * 1024 * 1024
MEDIA_LIMIT = 256 * 1024 * 1024


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or any(ord(char) < 32 for char in name)
        or path.as_posix() != name
    )


def valid_box(value: Any, width: float = 1, height: float = 1) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(
        type(number) in (int, float) and math.isfinite(number) for number in value
    ):
        return False
    x, y, w, h = value
    return (
        x >= 0
        and y >= 0
        and w > 0
        and h > 0
        and x + w <= width + 1e-6
        and y + h <= height + 1e-6
    )


def same_box(first: Any, second: Any) -> bool:
    return (
        valid_box(first)
        and valid_box(second)
        and all(
            math.isclose(a, b, abs_tol=1e-6) for a, b in zip(first, second, strict=True)
        )
    )


def object_identity(row: dict) -> tuple[str, str]:
    return str(row.get("image_filename", "")), str(row.get("object_key", ""))


class ExportCheck:
    def __init__(self, archive: zipfile.ZipFile) -> None:
        self.archive = archive
        self.issues: list[dict] = []
        self.metadata: dict = {}
        self.rows: dict[str, list[dict]] = {}
        self.coco: dict = {}
        self.members: dict[str, zipfile.ZipInfo] = {}
        self.media: dict[str, str] = {}
        self.sizes: dict[str, tuple[int, int]] = {}
        self.overlays: dict[str, list[dict]] = defaultdict(list)

    def issue(
        self, code: str, message: str, filename: str = "", severity: str = "error"
    ) -> None:
        self.issues.append(
            dict(code=code, message=message, filename=filename, severity=severity)
        )

    def read_json(self, name: str, *, lines: bool = False) -> Any:
        fallback = [] if lines else {}
        info = self.members.get(name)
        if info is None:
            self.issue("missing_metadata", f"Missing required file: {name}")
            return fallback
        if info.file_size > METADATA_LIMIT:
            self.issue("metadata_limit", f"Metadata exceeds 64 MiB: {name}")
            return fallback
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                text = self.archive.read(info).decode("utf-8")
            result = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if lines
                else json.loads(text)
            )
            canonical(result)
            if lines and (
                not isinstance(result, list)
                or not all(isinstance(row, dict) for row in result)
            ):
                raise ValueError("expected JSON objects on each line")
            if not lines and not isinstance(result, dict):
                raise ValueError("expected JSON object")
            return result
        except (
            ValueError,
            UnicodeError,
            OSError,
            RuntimeError,
            zipfile.BadZipFile,
            NotImplementedError,
            UserWarning,
        ) as exc:
            self.issue("invalid_metadata", f"Cannot read {name}: {exc}")
            return fallback

    def inspect(self) -> None:
        infos = self.archive.infolist()
        if len(infos) > 100_000:
            self.issue(
                "entry_limit", "Archive exceeds 100,000 entries; inspection stopped."
            )
            return
        metadata_bytes = 0
        for info in infos:
            name = info.filename
            if info.is_dir():
                continue
            if not safe_name(name) or stat.S_ISLNK(info.external_attr >> 16):
                self.issue("unsafe_member", f"Unsafe archive entry: {name}")
                continue
            if name in self.members:
                self.issue("duplicate_member", f"Duplicate archive entry: {name}")
                continue
            self.members[name] = info
            if not name.startswith("media/"):
                metadata_bytes += info.file_size
        if metadata_bytes > 128 * 1024 * 1024:
            self.issue(
                "metadata_limit", "Total metadata exceeds 128 MiB; inspection stopped."
            )
            return
        self.metadata = self.read_json("bundle.json")
        for name in JSONL_FILES:
            self.rows[name] = self.read_json(name, lines=True)
        self.coco = self.read_json("views/coco.json")
        for key in ("images", "annotations", "categories"):
            value = self.coco.get(key)
            if not isinstance(value, list) or not all(
                isinstance(row, dict) for row in value
            ):
                self.issue("invalid_coco", f"COCO {key} must be an array of objects.")
                self.coco[key] = []
        if (
            self.metadata.get("schema_version") != SCHEMA
            or self.metadata.get("rule_version") != RULE
        ):
            self.issue(
                "unsupported_version",
                "Only canonical bundle v2 / training-readiness-v3 is supported.",
            )
            return
        if self.coco.get("info") != {"schema_version": SCHEMA, "rule_version": RULE}:
            self.issue(
                "coco_version", "COCO schema/rule versions differ from bundle metadata."
            )
        self.check_media()
        self.check_manifest()
        self.check_training_views()

    def check_media(self) -> None:
        for name, info in self.members.items():
            if not name.startswith("media/"):
                continue
            filename = PurePosixPath(name).name
            if filename in self.media:
                self.issue(
                    "ambiguous_media",
                    "Multiple archive paths use this image filename.",
                    filename,
                )
                continue
            self.media[filename] = name
            if info.file_size > MEDIA_LIMIT:
                self.issue(
                    "media_limit",
                    "Image exceeds the 256 MiB inspection limit.",
                    filename,
                )
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with self.archive.open(info) as stream, Image.open(stream) as image:
                        if image.format not in {"JPEG", "PNG", "WEBP"}:
                            raise ValueError("unsupported image format")
                        if image.width * image.height > 40_000_000:
                            raise ValueError("image exceeds 40 megapixels")
                        image.load()
                        self.sizes[filename] = image.size
                        while stream.read(1024 * 1024):
                            pass
            except (
                OSError,
                ValueError,
                RuntimeError,
                zipfile.BadZipFile,
                NotImplementedError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as exc:
                self.issue(
                    "unreadable_media",
                    f"Image cannot be decoded or fails ZIP integrity: {exc}",
                    filename,
                )

    def check_manifest(self) -> None:
        manifest = self.rows["manifest.jsonl"]
        expected_counts: Counter = Counter()
        identities: set[tuple] = set()
        by_file: dict[str, list[dict]] = defaultdict(list)
        for row in manifest:
            filename = row.get("image_filename")
            view, decision = row.get("view"), row.get("decision")
            reasons = row.get("reasons")
            if (
                not isinstance(filename, str)
                or not filename
                or view not in VIEWS
                or decision not in ("included", "excluded")
                or not isinstance(reasons, list)
                or not reasons
                or not all(isinstance(reason, str) for reason in reasons)
            ):
                self.issue(
                    "invalid_manifest",
                    "Manifest row has invalid filename, view, decision or reasons.",
                )
                continue
            key = (filename, view, str(row.get("object_key", "image")))
            if key in identities:
                self.issue(
                    "duplicate_decision",
                    f"Repeated {view} decision for {key[2]}.",
                    filename,
                )
            identities.add(key)
            by_file[filename].append(row)
            expected_counts[f"{view}_{decision}"] += 1
            if row.get("rule_version") != RULE:
                self.issue(
                    "rule_version",
                    "Manifest row uses a different rule version.",
                    filename,
                )
            if view != "od_negative" and not isinstance(row.get("object_key"), str):
                self.issue(
                    "object_identity", "Object decision has no object_key.", filename
                )
            if (
                decision == "included"
                and view != "unresolved"
                and filename not in self.media
            ):
                self.issue(
                    "missing_training_media",
                    f"{view} includes an image absent from the ZIP.",
                    filename,
                )
        for filename, rows in by_file.items():
            marked_missing = any("media_missing" in row["reasons"] for row in rows)
            if marked_missing != (filename not in self.media):
                self.issue(
                    "media_manifest_mismatch",
                    "Media presence contradicts the manifest's missing-media reasons.",
                    filename,
                )
        for filename in self.media.keys() - by_file.keys():
            self.issue(
                "unreferenced_media", "Image has no manifest decision.", filename
            )
        expected_counts.update(
            subjects=len(self.rows["subjects.jsonl"]),
            facts=len(self.rows["facts.jsonl"]),
        )
        if self.metadata.get("counts") != dict(expected_counts):
            self.issue(
                "count_mismatch",
                "bundle.json counts differ from the actual manifest, subjects or facts.",
            )
        for view in VIEWS:
            expected = Counter(
                canonical(row)
                for row in manifest
                if row.get("view") == view and row.get("decision") == "included"
            )
            actual = Counter(canonical(row) for row in self.rows[f"views/{view}.jsonl"])
            if expected != actual:
                self.issue(
                    "view_mismatch",
                    f"views/{view}.jsonl differs from included manifest decisions.",
                )
        if not manifest:
            self.issue(
                "empty_bundle",
                "The archive has no manifest records; nothing to visually verify.",
                severity="warning",
            )

    def unique_index(self, rows: list[dict], key: str, label: str) -> dict:
        result = {}
        for row in rows:
            value = row.get(key)
            if type(value) is not int or value < 1 or value in result:
                self.issue(
                    "duplicate_or_invalid_id",
                    f"{label} has a missing, invalid or duplicate {key}.",
                )
                continue
            result[value] = row
        return result

    def check_training_views(self) -> None:
        manifest = self.rows["manifest.jsonl"]
        objects = {}
        sample_ids: dict[tuple[str, str], str] = {}
        for row in manifest:
            sample = str(row.get("sample_id", ""))
            if not re.fullmatch(r"[0-9a-f]{32}:[0-9a-f]{24}", sample):
                self.issue(
                    "sample_id",
                    "Sample ID must contain the labeling installation ID and local sample hash.",
                )
            identity = (
                str(row.get("image_filename")),
                str(row.get("object_key", "image")),
            )
            if identity in sample_ids and sample_ids[identity] != sample:
                self.issue("sample_id", "Training views disagree about a sample ID.")
            sample_ids[identity] = sample
        for row in self.rows["objects.jsonl"]:
            key = object_identity(row)
            if not all(key) or key in objects:
                self.issue(
                    "object_identity", "Missing or duplicate object identity.", key[0]
                )
            objects[key] = row
            if row.get("sample_id") != sample_ids.get(key):
                self.issue("sample_id", "Object and manifest sample IDs disagree.")
        expected_od = Counter(
            object_identity(row)
            for row in manifest
            if row.get("view") == "od_positive" and row.get("decision") == "included"
        )
        expected_cls = Counter(
            object_identity(row)
            for row in manifest
            if row.get("view") == "cls_positive" and row.get("decision") == "included"
        )
        negative = {
            str(row.get("image_filename"))
            for row in manifest
            if row.get("view") == "od_negative" and row.get("decision") == "included"
        }
        images = self.unique_index(self.coco["images"], "id", "COCO images")
        annotations = self.unique_index(
            self.coco["annotations"], "id", "COCO annotations"
        )
        categories = self.unique_index(self.coco["categories"], "id", "COCO categories")
        actual_od: Counter = Counter()
        actual_cls: Counter = Counter()
        coco_files: list[str] = []
        for row in images.values():
            path = row.get("file_name")
            filename = PurePosixPath(path).name if isinstance(path, str) else ""
            coco_files.append(filename)
            if not filename or path != self.media.get(filename):
                self.issue(
                    "coco_media_reference",
                    "COCO file_name does not reference an image in this archive.",
                    filename,
                )
            if (
                filename in self.sizes
                and (row.get("width"), row.get("height")) != self.sizes[filename]
            ):
                self.issue(
                    "image_dimensions",
                    "COCO dimensions differ from the decoded image.",
                    filename,
                )
        expected_images = negative | {filename for filename, _ in expected_od}
        if Counter(coco_files) != Counter(expected_images):
            self.issue(
                "coco_image_set",
                "COCO images differ from the positive/negative image set in the manifest.",
            )
        for row in annotations.values():
            image_id = row.get("image_id")
            image = images.get(image_id) if type(image_id) is int else None
            if image is None:
                self.issue(
                    "annotation_image",
                    "COCO annotation references an unknown image_id.",
                )
                continue
            path = image.get("file_name")
            filename = PurePosixPath(path).name if isinstance(path, str) else ""
            key = (filename, str(row.get("object_key", "")))
            actual_od[key] += 1
            category = row.get("category_id")
            if (
                type(category) is not int
                or category not in categories
                or categories[category].get("name") != "bird"
            ):
                self.issue(
                    "annotation_category",
                    "COCO annotation does not reference the bird category.",
                    filename,
                )
            if filename in negative:
                self.issue(
                    "negative_has_boxes",
                    "A no-bird image contains exported positive boxes.",
                    filename,
                )
            size = self.sizes.get(filename)
            bbox = row.get("bbox")
            if not size or not valid_box(bbox, *size):
                self.issue(
                    "invalid_od_box",
                    f"Invalid or out-of-image COCO box for {key[1]}.",
                    filename,
                )
                continue
            normalized = [
                bbox[0] / size[0],
                bbox[1] / size[1],
                bbox[2] / size[0],
                bbox[3] / size[1],
            ]
            if not same_box(normalized, row.get("bbox_normalized")):
                self.issue(
                    "box_units",
                    "Pixel and normalized COCO coordinates disagree.",
                    filename,
                )
            self.check_object(
                row, objects.get(key), normalized, filename, classifier=False
            )
            self.overlays[filename].append(
                dict(kind="od", box=bbox, label=f"Box training · {key[1]}")
            )
        for row in self.rows["views/classifier_ready.jsonl"]:
            key = object_identity(row)
            filename = key[0]
            actual_cls[key] += 1
            if (
                row.get("media_path") != self.media.get(filename)
                or filename not in self.media
            ):
                self.issue(
                    "classifier_media_reference",
                    "Classifier media_path does not reference an image in this archive.",
                    filename,
                )
            bbox = row.get("bbox_xywh_normalized")
            if not valid_box(bbox):
                self.issue(
                    "invalid_cls_box",
                    f"Invalid normalized classifier box for {key[1]}.",
                    filename,
                )
                continue
            if (
                not isinstance(row.get("species_key"), str)
                or not row["species_key"].strip()
            ):
                self.issue(
                    "missing_species",
                    "Classifier example has no species label.",
                    filename,
                )
            self.check_object(row, objects.get(key), bbox, filename, classifier=True)
            size = self.sizes.get(filename)
            if size:
                pixels = [
                    bbox[0] * size[0],
                    bbox[1] * size[1],
                    bbox[2] * size[0],
                    bbox[3] * size[1],
                ]
                self.overlays[filename].append(
                    dict(
                        kind="cls",
                        box=pixels,
                        label=f"{row.get('species_key', '')} · {key[1]}",
                    )
                )
        self.compare_membership(expected_od, actual_od, "od_membership", "COCO")
        self.compare_membership(
            expected_cls, actual_cls, "cls_membership", "Classifier"
        )

    def compare_membership(
        self, expected: Counter, actual: Counter, code: str, label: str
    ) -> None:
        for filename, key in sorted(expected.keys() | actual.keys()):
            identity = (filename, key)
            if expected[identity] != actual[identity]:
                self.issue(
                    code,
                    f"{label} {key}: expected {expected[identity]} example(s), found {actual[identity]}.",
                    filename,
                )

    def check_object(
        self, row: dict, obj: dict | None, box: list, filename: str, *, classifier: bool
    ) -> None:
        if obj is not None and row.get("sample_id") != obj.get("sample_id"):
            self.issue(
                "sample_id",
                "Training example and object sample IDs disagree.",
                filename,
            )
        if obj is None:
            self.issue(
                "missing_object",
                "Training example has no matching objects.jsonl record.",
                filename,
            )
            return
        if not same_box(box, obj.get("bbox")):
            self.issue(
                "object_box",
                "Training box differs from the exported object's box.",
                filename,
            )
        if obj.get("presence") != "present" or obj.get("active") is not True:
            self.issue(
                "object_state",
                "Training example refers to an inactive or absent bird.",
                filename,
            )
        if classifier and row.get("species_key") != obj.get("species_key"):
            self.issue(
                "object_species",
                "Training species differs from the exported human object label.",
                filename,
            )
        for key in ("detection_id", "manual_object_id"):
            if row.get(key) != obj.get(key):
                self.issue(
                    "object_reference",
                    f"Training {key} differs from the object record.",
                    filename,
                )


def write_report(source: Path, destination: Path, *, preview_limit: int = 60) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    preview_dir = destination / "previews"
    preview_dir.mkdir()
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    records = []
    check: ExportCheck | None = None
    try:
        with zipfile.ZipFile(source) as archive:
            check = ExportCheck(archive)
            check.inspect()
            decisions: dict[str, list[dict]] = defaultdict(list)
            for row in check.rows.get("manifest.jsonl", []):
                decisions[str(row.get("image_filename", ""))].append(row)
            problematic = {issue["filename"] for issue in check.issues}
            filenames = sorted(
                set(decisions) | set(check.media),
                key=lambda name: (name not in problematic, name),
            )
            shown = 0
            for index, filename in enumerate(filenames):
                record = dict(
                    anchor=f"image-{index}",
                    filename=filename,
                    preview=None,
                    width=1,
                    height=1,
                    boxes=check.overlays[filename],
                    decisions=decisions[filename],
                    problems=[
                        issue for issue in check.issues if issue["filename"] == filename
                    ],
                )
                if filename in check.sizes:
                    record["width"], record["height"] = check.sizes[filename]
                    if not preview_limit or shown < preview_limit:
                        with (
                            archive.open(check.media[filename]) as stream,
                            Image.open(stream) as image,
                        ):
                            image.thumbnail((1200, 900))
                            image.convert("RGB").save(
                                preview_dir / f"{index}.jpg", quality=85
                            )
                        record["preview"] = f"previews/{index}.jpg"
                        shown += 1
                records.append(record)
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as exc:
        if check is None:
            check = ExportCheck(None)  # type: ignore[arg-type]
        check.issue("unreadable_archive", f"Cannot fully inspect the archive: {exc}")
    assert check is not None
    errors = sum(issue["severity"] == "error" for issue in check.issues)
    result = dict(
        checker_version=1,
        source=source.name,
        sha256=digest.hexdigest(),
        created_at=datetime.now(UTC).isoformat(),
        bundle_id=check.metadata.get("bundle_id"),
        passed=not errors,
        errors=errors,
        warnings=sum(issue["severity"] == "warning" for issue in check.issues),
        summary=dict(
            images=len(check.media),
            decoded_images=len(check.sizes),
            manifest_images=len(records),
            box_examples=len(check.coco.get("annotations", [])),
            species_examples=len(check.rows.get("views/classifier_ready.jsonl", [])),
            previews=sum(bool(row["preview"]) for row in records),
        ),
        issues=check.issues,
        scope="ZIP consistency only; no live database comparison or biological label verification.",
    )
    (destination / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(["html"]),
    )
    html = environment.get_template("training_export_report.html").render(
        report=result,
        records=records,
        anchors={record["filename"]: record["anchor"] for record in records},
    )
    (destination / "index.html").write_text(html, encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Finished canonical export ZIP")
    parser.add_argument(
        "--report", type=Path, help="New output directory (never overwritten)"
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=60,
        help="Maximum previews; 0 = all. All media are checked regardless (default: 60).",
    )
    args = parser.parse_args(argv)
    if args.preview_limit < 0:
        parser.error("--preview-limit must be non-negative")
    if not args.archive.is_file():
        parser.error("archive does not exist or is not a file")
    destination = args.report or args.archive.with_name(
        args.archive.stem + "-check-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    try:
        result = write_report(
            args.archive, destination, preview_limit=args.preview_limit
        )
    except OSError as exc:
        print(f"Cannot create report: {exc}", file=sys.stderr)
        return 2
    print(
        f"{'PASS' if result['passed'] else 'FAIL'}: {result['errors']} errors, {result['warnings']} warnings"
    )
    print(f"HTML: {(destination / 'index.html').resolve()}")
    print(f"JSON: {(destination / 'report.json').resolve()}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
