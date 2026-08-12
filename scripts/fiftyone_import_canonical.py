#!/usr/bin/env python3
"""Import a canonical WatchMyBirds bundle into optional FiftyOne QA views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.canonical_dataset import MANIFEST_SCHEMA_VERSION, RULE_VERSION  # noqa: E402

SAVED_VIEWS = {
    "QA — all canonical samples": "qa",
    "Unresolved facts": "unresolved",
    "OD ready": "od_ready",
    "CLS ready": "cls_ready",
    "Hard negatives": "hard_negative",
    "Targeted task selection": "targeted_task",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _safe_extract(archive_path: Path, workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise ValueError("bundle contains an unsafe path") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            payload = archive.read(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"workspace file differs: {member.filename}")
            target.write_bytes(payload)
    return root


def prepare_bundle(source: Path, workspace: Path | None = None) -> Path:
    if source.is_dir():
        return source.resolve()
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("bundle must be a canonical ZIP or extracted directory")
    if workspace is None:
        raise ValueError("--workspace is required when importing a ZIP")
    return _safe_extract(source, workspace)


def _image_sample_id(installation_ids: list[str], filename: str) -> str:
    identity = ",".join(installation_ids) + ":" + filename
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def build_sample_specs(bundle_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported canonical bundle schema")
    if metadata.get("rule_version") != RULE_VERSION:
        raise ValueError("unsupported canonical rule version")

    manifest = _read_jsonl(bundle_dir / "manifest.jsonl")
    facts = _read_jsonl(bundle_dir / "facts.jsonl")
    subjects = _read_jsonl(bundle_dir / "subjects.jsonl")
    classifier = _read_jsonl(bundle_dir / "views" / "classifier_ready.jsonl")
    coco = json.loads((bundle_dir / "views" / "coco.json").read_text(encoding="utf-8"))

    manifest_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    facts_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subjects_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cls_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        manifest_by_file[str(row["image_filename"])].append(row)
    for row in facts:
        facts_by_file[str(row["image_filename"])].append(row)
    for row in subjects:
        subjects_by_file[str(row["image_filename"])].append(row)
    for row in classifier:
        cls_by_file[str(row["image_filename"])].append(row)

    coco_images = {int(row["id"]): row for row in coco.get("images", [])}
    od_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        image = coco_images.get(int(annotation["image_id"]))
        if image:
            od_by_file[Path(str(image["file_name"])).name].append(annotation)

    specs: list[dict[str, Any]] = []
    for filename in sorted(manifest_by_file):
        media_matches = sorted((bundle_dir / "media").glob(f"*/{filename}"))
        if not media_matches:
            continue
        rows = manifest_by_file[filename]
        included_views = {
            str(row["view"])
            for row in rows
            if row.get("decision") == "included"
        }
        tags = ["qa"]
        if "od_positive" in included_views or "od_negative" in included_views:
            tags.append("od_ready")
        if "od_negative" in included_views:
            tags.append("hard_negative")
        if "cls_positive" in included_views:
            tags.append("cls_ready")
        if "unresolved" in included_views:
            tags.extend(("unresolved", "targeted_task"))

        file_facts = facts_by_file.get(filename, [])
        installation_ids = sorted(
            {str(row["installation_id"]) for row in file_facts if row.get("installation_id")}
        )
        app_versions = sorted(
            {str(row["app_version"]) for row in file_facts if row.get("app_version")}
        )
        detector_versions = sorted(
            {
                str(row["detector_model_version"])
                for row in subjects_by_file.get(filename, [])
                if row.get("detector_model_version")
            }
        )
        classifier_versions = sorted(
            {
                str(row["classifier_model_version"])
                for row in subjects_by_file.get(filename, [])
                if row.get("classifier_model_version")
            }
        )
        specs.append(
            {
                "filepath": str(media_matches[0].resolve()),
                "tags": sorted(set(tags)),
                "wmb_sample_id": _image_sample_id(installation_ids, filename),
                "wmb_bundle_id": str(metadata["bundle_id"]),
                "wmb_schema_version": str(metadata["schema_version"]),
                "wmb_rule_version": str(metadata["rule_version"]),
                "wmb_installation_ids": installation_ids,
                "wmb_app_versions": app_versions,
                "wmb_detector_versions": detector_versions,
                "wmb_classifier_versions": classifier_versions,
                "wmb_facts_json": json.dumps(file_facts, sort_keys=True, separators=(",", ":")),
                "wmb_manifest_json": json.dumps(rows, sort_keys=True, separators=(",", ":")),
                "od_detections": od_by_file.get(filename, []),
                "cls_detections": cls_by_file.get(filename, []),
                "hard_negative": "od_negative" in included_views,
            }
        )
    return metadata, specs


def import_into_fiftyone(
    metadata: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    dataset_name: str | None = None,
    fo_module=None,
):
    if fo_module is None:
        try:
            import fiftyone as fo_module
        except ImportError as exc:
            raise RuntimeError(
                "FiftyOne is optional; install requirements-fiftyone.txt on the developer workstation"
            ) from exc

    name = dataset_name or f"watchmybirds-{str(metadata['bundle_id'])[:12]}"
    if fo_module.dataset_exists(name):
        dataset = fo_module.load_dataset(name)
    else:
        dataset = fo_module.Dataset(name, persistent=True)

    samples = []
    for spec in specs:
        sample = fo_module.Sample(filepath=spec["filepath"], tags=spec["tags"])
        for field in (
            "wmb_sample_id",
            "wmb_bundle_id",
            "wmb_schema_version",
            "wmb_rule_version",
            "wmb_installation_ids",
            "wmb_app_versions",
            "wmb_detector_versions",
            "wmb_classifier_versions",
            "wmb_facts_json",
            "wmb_manifest_json",
        ):
            sample[field] = spec[field]
        sample["od_ground_truth"] = fo_module.Detections(
            detections=[
                fo_module.Detection(
                    label="bird",
                    bounding_box=[float(value) for value in annotation["bbox_normalized"]],
                    wmb_detection_id=int(annotation["detection_id"]),
                )
                for annotation in spec["od_detections"]
            ]
        )
        sample["cls_ground_truth"] = fo_module.Detections(
            detections=[
                fo_module.Detection(
                    label=str(item["species_key"]),
                    bounding_box=[float(value) for value in item["bbox_xywh_normalized"]],
                    wmb_detection_id=int(item["detection_id"]),
                )
                for item in spec["cls_detections"]
            ]
        )
        if spec["hard_negative"]:
            sample["image_ground_truth"] = fo_module.Classification(label="no_bird")
        samples.append(sample)

    dataset.merge_samples(
        samples,
        key_field="wmb_sample_id",
        overwrite=True,
        expand_schema=True,
    )
    dataset.info.update(
        {
            "wmb_bundle_id": metadata["bundle_id"],
            "wmb_schema_version": metadata["schema_version"],
            "wmb_rule_version": metadata["rule_version"],
            "wmb_snapshot": metadata.get("snapshot", {}),
        }
    )
    dataset.save()
    for view_name, tag in SAVED_VIEWS.items():
        dataset.save_view(view_name, dataset.match_tags(tag), overwrite=True)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a canonical WatchMyBirds bundle into reproducible FiftyOne QA views."
    )
    parser.add_argument("bundle", type=Path, help="Canonical ZIP or extracted directory")
    parser.add_argument("--workspace", type=Path, help="Persistent extraction directory for ZIP input")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count samples without importing FiftyOne",
    )
    parser.add_argument("--launch", action="store_true", help="Launch the FiftyOne App after import")
    args = parser.parse_args()

    bundle_dir = prepare_bundle(args.bundle, args.workspace)
    metadata, specs = build_sample_specs(bundle_dir)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "bundle_id": metadata["bundle_id"],
                    "rule_version": metadata["rule_version"],
                    "samples": len(specs),
                    "saved_views": list(SAVED_VIEWS),
                },
                sort_keys=True,
            )
        )
        return 0
    dataset = import_into_fiftyone(metadata, specs, dataset_name=args.dataset_name)
    print(f"{dataset.name} {len(specs)} {metadata['bundle_id']}")
    if args.launch:
        import fiftyone as fo

        session = fo.launch_app(dataset)
        session.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
