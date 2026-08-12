"""Optional FiftyOne workflow contracts without requiring FiftyOne at runtime."""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import pytest

from core.canonical_dataset import MANIFEST_SCHEMA_VERSION, RULE_VERSION
from scripts.fiftyone_import_canonical import (
    SAVED_VIEWS,
    build_sample_specs,
    import_into_fiftyone,
    prepare_bundle,
)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.fixture
def extracted_bundle(tmp_path):
    root = tmp_path / "bundle"
    media = root / "media" / "2026-08-10"
    media.mkdir(parents=True)
    filename = "20260810_120000_bird.jpg"
    (media / filename).write_bytes(b"bird-image")
    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rule_version": RULE_VERSION,
        "bundle_id": "a" * 64,
        "snapshot": {"max_fact_id": 3},
    }
    (root / "bundle.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = [
        {
            "sample_id": "object-one",
            "view": view,
            "decision": "included",
            "reasons": [reason],
            "image_filename": filename,
            "detection_id": 7,
            "rule_version": RULE_VERSION,
        }
        for view, reason in (
            ("od_positive", "explicit_object_bird_and_suitable_bbox"),
            ("cls_positive", "explicit_object_bird_and_species"),
            ("unresolved", "frame_has_unresolved_objects"),
        )
    ]
    _write_jsonl(root / "manifest.jsonl", manifest)
    _write_jsonl(
        root / "facts.jsonl",
        [
            {
                "fact_id": 1,
                "image_filename": filename,
                "detection_id": 7,
                "fact_type": "bird_presence",
                "answer_value": "present",
                "installation_id": "station-anonymous",
                "app_version": "0.6.0",
            }
        ],
    )
    _write_jsonl(
        root / "subjects.jsonl",
        [
            {
                "subject_id": 1,
                "image_filename": filename,
                "detection_id": 7,
                "detector_model_version": "det-v1",
                "classifier_model_version": "cls-v2",
            }
        ],
    )
    _write_jsonl(
        root / "views" / "classifier_ready.jsonl",
        [
            {
                "image_filename": filename,
                "detection_id": 7,
                "species_key": "Parus_major",
                "bbox_xywh_normalized": [0.1, 0.2, 0.3, 0.4],
            }
        ],
    )
    (root / "views" / "coco.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": f"media/2026-08-10/{filename}"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "detection_id": 7,
                        "bbox_normalized": [0.1, 0.2, 0.3, 0.4],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root


class _FakeSample(dict):
    def __init__(self, *, filepath, tags):
        super().__init__()
        self.filepath = filepath
        self.tags = tags


class _FakeLabel:
    def __init__(self, **kwargs):
        self.values = kwargs


class _FakeDetections:
    def __init__(self, *, detections):
        self.detections = detections


class _FakeDataset:
    registry = {}

    def __init__(self, name, persistent=False):
        self.name = name
        self.persistent = persistent
        self.samples = {}
        self.info = {}
        self.saved_views = {}
        self.registry[name] = self

    def merge_samples(self, samples, *, key_field, **_kwargs):
        for sample in samples:
            self.samples[sample[key_field]] = sample

    def save(self):
        return None

    def match_tags(self, tag):
        return ("tag", tag)

    def save_view(self, name, view, *, overwrite=False):
        assert overwrite is True
        self.saved_views[name] = view


def _fake_fiftyone():
    _FakeDataset.registry = {}
    return SimpleNamespace(
        Dataset=_FakeDataset,
        Sample=_FakeSample,
        Detection=_FakeLabel,
        Detections=_FakeDetections,
        Classification=_FakeLabel,
        dataset_exists=lambda name: name in _FakeDataset.registry,
        load_dataset=lambda name: _FakeDataset.registry[name],
    )


def test_sample_specs_preserve_stable_ids_provenance_and_view_tags(
    extracted_bundle,
) -> None:
    metadata, first = build_sample_specs(extracted_bundle)
    _, second = build_sample_specs(extracted_bundle)

    assert metadata["bundle_id"] == "a" * 64
    assert first == second
    assert first[0]["wmb_sample_id"] == second[0]["wmb_sample_id"]
    assert first[0]["wmb_installation_ids"] == ["station-anonymous"]
    assert {"qa", "od_ready", "cls_ready", "unresolved", "targeted_task"} <= set(
        first[0]["tags"]
    )


def test_reimport_merges_by_stable_id_and_saves_all_qa_views(extracted_bundle) -> None:
    metadata, specs = build_sample_specs(extracted_bundle)
    fake = _fake_fiftyone()

    first = import_into_fiftyone(metadata, specs, fo_module=fake)
    second = import_into_fiftyone(metadata, specs, fo_module=fake)

    assert first is second
    assert len(second.samples) == 1
    assert set(second.saved_views) == set(SAVED_VIEWS)
    assert second.info["wmb_bundle_id"] == "a" * 64


def test_zip_import_requires_workspace_and_rejects_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "no")

    with pytest.raises(ValueError, match="workspace"):
        prepare_bundle(archive)
    with pytest.raises(ValueError, match="unsafe path"):
        prepare_bundle(archive, tmp_path / "workspace")
