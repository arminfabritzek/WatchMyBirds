"""Exercise the offline checker against real exports and damaged ZIP files."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PIL import Image

from core.human_label_core import HumanAnswer, record_human_answer
from scripts.check_training_export import main, write_report
from tests import test_canonical_dataset_bundle as bundle_tests
from web.services.canonical_dataset_service import build_bundle, write_canonical_bundle


@pytest.fixture
def exported_zip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    fixture = bundle_tests.canonical_case.__wrapped__(monkeypatch, tmp_path)
    conn, manager, provenance, _, _, cls_ids = next(fixture)
    try:
        for path in manager.originals_dir.rglob("*.jpg"):
            Image.new("RGB", (1000, 800), "#92a18c").save(path)
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename="20260810_090200_cls.jpg",
                detection_id=cls_ids[0],
                bbox_quality="suitable",
            ),
            provenance,
        )
        conn.commit()
        bundle = build_bundle(conn, path_resolver=manager.get_original_path)
        path = tmp_path / "export.zip"
        write_canonical_bundle(
            bundle, path_resolver=manager.get_original_path, destination=path
        )
        yield path
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


def mutate(source: Path, name: str, change: Callable[[object], object]) -> Path:
    target = source.with_name("changed.zip")
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as modified:
        for entry in original.infolist():
            payload = original.read(entry)
            if entry.filename == name:
                if name.endswith(".jsonl"):
                    value = [json.loads(line) for line in payload.splitlines()]
                    payload = "".join(
                        json.dumps(row) + "\n" for row in change(value)
                    ).encode()
                else:
                    payload = json.dumps(change(json.loads(payload))).encode()
            modified.writestr(entry, payload)
    return target


def test_real_export_passes_and_source_is_unchanged(
    exported_zip: Path, tmp_path: Path
) -> None:
    before = hashlib.sha256(exported_zip.read_bytes()).hexdigest()
    result = write_report(exported_zip, tmp_path / "report")
    assert result["passed"], result["issues"]
    assert result["summary"]["images"] == 3
    assert result["summary"]["box_examples"] == 1
    assert result["summary"]["species_examples"] == 1
    assert (
        result["sha256"]
        == before
        == hashlib.sha256(exported_zip.read_bytes()).hexdigest()
    )
    html = (tmp_path / "report/index.html").read_text()
    assert "Cyanistes_caeruleus" in html
    assert '<rect class="od"' in html
    assert '<rect class="cls"' in html
    assert "script-src 'none'" in html
    assert not list((tmp_path / "report").rglob("*.zip"))
    assert len(list((tmp_path / "report/previews").glob("*.jpg"))) == 3


@pytest.mark.parametrize(
    ("filename", "change", "code"),
    [
        ("views/coco.json", lambda doc: {**doc, "annotations": []}, "od_membership"),
        ("views/classifier_ready.jsonl", lambda rows: [], "cls_membership"),
        (
            "views/classifier_ready.jsonl",
            lambda rows: [{**rows[0], "species_key": "Wrong_species"}],
            "object_species",
        ),
        (
            "views/classifier_ready.jsonl",
            lambda rows: [{**rows[0], "bbox_xywh_normalized": [0.9, 0.2, 0.5, 0.3]}],
            "invalid_cls_box",
        ),
        (
            "views/coco.json",
            lambda doc: {
                **doc,
                "annotations": [{**doc["annotations"][0], "bbox": [0, 0, 3000, 10]}],
            },
            "invalid_od_box",
        ),
        (
            "views/coco.json",
            lambda doc: {
                **doc,
                "images": [{**row, "width": 5} for row in doc["images"]],
            },
            "image_dimensions",
        ),
        (
            "views/coco.json",
            lambda doc: {
                **doc,
                "annotations": [{**doc["annotations"][0], "image_id": 9000}],
            },
            "annotation_image",
        ),
        ("bundle.json", lambda doc: {**doc, "counts": {}}, "count_mismatch"),
        ("views/cls_positive.jsonl", lambda rows: [], "view_mismatch"),
        ("views/classifier_ready.jsonl", lambda rows: rows + rows, "cls_membership"),
        ("views/coco.json", lambda doc: {**doc, "images": "broken"}, "invalid_coco"),
    ],
)
def test_inconsistencies_fail(
    exported_zip: Path, tmp_path: Path, filename: str, change: Callable, code: str
) -> None:
    altered = mutate(exported_zip, filename, change)
    result = write_report(altered, tmp_path / "report")
    assert not result["passed"]
    assert code in {issue["code"] for issue in result["issues"]}
    assert (tmp_path / "report/index.html").is_file()


@pytest.mark.parametrize(
    "damage",
    ["missing", "corrupt", "duplicate", "traversal", "symlink", "malformed_json"],
)
def test_archive_damage_is_reported(
    exported_zip: Path, tmp_path: Path, damage: str
) -> None:
    damaged = tmp_path / "damaged.zip"
    with (
        zipfile.ZipFile(exported_zip) as original,
        zipfile.ZipFile(damaged, "w") as output,
    ):
        for index, info in enumerate(original.infolist()):
            if damage == "missing" and info.filename.startswith("media/"):
                continue
            payload = original.read(info)
            if damage == "corrupt" and info.filename.startswith("media/"):
                payload = b"not an image"
            if damage == "malformed_json" and info.filename == "manifest.jsonl":
                payload = b"not json\n"
            output.writestr(info, payload)
            if damage == "duplicate" and index == 0:
                with pytest.warns(UserWarning):
                    output.writestr(info, payload)
        if damage == "traversal":
            output.writestr("../escaped.txt", b"escape")
        if damage == "symlink":
            link = zipfile.ZipInfo("media/link.jpg")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            output.writestr(link, b"/etc/passwd")
    result = write_report(damaged, tmp_path / "report")
    assert not result["passed"]
    assert not (tmp_path / "escaped.txt").exists()


def test_preview_limit_does_not_limit_validation(
    exported_zip: Path, tmp_path: Path
) -> None:
    result = write_report(exported_zip, tmp_path / "report", preview_limit=1)
    assert result["summary"]["previews"] == 1
    assert result["summary"]["decoded_images"] == 3
    assert result["passed"]


def test_report_escapes_archive_text(exported_zip: Path, tmp_path: Path) -> None:
    altered = mutate(
        exported_zip,
        "bundle.json",
        lambda doc: {**doc, "bundle_id": "<script>alert(1)</script>"},
    )
    write_report(altered, tmp_path / "report")
    html = (tmp_path / "report/index.html").read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_cli_exit_codes_and_existing_output(exported_zip: Path, tmp_path: Path) -> None:
    destination = tmp_path / "report"
    assert main([str(exported_zip), "--report", str(destination)]) == 0
    saved = (destination / "report.json").read_bytes()
    assert main([str(exported_zip), "--report", str(destination)]) == 2
    assert (destination / "report.json").read_bytes() == saved
    invalid = tmp_path / "not-a-zip.zip"
    invalid.write_bytes(b"invalid")
    assert main([str(invalid), "--report", str(tmp_path / "bad")]) == 1
    assert (tmp_path / "bad/index.html").is_file()


@pytest.mark.parametrize("value", [None, 1, [], "wrong", {"annotations": 42}])
def test_malformed_coco_still_produces_failure_report(
    exported_zip: Path, tmp_path: Path, value: object
) -> None:
    altered = mutate(exported_zip, "views/coco.json", lambda _: value)
    result = write_report(altered, tmp_path / "report")
    assert not result["passed"]
    assert (tmp_path / "report/index.html").is_file()


def test_expected_exclusions_are_not_export_failures(
    exported_zip: Path, tmp_path: Path
) -> None:
    with zipfile.ZipFile(exported_zip) as archive:
        manifest = [
            json.loads(line) for line in archive.read("manifest.jsonl").splitlines()
        ]
    assert any(row["decision"] == "excluded" for row in manifest)
    result = write_report(exported_zip, tmp_path / "report")
    assert result["passed"]
    assert (
        "frame_has_unresolved_objects" in (tmp_path / "report/index.html").read_text()
    )
