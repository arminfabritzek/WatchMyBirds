"""Tests for the per-variant metadata block in the detector registry payload.

The Settings AI panel shows a meta sub-line under each variant id
(variant size, input resolution, recall, conf/iou, release date). These
come from ``<id>_metrics.json`` and ``<id>_model_config.yaml`` next to
the weights in the model dir. Not-installed rows fall back to hints
derived from the id itself (release date, variant-from-token).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from web.services.model_registry_service import (
    _build_variant_metadata,
    _compute_variant_tags,
    _released_from_id,
    _sort_variants_newest_first,
    _variant_from_id,
    build_detector_registry_payload,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload))
    else:
        path.write_text(payload)


def test_variant_from_id_recognizes_tiny_and_s():
    assert _variant_from_id("20260421_yolox_s_locator_640_v4") == "s"
    assert _variant_from_id("20260420_prodcal_yolox_tiny_640_mosaic0p5") == "tiny"
    # Legacy / unknown shapes -> None, no guessing
    assert _variant_from_id("20250810_215216") is None


def test_released_from_id_extracts_date_prefix():
    assert _released_from_id("20260421_yolox_s_locator_640_v4") == "2026-04-21"
    assert _released_from_id("20260420_prodcal_yolox_tiny_640_mosaic0p5") == "2026-04-20"
    # Non-numeric prefix -> None
    assert _released_from_id("latest") is None


def test_build_variant_metadata_reads_metrics_json(tmp_path: Path):
    """Installed variant with metrics.json: recall and conf flow through."""
    mid = "20260417_1636_yolox_tiny_640_mosaic0p5"
    _write(
        tmp_path / f"{mid}_metrics.json",
        {
            "version": mid,
            "chosen_threshold": {
                "conf": 0.15,
                "bird_recall": 0.993,
                "bird_precision": 0.9914,
                "f1": 0.9922,
            },
            "train_info": {"trained_at": "2026-04-17"},
        },
    )

    meta = _build_variant_metadata(str(tmp_path), mid, {})
    assert meta["bird_recall"] == pytest.approx(0.993)
    assert meta["f1"] == pytest.approx(0.9922)
    assert meta["conf"] == pytest.approx(0.15)
    assert meta["trained_at"] == "2026-04-17"
    # Id-derived fallbacks still fire for variant + released
    assert meta["variant"] == "tiny"
    assert meta["released"] == "2026-04-17"


def test_build_variant_metadata_yaml_overrides_conf_from_metrics(tmp_path: Path):
    """YAML is the authoritative runtime threshold source; its conf wins."""
    mid = "20260417_yolox_tiny_locator_ep120"
    _write(
        tmp_path / f"{mid}_metrics.json",
        {"chosen_threshold": {"conf": 0.10, "bird_recall": 0.99}},
    )
    _write(
        tmp_path / f"{mid}_model_config.yaml",
        """
detection:
  confidence_threshold: 0.15
  nms_iou_threshold: 0.5
  input_size: [640, 640]
  architecture: yolox_tiny_locator_5cls
meta:
  num_classes: 5
  trained_at: '2026-04-17'
""",
    )

    meta = _build_variant_metadata(str(tmp_path), mid, {})
    # YAML conf (0.15) wins over metrics.json conf (0.10)
    assert meta["conf"] == pytest.approx(0.15)
    assert meta["iou"] == pytest.approx(0.5)
    assert meta["input_size"] == [640, 640]
    assert meta["variant"] == "tiny"
    assert meta["num_classes"] == 5


def test_build_variant_metadata_no_local_files_uses_id_fallbacks(tmp_path: Path):
    """Not-installed variant: nothing on disk, but id reveals date + variant."""
    mid = "20260421_yolox_s_locator_640_v4"

    meta = _build_variant_metadata(str(tmp_path), mid, {})
    assert meta == {
        "variant": "s",
        "released": "2026-04-21",
    }
    # No metrics -> no recall, no conf, no thresholds
    assert "bird_recall" not in meta
    assert "conf" not in meta


def test_build_variant_metadata_registry_variant_wins_over_id_guess(tmp_path: Path):
    """When the registry entry already carries ``variant``, trust it."""
    mid = "20260420_prodcal_yolox_tiny_640_mosaic0p5"
    meta = _build_variant_metadata(
        str(tmp_path), mid, {"variant": "TINY "}  # trimmed + lowercased
    )
    assert meta["variant"] == "tiny"


def test_build_variant_metadata_handles_missing_yaml_gracefully(tmp_path: Path):
    """metrics.json on disk but no YAML: still returns what it has."""
    mid = "foo"
    _write(
        tmp_path / f"{mid}_metrics.json",
        {"chosen_threshold": {"bird_recall": 0.9}},
    )

    meta = _build_variant_metadata(str(tmp_path), mid, {})
    assert meta["bird_recall"] == pytest.approx(0.9)
    assert "conf" not in meta


def test_build_variant_metadata_handles_malformed_metrics(tmp_path: Path):
    """Broken metrics.json must not crash the registry payload build."""
    mid = "broken"
    _write(tmp_path / f"{mid}_metrics.json", "not-json{{{")
    # Should not raise
    meta = _build_variant_metadata(str(tmp_path), mid, {})
    assert isinstance(meta, dict)


@pytest.fixture
def _isolate_model_dir(tmp_path: Path, monkeypatch):
    """Point the registry service at tmp_path/object_detection."""
    model_base = tmp_path
    od_dir = model_base / "object_detection"
    od_dir.mkdir()
    # get_config() returns MODEL_BASE_PATH; force it at tmp_path.
    from web.services import model_registry_service as svc

    original = svc.get_config

    def fake_config():
        real = dict(original())
        real["MODEL_BASE_PATH"] = str(model_base)
        return real

    monkeypatch.setattr(svc, "get_config", fake_config)
    return od_dir


def test_registry_payload_includes_metadata_per_variant(_isolate_model_dir: Path):
    """End-to-end: build_detector_registry_payload attaches metadata blocks."""
    od = _isolate_model_dir
    active = "20260417_1636_yolox_tiny_640_mosaic0p5"
    remote_latest = "20260421_yolox_s_locator_640_v4"

    _write(
        od / "latest_models.json",
        {
            "latest": active,
            "weights_path": f"object_detection/{active}_best.onnx",
            "labels_path": f"object_detection/{active}_labels.json",
            "pinned_models": {
                active: {
                    "weights_path": f"object_detection/{active}_best.onnx",
                    "labels_path": f"object_detection/{active}_labels.json",
                },
                remote_latest: {
                    "weights_path": f"object_detection/{remote_latest}_best.onnx",
                    "labels_path": f"object_detection/{remote_latest}_labels.json",
                },
            },
        },
    )
    # Active has weights + metrics on disk (installed)
    (od / f"{active}_best.onnx").write_bytes(b"x")
    (od / f"{active}_labels.json").write_text('["bird"]')
    _write(
        od / f"{active}_metrics.json",
        {"chosen_threshold": {"conf": 0.15, "bird_recall": 0.993}},
    )
    # remote_latest not installed: no files

    payload = build_detector_registry_payload(detector=None)
    by_id = {v["id"]: v for v in payload["variants"]}

    # Installed variant has rich metadata
    active_meta = by_id[active]["metadata"]
    assert active_meta["bird_recall"] == pytest.approx(0.993)
    assert active_meta["conf"] == pytest.approx(0.15)
    assert active_meta["variant"] == "tiny"
    assert active_meta["released"] == "2026-04-17"

    # Not-installed variant gets id-derived hints only
    new_meta = by_id[remote_latest]["metadata"]
    assert new_meta["variant"] == "s"
    assert new_meta["released"] == "2026-04-21"
    assert "bird_recall" not in new_meta  # no metrics file on disk


# ---------------------------------------------------------------------------
# Tags + sorting
# ---------------------------------------------------------------------------


def _mkv(vid: str, *, variant: str | None = None,
         released: str | None = None,
         bird_recall: float | None = None) -> dict:
    """Minimal variant dict for tag/sort unit tests."""
    meta: dict = {}
    if variant:
        meta["variant"] = variant
    if released:
        meta["released"] = released
    if bird_recall is not None:
        meta["bird_recall"] = bird_recall
    return {"id": vid, "metadata": meta}


def test_compute_tags_newest_falls_back_to_plain_when_no_variant_info():
    """Entries without a detected size fall back to a single 'Newest'."""
    variants = [
        _mkv("old", released="2026-04-17"),
        _mkv("mid", released="2026-04-20"),
        _mkv("new", released="2026-04-21"),
    ]
    tags = _compute_variant_tags(variants)
    assert tags["new"] == ["Latest"]
    assert tags["mid"] == []
    assert tags["old"] == []


def test_compute_tags_newest_breaks_date_tie_by_bird_recall():
    """Three variants released on the same date — the one with highest
    bird_recall wins 'Newest', the others don't get it. Tagging all three
    'Newest' makes the hint meaningless (observed 2026-04-22 on Docker
    with _v3_noswarm + _BROKEN + _v4 all at 2026-04-21)."""
    variants = [
        _mkv("a", released="2026-04-21", bird_recall=0.98),
        _mkv("b", released="2026-04-21", bird_recall=0.99),  # newest winner
        _mkv("c", released="2026-04-21", bird_recall=0.97),
        _mkv("d", released="2026-04-20", bird_recall=0.95),
    ]
    tags = _compute_variant_tags(variants)
    assert tags["b"] == ["Latest", "Highest recall"]
    assert "Latest" not in tags["a"]
    assert "Latest" not in tags["c"]
    assert "Latest" not in tags["d"]  # older date, never newest


def test_compute_tags_newest_skipped_when_date_ties_and_recall_missing():
    """If the top-dated variants don't all have recall, skip the tag —
    comparing something-with-recall to something-without would be unfair."""
    variants = [
        _mkv("a", released="2026-04-21", bird_recall=0.99),
        _mkv("b", released="2026-04-21"),  # no recall
    ]
    tags = _compute_variant_tags(variants)
    assert "Latest" not in tags["a"]
    assert "Latest" not in tags["b"]


def test_compute_tags_newest_skipped_when_date_and_recall_tie():
    variants = [
        _mkv("a", released="2026-04-21", bird_recall=0.99),
        _mkv("b", released="2026-04-21", bird_recall=0.99),
    ]
    tags = _compute_variant_tags(variants)
    assert "Latest" not in tags["a"]
    assert "Latest" not in tags["b"]


def test_compute_tags_newest_per_variant_family():
    """With both s and tiny present, each family gets its own 'Newest X'
    so the user always sees a latest choice per hardware class."""
    variants = [
        _mkv("s_old", variant="s", released="2026-04-17"),
        _mkv("s_new", variant="s", released="2026-04-21"),
        _mkv("tiny_old", variant="tiny", released="2026-04-17"),
        _mkv("tiny_new", variant="tiny", released="2026-04-20"),
    ]
    tags = _compute_variant_tags(variants)
    assert "Latest s" in tags["s_new"]
    assert "Latest tiny" in tags["tiny_new"]
    # Older ones don't get the latest tag
    assert "Latest s" not in tags["s_old"]
    assert "Latest tiny" not in tags["tiny_old"]
    # And there's no cross-family bleed: tiny_new (2026-04-20) is older
    # than s_new (2026-04-21) but still earns "Latest tiny".
    assert "Latest" not in tags["s_new"]  # only the per-family label
    assert "Latest" not in tags["tiny_new"]


def test_compute_tags_faster_vs_more_accurate_needs_both_sizes():
    variants = [
        _mkv("s_model", variant="s"),
        _mkv("tiny_model", variant="tiny"),
    ]
    tags = _compute_variant_tags(variants)
    assert "Bigger, slower, better" in tags["s_model"]
    assert "Small, faster" in tags["tiny_model"]


def test_compute_tags_speed_badges_skipped_when_only_one_size_present():
    """'Faster' means nothing without an 's' sibling to compare against."""
    variants = [
        _mkv("tiny_a", variant="tiny"),
        _mkv("tiny_b", variant="tiny"),
    ]
    tags = _compute_variant_tags(variants)
    assert tags["tiny_a"] == []
    assert tags["tiny_b"] == []


def test_compute_tags_highest_recall_single_winner():
    variants = [
        _mkv("low", bird_recall=0.95),
        _mkv("mid", bird_recall=0.98),
        _mkv("top", bird_recall=0.993),
    ]
    tags = _compute_variant_tags(variants)
    assert tags["top"] == ["Highest recall"]
    assert tags["mid"] == []


def test_compute_tags_highest_recall_skipped_on_tie():
    """Ties are ambiguous — skip the badge rather than pick arbitrarily."""
    variants = [
        _mkv("a", bird_recall=0.99),
        _mkv("b", bird_recall=0.99),
    ]
    tags = _compute_variant_tags(variants)
    assert tags["a"] == []
    assert tags["b"] == []


def test_sort_variants_newest_first():
    variants = [
        _mkv("b", released="2026-04-17"),
        _mkv("a", released="2026-04-21"),
        _mkv("c", released="2026-04-20"),
        _mkv("undated"),
    ]
    order = [v["id"] for v in _sort_variants_newest_first(variants)]
    assert order == ["a", "c", "b", "undated"]


def test_registry_payload_sorts_newest_first_and_attaches_tags(_isolate_model_dir: Path):
    """End-to-end: payload variants arrive sorted + tagged."""
    od = _isolate_model_dir
    older_tiny = "20260417_1636_yolox_tiny_640_mosaic0p5"
    newer_s = "20260421_yolox_s_locator_640_v4"
    newer_tiny = "20260420_prodcal_yolox_tiny_640_mosaic0p5"

    _write(
        od / "latest_models.json",
        {
            "latest": older_tiny,
            "weights_path": f"object_detection/{older_tiny}_best.onnx",
            "labels_path": f"object_detection/{older_tiny}_labels.json",
            "pinned_models": {
                older_tiny: {
                    "weights_path": f"object_detection/{older_tiny}_best.onnx",
                    "labels_path": f"object_detection/{older_tiny}_labels.json",
                },
                newer_s: {
                    "weights_path": f"object_detection/{newer_s}_best.onnx",
                    "labels_path": f"object_detection/{newer_s}_labels.json",
                },
                newer_tiny: {
                    "weights_path": f"object_detection/{newer_tiny}_best.onnx",
                    "labels_path": f"object_detection/{newer_tiny}_labels.json",
                },
            },
        },
    )
    (od / f"{older_tiny}_best.onnx").write_bytes(b"x")
    (od / f"{older_tiny}_labels.json").write_text('["bird"]')
    _write(
        od / f"{older_tiny}_metrics.json",
        {"chosen_threshold": {"bird_recall": 0.993, "conf": 0.15}},
    )

    payload = build_detector_registry_payload(detector=None)
    order = [v["id"] for v in payload["variants"]]
    # Newest first: 2026-04-21, 2026-04-20, 2026-04-17
    assert order == [newer_s, newer_tiny, older_tiny]

    tags_by_id = {v["id"]: v["tags"] for v in payload["variants"]}
    # Per-family "Latest" tags: one per variant size
    assert "Latest s" in tags_by_id[newer_s]
    assert "Latest tiny" in tags_by_id[newer_tiny]
    assert "Latest tiny" not in tags_by_id[older_tiny]
    assert "Bigger, slower, better" in tags_by_id[newer_s]  # s vs tiny siblings exist
    assert "Small, faster" in tags_by_id[newer_tiny]
    assert "Small, faster" in tags_by_id[older_tiny]
    # Highest recall only on the one with metrics
    assert "Highest recall" in tags_by_id[older_tiny]


# ---------------------------------------------------------------------------
# HF-known whitelist filter (hides local-only non-active variants from UI)
# ---------------------------------------------------------------------------


def test_payload_filters_local_only_variants_when_hf_snapshot_present(_isolate_model_dir: Path):
    """Docker-volume scenario: legacy _BROKEN / dev artefacts vanish from
    the picker but stay on disk. Only HF-advertised variants plus the
    active one remain visible.
    """
    od = _isolate_model_dir
    active = "20260420_prodcal_yolox_s_640_mosaic0p5"  # local, not on HF
    hf_s = "20260421_yolox_s_locator_640_v4"           # HF latest
    hf_tiny = "20260420_prodcal_yolox_tiny_640_mosaic0p5"  # HF pinned
    stale_broken = "20260421_v3_noswarm_yolox_s_640_BROKEN"
    stale_old = "20260417_1512_yolox_s_640_mosaic0p5"

    _write(
        od / "latest_models.json",
        {
            "latest": active,
            "weights_path": f"object_detection/{active}_best.onnx",
            "labels_path": f"object_detection/{active}_labels.json",
            "hf_known_ids": [hf_s, hf_tiny],  # authoritative HF snapshot
            "pinned_models": {
                active: {
                    "weights_path": f"object_detection/{active}_best.onnx",
                    "labels_path": f"object_detection/{active}_labels.json",
                },
                hf_s: {
                    "weights_path": f"object_detection/{hf_s}_best.onnx",
                    "labels_path": f"object_detection/{hf_s}_labels.json",
                },
                hf_tiny: {
                    "weights_path": f"object_detection/{hf_tiny}_best.onnx",
                    "labels_path": f"object_detection/{hf_tiny}_labels.json",
                },
                stale_broken: {
                    "weights_path": f"object_detection/{stale_broken}_best.onnx",
                    "labels_path": f"object_detection/{stale_broken}_labels.json",
                },
                stale_old: {
                    "weights_path": f"object_detection/{stale_old}_best.onnx",
                    "labels_path": f"object_detection/{stale_old}_labels.json",
                },
            },
        },
    )
    # Make active + stale weights all present on disk (simulating the
    # real Docker volume where these files linger).
    for mid in (active, stale_broken, stale_old):
        (od / f"{mid}_best.onnx").write_bytes(b"x")
        (od / f"{mid}_labels.json").write_text('["bird"]')

    payload = build_detector_registry_payload(detector=None)
    ids = {v["id"] for v in payload["variants"]}

    # Visible: HF-advertised + the active one (even though active is not on HF)
    assert active in ids, "active local-only variant must remain visible"
    assert hf_s in ids
    assert hf_tiny in ids
    # Hidden: stale local-only artefacts
    assert stale_broken not in ids, "_BROKEN local-only variant must be hidden"
    assert stale_old not in ids, "stale local-only variant must be hidden"


def test_classifier_payload_tags_hf_latest_when_local_active_diverges(tmp_path: Path, monkeypatch):
    """Classifier sibling of the detector test — same preservation-guard
    divergence scenario, verified against build_classifier_registry_payload.

    Real-world case (2026-04-22 RPi): HF announces
    ``20260421_161805`` but its files are not on disk. The guard keeps
    ``20250817_213043`` locally, yet the UI must tag HF's advertised
    latest (161805) as ``is_hf_latest`` and the preserved one as
    ``is_active``.
    """
    cls_dir = tmp_path / "classifier"
    cls_dir.mkdir()

    local_active = "20250817_213043"
    hf_latest = "20260421_161805"

    payload_json = {
        "latest": local_active,
        "weights_path": f"classifier/{local_active}_best.onnx",
        "classes_path": f"classifier/{local_active}_classes.txt",
        "hf_latest_advertised": hf_latest,
        "hf_known_ids": [local_active, hf_latest],
        "pinned_models": {
            local_active: {
                "weights_path": f"classifier/{local_active}_best.onnx",
                "classes_path": f"classifier/{local_active}_classes.txt",
            },
            hf_latest: {
                "weights_path": f"classifier/{hf_latest}_best.onnx",
                "classes_path": f"classifier/{hf_latest}_classes.txt",
            },
        },
    }
    _write(cls_dir / "latest_models.json", payload_json)
    # Only the preserved active has files on disk.
    (cls_dir / f"{local_active}_best.onnx").write_bytes(b"x")
    (cls_dir / f"{local_active}_classes.txt").write_text("bird\n")

    from web.services import model_registry_service as svc

    original = svc.get_config

    def fake_config():
        real = dict(original())
        real["MODEL_BASE_PATH"] = str(tmp_path)
        return real

    monkeypatch.setattr(svc, "get_config", fake_config)

    payload = svc.build_classifier_registry_payload(classifier=None)
    by_id = {v["id"]: v for v in payload["variants"]}

    # Latest badge follows HF's view…
    assert by_id[hf_latest]["is_hf_latest"] is True
    assert by_id[local_active]["is_hf_latest"] is False
    # …while Active follows what's actually loadable locally.
    assert by_id[local_active]["is_active"] is True
    assert by_id[hf_latest]["is_active"] is False


def test_payload_tags_hf_latest_even_when_local_active_diverges(_isolate_model_dir: Path):
    """When the preservation guard keeps a different local id as active,
    the ``Latest`` badge must still land on HF's advertised latest.

    Real-world case (2026-04-22, classifier): HF announces
    ``20260421_161805`` but its files are not on disk, so the guard
    keeps local ``20250817_213043`` as active. The UI previously tagged
    the local id as Latest because ``is_hf_latest`` read from the
    top-level ``latest`` field (which is the preserved local pointer).
    The fix reads ``hf_latest_advertised`` — HF's own view, persisted
    separately on every successful merge — so the correct row wins.
    """
    od = _isolate_model_dir
    local_active = "20250817_213043"
    hf_latest = "20260421_161805"

    _write(
        od / "latest_models.json",
        {
            "latest": local_active,  # preservation guard kept this
            "weights_path": f"object_detection/{local_active}_best.onnx",
            "labels_path": f"object_detection/{local_active}_labels.json",
            "hf_latest_advertised": hf_latest,  # HF's own view
            "hf_known_ids": [local_active, hf_latest],
            "pinned_models": {
                local_active: {
                    "weights_path": f"object_detection/{local_active}_best.onnx",
                    "labels_path": f"object_detection/{local_active}_labels.json",
                },
                hf_latest: {
                    "weights_path": f"object_detection/{hf_latest}_best.onnx",
                    "labels_path": f"object_detection/{hf_latest}_labels.json",
                },
            },
        },
    )
    (od / f"{local_active}_best.onnx").write_bytes(b"x")
    (od / f"{local_active}_labels.json").write_text('["bird"]')

    payload = build_detector_registry_payload(detector=None)
    by_id = {v["id"]: v for v in payload["variants"]}

    # HF's latest wins the is_hf_latest flag and the "Latest" tag —
    # regardless of which id the guard preserved locally.
    assert by_id[hf_latest]["is_hf_latest"] is True
    assert by_id[local_active]["is_hf_latest"] is False
    # Active flag still tracks what's actually loaded locally.
    assert by_id[local_active]["is_active"] is True
    assert by_id[hf_latest]["is_active"] is False


def test_payload_shows_all_variants_when_no_hf_snapshot_available(_isolate_model_dir: Path):
    """Fresh install / offline first start: the hf_known_ids list is
    absent, so the filter degrades gracefully to showing every variant.
    Without this safeguard the UI would be empty until the first
    successful HF fetch completes.
    """
    od = _isolate_model_dir
    active = "locally_shipped_model"
    extra = "another_local_model"

    _write(
        od / "latest_models.json",
        {
            "latest": active,
            "weights_path": f"object_detection/{active}_best.onnx",
            "labels_path": f"object_detection/{active}_labels.json",
            # Note: no hf_known_ids key
            "pinned_models": {
                active: {
                    "weights_path": f"object_detection/{active}_best.onnx",
                    "labels_path": f"object_detection/{active}_labels.json",
                },
                extra: {
                    "weights_path": f"object_detection/{extra}_best.onnx",
                    "labels_path": f"object_detection/{extra}_labels.json",
                },
            },
        },
    )
    (od / f"{active}_best.onnx").write_bytes(b"x")
    (od / f"{active}_labels.json").write_text('["bird"]')

    payload = build_detector_registry_payload(detector=None)
    ids = {v["id"] for v in payload["variants"]}
    assert active in ids
    assert extra in ids, "without HF snapshot, all known variants stay visible"
