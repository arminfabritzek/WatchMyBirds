"""Tests for the new pin + preservation behaviour in utils.model_downloader.

Running the YOLOX model locally on the Pi before it exists on
HuggingFace means the remote fetch would otherwise overwrite our local
latest_models.json pointer and yank the pointer back to the 29-species
FasterRCNN. Two fixes tested here:

- WMB_PINNED_MODEL_ID skips the HF fetch entirely and uses the local
  JSON. Enables future "version selection" by adding entries under
  ``pinned_models`` in the local JSON.
- Preservation guard: when the remote latest points at files that are
  NOT on disk and the local pointer IS fully resolvable, we keep the
  local active pointer while merging remote variants into the local
  registry (unless WMB_FORCE_REMOTE_REFRESH is set).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from utils.model_downloader import (
    FORCE_REFRESH_ENV_VAR,
    PIN_ENV_VAR,
    PIN_ENV_VAR_PREFIX,
    fetch_latest_json,
    load_latest_identifier,
    set_latest_model_id,
)

BASE_URL = "https://example.test/object_detection"


@pytest.fixture(autouse=True)
def _isolate_pin_env(monkeypatch):
    """Clear both the generic and the task-specific pin env vars so tests
    don't leak configuration between each other or from the CI host."""
    for key in (
        PIN_ENV_VAR,
        f"{PIN_ENV_VAR_PREFIX}_OBJECT_DETECTION",
        f"{PIN_ENV_VAR_PREFIX}_CLASSIFIER",
        FORCE_REFRESH_ENV_VAR,
    ):
        monkeypatch.delenv(key, raising=False)


def _write_local(cache_dir: Path, payload: dict) -> None:
    (cache_dir / "latest_models.json").write_text(json.dumps(payload))


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------


def test_task_specific_pin_targets_only_matching_cache_dir(tmp_path, monkeypatch):
    """WMB_PINNED_MODEL_ID_OBJECT_DETECTION pins only the detector cache dir."""
    det_dir = tmp_path / "object_detection"
    det_dir.mkdir()
    cls_dir = tmp_path / "classifier"
    cls_dir.mkdir()

    _write_local(
        det_dir,
        {"latest": "detector_v99", "weights_path": "object_detection/x.onnx",
         "labels_path": "object_detection/x.json"},
    )
    _write_local(
        cls_dir,
        {"latest": "classifier_v7", "weights_path": "classifier/c.onnx",
         "labels_path": "classifier/c.json"},
    )

    monkeypatch.setenv(f"{PIN_ENV_VAR_PREFIX}_OBJECT_DETECTION", "detector_v99")

    # Detector: pin active, HF skipped
    with patch("utils.model_downloader.requests.get") as mock_get:
        det_data = fetch_latest_json(
            "https://hf.example/object_detection", str(det_dir)
        )
        mock_get.assert_not_called()
    assert det_data["latest"] == "detector_v99"

    # Classifier: pin NOT active for this task, HF is called.
    # The remote response will fail via ConnectionError; we fall back to local.
    with patch(
        "utils.model_downloader.requests.get",
        side_effect=requests.ConnectionError("offline"),
    ) as mock_get:
        cls_data = fetch_latest_json(
            "https://hf.example/classifier", str(cls_dir)
        )
        mock_get.assert_called_once()
    assert cls_data["latest"] == "classifier_v7"


def test_generic_pin_is_fallback_when_no_task_specific_set(tmp_path, monkeypatch):
    """Bare WMB_PINNED_MODEL_ID still works as a last-resort pin."""
    det_dir = tmp_path / "object_detection"
    det_dir.mkdir()
    _write_local(
        det_dir,
        {"latest": "pinned_legacy_v1",
         "weights_path": "object_detection/p.onnx",
         "labels_path": "object_detection/p.json"},
    )
    monkeypatch.setenv(PIN_ENV_VAR, "pinned_legacy_v1")

    with patch("utils.model_downloader.requests.get") as mock_get:
        data = fetch_latest_json(
            "https://hf.example/object_detection", str(det_dir)
        )
        mock_get.assert_not_called()
    assert data["latest"] == "pinned_legacy_v1"


def test_pin_skips_remote_fetch(tmp_path, monkeypatch):
    """When WMB_PINNED_MODEL_ID matches local latest, no HF call happens."""
    _write_local(
        tmp_path,
        {
            "latest": "20260417_crazy_detector_locator",
            "weights_path": "object_detection/20260417_crazy_detector_locator_best.onnx",
            "labels_path": "object_detection/20260417_crazy_detector_locator_labels.json",
        },
    )
    monkeypatch.setenv(PIN_ENV_VAR, "20260417_crazy_detector_locator")

    with patch("utils.model_downloader.requests.get") as mock_get:
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    mock_get.assert_not_called()
    assert data["latest"] == "20260417_crazy_detector_locator"


def test_pin_matches_pinned_models_entry(tmp_path, monkeypatch):
    """Pin can point at an alternate version listed under pinned_models."""
    _write_local(
        tmp_path,
        {
            "latest": "20260417_crazy_detector_locator",
            "weights_path": "object_detection/20260417_crazy_detector_locator_best.onnx",
            "labels_path": "object_detection/20260417_crazy_detector_locator_labels.json",
            "pinned_models": {
                "20250810_215216": {
                    "weights_path": "object_detection/20250810_215216_best.onnx",
                    "labels_path": "object_detection/20250810_215216_labels.json",
                }
            },
        },
    )
    monkeypatch.setenv(PIN_ENV_VAR, "20250810_215216")

    with patch("utils.model_downloader.requests.get") as mock_get:
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    mock_get.assert_not_called()
    assert data["latest"] == "20250810_215216"
    assert data["weights_path"].endswith("20250810_215216_best.onnx")
    assert data["labels_path"].endswith("20250810_215216_labels.json")


def test_pin_rejects_unknown_identifier(tmp_path, monkeypatch):
    _write_local(
        tmp_path,
        {
            "latest": "20260417_crazy_detector_locator",
            "weights_path": "object_detection/20260417_crazy_detector_locator_best.onnx",
            "labels_path": "object_detection/20260417_crazy_detector_locator_labels.json",
        },
    )
    monkeypatch.setenv(PIN_ENV_VAR, "some_unknown_id_v99")
    with pytest.raises(ValueError, match="does not match"):
        fetch_latest_json(BASE_URL, str(tmp_path))


def test_pin_without_local_json_errors(tmp_path, monkeypatch):
    monkeypatch.setenv(PIN_ENV_VAR, "anything")
    with pytest.raises(FileNotFoundError, match="no local latest_models.json"):
        fetch_latest_json(BASE_URL, str(tmp_path))


# ---------------------------------------------------------------------------
# Preservation guard
# ---------------------------------------------------------------------------


def _mock_remote(payload: dict):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return payload

    return _Resp()


def test_preservation_keeps_local_when_remote_files_missing(tmp_path):
    """Remote points at files we don't have -> keep local active pointer intact."""
    _write_local(
        tmp_path,
        {
            "latest": "20260417_crazy_detector_locator",
            "weights_path": "object_detection/20260417_crazy_detector_locator_best.onnx",
            "labels_path": "object_detection/20260417_crazy_detector_locator_labels.json",
        },
    )
    # Make local files resolvable on disk (preservation guard checks this)
    _touch(tmp_path / "20260417_crazy_detector_locator_best.onnx")
    _touch(tmp_path / "20260417_crazy_detector_locator_labels.json")

    remote_payload = {
        "latest": "20250810_215216",
        "weights_path": "object_detection/20250810_215216_best.onnx",
        "labels_path": "object_detection/20250810_215216_labels.json",
    }
    # NOTE: we don't write the 20250810_* files in tmp_path, so the remote
    # points at files that don't exist locally.

    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    # Preservation kicks in: local pointer wins
    assert data["latest"] == "20260417_crazy_detector_locator"
    # And the local JSON active pointer was NOT overwritten.
    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk["latest"] == "20260417_crazy_detector_locator"


def test_preservation_merges_remote_variants_when_remote_latest_missing(tmp_path):
    """Docker/RPi UX: keep the running model, but show new variants in Settings."""
    local_id = "20260417_1512_yolox_s_640_mosaic0p5"
    remote_s = "20260420_prodcal_yolox_s_640_mosaic0p5"
    remote_tiny = "20260420_prodcal_yolox_tiny_640_mosaic0p5"

    _write_local(
        tmp_path,
        {
            "latest": local_id,
            "weights_path": f"object_detection/{local_id}_best.onnx",
            "labels_path": f"object_detection/{local_id}_labels.json",
            "pinned_models": {
                local_id: {
                    "weights_path": f"object_detection/{local_id}_best.onnx",
                    "labels_path": f"object_detection/{local_id}_labels.json",
                    "active_precision": "int8_qdq",
                }
            },
        },
    )
    _touch(tmp_path / f"{local_id}_best.onnx")
    _touch(tmp_path / f"{local_id}_labels.json")

    remote_payload = {
        "latest": remote_tiny,
        # Use a higher-priority top-level alias here to prove the merge does
        # not leave a stale remote active path next to the preserved local id.
        "weights_path_onnx": f"object_detection/{remote_tiny}_best.onnx",
        "labels_path": f"object_detection/{remote_tiny}_labels.json",
        "pinned_models": {
            local_id: {
                "weights_path": f"object_detection/{local_id}_best.onnx",
                "labels_path": f"object_detection/{local_id}_labels.json",
            },
            remote_s: {
                "weights_path": f"object_detection/{remote_s}_best.onnx",
                "labels_path": f"object_detection/{remote_s}_labels.json",
            },
            remote_tiny: {
                "weights_path": f"object_detection/{remote_tiny}_best.onnx",
                "labels_path": f"object_detection/{remote_tiny}_labels.json",
            },
        },
    }

    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    assert data["latest"] == local_id
    assert data["weights_path"] == f"object_detection/{local_id}_best.onnx"
    assert "weights_path_onnx" not in data
    assert sorted(data["pinned_models"]) == [local_id, remote_s, remote_tiny]
    assert data["pinned_models"][local_id]["active_precision"] == "int8_qdq"

    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk == data


def test_preservation_surfaces_remote_latest_even_when_not_in_pinned_models(tmp_path):
    """HF publisher bumps ``latest`` to a new id but only lists older ids
    under ``pinned_models``. End users on Docker/RPi must still see the new
    variant in Settings so they can click "Install & switch" — otherwise the
    only workaround is ``WMB_FORCE_REMOTE_REFRESH=1``, which typical
    operators cannot set.
    """
    local_id = "20260420_prodcal_yolox_s_640_mosaic0p5"
    remote_latest = "20260421_yolox_s_locator_640_v4"
    remote_tiny = "20260420_prodcal_yolox_tiny_640_mosaic0p5"

    _write_local(
        tmp_path,
        {
            "latest": local_id,
            "weights_path": f"object_detection/{local_id}_best.onnx",
            "labels_path": f"object_detection/{local_id}_labels.json",
        },
    )
    _touch(tmp_path / f"{local_id}_best.onnx")
    _touch(tmp_path / f"{local_id}_labels.json")

    # Remote shape mirrors the real HF payload (2026-04-21): ``latest`` points
    # at a new id that is NOT listed under pinned_models, pinned_models only
    # contains an older sibling.
    remote_payload = {
        "latest": remote_latest,
        "project_name": "yolox-s-locator-5cls",
        "weights_path": f"object_detection/{remote_latest}_best.onnx",
        "labels_path": f"object_detection/{remote_latest}_labels.json",
        "config_path": f"object_detection/{remote_latest}_model_config.yaml",
        "weights_int8_path": f"object_detection/{remote_latest}_best_int8.onnx",
        "pinned_models": {
            remote_tiny: {
                "project_name": "yolox-tiny-locator-5cls",
                "variant": "tiny",
                "weights_path": f"object_detection/{remote_tiny}_best.onnx",
                "labels_path": f"object_detection/{remote_tiny}_labels.json",
            },
        },
    }

    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    # Preservation keeps the local active pointer intact.
    assert data["latest"] == local_id
    # But every variant the publisher shipped is now reachable from the UI:
    # the new ``remote_latest`` AND the older pinned sibling AND the local id.
    merged = data["pinned_models"]
    assert remote_latest in merged, (
        "remote latest must be surfaced as a variant even when the publisher "
        "did not list it under pinned_models"
    )
    assert merged[remote_latest]["weights_path"] == (
        f"object_detection/{remote_latest}_best.onnx"
    )
    assert merged[remote_latest]["labels_path"] == (
        f"object_detection/{remote_latest}_labels.json"
    )
    assert remote_tiny in merged
    # The explicit publisher entry for the tiny is preserved verbatim
    # (synth does not clobber pinned_models entries the publisher shipped).
    assert merged[remote_tiny]["variant"] == "tiny"

    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk == data


def test_preservation_prunes_stale_local_only_variants_without_weights(tmp_path):
    """Local-only entries that HF has removed AND have no weights on disk
    are rubble (UI shows "Not installed" but install would 404). Drop them.

    Safety: keep the active pointer and any entry whose weights are on disk.
    """
    active_id = "20260420_prodcal_yolox_s_640_mosaic0p5"
    remote_latest = "20260421_yolox_s_locator_640_v4"
    remote_tiny = "20260420_prodcal_yolox_tiny_640_mosaic0p5"
    stale_broken = "20260421_v3_noswarm_yolox_s_640_BROKEN"
    stale_noswarm = "20260421_v3_noswarm_yolox_s_640"
    stale_1512 = "20260417_1512_yolox_s_640_mosaic0p5"
    locally_installed = "20260417_1636_yolox_tiny_640_mosaic0p5"

    _write_local(
        tmp_path,
        {
            "latest": active_id,
            "weights_path": f"object_detection/{active_id}_best.onnx",
            "labels_path": f"object_detection/{active_id}_labels.json",
            "pinned_models": {
                active_id: {
                    "weights_path": f"object_detection/{active_id}_best.onnx",
                    "labels_path": f"object_detection/{active_id}_labels.json",
                },
                stale_broken: {
                    "weights_path": f"object_detection/{stale_broken}_best.onnx",
                    "labels_path": f"object_detection/{stale_broken}_labels.json",
                },
                stale_noswarm: {
                    "weights_path": f"object_detection/{stale_noswarm}_best.onnx",
                    "labels_path": f"object_detection/{stale_noswarm}_labels.json",
                },
                stale_1512: {
                    "weights_path": f"object_detection/{stale_1512}_best.onnx",
                    "labels_path": f"object_detection/{stale_1512}_labels.json",
                },
                locally_installed: {
                    "weights_path": f"object_detection/{locally_installed}_best.onnx",
                    "labels_path": f"object_detection/{locally_installed}_labels.json",
                },
            },
        },
    )
    # Only the active and the locally-installed variant have files on disk.
    _touch(tmp_path / f"{active_id}_best.onnx")
    _touch(tmp_path / f"{active_id}_labels.json")
    _touch(tmp_path / f"{locally_installed}_best.onnx")
    _touch(tmp_path / f"{locally_installed}_labels.json")

    # HF on 2026-04-21: publisher removed every stale id; only advertises _v4
    # as latest and _prodcal_tiny under pinned_models.
    remote_payload = {
        "latest": remote_latest,
        "weights_path": f"object_detection/{remote_latest}_best.onnx",
        "labels_path": f"object_detection/{remote_latest}_labels.json",
        "pinned_models": {
            remote_tiny: {
                "variant": "tiny",
                "weights_path": f"object_detection/{remote_tiny}_best.onnx",
                "labels_path": f"object_detection/{remote_tiny}_labels.json",
            },
        },
    }

    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    assert data["latest"] == active_id  # preservation still wins
    merged = data["pinned_models"]
    # Kept:
    assert active_id in merged, "active pointer must never be pruned"
    assert locally_installed in merged, "entries with weights on disk are kept"
    assert remote_latest in merged, "remote latest is surfaced for install"
    assert remote_tiny in merged, "remote pinned entries are surfaced"
    # Pruned:
    assert stale_broken not in merged, "stale local-only _BROKEN must be dropped"
    assert stale_noswarm not in merged
    assert stale_1512 not in merged

    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk == data


def test_preservation_does_not_prune_locally_installed_variants(tmp_path):
    """If a variant's weights are on disk (user installed it), we keep the
    pinned_models entry even when HF removed the variant — the user can
    still switch to it locally.
    """
    active_id = "active"
    orphaned_but_installed = "orphan_with_files"

    _write_local(
        tmp_path,
        {
            "latest": active_id,
            "weights_path": f"object_detection/{active_id}_best.onnx",
            "labels_path": f"object_detection/{active_id}_labels.json",
            "pinned_models": {
                active_id: {
                    "weights_path": f"object_detection/{active_id}_best.onnx",
                    "labels_path": f"object_detection/{active_id}_labels.json",
                },
                orphaned_but_installed: {
                    "weights_path": f"object_detection/{orphaned_but_installed}_best.onnx",
                    "labels_path": f"object_detection/{orphaned_but_installed}_labels.json",
                },
            },
        },
    )
    _touch(tmp_path / f"{active_id}_best.onnx")
    _touch(tmp_path / f"{active_id}_labels.json")
    _touch(tmp_path / f"{orphaned_but_installed}_best.onnx")
    _touch(tmp_path / f"{orphaned_but_installed}_labels.json")

    # HF forgot about both ids — only advertises something new.
    remote_payload = {
        "latest": "brand_new",
        "weights_path": "object_detection/brand_new_best.onnx",
        "labels_path": "object_detection/brand_new_labels.json",
    }
    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    merged = data["pinned_models"]
    assert orphaned_but_installed in merged, (
        "installed variants must not be pruned even if HF drops them"
    )
    assert active_id in merged
    assert "brand_new" in merged, "new HF latest surfaced"


def test_force_refresh_overrides_preservation(tmp_path, monkeypatch):
    """WMB_FORCE_REMOTE_REFRESH=1 bypasses the guard."""
    _write_local(
        tmp_path,
        {
            "latest": "20260417_crazy_detector_locator",
            "weights_path": "object_detection/20260417_crazy_detector_locator_best.onnx",
            "labels_path": "object_detection/20260417_crazy_detector_locator_labels.json",
        },
    )
    _touch(tmp_path / "20260417_crazy_detector_locator_best.onnx")
    _touch(tmp_path / "20260417_crazy_detector_locator_labels.json")

    remote_payload = {
        "latest": "20250810_215216",
        "weights_path": "object_detection/20250810_215216_best.onnx",
        "labels_path": "object_detection/20250810_215216_labels.json",
    }
    monkeypatch.setenv(FORCE_REFRESH_ENV_VAR, "1")

    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    # Force refresh: remote wins even though its files are missing
    assert data["latest"] == "20250810_215216"
    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk["latest"] == "20250810_215216"


def test_local_wins_on_conflict_even_when_both_usable(tmp_path):
    """New policy (needed so UI switches survive the live detector reload).

    Scenario: UI has written latest_models.json to point at new_id. The
    install endpoint fetched new_id's weights, so both old_id and new_id
    files are on disk. HF still advertises old_id as latest. Without
    this guard the reload path would overwrite the user's choice with
    HF's pointer — observed live on RPi 2026-04-17: UI switched to
    yolox_s, detector reloaded yolox_tiny because HF wrote back.
    """
    _write_local(
        tmp_path,
        {
            "latest": "new_id",
            "weights_path": "object_detection/new_best.onnx",
            "labels_path": "object_detection/new_labels.json",
        },
    )
    _touch(tmp_path / "old_best.onnx")
    _touch(tmp_path / "old_labels.json")
    _touch(tmp_path / "new_best.onnx")
    _touch(tmp_path / "new_labels.json")

    remote_payload = {
        "latest": "old_id",
        "weights_path": "object_detection/old_best.onnx",
        "labels_path": "object_detection/old_labels.json",
    }
    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    # Local choice preserved, on-disk JSON untouched.
    assert data["latest"] == "new_id"
    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk["latest"] == "new_id"


def test_remote_wins_when_local_json_missing(tmp_path):
    """Fresh install (no latest_models.json yet): HF is authoritative."""
    _touch(tmp_path / "new_best.onnx")
    _touch(tmp_path / "new_labels.json")

    remote_payload = {
        "latest": "new_id",
        "weights_path": "object_detection/new_best.onnx",
        "labels_path": "object_detection/new_labels.json",
    }
    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    assert data["latest"] == "new_id"
    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk["latest"] == "new_id"


def test_force_refresh_overrides_local_preservation(tmp_path, monkeypatch):
    """WMB_FORCE_REMOTE_REFRESH=1 escape hatch still works: when the
    operator explicitly wants HF's view to win, we persist remote even
    though local is a usable different id."""
    monkeypatch.setenv("WMB_FORCE_REMOTE_REFRESH", "1")
    _write_local(
        tmp_path,
        {
            "latest": "local_choice",
            "weights_path": "object_detection/local_choice_best.onnx",
            "labels_path": "object_detection/local_choice_labels.json",
        },
    )
    _touch(tmp_path / "local_choice_best.onnx")
    _touch(tmp_path / "local_choice_labels.json")
    _touch(tmp_path / "hf_choice_best.onnx")
    _touch(tmp_path / "hf_choice_labels.json")

    remote_payload = {
        "latest": "hf_choice",
        "weights_path": "object_detection/hf_choice_best.onnx",
        "labels_path": "object_detection/hf_choice_labels.json",
    }
    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    assert data["latest"] == "hf_choice"
    on_disk = json.loads((tmp_path / "latest_models.json").read_text())
    assert on_disk["latest"] == "hf_choice"


def test_network_failure_falls_back_to_local(tmp_path):
    _write_local(
        tmp_path,
        {
            "latest": "local_only",
            "weights_path": "object_detection/local_only_best.onnx",
            "labels_path": "object_detection/local_only_labels.json",
        },
    )

    with patch(
        "utils.model_downloader.requests.get",
        side_effect=requests.ConnectionError("offline"),
    ):
        data = fetch_latest_json(BASE_URL, str(tmp_path))

    assert data["latest"] == "local_only"


def test_set_latest_model_id_rewrites_latest(tmp_path):
    """set_latest_model_id flips latest_models.json['latest'] + paths."""
    _write_local(
        tmp_path,
        {
            "latest": "default_v1",
            "weights_path": "object_detection/default_v1_best.onnx",
            "labels_path": "object_detection/default_v1_labels.json",
            "pinned_models": {
                "alt_v2": {
                    "weights_path": "object_detection/alt_v2_best.onnx",
                    "labels_path": "object_detection/alt_v2_labels.json",
                }
            },
        },
    )
    # Make alt_v2's files exist so _local_payload_is_usable passes.
    _touch(tmp_path / "alt_v2_best.onnx")
    _touch(tmp_path / "alt_v2_labels.json")

    set_latest_model_id(str(tmp_path), "alt_v2")

    updated = json.loads((tmp_path / "latest_models.json").read_text())
    assert updated["latest"] == "alt_v2"
    assert updated["weights_path"].endswith("alt_v2_best.onnx")
    assert updated["labels_path"].endswith("alt_v2_labels.json")
    # pinned_models block is preserved.
    assert "alt_v2" in updated.get("pinned_models", {})


def test_set_latest_model_id_rejects_unknown(tmp_path):
    _write_local(
        tmp_path,
        {
            "latest": "default_v1",
            "weights_path": "object_detection/default_v1_best.onnx",
            "labels_path": "object_detection/default_v1_labels.json",
        },
    )
    _touch(tmp_path / "default_v1_best.onnx")
    _touch(tmp_path / "default_v1_labels.json")

    with pytest.raises(ValueError, match="does not match"):
        set_latest_model_id(str(tmp_path), "not_a_known_model")


def test_set_latest_model_id_rejects_variant_with_missing_files(tmp_path):
    """A variant listed under pinned_models but whose files aren't on disk is rejected."""
    _write_local(
        tmp_path,
        {
            "latest": "default_v1",
            "weights_path": "object_detection/default_v1_best.onnx",
            "labels_path": "object_detection/default_v1_labels.json",
            "pinned_models": {
                "phantom_v3": {
                    "weights_path": "object_detection/phantom_v3_best.onnx",
                    "labels_path": "object_detection/phantom_v3_labels.json",
                }
            },
        },
    )
    _touch(tmp_path / "default_v1_best.onnx")
    _touch(tmp_path / "default_v1_labels.json")
    # phantom_v3 files deliberately NOT touched.

    with pytest.raises(FileNotFoundError, match="missing on disk"):
        set_latest_model_id(str(tmp_path), "phantom_v3")


def test_set_latest_model_id_empty_raises(tmp_path):
    _write_local(
        tmp_path,
        {
            "latest": "default_v1",
            "weights_path": "object_detection/default_v1_best.onnx",
            "labels_path": "object_detection/default_v1_labels.json",
        },
    )
    with pytest.raises(ValueError, match="non-empty"):
        set_latest_model_id(str(tmp_path), "")


def test_set_latest_model_id_is_atomic(tmp_path):
    """Write must not leave a dangling .tmp file behind."""
    _write_local(
        tmp_path,
        {
            "latest": "v1",
            "weights_path": "object_detection/v1_best.onnx",
            "labels_path": "object_detection/v1_labels.json",
            "pinned_models": {
                "v2": {
                    "weights_path": "object_detection/v2_best.onnx",
                    "labels_path": "object_detection/v2_labels.json",
                }
            },
        },
    )
    _touch(tmp_path / "v1_best.onnx")
    _touch(tmp_path / "v1_labels.json")
    _touch(tmp_path / "v2_best.onnx")
    _touch(tmp_path / "v2_labels.json")

    set_latest_model_id(str(tmp_path), "v2")
    assert not (tmp_path / "latest_models.json.tmp").exists()


def test_set_latest_model_id_without_local_json_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No latest_models.json"):
        set_latest_model_id(str(tmp_path), "anything")


def test_load_latest_identifier_reads_current_pointer(tmp_path):
    _write_local(
        tmp_path,
        {
            "latest": "20260417_crazy_detector_locator",
            "weights_path": "object_detection/20260417_crazy_detector_locator_best.onnx",
            "labels_path": "object_detection/20260417_crazy_detector_locator_labels.json",
        },
    )
    assert (
        load_latest_identifier(str(tmp_path)) == "20260417_crazy_detector_locator"
    )


# ---------------------------------------------------------------------------
# ensure_model_files — autofetch path pulls companion files too
# ---------------------------------------------------------------------------


def test_ensure_model_files_also_fetches_companions(tmp_path):
    """Regression guard for the NAS cold-start bug (2026-04-18):
    when ensure_model_files boots a fresh deployment from HF, it must
    pull not just weights+labels but also _model_config.yaml and
    _metrics.json — otherwise the pin endpoint falls back to hardcoded
    thresholds and the AI panel shows null metrics. Observed live:
    Tiny was installed via startup-autofetch, S via /install endpoint;
    Tiny lacked the YAML (hence the bug), S had it.
    """
    from utils.model_downloader import ensure_model_files

    mid = "20260417_yolox_tiny_640_mosaic0p5"
    remote_payload = {
        "latest": mid,
        "weights_path": f"object_detection/{mid}_best.onnx",
        "labels_path": f"object_detection/{mid}_labels.json",
    }

    calls: list[tuple[str, str]] = []

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        calls.append((url, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"fake-fetched")
        return True

    with (
        patch(
            "utils.model_downloader.requests.get",
            return_value=_mock_remote(remote_payload),
        ),
        patch("utils.model_downloader._download_file", side_effect=fake_download),
    ):
        weights, labels = ensure_model_files(
            BASE_URL, str(tmp_path), "weights_path", "labels_path"
        )

    # Weights + labels returned as before.
    assert weights.endswith(f"{mid}_best.onnx")
    assert labels.endswith(f"{mid}_labels.json")

    # Companions landed too.
    assert (tmp_path / f"{mid}_model_config.yaml").exists()
    assert (tmp_path / f"{mid}_metrics.json").exists()

    # Four _download_file invocations: weights, labels, yaml, metrics.
    assert len(calls) == 4
    fetched_dests = [c[1] for c in calls]
    assert any(d.endswith("_best.onnx") for d in fetched_dests)
    assert any(d.endswith("_labels.json") for d in fetched_dests)
    assert any(d.endswith("_model_config.yaml") for d in fetched_dests)
    assert any(d.endswith("_metrics.json") for d in fetched_dests)


def test_ensure_model_files_tolerates_missing_companions(tmp_path):
    """Older releases without a YAML/metrics companion must still boot.
    Autofetch swallows the companion 404s and returns weights+labels."""
    from utils.model_downloader import ensure_model_files

    mid = "20240101_legacy_release"
    remote_payload = {
        "latest": mid,
        "weights_path": f"object_detection/{mid}_best.onnx",
        "labels_path": f"object_detection/{mid}_labels.json",
    }

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        # Simulate HF 404 for companions; weights/labels succeed.
        if url.endswith((".yaml", "_metrics.json")):
            return False
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"fake-fetched")
        return True

    with (
        patch(
            "utils.model_downloader.requests.get",
            return_value=_mock_remote(remote_payload),
        ),
        patch("utils.model_downloader._download_file", side_effect=fake_download),
    ):
        weights, labels = ensure_model_files(
            BASE_URL, str(tmp_path), "weights_path", "labels_path"
        )

    assert os.path.exists(weights)
    assert os.path.exists(labels)
    # Companions are NOT on disk — the autofetch tolerated the 404.
    assert not (tmp_path / f"{mid}_model_config.yaml").exists()
    assert not (tmp_path / f"{mid}_metrics.json").exists()


def test_ensure_model_files_skips_companions_when_flag_false(tmp_path):
    """with_companions=False suppresses the companion fetch entirely.
    Classifier lineage uses this — releases there ship only weights +
    classes.txt, and the 3x retries per companion (~6 s) plus the
    trailing ERROR log lines are just noise for that path.
    """
    from utils.model_downloader import ensure_model_files

    mid = "20250817_213043"
    remote_payload = {
        "latest": mid,
        "weights_path": f"classifier/{mid}_best.onnx",
        "classes_path": f"classifier/{mid}_classes.txt",
    }

    calls: list[tuple[str, str]] = []

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        calls.append((url, dest))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"fake-fetched")
        return True

    with (
        patch(
            "utils.model_downloader.requests.get",
            return_value=_mock_remote(remote_payload),
        ),
        patch("utils.model_downloader._download_file", side_effect=fake_download),
    ):
        weights, classes = ensure_model_files(
            BASE_URL,
            str(tmp_path),
            "weights_path",
            "classes_path",
            with_companions=False,
        )

    # Core files still downloaded.
    assert os.path.exists(weights)
    assert os.path.exists(classes)
    # Exactly two downloads — no YAML/metrics attempted.
    assert len(calls) == 2
    for url, _ in calls:
        assert not url.endswith(".yaml")
        assert not url.endswith("_metrics.json")


# ---------------------------------------------------------------------------
# prune_legacy_fasterrcnn_models — startup migration cleanup
# ---------------------------------------------------------------------------


def _write_legacy_fasterrcnn_layout(model_dir: Path, mid: str = "20250810_215216") -> None:
    """Create a realistic FasterRCNN-era directory (dict-form 29-species
    labels.json + weights + latest_models.json pointing at it)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    legacy_labels = {str(i): f"Species_{i}" for i in range(29)}
    (model_dir / f"{mid}_labels.json").write_text(json.dumps(legacy_labels))
    (model_dir / f"{mid}_best.onnx").write_bytes(b"fake-fasterrcnn-weights")
    (model_dir / "latest_models.json").write_text(
        json.dumps(
            {
                "latest": mid,
                "weights_path": f"object_detection/{mid}_best.onnx",
                "labels_path": f"object_detection/{mid}_labels.json",
                "pinned_models": {
                    mid: {
                        "weights_path": f"object_detection/{mid}_best.onnx",
                        "labels_path": f"object_detection/{mid}_labels.json",
                    }
                },
            }
        )
    )
    (model_dir / "model_metadata.json").write_text(
        json.dumps({"framework": "fasterrcnn", "inference_thresholds": {"confidence": 0.5}})
    )


def _write_yolox_layout(model_dir: Path, mid: str = "20260417_yolox_tiny_locator_ep120") -> None:
    """Create a realistic YOLOX directory (list-form 5-class labels + yaml)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    yolox_labels = ["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"]
    (model_dir / f"{mid}_labels.json").write_text(json.dumps(yolox_labels))
    (model_dir / f"{mid}_best.onnx").write_bytes(b"fake-yolox-weights")
    (model_dir / f"{mid}_model_config.yaml").write_text(
        "detection:\n  confidence_threshold: 0.15\n  output_format: yolox_raw\n"
    )
    (model_dir / "latest_models.json").write_text(
        json.dumps(
            {
                "latest": mid,
                "weights_path": f"object_detection/{mid}_best.onnx",
                "labels_path": f"object_detection/{mid}_labels.json",
                "pinned_models": {
                    mid: {
                        "weights_path": f"object_detection/{mid}_best.onnx",
                        "labels_path": f"object_detection/{mid}_labels.json",
                    }
                },
            }
        )
    )


def test_prune_legacy_removes_fasterrcnn_artefacts(tmp_path):
    """In-place upgrade path: the deployment had FasterRCNN files laying
    around from an older release. The startup cleanup wipes them so the
    HF autofetch below will provision the current YOLOX release.
    """
    from utils.model_downloader import prune_legacy_fasterrcnn_models

    _write_legacy_fasterrcnn_layout(tmp_path)
    removed = prune_legacy_fasterrcnn_models(str(tmp_path))

    # Legacy files are gone.
    assert not (tmp_path / "20250810_215216_best.onnx").exists()
    assert not (tmp_path / "20250810_215216_labels.json").exists()
    assert not (tmp_path / "latest_models.json").exists()
    assert not (tmp_path / "model_metadata.json").exists()
    assert any(p.endswith("_best.onnx") for p in removed)
    assert any(p.endswith("_labels.json") for p in removed)
    assert any(p.endswith("latest_models.json") for p in removed)


def test_prune_legacy_is_noop_on_yolox_layout(tmp_path):
    """Running the cleanup on a clean YOLOX deployment must touch
    nothing. This guarantees the cleanup can run on every startup
    without risk to modern installations."""
    from utils.model_downloader import prune_legacy_fasterrcnn_models

    _write_yolox_layout(tmp_path)
    mid = "20260417_yolox_tiny_locator_ep120"
    expected_files = [
        f"{mid}_best.onnx",
        f"{mid}_labels.json",
        f"{mid}_model_config.yaml",
        "latest_models.json",
    ]
    removed = prune_legacy_fasterrcnn_models(str(tmp_path))

    assert removed == []
    for f in expected_files:
        assert (tmp_path / f).exists(), f"YOLOX file {f} was wrongly removed"


def test_prune_legacy_is_noop_on_empty_dir(tmp_path):
    """Fresh deployment without any latest_models.json: cleanup is a no-op
    (no legacy detection possible, autofetch provisions everything later)."""
    from utils.model_downloader import prune_legacy_fasterrcnn_models

    assert prune_legacy_fasterrcnn_models(str(tmp_path)) == []


def test_prune_legacy_is_noop_on_missing_dir(tmp_path):
    """If the dir itself does not exist (first-boot on a fresh Pi where
    /opt/app/data/models has not been created yet), cleanup does not
    raise. DetectionManager init will create the dir via ensure_model_files."""
    from utils.model_downloader import prune_legacy_fasterrcnn_models

    nonexistent = tmp_path / "does-not-exist" / "object_detection"
    assert prune_legacy_fasterrcnn_models(str(nonexistent)) == []


def test_prune_legacy_leaves_yolox_when_mixed(tmp_path):
    """Half-migrated deployment with both a legacy FasterRCNN variant
    AND a modern YOLOX variant listed in pinned_models: only the
    FasterRCNN files are removed. The YOLOX files stay because their
    labels.json is a list, not a dict with >=20 numeric keys."""
    from utils.model_downloader import prune_legacy_fasterrcnn_models

    # Build a mixed layout: pinned_models has both; latest points at YOLOX.
    legacy_mid = "20250810_215216"
    yolox_mid = "20260417_yolox_tiny_locator_ep120"

    # Legacy files
    legacy_labels = {str(i): f"Species_{i}" for i in range(29)}
    (tmp_path / f"{legacy_mid}_labels.json").write_text(json.dumps(legacy_labels))
    (tmp_path / f"{legacy_mid}_best.onnx").write_bytes(b"legacy")

    # YOLOX files
    (tmp_path / f"{yolox_mid}_labels.json").write_text(
        json.dumps(["bird", "squirrel", "cat", "marten_mustelid", "hedgehog"])
    )
    (tmp_path / f"{yolox_mid}_best.onnx").write_bytes(b"yolox")

    # latest_models.json pinning both
    (tmp_path / "latest_models.json").write_text(
        json.dumps(
            {
                "latest": yolox_mid,
                "pinned_models": {
                    legacy_mid: {
                        "weights_path": f"object_detection/{legacy_mid}_best.onnx",
                        "labels_path": f"object_detection/{legacy_mid}_labels.json",
                    },
                    yolox_mid: {
                        "weights_path": f"object_detection/{yolox_mid}_best.onnx",
                        "labels_path": f"object_detection/{yolox_mid}_labels.json",
                    },
                },
            }
        )
    )

    removed = prune_legacy_fasterrcnn_models(str(tmp_path))

    # YOLOX files preserved
    assert (tmp_path / f"{yolox_mid}_best.onnx").exists()
    assert (tmp_path / f"{yolox_mid}_labels.json").exists()

    # Legacy files gone
    assert not (tmp_path / f"{legacy_mid}_best.onnx").exists()
    assert not (tmp_path / f"{legacy_mid}_labels.json").exists()

    # latest_models.json is also wiped so autofetch can rewrite it cleanly
    # (the preservation guard would otherwise see a local file with
    # unusable pinned_models entries).
    assert not (tmp_path / "latest_models.json").exists()
    assert len(removed) >= 3


def test_ensure_model_files_regenerates_metadata_from_fresh_yaml(tmp_path):
    """Regression guard (2026-04-18 RPi cold-start bug).

    Before this fix: a cold-start autofetch pulled weights + labels +
    YAML + metrics, but never wrote model_metadata.json. The detector
    init then fell back to the hardcoded YOLOX_DEFAULT_CONF_THR (0.15)
    regardless of what the YAML said. Worse, if a previous boot left a
    stale model_metadata.json from a different variant around, the
    detector silently ran with the wrong thresholds.

    After this fix: _fetch_companion_files regenerates
    model_metadata.json whenever a YAML is on disk in the OBJECT_DETECTION
    dir, so cold-start and pin paths both converge on the same file.
    """
    from utils.model_downloader import ensure_model_files

    # Cache dir basename "object_detection" is what _task_name_from_cache_dir
    # uses to decide "this is the detector lineage; regenerate metadata".
    od_dir = tmp_path / "object_detection"
    od_dir.mkdir()

    mid = "20260417_yolox_s_640_mosaic0p5"
    remote_payload = {
        "latest": mid,
        "weights_path": f"object_detection/{mid}_best.onnx",
        "labels_path": f"object_detection/{mid}_labels.json",
    }

    # Real YAML content with S's conf=0.30 — the whole point of this
    # test is that 0.30 ends up in model_metadata.json, not Tiny's
    # hardcoded default 0.15.
    s_yaml_body = (
        "detection:\n"
        "  confidence_threshold: 0.30\n"
        "  nms_iou_threshold: 0.50\n"
        "  input_size: [640, 640]\n"
        "  input_format: BGR\n"
        "  input_normalize: false\n"
        "  output_format: yolox_raw\n"
        f"  architecture: yolox_s_locator_5cls\n"
        f"  weights_file: {mid}_best.onnx\n"
        "meta:\n"
        f"  version: {mid}\n"
        "  num_classes: 5\n"
        "metrics_at_chosen_threshold:\n"
        "  bird_recall: 0.99\n"
    )

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if url.endswith("_model_config.yaml"):
            with open(dest, "w") as f:
                f.write(s_yaml_body)
        else:
            with open(dest, "wb") as f:
                f.write(b"fake-fetched")
        return True

    with (
        patch(
            "utils.model_downloader.requests.get",
            return_value=_mock_remote(remote_payload),
        ),
        patch("utils.model_downloader._download_file", side_effect=fake_download),
    ):
        ensure_model_files(BASE_URL, str(od_dir), "weights_path", "labels_path")

    # The key assertion: model_metadata.json now exists AND reflects the YAML.
    metadata_path = od_dir / "model_metadata.json"
    assert metadata_path.exists(), "cold-start must produce model_metadata.json"
    meta = json.loads(metadata_path.read_text())
    assert meta["inference_thresholds"]["confidence"] == 0.30, meta
    assert meta["inference_thresholds"]["iou_nms"] == 0.50
    assert meta["variant"] == "s"


def test_ensure_model_files_overwrites_stale_metadata_from_previous_variant(tmp_path):
    """Regression guard for the stale-metadata half of the 2026-04-18 bug.

    If a previous boot left a ``model_metadata.json`` with Tiny's conf=0.15
    around, a cold-start that re-provisions S must NOT let the Tiny metadata
    stay. The regeneration runs whenever the YAML is on disk, even when the
    YAML was already there from a prior boot.
    """
    from utils.model_downloader import ensure_model_files

    od_dir = tmp_path / "object_detection"
    od_dir.mkdir()

    mid = "20260417_yolox_s_640_mosaic0p5"

    # Pre-existing stale metadata with Tiny's values (what the detector
    # would wrongly read on init without our fix).
    (od_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "framework": "yolox",
                "variant": "tiny",
                "inference_thresholds": {"confidence": 0.15, "iou_nms": 0.50},
            }
        )
    )

    # Pre-existing S YAML (simulates the previous session having pulled it).
    s_yaml = (
        "detection:\n"
        "  confidence_threshold: 0.30\n"
        "  nms_iou_threshold: 0.50\n"
        "  architecture: yolox_s_locator_5cls\n"
        "  weights_file: best.onnx\n"
        "meta:\n  num_classes: 5\n"
    )
    (od_dir / f"{mid}_model_config.yaml").write_text(s_yaml)

    # Pre-existing S weights + labels (so the download skips them via
    # "already present").
    (od_dir / f"{mid}_best.onnx").write_bytes(b"fake-weights")
    (od_dir / f"{mid}_labels.json").write_text(json.dumps(["bird"]))
    (od_dir / f"{mid}_metrics.json").write_text(json.dumps({"aggregate": {}}))

    remote_payload = {
        "latest": mid,
        "weights_path": f"object_detection/{mid}_best.onnx",
        "labels_path": f"object_detection/{mid}_labels.json",
    }

    with patch(
        "utils.model_downloader.requests.get",
        return_value=_mock_remote(remote_payload),
    ):
        ensure_model_files(BASE_URL, str(od_dir), "weights_path", "labels_path")

    # Metadata now reflects S, not the stale Tiny values.
    meta = json.loads((od_dir / "model_metadata.json").read_text())
    assert meta["inference_thresholds"]["confidence"] == 0.30
    assert meta["variant"] == "s"


def test_ensure_model_files_skips_metadata_regen_for_classifier(tmp_path):
    """The classifier ships a YAML (per the 2026-04-18 HF layout) but
    does not consume model_metadata.json at runtime. Writing one there
    would just litter the filesystem."""
    from utils.model_downloader import ensure_model_files

    cls_dir = tmp_path / "classifier"
    cls_dir.mkdir()

    mid = "20250817_213043"
    remote_payload = {
        "latest": mid,
        "weights_path": f"classifier/{mid}_best.onnx",
        "classes_path": f"classifier/{mid}_classes.txt",
    }

    def fake_download(url: str, dest: str, *args, **kwargs) -> bool:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if url.endswith("_model_config.yaml"):
            with open(dest, "w") as f:
                f.write(
                    "detection:\n"
                    "  confidence_threshold: null\n"
                    "  architecture: efficientnet_b3\n"
                    "  weights_file: best.onnx\n"
                    "meta:\n  num_classes: 29\n"
                )
        else:
            with open(dest, "wb") as f:
                f.write(b"fake")
        return True

    with (
        patch(
            "utils.model_downloader.requests.get",
            return_value=_mock_remote(remote_payload),
        ),
        patch("utils.model_downloader._download_file", side_effect=fake_download),
    ):
        ensure_model_files(BASE_URL, str(cls_dir), "weights_path", "classes_path")

    # No model_metadata.json in the classifier dir — that file is only
    # meaningful for the detector.
    assert not (cls_dir / "model_metadata.json").exists()


def test_prune_legacy_short_dict_labels_not_treated_as_legacy(tmp_path):
    """Dict-form labels.json with few entries (<20) is NOT classified as
    legacy. Guard against accidentally nuking a small custom model."""
    from utils.model_downloader import prune_legacy_fasterrcnn_models

    mid = "20260101_custom_2class"
    (tmp_path / f"{mid}_labels.json").write_text(json.dumps({"0": "bird", "1": "squirrel"}))
    (tmp_path / f"{mid}_best.onnx").write_bytes(b"custom")
    (tmp_path / "latest_models.json").write_text(
        json.dumps(
            {
                "latest": mid,
                "pinned_models": {
                    mid: {
                        "weights_path": f"object_detection/{mid}_best.onnx",
                        "labels_path": f"object_detection/{mid}_labels.json",
                    }
                },
            }
        )
    )
    assert prune_legacy_fasterrcnn_models(str(tmp_path)) == []
    assert (tmp_path / f"{mid}_best.onnx").exists()
