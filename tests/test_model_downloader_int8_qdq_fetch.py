"""Tests for the INT8/QDQ weight bundle fetch in utils.model_downloader.

The companion-fetch path was previously fp32-only — INT8/QDQ files
existed on HuggingFace but were never pulled to disk, leaving the
settings UI int8 chip permanently disabled. These tests pin the new
behaviour: the QDQ primary + every declared fallback is downloaded
when the operator pins a detector variant, and pre-existing files
are left alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.model_downloader import (
    _fetch_int8_qdq_weights,
    _prefetch_all_int8_qdq_weights,
)

BASE_URL = "https://hf.example/object_detection"


@pytest.fixture(autouse=True)
def _allow_test_host(monkeypatch):
    monkeypatch.setenv(
        "WMB_ALLOWED_DOWNLOAD_HOSTS",
        "huggingface.co,cdn-lfs.huggingface.co,hf.example",
    )


def _write_local(cache_dir: Path, payload: dict) -> None:
    (cache_dir / "latest_models.json").write_text(json.dumps(payload))


def test_fetch_int8_qdq_pulls_primary_and_all_fallbacks(tmp_path: Path):
    """Primary QDQ + every declared fallback is downloaded once."""
    model_id = "20260521-1930_yolox_s_locator_640_mosaic0p75_v3_coco"
    _write_local(
        tmp_path,
        {
            "pinned_models": {
                model_id: {
                    "weights_int8_qdq_path": (
                        f"object_detection/{model_id}_best_int8_qdq.onnx"
                    ),
                    "weights_int8_qdq_fallback_paths": [
                        f"object_detection/{model_id}_best_int8_qdq.onnx",
                        f"object_detection/{model_id}_best_int8_qdq_pt.onnx",
                        f"object_detection/{model_id}_best_int8_qdq_u8a.onnx",
                    ],
                }
            }
        },
    )

    downloaded: list[str] = []

    def fake_download(url, dest, **_):
        downloaded.append(url)
        Path(dest).write_bytes(b"qdq")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_int8_qdq_weights(BASE_URL, str(tmp_path), model_id)

    # Primary + 2 distinct fallbacks (the dedupe drops the repeat of primary).
    assert sorted(Path(p).name for p in downloaded) == [
        f"{model_id}_best_int8_qdq.onnx",
        f"{model_id}_best_int8_qdq_pt.onnx",
        f"{model_id}_best_int8_qdq_u8a.onnx",
    ]
    for name in (
        f"{model_id}_best_int8_qdq.onnx",
        f"{model_id}_best_int8_qdq_pt.onnx",
        f"{model_id}_best_int8_qdq_u8a.onnx",
    ):
        assert (tmp_path / name).exists()


def test_fetch_int8_qdq_skips_files_already_on_disk(tmp_path: Path):
    """Pre-existing files are not re-downloaded (idempotent)."""
    model_id = "v3_coco"
    primary = f"object_detection/{model_id}_best_int8_qdq.onnx"
    fallback = f"object_detection/{model_id}_best_int8_qdq_pt.onnx"
    _write_local(
        tmp_path,
        {
            "pinned_models": {
                model_id: {
                    "weights_int8_qdq_path": primary,
                    "weights_int8_qdq_fallback_paths": [primary, fallback],
                }
            }
        },
    )
    # Pre-stage the primary on disk; only the fallback should be fetched.
    (tmp_path / Path(primary).name).write_bytes(b"existing")

    downloaded: list[str] = []

    def fake_download(url, dest, **_):
        downloaded.append(url)
        Path(dest).write_bytes(b"qdq")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_int8_qdq_weights(BASE_URL, str(tmp_path), model_id)

    assert downloaded == [f"{BASE_URL}/{model_id}_best_int8_qdq_pt.onnx"]
    assert (tmp_path / f"{model_id}_best_int8_qdq.onnx").read_bytes() == b"existing"


def test_fetch_int8_qdq_skips_variant_without_qdq_paths(tmp_path: Path):
    """Older variants (only weights_int8_path, no _qdq) are skipped cleanly.

    The loader's precision plan reads only ``weights_int8_qdq_*`` fields,
    so pulling a bare ``_int8.onnx`` would cost bytes for no behavioural
    effect. Verify the helper short-circuits with no downloads.
    """
    model_id = "v10_balanced"
    _write_local(
        tmp_path,
        {
            "pinned_models": {
                model_id: {
                    "weights_int8_path": (
                        f"object_detection/{model_id}_best_int8.onnx"
                    ),
                }
            }
        },
    )

    with patch("utils.model_downloader._download_file") as mock_dl:
        _fetch_int8_qdq_weights(BASE_URL, str(tmp_path), model_id)

    mock_dl.assert_not_called()


def test_fetch_int8_qdq_tolerates_individual_404(tmp_path: Path):
    """A 404 on a fallback does not abort the loop — other candidates run."""
    model_id = "v3_coco"
    _write_local(
        tmp_path,
        {
            "pinned_models": {
                model_id: {
                    "weights_int8_qdq_path": (
                        f"object_detection/{model_id}_best_int8_qdq.onnx"
                    ),
                    "weights_int8_qdq_fallback_paths": [
                        f"object_detection/{model_id}_best_int8_qdq.onnx",
                        f"object_detection/{model_id}_best_int8_qdq_pt.onnx",
                        f"object_detection/{model_id}_best_int8_qdq_u8a_pt.onnx",
                    ],
                }
            }
        },
    )

    def fake_download(url, dest, **_):
        if url.endswith("_pt.onnx") and "u8a" not in url:
            return False  # simulate 404 on the _pt fallback
        Path(dest).write_bytes(b"qdq")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_int8_qdq_weights(BASE_URL, str(tmp_path), model_id)

    assert (tmp_path / f"{model_id}_best_int8_qdq.onnx").exists()
    assert (tmp_path / f"{model_id}_best_int8_qdq_u8a_pt.onnx").exists()
    assert not (tmp_path / f"{model_id}_best_int8_qdq_pt.onnx").exists()


def test_fetch_int8_qdq_no_local_manifest_is_noop(tmp_path: Path):
    """Missing latest_models.json short-circuits without raising."""
    with patch("utils.model_downloader._download_file") as mock_dl:
        _fetch_int8_qdq_weights(BASE_URL, str(tmp_path), "any_id")
    mock_dl.assert_not_called()


def test_fetch_int8_qdq_unknown_variant_is_noop(tmp_path: Path):
    """latest_models.json present but the model_id is not in pinned_models."""
    _write_local(tmp_path, {"pinned_models": {"other_id": {}}})

    with patch("utils.model_downloader._download_file") as mock_dl:
        _fetch_int8_qdq_weights(BASE_URL, str(tmp_path), "missing_id")

    mock_dl.assert_not_called()


def test_prefetch_all_int8_qdq_iterates_pinned_models(tmp_path: Path):
    """Cold-start helper pre-fetches QDQ bundles for every pinned variant.

    The detector-lane guard is satisfied by naming the directory
    ``object_detection`` (mirrors the real cache layout).
    """
    cache_dir = tmp_path / "object_detection"
    cache_dir.mkdir()
    _write_local(
        cache_dir,
        {
            "pinned_models": {
                "v3_a": {
                    "weights_int8_qdq_path": "object_detection/v3_a_best_int8_qdq.onnx",
                },
                "v3_b": {
                    "weights_int8_qdq_path": "object_detection/v3_b_best_int8_qdq.onnx",
                    "weights_int8_qdq_fallback_paths": [
                        "object_detection/v3_b_best_int8_qdq.onnx",
                        "object_detection/v3_b_best_int8_qdq_pt.onnx",
                    ],
                },
                # Pre-QDQ variant — must be skipped without raising.
                "old": {
                    "weights_int8_path": "object_detection/old_best_int8.onnx",
                },
            }
        },
    )

    fetched: list[str] = []

    def fake_download(_url, dest, **_kwargs):
        fetched.append(Path(dest).name)
        Path(dest).write_bytes(b"qdq")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _prefetch_all_int8_qdq_weights(BASE_URL, str(cache_dir))

    assert sorted(fetched) == [
        "v3_a_best_int8_qdq.onnx",
        "v3_b_best_int8_qdq.onnx",
        "v3_b_best_int8_qdq_pt.onnx",
    ]


def test_prefetch_all_int8_qdq_skips_non_detector_lane(tmp_path: Path):
    """Classifier lane has no QDQ bundle — helper bails out early.

    The detector-lane guard is keyed off the cache-dir basename, so
    naming the dir ``classifier`` (matches the real layout) must short-
    circuit the helper before any download is attempted, even when the
    manifest accidentally contains QDQ paths.
    """
    cache_dir = tmp_path / "classifier"
    cache_dir.mkdir()
    _write_local(
        cache_dir,
        {
            "pinned_models": {
                "cls_v1": {
                    "weights_int8_qdq_path": ("classifier/cls_v1_best_int8_qdq.onnx"),
                }
            }
        },
    )

    with patch("utils.model_downloader._download_file") as mock_dl:
        _prefetch_all_int8_qdq_weights(BASE_URL, str(cache_dir))

    mock_dl.assert_not_called()


def test_prefetch_all_int8_qdq_tolerates_per_variant_exception(tmp_path: Path):
    """A crash on one variant must not abort the iteration."""
    cache_dir = tmp_path / "object_detection"
    cache_dir.mkdir()
    _write_local(
        cache_dir,
        {
            "pinned_models": {
                "v_bad": {
                    "weights_int8_qdq_path": "object_detection/v_bad_best_int8_qdq.onnx",
                },
                "v_good": {
                    "weights_int8_qdq_path": "object_detection/v_good_best_int8_qdq.onnx",
                },
            }
        },
    )

    fetched: list[str] = []

    def fake_download(url, dest, **_):
        if "v_bad" in url:
            raise RuntimeError("simulated network blow-up")
        fetched.append(Path(dest).name)
        Path(dest).write_bytes(b"qdq")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _prefetch_all_int8_qdq_weights(BASE_URL, str(cache_dir))

    assert fetched == ["v_good_best_int8_qdq.onnx"]
