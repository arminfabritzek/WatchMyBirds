"""Tests for the UI-driven force-refresh of companion YAMLs.

When the operator clicks "Switch" on a variant in the Settings UI, the
detector pin endpoint regenerates model_metadata.json. Pre-2026-05-14
this read the locally-cached YAML, which silently masked any new
fields the model author had added to the HF YAML in the meantime
(e.g. ``suppressed_classes`` or ``confidence_threshold_per_class``).

Now the pin endpoint passes ``refresh_companions=True`` so the YAML +
metrics get re-pulled from HF before regeneration. The cold-start
autofetch path keeps ``force_refresh=False`` so a service reboot
doesn't hit HF for every variant whose cache is already valid.

These tests use monkeypatch on ``_download_file`` to avoid real
network. The atomic-write (tmp + rename) plus the failure semantics
(local cache survives a network error) are verified end-to-end.
"""

from __future__ import annotations

from unittest.mock import patch

from utils.model_downloader import _download_file, _fetch_companion_files


def _make_companion_paths(tmp_path, model_id):
    yaml = tmp_path / f"{model_id}_model_config.yaml"
    metrics = tmp_path / f"{model_id}_metrics.json"
    return yaml, metrics


# ---------------------------------------------------------------------------
# _download_file: force=True semantics
# ---------------------------------------------------------------------------


def test_download_file_default_skips_when_exists(tmp_path, monkeypatch):
    """``force=False`` (default) returns True without fetching if file exists."""
    dest = tmp_path / "x.yaml"
    dest.write_text("local-cached")
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    ok = _download_file(
        "https://huggingface.co/x.yaml", str(dest), base_dir=str(tmp_path)
    )
    assert ok is True
    assert dest.read_text() == "local-cached"  # unchanged


def test_download_file_force_overwrites_existing(tmp_path, monkeypatch):
    """``force=True`` re-fetches even when the file exists; result is the
    new content."""
    dest = tmp_path / "x.yaml"
    dest.write_text("stale-cached")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=8192):
            yield b"fresh-from-hf"

    fake_resp = FakeResponse()
    monkeypatch.setattr(
        "utils.model_downloader.requests.get",
        lambda *a, **kw: fake_resp,
    )
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    ok = _download_file(
        "https://huggingface.co/x.yaml",
        str(dest),
        base_dir=str(tmp_path),
        force=True,
    )
    assert ok is True
    assert dest.read_text() == "fresh-from-hf"


def test_download_file_force_network_error_keeps_local_intact(tmp_path, monkeypatch):
    """When force-refresh fails mid-stream, the existing local file stays
    intact thanks to the tmp+rename atomic write."""
    dest = tmp_path / "x.yaml"
    dest.write_text("known-good-cache")

    import requests as _req

    def fake_get(*a, **kw):
        raise _req.RequestException("simulated network error")

    monkeypatch.setattr("utils.model_downloader.requests.get", fake_get)
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    ok = _download_file(
        "https://huggingface.co/x.yaml",
        str(dest),
        base_dir=str(tmp_path),
        force=True,
        retries=1,
    )
    assert ok is False
    # Existing cache must NOT have been clobbered by the failed tmp write
    assert dest.exists()
    assert dest.read_text() == "known-good-cache"
    # No leftover .tmp file
    assert not (tmp_path / "x.yaml.tmp").exists()


# ---------------------------------------------------------------------------
# _fetch_companion_files: force_refresh propagation
# ---------------------------------------------------------------------------


def test_fetch_companion_default_skips_when_exists(tmp_path):
    """Default cold-start path: existing YAML / metrics are NOT re-fetched."""
    yaml, metrics = _make_companion_paths(tmp_path, "model_A")
    yaml.write_text("stale-yaml")
    metrics.write_text("{}")

    calls = []

    def fake_download(url, dest, **kw):
        calls.append((url, dest, kw))
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/od", str(tmp_path), "model_A")

    # Both files exist; no download calls
    assert calls == []
    assert yaml.read_text() == "stale-yaml"


def test_fetch_companion_force_refresh_overwrites(tmp_path):
    """``force_refresh=True`` triggers a fresh download even when the
    local file already exists."""
    yaml, metrics = _make_companion_paths(tmp_path, "model_B")
    yaml.write_text("stale-yaml")
    metrics.write_text("{}")

    calls = []

    def fake_download(url, dest, **kw):
        calls.append((url, dest, kw))
        # Simulate a successful fresh fetch
        with open(dest, "w") as fh:
            if dest.endswith(".yaml"):
                fh.write("fresh-yaml-with-suppressed-classes")
            else:
                fh.write('{"fresh": true}')
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files(
            "https://hf.example/od", str(tmp_path), "model_B", force_refresh=True
        )

    # Two _download_file calls with force=True
    assert len(calls) == 2
    for _url, _dest, kwargs in calls:
        assert kwargs.get("force") is True
    assert yaml.read_text() == "fresh-yaml-with-suppressed-classes"
    assert metrics.read_text() == '{"fresh": true}'


def test_fetch_companion_force_refresh_keyword_only(tmp_path):
    """``force_refresh`` is keyword-only — positional calls must fail.

    Guards against accidental misuse where a caller might pass a 4th
    positional arg thinking it's something else.
    """
    import inspect

    sig = inspect.signature(_fetch_companion_files)
    param = sig.parameters["force_refresh"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


def test_fetch_companion_force_refresh_failure_keeps_local(tmp_path, monkeypatch):
    """Force-refresh + network error → existing local cache stays alive
    (end-to-end atomic-write semantics through _fetch_companion_files)."""
    yaml, _metrics = _make_companion_paths(tmp_path, "model_C")
    yaml.write_text("known-good-cache")

    import requests as _req

    def fake_get(*a, **kw):
        raise _req.RequestException("simulated")

    monkeypatch.setattr("utils.model_downloader.requests.get", fake_get)
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)

    _fetch_companion_files(
        "https://hf.example/od", str(tmp_path), "model_C", force_refresh=True
    )

    assert yaml.exists()
    assert yaml.read_text() == "known-good-cache"


# ---------------------------------------------------------------------------
# api_v1._regenerate_metadata_for_variant: refresh_companions propagation
# ---------------------------------------------------------------------------


def test_regenerate_metadata_for_variant_force_refreshes_companions(
    tmp_path, monkeypatch
):
    """The detector pin endpoint sets ``refresh_companions=True`` so
    ``_regenerate_metadata_for_variant`` calls ``_fetch_companion_files``
    with the force flag before reading the local YAML."""
    from web.blueprints import api_v1 as api_mod

    model_id = "20260513_yolox_s_locator_640_mosaic0p75_v2_coco"
    # Write a "fresh" YAML that the function will read
    yaml_path = tmp_path / f"{model_id}_model_config.yaml"
    yaml_path.write_text(
        "detection:\n"
        "  confidence_threshold: 0.30\n"
        "  nms_iou_threshold: 0.5\n"
        "  input_size: [640, 640]\n"
        "  architecture: yolox_s_locator_v2_6cls\n"
        "  suppressed_classes:\n"
        "    - person\n"
        "meta:\n"
        "  num_classes: 6\n"
        "  classes: [bird, squirrel, cat, marten_mustelid, hedgehog, person]\n"
    )

    fetch_calls = []

    def fake_fetch(base_url, model_dir, mid, *, force_refresh=False):
        fetch_calls.append({"base_url": base_url, "mid": mid, "force": force_refresh})

    monkeypatch.setattr("utils.model_downloader._fetch_companion_files", fake_fetch)

    out = api_mod._regenerate_metadata_for_variant(
        str(tmp_path), model_id, refresh_companions=True
    )

    assert out is not None
    assert len(fetch_calls) == 1
    assert fetch_calls[0]["force"] is True
    assert fetch_calls[0]["mid"] == model_id

    # And the metadata that landed actually carries the new field
    import json

    metadata = json.loads((tmp_path / "model_metadata.json").read_text())
    assert metadata["inference_thresholds"]["suppressed_classes"] == ["person"]


def test_regenerate_metadata_for_variant_cold_start_does_not_refresh(
    tmp_path, monkeypatch
):
    """The cold-start path (no UI click) keeps ``refresh_companions=False``
    so reboots don't hit HF when the local cache is already valid."""
    from web.blueprints import api_v1 as api_mod

    model_id = "abc"
    (tmp_path / f"{model_id}_model_config.yaml").write_text(
        "detection: {confidence_threshold: 0.3, input_size: [640, 640]}\n"
        "meta: {num_classes: 5}\n"
    )

    fetch_calls = []

    def fake_fetch(*a, **kw):
        fetch_calls.append(kw)

    monkeypatch.setattr("utils.model_downloader._fetch_companion_files", fake_fetch)

    api_mod._regenerate_metadata_for_variant(str(tmp_path), model_id)

    assert fetch_calls == []  # default refresh_companions=False, no fetch
