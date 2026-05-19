"""Tests for ETag-based content-freshness refresh of companion YAMLs.

When the model-dev re-pushes a bundle under the **same** ``model_id``
with changed companion content (the 2026-05-20 classifier N+1 case),
the App must detect the change and re-download the YAML. The old
existence-only skip path silently kept stale content.

This file covers four matrix cells from the fix-plan acceptance:

1. local present + HF unchanged (same ETag) -> no body download
2. local present + HF changed (new ETag)    -> body downloaded
3. local missing                            -> body downloaded (regression)
4. local present + HF changed + GET fails   -> local untouched (atomic)

Plus the WMB_FORCE_REMOTE_REFRESH env-var path that bypasses the
freshness sidecar entirely for a one-shot operator recovery.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import requests

from utils.model_downloader import (
    COMPANION_FRESHNESS_FILENAME,
    FORCE_REFRESH_ENV_VAR,
    _companion_is_fresh,
    _fetch_companion_files,
    _read_companion_freshness,
    _write_companion_freshness_entry,
)


def _seed_sidecar(model_dir: Path, basename: str, etag: str) -> None:
    """Pre-stage a freshness sidecar entry for *basename* with *etag*."""
    _write_companion_freshness_entry(str(model_dir), basename, etag, "deadbeef")


def _fake_head(status: int, etag: str | None):
    """Build a fake response object compatible with the head-path."""

    class FakeResp:
        status_code = status
        headers = {"ETag": etag} if etag is not None else {}

    return FakeResp()


# ---------------------------------------------------------------------------
# _companion_is_fresh: unit cells
# ---------------------------------------------------------------------------


def test_companion_is_fresh_etag_match_returns_true(tmp_path, monkeypatch):
    """Sidecar ETag matches remote ETag -> fresh (no body download)."""
    basename = "m_model_config.yaml"
    _seed_sidecar(tmp_path, basename, '"abc123"')
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"abc123"'),
    )
    assert (
        _companion_is_fresh("https://hf.example/" + basename, str(tmp_path), basename)
        is True
    )


def test_companion_is_fresh_304_returns_true(tmp_path, monkeypatch):
    """HF responds 304 Not Modified -> fresh."""
    basename = "m_model_config.yaml"
    _seed_sidecar(tmp_path, basename, '"abc123"')
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(304, None),
    )
    assert (
        _companion_is_fresh("https://hf.example/" + basename, str(tmp_path), basename)
        is True
    )


def test_companion_is_fresh_etag_changed_returns_false(tmp_path, monkeypatch):
    """Sidecar ETag stale -> not fresh (caller will redownload)."""
    basename = "m_model_config.yaml"
    _seed_sidecar(tmp_path, basename, '"old-etag"')
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"new-etag"'),
    )
    assert (
        _companion_is_fresh("https://hf.example/" + basename, str(tmp_path), basename)
        is False
    )


def test_companion_is_fresh_no_sidecar_treats_as_stale(tmp_path, monkeypatch):
    """No sidecar entry -> not fresh (caller will redownload to populate it)."""
    basename = "m_model_config.yaml"
    # no sidecar written
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"any-etag"'),
    )
    assert (
        _companion_is_fresh("https://hf.example/" + basename, str(tmp_path), basename)
        is False
    )


def test_companion_is_fresh_network_failure_is_safe(tmp_path, monkeypatch):
    """Transient network error -> treat as fresh (fail safe, no redownload storm)."""
    basename = "m_model_config.yaml"
    _seed_sidecar(tmp_path, basename, '"abc123"')

    def fail(*_, **__):
        raise requests.RequestException("simulated outage")

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr("utils.model_downloader.requests.head", fail)
    assert (
        _companion_is_fresh("https://hf.example/" + basename, str(tmp_path), basename)
        is True
    )


def test_companion_is_fresh_no_etag_header_treats_as_fresh(tmp_path, monkeypatch):
    """HF returns 200 but no ETag header -> treat as fresh (CDN misconfig defense)."""
    basename = "m_model_config.yaml"
    _seed_sidecar(tmp_path, basename, '"abc123"')
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, None),
    )
    assert (
        _companion_is_fresh("https://hf.example/" + basename, str(tmp_path), basename)
        is True
    )


# ---------------------------------------------------------------------------
# _fetch_companion_files: integration with freshness check
# ---------------------------------------------------------------------------


def test_fetch_companion_local_present_hf_unchanged_no_download(tmp_path, monkeypatch):
    """Cell 1: local present + HF unchanged -> no body download.

    The bug-fix's whole reason for existing: when content is genuinely
    unchanged, we still want the steady-state quiet behaviour of the
    original existence-only skip path.
    """
    yaml = tmp_path / "model_A_model_config.yaml"
    metrics = tmp_path / "model_A_metrics.json"
    yaml.write_text("local-cached-content")
    metrics.write_text("{}")
    _seed_sidecar(tmp_path, "model_A_model_config.yaml", '"matching-etag"')
    _seed_sidecar(tmp_path, "model_A_metrics.json", '"metrics-etag"')

    def head_for(url, **_):
        # HF returns matching ETag for both companions.
        if url.endswith("model_config.yaml"):
            return _fake_head(200, '"matching-etag"')
        return _fake_head(200, '"metrics-etag"')

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr("utils.model_downloader.requests.head", head_for)

    download_calls = []

    def fake_download(url, dest, **kw):
        download_calls.append((url, dest))
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    assert download_calls == [], (
        "Both ETags matched -> no body downloads expected. Got: "
        f"{[u for u, _ in download_calls]}"
    )
    assert yaml.read_text() == "local-cached-content"


def test_fetch_companion_local_present_hf_changed_downloads(tmp_path, monkeypatch):
    """Cell 2: local present + HF changed -> body downloaded.

    This is the regression of the 2026-05-20 bug: the model-dev re-pushed
    new content under the same model_id; the App must detect it and pull
    the new body.
    """
    yaml = tmp_path / "model_A_model_config.yaml"
    yaml.write_text("stale-pre-refit")
    _seed_sidecar(tmp_path, "model_A_model_config.yaml", '"old-etag"')

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"new-refit-etag"'),
    )

    download_calls = []

    def fake_download(url, dest, **kw):
        download_calls.append((url, dest))
        Path(dest).write_text("fresh-post-refit")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    yaml_calls = [c for c in download_calls if c[0].endswith("_model_config.yaml")]
    assert len(yaml_calls) == 1, (
        "ETag changed -> exactly one YAML body download expected. Got: "
        f"{[u for u, _ in download_calls]}"
    )
    assert yaml.read_text() == "fresh-post-refit"


def test_fetch_companion_local_missing_downloads(tmp_path, monkeypatch):
    """Cell 3: local missing -> body downloaded (regression test for the
    existing missing-file path)."""
    # No local file at start.
    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    # The HEAD path should not even be reached because the file is missing;
    # but if it were, return something harmless.
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"new"'),
    )

    download_calls = []

    def fake_download(url, dest, **kw):
        download_calls.append((url, dest))
        Path(dest).write_text("fetched-fresh")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    yaml_calls = [c for c in download_calls if c[0].endswith("_model_config.yaml")]
    assert len(yaml_calls) == 1
    yaml_path = tmp_path / "model_A_model_config.yaml"
    assert yaml_path.exists() and yaml_path.read_text() == "fetched-fresh"


def test_fetch_companion_changed_then_network_failure_keeps_local(
    tmp_path, monkeypatch
):
    """Cell 4: HF says changed, but the body GET then fails -> local stays
    intact thanks to atomic tmp+rename."""
    yaml = tmp_path / "model_A_model_config.yaml"
    yaml.write_text("known-good-pre-refit")
    _seed_sidecar(tmp_path, "model_A_model_config.yaml", '"old-etag"')

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"new-etag"'),
    )

    def failing_get(*_, **__):
        raise requests.RequestException("simulated mid-stream failure")

    monkeypatch.setattr("utils.model_downloader.requests.get", failing_get)

    _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    # Local file must be untouched
    assert yaml.read_text() == "known-good-pre-refit"
    # No leftover .tmp
    assert not (tmp_path / "model_A_model_config.yaml.tmp").exists()


# ---------------------------------------------------------------------------
# Sidecar persistence
# ---------------------------------------------------------------------------


def test_sidecar_persists_etag_after_download(tmp_path, monkeypatch):
    """After a successful download, the freshness sidecar carries the new
    ETag, so the next freshness check can skip on it."""
    yaml = tmp_path / "model_A_model_config.yaml"
    # local starts missing

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr(
        "utils.model_downloader.requests.head",
        lambda *a, **kw: _fake_head(200, '"persist-me"'),
    )

    def fake_download(url, dest, **kw):
        Path(dest).write_text("fresh")
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    sidecar = _read_companion_freshness(str(tmp_path))
    entry = sidecar.get("model_A_model_config.yaml")
    assert entry is not None, f"sidecar missing entry; got keys: {list(sidecar.keys())}"
    assert entry.get("etag") == '"persist-me"'
    # sha256 also recorded (non-empty hex string)
    assert entry.get("sha256")
    # fetched_at non-empty ISO string
    assert entry.get("fetched_at")


def test_sidecar_corrupt_treated_as_empty(tmp_path):
    """A garbage sidecar file does not crash freshness reads — it is
    treated as 'no cached signal' so the next HEAD goes out unconditionally."""
    (tmp_path / COMPANION_FRESHNESS_FILENAME).write_text("{ not json")
    assert _read_companion_freshness(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# WMB_FORCE_REMOTE_REFRESH env-var path (Slice 3 recovery)
# ---------------------------------------------------------------------------


def test_env_force_refresh_bypasses_sidecar(tmp_path, monkeypatch):
    """Operator recovery path: setting WMB_FORCE_REMOTE_REFRESH=1 forces
    a re-download of companions, ignoring the freshness sidecar.

    The intended workflow is: operator sets the env var, restarts the
    service; the next detector reload picks up the fresh content even
    if the sidecar still says 'fresh' (e.g. ETag never changed because
    of a CDN edge case)."""
    yaml = tmp_path / "model_A_model_config.yaml"
    yaml.write_text("pre-refresh")
    _seed_sidecar(tmp_path, "model_A_model_config.yaml", '"matching-etag"')

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    # If the freshness path were consulted, this HEAD would say "fresh"
    # and the download below would NOT happen. The env-var must skip
    # the freshness path entirely.
    head_calls = []

    def fake_head(*a, **kw):
        head_calls.append(a)
        return _fake_head(200, '"matching-etag"')

    monkeypatch.setattr("utils.model_downloader.requests.head", fake_head)

    download_calls = []

    def fake_download(url, dest, **kw):
        download_calls.append((url, dest, kw.get("force")))
        Path(dest).write_text("post-refresh")
        return True

    monkeypatch.setenv(FORCE_REFRESH_ENV_VAR, "1")
    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    # The env-var path must force the download even when the sidecar
    # would have said "fresh."
    yaml_calls = [c for c in download_calls if c[0].endswith("_model_config.yaml")]
    assert len(yaml_calls) == 1, (
        "WMB_FORCE_REMOTE_REFRESH should force exactly one download. Got: "
        f"{[u for u, _, _ in download_calls]}"
    )
    assert yaml_calls[0][2] is True, "Expected force=True on the _download_file call"
    assert yaml.read_text() == "post-refresh"


def test_env_force_refresh_off_keeps_freshness_path(tmp_path, monkeypatch):
    """Sanity-check the inverse: with WMB_FORCE_REMOTE_REFRESH unset and
    a matching sidecar, the freshness path skips the download."""
    yaml = tmp_path / "model_A_model_config.yaml"
    metrics = tmp_path / "model_A_metrics.json"
    yaml.write_text("steady-state")
    metrics.write_text("{}")
    _seed_sidecar(tmp_path, "model_A_model_config.yaml", '"steady-yaml"')
    _seed_sidecar(tmp_path, "model_A_metrics.json", '"steady-metrics"')

    def head_for(url, **_):
        if url.endswith("model_config.yaml"):
            return _fake_head(200, '"steady-yaml"')
        return _fake_head(200, '"steady-metrics"')

    monkeypatch.setattr("utils.model_downloader._safe_download_url", lambda u: u)
    monkeypatch.setattr("utils.model_downloader.requests.head", head_for)
    monkeypatch.delenv(FORCE_REFRESH_ENV_VAR, raising=False)

    download_calls = []

    def fake_download(url, dest, **kw):
        download_calls.append((url, dest))
        return True

    with patch("utils.model_downloader._download_file", side_effect=fake_download):
        _fetch_companion_files("https://hf.example/cls", str(tmp_path), "model_A")

    assert download_calls == [], "Steady-state: no downloads expected"
    assert yaml.read_text() == "steady-state"
