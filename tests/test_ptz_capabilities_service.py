"""Tests for the PTZ capabilities service + the core probe_capabilities.

PTZ capability-probe service tests.

Three test groups:

1. **Decoding** — ``PtzClient.get_capabilities`` correctly extracts the
   declared union from canned ``GetServiceCapabilities`` + ``GetNodes``
   responses (the on-the-wire shape ``onvif-zeep`` returns).
2. **Caching** — ``probe_capabilities`` re-uses cached results within
   the TTL and bypasses the cache with ``force_refresh=True``.
3. **Service + route** — the thin ``web/services/ptz_capabilities_service``
   wrapper delegates correctly, and the unknown-camera path raises
   ``ValueError`` so the Flask route can return 404.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from camera.ptz_client import PtzClient
from core import ptz_core
from web.services import ptz_capabilities_service

# ---------------------------------------------------------------------------
# Helpers: canned ONVIF response shapes
# ---------------------------------------------------------------------------


class _Attr:
    """Tiny stand-in for zeep's attribute-access objects.

    ONVIF responses from ``onvif-zeep`` mix dict-like and attr-like
    access depending on the WSDL. ``camera.ptz_client._get_value``
    handles both — we exercise the attr-style here, since the dict
    fallback is exercised by the existing PTZ tests.
    """

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_node(
    *,
    token: str = "PTZ-Node-Token",
    name: str = "PTZ-Node",
    max_presets: int = 32,
    home: bool = False,
    cont_pt: bool = True,
    cont_zoom: bool = True,
    rel_pt: bool = True,
    rel_zoom: bool = True,
    abs_pt: bool = True,
    abs_zoom: bool = True,
) -> _Attr:
    """Build a single GetNodes-style node with the requested spaces.

    Spaces are encoded as 1-element lists when supported and empty
    lists when not — that's what zeep returns; ``get_capabilities``
    coerces both to bool.
    """

    def space(supported: bool) -> list[dict[str, Any]]:
        return [{"URI": "x"}] if supported else []

    return _Attr(
        token=token,
        Name=name,
        SupportedPTZSpaces=_Attr(
            AbsolutePanTiltPositionSpace=space(abs_pt),
            AbsoluteZoomPositionSpace=space(abs_zoom),
            RelativePanTiltTranslationSpace=space(rel_pt),
            RelativeZoomTranslationSpace=space(rel_zoom),
            ContinuousPanTiltVelocitySpace=space(cont_pt),
            ContinuousZoomVelocitySpace=space(cont_zoom),
        ),
        MaximumNumberOfPresets=max_presets,
        HomeSupported=home,
    )


def _fake_ptz_service(
    *,
    service_caps: _Attr | None = None,
    nodes: list[_Attr] | None = None,
    raise_on_service_caps: Exception | None = None,
    raise_on_nodes: Exception | None = None,
) -> MagicMock:
    """Build a mock that mimics what PtzClient._ensure_services returns.

    The mock has the two methods PtzClient.get_capabilities actually
    calls: ``GetServiceCapabilities()`` and ``GetNodes()``.
    """
    svc = MagicMock()
    if raise_on_service_caps:
        svc.GetServiceCapabilities.side_effect = raise_on_service_caps
    else:
        svc.GetServiceCapabilities.return_value = service_caps or _Attr(
            MoveStatus=False,
            StatusPosition=False,
            EFlip=False,
            Reverse=False,
        )
    if raise_on_nodes:
        svc.GetNodes.side_effect = raise_on_nodes
    else:
        svc.GetNodes.return_value = nodes if nodes is not None else [_make_node()]
    return svc


# ---------------------------------------------------------------------------
# Decoding tests — PtzClient.get_capabilities
# ---------------------------------------------------------------------------


def _patched_client(svc_mock: MagicMock) -> PtzClient:
    """Build a PtzClient with ``_ensure_services`` short-circuited."""
    client = PtzClient(ip="0.0.0.0", port=80, username="u", password="p")
    client._ensure_services = lambda: (svc_mock, "ProfileToken")  # type: ignore[method-assign]
    return client


def test_get_capabilities_full_support():
    """When every space is populated and Home is supported, the declared
    union mirrors the inputs and max_presets reflects the node."""
    svc = _fake_ptz_service(
        service_caps=_Attr(
            MoveStatus=True, StatusPosition=True, EFlip=False, Reverse=False
        ),
        nodes=[_make_node(max_presets=64, home=True)],
    )
    client = _patched_client(svc)

    result = client.get_capabilities()

    assert result["declared"]["continuous_pan_tilt"] is True
    assert result["declared"]["continuous_zoom"] is True
    assert result["declared"]["relative_pan_tilt"] is True
    assert result["declared"]["relative_zoom"] is True
    assert result["declared"]["absolute_pan_tilt"] is True
    assert result["declared"]["absolute_zoom"] is True
    assert result["declared"]["home_position"] is True
    assert result["declared"]["move_status"] is True
    assert result["declared"]["status_position"] is True
    assert result["declared"]["max_presets"] == 64
    assert result["error"] is None
    assert len(result["nodes"]) == 1


def test_get_capabilities_continuous_only_cam():
    """The probe cam from the actual probe run on 2026-05-17: declares
    continuous + relative + absolute via GetNodes but MoveStatus and
    StatusPosition are False at the service-caps level."""
    svc = _fake_ptz_service(
        service_caps=_Attr(
            MoveStatus=False,
            StatusPosition=False,
            EFlip=False,
            Reverse=False,
        ),
        nodes=[_make_node(max_presets=32, home=False)],
    )
    client = _patched_client(svc)
    result = client.get_capabilities()

    assert result["declared"]["move_status"] is False
    assert result["declared"]["status_position"] is False
    assert result["declared"]["max_presets"] == 32
    # GetNodes declares continuous, even though empirically only
    # Continuous works on this firmware — the empirical bit comes
    # from the standalone probe tool, not from this read-only function.
    assert result["declared"]["continuous_pan_tilt"] is True


def test_get_capabilities_union_across_multiple_nodes():
    """When a cam reports two nodes with partial spaces, the declared
    union should be True if ANY node has that capability."""
    node_a = _make_node(
        cont_pt=True, cont_zoom=False, rel_pt=False, abs_pt=False, abs_zoom=False,
        rel_zoom=False, max_presets=8,
    )
    node_b = _make_node(
        cont_pt=False, cont_zoom=True, rel_pt=True, abs_pt=False, abs_zoom=False,
        rel_zoom=False, max_presets=16,
    )
    svc = _fake_ptz_service(nodes=[node_a, node_b])
    client = _patched_client(svc)

    result = client.get_capabilities()

    assert result["declared"]["continuous_pan_tilt"] is True   # from node_a
    assert result["declared"]["continuous_zoom"] is True       # from node_b
    assert result["declared"]["relative_pan_tilt"] is True     # from node_b
    assert result["declared"]["absolute_pan_tilt"] is False    # neither
    assert result["declared"]["max_presets"] == 16             # max across nodes
    assert len(result["nodes"]) == 2


def test_get_capabilities_tolerates_service_caps_error():
    """If GetServiceCapabilities raises but GetNodes works, the result
    is partial — declared union is still populated from GetNodes."""
    svc = _fake_ptz_service(
        raise_on_service_caps=RuntimeError("SOAP fault"),
        nodes=[_make_node()],
    )
    client = _patched_client(svc)

    result = client.get_capabilities()

    assert result["service_capabilities"] is None
    assert result["error"] is not None
    assert "service_capabilities" in (result["error"] or "")
    # But the node-based declared bits are still there.
    assert result["declared"]["continuous_pan_tilt"] is True


def test_get_capabilities_tolerates_nodes_error():
    """If GetNodes raises but GetServiceCapabilities works, declared
    bits from the service caps still surface."""
    svc = _fake_ptz_service(
        service_caps=_Attr(
            MoveStatus=True, StatusPosition=True, EFlip=False, Reverse=False
        ),
        raise_on_nodes=RuntimeError("malformed WSDL"),
    )
    client = _patched_client(svc)

    result = client.get_capabilities()

    assert result["declared"]["move_status"] is True
    assert result["declared"]["status_position"] is True
    # node-derived bits stay False (no nodes parsed).
    assert result["declared"]["continuous_pan_tilt"] is False
    assert result["error"] is not None
    assert "nodes" in (result["error"] or "")


# ---------------------------------------------------------------------------
# Caching tests — core.ptz_core.probe_capabilities
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    """Each test starts with an empty capability cache."""
    ptz_core.clear_capabilities_cache()
    yield
    ptz_core.clear_capabilities_cache()


def _stub_camera_lookup(camera_id: int = 7, ip: str = "10.0.0.42") -> dict[str, Any]:
    return {
        "id": camera_id,
        "ip": ip,
        "port": 80,
        "username": "admin",
        "password": "",  # include_password=False at the storage layer
        "ptz": {},
    }


def test_probe_capabilities_uses_cache_within_ttl():
    """A second call within the TTL returns the cached dict and never
    builds a new client."""
    fake_client = MagicMock()
    fake_client.get_capabilities.return_value = {
        "service_capabilities": None,
        "nodes": [],
        "declared": {"continuous_pan_tilt": True, "max_presets": 32},
        "error": None,
    }

    storage = MagicMock()
    storage.get_camera.return_value = _stub_camera_lookup()

    with patch.object(ptz_core, "get_camera_storage", return_value=storage), patch.object(
        ptz_core, "_client_for_camera", return_value=fake_client
    ):
        first = ptz_core.probe_capabilities(7)
        second = ptz_core.probe_capabilities(7)

    assert first["from_cache"] is False
    assert second["from_cache"] is True
    # The client was created exactly once — cache hit must NOT
    # re-build the PtzClient (which would re-issue ONVIF calls).
    fake_client.get_capabilities.assert_called_once()


def test_probe_capabilities_force_refresh_bypasses_cache():
    """``force_refresh=True`` issues a fresh probe even if a cached
    entry is still warm."""
    fake_client = MagicMock()
    fake_client.get_capabilities.return_value = {
        "service_capabilities": None,
        "nodes": [],
        "declared": {"continuous_pan_tilt": True, "max_presets": 32},
        "error": None,
    }
    storage = MagicMock()
    storage.get_camera.return_value = _stub_camera_lookup()

    with patch.object(ptz_core, "get_camera_storage", return_value=storage), patch.object(
        ptz_core, "_client_for_camera", return_value=fake_client
    ):
        ptz_core.probe_capabilities(7)
        ptz_core.probe_capabilities(7, force_refresh=True)

    assert fake_client.get_capabilities.call_count == 2


def test_probe_capabilities_unknown_camera_raises_value_error():
    """A bogus camera_id surfaces as ValueError so the Flask route can
    return 404 cleanly."""
    storage = MagicMock()
    storage.get_camera.return_value = None
    with patch.object(ptz_core, "get_camera_storage", return_value=storage):
        with pytest.raises(ValueError):
            ptz_core.probe_capabilities(999)


def test_probe_capabilities_expired_cache_re_probes():
    """When the cached entry is older than the TTL, the next call
    fetches fresh data."""
    fake_client = MagicMock()
    fake_client.get_capabilities.return_value = {
        "service_capabilities": None,
        "nodes": [],
        "declared": {"continuous_pan_tilt": True, "max_presets": 32},
        "error": None,
    }
    storage = MagicMock()
    storage.get_camera.return_value = _stub_camera_lookup()

    with patch.object(ptz_core, "get_camera_storage", return_value=storage), patch.object(
        ptz_core, "_client_for_camera", return_value=fake_client
    ):
        ptz_core.probe_capabilities(7)
        # Reach inside and age the cached entry past the TTL.
        with ptz_core._capabilities_cache_lock:
            ts, payload = ptz_core._capabilities_cache[7]
            ptz_core._capabilities_cache[7] = (
                ts - (ptz_core._CAPABILITIES_CACHE_TTL_SEC + 1.0),
                payload,
            )
        ptz_core.probe_capabilities(7)

    assert fake_client.get_capabilities.call_count == 2


# ---------------------------------------------------------------------------
# Service wrapper tests
# ---------------------------------------------------------------------------


def test_service_delegates_to_core_probe():
    """The web-services wrapper is a one-liner; this just confirms it
    actually forwards to core."""
    payload = {
        "camera_id": 7,
        "probed_at": time.time(),
        "from_cache": False,
        "ip": "10.0.0.42",
        "declared": {"continuous_pan_tilt": True},
        "service_capabilities": None,
        "nodes": [],
        "error": None,
    }
    with patch.object(ptz_core, "probe_capabilities", return_value=payload) as mock:
        out = ptz_capabilities_service.probe_capabilities(7, force_refresh=True)

    mock.assert_called_once_with(7, force_refresh=True)
    assert out is payload


def test_service_clear_cache_delegates():
    with patch.object(ptz_core, "clear_capabilities_cache") as mock:
        ptz_capabilities_service.clear_cache(7)
    mock.assert_called_once_with(7)


# ---------------------------------------------------------------------------
# Empirical-loader tests — core.ptz_core._load_empirical_from_disk
# ---------------------------------------------------------------------------
#
# The probe tool writes OUTPUT_DIR/ptz_capabilities/cam<id>.yaml; the core
# reads from exactly that path. No lab-scanning, no IP-matching — the
# probe is now the authoritative writer.


def _write_empirical_cache(
    output_dir: Any,
    camera_id: int,
    *,
    ip: str = "10.0.0.42",
    continuous_works: bool = True,
    relative_works: bool = False,
    absolute_works: bool = False,
    movestatus_transitions: bool = False,
    recommended_strategy: str = "continuous_pulse",
    probed_at: str = "20260517_235927",
    empirical_override: dict[str, Any] | None = None,
) -> Any:
    """Write a minimal empirical cache file in the shape WMB's probe-tool
    will produce. ``empirical_override`` lets the caller drop in an empty
    dict to exercise the "no empirical block" branch."""
    import yaml

    cap_dir = output_dir / "ptz_capabilities"
    cap_dir.mkdir(parents=True, exist_ok=True)

    if empirical_override is not None:
        empirical = empirical_override
    else:
        empirical = {
            "continuous_works": continuous_works,
            "relative_works": relative_works,
            "absolute_works": absolute_works,
            "movestatus_transitions": movestatus_transitions,
        }

    payload = {
        "camera_id": int(camera_id),
        "probed_at": probed_at,
        "connection": {"ip": ip},
        "empirical": empirical,
        "recommended_strategy": recommended_strategy,
    }
    target = cap_dir / f"cam{int(camera_id)}.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return target


def _fake_pm_for(tmp_path: Any) -> MagicMock:
    """Return a MagicMock PathManager whose get_ptz_capabilities_path
    resolves under tmp_path/ptz_capabilities/cam<id>.yaml."""
    fake_pm = MagicMock()

    def _path_for(cam_id: int) -> Any:
        cap_dir = tmp_path / "ptz_capabilities"
        cap_dir.mkdir(parents=True, exist_ok=True)
        return cap_dir / f"cam{int(cam_id)}.yaml"

    fake_pm.get_ptz_capabilities_path.side_effect = _path_for
    return fake_pm


def test_load_empirical_returns_none_when_no_file(tmp_path, monkeypatch):
    """No cache file → loader returns None (yellow pills in UI)."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    assert ptz_core._load_empirical_from_disk(7) is None


def test_load_empirical_returns_none_when_pathmanager_unavailable(monkeypatch):
    """A broken PathManager (e.g., import error in tests) must yield
    None, not raise — capability pills are best-effort UI sugar."""
    def _boom() -> MagicMock:
        raise RuntimeError("path manager not initialised")

    monkeypatch.setattr(ptz_core, "get_path_manager", _boom)
    assert ptz_core._load_empirical_from_disk(7) is None


def test_load_empirical_reads_canonical_file(tmp_path, monkeypatch):
    """The exact scenario after Run 3 of the probe on 2026-05-17:
    cam0's file says relative/absolute broken, continuous works."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_empirical_cache(
        tmp_path,
        camera_id=0,
        ip="192.168.1.100",
        continuous_works=True,
        relative_works=False,
        absolute_works=False,
        movestatus_transitions=False,
        recommended_strategy="continuous_pulse",
    )

    result = ptz_core._load_empirical_from_disk(0)

    assert result is not None
    assert result["continuous_works"] is True
    assert result["relative_works"] is False
    assert result["absolute_works"] is False
    assert result["movestatus_transitions"] is False
    assert result["recommended_strategy"] == "continuous_pulse"
    assert result["report_timestamp"] == "20260517_235927"


def test_load_empirical_per_camera_isolation(tmp_path, monkeypatch):
    """cam0's file must not bleed into cam1's lookup."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_empirical_cache(tmp_path, camera_id=0, relative_works=False)

    assert ptz_core._load_empirical_from_disk(0) is not None
    assert ptz_core._load_empirical_from_disk(1) is None


def test_load_empirical_tolerates_malformed_yaml(tmp_path, monkeypatch):
    """If a hand-edit corrupts the cache file, the loader returns None
    rather than crashing the Settings page."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    cap_dir = tmp_path / "ptz_capabilities"
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / "cam7.yaml").write_text(
        "not: valid: yaml: at: all: :\n  - [\n", encoding="utf-8"
    )
    assert ptz_core._load_empirical_from_disk(7) is None


def test_load_empirical_forwards_follow_zoom_max_burst_sec(tmp_path, monkeypatch):
    """The near-focus zoom-budget field written by the in-UI probe
    wizard's finalize must round-trip through the loader.

    Regression for a read-side read-side miss: the loader's strict
    whitelist was dropping the new key silently, so apply_near_focus_budget
    saw an empirical dict without the field and reported 'No near-focus
    budget recorded yet — run the wizard's near-focus step first' even
    on cams that HAD been probed.
    """
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_empirical_cache(
        tmp_path,
        camera_id=0,
        empirical_override={
            "continuous_works": True,
            "relative_works": False,
            "absolute_works": False,
            "movestatus_transitions": False,
            "follow_zoom_max_burst_sec": 0.75,
        },
    )

    result = ptz_core._load_empirical_from_disk(0)

    assert result is not None
    assert result["follow_zoom_max_burst_sec"] == 0.75


def test_load_empirical_omits_follow_zoom_when_absent(tmp_path, monkeypatch):
    """Cams probed before read-side have no follow_zoom_max_burst_sec
    in their cache. The loader must not insert a default — absence is
    meaningful (apply path surfaces 'run the step first')."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_empirical_cache(tmp_path, camera_id=0, continuous_works=True)

    result = ptz_core._load_empirical_from_disk(0)

    assert result is not None
    assert "follow_zoom_max_burst_sec" not in result


def test_load_empirical_returns_none_when_empirical_key_missing(tmp_path, monkeypatch):
    """A file with no `empirical:` block (e.g., a stub written by an
    aborted probe run) yields None — partial files don't masquerade
    as confirmed-good data."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_empirical_cache(tmp_path, camera_id=3, empirical_override={})
    assert ptz_core._load_empirical_from_disk(3) is None


def test_probe_capabilities_merges_disk_empirical(tmp_path, monkeypatch):
    """End-to-end: probe_capabilities reads the cam's empirical file
    from disk and surfaces the gap between declared and empirical
    in the result. This is what powers the operator-facing tri-state pills."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))
    _write_empirical_cache(
        tmp_path,
        camera_id=3,
        ip="192.168.1.100",
        relative_works=False,
        absolute_works=False,
    )

    fake_client = MagicMock()
    fake_client.get_capabilities.return_value = {
        "service_capabilities": None,
        "nodes": [],
        "declared": {
            "continuous_pan_tilt": True,
            "relative_pan_tilt": True,
            "absolute_pan_tilt": True,
            "max_presets": 32,
        },
        "error": None,
    }
    storage = MagicMock()
    storage.get_camera.return_value = _stub_camera_lookup(
        camera_id=3, ip="192.168.1.100"
    )

    with patch.object(
        ptz_core, "get_camera_storage", return_value=storage
    ), patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        result = ptz_core.probe_capabilities(3)

    # Declared says all three move types are supported …
    assert result["declared"]["relative_pan_tilt"] is True
    assert result["declared"]["absolute_pan_tilt"] is True
    # … but the empirical file (written by the probe tool) shows they
    # don't actually work. This is the gap the tri-state pills surface.
    assert result["empirical"]["relative_works"] is False
    assert result["empirical"]["absolute_works"] is False
    assert result["empirical"]["continuous_works"] is True


def test_probe_capabilities_empirical_is_none_for_unprobed_cam(
    tmp_path, monkeypatch
):
    """Fresh operator: cam saved in WMB, probe never run for it →
    empirical=None → Settings UI renders yellow ?-pills + advises to
    run the probe."""
    monkeypatch.setattr(ptz_core, "get_path_manager", lambda: _fake_pm_for(tmp_path))

    fake_client = MagicMock()
    fake_client.get_capabilities.return_value = {
        "service_capabilities": None,
        "nodes": [],
        "declared": {"continuous_pan_tilt": True, "max_presets": 32},
        "error": None,
    }
    storage = MagicMock()
    storage.get_camera.return_value = _stub_camera_lookup()

    with patch.object(
        ptz_core, "get_camera_storage", return_value=storage
    ), patch.object(ptz_core, "_client_for_camera", return_value=fake_client):
        result = ptz_core.probe_capabilities(7)

    assert result["empirical"] is None


# ---------------------------------------------------------------------------
# Probe-tool smoke tests — scripts.ptz_probe
# ---------------------------------------------------------------------------
#
# Ensures the public operator-facing tool stays importable + invocable
# from the repo. These are NOT functional probe tests (those require a
# real cam) — just guards against accidentally breaking `python -m
# scripts.ptz_probe` via a refactor.


def test_scripts_ptz_probe_package_importable():
    """The probe tool's core module must import cleanly. Catches
    accidental sys.path / __init__.py breakage."""
    from scripts.ptz_probe import core as probe_core

    # A handful of expected public API surfaces from the core.
    for name in (
        "connect",
        "get_device_info",
        "get_ptz_nodes",
        "get_service_capabilities",
        "continuous_move",
        "relative_move",
        "absolute_move",
        "emergency_stop",
    ):
        assert hasattr(probe_core, name), (
            f"scripts.ptz_probe.core missing expected attr {name!r}"
        )


def test_scripts_ptz_probe_help_exits_zero():
    """`python -m scripts.ptz_probe --help` must exit 0 with usage on
    stdout — protects the documented operator invocation."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "scripts.ptz_probe", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"--help exited non-zero. stderr:\n{result.stderr}"
    )
    assert "--camera-id" in result.stdout, (
        "Operator-facing --camera-id flag missing from --help output."
    )
    assert "--output-dir" in result.stdout, (
        "Operator-facing --output-dir flag missing from --help output."
    )
