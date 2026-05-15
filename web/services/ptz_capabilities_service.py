"""
PTZ Capabilities Service - thin web wrapper around the core
capability-probe.

Slice 4-C of 2026-05-17_PTZ_capability-probe-and-integration.
Exposes the declared ONVIF capabilities of a saved camera to the
Settings UI without invoking any movement on the cam. Empirical
move tests live in the standalone probe tool under
``agent_handoff/lab/experiments/ptz_probe/`` — this service is
``GetServiceCapabilities`` + ``GetNodes``-grade only.

Boundary: this module follows the H-01 invariant (web/services/*
imports only from core/*, stdlib/typing, config, logging_config,
utils.*, other web.services.*). Camera-protocol details live in
``camera/ptz_client.py``; the core layer mediates.
"""

from typing import Any

from core import ptz_core


def probe_capabilities(
    camera_id: int, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Return the declared PTZ capabilities of a saved camera.

    Cached for 60s per camera in core. ``force_refresh=True``
    bypasses the cache (used by the Settings UI's "Re-probe"
    button so the operator can verify after firmware/network
    changes without waiting out the TTL).

    Raises ``ValueError`` if the camera_id is unknown.
    """
    return ptz_core.probe_capabilities(camera_id, force_refresh=force_refresh)


def clear_cache(camera_id: int | None = None) -> None:
    """Drop cached capability-probe results.

    Mostly used by tests; production code prefers the
    ``force_refresh=True`` route through ``probe_capabilities``.
    """
    ptz_core.clear_capabilities_cache(camera_id)
