"""
PTZ Empirical Probe Service - thin web wrapper.

Free-form probe flow: operator picks tests in any order from a flat
list, re-runs are first-class, finalize is explicit.

Exposes the core.ptz_empirical_probe state machine to the Flask layer
without leaking any camera/ or detector/ imports (H-01 boundary).
"""

from typing import Any

from core import ptz_empirical_probe


def start_session(
    camera_id: int, *, probe_slot: str | None = None
) -> dict[str, Any]:
    """Start (or resume) a probe session. Returns the full case list
    + any already-recorded verdicts so the wizard can render its
    dashboard without a second request.

    ``probe_slot`` is the operator-picked preset token (1–32 on this
    firmware). When set, the wizard's preset block writes to that slot
    via SetPreset and verifies round-trip via GotoPreset. When None or
    empty, the preset block degrades to overview-goto verification.

    Raises:
        ValueError: when no overview preset is configured, or when
            probe_slot equals overview_preset after canonicalization.
        RuntimeError: when the AutoPtzController could not be paused.
    """
    session = ptz_empirical_probe.start_session(camera_id, probe_slot=probe_slot)
    return _session_to_payload(session)


def session_status(camera_id: int) -> dict[str, Any] | None:
    """Return the current session metadata + all cases + verdicts, or
    None when no probe is in flight."""
    session = ptz_empirical_probe.get_session(camera_id)
    if session is None:
        return None
    return _session_to_payload(session)


def execute_step(camera_id: int, step_id: str) -> dict[str, Any]:
    """Fire the ONVIF move for a specific step (free-form addressing).

    Raises:
        ValueError: no session in flight, or step_id unknown.
        ExecuteInFlightError: another step is currently executing.
    """
    return ptz_empirical_probe.execute_step(camera_id, step_id)


def record_feedback(
    camera_id: int, *, step_id: str, feedback: str, comment: str = ""
) -> dict[str, Any]:
    """Record the operator's verdict ("yes"/"no"/"skip") for a step.

    Free-form: no advance, no auto-finalize. The wizard calls
    :func:`finalize_session` explicitly when the operator clicks
    Finish probe.
    """
    return ptz_empirical_probe.record_feedback(
        camera_id, step_id=step_id, feedback=feedback, comment=comment
    )


def finalize_session(camera_id: int) -> dict[str, Any]:
    """Operator-triggered finalize: write the empirical YAML, resume
    Auto-PTZ, clear the session.

    Returns ``{cache_path, cache_error, verdict_count, total_steps}``.
    The wizard's done modal shows a warning when ``cache_error`` is
    non-empty (typically a permissions issue on the data dir).
    """
    return ptz_empirical_probe.finalize_session(camera_id)


def apply_near_focus_budget(camera_id: int) -> dict[str, Any]:
    """Read the most recently probed follow_zoom_max_burst_sec from the
    cache YAML and apply it to the camera's live PTZ config.

    All reads + writes route through core.ptz_core (which owns the
    storage boundary) so this service stays inside the H-01 import
    invariant — no direct utils.* imports.

    Returns ``{"applied": bool, "value": float | None, "reason": str}``.
    ``applied=False`` with a human-readable reason when the cache has
    no budget recorded yet (operator hasn't run the near-focus step,
    or rated it "no"). The wizard surfaces the reason in the Apply
    button's tooltip.
    """
    from core import ptz_core

    empirical = ptz_core._load_empirical_from_disk(int(camera_id)) or {}
    value = empirical.get("follow_zoom_max_burst_sec")
    if value is None:
        return {
            "applied": False,
            "value": None,
            "reason": (
                "No near-focus budget recorded yet — run the wizard's "
                "near-focus step first."
            ),
        }
    current = ptz_core.get_ptz_config(int(camera_id))
    if current is None:
        return {
            "applied": False,
            "value": float(value),
            "reason": f"Camera {camera_id} not found",
        }
    merged = dict(current)
    merged["follow_zoom_max_burst_sec"] = float(value)
    updated = ptz_core.update_ptz_config(int(camera_id), merged)
    if updated is None:
        return {
            "applied": False,
            "value": float(value),
            "reason": "Failed to write camera config",
        }
    return {
        "applied": True,
        "value": float(value),
        "reason": "",
    }


def abort_session(camera_id: int) -> bool:
    """Emergency-stop the cam, persist partial verdicts as a YAML,
    clear the session, resume Auto-PTZ.

    Returns True if a session was aborted, False if there was nothing
    to abort. Safe to call repeatedly.
    """
    return ptz_empirical_probe.abort_session(camera_id)


# ---------------------------------------------------------------------------
# Internal: shape the session for HTTP responses
# ---------------------------------------------------------------------------


def _session_to_payload(session: Any) -> dict[str, Any]:
    """Strip the dataclass internals down to what the wizard UI needs.

    The full case list is included so the wizard's flat-list dashboard
    can render every test card without a second request. Sessions are
    bounded (~33 cases on a fully-declared cam) so payload size is
    not a concern.
    """
    return {
        "camera_id": session.camera_id,
        "camera_ip": session.camera_ip,
        "cases": list(session.cases),  # full case list for the flat UI
        "verdict_count": session.current_index,
        "total_steps": session.total_steps(),
        "done": session.is_done(),
        "aborted": session.aborted,
        "started_at": session.started_at,
        "finished_at": session.finished_at,
        "probe_slot": getattr(session, "probe_slot", "") or "",
        "overview_preset": getattr(session, "overview_preset", "") or "",
        "results": {
            sid: {
                "feedback": r.feedback,
                "comment": r.comment,
                "executed_at": r.executed_at,
                "onvif_error": r.onvif_error,
                # Empty for most kinds; near_focus accumulates bursts so the
                # wizard can show "N bursts so far" while the operator clicks.
                "poll_samples": list(r.poll_samples or []),
            }
            for sid, r in session.results.items()
        },
    }
