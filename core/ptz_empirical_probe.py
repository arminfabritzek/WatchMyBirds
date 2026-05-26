"""
PTZ Empirical Probe — in-UI wizard backend.

State-machine + persistence + finalize for the operator-attended PTZ
capability probe that runs from the WMB Settings/Stream-page wizard.

The wizard walks the operator through the same 10 move tests the CLI
tool runs (`python -m scripts.ptz_probe`), but each step is a separate
HTTP request: start → execute step → operator watches camera → operator
clicks ✓/✗/skip → record → advance. The session lives in this module's
in-memory dict between requests; on every step it's persisted to YAML
on disk so a browser refresh or app restart can resume mid-probe.

When the session finalizes, the empirical summary is written to
`OUTPUT_DIR/ptz_capabilities/cam<id>.yaml` — the canonical file the
Settings-UI tri-state pills read. Same format as
`scripts/ptz_probe/__main__.py::_write_wmb_capabilities_cache` so the
in-UI and CLI paths produce interchangeable output.

The wizard takes exclusive control of the camera via
`AutoPtzController.pause_for_external` — detection-driven moves are
suppressed for the duration of the session so the controller cannot
fight the wizard's commands.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from camera.ptz_client import PtzClient
from core import ptz_core
from utils.log_safety import safe_log_value as _slv
from utils.path_manager import get_path_manager

logger = logging.getLogger(__name__)


# Movestatus-poll window in seconds. Module-level constant so tests can
# monkey-patch it down to ~0.1s for fast iteration without touching the
# wizard-facing default of 3 seconds.
_MOVESTATUS_POLL_WINDOW_SEC = 3.0

# Operator-verdict literals. The wizard JS sends these verbatim and the
# rollup logic in _finalize_session checks `feedback == FEEDBACK_YES`.
# Hoisting them avoids drift between the validator, the rollup, and the
# StepResult default.
FEEDBACK_YES = "yes"
FEEDBACK_NO = "no"
FEEDBACK_SKIP = "skip"
FEEDBACK_PENDING = "pending"
VALID_FEEDBACK = frozenset({FEEDBACK_YES, FEEDBACK_NO, FEEDBACK_SKIP})


# ---------------------------------------------------------------------------
# Test-case definitions
# ---------------------------------------------------------------------------
#
# Each case is a dict with:
#   id          — stable identifier, also used in result keys
#   kind        — "continuous" | "relative" | "absolute" | "movestatus"
#   description — short label shown in the wizard step card
#   expectation — what the operator should look for in the live stream
#   purpose     — *why* this test exists (helps the operator judge edge cases)
#   inputs      — kwargs handed to the ONVIF move call (pan/tilt/zoom/...)
#
# These mirror scripts/ptz_probe/__main__.py::_continuous_cases /
# _relative_cases / _absolute_cases. Kept independent (not imported)
# because core/ should not depend on scripts/; a future shared
# probe-case module should be extracted from here and the CLI together.


def _case_by_id(session: ProbeSession, step_id: str) -> dict[str, Any] | None:
    """Find a case dict by its step_id within a session.

    Returns the case dict (with id/kind/description/expectation/purpose/inputs)
    or None if no such step exists in this session's case list. Used by
    the ID-addressed execute / feedback APIs the wizard's flat-list UI
    drives — replaces the linear current_step() cursor.
    """
    sid = str(step_id)
    for case in session.cases:
        if str(case.get("id")) == sid:
            return case
    return None


def _continuous_cases(declared: dict[str, bool]) -> list[dict[str, Any]]:
    """All continuous-move test cases, gated by declared axes.

    Mirrors `scripts/ptz_probe/__main__.py::_continuous_cases` 1:1 so the
    in-UI wizard and CLI tool produce comparable reports.
    """
    cases: list[dict[str, Any]] = []
    if declared.get("continuous_pan_tilt"):
        cases += [
            {
                "id": "c_pan_right_slow",
                "kind": "continuous",
                "description": "pan right, moderate velocity, 300ms burst",
                "expectation": "Camera turns clearly to the right (a few degrees), then stops within ~300ms.",
                "purpose": "Baseline: does ContinuousMove work at all? Magnitude matches the stream-page joystick (0.5) so a working joystick predicts a working probe — cheap cams often need ≥0.3 to overcome motor startup torque on the tilt axis.",
                "inputs": {"pan": 0.5, "duration_sec": 0.3},
            },
            {
                "id": "c_pan_right_fast",
                "kind": "continuous",
                "description": "pan right, FAST velocity, 300ms burst",
                "expectation": "Camera snaps clearly to the right, then stops. Should move noticeably more than the slow burst.",
                "purpose": "Tests whether the velocity scalar actually scales above the baseline — many cheap cams collapse 0.5 and 0.8 to the same speed.",
                "inputs": {"pan": 0.8, "duration_sec": 0.3},
            },
            {
                "id": "c_pan_left_slow",
                "kind": "continuous",
                "description": "pan left, moderate velocity, 300ms burst",
                "expectation": "Camera turns clearly to the LEFT, then stops.",
                "purpose": "Sign-correctness check — does negative pan = left?",
                "inputs": {"pan": -0.5, "duration_sec": 0.3},
            },
            {
                "id": "c_tilt_up_slow",
                "kind": "continuous",
                "description": "tilt up, moderate velocity, 300ms burst",
                "expectation": "Camera tilts upward, then stops.",
                "purpose": "Confirms tilt works and positive tilt = up (some cams invert). Tilt motors typically need more torque than pan to start moving — using the joystick's 0.5 magnitude avoids false negatives.",
                "inputs": {"tilt": 0.5, "duration_sec": 0.3},
            },
            {
                "id": "c_tilt_down_slow",
                "kind": "continuous",
                "description": "tilt down, moderate velocity, 300ms burst",
                "expectation": "Camera tilts downward, then stops.",
                "purpose": "Tilt sign-correctness in the other direction.",
                "inputs": {"tilt": -0.5, "duration_sec": 0.3},
            },
            {
                "id": "c_pan_long",
                "kind": "continuous",
                "description": "pan right, slow velocity, up to 1500ms burst",
                "expectation": "Camera turns right smooth and steady, then stops cleanly without overshooting. The wizard asks for 1500ms but many cheap cams ignore the requested duration and use their own burst length (~800-1200ms is typical) — that's fine, the test is about whether the Stop() call cleanly halts whatever motion was in progress, not about exactly 1.5s of motion.",
                "purpose": "Stop-timing check — if the cam keeps drifting after the Stop() call, ContinuousMove tracking will overshoot. Magnitude stays low (0.2) to expose drift; raising it would mask it. Rate ✓ if motion stopped cleanly regardless of how long the burst actually lasted.",
                "inputs": {"pan": 0.2, "duration_sec": 1.5},
            },
        ]
    if declared.get("continuous_zoom"):
        cases += [
            {
                "id": "c_zoom_in_slow",
                "kind": "continuous",
                "description": "zoom in, slow velocity, 300ms burst",
                "expectation": "Camera zooms in slightly (image gets bigger).",
                "purpose": "Confirms continuous zoom works.",
                "inputs": {"zoom": 0.2, "duration_sec": 0.3},
            },
            {
                "id": "c_zoom_out_slow",
                "kind": "continuous",
                "description": "zoom out, slow velocity, 300ms burst",
                "expectation": "Camera zooms out (image gets wider).",
                "purpose": "Zoom sign-correctness — does negative zoom = wider?",
                "inputs": {"zoom": -0.2, "duration_sec": 0.3},
            },
        ]
    return cases


def _relative_cases(declared: dict[str, bool]) -> list[dict[str, Any]]:
    """Relative-move amplitude ladder + sign/repeat checks per axis.

    Mirrors `scripts/ptz_probe/__main__.py::_relative_cases`.
    """
    cases: list[dict[str, Any]] = []
    if declared.get("relative_pan_tilt"):
        cases += [
            {
                "id": "r_pan_right_tiny",
                "kind": "relative",
                "description": "relative pan +0.02 (TINY step right) — SAFETY: hit ⚠ Halt motion below if the camera keeps panning",
                "expectation": "Camera nudges very slightly to the right — barely perceptible step, NOT a continuous move. If the cam keeps moving past ~1 second, this firmware interprets Relative as velocity-style: hit the ⚠ Halt motion button to stop it, then rate ✗.",
                "purpose": "Amplitude rung 1. If visible-but-tiny, Translation IS a position delta. If it 'keeps moving', Translation is being velocity-interpreted.",
                "inputs": {"pan": 0.02, "speed": 0.5},
            },
            {
                "id": "r_pan_right_tiny_repeat",
                "kind": "relative",
                "description": "relative pan +0.02 AGAIN (reproducibility check)",
                "expectation": "Camera nudges the SAME tiny step again.",
                "purpose": "Reproducibility — if Δ1 ≠ Δ2, RelativeMove is useless for closed-loop tracking.",
                "inputs": {"pan": 0.02, "speed": 0.5},
            },
            {
                "id": "r_pan_right_small",
                "kind": "relative",
                "description": "relative pan +0.05 (small step right)",
                "expectation": "Camera moves slightly more than the +0.02 step, still a discrete movement that stops on its own.",
                "purpose": "Amplitude rung 2 — does the step size scale?",
                "inputs": {"pan": 0.05, "speed": 0.5},
            },
            {
                "id": "r_pan_right_medium",
                "kind": "relative",
                "description": "relative pan +0.10 (medium step right)",
                "expectation": "Camera moves a clearly visible step right and stops on its own.",
                "purpose": "Amplitude rung 3. If all three rungs hit an endstop, RelativeMove is velocity-style on this firmware.",
                "inputs": {"pan": 0.10, "speed": 0.5},
            },
            {
                "id": "r_pan_left_small",
                "kind": "relative",
                "description": "relative pan -0.05 (small step LEFT)",
                "expectation": "Camera moves slightly to the LEFT.",
                "purpose": "Sign-correctness for relative pan.",
                "inputs": {"pan": -0.05, "speed": 0.5},
            },
            {
                "id": "r_tilt_up_small",
                "kind": "relative",
                "description": "relative tilt +0.05 (small step up)",
                "expectation": "Camera tilts slightly upward.",
                "purpose": "Relative tilt works + positive = up.",
                "inputs": {"tilt": 0.05, "speed": 0.5},
            },
            {
                "id": "r_tilt_down_small",
                "kind": "relative",
                "description": "relative tilt -0.05 (small step down)",
                "expectation": "Camera tilts slightly downward.",
                "purpose": "Relative tilt sign-correctness in the other direction.",
                "inputs": {"tilt": -0.05, "speed": 0.5},
            },
        ]
    if declared.get("relative_zoom"):
        cases += [
            {
                "id": "r_zoom_in_small",
                "kind": "relative",
                "description": "relative zoom +0.05 (small zoom in)",
                "expectation": "Camera zooms in by a small notch.",
                "purpose": "Relative zoom works + positive = closer.",
                "inputs": {"zoom": 0.05, "speed": 0.5},
            },
            {
                "id": "r_zoom_out_small",
                "kind": "relative",
                "description": "relative zoom -0.05 (small zoom out)",
                "expectation": "Camera zooms out by a small notch (back roughly to where it was).",
                "purpose": "Relative zoom sign-correctness.",
                "inputs": {"zoom": -0.05, "speed": 0.5},
            },
        ]
    return cases


def _absolute_cases(declared: dict[str, bool]) -> list[dict[str, Any]]:
    """Absolute-move tests gated by declared axes.

    Mirrors `scripts/ptz_probe/__main__.py::_absolute_cases`.
    """
    cases: list[dict[str, Any]] = []
    if declared.get("absolute_pan_tilt"):
        cases += [
            {
                "id": "a_center",
                "kind": "absolute",
                "description": "absolute pan=0, tilt=0 (move to centre) — SAFETY: hit ⚠ Halt motion if the camera keeps panning",
                "expectation": "Camera moves to its CENTRE position and stops within ~3 seconds. If it keeps moving past that, this firmware interprets AbsoluteMove as velocity-style: hit ⚠ Halt motion, then rate ✗ — absolute coords are unusable on this cam.",
                "purpose": "Does the cam have a meaningful (0,0) origin? Some cheap cams treat (0,0) as 'pan to endstop'.",
                "inputs": {"pan": 0.0, "tilt": 0.0, "speed": 0.5},
            },
            {
                "id": "a_right_half",
                "kind": "absolute",
                "description": "absolute pan=+0.5, tilt=0",
                "expectation": "Camera moves to roughly 50% to the right of centre.",
                "purpose": "Tests whether absolute coordinates address a stable world position.",
                "inputs": {"pan": 0.5, "tilt": 0.0, "speed": 0.5},
            },
            {
                "id": "a_left_half",
                "kind": "absolute",
                "description": "absolute pan=-0.5, tilt=0",
                "expectation": "Camera moves to roughly 50% to the left of centre.",
                "purpose": "Confirms the absolute coordinate space is symmetric.",
                "inputs": {"pan": -0.5, "tilt": 0.0, "speed": 0.5},
            },
        ]
    if declared.get("absolute_zoom"):
        cases += [
            {
                "id": "a_zoom_quarter",
                "kind": "absolute",
                "description": "absolute zoom=0.25 (modest zoom)",
                "expectation": "Camera zooms to ~25% of its zoom range.",
                "purpose": "Absolute zoom positioning works.",
                "inputs": {"zoom": 0.25, "speed": 0.5},
            },
            {
                "id": "a_zoom_zero",
                "kind": "absolute",
                "description": "absolute zoom=0 (fully wide)",
                "expectation": "Camera zooms fully out to widest view.",
                "purpose": "Confirms zoom=0 is the wide end (not the tele end).",
                "inputs": {"zoom": 0.0, "speed": 0.5},
            },
        ]
    return cases


def _homecome_case(overview_preset: str, after_block: str) -> dict[str, Any]:
    """Inter-block Home-return — gotos the overview preset and waits.

    Inserted between continuous/relative/absolute blocks so cumulative
    drift doesn't poison later tests, mirroring the CLI's between-block
    `goto_preset(home_token)`.

    ``after_block`` distinguishes the three inter-block returns ("continuous",
    "relative", "absolute"). Without it, all three cases share the same
    case-ID and collide in the UI's flat-list dashboard + session.results
    dict (one row rendered, one verdict storable, but total_steps counts
    three — the operator can't reach 100%).
    """
    block_label = str(after_block or "block").strip() or "block"
    return {
        "id": f"home_return_after_{block_label}",
        "kind": "homecome",
        "description": f"return to overview preset {overview_preset} (after {block_label} block)",
        "expectation": "Camera flies back to the Overview position before the next block starts.",
        "purpose": "Resets cumulative drift between test blocks.",
        "inputs": {"preset_token": overview_preset, "settle_sec": 4.0},
    }


def _preset_cases(probe_slot: str | None, overview_preset: str) -> list[dict[str, Any]]:
    """Preset SetPreset + GotoPreset round-trip + overview-goto verification.

    The CLI's `_ensure_home_preset` runs this implicitly; the wizard
    surfaces it as its own block so the operator can confirm presets
    are usable on this firmware. `probe_slot` is the operator-chosen
    1..32 slot (None = block skipped).
    """
    if not probe_slot:
        # No slot picked — only verify Goto on the existing overview.
        return [
            {
                "id": "p_goto_overview",
                "kind": "preset_goto",
                "description": f"GotoPreset({overview_preset}) — fly to overview",
                "expectation": "Camera flies smoothly to the Overview preset position.",
                "purpose": "Verifies the overview preset still works after the move tests.",
                "inputs": {"preset_token": overview_preset, "settle_sec": 4.0},
            },
        ]
    return [
        {
            "id": "p_set_probe_slot",
            "kind": "preset_set",
            "description": f"SetPreset on slot {probe_slot} at the current camera view",
            "expectation": "Camera does NOT move. The current view is saved under the probe slot.",
            "purpose": "Tests whether SetPreset actually stores a slot (some firmware silently no-ops).",
            "inputs": {"preset_token": probe_slot, "preset_name": f"WMB_probe_{probe_slot}"},
        },
        {
            "id": "p_goto_overview_first",
            "kind": "preset_goto",
            "description": f"GotoPreset({overview_preset}) — leave the probe view",
            "expectation": "Camera flies to the Overview preset.",
            "purpose": "Move away from the probe slot's position so the next test can prove the goto really moved.",
            "inputs": {"preset_token": overview_preset, "settle_sec": 4.0},
        },
        {
            "id": "p_goto_probe_slot",
            "kind": "preset_goto",
            "description": f"GotoPreset({probe_slot}) — return to the saved view",
            "expectation": "Camera flies back to EXACTLY the view that was saved in the previous step.",
            "purpose": "Round-trip — if the cam ends up at the same view as p_set_probe_slot, SetPreset+GotoPreset work end-to-end.",
            "inputs": {"preset_token": probe_slot, "settle_sec": 4.0},
        },
    ]


def _multi_goto_case(overview_preset: str, probe_slot: str | None) -> dict[str, Any]:
    """Three gotos in a row alternating overview ↔ probe-slot (or just overview).

    Tests whether the cam handles command queueing — some firmware
    drops or re-orders rapid-fire gotos.
    """
    if probe_slot:
        sequence = [overview_preset, probe_slot, overview_preset]
        desc_suffix = f"{overview_preset} → {probe_slot} → {overview_preset}"
    else:
        sequence = [overview_preset] * 3
        desc_suffix = f"{overview_preset} × 3"
    return {
        "id": "p_multi_goto",
        "kind": "multi_goto",
        "description": f"3 gotos back-to-back ({desc_suffix})",
        "expectation": "Camera visits each preset in order. The FINAL position must be the last preset in the list.",
        "purpose": "Detects firmware that drops queued gotos or only honours the last one — affects how WMB issues counter-gotos.",
        "inputs": {"sequence": sequence, "settle_sec_each": 4.0},
    }


def _goto_speed_cases(overview_preset: str, probe_slot: str | None) -> list[dict[str, Any]]:
    """GotoPreset at three different speeds (0.2 / 0.5 / 0.8).

    Tests whether the cam respects the ONVIF Speed parameter on
    preset jumps — many cheap cams ignore it and run at fixed speed.
    Uses probe_slot as the target so the cam has somewhere to fly TO;
    falls back to overview if no probe slot was set.
    """
    target = probe_slot or overview_preset
    return [
        {
            "id": "p_goto_speed_slow",
            "kind": "preset_goto",
            "description": f"GotoPreset({target}) at speed=0.2 (slow)",
            "expectation": "Camera flies to the target. Note how long it takes — many cheap cams ignore the speed parameter and run at a fixed pace. Rate ✓ only if you see a CLEAR difference vs. the next two cases; if all three feel identical, rate this one ✗ (speed is ignored).",
            "purpose": "Speed rung 1. Three cases together prove whether the cam respects the ONVIF Speed scalar on preset jumps.",
            "inputs": {"preset_token": target, "speed": 0.2, "settle_sec": 6.0,
                       "prep_goto_token": overview_preset if target != overview_preset else None},
        },
        {
            "id": "p_goto_speed_mid",
            "kind": "preset_goto",
            "description": f"GotoPreset({target}) at speed=0.5 (medium)",
            "expectation": "Should be noticeably faster than 0.2. If it looks identical to the slow case, the cam ignores the speed parameter — rate ✗.",
            "purpose": "Speed rung 2 — does the scalar actually scale?",
            "inputs": {"preset_token": target, "speed": 0.5, "settle_sec": 4.0,
                       "prep_goto_token": overview_preset if target != overview_preset else None},
        },
        {
            "id": "p_goto_speed_fast",
            "kind": "preset_goto",
            "description": f"GotoPreset({target}) at speed=0.8 (fast)",
            "expectation": "Should snap to the target — clearly faster than 0.5. If identical to slow/mid, the cam ignores the speed parameter — rate ✗.",
            "purpose": "Speed rung 3. If all three feel the same, goto_speed_scales is reported as false in the empirical YAML.",
            "inputs": {"preset_token": target, "speed": 0.8, "settle_sec": 4.0,
                       "prep_goto_token": overview_preset if target != overview_preset else None},
        },
    ]


def _near_focus_case(declared: dict[str, bool]) -> list[dict[str, Any]]:
    """Near-focus zoom-budget discovery — operator drives a series of
    small zoom-in bursts and stops when the live preview goes blurry.

    Only emitted when continuous zoom is declared; for cams without
    continuous zoom there is no time-integrated budget to discover.
    Each operator click on the wizard's "Zoom in more" button re-runs
    the SAME step_id, and the burst duration is appended to the
    StepResult.poll_samples list. Finalize sums the bursts (excluding
    the last one when feedback="yes", since that burst is what
    produced the blur) into the follow_zoom_max_burst_sec field.
    """
    if not declared.get("continuous_zoom"):
        return []
    return [
        {
            "id": "nf_zoom_budget",
            "kind": "near_focus",
            "description": "Find your lens's near-focus zoom limit (follow-mode budget)",
            "expectation": (
                "Click 'Zoom in more' repeatedly. Watch the live preview. "
                "The moment a click produces a BLURRY image, hit 'STOP — "
                "blurry'. If you're happy with the framing before any blur "
                "appears, hit 'Done — keep this zoom' instead."
            ),
            "purpose": (
                "Follow-mode auto-PTZ has no absolute zoom feedback on "
                "most cheap cams (GetStatus stub), so it caps the total "
                "time it spends driving zoom-in per bird. Finding your "
                "lens's near-focus limit empirically prevents blurred "
                "auto-framed shots of close subjects (e.g. feeders <1 m)."
            ),
            "inputs": {"zoom": 0.2, "duration_sec": 0.25},
        }
    ]


def _movestatus_case() -> dict[str, Any]:
    """The MoveStatus-poll diagnostic — runs ContinuousMove + polls GetStatus
    to check whether MoveStatus actually transitions."""
    return {
        "id": "movestatus_poll",
        "kind": "movestatus",
        "description": "ContinuousMove pan+0.3 for 1s, poll GetStatus 3s",
        "expectation": "GetStatus samples should show MoveStatus going MOVING→IDLE, not stuck on one value.",
        "purpose": "Tells us whether closed-loop tracking (wait-until-idle) is possible on this cam.",
        "inputs": {
            "pan": 0.3,
            "duration_sec": 1.0,
            "poll_window_sec": _MOVESTATUS_POLL_WINDOW_SEC,
        },
    }


def cases_for(
    declared: dict[str, bool],
    *,
    overview_preset: str = "",
    probe_slot: str | None = None,
) -> list[dict[str, Any]]:
    """Return the test cases the wizard should walk through.

    Block order matches the CLI:
      1. Continuous (gated by declared continuous_*)
      2. Home-return
      3. Relative (gated by declared relative_*)
      4. Home-return
      5. Absolute (gated by declared absolute_*)
      6. Home-return
      7. Preset (SetPreset + GotoPreset round-trip on probe_slot)
      8. Multi-goto sequence
      9. Goto at three speeds
     10. MoveStatus poll diagnostic

    Home-return blocks are skipped when there is no overview_preset to
    return to. The preset/multi-goto/speed blocks degrade gracefully
    when probe_slot is None (operator opted out of writing a slot).
    """
    cases: list[dict[str, Any]] = []

    home = (overview_preset or "").strip()

    def _maybe_homecome(after_block: str) -> None:
        if home:
            cases.append(_homecome_case(home, after_block=after_block))

    continuous = _continuous_cases(declared)
    if continuous:
        cases.extend(continuous)
        _maybe_homecome("continuous")

    # Near-focus zoom budget — operator-attended discovery of the lens's
    # near-focus limit. Sits right after continuous + homecome so the
    # lens starts from the overview's wide-angle reference point. We
    # also add a homecome after the step so the cam recovers to wide-
    # angle whichever button the operator ended on (STOP / Done /
    # never-executed).
    near_focus = _near_focus_case(declared)
    if near_focus:
        cases.extend(near_focus)
        _maybe_homecome("near_focus")

    relative = _relative_cases(declared)
    if relative:
        cases.extend(relative)
        _maybe_homecome("relative")

    absolute = _absolute_cases(declared)
    if absolute:
        cases.extend(absolute)
        _maybe_homecome("absolute")

    if home:
        cases.extend(_preset_cases(probe_slot, home))
        cases.append(_multi_goto_case(home, probe_slot))
        cases.extend(_goto_speed_cases(home, probe_slot))

    # MoveStatus poll always runs — even if MoveStatus is undeclared,
    # it's worth confirming the diagnostic comes back stuck.
    cases.append(_movestatus_case())

    return cases


# ---------------------------------------------------------------------------
# Session dataclass + in-memory registry
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """One operator-recorded test outcome."""

    step_id: str
    feedback: str  # "yes" | "no" | "skip" | "pending"
    comment: str = ""
    executed_at: float = 0.0  # unix seconds; 0 if never run
    poll_samples: list[dict[str, Any]] = field(default_factory=list)
    onvif_error: str = ""


@dataclass
class ProbeSession:
    """A single in-flight probe session for one camera.

    Lifecycle:
      created → execute step 0 → feedback → execute step 1 → … → finalize → done
    """

    camera_id: int
    camera_ip: str
    cases: list[dict[str, Any]]
    current_index: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    aborted: bool = False
    results: dict[str, StepResult] = field(default_factory=dict)
    # Operator-picked preset slot the wizard may write to in the preset
    # block. "" / None when the operator opted out — preset cases then
    # only verify the existing overview goto.
    probe_slot: str = ""
    overview_preset: str = ""

    def total_steps(self) -> int:
        return len(self.cases)

    def current_step(self) -> dict[str, Any] | None:
        if 0 <= self.current_index < len(self.cases):
            return self.cases[self.current_index]
        return None

    def is_done(self) -> bool:
        return self.current_index >= len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for persistence + API responses."""
        return {
            "camera_id": self.camera_id,
            "camera_ip": self.camera_ip,
            "cases": copy.deepcopy(self.cases),
            "current_index": self.current_index,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "aborted": self.aborted,
            "results": {sid: asdict(r) for sid, r in self.results.items()},
            "probe_slot": self.probe_slot,
            "overview_preset": self.overview_preset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeSession:
        """Rehydrate a session from its persisted dict."""
        sess = cls(
            camera_id=int(data["camera_id"]),
            camera_ip=str(data.get("camera_ip") or ""),
            cases=list(data.get("cases") or []),
            current_index=int(data.get("current_index", 0)),
            started_at=float(data.get("started_at") or 0.0),
            finished_at=float(data.get("finished_at") or 0.0),
            aborted=bool(data.get("aborted")),
            probe_slot=str(data.get("probe_slot") or ""),
            overview_preset=str(data.get("overview_preset") or ""),
        )
        for sid, r in (data.get("results") or {}).items():
            sess.results[sid] = StepResult(
                step_id=str(r.get("step_id") or sid),
                feedback=str(r.get("feedback") or FEEDBACK_PENDING),
                comment=str(r.get("comment") or ""),
                executed_at=float(r.get("executed_at") or 0.0),
                poll_samples=list(r.get("poll_samples") or []),
                onvif_error=str(r.get("onvif_error") or ""),
            )
        return sess


# Module-level registry. One session per camera_id at a time. Guarded by
# the same lock pattern as the capability cache. Sessions also live on
# disk (see _persist_session) so an app restart or browser refresh can
# resume mid-probe without losing operator progress.
_session_lock = threading.Lock()
_sessions: dict[int, ProbeSession] = {}

# Per-camera execute-in-flight flags. Set while ``execute_current_step``
# is running its ONVIF moves; cleared in a finally block. Used by
# ``record_feedback`` to refuse advance-while-executing: otherwise a
# multi-goto's 12-second blocking execute could be undercut by a Skip
# that ends up overwritten when execute finally returns.
_executing: set[int] = set()


class ExecuteInFlightError(RuntimeError):
    """Raised when a feedback/execute call arrives mid-execute.

    Surfaced as HTTP 409 by the route layer. The wizard JS shows a
    "wait, test still running" hint and retries once the step finishes.
    """

# Reference to the live AutoPtzController held by the detection manager.
# Set via register_auto_ptz_controller() at app boot — the wizard uses
# it to pause/resume detection-driven moves. None in tests / standalone.
_auto_ptz_controller: Any | None = None


def register_auto_ptz_controller(controller: Any | None) -> None:
    """Register the live AutoPtzController so the probe can pause/resume it.

    Called once at app boot from the same place that instantiates the
    DetectionManager (next to where api_v1.detection_manager is wired).
    Tests can pass a MagicMock to exercise the pause/resume flow without
    a real controller.
    """
    global _auto_ptz_controller
    _auto_ptz_controller = controller


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _session_path(camera_id: int):
    """Return the disk path for this cam's session file. None on PathManager failure."""
    try:
        pm = get_path_manager()
        # Reuse the same dir as the canonical empirical-cache file; the
        # .session.yaml suffix keeps them distinct.
        target = pm.get_ptz_capabilities_path(camera_id)
        return target.parent / f"cam{int(camera_id)}.session.yaml"
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "PathManager unavailable for session path (cam %s): %s",
            _slv(str(camera_id)),
            exc,
        )
        return None


def _persist_session(session: ProbeSession) -> None:
    """Best-effort write of session state to disk."""
    path = _session_path(session.camera_id)
    if path is None:
        return
    try:
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(session.to_dict(), fh, sort_keys=False, allow_unicode=True)
    except OSError as exc:
        logger.debug(
            "Failed to persist probe session for cam %s: %s",
            _slv(str(session.camera_id)),
            exc,
        )


def _load_session_from_disk(camera_id: int) -> ProbeSession | None:
    """Read a session from disk, if it exists and is parseable."""
    path = _session_path(camera_id)
    if path is None or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return ProbeSession.from_dict(data)
    except (OSError, yaml.YAMLError, KeyError, ValueError) as exc:
        logger.debug(
            "Unreadable probe session for cam %s: %s", _slv(str(camera_id)), exc
        )
        return None


def _delete_session_file(camera_id: int) -> None:
    """Remove the on-disk session file after finalize/abort."""
    path = _session_path(camera_id)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug(
            "Failed to delete probe session for cam %s: %s",
            _slv(str(camera_id)),
            exc,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_session(camera_id: int) -> ProbeSession | None:
    """Return the in-flight session for this cam, in memory only.

    Pure read — does NOT touch the AutoPtzController and does NOT load
    from disk. Used by ``/status`` poll endpoints and any code that
    wants to peek without claiming the camera. To resume a session
    persisted to disk (e.g. after an app restart) call
    :func:`claim_session` instead, which is what ``/start`` does.
    """
    cam_id = int(camera_id)
    with _session_lock:
        return _sessions.get(cam_id)


def _hydrate_results_from_cache_yaml(session: ProbeSession) -> int:
    """Pre-populate session.results from the camera's previously written
    empirical YAML at OUTPUT_DIR/ptz_capabilities/cam<id>.yaml.

    Only the ``operator_verdicts`` block is consumed — feedback +
    comment, scoped to step IDs that exist in the freshly-built case
    list. Unknown step IDs (firmware drift, declared-caps mismatch)
    are dropped silently. Missing or unreadable file is a no-op.

    The near-focus step is deliberately not pre-loaded with samples —
    those aren't preserved in the YAML, and a stale burst history
    would be misleading. Operator who wants to re-do near-focus just
    clicks "Zoom in once" to start fresh.

    Returns the count of seeded verdicts (for logging only).
    """
    try:
        pm = get_path_manager()
        cache_path = pm.get_ptz_capabilities_path(session.camera_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Skipping cache hydrate for cam %s — PathManager unavailable: %s",
            session.camera_id, exc,
        )
        return 0
    if not cache_path.exists():
        return 0
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Cache YAML at %s is unreadable; starting fresh probe: %s",
            cache_path, exc,
        )
        return 0
    verdicts = (payload or {}).get("operator_verdicts") or {}
    if not isinstance(verdicts, dict):
        return 0
    known_ids = {str(c["id"]) for c in session.cases}
    seeded = 0
    for raw_id, raw in verdicts.items():
        sid = str(raw_id)
        if sid not in known_ids or not isinstance(raw, dict):
            continue
        fb = str(raw.get("feedback") or FEEDBACK_PENDING).lower()
        if fb not in (FEEDBACK_YES, FEEDBACK_NO, FEEDBACK_SKIP):
            continue
        session.results[sid] = StepResult(
            step_id=sid,
            feedback=fb,
            comment=str(raw.get("comment") or ""),
            # executed_at=0 so the wizard renders "Run test" (not "Re-run
            # test") — the operator re-runs only what they want to retest.
            executed_at=0.0,
        )
        seeded += 1
    session.current_index = sum(
        1 for r in session.results.values()
        if r.feedback != FEEDBACK_PENDING
    )
    return seeded


def claim_session(camera_id: int) -> ProbeSession | None:
    """Return the in-flight session, loading from disk if needed.

    Disk-recovery path: when the app restarted mid-probe but the
    session YAML survives, this reseats the session into memory AND
    re-pauses Auto-PTZ. Only call this from operator-initiated entry
    points (``/start``) — never from polling endpoints, because the
    re-pause is a side effect the operator should observe explicitly.
    """
    cam_id = int(camera_id)
    with _session_lock:
        session = _sessions.get(cam_id)
    if session is not None:
        return session

    session = _load_session_from_disk(cam_id)
    if session is None:
        return None
    with _session_lock:
        _sessions[cam_id] = session
    if _auto_ptz_controller is not None:
        try:
            _auto_ptz_controller.pause_for_external("empirical probe")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to re-pause AutoPtzController on session resume: %s",
                exc,
            )
    return session


def start_session(
    camera_id: int, *, probe_slot: str | None = None
) -> ProbeSession:
    """Start (or resume) a probe session for the given camera.

    Refuses if no Home preset is configured for the cam — the probe
    needs a safe return point between tests. Caller should configure
    Home via the existing Settings → presets workflow first.

    ``probe_slot`` is an operator-picked preset token (1–32 on this
    firmware's slot range). When set, the preset block writes to it
    via SetPreset and verifies via GotoPreset. When None or empty,
    the preset block degrades to overview-goto verification only —
    no SetPreset is issued.

    Idempotent: a second call with an in-flight session returns the
    same session, unchanged. The slot picked at first-start sticks.
    """
    cam_id = int(camera_id)

    # Idempotent: existing session wins. claim_session resurrects a
    # disk-persisted session from a prior crash AND re-pauses Auto-PTZ
    # — which is what we want for an operator-initiated /start.
    existing = claim_session(cam_id)
    if existing is not None and not existing.is_done() and not existing.aborted:
        return existing

    # Probe needs the declared capability set first to know which case
    # blocks to include. This re-uses the cached probe_capabilities call.
    cap_data = ptz_core.probe_capabilities(cam_id)
    declared = cap_data.get("declared") or {}
    cam_ip = str(cap_data.get("ip") or "")

    # Refuse to start without a Home preset. The probe MUST return to
    # a known position between tests; without that, an absolute-move
    # to (0,0,0) on a cheap cam can leave the operator's view stuck at
    # an endstop with no recovery.
    config = ptz_core.get_ptz_config(cam_id) or {}
    overview = str(config.get("overview_preset") or "").strip()
    if not overview:
        raise ValueError(
            "No overview preset configured for this camera. Set one in "
            "Settings → Auto-PTZ before starting the empirical probe — "
            "the probe needs a safe return point between tests."
        )

    slot = (probe_slot or "").strip()
    if slot:
        # Canonicalize: the operator types "20" but the cam returns
        # "Preset020" from GetPresets. A literal string compare against
        # overview would miss the collision, so we resolve the slot to
        # whichever existing preset token shares its integer suffix —
        # if any — before checking.
        slot = _canonicalize_preset_slot(cam_id, slot)
        if slot == overview:
            raise ValueError(
                f"Probe slot resolves to {slot!r}, which is the same as "
                "the overview preset — the wizard would overwrite the "
                "operator's overview. Pick a different slot, or leave "
                "the field blank to skip SetPreset."
            )

    cases = cases_for(declared, overview_preset=overview, probe_slot=slot or None)
    if not cases:
        raise ValueError(
            "Camera declares no PTZ capabilities to test. Run the "
            "declared-capabilities probe (Re-probe button in Settings) "
            "first to refresh the ONVIF capability list."
        )

    session = ProbeSession(
        camera_id=cam_id,
        camera_ip=cam_ip,
        cases=cases,
        current_index=0,
        started_at=time.time(),
        probe_slot=slot,
        overview_preset=overview,
    )

    # Hydrate from any previously written cache YAML for this cam so a
    # second probe doesn't lose the operator's earlier verdicts. Only
    # step IDs present in the freshly-built case list are seeded;
    # foreign IDs (firmware drift, schema changes) are dropped. Failure
    # is non-fatal — wizard just renders everything as "not tested".
    try:
        seeded = _hydrate_results_from_cache_yaml(session)
        if seeded > 0:
            logger.info(
                "Pre-seeded %d operator verdicts for cam %s from prior probe",
                seeded, cam_id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cache hydrate failed for cam %s; continuing with fresh session: %s",
            cam_id, exc,
        )

    # Pause Auto-PTZ BEFORE recording the session so the controller
    # cannot issue a detection-driven move into the wizard's exclusive
    # window. If pause fails, refuse to start — we don't want the wizard
    # and the controller fighting over the cam.
    paused_here = False
    if _auto_ptz_controller is not None:
        try:
            _auto_ptz_controller.pause_for_external("empirical probe")
            paused_here = True
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not pause Auto-PTZ controller; refusing to start probe: {exc}"
            ) from exc

    # Roll the pause back if anything between here and a successful
    # return raises — otherwise Auto-PTZ stays locked until app restart
    # with no session to clean up.
    try:
        with _session_lock:
            _sessions[cam_id] = session
        _persist_session(session)
    except Exception:
        if paused_here and _auto_ptz_controller is not None:
            try:
                _auto_ptz_controller.resume_from_external()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to resume AutoPtzController after start_session "
                    "rollback: %s",
                    exc,
                )
        with _session_lock:
            _sessions.pop(cam_id, None)
        raise
    return session


def execute_step(camera_id: int, step_id: str) -> dict[str, Any]:
    """Fire the ONVIF move for a specific step (free-form addressing).

    Returns a dict with `step_id`, `executed_at`, `kind`, `inputs`,
    `onvif_error` (empty on success) and, for movestatus_poll,
    `poll_samples`. The result is recorded in session.results[step_id]
    so feedback can finalise the operator's verdict.

    Free-form: the wizard surfaces all cases at once and the operator
    runs them in whatever order makes sense. ID-addressed instead of
    cursor-driven so re-runs and out-of-order runs are first-class.
    """
    cam_id = int(camera_id)
    session = get_session(cam_id)
    if session is None:
        raise ValueError(f"No probe session in progress for camera {cam_id}")
    case = _case_by_id(session, step_id)
    if case is None:
        raise ValueError(
            f"Unknown step_id {step_id!r} for camera {cam_id}"
        )

    # Mark execute-in-flight under the session lock so a concurrent
    # /feedback request observes the flag and returns 409 instead of
    # racing this step's StepResult write below.
    with _session_lock:
        if cam_id in _executing:
            raise ExecuteInFlightError(
                f"Probe step is still running for camera {cam_id}; "
                "wait for it to finish before issuing another command."
            )
        _executing.add(cam_id)

    client = _build_client(cam_id)
    result = StepResult(
        step_id=str(case["id"]),
        feedback=FEEDBACK_PENDING,
        executed_at=time.time(),
    )

    try:
        kind = case["kind"]
        inputs = case["inputs"]
        if kind == "continuous":
            client.continuous_move(
                pan=float(inputs.get("pan", 0.0)),
                tilt=float(inputs.get("tilt", 0.0)),
                zoom=float(inputs.get("zoom", 0.0)),
                duration_ms=int(float(inputs.get("duration_sec", 0.3)) * 1000),
            )
        elif kind == "relative":
            client.relative_move(
                pan=float(inputs.get("pan", 0.0)),
                tilt=float(inputs.get("tilt", 0.0)),
                zoom=float(inputs.get("zoom", 0.0)),
                speed=float(inputs["speed"]) if "speed" in inputs else None,
            )
        elif kind == "absolute":
            client.absolute_move(
                pan=float(inputs.get("pan", 0.0)),
                tilt=float(inputs.get("tilt", 0.0)),
                zoom=float(inputs.get("zoom", 0.0)),
                speed=float(inputs["speed"]) if "speed" in inputs else None,
            )
        elif kind in ("preset_goto", "homecome"):
            # Optional prep-goto used by goto-at-speed cases: fly to a
            # neutral spot first so the next goto has somewhere to fly TO,
            # otherwise speed=0.2 on a cam that's already at the target
            # is invisible to the operator.
            prep = inputs.get("prep_goto_token")
            if prep:
                try:
                    client.goto_preset(str(prep))
                    settle = float(inputs.get("prep_settle_sec", 3.0))
                    if settle > 0:
                        time.sleep(settle)
                except Exception as prep_exc:  # noqa: BLE001
                    logger.warning(
                        "Probe step %s prep-goto(%s) failed: %s",
                        _slv(case["id"]), _slv(str(prep)), prep_exc,
                    )
            client.goto_preset(
                str(inputs["preset_token"]),
                speed=float(inputs["speed"]) if "speed" in inputs else None,
            )
        elif kind == "preset_set":
            client.set_preset(
                name=str(inputs.get("preset_name") or f"WMB_probe_{inputs['preset_token']}"),
                preset_token=str(inputs["preset_token"]),
            )
        elif kind == "multi_goto":
            # Fire 3 gotos back-to-back with a settle gap between each.
            # The cam must end at sequence[-1]; the operator confirms
            # visually. ONVIF errors on any goto fail the whole step.
            sequence = list(inputs.get("sequence") or [])
            if not sequence:
                raise ValueError("multi_goto case has empty sequence")
            settle_each = float(inputs.get("settle_sec_each", 4.0))
            for token in sequence:
                client.goto_preset(str(token))
                if settle_each > 0:
                    time.sleep(settle_each)
        elif kind == "near_focus":
            # One zoom-in burst per execute call. The wizard re-invokes
            # execute_step for the same step_id once per "Zoom in more"
            # click; each call appends its burst-duration sample to the
            # existing StepResult so finalize can sum them into the
            # follow_zoom_max_burst_sec budget.
            burst_sec = float(inputs.get("duration_sec", 0.25))
            client.continuous_move(
                zoom=float(inputs.get("zoom", 0.2)),
                duration_ms=int(burst_sec * 1000),
            )
            result.poll_samples = [
                {"burst_sec": burst_sec, "executed_at": result.executed_at}
            ]
        elif kind == "movestatus":
            client.continuous_move(
                pan=float(inputs.get("pan", 0.3)),
                duration_ms=int(float(inputs.get("duration_sec", 1.0)) * 1000),
            )
            # Read the module-level constant at call time so test monkey-
            # patching takes effect even when the case dict was built earlier.
            poll_window = float(
                inputs.get("poll_window_sec", _MOVESTATUS_POLL_WINDOW_SEC)
            )
            if poll_window > _MOVESTATUS_POLL_WINDOW_SEC:
                poll_window = _MOVESTATUS_POLL_WINDOW_SEC
            poll_interval = 0.05 if poll_window < 1.0 else 0.2
            # Use the structured-sample API from PtzClient — same code
            # path the standalone CLI tool uses, so empirical behaviour
            # is identical between in-UI wizard and CLI probe.
            samples = client.poll_move_status(
                duration_sec=poll_window, interval_sec=poll_interval
            )
            # Persist as plain dicts so YAML serialisation stays clean.
            result.poll_samples = [
                {
                    "pan": s.pan,
                    "tilt": s.tilt,
                    "zoom": s.zoom,
                    "move_status_pan_tilt": s.move_status_pan_tilt,
                    "move_status_zoom": s.move_status_zoom,
                    "utc_time": s.utc_time,
                    "error": s.error,
                }
                for s in samples
            ]
        else:
            raise ValueError(f"Unknown probe-step kind: {kind!r}")
    except Exception as exc:  # noqa: BLE001 — operator must see the error
        result.onvif_error = str(exc)
        logger.warning(
            "Probe step %s failed for cam %s: %s",
            _slv(case["id"]),
            _slv(str(cam_id)),
            exc,
        )
    finally:
        with _session_lock:
            _executing.discard(cam_id)

    with _session_lock:
        existing = session.results.get(case["id"])
        if case["kind"] == "near_focus":
            # Near-focus has unique semantics: each execute_step is one
            # zoom-in burst, and the operator clicks "Zoom in once"
            # repeatedly to accumulate a budget. Clicking it on a
            # hydrated-yes step is also valid — the operator is starting
            # a fresh discovery, so we reset feedback to PENDING and
            # accumulate from the new burst onwards.
            if existing is None or existing.feedback != FEEDBACK_PENDING:
                # No prior pending session — start a fresh sample list.
                # The hydrated "yes" verdict (if any) is discarded as
                # soon as the operator re-engages with this step; the
                # next STOP/Done click writes the new verdict.
                session.results[case["id"]] = result
            else:
                # Pending result already in flight — append.
                existing.poll_samples.extend(result.poll_samples)
                existing.executed_at = result.executed_at
                existing.onvif_error = result.onvif_error
        elif existing is None or existing.feedback == FEEDBACK_PENDING:
            # Default kinds: only write the pending result when nothing
            # has been recorded for this step yet. Respects an operator
            # verdict that came in concurrently.
            session.results[case["id"]] = result
        merged = session.results.get(case["id"], result)
    _persist_session(session)
    return {
        "step_id": case["id"],
        "kind": case["kind"],
        "inputs": copy.deepcopy(case["inputs"]),
        "executed_at": merged.executed_at,
        "onvif_error": merged.onvif_error,
        "poll_samples": copy.deepcopy(merged.poll_samples),
    }


def record_feedback(
    camera_id: int, step_id: str, feedback: str, comment: str = ""
) -> dict[str, Any]:
    """Record the operator's verdict for a specific step (free-form).

    Returns a dict with ``step_id``, the resolved ``feedback``, and the
    total ``verdict_count`` so the UI can show progress without owning
    the cursor.

    ``feedback`` is "yes" | "no" | "skip" — anything else is normalised
    to "skip" with a logged warning. Re-rating an already-rated step
    overwrites the verdict (the wizard UI confirms first).

    No auto-finalise: the operator explicitly calls
    :func:`finalize_session` when they're satisfied with the verdicts.
    """
    cam_id = int(camera_id)
    # Block advance-during-execute. Without this, a multi-goto step's
    # 12-second blocking execute can be undercut by a Skip that ends up
    # overwritten when execute finally returns. The wizard JS surfaces
    # 409 as a "wait, still running" hint and retries on completion.
    with _session_lock:
        if cam_id in _executing:
            raise ExecuteInFlightError(
                f"Probe step is still running for camera {cam_id}; "
                "wait for the move to complete before rating it."
            )
    session = get_session(cam_id)
    if session is None:
        raise ValueError(f"No probe session in progress for camera {cam_id}")

    case = _case_by_id(session, step_id)
    if case is None:
        raise ValueError(
            f"Unknown step_id {step_id!r} for camera {cam_id}"
        )

    fb = str(feedback or "").lower().strip()
    if fb not in VALID_FEEDBACK:
        logger.warning("Unknown probe feedback %r, treating as skip", feedback)
        fb = FEEDBACK_SKIP

    with _session_lock:
        result = session.results.get(case["id"])
        if result is None:
            # Operator rated a step without running it first — that's
            # allowed in free-form mode (Skip everything is a valid flow).
            result = StepResult(step_id=case["id"], feedback=fb, comment=str(comment))
        else:
            result.feedback = fb
            result.comment = str(comment)
        session.results[case["id"]] = result
        # Keep current_index as a verdict count so the disk shape stays
        # compatible with older session.yaml files; it has no cursor
        # semantics in free-form mode any more.
        session.current_index = sum(
            1 for r in session.results.values()
            if r.feedback != FEEDBACK_PENDING
        )

    _persist_session(session)

    return {
        "step_id": case["id"],
        "feedback": fb,
        "verdict_count": session.current_index,
        "total_steps": len(session.cases),
    }


def finalize_session(camera_id: int) -> dict[str, Any]:
    """Operator-triggered finalize: roll up verdicts, write cache YAML,
    drop the session, resume Auto-PTZ.

    Free-form mode means the operator decides when they have enough
    verdicts to commit. Returns ``{cache_path, cache_error,
    verdict_count, total_steps}`` for the wizard's done modal.

    Raises ``ValueError`` if there is no session in progress for this
    camera. Idempotent against an empty results dict — produces a
    cache with all flags ``false`` rather than an error.
    """
    cam_id = int(camera_id)
    session = get_session(cam_id)
    if session is None:
        raise ValueError(f"No probe session in progress for camera {cam_id}")
    with _session_lock:
        session.finished_at = time.time()
        verdict_count = sum(
            1 for r in session.results.values()
            if r.feedback != FEEDBACK_PENDING
        )
        total_steps = len(session.cases)
    cache_path, cache_error = _finalize_session(cam_id)
    return {
        "cache_path": cache_path,
        "cache_error": cache_error,
        "verdict_count": verdict_count,
        "total_steps": total_steps,
    }


def abort_session(camera_id: int) -> bool:
    """Emergency-stop the cam, persist any partial verdicts, clear the
    session, resume Auto-PTZ.

    Free-form mode: any verdicts the operator already recorded are
    preserved as a partial cam<id>.yaml before the session is torn
    down. The wizard treats Abort the same as a normal finish for
    cache-write purposes — Settings pills update based on whatever
    the operator confirmed.

    Returns True if a session was aborted, False if there was nothing
    to abort. Safe to call repeatedly.
    """
    cam_id = int(camera_id)
    with _session_lock:
        session = _sessions.get(cam_id)
    if session is None:
        return False

    session.aborted = True
    session.finished_at = time.time()

    # Emergency-stop the cam BEFORE finalize, so the partial YAML reflects
    # the actual stopped state, not a still-moving cam.
    try:
        client = _build_client(cam_id)
        client.stop(pan_tilt=True, zoom=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Emergency stop failed for cam %s during abort: %s", cam_id, exc
        )

    # Write partial YAML + clean up the session + resume controller —
    # _finalize_session handles all three. We don't need to repeat any of
    # them here.
    try:
        _finalize_session(cam_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to write partial cache during abort for cam %s: %s",
            cam_id,
            exc,
        )
        # Belt-and-suspenders: ensure cleanup happens even if finalize
        # raised before its own cleanup ran.
        with _session_lock:
            _sessions.pop(cam_id, None)
        _delete_session_file(cam_id)
        if _auto_ptz_controller is not None:
            try:
                _auto_ptz_controller.resume_from_external()
            except Exception as resume_exc:  # noqa: BLE001
                logger.warning(
                    "Failed to resume AutoPtzController on abort: %s",
                    resume_exc,
                )

    return True


def _finalize_session(camera_id: int) -> tuple[str, str]:
    """Compute the empirical summary, write the canonical cache file,
    clean up.

    Returns ``(cache_path, cache_error)``:
      - on success: ``(path, "")``
      - on cache-write failure: ``("", error_message)`` — session is
        still cleaned up and Auto-PTZ resumed; the operator just
        needs to know the pills won't update.
      - when there's no session to finalise: ``("", "")``.
    """
    cam_id = int(camera_id)
    session = get_session(cam_id)
    if session is None:
        return "", ""

    # Roll up per-kind verdicts: a kind "works" if AT LEAST ONE case of
    # that kind was rated yes by the operator. This is deliberately
    # generous — broken cams typically fail every case of the broken
    # kind (e.g., relative on the operator's IPCAM). One working case
    # is rare but valid evidence.
    # Tri-state rollup helpers. The "not tested" state — distinct from
    # "tested and broken" — is the load-bearing distinction this whole
    # design rests on. "skip" and "pending" both count as "not tested"
    # for boolean rollups: skip is a deliberate non-decision (operator
    # saw the step and chose not to rate it), which carries no evidence
    # about the cam's behaviour either way. The near-focus step has its
    # own meaning for "skip" ("Done — keep this zoom") and reads its
    # verdicts directly, not through these helpers.
    def _untested(feedback: str) -> bool:
        return feedback in (FEEDBACK_PENDING, FEEDBACK_SKIP)

    def _tristate_any_yes(kind: str) -> bool | None:
        any_case = False
        any_rated = False
        for case in session.cases:
            if case["kind"] != kind:
                continue
            any_case = True
            r = session.results.get(case["id"])
            if r is None or _untested(r.feedback):
                continue
            any_rated = True
            if r.feedback == FEEDBACK_YES:
                return True
        if not any_case:
            return None  # no case of this kind in the session — nothing to say
        if not any_rated:
            return None  # cases exist but operator never rated any
        return False  # at least one rated, none said yes

    def _tristate_all_yes(kind: str) -> bool | None:
        # Block-all-yes — used for the speed triplet where "scales" only
        # makes sense if every rung passed. Returns None when no case of
        # the kind was rated; False when at least one was rated non-yes.
        any_case = False
        any_rated = False
        all_rated_yes = True
        for case in session.cases:
            if case["kind"] != kind:
                continue
            any_case = True
            r = session.results.get(case["id"])
            if r is None or _untested(r.feedback):
                all_rated_yes = False
                continue
            any_rated = True
            if r.feedback != FEEDBACK_YES:
                all_rated_yes = False
        if not any_case:
            return None
        if not any_rated:
            return None
        return all_rated_yes

    def _tristate_yes_for_ids(ids: set[str]) -> bool | None:
        if not ids:
            return None
        any_rated = False
        all_yes = True
        for sid in ids:
            r = session.results.get(sid)
            if r is None or _untested(r.feedback):
                all_yes = False
                continue
            any_rated = True
            if r.feedback != FEEDBACK_YES:
                all_yes = False
        if not any_rated:
            return None
        return all_yes

    speed_ids = {
        c["id"] for c in session.cases
        if c["kind"] == "preset_goto" and c["id"].startswith("p_goto_speed_")
    }
    preset_set_ids = {c["id"] for c in session.cases if c["kind"] == "preset_set"}
    multi_goto_ids = {c["id"] for c in session.cases if c["kind"] == "multi_goto"}

    # Near-focus zoom budget: the operator clicked "Zoom in more" N
    # times, each click contributes one burst-duration sample. The
    # feedback tells us which sum to commit:
    #   "yes"   → STOP (just got blurry). Last burst was over the
    #             limit — budget = sum minus the last sample.
    #   "skip"  → Done (happy with this zoom, no blur seen). Budget =
    #             full sum.
    #   "no" / "pending" → operator didn't reach a verdict, or said
    #             the step failed. Don't write a budget; leave the
    #             field absent so the runtime keeps its 0.0 default.
    follow_zoom_max_burst_sec: float | None = None
    for case in session.cases:
        if case["kind"] != "near_focus":
            continue
        r = session.results.get(case["id"])
        if r is None:
            continue
        samples = list(r.poll_samples or [])
        if not samples:
            continue
        bursts = [float(s.get("burst_sec", 0.0)) for s in samples]
        if r.feedback == FEEDBACK_YES:
            # Last burst caused the blur — exclude it.
            follow_zoom_max_burst_sec = round(sum(bursts[:-1]), 3)
        elif r.feedback == FEEDBACK_SKIP:
            follow_zoom_max_burst_sec = round(sum(bursts), 3)
        break

    empirical: dict[str, Any] = {
        "continuous_works": _tristate_any_yes("continuous"),
        "relative_works": _tristate_any_yes("relative"),
        "absolute_works": _tristate_any_yes("absolute"),
        "movestatus_transitions": _tristate_any_yes("movestatus"),
        # Presets: SetPreset only counts as "working" when both write
        # and round-trip read succeed. When the operator opted out of
        # picking a slot, preset_set_ids is empty → not_tested (None),
        # NOT False. False would imply we tried and the cam refused.
        "preset_set_works": _tristate_yes_for_ids(preset_set_ids),
        "preset_goto_works": _tristate_any_yes("preset_goto"),
        "multi_goto_works": _tristate_yes_for_ids(multi_goto_ids),
        # goto_speed_scales requires the full speed triplet, all yes.
        # No speed cases → not_tested (None) instead of the old "False".
        "goto_speed_scales": _tristate_all_yes("preset_goto") if speed_ids else None,
    }
    if follow_zoom_max_burst_sec is not None:
        empirical["follow_zoom_max_burst_sec"] = follow_zoom_max_burst_sec

    # Recommended strategy mirrors the lab tool's logic, but now
    # honestly: an "unknown" strategy means the operator never rated
    # any move-type case, so we have no empirical basis to recommend.
    # Callers must handle "unknown" explicitly — silently defaulting
    # to presets_only (the pre-tri-state behaviour) is exactly the
    # lie this refactor removes.
    if empirical["absolute_works"] is True:
        strategy = "absolute"
    elif empirical["relative_works"] is True:
        strategy = "relative"
    elif empirical["continuous_works"] is True:
        strategy = "continuous_pulse"
    elif empirical["preset_goto_works"] is True:
        strategy = "presets_only"
    elif all(
        empirical[k] is None
        for k in ("absolute_works", "relative_works",
                  "continuous_works", "preset_goto_works")
    ):
        strategy = "unknown"
    else:
        # Mix of rated-no and not-tested with no rated-yes — the
        # operator tested SOMETHING and it broke. Honest fallback
        # is presets_only (because presets at least work on most
        # cams) but mark it as evidence-based, not "default".
        strategy = "presets_only"

    payload = {
        "camera_id": cam_id,
        "probed_at": time.strftime(
            "%Y%m%d_%H%M%S", time.gmtime(session.finished_at or time.time())
        ),
        "connection": {"ip": session.camera_ip},
        "empirical": empirical,
        "recommended_strategy": strategy,
        "probe_slot_used": session.probe_slot,
        "overview_preset": session.overview_preset,
        # Per-step verdicts so a follow-up debugging session can see
        # which specific cases the operator rated yes/no/skip.
        "operator_verdicts": {
            sid: {"feedback": r.feedback, "comment": r.comment}
            for sid, r in session.results.items()
        },
    }

    cache_path_str = ""
    cache_error = ""
    try:
        pm = get_path_manager()
        cache_path = pm.get_ptz_capabilities_path(cam_id)
        with cache_path.open("w", encoding="utf-8") as fh:
            fh.write("# WMB empirical PTZ capabilities — written by in-UI probe wizard\n")
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
        cache_path_str = str(cache_path)
        # Drop the in-memory capability cache so the next Settings-UI
        # open re-reads the freshly-written file.
        ptz_core.clear_capabilities_cache(cam_id)
    except Exception as exc:  # noqa: BLE001
        cache_error = str(exc) or exc.__class__.__name__
        logger.warning(
            "Failed to write canonical empirical cache for cam %s: %s",
            cam_id,
            exc,
        )

    # Session is done — drop from memory, clean disk.
    with _session_lock:
        _sessions.pop(cam_id, None)
    _delete_session_file(cam_id)

    if _auto_ptz_controller is not None:
        try:
            _auto_ptz_controller.resume_from_external()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to resume AutoPtzController on finalize: %s", exc)

    return cache_path_str, cache_error


def _build_client(camera_id: int) -> PtzClient:
    """Build a fresh PtzClient for the given cam. Internal — public
    callers should go through execute_current_step etc."""
    return ptz_core._client_for_camera(camera_id)


def _canonicalize_preset_slot(camera_id: int, raw: str) -> str:
    """Resolve a numeric slot like ``"20"`` to the cam's canonical
    token (e.g. ``"Preset020"``).

    The operator's slot picker accepts integers 1–32. The cam reports
    its existing presets as ``Preset<NNN>`` zero-padded; the bare
    integer and the canonical token both point at the same physical
    slot, but a literal string compare against the overview preset
    misses the collision. This helper queries the live preset list
    and returns whichever token shares the operator's integer suffix.

    Returns the canonical token when a match is found; otherwise the
    raw input (a slot the operator wants to create fresh).
    """
    try:
        slot_int = int(raw)
    except (TypeError, ValueError):
        return raw
    try:
        client = _build_client(camera_id)
        presets = client.list_presets()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Preset list lookup failed for cam %s during canonicalize: %s",
            _slv(str(camera_id)),
            exc,
        )
        return raw
    for preset in presets:
        token = str(getattr(preset, "token", "") or "").strip()
        if not token:
            continue
        # Extract the trailing integer from tokens like "Preset020".
        digits = ""
        for ch in reversed(token):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        if digits and int(digits) == slot_int:
            return token
    return raw


def reset_for_tests() -> None:
    """Test-only: clear all sessions and the controller reference."""
    global _auto_ptz_controller
    with _session_lock:
        _sessions.clear()
        _executing.clear()
    _auto_ptz_controller = None
