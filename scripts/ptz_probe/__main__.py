#!/usr/bin/env python3
"""
PTZ Probe — interactive CLI.

Walks a single ONVIF PTZ camera through every move type WatchMyBirds
might ever want to use, asks the operator after each move whether the
camera actually did the right thing, and writes a structured report:

    ptz_probe_report_<timestamp>.yaml   (human + machine)
    ptz_probe_report_<timestamp>.json   (machine)
    ptz_probe_report_<timestamp>.log    (raw ONVIF traffic + debug)

Usage:
    cd agent_handoff/lab/experiments/ptz_probe
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python ptz_probe.py
    # or with prefilled connection args:
    python ptz_probe.py --ip 192.168.1.100 --port 80 --user admin --pass secret

Safety:
    A "Home" preset is mandatory before any move test runs. The probe
    refuses to start the move sequence if no Home preset is set. Between
    test blocks it returns to Home automatically.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

# Recommended invocation is `python -m scripts.ptz_probe` from the WMB
# repo root, which resolves this import cleanly. If somebody runs the
# file directly (`python scripts/ptz_probe/__main__.py`), the package
# isn't on sys.path — we add the repo root and re-try.
try:
    from scripts.ptz_probe import core
except ImportError:
    import sys as _sys
    _repo_root = str(Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from scripts.ptz_probe import core

SETTINGS_FILE = Path(__file__).resolve().parent / ".ptz_probe_settings.json"

# Time to wait after issuing a goto_preset(Home) between test blocks.
# Preset jumps are FAST on this cam — much faster than continuous pan/tilt
# at default speed — so 2s is enough for the cam to arrive even without
# MoveStatus feedback.
HOME_SETTLE_SEC = 2.0

# Default preset slot for the probe's own Home preset. Operator's slot
# convention: 1–7 = preset-mode zones, 8 = auto-re-focus, 11–17 = grid-mode
# cells. Slots 20+ are free for ad-hoc use. Cam declares max 32 presets, so
# slot 20 is safely inside the physical limit AND clear of all conventions.
DEFAULT_HOME_SLOT = 20

logger = logging.getLogger("ptz_probe")


# ---------------------------------------------------------------------------
# Settings persistence (last-used cam for next run)
# ---------------------------------------------------------------------------


def load_settings() -> dict[str, Any]:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_settings(s: dict[str, Any]) -> None:
    safe = {k: v for k, v in s.items() if k != "password"}  # never persist pw
    try:
        SETTINGS_FILE.write_text(json.dumps(safe, indent=2))
    except Exception as exc:
        logger.warning("could not save settings: %s", exc)


# ---------------------------------------------------------------------------
# Tiny CLI helpers
# ---------------------------------------------------------------------------


def hr(char: str = "─", n: int = 70) -> None:
    print(char * n)


def title(text: str) -> None:
    print()
    hr("═")
    print(f"  {text}")
    hr("═")


def section(text: str) -> None:
    print()
    print(f"── {text} " + "─" * max(0, 66 - len(text)))


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return ans or default


def ask_yn(prompt: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        ans = ask(f"{prompt} ({default_str})").lower()
        if ans == "":
            return default
        if ans in ("y", "yes", "j", "ja"):
            return True
        if ans in ("n", "no", "nein"):
            return False


def ask_feedback(prompt: str) -> str:
    """Ask y/n/r/skip for one move test. Default (Enter) = yes."""
    while True:
        ans = ask(
            f"{prompt} (Y=correct / n=wrong / r=repeat / s=skip-rest)"
        ).lower()
        if ans in ("", "y", "yes"):
            return "yes"
        if ans in ("n", "no"):
            return "no"
        if ans in ("r", "repeat"):
            return "repeat"
        if ans in ("s", "skip"):
            return "skip"


def ask_comment() -> str:
    """Optional free-text note. Empty line = no comment."""
    try:
        ans = input("  Comment (optional, Enter to skip): ").strip()
    except EOFError:
        return ""
    return ans


def _flush_stdin() -> None:
    """Discard any pending stdin bytes left over from previous prompts.

    Without this, a stray 'yy' or 'ggo' typed at an earlier prompt can
    auto-answer the next gate without the user pressing anything. This is
    the most likely explanation for "Enter didn't work" — the gate was
    self-answered by a leftover character before the user even pressed
    Enter. Works on Unix (POSIX termios), no-op on platforms without it.
    """
    try:
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except Exception:
        # Non-interactive stdin, Windows, or other platform without termios —
        # there is nothing safe to do, so silently skip.
        pass


def ask_confirm(prompt: str) -> None:
    """Pause gate. Press Enter to continue, Ctrl-C to abort.

    Discards any leftover stdin bytes from previous prompts BEFORE waiting,
    so a stray character cannot auto-answer this gate. That way Enter
    really means Enter — the user is the only one who can open it.
    """
    _flush_stdin()
    try:
        input(f"  {prompt} ")
    except EOFError:
        print("  (EOF — treating as abort)")
        sys.exit(130)


def ask_go_or_skip(prompt: str) -> str:
    """Three-way gate before firing a move.

    Returns:
      'go'         - operator pressed Enter, run the move
      'skip-test'  - operator typed 's', skip THIS test only
      'skip-block' - operator typed 'S' or 'skip', skip rest of this block

    Ctrl-C still bypasses everything as emergency stop.
    """
    _flush_stdin()
    while True:
        try:
            ans = input(f"  {prompt} ")
        except EOFError:
            print("  (EOF — treating as abort)")
            sys.exit(130)
        stripped = ans.strip()
        if stripped == "":
            return "go"
        if stripped == "s":
            return "skip-test"
        if stripped in ("S", "skip", "SKIP"):
            return "skip-block"
        print("  ↪ Enter = run move | s = skip this test | S = skip rest of block")


def ask_drift(prompt: str) -> str:
    """Three-way for stop-drift observation after a Continuous move.

    'clean' = cam stopped exactly when Stop() arrived, no after-drift
    'tiny'  = cam drifted a small amount (visible but well under 1 frame width)
    'large' = cam kept moving noticeably after Stop()

    Default (Enter) = clean.
    """
    while True:
        ans = ask(
            f"{prompt} (S=stopped-cleanly / t=tiny-after-move / l=large-after-move)"
        ).lower()
        if ans in ("", "s", "stopped", "c", "clean"):
            return "clean"
        if ans in ("t", "tiny"):
            return "tiny"
        if ans in ("l", "large"):
            return "large"


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_probe(args: argparse.Namespace) -> int:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    here = Path(__file__).resolve().parent
    log_path = here / f"ptz_probe_report_{timestamp}.log"
    yaml_path = here / f"ptz_probe_report_{timestamp}.yaml"
    json_path = here / f"ptz_probe_report_{timestamp}.json"

    _setup_logging(log_path, verbose=args.verbose)
    logger.info("PTZ probe starting, report timestamp: %s", timestamp)

    saved = load_settings()
    ip = args.ip or saved.get("ip") or ask("Camera IP")
    port = int(args.port or saved.get("port") or ask("ONVIF port", "80"))
    user = args.user or saved.get("username") or ask("ONVIF username", "admin")
    if args.password:
        password = args.password
    elif os.environ.get("WMB_CAM_PASSWORD"):
        password = os.environ["WMB_CAM_PASSWORD"]
        print("  (using password from $WMB_CAM_PASSWORD)")
    else:
        # Use getpass so the password is not echoed.
        try:
            import getpass

            password = getpass.getpass("  ONVIF password: ")
        except Exception:
            password = ask("ONVIF password")

    if not ip:
        print("ERROR: IP is required.")
        return 2

    title(f"PTZ Probe — {ip}:{port} as {user!r}")

    if not args.no_prompts:
        section("Pre-flight checklist")
        print()
        print("  Before we connect, please confirm:")
        print()
        print(f"    [ ] Camera is powered on and reachable at {ip}:{port}")
        print(f"    [ ] You have a live view of the camera (cam app, RTSP")
        print(f"        viewer, or WMB stream page) so you can SEE moves")
        print(f"    [ ] Operator preset slots 1–7, 8, 11–17 are intact")
        print(f"        and you accept that slot {DEFAULT_HOME_SLOT} will be")
        print(f"        overwritten with the probe's Home preset")
        print(f"    [ ] You can stop the probe at any time with Ctrl-C —")
        print(f"        emergency stop fires and the cam is left where it is")
        print()
        print()
        if not ask_yn("All set — start the probe?", default=True):
            print("  → aborted by operator")
            return 0
        print()

    try:
        conn = core.connect(ip=ip, port=port, username=user, password=password)
    except ConnectionError as exc:
        print(f"\n  CONNECT FAILED: {exc}")
        return 3

    save_settings({"ip": ip, "port": port, "username": user})

    # Install an emergency-stop handler so Ctrl-C halts the cam.
    # Note: all completed test results are already on disk via incremental
    # save in _run_move_tests, so Ctrl-C does NOT lose any answered tests.
    def _sigint(_sig, _frm):
        print("\n\n  ⛔  Ctrl-C — emergency stop")
        core.emergency_stop(conn)
        print(f"  📄  Partial reports already saved at:")
        print(f"      {yaml_path}")
        print(f"      {json_path}")
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint)

    report: dict[str, Any] = {
        "meta": {
            "timestamp": timestamp,
            "tool": "ptz_probe",
            "tool_version": "0.1.0",
        },
        "connection": {
            "ip": ip,
            "port": port,
            "username": user,
        },
    }

    # --- A. identity ----------------------------------------------------
    section("A. Device identity")
    info = core.get_device_info(conn)
    print(f"  Manufacturer    : {info.manufacturer}")
    print(f"  Model           : {info.model}")
    print(f"  Firmware        : {info.firmware_version}")
    print(f"  Serial          : {info.serial_number}")
    print(f"  Hardware ID     : {info.hardware_id}")
    report["device"] = asdict(info)

    section("A2. Services advertised")
    services = core.get_services(conn)
    for svc in services:
        print(f"  {svc['version']:>6}  {svc['namespace']}")
    report["services"] = services

    if not conn.ptz_service:
        print("\n  ⚠️   No PTZ service on this camera. Probe ends here.")
        _write_reports(report, yaml_path, json_path)
        return 0

    # --- B. profiles ----------------------------------------------------
    section("B. Media profiles")
    if not conn.profiles:
        print("  (no profiles)")
    for p in conn.profiles:
        marker = " ←active" if p.token == conn.active_profile_token else ""
        ptz_flag = "ptz" if p.has_ptz else "no-ptz"
        print(f"  [{p.index}] {p.name!r:<30}  {ptz_flag:<6}  token={p.token}{marker}")
    report["profiles"] = [asdict(p) for p in conn.profiles]
    report["active_profile_token"] = conn.active_profile_token

    if len(conn.profiles) > 1 and not args.no_prompts:
        if ask_yn("Multiple profiles — pick a different one for PTZ?", default=False):
            idx = int(ask("Profile index to activate", "0"))
            sel = core.select_profile(conn, idx)
            if sel:
                print(f"  → active: {sel.name} ({sel.token})")
                report["active_profile_token"] = sel.token

    # --- C. capability discovery ---------------------------------------
    section("C. PTZ service capabilities (GetServiceCapabilities)")
    svc_caps = core.get_service_capabilities(conn)
    for k, v in svc_caps.items():
        if k == "raw":
            continue
        print(f"  {k:<35}: {v}")
    report["ptz_service_capabilities"] = svc_caps

    section("D. PTZ node capabilities (GetNodes)")
    nodes = core.get_ptz_nodes(conn)
    if not nodes:
        print("  (no PTZ nodes returned)")
    for nc in nodes:
        print(f"\n  Node {nc.node_token!r}  name={nc.name!r}")
        print(f"    continuous_pan_tilt : {nc.supports_continuous_pan_tilt}")
        print(f"    continuous_zoom     : {nc.supports_continuous_zoom}")
        print(f"    relative_pan_tilt   : {nc.supports_relative_pan_tilt}")
        print(f"    relative_zoom       : {nc.supports_relative_zoom}")
        print(f"    absolute_pan_tilt   : {nc.supports_absolute_pan_tilt}")
        print(f"    absolute_zoom       : {nc.supports_absolute_zoom}")
        print(f"    home_position       : supported={nc.supports_home_position} "
              f"fixed={nc.fixed_home_position}")
        print(f"    max_presets         : {nc.maximum_number_of_presets}")
        if nc.pan_tilt_spaces:
            print(f"    pan_tilt_spaces:")
            for sp in nc.pan_tilt_spaces:
                print(f"      • {sp}")
        if nc.zoom_spaces:
            print(f"    zoom_spaces:")
            for sp in nc.zoom_spaces:
                print(f"      • {sp}")
    report["ptz_nodes"] = [asdict(n) for n in nodes]

    declared = _declared_capabilities(nodes)
    report["declared_capabilities_union"] = declared

    # --- E. status reading ---------------------------------------------
    section("E. GetStatus — does the cam report position & move-state?")
    st = core.get_status(conn)
    print(f"  pan/tilt/zoom        : {st.pan}, {st.tilt}, {st.zoom}")
    print(f"  move_status pan_tilt : {st.move_status_pan_tilt}")
    print(f"  move_status zoom     : {st.move_status_zoom}")
    print(f"  utc_time             : {st.utc_time}")
    print(f"  error                : {st.error}")
    report["initial_status"] = asdict(st)

    # --- E2. Stream sight-check ----------------------------------------
    section("E2. Live stream — confirm you can SEE the cam before moving it")
    print()
    snapshot_uri = core.get_snapshot_uri(conn)
    print("  Before any move runs, please open ONE of these in a viewer so")
    print("  you can watch the camera in real time:")
    print()
    for p in conn.profiles:
        if p.stream_uri:
            marker = " ←active" if p.token == conn.active_profile_token else ""
            print(f"    • RTSP ({p.name}): {p.stream_uri}{marker}")
    if snapshot_uri:
        print(f"    • Snapshot JPEG: {snapshot_uri}")
    print(f"    • go2rtc UI (if running locally): http://localhost:1984")
    print(f"    • WMB stream page (if app is running): http://localhost:8050")
    print()
    print("  Quick RTSP preview from terminal (Mac with ffplay installed):")
    if conn.profiles and conn.profiles[0].stream_uri:
        print(f"    ffplay -fflags nobuffer -rtsp_transport tcp \\")
        print(f"      {conn.profiles[0].stream_uri}")
    print()
    report["stream_check"] = {
        "rtsp_uris": [p.stream_uri for p in conn.profiles if p.stream_uri],
        "snapshot_uri": snapshot_uri,
    }
    if not ask_yn("Do you have a live view and can SEE the camera now?",
                  default=True):
        print()
        print("  ⚠️   Without a live view you cannot evaluate move tests.")
        print("       Aborting before any cam state is changed.")
        return 0

    # --- F. presets & Home safety check --------------------------------
    section("F. Presets")
    presets = core.get_presets(conn)
    if presets:
        for p in presets:
            print(f"  • token={p.token!r:<10}  name={p.name!r}")
    else:
        print("  (no presets defined)")
    report["presets"] = [asdict(p) for p in presets]

    home_token = _ensure_home_preset(conn, presets, args.no_prompts)
    if home_token is None:
        print("\n  ⚠️   No Home preset set. Move tests will be SKIPPED for safety.")
        report["home_preset_token"] = None
        report["move_tests_skipped"] = True
        _write_reports(report, yaml_path, json_path)
        return 0
    report["home_preset_token"] = home_token
    report["move_tests_skipped"] = False

    # Save the capability-discovery half NOW so Ctrl-C during move tests
    # still leaves a useful report behind.
    _write_reports(report, yaml_path, json_path)

    # --- G. move tests -------------------------------------------------
    move_results: list[dict[str, Any]] = []
    if args.no_prompts or ask_yn("Start move tests now?", default=True):
        # Persist progress after EVERY test result so Ctrl-C never loses
        # data. The reports get rewritten atomically each time.
        def _save_progress(current_results: list[dict[str, Any]]) -> None:
            report["move_tests"] = list(current_results)
            _write_reports(report, yaml_path, json_path)

        move_results = _run_move_tests(
            conn,
            home_token=home_token,
            declared=declared,
            skip_absolute=args.skip_absolute,
            skip_relative=args.skip_relative,
            skip_continuous=args.skip_continuous,
            save_progress=_save_progress,
        )
    else:
        print("  → move tests skipped by operator")
    report["move_tests"] = move_results

    # --- H. summary ----------------------------------------------------
    section("H. Summary")
    summary = _summarise(report)
    for line in summary["lines"]:
        print(f"  {line}")
    report["summary"] = summary

    # --- I. overall notes (free-form) ---------------------------------
    section("I. Overall notes")
    if not args.no_prompts:
        print()
        print("  Anything else worth recording about this cam? E.g.")
        print("    - visible jitter on stop")
        print("    - stream lag in seconds")
        print("    - mechanical noises during specific moves")
        print("    - anything that didn't fit a single test case")
        print()
        print("  Leave blank to skip.")
        print()
        report["overall_notes"] = ask_comment()
    else:
        report["overall_notes"] = ""
    print()

    print()
    print(f"  → returning to Home preset before exit")
    core.goto_preset(conn, home_token, settle_sec=2.0)

    _write_reports(report, yaml_path, json_path)
    print()
    print(f"  📄 Reports written:")
    print(f"     {yaml_path}")
    print(f"     {json_path}")
    print(f"     {log_path}")

    # Canonical WMB cache write: when --camera-id is set, write the
    # empirical summary to OUTPUT_DIR/ptz_capabilities/cam<id>.yaml.
    # WMB's Settings UI reads from exactly this path to render tri-state
    # capability pills.
    #
    # OUTPUT_DIR resolution order:
    #   1. --output-dir flag (explicit, wins)
    #   2. $WMB_OUTPUT_DIR env var
    #   3. WMB's own config.get_config()['OUTPUT_DIR'] — works zero-config
    #      when the script is run from inside a WMB checkout (Mac dev or
    #      RPi prod), because the WMB package is then on sys.path
    output_dir_arg = args.output_dir or os.getenv("WMB_OUTPUT_DIR", "")
    if args.camera_id is not None and not output_dir_arg:
        try:
            from config import get_config as _wmb_get_config

            output_dir_arg = str(_wmb_get_config().get("OUTPUT_DIR", "") or "")
            if output_dir_arg:
                print(f"  ↳ Using WMB config OUTPUT_DIR: {output_dir_arg}")
        except Exception:
            # WMB not importable from here (probe run standalone) — that's
            # fine, we just fall through to the "no output_dir" warning.
            output_dir_arg = ""

    if args.camera_id is not None and output_dir_arg:
        cache_path = _write_wmb_capabilities_cache(
            report, int(args.camera_id), Path(output_dir_arg)
        )
        if cache_path is not None:
            print(f"     {cache_path}  ← WMB Settings-UI cache")
        else:
            print(
                f"  ⚠ Could not write WMB cache to "
                f"{Path(output_dir_arg) / 'ptz_capabilities'}"
            )
    elif args.camera_id is not None and not output_dir_arg:
        print(
            "  ⚠ --camera-id set but no OUTPUT_DIR resolvable "
            "(--output-dir, $WMB_OUTPUT_DIR, or WMB config) — "
            "skipping WMB cache write."
        )
    elif args.camera_id is None and output_dir_arg:
        print(
            "  ⚠ --output-dir / WMB_OUTPUT_DIR set but no --camera-id — "
            "skipping WMB cache write (need both)."
        )

    return 0


# ---------------------------------------------------------------------------
# Home preset workflow
# ---------------------------------------------------------------------------


def _ensure_home_preset(
    conn: core.Connection,
    presets: list[core.PresetInfo],
    no_prompts: bool,
) -> str | None:
    """Return the token of a Home preset, creating one if needed.

    Looks for an existing preset named 'Home' (case-insensitive). If none
    exists, asks the operator to manually aim the cam and confirms before
    saving the current position as 'Home'. Returns None if the operator
    declines.
    """
    section("F2. Home preset (safety net)")
    for p in presets:
        if p.name.strip().lower() in {"home", "wmb_home", "ptz_probe_home"}:
            print(f"  Found existing Home preset: token={p.token!r} name={p.name!r}")
            if no_prompts or ask_yn("Use this as Home?", default=True):
                return p.token

    print()
    print("  No 'Home' preset found. The probe will refuse to run move tests")
    print("  unless you set one now — it is the safety fallback used between")
    print("  test blocks.")
    if no_prompts:
        print("  --no-prompts is active; skipping Home setup.")
        return None
    if not ask_yn("Set current camera position as Home now?", default=True):
        return None

    print()
    print(f"  Operator's slot convention:")
    print(f"    1–7   preset-mode zones")
    print(f"    8     auto-re-focus")
    print(f"    11–17 grid-mode cells")
    print(f"    20+   free for ad-hoc use")
    print()
    print(f"  Cam declares max 32 presets but lists 256 — slots above ~32 may")
    print(f"  silently no-op or modulo-map. Default slot {DEFAULT_HOME_SLOT} is")
    print(f"  clear of all conventions AND well within the 32-slot hardware limit.")
    print()
    slot_str = ask(f"Which preset slot to use for Home?", str(DEFAULT_HOME_SLOT))
    try:
        slot_num = int(slot_str)
    except ValueError:
        print(f"  Invalid slot number — falling back to {DEFAULT_HOME_SLOT}.")
        slot_num = DEFAULT_HOME_SLOT
    requested_token = f"Preset{slot_num:03d}"
    print(f"  → will REQUEST cam to save Home as token={requested_token!r}")
    print(f"     (cam may return a different token — we will verify)")

    print()
    print()
    print("  ➜  Aim the camera manually to a safe overview position now.")
    print("      (use the cam's native app, or the WMB UI if it is running)")
    print()
    ask_confirm("Cam aimed at the Home position? Press Enter to SAVE it:")
    print()

    result, returned_token = core.set_preset(
        conn, preset_name="ptz_probe_home", preset_token=requested_token,
    )
    if not result.success:
        print(f"  SetPreset failed: {result.error}")
        return None
    print(f"  → cam returned: token={returned_token!r}")
    if returned_token != requested_token:
        print(f"  ⚠️  Cam ignored our token request "
              f"(asked for {requested_token!r}, got {returned_token!r})")

    # Verify by physically moving away and back, then asking the operator.
    if not returned_token:
        print("  ⚠️  No token returned — cannot verify.")
        return None

    print()
    print()
    print("  ── Verification: does the returned token actually point to your aim? ──")
    print()
    print("  Verification has TWO separate steps. After each step the probe")
    print("  pauses so you can look at the camera before the next step fires:")
    print()
    print("    Step 1: drive the cam visibly AWAY from Home (1.5s pan-right).")
    print("            You confirm the cam has actually moved away.")
    print(f"    Step 2: send GotoPreset({returned_token!r}) and wait for the cam")
    print(f"            to return to YOUR ORIGINAL AIM. You confirm the cam")
    print(f"            is back where you set it.")
    print()
    ask_confirm("Ready for Step 1 (drive AWAY from Home)? Press Enter to start:")
    print()
    print(f"  Step 1: driving cam pan-right (ContinuousMove, 1.5s burst)")
    core.continuous_move(conn, pan=0.2, duration_sec=1.5)
    time.sleep(1.0)  # let mechanical settle
    print(f"  Step 1 complete — cam should now be visibly RIGHT of Home.")
    print()
    if not ask_yn("Has the cam visibly moved AWAY from Home?", default=True):
        print()
        print("  ⚠️   Cam did not visibly move on ContinuousMove —")
        print("       cannot verify Home preset. Falling through.")
        return None

    print()
    ask_confirm("Ready for Step 2 (return to Home)? Press Enter to start:")
    print()
    print(f"  Step 2: GotoPreset({returned_token!r}) — waiting "
          f"{HOME_SETTLE_SEC:.0f}s for cam to arrive")
    core.goto_preset(conn, returned_token, settle_sec=HOME_SETTLE_SEC)
    print()
    print()
    print("  Now look at the camera:")
    print("    - If it returned to YOUR ORIGINAL AIM → Home is verified.")
    print("    - If it is somewhere else → SetPreset did not stick at the")
    print(f"      requested slot {requested_token!r}.")
    print()
    if ask_yn("Is the camera back AT THE POSITION you originally aimed for?",
              default=True):
        print(f"  ✓  Home verified: token={returned_token!r}")
        return returned_token
    print()
    print(f"  ✗  Verification FAILED.")
    print(f"     The cam saved your aim into a slot that's NOT what we requested,")
    print(f"     OR the SetPreset write was a no-op (cam returned OK but stored")
    print(f"     nothing). This is the {requested_token!r}-doesn't-stick problem.")
    print()
    print(f"     Recommended next step: pick a slot inside the declared 32 limit")
    print(f"     (any of 20–32 is safe — clear of your conventions and within")
    print(f"     the cam's hardware preset limit).")
    return None


# ---------------------------------------------------------------------------
# Move-test sequence
# ---------------------------------------------------------------------------


def _run_move_tests(
    conn: core.Connection,
    *,
    home_token: str,
    declared: dict[str, bool],
    skip_absolute: bool,
    skip_relative: bool,
    skip_continuous: bool,
    save_progress: Any = None,
) -> list[dict[str, Any]]:
    """The sequenced move-test plan. Each step prompts for feedback.

    If save_progress is callable, it is called after every test result
    with the current results list — used for incremental save so the
    operator never loses progress on Ctrl-C.
    """
    results: list[dict[str, Any]] = []

    def go_home() -> None:
        print(f"    ↩  returning to Home preset ({HOME_SETTLE_SEC:.0f}s settle)")
        r = core.goto_preset(conn, home_token, settle_sec=HOME_SETTLE_SEC)
        if not r.success:
            print(f"    !! goto_preset(Home) failed: {r.error}")

    def _persist() -> None:
        if save_progress is not None:
            try:
                save_progress(results)
            except Exception as exc:
                logger.warning("incremental save failed: %s", exc)

    def _run_block(cases: list[dict[str, Any]]) -> bool:
        """Run a block of test cases. Returns True if completed, False if
        the operator chose to skip the rest of the block."""
        for case in cases:
            # Repeat loop: a single case may be run multiple times if the
            # operator wants. Each run REPLACES the previous result for
            # that case in the results list.
            placeholder_idx = len(results)
            results.append({"case_id": case["id"], "feedback": "pending"})
            _persist()

            while True:
                res = _run_one_case(conn, case)
                results[placeholder_idx] = res
                _persist()  # save IMMEDIATELY after every result

                if res["feedback"] == "skip":
                    return False  # skip rest of block
                if res["feedback"] == "skip-test":
                    go_home()
                    break  # next case in block
                if res["feedback"] == "repeat":
                    print()
                    print("    🔁 repeating this test — returning home first")
                    go_home()
                    continue  # re-run same case, will overwrite result
                # feedback was yes / no / anything else → next case
                go_home()
                break
        return True

    # ── Continuous ────────────────────────────────────────────────────
    if skip_continuous:
        print("\n  [continuous tests skipped by flag]")
    elif not declared["continuous_pan_tilt"] and not declared["continuous_zoom"]:
        print("\n  [continuous not declared in GetNodes — skipping]")
    else:
        section("G1. Continuous move tests")
        _print_block_briefing(
            "Block 1 — ContinuousMove (velocity + duration model)",
            [
                "These tests send the cam a velocity vector and then a Stop after",
                "a fixed duration. This is what WatchMyBirds uses today. We want",
                "to know: does velocity scaling work, does Stop actually stop, and",
                "does the cam respect the sign of pan/tilt/zoom values.",
                f"Cases in this block: {len(_continuous_cases(declared))}.",
                "Before each move:  Enter = run | s = skip this test | S = skip rest of block.",
                "After each move:  Y = correct / n = wrong / r = repeat | s = skip rest.",
                "Free-text comment is optional after each.",
            ],
        )
        _run_block(_continuous_cases(declared))

    # ── Relative ──────────────────────────────────────────────────────
    if skip_relative:
        print("\n  [relative tests skipped by flag]")
    elif not declared["relative_pan_tilt"] and not declared["relative_zoom"]:
        print("\n  [relative not declared in GetNodes — skipping]")
    else:
        section("G2. Relative move tests")
        _print_block_briefing(
            "Block 2 — RelativeMove (amplitude ladder + reproducibility)",
            [
                "These tests ask the cam to move BY a given amount from its current",
                "position. Settle window is 4s per case (cam has no MoveStatus).",
                "The amplitude ladder (+0.02 → +0.05 → +0.10) tests whether the",
                "Translation parameter is honoured as a discrete position delta",
                "or interpreted as a velocity-style endless move.",
                f"Cases in this block: {len(_relative_cases(declared))}.",
                "If the FIRST relative test goes endless, hit S to skip the rest",
                "of this block — all amplitudes will behave the same way.",
                "Before each move:  Enter = run | s = skip this test | S = skip rest of block.",
            ],
        )
        _run_block(_relative_cases(declared))

    # ── Absolute ──────────────────────────────────────────────────────
    if skip_absolute:
        print("\n  [absolute tests skipped by flag]")
    elif not declared["absolute_pan_tilt"] and not declared["absolute_zoom"]:
        print("\n  [absolute not declared in GetNodes — skipping]")
    else:
        section("G3. Absolute move tests")
        _print_block_briefing(
            "Block 3 — AbsoluteMove (position model)",
            [
                "These tests ask the cam to move TO a specific coordinate.",
                "If this works reproducibly, we can address world positions",
                "directly — the richest auto-PTZ strategy.",
                "WARNING: on some cheap cams AbsoluteMove(0,0) is NOT the centre",
                "but the last endstop. The first test centres the cam — watch it.",
                "If it flies into a wall, hit Ctrl-C (emergency stop) and answer 'n'.",
                f"Cases in this block: {len(_absolute_cases(declared))}.",
            ],
        )
        if not ask_yn("Run absolute-move tests?", default=False):
            print("  → absolute tests skipped by operator")
        else:
            _run_block(_absolute_cases(declared))

    # ── MoveStatus polling test ───────────────────────────────────────
    section("G4. MoveStatus polling — does GetStatus actually update?")
    print()
    print("  This issues a continuous pan-right for 1.0s then samples")
    print("  GetStatus every 200ms for 3 seconds. Watching whether")
    print("  MoveStatus is MOVING during the move and IDLE after.")
    if ask_yn("Run MoveStatus poll test?", default=True):
        poll_result = _run_movestatus_poll(conn)
        results.append(poll_result)
        go_home()

    return results


def _run_one_case(conn: core.Connection, case: dict[str, Any]) -> dict[str, Any]:
    print()
    print()
    print(f"  ▶  {case['id']}")
    print(f"     test    : {case['description']}")
    if case.get("expectation"):
        print(f"     expect  : {case['expectation']}")
    if case.get("purpose"):
        # Wrap purpose to keep terminal readable
        purpose = case["purpose"]
        for line in _wrap_lines(purpose, prefix_first="     why     : ",
                                prefix_rest="              "):
            print(line)

    # Gate before move: operator can run, skip just this test, or skip the
    # whole block. Stray-char immunity via stdin flush inside ask_go_or_skip.
    print()
    gate = ask_go_or_skip(
        "Enter = run move | s = skip this test | S = skip rest of block:"
    )
    if gate == "skip-test":
        print("  → skipping this test (no move issued)")
        return {
            "case_id": case["id"],
            "description": case["description"],
            "expectation": case.get("expectation", ""),
            "purpose": case.get("purpose", ""),
            "kind": case["kind"],
            "input": {k: case[k] for k in case
                      if k not in ("id", "description", "kind",
                                   "expectation", "purpose")},
            "onvif_success": None,
            "feedback": "skip-test",
            "comment": "",
        }
    if gate == "skip-block":
        print("  → skipping rest of this block (no move issued)")
        return {
            "case_id": case["id"],
            "description": case["description"],
            "expectation": case.get("expectation", ""),
            "purpose": case.get("purpose", ""),
            "kind": case["kind"],
            "input": {k: case[k] for k in case
                      if k not in ("id", "description", "kind",
                                   "expectation", "purpose")},
            "onvif_success": None,
            "feedback": "skip",
            "comment": "",
        }
    print()

    print(f"     pre-status: {_short_status(core.get_status(conn))}")
    kind = case["kind"]
    if kind == "continuous":
        result = core.continuous_move(
            conn,
            pan=case.get("pan", 0.0),
            tilt=case.get("tilt", 0.0),
            zoom=case.get("zoom", 0.0),
            duration_sec=case.get("duration_sec", 0.3),
        )
    elif kind == "relative":
        result = core.relative_move(
            conn,
            pan=case.get("pan", 0.0),
            tilt=case.get("tilt", 0.0),
            zoom=case.get("zoom", 0.0),
            speed=case.get("speed", 0.5),
            settle_sec=case.get("settle_sec", 1.5),
        )
    elif kind == "absolute":
        result = core.absolute_move(
            conn,
            pan=case.get("pan", 0.0),
            tilt=case.get("tilt", 0.0),
            zoom=case.get("zoom", 0.0),
            speed=case.get("speed", 0.5),
            settle_sec=case.get("settle_sec", 2.0),
        )
    else:
        result = core.MoveResult(success=False, command=case["id"], error=f"unknown kind {kind}")

    print(f"     onvif-call : {result.command}  → success={result.success}")
    if result.error:
        print(f"     error      : {result.error}")
    print(f"     post-status: {_short_status(result.status_after)}")

    # Drift observation only after Continuous moves — Relative/Absolute
    # have their own settling baked into the ONVIF call.
    drift_observation = ""
    if kind == "continuous" and result.success:
        print()
        print("     The probe just sent the Stop command. Now WATCH THE CAMERA")
        print("     for 2 seconds — the cam SHOULD be standing perfectly still.")
        print("     If it keeps moving for a moment after Stop, that is 'drift'")
        print("     and it would make smooth tracking impossible (the camera")
        print("     would always overshoot its target by that much).")
        time.sleep(2.0)
        print()
        print()
        print("  → Question: after the Stop command, did the camera ...")
        print("     S = stop cleanly, perfectly still (the good case)")
        print("     t = drift a TINY bit after Stop (a hair's worth)")
        print("     l = keep moving NOTICEABLY after Stop (a large drift)")
        print()
        drift_observation = ask_drift("After Stop, did the cam")

    print()
    print()
    print(f"  → Question: was the move correct?")
    print(f"     (expected: {case.get('expectation', '<no expectation set>')})")
    print()
    feedback = ask_feedback("Did the camera do the expected thing?")
    if feedback in ("skip", "repeat"):
        # No comment prompt for skip/repeat — operator wants to move on
        # (skip) or re-run the test (repeat).
        comment = ""
    else:
        print()
        print()
        print("  → Optional: anything worth noting about this test?")
        print()
        comment = ask_comment()
    print()
    input_fields = {
        k: case[k] for k in case
        if k not in ("id", "description", "kind", "expectation", "purpose")
    }
    return {
        "case_id": case["id"],
        "description": case["description"],
        "expectation": case.get("expectation", ""),
        "purpose": case.get("purpose", ""),
        "kind": kind,
        "input": input_fields,
        "onvif_success": result.success,
        "onvif_error": result.error,
        "duration_sec": result.duration_sec,
        "status_before": asdict(result.status_before) if result.status_before else None,
        "status_after": asdict(result.status_after) if result.status_after else None,
        "feedback": feedback,
        "comment": comment,
        "post_stop_drift_observed": drift_observation,
    }


def _run_movestatus_poll(conn: core.Connection) -> dict[str, Any]:
    """Issue a 1s continuous pan, then sample status for 3s. Report transitions."""
    print()
    print("    ▶  issuing ContinuousMove(pan=+0.3, 1.0s)")
    move = core.continuous_move(conn, pan=0.3, duration_sec=1.0)
    print(f"       → {move.command} success={move.success}")
    print("    ▶  polling GetStatus every 200ms for 3.0s after stop")
    samples = core.poll_move_status(conn, max_wait_sec=3.0, poll_interval_sec=0.2)
    transitions: list[str] = []
    last: str | None = None
    for s in samples:
        cur = s.move_status_pan_tilt or "?"
        if cur != last:
            transitions.append(cur)
            last = cur
    print(f"       transitions seen: {' → '.join(transitions) if transitions else '(none)'}")
    print()
    print()
    print("  → Question: did MoveStatus actually transition?")
    print("     Expected a series like IDLE → MOVING → IDLE. If you see only")
    print("     one value (e.g. IDLE), the cam does not report move state and")
    print("     wait-until-idle tracking is impossible on this firmware.")
    print()
    feedback = ask_feedback("Did MoveStatus actually transition (not stuck at one value)?")
    if feedback != "skip":
        print()
        print()
        print("  → Optional: anything to note about the transition pattern?")
        print()
        comment = ask_comment()
    else:
        comment = ""
    print()
    return {
        "case_id": "movestatus_poll",
        "description": "ContinuousMove pan+0.3 for 1s, poll GetStatus 3s",
        "expectation": "Series of GetStatus samples should show MoveStatus going "
                       "from MOVING (during the move) to IDLE (after the stop), "
                       "not stuck on one value.",
        "purpose": "Tells us whether closed-loop tracking (wait-until-idle) is "
                   "possible at all on this cam, or whether we must rely on "
                   "fixed-time settling.",
        "kind": "diagnostic",
        "input": {"pan": 0.3, "duration_sec": 1.0, "poll_window_sec": 3.0},
        "onvif_success": move.success,
        "onvif_error": move.error,
        "samples": [asdict(s) for s in samples],
        "transitions": transitions,
        "feedback": feedback,
        "comment": comment,
    }


# ---------------------------------------------------------------------------
# Test-case generators
# ---------------------------------------------------------------------------


def _continuous_cases(declared: dict[str, bool]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if declared["continuous_pan_tilt"]:
        cases += [
            {
                "id": "c_pan_right_slow",
                "description": "pan right, slow velocity, 300ms burst",
                "kind": "continuous", "pan": 0.2, "duration_sec": 0.3,
                "expectation": "Camera turns slightly to the right (a few degrees), "
                               "then stops within ~300ms.",
                "purpose": "Baseline: does ContinuousMove work at all with a small "
                           "velocity and short duration? This is the move shape "
                           "WatchMyBirds uses today.",
            },
            {
                "id": "c_pan_right_fast",
                "description": "pan right, FAST velocity, 300ms burst",
                "kind": "continuous", "pan": 0.8, "duration_sec": 0.3,
                "expectation": "Camera snaps clearly to the right, then stops. "
                               "Should move noticeably more than the slow burst.",
                "purpose": "Tests whether the velocity scalar actually scales — "
                           "many cheap cams ignore the value and run at fixed speed.",
            },
            {
                "id": "c_pan_left_slow",
                "description": "pan left, slow velocity, 300ms burst",
                "kind": "continuous", "pan": -0.2, "duration_sec": 0.3,
                "expectation": "Camera turns slightly to the LEFT, then stops.",
                "purpose": "Sign-correctness check — does negative pan = left?",
            },
            {
                "id": "c_tilt_up_slow",
                "description": "tilt up, slow velocity, 300ms burst",
                "kind": "continuous", "tilt": 0.2, "duration_sec": 0.3,
                "expectation": "Camera tilts upward, then stops.",
                "purpose": "Confirms tilt works and positive tilt = up "
                           "(some cams invert).",
            },
            {
                "id": "c_tilt_down_slow",
                "description": "tilt down, slow velocity, 300ms burst",
                "kind": "continuous", "tilt": -0.2, "duration_sec": 0.3,
                "expectation": "Camera tilts downward, then stops.",
                "purpose": "Tilt sign-correctness in the other direction.",
            },
            {
                "id": "c_pan_long",
                "description": "pan right, slow velocity, 1500ms burst",
                "kind": "continuous", "pan": 0.2, "duration_sec": 1.5,
                "expectation": "Camera turns right for ~1.5s, smooth and steady, "
                               "then stops cleanly without overshooting.",
                "purpose": "Stop-timing check — if the cam keeps drifting after "
                           "the Stop() call, ContinuousMove tracking will overshoot.",
            },
        ]
    if declared["continuous_zoom"]:
        cases += [
            {
                "id": "c_zoom_in_slow",
                "description": "zoom in, slow velocity, 300ms burst",
                "kind": "continuous", "zoom": 0.2, "duration_sec": 0.3,
                "expectation": "Camera zooms in slightly (image gets bigger).",
                "purpose": "Confirms continuous zoom works.",
            },
            {
                "id": "c_zoom_out_slow",
                "description": "zoom out, slow velocity, 300ms burst",
                "kind": "continuous", "zoom": -0.2, "duration_sec": 0.3,
                "expectation": "Camera zooms out (image gets wider).",
                "purpose": "Zoom sign-correctness — does negative zoom = wider?",
            },
        ]
    return cases


def _relative_cases(declared: dict[str, bool]) -> list[dict[str, Any]]:
    """Amplitude ladder: tiny → small → medium, plus reproducibility check.

    Settle time is 4.0s (vs 1.5s in v1) because run 1 showed the cam needs
    much longer than the previous default for slow translations to finish.
    If +0.02 produces a visible tiny step, RelativeMove honours Translation
    as a position delta. If +0.02 keeps moving identically to +0.10, we have
    proof Translation is being interpreted as velocity-style on this cam.
    """
    cases: list[dict[str, Any]] = []
    if declared["relative_pan_tilt"]:
        cases += [
            {
                "id": "r_pan_right_tiny",
                "description": "relative pan +0.02 (TINY step right)",
                "kind": "relative", "pan": 0.02, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera nudges very slightly to the right — barely "
                               "perceptible step, NOT a continuous move.",
                "purpose": "Amplitude ladder rung 1. If this is visible-but-tiny, "
                           "Translation IS a position delta. If this 'keeps moving' "
                           "like the prior run's +0.10, Translation is being "
                           "interpreted as velocity-style on this cam.",
            },
            {
                "id": "r_pan_right_tiny_repeat",
                "description": "relative pan +0.02 AGAIN (reproducibility check)",
                "kind": "relative", "pan": 0.02, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera nudges the SAME tiny step again.",
                "purpose": "Reproducibility — if Δ1 ≠ Δ2, RelativeMove is useless "
                           "for closed-loop tracking. Most important test for the "
                           "relative-mode auto-PTZ plan.",
            },
            {
                "id": "r_pan_right_small",
                "description": "relative pan +0.05 (small step right)",
                "kind": "relative", "pan": 0.05, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera moves slightly more than the +0.02 step, "
                               "still a discrete movement that stops on its own.",
                "purpose": "Amplitude ladder rung 2. Does the step size scale?",
            },
            {
                "id": "r_pan_right_medium",
                "description": "relative pan +0.10 (medium step right)",
                "kind": "relative", "pan": 0.10, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera moves a clearly visible step right and "
                               "stops on its own within ~4 seconds.",
                "purpose": "Amplitude ladder rung 3 — the size that 'kept moving' "
                           "in run 1. With the longer settle window we should now "
                           "see if it actually completes or keeps drifting forever.",
            },
            {
                "id": "r_pan_left_small",
                "description": "relative pan -0.05 (small step LEFT)",
                "kind": "relative", "pan": -0.05, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera moves slightly to the LEFT.",
                "purpose": "Sign-correctness for relative pan.",
            },
            {
                "id": "r_tilt_up_small",
                "description": "relative tilt +0.05 (small step up)",
                "kind": "relative", "tilt": 0.05, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera tilts slightly upward.",
                "purpose": "Relative tilt works + positive = up.",
            },
            {
                "id": "r_tilt_down_small",
                "description": "relative tilt -0.05 (small step down)",
                "kind": "relative", "tilt": -0.05, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera tilts slightly downward.",
                "purpose": "Relative tilt sign-correctness in the other direction.",
            },
        ]
    if declared["relative_zoom"]:
        cases += [
            {
                "id": "r_zoom_in_small",
                "description": "relative zoom +0.05 (small zoom in)",
                "kind": "relative", "zoom": 0.05, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera zooms in by a small notch.",
                "purpose": "Relative zoom works + positive = closer.",
            },
            {
                "id": "r_zoom_out_small",
                "description": "relative zoom -0.05 (small zoom out)",
                "kind": "relative", "zoom": -0.05, "speed": 0.5, "settle_sec": 4.0,
                "expectation": "Camera zooms out by a small notch (back roughly "
                               "to where it was before).",
                "purpose": "Relative zoom sign-correctness.",
            },
        ]
    return cases


def _absolute_cases(declared: dict[str, bool]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if declared["absolute_pan_tilt"]:
        cases += [
            {
                "id": "a_center",
                "description": "absolute pan=0, tilt=0 (move to centre of pan/tilt range)",
                "kind": "absolute", "pan": 0.0, "tilt": 0.0, "speed": 0.5,
                "expectation": "Camera moves to its CENTRE position (whatever that "
                               "means for this cam — often the mounted-forward direction).",
                "purpose": "Does the cam have a meaningful (0,0) origin? Some cheap "
                           "cams treat (0,0) as 'last endstop', which would be a bug.",
            },
            {
                "id": "a_right_half",
                "description": "absolute pan=+0.5, tilt=0",
                "kind": "absolute", "pan": 0.5, "tilt": 0.0, "speed": 0.5,
                "expectation": "Camera moves to roughly 50% to the right of centre.",
                "purpose": "Tests whether absolute coordinates address a stable "
                           "world position — key requirement for Schritt 3.",
            },
            {
                "id": "a_left_half",
                "description": "absolute pan=-0.5, tilt=0",
                "kind": "absolute", "pan": -0.5, "tilt": 0.0, "speed": 0.5,
                "expectation": "Camera moves to roughly 50% to the left of centre.",
                "purpose": "Confirms the absolute coordinate space is symmetric.",
            },
        ]
    if declared["absolute_zoom"]:
        cases += [
            {
                "id": "a_zoom_quarter",
                "description": "absolute zoom=0.25 (modest zoom)",
                "kind": "absolute", "zoom": 0.25, "speed": 0.5,
                "expectation": "Camera zooms to ~25% of its zoom range.",
                "purpose": "Absolute zoom positioning works.",
            },
            {
                "id": "a_zoom_zero",
                "description": "absolute zoom=0 (fully wide)",
                "kind": "absolute", "zoom": 0.0, "speed": 0.5,
                "expectation": "Camera zooms fully out to widest view.",
                "purpose": "Confirms zoom=0 is the wide end (not, say, the tele end).",
            },
        ]
    return cases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _declared_capabilities(nodes: list[core.NodeCapabilities]) -> dict[str, bool]:
    """Union across all nodes — if any node supports X, treat X as available."""
    d = {
        "continuous_pan_tilt": False,
        "continuous_zoom": False,
        "relative_pan_tilt": False,
        "relative_zoom": False,
        "absolute_pan_tilt": False,
        "absolute_zoom": False,
        "home_position": False,
    }
    for n in nodes:
        d["continuous_pan_tilt"] |= n.supports_continuous_pan_tilt
        d["continuous_zoom"] |= n.supports_continuous_zoom
        d["relative_pan_tilt"] |= n.supports_relative_pan_tilt
        d["relative_zoom"] |= n.supports_relative_zoom
        d["absolute_pan_tilt"] |= n.supports_absolute_pan_tilt
        d["absolute_zoom"] |= n.supports_absolute_zoom
        d["home_position"] |= n.supports_home_position
    return d


def _wrap_lines(text: str, *, prefix_first: str, prefix_rest: str,
                width: int = 76) -> list[str]:
    """Word-wrap text into terminal-friendly lines with two prefixes."""
    import textwrap
    body_width = max(20, width - len(prefix_first))
    wrapped = textwrap.wrap(text, width=body_width) or [""]
    out = [f"{prefix_first}{wrapped[0]}"]
    for line in wrapped[1:]:
        out.append(f"{prefix_rest}{line}")
    return out


def _print_block_briefing(title_line: str, body_lines: list[str]) -> None:
    """Print a multi-line briefing block before each move-test block."""
    print()
    print()
    print(f"  ╔══ {title_line}")
    for line in body_lines:
        for wrapped in _wrap_lines(line, prefix_first="  ║   ", prefix_rest="  ║   "):
            print(wrapped)
    print(f"  ╚{'═' * 70}")
    print()
    print()
    ask_confirm("Ready for this block? Press Enter to start (Ctrl-C to abort):")
    print()


def _short_status(s: core.StatusSample | None) -> str:
    if s is None:
        return "(none)"
    if s.error:
        return f"error={s.error}"
    return (
        f"pan={s.pan} tilt={s.tilt} zoom={s.zoom} "
        f"mv_pt={s.move_status_pan_tilt} mv_z={s.move_status_zoom}"
    )


def _summarise(report: dict[str, Any]) -> dict[str, Any]:
    """Build the verdict block: what works, what doesn't, recommended strategy."""
    results = report.get("move_tests", []) or []
    declared = report.get("declared_capabilities_union", {}) or {}

    def yes_count(prefix: str) -> int:
        return sum(
            1 for r in results
            if r.get("case_id", "").startswith(prefix) and r.get("feedback") == "yes"
        )

    def any_yes(prefix: str) -> bool:
        return yes_count(prefix) > 0

    empirical = {
        "continuous_works": any_yes("c_"),
        "relative_works": any_yes("r_"),
        "absolute_works": any_yes("a_"),
        "movestatus_transitions": any(
            r.get("case_id") == "movestatus_poll" and r.get("feedback") == "yes"
            for r in results
        ),
    }

    relative_repro = None
    by_id = {r.get("case_id"): r for r in results}
    base = by_id.get("r_pan_right_tiny")
    repeat = by_id.get("r_pan_right_tiny_repeat")
    if base and repeat and base.get("feedback") == "yes" and repeat.get("feedback") == "yes":
        a1 = (base.get("status_after") or {}).get("pan")
        b0 = (base.get("status_before") or {}).get("pan")
        a2 = (repeat.get("status_after") or {}).get("pan")
        b2 = (repeat.get("status_before") or {}).get("pan")
        if a1 is not None and b0 is not None and a2 is not None and b2 is not None:
            d1 = float(a1) - float(b0)
            d2 = float(a2) - float(b2)
            relative_repro = {
                "delta_first": d1,
                "delta_second": d2,
                "abs_difference": abs(d1 - d2),
            }

    # Strategy verdict, ordered by preference for the WMB use-case:
    # absolute > relative > continuous > preset-only.
    if empirical["absolute_works"]:
        strategy = "absolute"
    elif empirical["relative_works"]:
        strategy = "relative"
    elif empirical["continuous_works"]:
        strategy = "continuous_pulse"
    else:
        strategy = "presets_only"

    lines = [
        f"declared by cam : continuous={declared.get('continuous_pan_tilt') or declared.get('continuous_zoom')}, "
        f"relative={declared.get('relative_pan_tilt') or declared.get('relative_zoom')}, "
        f"absolute={declared.get('absolute_pan_tilt') or declared.get('absolute_zoom')}",
        f"empirically OK : continuous={empirical['continuous_works']}, "
        f"relative={empirical['relative_works']}, "
        f"absolute={empirical['absolute_works']}",
        f"movestatus     : transitions correctly = {empirical['movestatus_transitions']}",
        f"recommended    : {strategy}",
    ]
    if relative_repro is not None:
        lines.append(
            f"relative repro : Δ1={relative_repro['delta_first']:.4f}  "
            f"Δ2={relative_repro['delta_second']:.4f}  "
            f"|Δ1-Δ2|={relative_repro['abs_difference']:.4f}"
        )

    return {
        "declared": declared,
        "empirical": empirical,
        "relative_reproducibility": relative_repro,
        "recommended_strategy": strategy,
        "lines": lines,
    }


def _write_reports(report: dict[str, Any], yaml_path: Path, json_path: Path) -> None:
    """Write the report atomically (temp file + rename) to BOTH paths.

    Atomic so a Ctrl-C mid-write cannot leave a partial/corrupt report.
    Called after every test result so the user never loses work.
    """
    import os

    def default(o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return str(o)

    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with json_tmp.open("w") as f:
        json.dump(report, f, indent=2, default=default)
    os.replace(json_tmp, json_path)

    yaml_tmp = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    with yaml_tmp.open("w") as f:
        f.write("# PTZ Probe report — auto-generated\n")
        yaml.dump(report, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    os.replace(yaml_tmp, yaml_path)


def _write_wmb_capabilities_cache(
    report: dict[str, Any], camera_id: int, output_dir: Path
) -> Path | None:
    """Write the canonical WMB empirical-capabilities cache.

    Path: ``<output_dir>/ptz_capabilities/cam<id>.yaml``. WMB's
    ``core.ptz_core._load_empirical_from_disk`` reads exactly this
    file to render the Settings-UI tri-state pills (green=declared+
    empirical✓, red=declared but empirical✗, etc).

    Schema (kept minimal — only what WMB actually consumes):

        camera_id: 0
        probed_at: <ISO-ish timestamp from the report meta>
        connection:
          ip: <cam ip>
        empirical:
          continuous_works: bool
          relative_works:   bool
          absolute_works:   bool
          movestatus_transitions: bool
        recommended_strategy: <continuous_pulse | relative | absolute | presets_only>

    Returns the written path, or None on failure (caller decides
    whether to surface that to the operator — the probe should not
    abort on cache-write failure).
    """
    import os

    summary = report.get("summary") or {}
    empirical = summary.get("empirical")
    if not empirical:
        # Nothing to publish yet — probably the user aborted before
        # any move tests ran.
        return None

    cache_dir = output_dir / "ptz_capabilities"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    payload = {
        "camera_id": int(camera_id),
        "probed_at": (report.get("meta") or {}).get("timestamp") or "",
        "connection": {
            "ip": (report.get("connection") or {}).get("ip") or "",
        },
        "empirical": {
            "continuous_works": bool(empirical.get("continuous_works", False)),
            "relative_works": bool(empirical.get("relative_works", False)),
            "absolute_works": bool(empirical.get("absolute_works", False)),
            "movestatus_transitions": bool(
                empirical.get("movestatus_transitions", False)
            ),
        },
        "recommended_strategy": str(summary.get("recommended_strategy") or ""),
    }

    target = cache_dir / f"cam{int(camera_id)}.yaml"
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write("# WMB empirical PTZ capabilities — written by ptz_probe\n")
            yaml.dump(
                payload,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        os.replace(tmp, target)
    except OSError:
        return None

    return target


def _setup_logging(log_path: Path, verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)

    # Quieten zeep noise unless --verbose.
    if not verbose:
        for name in ("zeep", "zeep.transports", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe an ONVIF PTZ camera for capabilities and behaviour.",
    )
    parser.add_argument("--ip", help="Camera IP address")
    parser.add_argument("--port", type=int, help="ONVIF port (commonly 80 or 8080)")
    parser.add_argument("--user", help="ONVIF username")
    parser.add_argument("--password", help="ONVIF password (omit to be prompted)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging to stderr in addition to log file")
    parser.add_argument("--no-prompts", action="store_true",
                        help="Skip interactive prompts where defaults exist "
                             "(used for non-interactive smoke runs).")
    parser.add_argument("--skip-continuous", action="store_true",
                        help="Skip the continuous-move test block")
    parser.add_argument("--skip-relative", action="store_true",
                        help="Skip the relative-move test block")
    parser.add_argument("--skip-absolute", action="store_true",
                        help="Skip the absolute-move test block")
    parser.add_argument("--camera-id", type=int,
                        help="WMB camera_id this probe corresponds to. "
                             "When set together with --output-dir, the "
                             "probe also writes "
                             "OUTPUT_DIR/ptz_capabilities/cam<id>.yaml — "
                             "the canonical empirical-capabilities cache "
                             "WMB's Settings UI reads.")
    parser.add_argument("--output-dir",
                        help="WMB OUTPUT_DIR. When set together with "
                             "--camera-id, the probe writes the canonical "
                             "empirical-capabilities cache there. Default: "
                             "$WMB_OUTPUT_DIR if set, else nothing written.")
    args = parser.parse_args()

    try:
        return run_probe(args)
    except KeyboardInterrupt:
        print("\n  interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
