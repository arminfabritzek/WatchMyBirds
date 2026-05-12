"""
Anonymous opt-in usage heartbeat.

Default OFF. The user must explicitly toggle telemetry on in
Settings -> Privacy. There is no banner, no popup, no nag.

What we send (only when enabled):
    {
      "installation_id":   <32 hex chars, generated once on first opt-in>,
      "app_version":       "v0.X.Y",
      "os":                "linux" | "darwin" | "windows",
      "arch":              "aarch64" | "x86_64" | "armv7l",
      "cpu_count":         int,
      "total_ram_gb":      int (rounded whole GB),
      "python_version":    "3.12.3",
      "detector_variant":  e.g. "yolox-tiny-int8" | "fasterrcnn" | "unknown"
    }

What we never send: IP, country, locale, hostname, MAC, exact RAM
bytes, kernel version, Pi model string, observation count, species
names, image paths, error messages, uptime.

Cadence: once per UTC date, idempotent (the Worker's PRIMARY KEY on
(installation_id, date) absorbs duplicate pings).

See docs/PRIVACY.md and the project's Settings -> Privacy page for
the human-readable version.
"""

from __future__ import annotations

import logging
import os
import platform
import secrets
import sys
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Locked at the protocol level so Worker UA-allowlist matches.
# Worker regex: /^WatchMyBirds-Heartbeat\/[\w.+-]+/
USER_AGENT = "WatchMyBirds-Heartbeat/1.0"

# Marker file lives next to settings.yaml so reset/migration is implicit
# (delete OUTPUT_DIR -> heartbeat history is also gone).
LAST_SENT_FILENAME = "telemetry_last_sent.txt"

# Module-level wake-up Event: lets the toggle endpoint poke the
# scheduler thread out of its sleep so the first heartbeat after
# toggle-on goes within ~10ms instead of waiting up to one full
# check_interval cycle.
_wake_event = threading.Event()


def wake_now() -> None:
    """Wake the scheduler thread out of its current sleep.

    Idempotent: safe to call when no scheduler is running, safe to
    call repeatedly. The scheduler's loop calls _wake_event.clear()
    at the top of each iteration, so a single set() triggers exactly
    one extra iteration rather than putting the loop into a tight
    spin.
    """
    _wake_event.set()


def _get_last_sent_path(output_dir: str) -> Path:
    return Path(output_dir) / LAST_SENT_FILENAME


def _read_last_sent_date(output_dir: str) -> str:
    """Returns the UTC date string ('YYYY-MM-DD') of the last successful
    ping, or empty string if never sent / file missing / corrupted.
    """
    path = _get_last_sent_path(output_dir)
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    # Defensive: only accept well-formed YYYY-MM-DD; anything else
    # treated as "never sent" so we don't get stuck.
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return raw
    return ""


def _write_last_sent_date(output_dir: str, date_str: str) -> None:
    """Atomically write today's UTC date to the marker file.

    Atomic via tempfile-in-same-dir + os.replace (POSIX rename is
    atomic on the same filesystem). Crash mid-write leaves either
    the old contents or the new contents — never a half-written file.
    """
    path = _get_last_sent_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".telemetry_last_sent.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(date_str)
        os.replace(tmp_path, path)
    except Exception:
        # If anything went wrong, clean up the temp file. We deliberately
        # do not retry — next scheduler tick will try again.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            # tmp_path never got created; nothing to clean up.
            pass
        raise


def wipe_last_sent(output_dir: str) -> None:
    """Remove the marker file. Used by the rotate-UUID action so the
    next scheduler tick will send under the new UUID.
    """
    path = _get_last_sent_path(output_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        # Already gone; the rotate-UUID intent is already satisfied.
        pass


def _detect_os() -> str:
    """Low-resolution OS family. No kernel version, no distro string."""
    sys_platform = sys.platform
    if sys_platform.startswith("linux"):
        return "linux"
    if sys_platform == "darwin":
        return "darwin"
    if sys_platform.startswith("win"):
        return "windows"
    return "other"


def _detect_arch() -> str:
    """Low-resolution arch. 'aarch64', 'x86_64', 'armv7l', or 'unknown'.
    No CPU model string, no microarch.
    """
    machine = (platform.machine() or "").lower()
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine.startswith("armv7"):
        return "armv7l"
    return "unknown"


def _detect_cpu_count() -> int:
    """Whole-number CPU count. Defaults to 1 if detection fails."""
    try:
        n = os.cpu_count()
        if n is None or n < 1:
            return 1
        return int(n)
    except Exception:
        return 1


def _detect_total_ram_gb() -> int:
    """Rounded whole-GB total RAM. Reads /proc/meminfo on Linux without
    requiring psutil; on other OSes returns 0 (signals 'unknown' to the
    aggregation side, which is honest).

    Rounding deliberately to whole GB (not 7.7 -> 7.7 GB) to avoid
    memory-config fingerprinting. Pi 5 8GB and Pi 5 4GB are
    distinguishable; Pi 5 8GB-with-half-used-for-zram and Pi 5 8GB are
    not.
    """
    try:
        # Linux fast path: /proc/meminfo doesn't need psutil and is
        # already what Pi runs on.
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # Format: "MemTotal:  16335152 kB"
                    kb = int(parts[1])
                    return max(0, round(kb / (1024 * 1024)))
    except (OSError, ValueError, IndexError):
        # /proc/meminfo unavailable (non-Linux) or malformed; try psutil.
        pass

    # Non-Linux fallback via psutil if available.
    try:
        import psutil  # type: ignore[import-untyped]

        return max(0, round(psutil.virtual_memory().total / (1024**3)))
    except Exception:
        return 0


def _detect_app_version() -> str:
    """Read APP_VERSION (env or file at repo root). Falls back to 'unknown'.

    Order: APP_VERSION env var (set by Docker/systemd) wins; otherwise
    look for an APP_VERSION file next to the launching script, then one
    level up.
    """
    try:
        env_ver = os.environ.get("APP_VERSION", "").strip()
        if env_ver:
            return env_ver
        repo_root = Path(sys.argv[0]).resolve().parent
        for candidate in (repo_root / "APP_VERSION", repo_root.parent / "APP_VERSION"):
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip() or "unknown"
    except (OSError, AttributeError):
        # sys.argv[0] resolution can fail in odd embeddings (e.g. REPL).
        pass
    return "unknown"


def _detect_python_version() -> str:
    """e.g. '3.12.3'."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _detect_detector_variant() -> str:
    """Best-effort detector identifier. Returns 'unknown' if we can't
    figure it out — that's a valid value the Worker accepts.

    Strategy (in order):
      1. Read model_metadata.json next to the active OD model (most
         authoritative — written by the detector loader at boot).
         Build a low-resolution family string from `framework` +
         `variant`, e.g. "yolox-s" or "yolox-tiny". Avoids leaking
         the full training-run identifier (which contains dates and
         hyperparameters and would over-fingerprint installs).
      2. Fall back to explicit config keys if someone set them.
      3. "unknown" if neither is available.
    """
    # Path 1: read model_metadata.json
    try:
        from config import get_config

        cfg = get_config()
        model_base = cfg.get("MODEL_BASE_PATH", "./data/models")
        meta_path = Path(model_base) / "object_detection" / "model_metadata.json"
        if meta_path.exists():
            import json

            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            framework = str(meta.get("framework", "")).strip().lower()
            variant = str(meta.get("variant", "")).strip().lower()
            if framework and variant:
                # e.g. "yolox-s", "yolox-tiny" — low-res family identifier,
                # no dates, no hyperparams, no training-run noise.
                return f"{framework}-{variant}"[:64]
            if framework:
                return framework[:64]
    except (OSError, ValueError, KeyError, ImportError):
        # model_metadata.json missing/malformed or config not yet loaded;
        # fall through to the explicit-config path.
        pass

    # Path 2: explicit config override (legacy / tests)
    try:
        from config import get_config

        cfg = get_config()
        for key in ("DETECTOR_VARIANT", "OD_VARIANT", "DETECTOR", "OD_BACKEND"):
            v = cfg.get(key)
            if v:
                return str(v).lower().replace(" ", "-")[:64]
    except (ImportError, AttributeError):
        # Config not initialised yet (e.g. very early boot).
        pass

    return "unknown"


def _ensure_installation_id(cfg: dict) -> str:
    """Lazily generate a 32-hex-char UUID on first opt-in and persist it.

    Generated only when telemetry is being enabled — never on import,
    never on first boot if telemetry stays off.
    """
    existing = str(cfg.get("telemetry_installation_id", "") or "").strip().lower()
    # Validate shape: 32 lowercase hex chars (matches Worker regex).
    if len(existing) == 32 and all(c in "0123456789abcdef" for c in existing):
        return existing

    # Generate + persist. We import locally to avoid a cycle at module load.
    from utils.settings import load_settings_yaml, save_settings_yaml

    new_id = secrets.token_hex(16)  # 32 hex chars
    yaml_settings = load_settings_yaml(str(cfg["OUTPUT_DIR"]))
    yaml_settings["telemetry_installation_id"] = new_id
    save_settings_yaml(yaml_settings, str(cfg["OUTPUT_DIR"]))
    logger.info("Telemetry: generated new installation_id (first opt-in).")
    return new_id


def rotate_installation_id(output_dir: str) -> str:
    """Wipe persisted UUID and last-sent marker, generate a fresh UUID.

    Called by the Settings UI rotate-button. The next scheduler tick
    will send under the new UUID. Old UUID's rows in D1 stay until the
    daily 90d-cleanup cron deletes them.
    """
    from utils.settings import load_settings_yaml, save_settings_yaml

    yaml_settings = load_settings_yaml(output_dir)
    new_id = secrets.token_hex(16)
    yaml_settings["telemetry_installation_id"] = new_id
    save_settings_yaml(yaml_settings, output_dir)
    wipe_last_sent(output_dir)
    logger.info("Telemetry: installation_id rotated by user request.")
    return new_id


def build_payload(cfg: dict) -> dict:
    """Build the heartbeat payload from current runtime state.

    Pure function, easy to unit-test. The shape MUST match the Worker's
    ALLOWED_FIELDS allowlist exactly — extra or missing keys cause 400.

    Side effect: lazily generates and persists installation_id on first
    opt-in via _ensure_installation_id. Use build_payload_preview()
    instead if you need a side-effect-free preview (UI/diagnostics).
    """
    return {
        "installation_id": _ensure_installation_id(cfg),
        "app_version": _detect_app_version(),
        "os": _detect_os(),
        "arch": _detect_arch(),
        "cpu_count": _detect_cpu_count(),
        "total_ram_gb": _detect_total_ram_gb(),
        "python_version": _detect_python_version(),
        "detector_variant": _detect_detector_variant(),
    }


def build_payload_preview(cfg: dict) -> dict:
    """Build a preview of the heartbeat payload WITHOUT side effects.

    Used by the Settings UI's "Show what would be sent" button so the
    operator can inspect the exact payload before opting in.

    Differences from build_payload():
      - Does NOT generate or persist a UUID. If no UUID exists yet,
        returns the placeholder "<would-be-generated-on-opt-in>" so
        the operator sees a real-shaped JSON without us silently
        mutating settings.yaml.
      - All other fields are real (live OS/arch/cpu/ram detection)
        so the preview matches what the live ping would actually
        send.

    This function MUST NOT write to settings.yaml or to the last-sent
    marker. It is read-only.
    """
    existing_id = str(cfg.get("telemetry_installation_id", "") or "").strip().lower()
    if len(existing_id) == 32 and all(c in "0123456789abcdef" for c in existing_id):
        installation_id = existing_id
    else:
        installation_id = "<would-be-generated-on-opt-in>"

    return {
        "installation_id": installation_id,
        "app_version": _detect_app_version(),
        "os": _detect_os(),
        "arch": _detect_arch(),
        "cpu_count": _detect_cpu_count(),
        "total_ram_gb": _detect_total_ram_gb(),
        "python_version": _detect_python_version(),
        "detector_variant": _detect_detector_variant(),
    }


def _today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _send_heartbeat(endpoint: str, payload: dict) -> bool:
    """POST the heartbeat. Returns True on success (HTTP 204), False
    otherwise. ALL exceptions and non-204 responses are swallowed —
    we never surface telemetry failures to the user.
    """
    try:
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
            timeout=5,
        )
    except requests.RequestException as e:
        # Network down, DNS fail, TLS handshake fail, timeout — all
        # silent. The day's ping is just lost; tomorrow we try again.
        logger.debug("Telemetry send failed (network): %s", e)
        return False

    if resp.status_code == 204:
        return True

    # 400 = our payload doesn't match the Worker's allowlist. This is
    # a bug worth knowing about in logs but not worth surfacing to the
    # user. 404 = UA allowlist mismatch (constant we control). Anything
    # else is server-side and equally not the user's problem.
    logger.debug(
        "Telemetry send returned %s (expected 204): %s",
        resp.status_code,
        resp.text[:200],
    )
    return False


def is_enabled(cfg: dict) -> bool:
    """Strict 'is the telemetry toggle on?' check.

    Default-OFF is enforced here: anything other than literal True
    (the boolean, after settings.yaml round-trip) returns False.
    Missing key returns False. String 'true' returns False (we want the
    Settings UI to write a real bool, not a string).
    """
    val = cfg.get("telemetry_enabled", False)
    return val is True


def start_telemetry_scheduler(check_interval: int = 300):
    """Start the background telemetry scheduler thread.

    Args:
        check_interval: Seconds between toggle/cadence checks.
                        Default 5 min — short enough that a runtime
                        toggle-on triggers the first heartbeat within
                        a few minutes (good UX), long enough to be
                        invisible CPU-wise (12 polls/h, each is one
                        filesystem read + a boolean compare). The
                        "one heartbeat per UTC day" guarantee is
                        enforced by the last-sent marker file, NOT
                        by this interval — so changing it does not
                        change cloud-side data volume.

    Returns the daemon Thread for tests/inspection.
    """

    def _loop():
        logger.info("Telemetry scheduler started.")
        while True:
            # Clear the wake-event at the TOP of each iteration so a
            # set() that arrives DURING this iteration's work still
            # wakes the next sleep. If we cleared after the wait()
            # below instead, a wake_now() call between "tick done" and
            # "wait starts" would be lost.
            _wake_event.clear()

            try:
                from config import get_config

                cfg = get_config()

                if not is_enabled(cfg):
                    # Toggle off — do nothing. We do NOT touch the
                    # last_sent marker here; if the user toggles back
                    # on later, we should NOT re-ping for a date we've
                    # already pinged for.
                    pass
                else:
                    output_dir = str(cfg.get("OUTPUT_DIR", "./data/output"))
                    today = _today_utc()
                    last_sent = _read_last_sent_date(output_dir)
                    if last_sent != today:
                        endpoint = str(
                            cfg.get(
                                "telemetry_endpoint",
                                "https://heartbeat-wmb.starmin.de/v1/heartbeat",
                            )
                        )
                        payload = build_payload(cfg)
                        if _send_heartbeat(endpoint, payload):
                            _write_last_sent_date(output_dir, today)
                            logger.info("Telemetry: heartbeat sent (date=%s).", today)
                        # On failure: do NOT mark sent. Next tick retries.

            except Exception as e:
                logger.error("Telemetry scheduler error: %s", e, exc_info=True)

            # Sleep until either check_interval elapses OR wake_now()
            # is called (toggle-on triggers an immediate next tick).
            _wake_event.wait(check_interval)

    t = threading.Thread(target=_loop, name="TelemetryScheduler", daemon=True)
    t.start()
    return t
