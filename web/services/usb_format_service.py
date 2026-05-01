"""
USB Format Service — Web-layer logic for formatting USB sticks from the UI.

Handles:
  - Discovering attached USB block devices (read-only enumeration)
  - Validating a target the operator selected
  - Triggering rpi/format_backup_stick.sh via wmb-format-backup.service
  - Reading status as it progresses

The actual destructive work (mkfs.ext4 + parted) lives in the bash
script and runs as root via systemd. This module only:
  1. Tells the operator what's attached (so they pick correctly)
  2. Sets the target via systemd's set-environment
  3. Starts the unit non-blocking
  4. Polls /opt/app/data/usb_format_status.json for state updates

The script's own hard guards (USB-only, size limits, /opt/app safety
check) are the load-bearing security boundary; this Python layer
performs the same checks defensively but the script never trusts
that the caller did them.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path
from typing import Any

from logging_config import get_logger

logger = get_logger(__name__)

# Mirrors rpi/format_backup_stick.sh's STATUS_FILE constant.
STATUS_FILE = Path("/opt/app/data/usb_format_status.json")
HISTORY_FILE = Path("/opt/app/data/usb_format_history.jsonl")
SERVICE_UNIT = "wmb-format-backup.service"

# Match /dev/sd[a-z] only (no partitions, no other patterns). Anything
# else is rejected at the API boundary.
_VALID_DEV_RE = re.compile(r"^/dev/sd[a-z]$")

# Match the trigger lock so a second click while a format is running
# doesn't start a parallel attempt.
_TRIGGER_LOCK = threading.Lock()


# ----------------------------------------------------------------------
# Discovery: which USB sticks are attached?
# ----------------------------------------------------------------------


def _udev_property(device: str, key: str) -> str | None:
    """Read a single udev property, e.g. ID_BUS or ID_MODEL."""
    try:
        result = subprocess.run(
            ["udevadm", "info", "--query=property", "--name", device],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1] or None
    return None


def list_usb_block_devices() -> list[dict[str, Any]]:
    """Enumerate USB block devices the operator could plausibly format.

    Filters strictly: only whole-disk /dev/sd[a-z] devices reporting
    ID_BUS=usb and the kernel's removable=1 flag. Internal SATA
    drives (which can also report as /dev/sda on some hardware) are
    excluded by the bus check.

    Returns dicts with the fields the UI needs to render a picker.
    """
    out: list[dict[str, Any]] = []

    # Tree mode: also pull each disk's children so partition-level LABEL
    # and FSTYPE (where NTFS / FAT / ext4 actually live on a partitioned
    # stick) are visible. With `-d` we'd only see the whole-disk row,
    # whose LABEL/FSTYPE are empty for any normally-partitioned stick.
    try:
        result = subprocess.run(
            [
                "lsblk", "-J", "-b",
                "-o", "NAME,SIZE,LABEL,FSTYPE,MODEL,VENDOR,RM,TYPE",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("lsblk failed: %s", exc)
        return []

    if result.returncode != 0:
        logger.warning("lsblk returned %d: %s", result.returncode, result.stderr)
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    for entry in data.get("blockdevices", []) or []:
        name = entry.get("name") or ""
        if not name.startswith("sd") or len(name) != 3:
            # Filters /dev/sda, /dev/sdb, etc. but not /dev/sda1.
            continue
        if entry.get("type") and entry.get("type") != "disk":
            continue

        device = f"/dev/{name}"

        # Kernel removable flag. RPi UAS-attached internal SATA shows
        # up here too, so we additionally check ID_BUS.
        if str(entry.get("rm") or "0") != "1":
            # Some USB sticks erroneously report rm=0; we still allow
            # them through if ID_BUS=usb confirms.
            pass

        bus = _udev_property(device, "ID_BUS")
        if bus != "usb":
            continue

        size_bytes = entry.get("size")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            continue

        # Whole-disk LABEL/FSTYPE only populate for unpartitioned
        # ("superfloppy") sticks. For normal NTFS/FAT/ext4 sticks the
        # filesystem lives on /dev/sd?1 -- fall back to the first
        # partition child so the picker can show "currently: NTFS".
        label = entry.get("label")
        fstype = entry.get("fstype")
        if not (label or fstype):
            for child in entry.get("children") or []:
                if child.get("type") == "part":
                    label = label or child.get("label")
                    fstype = fstype or child.get("fstype")
                    if label or fstype:
                        break

        out.append({
            "device": device,
            "size_bytes": size_bytes,
            "model": entry.get("model") or _udev_property(device, "ID_MODEL"),
            "vendor": entry.get("vendor") or _udev_property(device, "ID_VENDOR"),
            "current_label": label,
            "current_fstype": fstype,
            "is_already_wmb_backup": label == "WMB-BACKUP",
        })

    return out


# ----------------------------------------------------------------------
# Trigger
# ----------------------------------------------------------------------


def is_format_supported() -> bool:
    """Refuse to surface the format UI on platforms where we'd fail."""
    script = Path("/opt/app/rpi/format_backup_stick.sh")
    if not script.is_file():
        return False
    # systemctl available?
    try:
        result = subprocess.run(
            ["systemctl", "--version"],
            capture_output=True, timeout=3,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def trigger_format(target_device: str, confirm_token: str) -> tuple[bool, str]:
    """Validate and start the format service.

    Returns (started, message). Caller polls get_format_status() for
    progress.
    """
    if not is_format_supported():
        return False, "Formatting from the UI is not available on this build."

    if not _VALID_DEV_RE.match(target_device or ""):
        return False, f"Invalid target device: {target_device!r}"

    if confirm_token != "FORMAT":
        return False, "Confirmation token missing or incorrect."

    # Re-validate the target is on our discovery list (defense in depth:
    # the UI should only let the operator pick from the list, but a
    # crafted POST shouldn't bypass that).
    valid_devices = {d["device"] for d in list_usb_block_devices()}
    if target_device not in valid_devices:
        return False, (
            f"Target {target_device} is not a recognised USB stick on this Pi."
        )

    # Cross-process guard: if a previous run is still active (status
    # file shows a non-terminal state), refuse rather than queue. Our
    # in-process _TRIGGER_LOCK below catches concurrent clicks within
    # the same Flask worker; this catches the case where Flask was
    # restarted mid-format or the operator opened a second browser.
    current = get_format_status()
    active_states = {
        "starting", "validating", "unmounting", "wiping",
        "partitioning", "formatting", "mounting",
    }
    if current.get("state") in active_states:
        return False, (
            f"A format is already running (state: {current.get('state')}). "
            "Please wait for it to finish before starting another."
        )

    if not _TRIGGER_LOCK.acquire(blocking=False):
        return False, "A format operation is already starting."

    try:
        # Hand the target + confirmation to the root service via a
        # trigger file written from this process. Earlier versions
        # used `systemctl set-environment`, but that requires root
        # (or a global polkit grant) and would either fail outright
        # or open a far broader authorisation hole than necessary.
        # The trigger file lives in /opt/app/data, owned by the app
        # user, mode 0644 -- root reads it once and then refuses to
        # proceed if the values don't match the script's own guards.
        trigger_path = Path("/opt/app/data/usb_format_trigger.json")
        try:
            trigger_path.parent.mkdir(parents=True, exist_ok=True)
            trigger_path.write_text(
                json.dumps({
                    "target_device": target_device,
                    "confirm": "FORMAT",
                })
            )
        except OSError as exc:
            logger.error("Could not write format trigger file: %s", exc)
            return False, f"Could not stage format trigger: {exc}"

        # --no-block: returns immediately; status comes from the file.
        try:
            subprocess.run(
                ["systemctl", "--no-block", "start", SERVICE_UNIT],
                check=True, capture_output=True, text=True, timeout=10,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("systemctl start %s failed: %s", SERVICE_UNIT, exc.stderr)
            # Polkit failure usually has a recognisable phrase.
            err = exc.stderr or str(exc)
            if "polkit" in err.lower() or "authority" in err.lower():
                return False, (
                    "Polkit refused the format action. The "
                    "watchmybirds user is not authorised to start "
                    f"{SERVICE_UNIT}."
                )
            return False, f"Failed to start format service: {err}"

        logger.info(
            "Format service triggered: target=%s",
            target_device,
        )
        return True, f"Format started on {target_device}."
    finally:
        _TRIGGER_LOCK.release()


# ----------------------------------------------------------------------
# Calibration history
# ----------------------------------------------------------------------


def get_format_history(limit: int = 10) -> list[dict[str, Any]]:
    """Read the rolling format-duration history.

    Each line is a JSON object: {ts, size_bytes, seconds}. The bash
    script appends one line on every successful format and trims to the
    last 20. The UI uses these to calibrate its remaining-time estimate
    instead of relying on the static 2.5 s/GB heuristic forever.

    Returns the last `limit` entries (newest last). Tolerates a missing
    or malformed file -- absence just means "no calibration yet, fall
    back to the heuristic".
    """
    if not HISTORY_FILE.is_file():
        return []
    try:
        with HISTORY_FILE.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        size = entry.get("size_bytes")
        secs = entry.get("seconds")
        # Sanity floor: a sub-5 s "format" is almost certainly a
        # validation failure that wrote success in error -- skip it
        # so it doesn't poison the regression.
        if isinstance(size, int) and isinstance(secs, (int, float)) \
                and size > 0 and secs >= 5:
            out.append({
                "ts": entry.get("ts"),
                "size_bytes": size,
                "seconds": float(secs),
            })
    return out


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------


def get_format_status() -> dict[str, Any]:
    """Read /opt/app/data/usb_format_status.json or return idle.

    The bash script writes this file at every state transition. If
    no file exists, no format has been run (or the file was cleaned
    up). Either way, idle.
    """
    if not STATUS_FILE.is_file():
        return {
            "state": "idle",
            "message": None,
            "ts": None,
            "target": None,
        }
    try:
        with STATUS_FILE.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"state": "error", "message": "Invalid status file format."}
        return data
    except (OSError, json.JSONDecodeError) as exc:
        return {"state": "error", "message": f"Cannot read status: {exc}"}


def clear_format_status() -> bool:
    """Wipe the status file (operator-facing 'Acknowledge'/'Dismiss')."""
    try:
        STATUS_FILE.unlink(missing_ok=True)
        return True
    except OSError as exc:
        logger.warning("clear_format_status failed: %s", exc)
        return False
