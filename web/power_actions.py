"""Shared helpers for reboot/shutdown actions via systemd/logind."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Literal

PowerAction = Literal["shutdown", "restart"]

_ACTION_CONFIG = {
    "shutdown": {
        "systemctl_action": "poweroff",
        "label": "Shutdown",
        "success_message": "System is shutting down...",
    },
    "restart": {
        "systemctl_action": "reboot",
        "label": "Restart",
        "success_message": "System is restarting...",
    },
}

POWER_MANAGEMENT_UNAVAILABLE_MESSAGE = (
    "Power management is intentionally disabled in non-systemd environments."
)


def is_power_management_available() -> bool:
    """Return True when systemd/systemctl power management is available."""
    return (
        os.path.isdir("/run/systemd/system") and shutil.which("systemctl") is not None
    )


def get_power_action_success_message(action: PowerAction) -> str:
    """Return the user-facing success message for a power action."""
    return _ACTION_CONFIG[action]["success_message"]


def schedule_power_action(
    action: PowerAction,
    logger,
    *,
    delay_seconds: float = 2.0,
) -> threading.Thread:
    """Schedule a delayed reboot/shutdown using systemctl via logind/polkit."""
    cfg = _ACTION_CONFIG[action]
    systemctl_action = cfg["systemctl_action"]
    label = cfg["label"]

    def _delayed() -> None:
        time.sleep(delay_seconds)
        try:
            subprocess.run(
                ["systemctl", "--no-ask-password", "--no-wall", systemctl_action],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.CalledProcessError as e:
            details = (e.stderr or e.stdout or "").strip()
            logger.error(f"{label} command failed (Exit {e.returncode}): {details}")
        except Exception as e:
            logger.error(f"{label} command failed: {e}")

    t = threading.Thread(target=_delayed, name=f"power-{action}", daemon=True)
    t.start()
    return t
