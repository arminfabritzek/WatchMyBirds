"""Retention Service — thin web-layer wrapper over core.retention_core.

Imports only core (H-01 enforcement test forbids utils.* here); the DB
connection is opened inside core.retention_core.
"""

from typing import Any

from core import retention_core


def preview() -> dict[str, Any]:
    """Preview the same deadlines the deletion job executes."""
    return retention_core.preview()


def start_run() -> dict[str, Any]:
    """Start the registered retention job, sharing its overlap lock."""
    if retention_core.activation_date(retention_core._settings_from_config()) is None:
        return {"status": "disabled", "reason": "Automatic deletion is off"}
    from web.services.nightly_job_hub import run_now

    return run_now("retention", reason="manual retention trigger")


def is_original_retention_deleted(filename: str) -> bool:
    """True iff the original for `filename` was removed by retention."""
    return retention_core.is_original_retention_deleted(filename)
