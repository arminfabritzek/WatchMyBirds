"""Storage-accounting service construction without web or infrastructure imports."""

from dataclasses import asdict
from typing import Any

from config import get_config
from core.deletion_plan import DEFAULT_SETTINGS
from core.retention_core import (
    _settings_from_config,
    activation_date,
    load_last_run_status,
)
from core.storage_scan import StorageScan


def deletion_defaults() -> dict:
    return asdict(DEFAULT_SETTINGS)


def saving_state(manager: Any) -> dict:
    persistence = getattr(manager, "persistence_service", None)
    pressure = getattr(persistence, "storage_pressure", None)
    status = pressure.status() if pressure is not None else {}
    return status if isinstance(status, dict) else {}


def create() -> StorageScan:
    return StorageScan(str(get_config()["OUTPUT_DIR"]), _settings_from_config)


def deletion_state() -> dict:
    last = load_last_run_status()
    result = last.get("progress", {})
    return {
        "enabled": activation_date(_settings_from_config()) is not None,
        "last_run_text": (
            f"Last cleanup {last.get('last_finished_at')}: {result.get('deleted', 0)} files, "
            f"{result.get('deleted_rows', 0)} rows, {result.get('freed_bytes', 0)} bytes removed; "
            f"{result.get('errors', 0)} errors."
            if last.get("last_finished_at")
            else "No cleanup run recorded."
        ),
    }
