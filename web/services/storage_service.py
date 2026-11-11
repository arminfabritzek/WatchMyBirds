"""Storage-accounting service construction without web or infrastructure imports."""

from dataclasses import asdict
from typing import Any

from config import get_config
from core.retention_core import (
    _settings_from_config,
    activation_date,
    load_last_run_status,
    settings_for_plan,
)
from core.storage_budget import DEFAULT_CAPTURE_BUDGET_BYTES, photo_save_budget
from core.storage_scan import StorageScan


def deletion_defaults() -> dict:
    return asdict(settings_for_plan(get_config()))


def saving_state(manager: Any) -> dict:
    persistence = getattr(manager, "persistence_service", None)
    pressure = getattr(persistence, "storage_pressure", None)
    status = pressure.status() if pressure is not None else {}
    return status if isinstance(status, dict) else {}


def capture_budget(manager: Any) -> int:
    budget = DEFAULT_CAPTURE_BUDGET_BYTES
    stored = saving_state(manager).get("capture_budget_bytes")
    if isinstance(stored, int):
        budget = max(budget, stored)
    capture = getattr(manager, "video_capture", None)
    resolution = getattr(capture, "resolution", None)
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        width, height = resolution
        if all(isinstance(value, int) and value > 0 for value in resolution):
            budget = max(budget, photo_save_budget(width * height * 3))
    return budget


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
