"""Storage-accounting service construction without web or infrastructure imports."""

from config import get_config
from core.retention_core import (
    _settings_from_config,
    activation_date,
    load_last_run_status,
)
from core.storage_scan import StorageScan


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
