"""Storage-accounting service construction without web or infrastructure imports."""

from config import get_config
from core.retention_core import _settings_from_config
from core.storage_scan import StorageScan


def create() -> StorageScan:
    return StorageScan(str(get_config()["OUTPUT_DIR"]), _settings_from_config)
