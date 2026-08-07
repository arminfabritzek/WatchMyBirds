"""Shared test isolation.

Several process-global caches survive between tests and leak one test's
paths into the next:

- ``config._CONFIG`` is loaded once, and ~17 modules bind the resulting
  dict at import time (``config = get_config()``). Rebinding
  ``config._CONFIG`` therefore does *not* reach them — they still hold
  the old dict. Because every module shares that one object, the fix is
  to replace its *contents* in place rather than the reference.
- ``utils.path_manager._instance`` caches OUTPUT_DIR on first use. The
  metadata burn-in path resolves originals through it rather than
  through the caller's path, so a stale instance makes one test read a
  previous test's tmp_path.
- The compute lease service is process-global and raises when a second
  ``init`` arrives with a different detection_manager.

Individual test modules may still reset these themselves; doing so is
harmless and stays compatible with the autouse fixture here.
"""

from __future__ import annotations

import pytest

import config as config_module
from utils import path_manager
from utils.db import connection as db_connection


def reset_config_in_place() -> None:
    """Force a config reload that every already-imported module can see.

    Modules that did ``config = get_config()`` at import time hold a
    reference to the dict, not to ``config._CONFIG``. Mutating the shared
    dict in place is what makes a monkeypatched env var reach them.
    """
    current = config_module._CONFIG
    config_module._CONFIG = None
    fresh = config_module.get_config()

    if current is not None and current is not fresh:
        current.clear()
        current.update(fresh)
        config_module._CONFIG = current


@pytest.fixture(autouse=True)
def _isolate_process_globals():
    """Drop cached singletons around every test."""
    path_manager._instance = None
    db_connection._schema_initialized_paths.clear()

    yield

    path_manager._instance = None
    db_connection._schema_initialized_paths.clear()

    try:
        from web.services import compute_lease_service
    except ImportError:  # pragma: no cover - service is optional in some runs
        return
    compute_lease_service.reset_compute_lease_service_for_testing()
