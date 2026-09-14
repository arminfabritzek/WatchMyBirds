"""Regression coverage for bounded retention SQL variable usage."""

import datetime as dt
import sqlite3

import pytest

from core import retention_core
from tests.retention_helpers import seed_image
from utils.db.connection import closing_connection

NOW = dt.datetime(2026, 6, 1, 12, 0, tzinfo=dt.UTC)
SETTINGS = {
    "RETENTION_ENABLED": True,
    "RETENTION_DAYS": 90,
    "RETENTION_PROTECT_FAVORITES": True,
    "RETENTION_PROTECT_UNREVIEWED": True,
}


@pytest.fixture(autouse=True)
def configured_output(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())


def test_planner_chunks_above_sqlite_variable_limit(tmp_path):
    output_dir = tmp_path / "output"
    protected_indexes = {3, 27, 51}
    missing_derivative_index = 12
    with closing_connection() as conn:
        for index in range(55):
            seed_image(
                conn,
                f"20260101_12{index:04d}_batch.jpg",
                output_dir,
                favorite=index in protected_indexes,
                write_thumbs=index != missing_derivative_index,
            )

        previous_limit = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 40)
        try:
            plan = retention_core.build_plan(conn, str(output_dir), SETTINGS, now=NOW)
        finally:
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)

    assert len(plan.deletable) == 51
    assert plan.protected_counts["export_relevant"] == 3
    assert plan.protected_counts["missing_derivative"] == 1
