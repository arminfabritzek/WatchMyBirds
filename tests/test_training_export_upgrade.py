from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask
from PIL import Image

from config import get_config
from core.deletion_plan import DEFAULT_SETTINGS, deletion_preview
from core.training_export import waiting_revisions
from tests.test_restore_database_merge import _make_db
from tests.test_surface_routes_local_db import local_db_app as local_db_app
from utils import backup
from utils.db import connection as db_connection
from utils.path_manager import PathManager
from utils.restore import _merge_database

FIXTURE = Path(__file__).parent / "fixtures/training_exports_0_5_2_via_0_5_9.sql"


@pytest.mark.parametrize("backup_kind", ["legacy_only", "current"])
def test_legacy_export_upgrade_download_receipt_backup_restore_and_merge(
    local_db_app: Flask, tmp_path: Path, backup_kind: str
) -> None:
    paths = PathManager(get_config()["OUTPUT_DIR"])
    source = sqlite3.connect(":memory:")
    source.executescript(FIXTURE.read_text())
    legacy_rows = source.execute("SELECT * FROM training_exports").fetchall()
    legacy_schema = source.execute(
        "SELECT sql FROM sqlite_master WHERE name='training_exports'"
    ).fetchone()[0]
    with sqlite3.connect(paths.get_database_path()) as restored:
        source.backup(restored)
    source.close()
    legacy_snapshot = tmp_path / "legacy.db"
    with patch.object(
        backup, "get_db_path", return_value=str(paths.get_database_path())
    ):
        assert backup._create_db_snapshot(legacy_snapshot)
    name = "20260101_120000_upgrade.jpg"
    original = paths.get_original_path(name)
    original.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 200)).save(original)
    db_connection._schema_initialized_paths.clear()
    with db_connection.closing_connection() as conn:
        assert len(waiting_revisions(conn, media_exists=lambda _: True)) == 1
        assert conn.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[0] == 0
        preview = deletion_preview(
            conn, DEFAULT_SETTINGS, datetime.now(UTC), paths=paths
        )
        assert preview["waiting"]["waiting_for_training"]["photos"] == 1
    with local_db_app.test_client() as client:
        with client.session_transaction() as session:
            session["authenticated"] = True
            session["_csrf_token"] = "upgrade-test"
        assert client.get("/admin/canonical-dataset").status_code == 200
        response = client.post(
            "/api/canonical-dataset/download",
            headers={"X-CSRF-Token": "upgrade-test"},
        )
        assert response.status_code == 200 and len(response.data) > 0
        with db_connection.closing_connection() as conn:
            export_id = conn.execute(
                "SELECT export_id FROM labeling_exports"
            ).fetchone()[0]
        again = client.get(f"/api/canonical-dataset/exports/{export_id}/download")
        assert again.status_code == 200 and again.data == response.data
        received = client.post(
            f"/api/canonical-dataset/exports/{export_id}/received",
            headers={"X-CSRF-Token": "upgrade-test"},
        )
        assert received.status_code == 303
        assert "Received" in client.get(received.location).get_data(as_text=True)
    with db_connection.closing_connection() as conn:
        assert [
            tuple(row) for row in conn.execute("SELECT * FROM training_exports")
        ] == legacy_rows
        assert (
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='training_exports'"
            ).fetchone()[0]
            == legacy_schema
        )
        assert not waiting_revisions(conn, media_exists=lambda _: True)
        assert (
            deletion_preview(conn, DEFAULT_SETTINGS, datetime.now(UTC), paths=paths)[
                "errors"
            ]
            == 0
        )
    snapshot = tmp_path / "current.db"
    with patch.object(
        backup, "get_db_path", return_value=str(paths.get_database_path())
    ):
        assert backup._create_db_snapshot(snapshot)
    with sqlite3.connect(snapshot) as restored:
        restored.row_factory = sqlite3.Row
        db_connection._init_schema(restored, existing_installation=True)
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert [
            tuple(row) for row in restored.execute("SELECT * FROM training_exports")
        ] == legacy_rows
        assert restored.execute("SELECT received_at FROM labeling_exports").fetchone()[
            0
        ]
    destination = _make_db(tmp_path, "destination")
    target_paths = PathManager(str(destination.parent))
    target_original = target_paths.get_original_path(name)
    target_original.parent.mkdir(parents=True, exist_ok=True)
    target_original.write_bytes(original.read_bytes())
    with patch.dict(get_config(), {"OUTPUT_DIR": str(destination.parent)}):
        selected_backup = legacy_snapshot if backup_kind == "legacy_only" else snapshot
        _merge_database(selected_backup)
        _merge_database(selected_backup)
        with db_connection.closing_connection() as merged:
            assert merged.execute("PRAGMA foreign_key_check").fetchall() == []
            assert len(waiting_revisions(merged, media_exists=lambda _: True)) == (
                backup_kind == "legacy_only"
            )
            assert merged.execute("SELECT COUNT(*) FROM labeling_exports").fetchone()[
                0
            ] == (backup_kind == "current")
            assert not merged.execute(
                "SELECT 1 FROM sqlite_master WHERE name='training_exports'"
            ).fetchone()
            assert (
                deletion_preview(
                    merged, DEFAULT_SETTINGS, datetime.now(UTC), paths=target_paths
                )["errors"]
                == 0
            )
