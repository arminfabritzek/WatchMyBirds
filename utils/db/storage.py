"""Read-only storage diagnostics and bounded views for canonical readiness."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def open_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("BEGIN")
    return conn


def database_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    size = int(conn.execute("PRAGMA main.page_size").fetchone()[0])
    pages = int(conn.execute("PRAGMA main.page_count").fetchone()[0])
    free = int(conn.execute("PRAGMA main.freelist_count").fetchone()[0])
    tables = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM main.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    counts = {}
    for table in tables:
        escaped = table.replace('"', '""')
        counts[table] = int(
            conn.execute(f'SELECT COUNT(*) FROM main."{escaped}"').fetchone()[0]
        )
    first, last = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM main.images"
    ).fetchone()
    objects, availability = [], "available"
    try:
        objects = [
            dict(row)
            for row in conn.execute(
                "SELECT name, SUM(pgsize) AS bytes FROM main.dbstat GROUP BY name ORDER BY bytes DESC"
            )
        ]
    except sqlite3.OperationalError:
        availability = "unavailable"
    return dict(
        page_size=size,
        page_count=pages,
        reusable_pages=free,
        allocated_bytes=pages * size,
        reusable_bytes=free * size,
        records=counts,
        first_timestamp=first,
        last_timestamp=last,
        object_sizes=objects,
        object_sizes_status=availability,
    )


def image_batches(
    conn: sqlite3.Connection, batch_size: int = 200
) -> Iterator[list[dict[str, Any]]]:
    cursor = ""
    while True:
        rows = conn.execute(
            """SELECT i.filename, i.timestamp, i.review_status,
            COALESCE(i.original_present,1) AS original_present,
            EXISTS(SELECT 1 FROM main.detections d WHERE d.image_filename=i.filename
                AND d.status='active' AND d.is_favorite=1) AS is_favorite
            FROM main.images i WHERE i.filename > ? ORDER BY i.filename LIMIT ?""",
            (cursor, max(1, min(batch_size, 400))),
        ).fetchall()
        if not rows:
            return
        yield [dict(row) for row in rows]
        cursor = str(rows[-1]["filename"])


def prepare_canonical_views(conn: sqlite3.Connection) -> None:
    # TEMP objects shadow only this read-only connection's main tables. The
    # canonical exporter sees complete frames for a bounded image selection.
    conn.execute("CREATE TEMP TABLE storage_selection(filename TEXT PRIMARY KEY)")
    for table, column in [
        ("images", "filename"),
        ("detections", "image_filename"),
        ("label_subjects", "image_filename"),
        ("manual_objects", "image_filename"),
        ("current_human_label_facts", "image_filename"),
    ]:
        conn.execute(f'''CREATE TEMP VIEW "{table}" AS SELECT source.* FROM main."{table}" source
            JOIN temp.storage_selection selected ON selected.filename=source."{column}"''')
    conn.execute("""CREATE TEMP VIEW human_label_facts AS SELECT source.*
        FROM main.human_label_facts source JOIN temp.label_subjects subject ON subject.subject_id=source.subject_id""")
    conn.execute("""CREATE TEMP VIEW manual_object_revisions AS SELECT source.*
        FROM main.manual_object_revisions source JOIN temp.manual_objects object ON object.manual_object_id=source.manual_object_id""")


def select_images(conn: sqlite3.Connection, filenames: list[str]) -> None:
    conn.execute("DELETE FROM temp.storage_selection")
    conn.executemany(
        "INSERT INTO temp.storage_selection VALUES (?)", ((name,) for name in filenames)
    )


def historical_images(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("""SELECT DISTINCT s.image_filename
        FROM temp.human_label_facts f JOIN temp.label_subjects s ON s.subject_id=f.subject_id
        WHERE f.assertion_state='retracted' OR EXISTS (
            SELECT 1 FROM temp.human_label_facts successor WHERE successor.supersedes_fact_id=f.fact_id)""")
    }
