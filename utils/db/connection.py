"""
Database Connection and Schema Management.

This module handles SQLite connection creation and schema initialization.
"""

import sqlite3
from pathlib import Path

from config import get_config

DB_FILENAME = "images.db"


def _get_db_path() -> Path:
    cfg = get_config()
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / DB_FILENAME


def get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    _init_schema(conn)
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            coco_json TEXT,
            downloaded_timestamp TEXT,
            detector_model_id TEXT,
            classifier_model_id TEXT,
            source_id INTEGER REFERENCES sources(source_id),
            content_hash TEXT
        );
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp DESC);"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_class_name TEXT,
            od_confidence REAL,
            od_model_id TEXT,
            created_at TEXT,
            FOREIGN KEY(image_filename) REFERENCES images(filename) ON DELETE CASCADE
        );
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_filename ON detections(image_filename);"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            cls_model_id TEXT,
            rank INTEGER DEFAULT 1,
            created_at TEXT,
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
        );
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_classifications_detection_id ON classifications(detection_id);"
    )

    _ensure_column_on_table(conn, "detections", "status", "TEXT DEFAULT 'active'")
    _ensure_column_on_table(conn, "classifications", "status", "TEXT DEFAULT 'active'")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_status_filename ON detections(status, image_filename);"
    )

    _ensure_column_on_table(conn, "detections", "score", "REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_score ON detections(score DESC);"
    )

    _ensure_column_on_table(conn, "detections", "agreement_score", "REAL")
    _ensure_column_on_table(conn, "detections", "detector_model_name", "TEXT")
    _ensure_column_on_table(conn, "detections", "detector_model_version", "TEXT")
    _ensure_column_on_table(conn, "detections", "classifier_model_name", "TEXT")
    _ensure_column_on_table(conn, "detections", "classifier_model_version", "TEXT")
    _ensure_column_on_table(conn, "detections", "thumbnail_path", "TEXT")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            uri TEXT,
            config_json TEXT,
            active INTEGER DEFAULT 1
        );
        """)
    _ensure_column_on_table(
        conn, "images", "source_id", "INTEGER REFERENCES sources(source_id)"
    )

    _ensure_column_on_table(conn, "images", "content_hash", "TEXT")

    # Review Queue: review_status (untagged | confirmed_bird | no_bird)
    _ensure_column_on_table(conn, "images", "review_status", "TEXT DEFAULT 'untagged'")
    _ensure_column_on_table(conn, "images", "review_updated_at", "TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_review_status_timestamp ON images(review_status, timestamp);"
    )

    # 1. Ensure Default Source Exists
    default_source_id = get_or_create_default_source(conn)
    # 2. Backfill existing images
    conn.execute(
        "UPDATE images SET source_id = ? WHERE source_id IS NULL", (default_source_id,)
    )
    conn.commit()


def get_or_create_default_source(conn: sqlite3.Connection) -> int:
    """Gets the ID of the default 'Default Camera' source, or creates it if missing."""
    row = conn.execute(
        "SELECT source_id FROM sources WHERE name='Default Camera'"
    ).fetchone()
    if row:
        return row[0]

    # Create if not exists
    cur = conn.execute(
        "INSERT INTO sources (name, type) VALUES (?, ?)", ("Default Camera", "ipcam")
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_user_import_source(conn: sqlite3.Connection) -> int:
    """Gets the ID of the 'User Import' source, or creates it if missing."""
    row = conn.execute(
        "SELECT source_id FROM sources WHERE name='User Import'"
    ).fetchone()
    if row:
        return row[0]

    # Create if not exists
    cur = conn.execute(
        "INSERT INTO sources (name, type) VALUES (?, ?)",
        ("User Import", "folder_upload"),
    )
    conn.commit()
    return cur.lastrowid


def _ensure_column(conn: sqlite3.Connection, column: str, coltype: str) -> None:
    _ensure_column_on_table(conn, "images", column, coltype)


def _ensure_column_on_table(
    conn: sqlite3.Connection, table: str, column: str, coltype: str
) -> None:
    cur = conn.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype};")
        conn.commit()
    # Add index for content_hash if added via migration (or if missing and column exists)
    if column == "content_hash" and table == "images":
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);"
        )
        conn.commit()
