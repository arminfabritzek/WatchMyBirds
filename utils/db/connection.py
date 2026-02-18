"""
Database Connection and Schema Management.

This module handles SQLite connection creation and schema initialization.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import get_config

DB_FILENAME = "images.db"

# Module-level cache: initialize schema once per database path.
# Tests patch OUTPUT_DIR, so schema init must be keyed by db path (not process-global).
_schema_initialized_paths: set[Path] = set()


def _get_db_path() -> Path:
    cfg = get_config()
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / DB_FILENAME


def get_connection() -> sqlite3.Connection:
    global _schema_initialized_paths
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    if db_path not in _schema_initialized_paths:
        _init_schema(conn)
        _schema_initialized_paths.add(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def closing_connection():
    """Context manager that creates a DB connection and guarantees it is closed.

    IMPORTANT: `with sqlite3.Connection as conn:` only manages transactions
    (commit/rollback) — it does NOT call conn.close(). This context manager
    ensures the file descriptor is released when the block exits.

    Usage:
        with closing_connection() as conn:
            conn.execute("SELECT ...")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
    _ensure_column_on_table(conn, "detections", "bbox_x", "REAL")
    _ensure_column_on_table(conn, "detections", "bbox_y", "REAL")
    _ensure_column_on_table(conn, "detections", "bbox_w", "REAL")
    _ensure_column_on_table(conn, "detections", "bbox_h", "REAL")
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

    # Frame resolution at capture time (tracks camera/resolution changes)
    _ensure_column_on_table(conn, "detections", "frame_width", "INTEGER")
    _ensure_column_on_table(conn, "detections", "frame_height", "INTEGER")



    # Detection Quality Rating (1-5 stars, computed or manual)
    _ensure_column_on_table(conn, "detections", "rating", "INTEGER")
    _ensure_column_on_table(conn, "detections", "rating_source", "TEXT DEFAULT 'auto'")


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

    # Deep Scan tracking (additive, no destructive migration)
    _ensure_column_on_table(conn, "images", "deep_scan_last_attempt_at", "TEXT")
    _ensure_column_on_table(conn, "images", "deep_scan_last_result", "TEXT")
    _ensure_column_on_table(conn, "images", "deep_scan_attempt_count", "INTEGER DEFAULT 0")

    # 1. Ensure Default Source Exists
    default_source_id = get_or_create_default_source(conn)
    # 2. Backfill existing images
    conn.execute(
        "UPDATE images SET source_id = ? WHERE source_id IS NULL", (default_source_id,)
    )


    conn.execute("""
        CREATE TABLE IF NOT EXISTS species_meta (
            scientific_name TEXT PRIMARY KEY,
            image_url TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Weather History Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temp_c REAL,
            precip_mm REAL,
            wind_kph REAL,
            condition_code INTEGER,
            is_day INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_weather_ts ON weather_logs(timestamp DESC);"
    )

    # Inbox ingest audit log (skip reasons, etc.). This must not affect gallery/review.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inbox_ingest_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            inbox_filename TEXT NOT NULL,
            content_hash TEXT,
            status TEXT NOT NULL,
            reason TEXT,
            source_id INTEGER,
            image_filename TEXT,
            details_json TEXT
        );
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inbox_ingest_events_created_at ON inbox_ingest_events(created_at DESC);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inbox_ingest_events_status ON inbox_ingest_events(status);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inbox_ingest_events_hash ON inbox_ingest_events(content_hash);"
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
