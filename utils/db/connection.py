"""
Database Connection and Schema Management.

This module handles SQLite connection creation and schema initialization.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import get_config

DB_FILENAME = "images.db"
SQLITE_MMAP_SIZE_BYTES = 2 * 1024 * 1024 * 1024

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
    # busy_timeout MUST be the very first PRAGMA on every new connection.
    # Every subsequent statement (`journal_mode=WAL`, `synchronous=NORMAL`,
    # `foreign_keys=ON`, the cache/mmap settings, even `optimize`) needs
    # to acquire a lock on the DB file briefly. If a writer (e.g. the
    # aesthetic-tagger bridge committing scores) is already holding a
    # write lock, those PRAGMAs throw `database is locked` instantly
    # unless busy_timeout has already been installed.
    #
    # 15000 ms: under combined detector + tagger + health-check load,
    # an SD-card-backed deploy can take longer than 5 s to release the
    # lock (fsync is the slow path). 15 s absorbs that; waiting once
    # is preferable to ERROR-spam in the log.
    conn.execute("PRAGMA busy_timeout=15000;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    # Keep the DB resident in memory rather
    # than re-reading pages from disk on every query. Originally proposed
    # by the upstream fork-PR; profiling on a representative dataset confirmed
    # the remaining hot path is dominated by SQL execute
    # + fetchall, which mmap collapses to memory access.
    #
    # - mmap_size=2 GB: maps current and near-future long-running DBs into
    #   the process's address space; reads become memcpy from the OS page
    #   cache. The mapping is virtual address space, not pre-allocated RAM.
    # - cache_size=100000 (negative would be KB; positive is page-count
    #   so 100k × 4 KB ≈ 400 MB cache, large enough for the working set).
    # - optimize=0x10002: enables both the always-on ANALYZE-when-needed
    #   logic and the optimize-on-close pass below.
    conn.execute(f"PRAGMA mmap_size={SQLITE_MMAP_SIZE_BYTES};")
    conn.execute("PRAGMA cache_size=100000;")
    conn.execute("PRAGMA optimize=0x10002;")
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
        # Let SQLite run any pending ANALYZE work cheaply on close;
        # paired with `PRAGMA optimize=0x10002;` in get_connection so
        # statistics stay current without a manual maintenance window.
        try:
            conn.execute("PRAGMA optimize;")
        except sqlite3.Error:
            # Stale conn or partial transaction; close anyway.
            pass
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
    conn.execute(
        """
        DELETE FROM classifications
        WHERE classification_id NOT IN (
            SELECT MAX(classification_id)
            FROM classifications
            GROUP BY detection_id, rank
        );
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_classifications_detection_rank ON classifications(detection_id, rank);"
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

    # Decision system fields (Plan v1)
    _ensure_column_on_table(conn, "detections", "decision_state", "TEXT")
    _ensure_column_on_table(conn, "detections", "bbox_quality", "REAL")
    _ensure_column_on_table(conn, "detections", "unknown_score", "REAL")
    _ensure_column_on_table(conn, "detections", "decision_reasons", "TEXT")
    _ensure_column_on_table(conn, "detections", "policy_version", "TEXT")
    _ensure_column_on_table(conn, "detections", "manual_species_override", "TEXT")
    _ensure_column_on_table(conn, "detections", "species_source", "TEXT")
    _ensure_column_on_table(conn, "detections", "species_updated_at", "TEXT")
    _ensure_column_on_table(conn, "detections", "manual_bbox_review", "TEXT")
    _ensure_column_on_table(conn, "detections", "bbox_reviewed_at", "TEXT")

    # CLS-v2 decision layer (added 2026-04-23 with classifier config YAMLs).
    # decision_level: 'species' | 'genus' | 'reject'. NULL for detections
    #   saved before the decision layer shipped — those are species-level
    #   by construction (the old pipeline had no genus fallback).
    # raw_species_name: top-1 species latin regardless of decision level.
    #   Lets us reconstruct what the classifier actually thought, even when
    #   cls_class_name was promoted to genus_sp. or cleared for reject.
    _ensure_column_on_table(conn, "detections", "decision_level", "TEXT")
    _ensure_column_on_table(conn, "detections", "raw_species_name", "TEXT")

    # Frame resolution at capture time (tracks camera/resolution changes)
    _ensure_column_on_table(conn, "detections", "frame_width", "INTEGER")
    _ensure_column_on_table(conn, "detections", "frame_height", "INTEGER")

    # Detection Quality Rating (1-5 stars, computed or manual)
    _ensure_column_on_table(conn, "detections", "rating", "INTEGER")
    _ensure_column_on_table(conn, "detections", "rating_source", "TEXT DEFAULT 'auto'")

    # Favorite flag (simple ❤️ toggle for cover image selection)
    _ensure_column_on_table(conn, "detections", "is_favorite", "INTEGER DEFAULT 0")

    # Aesthetic auto-tagging (nightly batch job, see scripts/aesthetic_tag_nightly.py).
    # aesthetic_score: float in [0, 1], CLIP zero-shot "facing camera" probability.
    # aesthetic_score_at: ISO-8601 timestamp of last computation; lets the job skip
    #   detections it has already scored on previous nights.
    _ensure_column_on_table(conn, "detections", "aesthetic_score", "REAL")
    _ensure_column_on_table(conn, "detections", "aesthetic_score_at", "TEXT")

    # is_gallery_eligible: boolean flag set by the aesthetic tagger to mark a detection
    #   as a model-picked gallery candidate. Kept strictly separate from is_favorite so
    #   that:
    #     - manual HUMAN favorites (is_favorite=1) remain a clean training gold-label
    #     - the tagger never overwrites HUMAN choices
    #     - gallery surfaces can render a "KI pick" badge on is_gallery_eligible=1 AND
    #       is_favorite=0 detections
    #   Backfill of legacy rating_source='auto' rows happens in _backfill_gallery_eligible
    #   (see below). See workflow/plans/2026-05-02_FEATURE_aesthetic-tagger-three-column-split.md.
    _ensure_column_on_table(
        conn, "detections", "is_gallery_eligible", "INTEGER DEFAULT 0"
    )
    _backfill_gallery_eligible(conn)
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_gallery_state_filename
        ON detections(status, lower(COALESCE(decision_state, '')), image_filename);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_gallery_state_score
        ON detections(status, lower(COALESCE(decision_state, '')), score DESC);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_detections_created_at_desc
        ON detections(created_at DESC);
        """
    )

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
    _ensure_column_on_table(
        conn, "images", "deep_scan_attempt_count", "INTEGER DEFAULT 0"
    )

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

    # Rescan Proposals — safe proposal persistence (never auto-overwrite)
    # Status flow: queued → ready → applied | discarded | failed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rescan_proposals (
            proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            target_detection_id INTEGER,
            image_filename TEXT NOT NULL,
            suggested_species TEXT,
            suggested_confidence REAL,
            suggested_score REAL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            topk_json TEXT,
            source_model TEXT,
            status TEXT DEFAULT 'queued',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            applied_at TEXT,
            FOREIGN KEY(image_filename) REFERENCES images(filename) ON DELETE CASCADE
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rescan_proposals_job_id ON rescan_proposals(job_id);"
    )

    # Training-Data Export (opt-in crowdsourcing of reviewed labels
    # to the upstream training dev). Rows get added either by an
    # explicit Export-modal selection or — when the auto-opt-in
    # setting is on — automatically each time the operator approves
    # a review event. The CSV / ZIP build path flips ``export_status``
    # from 'pending' to 'exported' after a successful download.
    # UNIQUE(detection_id) enforces at-most-one pool entry per
    # detection; INSERT OR IGNORE in the approval path relies on it.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_exports (
            export_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            detection_id INTEGER NOT NULL,
            export_status TEXT NOT NULL DEFAULT 'pending',
            exported_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(detection_id),
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_exports_batch_id ON training_exports(batch_id);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_exports_status ON training_exports(export_status);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rescan_proposals_status ON rescan_proposals(status);"
    )

    # Seen-species log for the "new species only" Telegram mode. One row per
    # species the operator has ever been notified about (by species_key, the
    # same Latin / OD-class identifier the rest of the stack uses). The mode
    # gates instant alerts on absence from this table; on first sighting we
    # INSERT, then send, then only future sightings of that species ever
    # alert again. Operators can wipe via the Settings "Reset known-species
    # list" button so re-tests after a model swap work.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_species (
            species_key TEXT PRIMARY KEY,
            first_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            first_image_filename TEXT,
            first_score REAL
        );
    """)

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


_BACKFILL_MARKER_KEY = "gallery_eligible_split_v1"


def _backfill_gallery_eligible(conn: sqlite3.Connection) -> None:
    """One-shot backfill: convert legacy auto-tagged favorites to is_gallery_eligible.

    Runs ONCE per database, gated by a marker row in `_migration_state`.

    Before the three-column split, the aesthetic tagger marked picks as
    is_favorite=1 with rating_source='auto'. After the split, those rows
    must move to is_gallery_eligible=1 / is_favorite=0 so that:

      - is_favorite is reserved for HUMAN clicks (gold-label for training)
      - is_gallery_eligible is the model-decision column

    DANGER WITHOUT THE MARKER: the heart-toggle endpoint historically did
    not stamp rating_source='manual' on HUMAN clicks, so the default
    rating_source='auto' would match the migration WHERE clause. Running
    the backfill on every app start would silently delete every HUMAN
    favorite created since the last restart. The marker prevents that.

    Detection of "no rating_source column" (very old DBs predating
    rating_source) short-circuits to a no-op — there is nothing to migrate.
    """
    cur = conn.execute("PRAGMA table_info(detections);")
    cols = {row[1] for row in cur.fetchall()}
    if "rating_source" not in cols or "is_gallery_eligible" not in cols:
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _migration_state (
            key TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        """
    )
    already = conn.execute(
        "SELECT 1 FROM _migration_state WHERE key = ?",
        (_BACKFILL_MARKER_KEY,),
    ).fetchone()
    if already:
        return

    conn.execute(
        """
        UPDATE detections
           SET is_gallery_eligible = 1,
               is_favorite = 0
         WHERE is_favorite = 1
           AND rating_source = 'auto'
        """
    )
    conn.execute(
        "INSERT INTO _migration_state(key, applied_at) VALUES (?, datetime('now'))",
        (_BACKFILL_MARKER_KEY,),
    )
    conn.commit()


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
