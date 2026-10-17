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
    existed_before_setup = db_path.exists()
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
        _init_schema(conn, existing_installation=existed_before_setup)
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


def _init_schema(
    conn: sqlite3.Connection, *, existing_installation: bool = False
) -> None:
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

    # CLS-v2 decision layer (added with the classifier config YAMLs).
    # decision_level: 'species' | 'genus' | 'reject'. NULL for detections
    #   saved before the decision layer shipped — those are species-level
    #   by construction (the old pipeline had no genus fallback).
    # raw_species_name: top-1 species latin regardless of decision level.
    #   Lets us reconstruct what the classifier actually thought, even when
    #   cls_class_name was promoted to genus_sp. or cleared for reject.
    _ensure_column_on_table(conn, "detections", "decision_level", "TEXT")
    _ensure_column_on_table(conn, "detections", "raw_species_name", "TEXT")
    _ensure_column_on_table(conn, "detections", "geography_json", "TEXT")

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
    #     - manual manual favorites (is_favorite=1) remain a clean training gold-label
    #     - the tagger never overwrites manual choices
    #     - gallery surfaces can render a "AI pick" badge on is_gallery_eligible=1 AND
    #       is_favorite=0 detections
    #   Backfill of legacy rating_source='auto' rows happens in _backfill_gallery_eligible
    #   (see below). md.
    _ensure_column_on_table(
        conn, "detections", "is_gallery_eligible", "INTEGER DEFAULT 0"
    )
    _backfill_gallery_eligible(conn)

    # Nightly crop quality scoring (see web/services/nightly_job_hub.py
    # + utils/image_ops.laplacian_sharpness/crop_brightness). Computed
    # once per crop by the nightly sharpness job. Purely additive aux
    # signal — never used to filter, gate, or drop detections.
    # sharpness_score: float, variance of Laplacian on a 256-px-normalised
    #   grayscale crop. Higher = sharper. NULL = unscored.
    # crop_brightness: float in [0,255], mean grayscale luminance.
    #   Companion metric so consumers can tell "blur" from "underexposed".
    _ensure_column_on_table(conn, "detections", "sharpness_score", "REAL")
    _ensure_column_on_table(conn, "detections", "crop_brightness", "REAL")
    # The nightly sharpness percentile ranks photos; it never changes admission.
    _ensure_column_on_table(conn, "detections", "quality_gallery_ok", "INTEGER")
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

    # Artifact retention: original-file presence marker (additive).
    # original_present=1 for every existing row (the original is on disk);
    # retention sets it to 0 and stamps original_deleted_at on removal.
    _ensure_column_on_table(conn, "images", "original_present", "INTEGER DEFAULT 1")
    _ensure_column_on_table(conn, "images", "original_deleted_at", "TEXT")

    # Deep Scan tracking (additive, no destructive migration)
    _ensure_column_on_table(conn, "images", "deep_scan_last_attempt_at", "TEXT")
    _ensure_column_on_table(conn, "images", "deep_scan_last_result", "TEXT")
    _ensure_column_on_table(
        conn, "images", "deep_scan_attempt_count", "INTEGER DEFAULT 0"
    )

    # PTZ context per frame (additive, NULL on legacy and non-PTZ rows).
    # Forward-compatible schema: the three coordinate slots (pan/tilt/zoom)
    # and ptz_position_at stay NULL in v1 and are reserved for v2, when
    # ONVIF GetStatus / AbsoluteMove data is wired in. Used by the gallery
    # ranker to prefer preset-targeted thumbnails over overview shots
    # (the camera is physically closer to the bird at preset rest).
    _ensure_column_on_table(conn, "images", "ptz_origin", "TEXT")
    _ensure_column_on_table(conn, "images", "ptz_preset_token", "TEXT")
    _ensure_column_on_table(conn, "images", "ptz_zone", "TEXT")
    _ensure_column_on_table(conn, "images", "ptz_state", "TEXT")
    _ensure_column_on_table(conn, "images", "ptz_camera_id", "INTEGER")
    _ensure_column_on_table(conn, "images", "ptz_pan", "REAL")
    _ensure_column_on_table(conn, "images", "ptz_tilt", "REAL")
    _ensure_column_on_table(conn, "images", "ptz_zoom", "REAL")
    _ensure_column_on_table(conn, "images", "ptz_position_at", "TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_images_ptz_origin
        ON images(ptz_origin) WHERE ptz_origin IS NOT NULL;
        """
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
    _ensure_column_on_table(conn, "rescan_proposals", "geography_json", "TEXT")

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

    # Reject-audit log: metadata-only record of detections the classifier
    # routed to ``decision_level='reject'`` (top-1 below review threshold,
    # no genus rescue). The detection itself is not persisted — no image,
    # no crop, no detections row, no images row when the frame had only
    # rejects. This keeps the FP-cluster signal
    # (bbox position + raw_species_name + counts) auditable without
    # filling disk and the Unclear-tab with non-bird crops from static
    # background objects (e.g. tree-branch FPs).
    #
    # No FK to images/detections: by design, the parent rows don't exist
    # for reject-only frames. The frame is identified by its filename
    # stamp (``frame_timestamp`` = ``YYYYMMDD_HHMMSS_ffffff``) so the
    # operator can cross-reference logs even though no file was written.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reject_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            frame_timestamp TEXT NOT NULL,
            frame_width INTEGER,
            frame_height INTEGER,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            od_class_name TEXT,
            od_confidence REAL,
            raw_species_name TEXT,
            top1_prob REAL,
            decision_state TEXT,
            decision_reasons TEXT,
            detector_model_id TEXT,
            classifier_model_id TEXT
        );
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reject_audit_created_at ON reject_audit(created_at DESC);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reject_audit_raw_species ON reject_audit(raw_species_name);"
    )
    # Composite bbox index supports the static-FP cluster detection
    # query ("which bbox positions cluster across reject rows?").
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_reject_audit_bbox ON reject_audit(bbox_x, bbox_y);"
    )

    _ensure_column_on_table(conn, "detections", "trashed_at", "TEXT")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS _migration_state (key TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    _init_human_label_schema(conn)
    _init_training_export_schema(conn)
    _init_station_event_review_schema(conn)
    _init_station_runtime_schema(conn)

    from utils.db.observations import recreate_observations_view

    recreate_observations_view(conn)

    conn.commit()
    try:
        migrate_human_answers(conn, existing_installation=existing_installation)
        _initialize_automatic_deletion(
            conn, existing_installation=existing_installation
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Human-answer migration failed; deletion remains disabled"
        )


def _initialize_automatic_deletion(
    conn: sqlite3.Connection, *, existing_installation: bool
) -> None:
    from datetime import UTC, datetime

    from utils.settings import load_settings_yaml, save_settings_yaml

    conn.execute("BEGIN IMMEDIATE")
    try:
        marker = "automatic_deletion_defaults_v1"
        if conn.execute(
            "SELECT 1 FROM _migration_state WHERE key=?", (marker,)
        ).fetchone():
            conn.rollback()
            return
        cfg = get_config()
        stamp = datetime.now(UTC).isoformat()
        settings = load_settings_yaml(str(cfg["OUTPUT_DIR"]))
        defaults = {
            "AUTOMATIC_DELETION_ENABLED": not existing_installation,
            "AUTOMATIC_DELETION_ENABLED_AT": None if existing_installation else stamp,
            "OLD_BIRD_PHOTOS": "keep_previews",
        }
        save_settings_yaml({**settings, **defaults}, str(cfg["OUTPUT_DIR"]))
        cfg.update(defaults)
        conn.execute("INSERT INTO _migration_state VALUES (?,?)", (marker, stamp))
        if existing_installation:
            conn.execute(
                "INSERT OR IGNORE INTO _migration_state VALUES ('human_answers_retention_notice',?)",
                (stamp,),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _init_training_export_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS labeling_exports (
            export_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            received_at TEXT,
            bundle_json TEXT NOT NULL,
            media_map_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS labeling_export_members (
            export_id TEXT NOT NULL REFERENCES labeling_exports(export_id),
            image_filename TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('subject', 'manual')),
            object_id INTEGER NOT NULL,
            revision_json TEXT NOT NULL,
            PRIMARY KEY(export_id, kind, object_id)
        );
        CREATE INDEX IF NOT EXISTS idx_labeling_export_member_revision
        ON labeling_export_members(kind, object_id, revision_json);
        CREATE INDEX IF NOT EXISTS idx_labeling_export_member_image
        ON labeling_export_members(image_filename);
    """)


def _init_human_label_schema(conn: sqlite3.Connection) -> None:
    """Install the additive canonical human-label schema."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_objects (
            manual_object_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'retracted')),
            bbox_x REAL NOT NULL CHECK(bbox_x BETWEEN 0.0 AND 1.0),
            bbox_y REAL NOT NULL CHECK(bbox_y BETWEEN 0.0 AND 1.0),
            bbox_w REAL NOT NULL CHECK(bbox_w > 0.0 AND bbox_w <= 1.0),
            bbox_h REAL NOT NULL CHECK(bbox_h > 0.0 AND bbox_h <= 1.0),
            species_key TEXT,
            species_state TEXT NOT NULL
                CHECK(species_state IN ('identified', 'unknown')),
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(bbox_x + bbox_w <= 1.0),
            CHECK(bbox_y + bbox_h <= 1.0),
            CHECK(
                (species_state = 'identified' AND length(trim(species_key)) > 0)
                OR (species_state = 'unknown' AND species_key IS NULL)
            ),
            FOREIGN KEY(image_filename) REFERENCES images(filename)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_manual_objects_image_status
        ON manual_objects(image_filename, status);
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_object_revisions (
            manual_revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            manual_object_id INTEGER NOT NULL,
            revision INTEGER NOT NULL CHECK(revision >= 1),
            operation TEXT NOT NULL CHECK(operation IN ('create', 'update')),
            asserted_facts TEXT NOT NULL,
            bbox_x REAL NOT NULL,
            bbox_y REAL NOT NULL,
            bbox_w REAL NOT NULL,
            bbox_h REAL NOT NULL,
            species_key TEXT,
            species_state TEXT NOT NULL
                CHECK(species_state IN ('identified', 'unknown')),
            context TEXT NOT NULL
                CHECK(context IN ('normal_correction', 'targeted_training')),
            source_kind TEXT NOT NULL CHECK(length(trim(source_kind)) > 0),
            source_ref TEXT,
            installation_id TEXT NOT NULL CHECK(length(trim(installation_id)) > 0),
            app_version TEXT NOT NULL CHECK(length(trim(app_version)) > 0),
            created_at TEXT NOT NULL CHECK(length(trim(created_at)) > 0),
            UNIQUE(manual_object_id, revision),
            FOREIGN KEY(manual_object_id) REFERENCES manual_objects(manual_object_id)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bird_add_requests (
            request_id TEXT PRIMARY KEY,
            image_filename TEXT NOT NULL REFERENCES images(filename) ON DELETE CASCADE,
            payload_json TEXT NOT NULL,
            result_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_object_requests (
            request_id TEXT PRIMARY KEY,
            manual_object_id INTEGER NOT NULL,
            revision INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(manual_object_id) REFERENCES manual_objects(manual_object_id)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS label_subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL CHECK(scope IN ('image', 'object')),
            image_filename TEXT NOT NULL,
            detection_id INTEGER,
            proposal_bbox_x REAL,
            proposal_bbox_y REAL,
            proposal_bbox_w REAL,
            proposal_bbox_h REAL,
            proposal_coordinate_space TEXT,
            proposal_species_key TEXT,
            detector_model_version TEXT,
            classifier_model_version TEXT,
            created_at TEXT NOT NULL,
            CHECK(
                (
                    scope = 'image'
                    AND detection_id IS NULL
                    AND proposal_bbox_x IS NULL
                    AND proposal_bbox_y IS NULL
                    AND proposal_bbox_w IS NULL
                    AND proposal_bbox_h IS NULL
                    AND proposal_coordinate_space IS NULL
                )
                OR
                (
                    scope = 'object'
                    AND detection_id IS NOT NULL
                    AND proposal_bbox_x BETWEEN 0.0 AND 1.0
                    AND proposal_bbox_y BETWEEN 0.0 AND 1.0
                    AND proposal_bbox_w > 0.0 AND proposal_bbox_w <= 1.0
                    AND proposal_bbox_h > 0.0 AND proposal_bbox_h <= 1.0
                    AND proposal_coordinate_space = 'frame_fraction_xywh_v1'
                )
            ),
            FOREIGN KEY(image_filename) REFERENCES images(filename)
                ON DELETE CASCADE,
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_label_subjects_image_unique
        ON label_subjects(image_filename)
        WHERE scope = 'image';
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_label_subjects_detection_unique
        ON label_subjects(detection_id)
        WHERE scope = 'object';
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_label_subject_object_matches_image
        BEFORE INSERT ON label_subjects
        WHEN NEW.scope = 'object'
         AND NOT EXISTS (
             SELECT 1
             FROM detections d
             WHERE d.detection_id = NEW.detection_id
               AND d.image_filename = NEW.image_filename
         )
        BEGIN
            SELECT RAISE(ABORT, 'object subject does not match detection image');
        END;
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS human_label_facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version INTEGER NOT NULL DEFAULT 1
                CHECK(schema_version = 1),
            subject_id INTEGER NOT NULL,
            fact_type TEXT NOT NULL CHECK(fact_type IN (
                'bird_presence',
                'bbox_quality',
                'bbox_correction',
                'species_identity',
                'detector_miss'
            )),
            assertion_state TEXT NOT NULL
                CHECK(assertion_state IN ('asserted', 'retracted')),
            answer_value TEXT,
            species_key TEXT,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            coordinate_space TEXT,
            context TEXT NOT NULL
                CHECK(context IN ('normal_correction', 'targeted_training')),
            source_kind TEXT NOT NULL CHECK(length(trim(source_kind)) > 0),
            source_ref TEXT,
            installation_id TEXT NOT NULL
                CHECK(length(trim(installation_id)) > 0),
            app_version TEXT NOT NULL CHECK(length(trim(app_version)) > 0),
            created_at TEXT NOT NULL CHECK(length(trim(created_at)) > 0),
            supersedes_fact_id INTEGER,
            CHECK(supersedes_fact_id IS NULL OR supersedes_fact_id != fact_id),
            CHECK(
                (
                    assertion_state = 'retracted'
                    AND answer_value IS NULL
                    AND species_key IS NULL
                    AND bbox_x IS NULL AND bbox_y IS NULL
                    AND bbox_w IS NULL AND bbox_h IS NULL
                    AND coordinate_space IS NULL
                )
                OR
                (
                    assertion_state = 'asserted'
                    AND (
                        (
                            fact_type = 'bird_presence'
                            AND answer_value IN ('present', 'absent')
                            AND species_key IS NULL
                            AND bbox_x IS NULL AND bbox_y IS NULL
                            AND bbox_w IS NULL AND bbox_h IS NULL
                            AND coordinate_space IS NULL
                        )
                        OR
                        (
                            fact_type = 'bbox_quality'
                            AND answer_value IN ('suitable', 'unsuitable')
                            AND species_key IS NULL
                            AND bbox_x IS NULL AND bbox_y IS NULL
                            AND bbox_w IS NULL AND bbox_h IS NULL
                            AND coordinate_space IS NULL
                        )
                        OR
                        (
                            fact_type = 'bbox_correction'
                            AND answer_value = 'provided'
                            AND species_key IS NULL
                            AND bbox_x BETWEEN 0.0 AND 1.0
                            AND bbox_y BETWEEN 0.0 AND 1.0
                            AND bbox_w > 0.0 AND bbox_w <= 1.0
                            AND bbox_h > 0.0 AND bbox_h <= 1.0
                            AND bbox_x + bbox_w <= 1.0
                            AND bbox_y + bbox_h <= 1.0
                            AND coordinate_space = 'frame_fraction_xywh_v1'
                        )
                        OR
                        (
                            fact_type = 'species_identity'
                            AND (
                                (
                                    answer_value IN ('confirmed', 'corrected')
                                    AND length(trim(species_key)) > 0
                                )
                                OR
                                (
                                    answer_value IN ('wrong', 'unknown')
                                    AND species_key IS NULL
                                )
                            )
                            AND bbox_x IS NULL AND bbox_y IS NULL
                            AND bbox_w IS NULL AND bbox_h IS NULL
                            AND coordinate_space IS NULL
                        )
                        OR
                        (
                            fact_type = 'detector_miss'
                            AND answer_value = 'reported'
                            AND species_key IS NULL
                            AND bbox_x IS NULL AND bbox_y IS NULL
                            AND bbox_w IS NULL AND bbox_h IS NULL
                            AND coordinate_space IS NULL
                        )
                    )
                )
            ),
            FOREIGN KEY(subject_id) REFERENCES label_subjects(subject_id)
                ON DELETE CASCADE,
            FOREIGN KEY(supersedes_fact_id) REFERENCES human_label_facts(fact_id)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_human_label_facts_root_unique
        ON human_label_facts(subject_id, fact_type)
        WHERE supersedes_fact_id IS NULL;
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_human_label_facts_successor_unique
        ON human_label_facts(supersedes_fact_id)
        WHERE supersedes_fact_id IS NOT NULL;
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_human_label_facts_subject_type
        ON human_label_facts(subject_id, fact_type, fact_id);
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_human_label_fact_scope
        BEFORE INSERT ON human_label_facts
        WHEN NOT EXISTS (
            SELECT 1
            FROM label_subjects s
            WHERE s.subject_id = NEW.subject_id
              AND (
                  (s.scope = 'image' AND NEW.fact_type IN (
                      'bird_presence', 'detector_miss'
                  ))
                  OR
                  (s.scope = 'object' AND NEW.fact_type IN (
                      'bird_presence', 'bbox_quality',
                      'bbox_correction', 'species_identity'
                  ))
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'fact type is invalid for subject scope');
        END;
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_human_label_fact_supersession_axis
        BEFORE INSERT ON human_label_facts
        WHEN NEW.supersedes_fact_id IS NOT NULL
         AND NOT EXISTS (
             SELECT 1
             FROM human_label_facts previous
             WHERE previous.fact_id = NEW.supersedes_fact_id
               AND previous.subject_id = NEW.subject_id
               AND previous.fact_type = NEW.fact_type
         )
        BEGIN
            SELECT RAISE(ABORT, 'superseded fact belongs to another axis');
        END;
        """
    )
    conn.execute("DROP VIEW IF EXISTS current_human_label_facts")
    conn.execute(
        """
        CREATE VIEW current_human_label_facts AS
        SELECT
            f.fact_id,
            f.schema_version,
            f.subject_id,
            s.scope,
            s.image_filename,
            s.detection_id,
            f.fact_type,
            f.assertion_state,
            f.answer_value,
            f.species_key,
            f.bbox_x,
            f.bbox_y,
            f.bbox_w,
            f.bbox_h,
            f.coordinate_space,
            f.context,
            f.source_kind,
            f.source_ref,
            f.installation_id,
            f.app_version,
            f.created_at,
            f.supersedes_fact_id,
            s.proposal_bbox_x,
            s.proposal_bbox_y,
            s.proposal_bbox_w,
            s.proposal_bbox_h,
            s.proposal_coordinate_space,
            s.proposal_species_key,
            s.detector_model_version,
            s.classifier_model_version
        FROM human_label_facts f
        JOIN label_subjects s ON s.subject_id = f.subject_id
        WHERE f.assertion_state = 'asserted'
          AND NOT EXISTS (
              SELECT 1
              FROM human_label_facts successor
              WHERE successor.supersedes_fact_id = f.fact_id
          );
        """
    )


def _init_station_event_review_schema(conn: sqlite3.Connection) -> None:
    """Retain the unused event-assessment schema for existing databases."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS station_event_reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key_snapshot TEXT NOT NULL,
            anchor_detection_id INTEGER NOT NULL,
            outcome TEXT NOT NULL CHECK(outcome IN (
                'verified_species', 'unresolved_bird', 'no_bird'
            )),
            candidate_species_key TEXT,
            species_key TEXT,
            evidence_quality TEXT NOT NULL CHECK(evidence_quality IN (
                'diagnostic', 'limited', 'insufficient'
            )),
            event_start TEXT NOT NULL,
            event_end TEXT NOT NULL,
            source_kind TEXT NOT NULL CHECK(length(trim(source_kind)) > 0),
            source_ref TEXT,
            installation_id TEXT NOT NULL CHECK(length(trim(installation_id)) > 0),
            app_version TEXT NOT NULL CHECK(length(trim(app_version)) > 0),
            created_at TEXT NOT NULL CHECK(length(trim(created_at)) > 0),
            CHECK(
                (outcome = 'verified_species' AND length(trim(species_key)) > 0)
                OR (outcome != 'verified_species' AND species_key IS NULL)
            ),
            FOREIGN KEY(anchor_detection_id) REFERENCES detections(detection_id)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS station_event_review_members (
            review_id INTEGER NOT NULL,
            detection_id INTEGER NOT NULL,
            PRIMARY KEY(review_id, detection_id),
            FOREIGN KEY(review_id) REFERENCES station_event_reviews(review_id)
                ON DELETE CASCADE,
            FOREIGN KEY(detection_id) REFERENCES detections(detection_id)
                ON DELETE CASCADE
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_station_event_review_members_detection
        ON station_event_review_members(detection_id, review_id DESC);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_station_event_reviews_created
        ON station_event_reviews(created_at DESC);
        """
    )


def _init_station_runtime_schema(conn: sqlite3.Connection) -> None:
    """Install the local observation-effort sample stream."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS station_runtime_samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sampled_at TEXT NOT NULL,
            sample_interval_seconds REAL NOT NULL
                CHECK(sample_interval_seconds > 0),
            app_online INTEGER NOT NULL CHECK(app_online IN (0, 1)),
            camera_online INTEGER NOT NULL CHECK(camera_online IN (0, 1)),
            stream_online INTEGER NOT NULL CHECK(stream_online IN (0, 1)),
            detector_ready INTEGER NOT NULL CHECK(detector_ready IN (0, 1)),
            detector_active INTEGER NOT NULL CHECK(detector_active IN (0, 1)),
            od_active INTEGER NOT NULL CHECK(od_active IN (0, 1)),
            od_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    _ensure_column_on_table(conn, "station_runtime_samples", "ptz_state", "TEXT")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_station_runtime_samples_time
        ON station_runtime_samples(sampled_at ASC);
        """
    )


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


def _ensure_column(conn: sqlite3.Connection, column: str, coltype: str) -> None:
    _ensure_column_on_table(conn, "images", column, coltype)


_BACKFILL_MARKER_KEY = "gallery_eligible_split_v1"


def _backfill_gallery_eligible(conn: sqlite3.Connection) -> None:
    """One-shot backfill: convert legacy auto-tagged favorites to is_gallery_eligible.

    Runs ONCE per database, gated by a marker row in `_migration_state`.

    Before the three-column split, the aesthetic tagger marked picks as
    is_favorite=1 with rating_source='auto'. After the split, those rows
    must move to is_gallery_eligible=1 / is_favorite=0 so that:

      - is_favorite is reserved for manual clicks (gold-label for training)
      - is_gallery_eligible is the model-decision column

    DANGER WITHOUT THE MARKER: the heart-toggle endpoint historically did
    not stamp rating_source='manual' on manual clicks, so the default
    rating_source='auto' would match the migration WHERE clause. Running
    the backfill on every app start would silently delete every manual
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


HUMAN_ANSWERS_MIGRATION = "human_answer_facts_v1"


def human_answers_migrated(conn: sqlite3.Connection) -> bool:
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='_migration_state'"
    ).fetchone():
        return False
    return (
        conn.execute(
            "SELECT 1 FROM _migration_state WHERE key=?", (HUMAN_ANSWERS_MIGRATION,)
        ).fetchone()
        is not None
    )


def migrate_human_answers(
    conn: sqlite3.Connection, *, existing_installation: bool = False
) -> dict[str, int]:
    """Carry historical answers over atomically; never resurrect a withdrawn axis."""
    import json
    from collections import Counter
    from dataclasses import replace
    from datetime import UTC, datetime
    from uuid import uuid4

    from core.human_label_core import (
        LabelProvenance,
        _derived_source_ref,
        append_fact,
        ensure_image_subject,
        ensure_object_subject,
        species_review_state,
    )
    from detectors.services.decision_policy_service import DecisionPolicyService

    if human_answers_migrated(conn):
        return {}
    previous_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    counts: Counter[str] = Counter()
    stamp = datetime.now(UTC).isoformat()
    provenance = LabelProvenance(
        installation_id="legacy",
        app_version="unknown",
        context="normal_correction",
        source_kind="watchmybirds_migration",
        source_ref="migration:human-answers",
        created_at=stamp,
    )
    policy = DecisionPolicyService()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if human_answers_migrated(conn):
            conn.rollback()
            return {}

        def history(subject: int, axis: str) -> bool:
            return (
                conn.execute(
                    "SELECT 1 FROM human_label_facts WHERE subject_id=? AND fact_type=? LIMIT 1",
                    (subject, axis),
                ).fetchone()
                is not None
            )

        def retract(fact: sqlite3.Row) -> None:
            append_fact(
                conn,
                subject_id=fact["subject_id"],
                fact_type=fact["fact_type"],
                answer_value=None,
                assertion_state="retracted",
                provenance=provenance,
            )
            counts["withdrawn_facts"] += 1

        detections = conn.execute(
            """SELECT d.*,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id=d.detection_id AND c.rank=1) AS model_species
            FROM detections d JOIN images i ON i.filename=d.image_filename ORDER BY d.detection_id"""
        ).fetchall()
        # A withdrawn contradictory absence still explains an old rejection.
        absent_ids = {
            row[0]
            for row in conn.execute(
                "SELECT detection_id FROM current_human_label_facts WHERE scope='object' AND fact_type='bird_presence' AND answer_value='absent'"
            )
        }
        conflict_objects: set[int] = set()
        blocked_species: set[int] = set()
        blocked_images: set[str] = set()
        # Resolve every image contradiction before importing either legacy axis.
        image_absences = conn.execute(
            """SELECT i.filename, f.* FROM images i
            LEFT JOIN current_human_label_facts f ON f.image_filename=i.filename
                AND f.scope='image' AND f.fact_type='bird_presence'
            WHERE f.answer_value='absent' OR (i.review_status='no_bird' AND NOT EXISTS (
                SELECT 1 FROM human_label_facts h JOIN label_subjects s USING(subject_id)
                WHERE s.scope='image' AND s.image_filename=i.filename AND h.fact_type='bird_presence'
            ))"""
        ).fetchall()
        for image_fact in image_absences:
            filename = image_fact["filename"]
            positive_ids = {
                fact[0]
                for fact in conn.execute(
                    """SELECT DISTINCT detection_id FROM current_human_label_facts
                    WHERE image_filename=? AND scope='object' AND (
                        (fact_type='bird_presence' AND answer_value='present') OR
                        (fact_type='species_identity' AND answer_value IN ('confirmed','corrected','unknown'))
                    )""",
                    (filename,),
                )
            }
            legacy_ids = {
                row[0]
                for row in conn.execute(
                    """SELECT d.detection_id FROM detections d WHERE d.image_filename=?
                    AND ((d.species_source='manual' AND COALESCE(d.manual_species_override,'')!='')
                        OR d.species_source='manual_unknown')
                    AND NOT EXISTS (SELECT 1 FROM human_label_facts f JOIN label_subjects s USING(subject_id)
                        WHERE s.detection_id=d.detection_id AND f.fact_type='species_identity')""",
                    (filename,),
                )
            }
            manual_count = conn.execute(
                "SELECT COUNT(*) FROM manual_objects WHERE image_filename=? AND status='active'",
                (filename,),
            ).fetchone()[0]
            if not (positive_ids or legacy_ids or manual_count):
                continue
            blocked_images.add(filename)
            blocked_species.update(legacy_ids)
            if image_fact["fact_id"] is not None:
                retract(image_fact)
                counts["image_object_conflicts"] += len(positive_ids | legacy_ids)
                counts["image_manual_conflicts"] += manual_count
                for did in positive_ids | legacy_ids:
                    conflict_objects.add(did)
                    for fact in conn.execute(
                        """SELECT * FROM current_human_label_facts WHERE detection_id=?
                        AND fact_type IN ('bird_presence','species_identity')""",
                        (did,),
                    ).fetchall():
                        retract(fact)
            else:
                counts["legacy_conflicts"] += len(legacy_ids)
                counts["legacy_object_conflicts"] += len(positive_ids)
                counts["legacy_manual_conflicts"] += manual_count
                conflict_objects.update(legacy_ids)
            conflict_objects.update(
                row[0]
                for row in conn.execute(
                    """SELECT d.detection_id FROM detections d WHERE d.image_filename=?
                    AND NOT EXISTS (SELECT 1 FROM current_human_label_facts f
                        WHERE f.detection_id=d.detection_id AND f.fact_type='bird_presence')""",
                    (filename,),
                )
            )
        for row in detections:
            did = row["detection_id"]
            species_history = (
                conn.execute(
                    """SELECT 1 FROM human_label_facts f JOIN label_subjects s USING(subject_id)
                WHERE s.detection_id=? AND f.fact_type='species_identity' LIMIT 1""",
                    (did,),
                ).fetchone()
                is not None
            )
            source = row["species_source"]
            old_species = row["manual_species_override"]
            legacy_species = bool(old_species) or source in {
                "manual",
                "manual_unknown",
                "manual_wrong",
            }
            needs_subject = legacy_species or row["manual_bbox_review"] in {
                "correct",
                "wrong",
            }
            subject = ensure_object_subject(conn, did) if needs_subject else None
            if subject is not None:
                if did not in blocked_species and not history(
                    subject, "species_identity"
                ):
                    value = {"manual_unknown": "unknown", "manual_wrong": "wrong"}.get(
                        source
                    )
                    if source == "manual" and old_species:
                        proposal = conn.execute(
                            "SELECT proposal_species_key FROM label_subjects WHERE subject_id=?",
                            (subject,),
                        ).fetchone()[0]
                        value = species_review_state("corrected", old_species, proposal)
                    if value:
                        answer_provenance = replace(
                            provenance, source_ref="answer:" + uuid4().hex
                        )
                        if value != "wrong" and not history(subject, "bird_presence"):
                            append_fact(
                                conn,
                                subject_id=subject,
                                fact_type="bird_presence",
                                answer_value="present",
                                provenance=replace(
                                    answer_provenance,
                                    source_ref=_derived_source_ref(
                                        answer_provenance.source_ref
                                    ),
                                ),
                            )
                        append_fact(
                            conn,
                            subject_id=subject,
                            fact_type="species_identity",
                            answer_value=value,
                            species_key=old_species
                            if value in {"confirmed", "corrected"}
                            else None,
                            provenance=answer_provenance,
                        )
                        counts["imported_species"] += 1
                if row["manual_bbox_review"] in {"correct", "wrong"} and not history(
                    subject, "bbox_quality"
                ):
                    append_fact(
                        conn,
                        subject_id=subject,
                        fact_type="bbox_quality",
                        answer_value="suitable"
                        if row["manual_bbox_review"] == "correct"
                        else "unsuitable",
                        provenance=provenance,
                    )
                    counts["imported_box_quality"] += 1
            reset_reason = (
                "conflict"
                if did in conflict_objects
                else "species_history"
                if species_history
                else "legacy_species"
                if legacy_species
                else None
            )
            reasons = []
            if reset_reason:
                reasons = ["replaced by a human answer before the update"]
                counts["reset_" + reset_reason] += 1
            elif row["decision_state"] == "confirmed":
                reasons = policy.admission_reasons(
                    score=row["score"],
                    species_key=row["model_species"] or row["od_class_name"],
                    geography_json=row["geography_json"],
                    allow_missing=True,
                )
                for reason in reasons:
                    counts["policy_" + reason] += 1
            if reasons:
                try:
                    prior = json.loads(row["decision_reasons"] or "[]")
                except (ValueError, TypeError):
                    prior = []
                prior = prior if isinstance(prior, list) else []
                conn.execute(
                    "UPDATE detections SET decision_state='uncertain', decision_reasons=? WHERE detection_id=?",
                    (json.dumps(list(dict.fromkeys([*prior, *reasons]))), did),
                )
                counts["changed_decisions"] += row["decision_state"] != "uncertain"
        for row in conn.execute(
            "SELECT filename FROM images WHERE review_status='no_bird' OR filename IN (SELECT image_filename FROM current_human_label_facts WHERE scope='image' AND fact_type='bird_presence' AND answer_value='absent')"
        ).fetchall():
            filename = row[0]
            if filename in blocked_images:
                continue
            subject = ensure_image_subject(conn, filename)
            current = conn.execute(
                "SELECT * FROM current_human_label_facts WHERE subject_id=? AND fact_type='bird_presence'",
                (subject,),
            ).fetchone()
            if not history(subject, "bird_presence"):
                image_fact_id = append_fact(
                    conn,
                    subject_id=subject,
                    fact_type="bird_presence",
                    answer_value="absent",
                    provenance=provenance,
                )
                counts["imported_image_absence"] += 1
                current = {"answer_value": "absent", "fact_id": image_fact_id}
            if current is not None and current["answer_value"] == "absent":
                for det in conn.execute(
                    "SELECT detection_id FROM detections WHERE image_filename=?",
                    (filename,),
                ).fetchall():
                    obj = ensure_object_subject(conn, det[0])
                    if not history(obj, "bird_presence"):
                        append_fact(
                            conn,
                            subject_id=obj,
                            fact_type="bird_presence",
                            answer_value="absent",
                            provenance=replace(
                                provenance,
                                source_ref=f"migration:image-absence:{current['fact_id']}",
                            ),
                        )
                        counts["imported_object_absence"] += 1
        absent_ids.update(
            r[0]
            for r in conn.execute(
                "SELECT detection_id FROM current_human_label_facts WHERE scope='object' AND fact_type='bird_presence' AND answer_value='absent'"
            )
        )
        for did in absent_ids:
            counts["restored_absence_objects"] += conn.execute(
                "UPDATE detections SET status='active' WHERE detection_id=? AND status='rejected'",
                (did,),
            ).rowcount
            conn.execute(
                "UPDATE classifications SET status='active' WHERE detection_id=? AND status='rejected'",
                (did,),
            )
        counts["dated_trash_objects"] = conn.execute(
            "UPDATE detections SET trashed_at=? WHERE status='rejected' AND trashed_at IS NULL",
            (stamp,),
        ).rowcount
        if existing_installation:
            conn.execute(
                "INSERT OR IGNORE INTO _migration_state(key,applied_at) VALUES ('human_answers_retention_notice',?)",
                (stamp,),
            )
        conn.execute(
            "INSERT INTO _migration_state(key,applied_at) VALUES (?,?)",
            (HUMAN_ANSWERS_MIGRATION, stamp),
        )
        conn.commit()
        return dict(counts)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.row_factory = previous_factory
