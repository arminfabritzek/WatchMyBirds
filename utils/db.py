import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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


def insert_image(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO images (
            filename,
            timestamp,
            coco_json,
            downloaded_timestamp,
            detector_model_id,
            classifier_model_id,
            source_id,
            content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("filename"),
            row.get("timestamp"),
            row.get("coco_json"),
            row.get("downloaded_timestamp", ""),
            row.get("detector_model_id", ""),
            row.get("classifier_model_id", ""),
            row.get("source_id"),
            row.get("content_hash"),
        ),
    )
    conn.commit()


def check_image_exists_by_hash(conn: sqlite3.Connection, content_hash: str) -> bool:
    """Checks if an image with the given SHA-256 hash already exists."""
    if not content_hash:
        return False
    row = conn.execute(
        "SELECT 1 FROM images WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row is not None


def insert_detection(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Inserts a detection record and returns its ID."""
    cur = conn.execute(
        """
        INSERT INTO detections (
            image_filename,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            od_class_name,
            od_confidence,
            od_model_id,
            created_at,
            score,
            agreement_score,
            detector_model_name,
            detector_model_version,
            classifier_model_name,
            classifier_model_version,
            thumbnail_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("image_filename"),
            row.get("bbox_x"),
            row.get("bbox_y"),
            row.get("bbox_w"),
            row.get("bbox_h"),
            row.get("od_class_name"),
            row.get("od_confidence"),
            row.get("od_model_id"),
            row.get("created_at"),
            row.get("score"),
            row.get("agreement_score"),
            row.get("detector_model_name"),
            row.get("detector_model_version"),
            row.get("classifier_model_name"),
            row.get("classifier_model_version"),
            row.get("thumbnail_path"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_classification(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    """Inserts a classification record and returns its ID."""
    cur = conn.execute(
        """
        INSERT INTO classifications (
            detection_id,
            cls_class_name,
            cls_confidence,
            cls_model_id,
            rank,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            row.get("detection_id"),
            row.get("cls_class_name"),
            row.get("cls_confidence"),
            row.get("cls_model_id"),
            row.get("rank", 1),
            row.get("created_at"),
        ),
    )
    conn.commit()
    return cur.lastrowid


def fetch_detections_for_gallery(
    conn: sqlite3.Connection,
    date_str_iso: str = None,
    limit: int = None,
    order_by: str = "score",
) -> list[sqlite3.Row]:
    """
    Returns detection-centric records for gallery display.
    """
    params = []
    where_clauses = ["d.status = 'active'"]

    if date_str_iso:
        date_prefix = date_str_iso.replace("-", "")
        where_clauses.append(
            "i.timestamp LIKE ? || '%'"
        )  # Using i.timestamp for filtering
        params.append(date_prefix)

    where_sql = " AND ".join(where_clauses)

    # Sort order
    if order_by == "time":
        order_clause = "ORDER BY i.timestamp DESC"
    else:  # default "score"
        order_clause = "ORDER BY d.score DESC, i.timestamp DESC"

    query = f"""
        SELECT
            d.detection_id,
            i.timestamp as image_timestamp,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.score,
            -- Virtual paths matching actual filesystem structure (YYYY-MM-DD folders)
            -- Prefer explicit thumbnail_path if available (for multi-detection support), else fallback to virtual crop
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) AS relative_path,
            i.filename as original_name,
            i.downloaded_timestamp,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence,
            -- Count of sibling detections on the same image (for multi-bird display)
            (SELECT COUNT(*) FROM detections d2 WHERE d2.image_filename = d.image_filename AND d2.status = 'active') as sibling_count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {where_sql}
        {order_clause}
    """

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    cur = conn.execute(query, params)
    return cur.fetchall()


def fetch_sibling_detections(
    conn: sqlite3.Connection, image_filename: str
) -> list[sqlite3.Row]:
    """
    Returns all active detections for a given image filename.
    Used to display all birds when viewing a multi-detection image in the modal.
    Includes bbox coordinates for bounding box visualization.
    """
    query = """
        SELECT
            d.detection_id,
            d.od_class_name,
            d.od_confidence,
            d.score,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.image_filename = ? AND d.status = 'active'
        ORDER BY d.score DESC
    """
    cur = conn.execute(query, (image_filename,))
    return cur.fetchall()


def fetch_day_count(conn: sqlite3.Connection, date_str_iso: str) -> int:
    """Returns COUNT(*) for a given date (YYYY-MM-DD)."""
    date_prefix = date_str_iso.replace("-", "")
    cur = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM detections d
        WHERE d.image_filename LIKE ? || '%'
        AND d.status = 'active';
        """,
        (date_prefix,),
    )
    row = cur.fetchone()
    return int(row["cnt"]) if row else 0


def fetch_hourly_counts(
    conn: sqlite3.Connection, date_str_iso: str
) -> list[sqlite3.Row]:
    """Returns hourly counts for a given date (YYYY-MM-DD)."""
    date_prefix = date_str_iso.replace("-", "")
    cur = conn.execute(
        """
        SELECT
            substr(d.image_filename, 10, 2) AS hour,
            COUNT(*) AS count
        FROM detections d
        WHERE d.image_filename LIKE ? || '%'
        AND d.status = 'active'
        GROUP BY hour
        ORDER BY hour;
        """,
        (date_prefix,),
    )
    return cur.fetchall()


def fetch_daily_covers(
    conn: sqlite3.Connection, min_score: float = 0.0
) -> list[sqlite3.Row]:
    """
    Returns the best detection (highest score) for each day to use as a cover.
    Includes bbox for dynamic cropping and image count per day.

    Args:
        min_score: Minimum score threshold for counting images (to match gallery display filter)
    """
    query = """
    WITH DailyBest AS (
        SELECT
            (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2)) as date_iso,
            d.image_filename,
            i.filename as original_name,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name,
            d.score,
            d.thumbnail_path,
            ROW_NUMBER() OVER (
                PARTITION BY (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2))
                ORDER BY d.score DESC
            ) as rn
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
    ),
    DayCounts AS (
        SELECT
            (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2)) as date_iso,
            COUNT(DISTINCT d.image_filename) as image_count
        FROM detections d
        WHERE d.status = 'active'
        AND (d.score IS NULL OR d.score >= ?)
        GROUP BY (substr(d.image_filename, 1, 4) || '-' || substr(d.image_filename, 5, 2) || '-' || substr(d.image_filename, 7, 2))
    )
    SELECT
        db.date_iso as date_key,
        REPLACE(db.original_name, '.jpg', '.webp') as optimized_name_virtual,
        (substr(db.image_filename, 1, 4) || '-' || substr(db.image_filename, 5, 2) || '-' || substr(db.image_filename, 7, 2) || '/' ||
         REPLACE(db.original_name, '.jpg', '.webp')) AS relative_path,
        db.bbox_x, db.bbox_y, db.bbox_w, db.bbox_h,
        (substr(db.image_filename, 1, 4) || '-' || substr(db.image_filename, 5, 2) || '-' || substr(db.image_filename, 7, 2) || '/' ||
         COALESCE(db.thumbnail_path, REPLACE(db.original_name, '.jpg', '_crop_1.webp'))) AS thumbnail_path_virtual,
        dc.image_count
    FROM DailyBest db
    JOIN DayCounts dc ON db.date_iso = dc.date_iso
    WHERE db.rn = 1
    ORDER BY date_key DESC;
    """
    cur = conn.execute(query, (min_score,))
    return cur.fetchall()


def reject_detections(conn: sqlite3.Connection, detection_ids: Iterable[int]) -> None:
    """
    Semantic Reject: Sets status of specific detections to 'rejected'.
    Does not delete files.
    Propagates to classifications.
    """
    ids = list(detection_ids)
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)

    # Update status of detections
    conn.execute(
        f"UPDATE detections SET status = 'rejected' WHERE detection_id IN ({placeholders})",
        ids,
    )
    # Also reject classifications for these rejected detections
    conn.execute(
        f"UPDATE classifications SET status = 'rejected' WHERE detection_id IN ({placeholders})",
        ids,
    )
    conn.commit()


def restore_detections(conn: sqlite3.Connection, detection_ids: Iterable[int]) -> None:
    """
    Restores rejected detections to active status. also restores associated classifications.
    Triggers legacy recalculation for affected images.
    """
    ids = list(detection_ids)
    if not ids:
        return

    placeholders = ",".join("?" for _ in ids)

    # 1. Restore Detections
    conn.execute(
        f"UPDATE detections SET status = 'active' WHERE detection_id IN ({placeholders})",
        ids,
    )

    # 2. Restore Classifications (cascading restore)
    conn.execute(
        f"UPDATE classifications SET status = 'active' WHERE detection_id IN ({placeholders})",
        ids,
    )

    conn.commit()


def purge_detections(
    conn: sqlite3.Connection,
    detection_ids: Iterable[int] = None,
    before_date: str = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Permanently deletes rejected detections (and cascades to classifications).
    Strictly DB-only. No file deletion.
    Requires strict scoping (ids or filter).
    """
    if not detection_ids and not before_date:
        raise ValueError("Purge requires explicit detection_ids or filter")

    where_clauses = ["status = 'rejected'"]
    params = []

    if detection_ids:
        ids = list(detection_ids)
        placeholders = ",".join("?" for _ in ids)
        where_clauses.append(f"detection_id IN ({placeholders})")
        params.extend(ids)

    if before_date:
        # Expects ISO date string 'YYYY-MM-DD'
        # image_timestamp usually starts with YYYYMMDD_
        date_prefix = before_date.replace("-", "")
        where_clauses.append("image_filename < ?")
        params.append(date_prefix)

    where_sql = " AND ".join(where_clauses)

    # Dry Run: Count what would be deleted
    count_cursor = conn.execute(
        f"SELECT COUNT(*), GROUP_CONCAT(detection_id) FROM detections WHERE {where_sql}",
        params,
    )
    row = count_cursor.fetchone()
    count = row[0]
    affected_ids_str = row[1] if row[1] else ""
    affected_ids = (
        [int(x) for x in affected_ids_str.split(",")] if affected_ids_str else []
    )

    if dry_run:
        return {"purged": False, "would_purge": count, "detection_ids": affected_ids}

    # Execute Purge
    conn.execute(f"DELETE FROM detections WHERE {where_sql}", params)
    conn.commit()

    return {"purged": True, "count": count, "detection_ids": affected_ids}


def update_downloaded_timestamp(
    conn: sqlite3.Connection, filenames: Iterable[str], download_ts: str
) -> None:
    names = list(filenames)
    if not names:
        return
    placeholders = ",".join("?" for _ in names)
    params = [download_ts] + names
    conn.execute(
        f"""
        UPDATE images
        SET downloaded_timestamp = ?
        WHERE filename IN ({placeholders});
        """,
        params,
    )
    conn.commit()


def fetch_detection_species_summary(
    conn: sqlite3.Connection, date_str_iso: str
) -> list[sqlite3.Row]:
    """
    Returns counts per species for a given date (YYYY-MM-DD), based on DETECTIONS.
    Species = classification class name if present.
    If no classification class name, we consider it 'Unclassified' (or handle OD class if desired,
    but per plan we rely on CLS).
    """
    date_prefix = date_str_iso.replace("-", "")

    # We want to group by the effective species name.
    # Current plan: Use cls_class_name.
    # Note: We need to join detections -> classifications.
    # If a detection has no classification, it will be skipped by inner join, or become 'Unclassified' via Left Join logic.
    # Let's count explicitly identified species via Classification first.

    # The requirement is: "Species = cls_class_name (if present), otherwise 'Unclassified' "

    cur = conn.execute(
        """
        SELECT
            COALESCE(cls.cls_class_name, 'Unclassified') as species,
            COUNT(d.detection_id) as count
        FROM detections d
        LEFT JOIN classifications cls ON d.detection_id = cls.detection_id AND cls.status = 'active'
            AND cls.rank = 1 -- Assuming rank 1 is top choice
        WHERE d.image_filename LIKE ? || '%'
        AND d.status = 'active'
        GROUP BY species
        ORDER BY count DESC;
        """,
        (date_prefix,),
    )
    return cur.fetchall()


def fetch_trash_items(
    conn: sqlite3.Connection,
    page: int = 1,
    limit: int = 50,
    species: str = None,
    before_date: str = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetches trashed items with pagination and filters.
    Returns heterogeneous items:
    - Rejected detections (trash_type='detection')
    - No-bird images (trash_type='image')

    Returns (items, total_count).
    """
    offset = (page - 1) * limit
    items = []

    # === Part 1: Rejected Detections ===
    det_where = ["d.status = 'rejected'"]
    det_params = []

    if species:
        det_where.append("""
            (d.od_class_name = ? OR EXISTS (
                SELECT 1 FROM classifications c
                WHERE c.detection_id = d.detection_id AND c.cls_class_name = ?
            ))
        """)
        det_params.extend([species, species])

    if before_date:
        date_prefix = before_date.replace("-", "")
        det_where.append("d.image_filename < ?")
        det_params.append(date_prefix)

    det_where_sql = " AND ".join(det_where)

    # Count detections
    det_count_row = conn.execute(
        f"SELECT COUNT(*) FROM detections d WHERE {det_where_sql}", det_params
    ).fetchone()
    det_count = det_count_row[0] if det_count_row else 0

    # === Part 2: No-Bird Images ===
    img_where = ["i.review_status = 'no_bird'"]
    img_params = []

    if before_date:
        date_prefix = before_date.replace("-", "")
        img_where.append("i.timestamp < ?")
        img_params.append(date_prefix)

    # Species filter doesn't apply to no-bird images (they have no species)
    img_where_sql = " AND ".join(img_where)

    img_count_row = conn.execute(
        f"SELECT COUNT(*) FROM images i WHERE {img_where_sql}", img_params
    ).fetchone()
    img_count = img_count_row[0] if img_count_row else 0

    total_count = det_count + img_count

    # === Fetch Items with UNION (sorted by timestamp DESC, paginated) ===
    # We use a UNION ALL to combine both types

    union_query = f"""
        SELECT
            'detection' as trash_type,
            CAST(d.detection_id AS TEXT) as item_id,
            i.timestamp as image_timestamp,
            i.filename as filename,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.od_class_name,
            d.od_confidence,
            d.created_at,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) as relative_path,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             COALESCE(d.thumbnail_path, REPLACE(i.filename, '.jpg', '_crop_1.webp'))) as thumbnail_path_virtual,
            (SELECT cls_class_name FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_class_name,
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {det_where_sql}

        UNION ALL

        SELECT
            'image' as trash_type,
            i.filename as item_id,
            i.timestamp as image_timestamp,
            i.filename as filename,
            NULL as bbox_x, NULL as bbox_y, NULL as bbox_w, NULL as bbox_h,
            NULL as od_class_name,
            NULL as od_confidence,
            i.review_updated_at as created_at,
            REPLACE(i.filename, '.jpg', '.webp') as optimized_name_virtual,
            (substr(i.timestamp, 1, 4) || '-' || substr(i.timestamp, 5, 2) || '-' || substr(i.timestamp, 7, 2) || '/' ||
             REPLACE(i.filename, '.jpg', '.webp')) as relative_path,
            NULL as thumbnail_path_virtual,
            NULL as cls_class_name,
            NULL as cls_confidence
        FROM images i
        WHERE {img_where_sql}

        ORDER BY image_timestamp DESC
        LIMIT ? OFFSET ?
    """

    all_params = det_params + img_params + [limit, offset]
    rows = conn.execute(union_query, all_params).fetchall()

    for row in rows:
        items.append(
            {
                "trash_type": row["trash_type"],
                "item_id": row["item_id"],  # detection_id (as str) or filename
                "detection_id": (
                    int(row["item_id"]) if row["trash_type"] == "detection" else None
                ),
                "filename": row["filename"],
                "image_timestamp": row["image_timestamp"],
                "image_optimized": row["optimized_name_virtual"],
                "relative_path": row["relative_path"],
                "thumbnail_path_virtual": row["thumbnail_path_virtual"],
                "bbox_x": row["bbox_x"],
                "bbox_y": row["bbox_y"],
                "bbox_w": row["bbox_w"],
                "bbox_h": row["bbox_h"],
                "od_class_name": row["od_class_name"],
                "od_confidence": row["od_confidence"],
                "cls_class_name": row["cls_class_name"],
                "cls_confidence": row["cls_confidence"],
                "created_at": row["created_at"],
            }
        )

    return items, total_count


def fetch_trash_count(conn: sqlite3.Connection) -> int:
    """
    Returns total number of trashed items (for badge).
    Includes: rejected detections + images with review_status='no_bird'.
    """
    # Count rejected detections
    det_row = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status = 'rejected'"
    ).fetchone()
    det_count = det_row[0] if det_row else 0

    # Count no_bird images
    img_row = conn.execute(
        "SELECT COUNT(*) FROM images WHERE review_status = 'no_bird'"
    ).fetchone()
    img_count = img_row[0] if img_row else 0

    return det_count + img_count


# =============================================================================
# Analytics Functions (All-Time Aggregations)
# =============================================================================


def fetch_all_time_daily_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Returns daily detection counts for all-time data.
    Output: list of rows with 'date_iso' (YYYY-MM-DD) and 'count'.
    """
    cur = conn.execute("""
        SELECT
            (substr(image_filename, 1, 4) || '-' ||
             substr(image_filename, 5, 2) || '-' ||
             substr(image_filename, 7, 2)) AS date_iso,
            COUNT(*) AS count
        FROM detections
        WHERE status = 'active'
        GROUP BY date_iso
        ORDER BY date_iso ASC
        """)
    return cur.fetchall()


def fetch_all_detection_times(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Returns time part (HHMMSS) of all active detections for KDE calculation.
    """
    cur = conn.execute("""
        SELECT substr(i.timestamp, 10, 6) as time_str
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
        """)
    return cur.fetchall()


def fetch_species_timestamps(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Returns (species, timestamp) for all active detections.
    Used for Ridgeplot/Heatmap activity analysis.
    """
    cur = conn.execute("""
        SELECT
            COALESCE(c.cls_class_name, d.od_class_name, 'Unknown') AS species,
            i.timestamp as image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c ON d.detection_id = c.detection_id AND c.status = 'active'
        WHERE d.status = 'active'
        """)
    return cur.fetchall()


def fetch_analytics_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Returns high-level summary stats for analytics dashboard.
    """
    # Total detections
    total_cursor = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status = 'active'"
    )
    total_detections = total_cursor.fetchone()[0] or 0

    # Total unique species
    species_cursor = conn.execute("""
        SELECT COUNT(DISTINCT COALESCE(c.cls_class_name, d.od_class_name)) AS total
        FROM detections d
        LEFT JOIN classifications c ON d.detection_id = c.detection_id AND c.status = 'active'
        WHERE d.status = 'active'
        """)
    total_species = species_cursor.fetchone()[0] or 0

    # Date range
    range_cursor = conn.execute("""
        SELECT
            MIN(substr(i.timestamp, 1, 4) || '-' ||
                substr(i.timestamp, 5, 2) || '-' ||
                substr(i.timestamp, 7, 2)) AS first_date,
            MAX(substr(i.timestamp, 1, 4) || '-' ||
                substr(i.timestamp, 5, 2) || '-' ||
                substr(i.timestamp, 7, 2)) AS last_date
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
        """)
    range_row = range_cursor.fetchone()

    return {
        "total_detections": total_detections,
        "total_species": total_species,
        "date_range": {
            "first": range_row["first_date"] if range_row else None,
            "last": range_row["last_date"] if range_row else None,
        },
    }


def fetch_orphan_images(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Returns images that have no detections at all (true orphan images).
    Images with rejected detections (in trash) are NOT orphans.
    These are candidates for cleanup.
    """
    query = """
    SELECT
        i.filename,
        i.timestamp
    FROM images i
    WHERE NOT EXISTS (
        SELECT 1 FROM detections d
        WHERE d.image_filename = i.filename
    )
    AND i.filename IS NOT NULL
    ORDER BY i.timestamp DESC;
    """
    cur = conn.execute(query)
    return cur.fetchall()


def delete_orphan_images(conn: sqlite3.Connection, filenames: Iterable[str]) -> int:
    """
    Deletes image rows from the database by filename.
    Returns the number of rows deleted.
    File deletion must be handled separately by file_gc.
    """
    names = list(filenames)
    if not names:
        return 0
    placeholders = ",".join("?" for _ in names)
    cur = conn.execute(f"DELETE FROM images WHERE filename IN ({placeholders})", names)
    conn.commit()
    return cur.rowcount


def fetch_orphan_count(conn: sqlite3.Connection) -> int:
    """Returns count of images with zero detections."""
    query = """
    SELECT COUNT(*)
    FROM images i
    WHERE NOT EXISTS (
        SELECT 1 FROM detections d
        WHERE d.image_filename = i.filename
    )
    """
    row = conn.execute(query).fetchone()
    return row[0] if row else 0


# =============================================================================
# Review Queue Functions
# =============================================================================


def fetch_review_queue_images(
    conn: sqlite3.Connection, save_threshold: float = 0.65
) -> list[sqlite3.Row]:
    """
    Returns images that need review:
    - review_status = 'untagged' AND
    - (no detections OR max(od_confidence) < save_threshold)

    Sorted by timestamp ASC (oldest first).
    """
    query = """
    SELECT
        i.filename,
        i.timestamp,
        i.review_status,
        (SELECT MAX(d.od_confidence) FROM detections d WHERE d.image_filename = i.filename) as max_od_conf,
        CASE
            WHEN NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)
            THEN 'orphan'
            ELSE 'low_confidence'
        END as review_reason
    FROM images i
    WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
    AND (
        NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)
        OR (SELECT MAX(d.od_confidence) FROM detections d WHERE d.image_filename = i.filename) < ?
    )
    AND i.filename IS NOT NULL
    ORDER BY i.timestamp ASC;
    """
    cur = conn.execute(query, (save_threshold,))
    return cur.fetchall()


def fetch_review_queue_count(
    conn: sqlite3.Connection, save_threshold: float = 0.65
) -> int:
    """
    Returns count of images needing review (for badge).
    Same criteria as fetch_review_queue_images.
    """
    query = """
    SELECT COUNT(*)
    FROM images i
    WHERE (i.review_status IS NULL OR i.review_status = 'untagged')
    AND (
        NOT EXISTS (SELECT 1 FROM detections d WHERE d.image_filename = i.filename)
        OR (SELECT MAX(d.od_confidence) FROM detections d WHERE d.image_filename = i.filename) < ?
    )
    AND i.filename IS NOT NULL;
    """
    row = conn.execute(query, (save_threshold,)).fetchone()
    return row[0] if row else 0


def restore_no_bird_images(conn: sqlite3.Connection, filenames: Iterable[str]) -> int:
    """
    Restores 'no_bird' images back to 'untagged' (returns them to Review Queue).
    Returns: number of rows updated.
    """
    names = list(filenames)
    if not names:
        return 0

    from datetime import datetime

    updated_at = datetime.now().isoformat()

    placeholders = ",".join("?" for _ in names)
    params = [updated_at] + names

    cur = conn.execute(
        f"""
        UPDATE images
        SET review_status = 'untagged', review_updated_at = ?
        WHERE filename IN ({placeholders})
        AND review_status = 'no_bird';
        """,
        params,
    )
    conn.commit()
    return cur.rowcount


def delete_no_bird_images(
    conn: sqlite3.Connection, filenames: Iterable[str] = None, delete_all: bool = False
) -> int:
    """
    Permanently deletes 'no_bird' images from the database.
    File deletion must be handled separately.

    Args:
        filenames: Specific files to delete (if None and delete_all=True, deletes all)
        delete_all: If True and filenames is None, deletes ALL no_bird images

    Returns: number of rows deleted.
    """
    if not filenames and not delete_all:
        return 0

    if filenames:
        names = list(filenames)
        placeholders = ",".join("?" for _ in names)
        cur = conn.execute(
            f"DELETE FROM images WHERE filename IN ({placeholders}) AND review_status = 'no_bird'",
            names,
        )
    else:
        # Delete all no_bird images
        cur = conn.execute("DELETE FROM images WHERE review_status = 'no_bird'")

    conn.commit()
    return cur.rowcount


def update_review_status(
    conn: sqlite3.Connection,
    filenames: Iterable[str],
    new_status: str,
    updated_at: str = None,
) -> int:
    """
    Updates review_status for specified images.
    Only updates images that are currently 'untagged' (no way back).

    new_status: 'confirmed_bird' | 'no_bird'
    Returns: number of rows updated.
    """
    names = list(filenames)
    if not names:
        return 0

    if new_status not in ("confirmed_bird", "no_bird"):
        raise ValueError(f"Invalid review status: {new_status}")

    if updated_at is None:
        from datetime import datetime

        updated_at = datetime.now().isoformat()

    placeholders = ",".join("?" for _ in names)
    params = [new_status, updated_at] + names

    cur = conn.execute(
        f"""
        UPDATE images
        SET review_status = ?, review_updated_at = ?
        WHERE filename IN ({placeholders})
        AND (review_status IS NULL OR review_status = 'untagged');
        """,
        params,
    )
    conn.commit()
    return cur.rowcount
