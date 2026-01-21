import os
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List

from config import get_config


DB_FILENAME = "images.db"


def _get_db_path() -> Path:
    cfg = get_config()
    output_dir = Path(cfg.get("OUTPUT_DIR", "/output"))
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
    conn.execute(
        """
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
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_timestamp ON images(timestamp DESC);"
    )

    conn.execute(
        """
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
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_detections_filename ON detections(image_filename);"
    )

    conn.execute(
        """
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
        """
    )
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
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            uri TEXT,
            config_json TEXT,
            active INTEGER DEFAULT 1
        );
        """
    )
    _ensure_column_on_table(conn, "images", "source_id", "INTEGER REFERENCES sources(source_id)")

    _ensure_column_on_table(conn, "images", "content_hash", "TEXT")

    # 1. Ensure Default Source Exists
    default_source_id = get_or_create_default_source(conn)
    # 2. Backfill existing images
    conn.execute(
        "UPDATE images SET source_id = ? WHERE source_id IS NULL",
        (default_source_id,)
    )
    conn.commit()


def get_or_create_default_source(conn: sqlite3.Connection) -> int:
    """Gets the ID of the default 'Default Camera' source, or creates it if missing."""
    row = conn.execute("SELECT source_id FROM sources WHERE name='Default Camera'").fetchone()
    if row:
        return row[0]
    
    # Create if not exists
    cur = conn.execute(
        "INSERT INTO sources (name, type) VALUES (?, ?)", 
        ("Default Camera", "ipcam")
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_user_import_source(conn: sqlite3.Connection) -> int:
    """Gets the ID of the 'User Import' source, or creates it if missing."""
    row = conn.execute("SELECT source_id FROM sources WHERE name='User Import'").fetchone()
    if row:
        return row[0]
    
    # Create if not exists
    cur = conn.execute(
        "INSERT INTO sources (name, type) VALUES (?, ?)", 
        ("User Import", "folder_upload")
    )
    conn.commit()
    return cur.lastrowid


def _ensure_column(conn: sqlite3.Connection, column: str, coltype: str) -> None:
    _ensure_column_on_table(conn, "images", column, coltype)

def _ensure_column_on_table(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    cur = conn.execute(f"PRAGMA table_info({table});")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype};")
        conn.commit()
    # Add index for content_hash if added via migration (or if missing and column exists)
    if column == "content_hash" and table == "images":
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash);")
        conn.commit()





def insert_image(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
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


def insert_detection(conn: sqlite3.Connection, row: Dict[str, Any]) -> int:
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


def insert_classification(conn: sqlite3.Connection, row: Dict[str, Any]) -> int:
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
    order_by: str = "score"
) -> List[sqlite3.Row]:
    """
    Returns detection-centric records for gallery display.
    """
    params = []
    where_clauses = ["d.status = 'active'"]
    
    if date_str_iso:
        date_prefix = date_str_iso.replace("-", "")
        where_clauses.append("i.timestamp LIKE ? || '%'") # Using i.timestamp for filtering
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
            (SELECT cls_confidence FROM classifications c WHERE c.detection_id = d.detection_id ORDER BY cls_confidence DESC LIMIT 1) as cls_confidence
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


def fetch_hourly_counts(conn: sqlite3.Connection, date_str_iso: str) -> List[sqlite3.Row]:
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


def fetch_daily_covers(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Returns the best detection (highest score) for each day to use as a cover.
    Includes bbox for dynamic cropping and detection count per day.
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
            COUNT(*) as detection_count
        FROM detections d
        WHERE d.status = 'active'
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
        dc.detection_count
    FROM DailyBest db
    JOIN DayCounts dc ON db.date_iso = dc.date_iso
    WHERE db.rn = 1
    ORDER BY date_key DESC;
    """
    cur = conn.execute(query)
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
        ids
    )

    # 2. Restore Classifications (cascading restore)
    conn.execute(
        f"UPDATE classifications SET status = 'active' WHERE detection_id IN ({placeholders})",
        ids
    )


    conn.commit()


def purge_detections(
    conn: sqlite3.Connection, 
    detection_ids: Iterable[int] = None, 
    before_date: str = None, 
    dry_run: bool = False
) -> Dict[str, Any]:
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
        params
    )
    row = count_cursor.fetchone()
    count = row[0]
    affected_ids_str = row[1] if row[1] else ""
    affected_ids = [int(x) for x in affected_ids_str.split(",")] if affected_ids_str else []

    if dry_run:
        return {
            "purged": False,
            "would_purge": count,
            "detection_ids": affected_ids
        }

    # Execute Purge
    conn.execute(f"DELETE FROM detections WHERE {where_sql}", params)
    conn.commit()

    return {
        "purged": True,
        "count": count,
        "detection_ids": affected_ids
    }


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
) -> List[sqlite3.Row]:
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
    
    # The requirement is: "Species = cls_class_name (wenn vorhanden), sonst 'Unclassified' "
    
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
    before_date: str = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetches rejected detections with pagination and filters.
    Returns (items, total_count).
    """
    offset = (page - 1) * limit
    
    where_clauses = ["d.status = 'rejected'"]
    params = []
    
    if species:
        where_clauses.append("""
            (d.od_class_name = ? OR EXISTS (
                SELECT 1 FROM classifications c 
                WHERE c.detection_id = d.detection_id AND c.cls_class_name = ?
            ))
        """)
        params.extend([species, species])
        
    if before_date:
        date_prefix = before_date.replace("-", "")
        where_clauses.append("d.image_filename < ?")
        params.append(date_prefix)
        
    where_sql = " AND ".join(where_clauses)
    
    # Total Count
    count_cursor = conn.execute(f"SELECT COUNT(*) FROM detections d WHERE {where_sql}", params)
    total_count = count_cursor.fetchone()[0]
    
    # Items
    params.extend([limit, offset])
    items_cursor = conn.execute(f"""
        SELECT 
            d.detection_id,
            i.timestamp as image_timestamp,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
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
        WHERE {where_sql}
        ORDER BY i.timestamp DESC
        LIMIT ? OFFSET ?
    """, params)
    
    items = []
    for row in items_cursor:
        items.append({
            "detection_id": row["detection_id"],
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
            "created_at": row["created_at"]
        })
        
    return items, total_count


def fetch_trash_count(conn: sqlite3.Connection) -> int:
    """Returns total number of rejected detections (for badge)."""
    row = conn.execute("SELECT COUNT(*) FROM detections WHERE status = 'rejected'").fetchone()
    return row[0] if row else 0


# =============================================================================
# Analytics Functions (All-Time Aggregations)
# =============================================================================

def fetch_all_time_daily_counts(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Returns daily detection counts for all-time data.
    Output: list of rows with 'date_iso' (YYYY-MM-DD) and 'count'.
    """
    cur = conn.execute(
        """
        SELECT 
            (substr(image_filename, 1, 4) || '-' || 
             substr(image_filename, 5, 2) || '-' || 
             substr(image_filename, 7, 2)) AS date_iso,
            COUNT(*) AS count
        FROM detections
        WHERE status = 'active'
        GROUP BY date_iso
        ORDER BY date_iso ASC
        """
    )
    return cur.fetchall()


def fetch_all_detection_times(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Returns time part (HHMMSS) of all active detections for KDE calculation.
    """
    cur = conn.execute(
        """
        SELECT substr(i.timestamp, 10, 6) as time_str
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
        """
    )
    return cur.fetchall()


def fetch_species_timestamps(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Returns (species, timestamp) for all active detections.
    Used for Ridgeplot/Heatmap activity analysis.
    """
    cur = conn.execute(
        """
        SELECT 
            COALESCE(c.cls_class_name, d.od_class_name, 'Unknown') AS species,
            i.timestamp as image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        LEFT JOIN classifications c ON d.detection_id = c.detection_id AND c.status = 'active'
        WHERE d.status = 'active'
        """
    )
    return cur.fetchall()






def fetch_analytics_summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Returns high-level summary stats for analytics dashboard.
    """
    # Total detections
    total_cursor = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status = 'active'"
    )
    total_detections = total_cursor.fetchone()[0] or 0
    
    # Total unique species
    species_cursor = conn.execute(
        """
        SELECT COUNT(DISTINCT COALESCE(c.cls_class_name, d.od_class_name)) AS total
        FROM detections d
        LEFT JOIN classifications c ON d.detection_id = c.detection_id AND c.status = 'active'
        WHERE d.status = 'active'
        """
    )
    total_species = species_cursor.fetchone()[0] or 0
    
    # Date range
    range_cursor = conn.execute(
        """
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
        """
    )
    range_row = range_cursor.fetchone()
    
    return {
        "total_detections": total_detections,
        "total_species": total_species,
        "date_range": {
            "first": range_row["first_date"] if range_row else None,
            "last": range_row["last_date"] if range_row else None
        }
    }


def fetch_orphan_images(conn: sqlite3.Connection) -> List[sqlite3.Row]:
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
    cur = conn.execute(
        f"DELETE FROM images WHERE filename IN ({placeholders})",
        names
    )
    conn.commit()
    return cur.rowcount
