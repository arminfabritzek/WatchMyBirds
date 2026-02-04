"""
Analytics Database Operations.

This module handles analytics-related database queries for dashboards
and reporting functionality.
"""

import sqlite3
from typing import Any


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
