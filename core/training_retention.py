"""Protect human training evidence, including previously protected legacy signals."""

import sqlite3


def protection_sources_any(
    conn: sqlite3.Connection, filenames: list[str]
) -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = {}
    batch_size = max(1, min(400, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)))
    for start in range(0, len(filenames), batch_size):
        batch = filenames[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            SELECT i.filename,
                EXISTS (SELECT 1 FROM label_subjects s JOIN human_label_facts f
                        ON f.subject_id=s.subject_id WHERE s.image_filename=i.filename) AS canonical_facts,
                EXISTS (SELECT 1 FROM manual_objects mo WHERE mo.image_filename=i.filename) AS manual_objects,
                EXISTS (
                    SELECT 1 FROM detections d WHERE d.image_filename=i.filename AND d.status='active'
                    AND (
                        i.review_status='no_bird'
                        OR (lower(COALESCE(d.decision_level,'')) IN ('species','species_review')
                            AND COALESCE(i.review_status,'') != 'no_bird'
                            AND (lower(COALESCE(d.species_source,''))='manual'
                                 OR COALESCE(d.manual_species_override,'') != ''))
                        OR (COALESCE(d.manual_species_override,'') != ''
                            AND d.manual_species_override != COALESCE(d.raw_species_name,''))
                        OR (d.is_favorite=1 AND d.rating_source='manual')
                    )
                ) AS legacy_review
            FROM images i WHERE i.filename IN ({placeholders})
        """,
            batch,
        ).fetchall()
        for row in rows:
            reasons = {
                name
                for name in ("canonical_facts", "manual_objects", "legacy_review")
                if row[name]
            }
            if reasons:
                relevant[row["filename"]] = reasons
    return relevant


def is_export_relevant_any(conn: sqlite3.Connection, filenames: list[str]) -> set[str]:
    return set(protection_sources_any(conn, filenames))
