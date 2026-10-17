"""Current station observations, independent of photo selection and event review."""

from __future__ import annotations

import sqlite3


def has_facts(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'current_human_label_facts'"
        ).fetchone()
        is not None
    )


def fact_sql(
    conn: sqlite3.Connection,
    axis: str,
    column: str = "answer_value",
    *,
    detection: str = "d",
    image: str = "i",
    scope: str = "object",
) -> str:
    if not has_facts(conn):
        return "NULL"
    subject = (
        f"f.detection_id = {detection}.detection_id"
        if scope == "object"
        else f"f.image_filename = {image}.filename"
    )
    return (
        f"(SELECT f.{column} FROM current_human_label_facts f WHERE {subject} "
        f"AND f.scope = '{scope}' AND f.fact_type = '{axis}' LIMIT 1)"
    )


def identity_sql(
    conn: sqlite3.Connection, fallback: str, *, detection: str = "d"
) -> str:
    answer = fact_sql(conn, "species_identity", detection=detection)
    species = fact_sql(conn, "species_identity", "species_key", detection=detection)
    return f"""CASE WHEN {answer} IN ('unknown', 'wrong') THEN 'Unknown_species'
        WHEN {answer} IN ('confirmed', 'corrected') THEN REPLACE({species}, ' ', '_')
        ELSE {fallback} END"""


def recreate_observations_view(conn: sqlite3.Connection) -> None:
    """Rebuild the current object projection at application startup."""
    conn.execute("DROP VIEW IF EXISTS current_observations")
    conn.execute("""
    CREATE VIEW current_observations AS
    WITH facts AS MATERIALIZED (
        SELECT detection_id, image_filename, scope,
            MAX(CASE WHEN fact_type = 'species_identity' THEN answer_value END) AS answer,
            MAX(CASE WHEN fact_type = 'species_identity' THEN species_key END) AS human_species,
            MAX(CASE WHEN fact_type = 'species_identity' THEN proposal_species_key END) AS proposal_species,
            MAX(CASE WHEN fact_type = 'species_identity' THEN created_at END) AS answered_at,
            MAX(CASE WHEN fact_type = 'bird_presence' THEN answer_value END) AS presence
        FROM current_human_label_facts GROUP BY scope, detection_id, image_filename
    ), refuted_species AS MATERIALIZED (
        SELECT DISTINCT REPLACE(proposal_species, ' ', '_') AS species
        FROM facts WHERE answer = 'wrong' AND proposal_species IS NOT NULL
    ), proposals AS (
        SELECT 'detection:' || d.detection_id AS object_key, d.detection_id,
            NULL AS manual_object_id, d.image_filename, i.timestamp,
            i.source_id, COALESCE(s.type, 'unknown') AS source_type,
            i.classifier_model_id, i.ptz_origin, COALESCE(i.original_present, 1) AS original_present,
            d.status, d.score, d.od_confidence, d.bbox_quality,
            d.unknown_score, c.cls_confidence, d.is_favorite, d.quality_gallery_ok,
            d.thumbnail_path, d.is_gallery_eligible, d.aesthetic_score, d.decision_state, d.decision_level, d.decision_reasons,
            f.answer,
            f.answered_at, f.presence, (f.scope IS NOT NULL) AS has_current_answer,
            REPLACE(CASE WHEN f.answer IN ('confirmed', 'corrected') THEN f.human_species
                ELSE COALESCE(c.cls_class_name, d.od_class_name) END, ' ', '_') AS proposed_species,
            CASE WHEN f.answer IN ('confirmed', 'corrected') THEN 'human'
                 ELSE 'automatic' END AS origin
        FROM detections d JOIN images i ON i.filename = d.image_filename
        LEFT JOIN sources s ON s.source_id = i.source_id
        LEFT JOIN classifications c ON c.detection_id = d.detection_id
            AND c.rank = 1 AND COALESCE(c.status, 'active') = 'active'
        LEFT JOIN facts f ON f.detection_id = d.detection_id AND f.scope = 'object'
        UNION ALL
        SELECT 'manual:' || m.manual_object_id, NULL, m.manual_object_id, m.image_filename,
            i.timestamp, i.source_id, COALESCE(s.type, 'unknown'),
            i.classifier_model_id, i.ptz_origin, COALESCE(i.original_present, 1),
            m.status, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL,
            CASE WHEN m.species_state = 'unknown' THEN 'unknown' ELSE 'corrected' END,
            m.updated_at, 'present', 1, REPLACE(m.species_key, ' ', '_'), 'human'
        FROM manual_objects m JOIN images i ON i.filename = m.image_filename
        LEFT JOIN sources s ON s.source_id = i.source_id
    ), evidence AS (
        SELECT *, CASE WHEN origin != 'automatic' THEN NULL
            WHEN COALESCE(decision_state, '') != 'confirmed'
                OR COALESCE(NULLIF(decision_level, ''), 'species') != 'species'
                THEN 'Species uncertain: automatic decision is unresolved' ||
                    CASE WHEN COALESCE(decision_reasons, '[]') != '[]'
                        THEN ': ' || decision_reasons ELSE '' END
            ELSE NULL END AS review_reason
        FROM proposals
    ), assessed AS (
        SELECT *,
            (status = 'active' AND COALESCE(presence, '') != 'absent') AS active,
            (answer IS NOT NULL OR COALESCE(presence, '') = 'absent') AS answered,
            COALESCE((status = 'active' AND COALESCE(presence, '') != 'absent'
                AND NULLIF(TRIM(proposed_species), '') IS NOT NULL
                AND proposed_species != 'Unknown_species'
                AND CASE WHEN answer IS NOT NULL
                    THEN answer IN ('confirmed', 'corrected')
                    ELSE decision_state = 'confirmed'
                        AND COALESCE(NULLIF(decision_level, ''), 'species') = 'species'
                END), 0) AS admitted
        FROM evidence
    ), observations AS (
        SELECT *, COALESCE(review_reason, CASE
            WHEN origin = 'automatic' AND proposed_species IN (SELECT species FROM refuted_species)
                THEN 'History hint: an earlier proposal of this species was refuted'
            END) AS warning,
            CASE WHEN admitted AND active
            THEN proposed_species ELSE 'Unknown_species' END AS species_key,
            (active AND presence = 'present' AND NOT admitted) AS unknown_bird
        FROM assessed
    )
    SELECT * FROM observations
    """)


def selection_sql(selection: dict[str, str] | None = None) -> tuple[str, list[str]]:
    selection = selection or {}
    clauses = ["1"]
    params: list[str] = []
    day = "CASE WHEN substr(timestamp, 5, 1) = '-' THEN substr(timestamp, 1, 10) ELSE substr(timestamp, 1, 4) || '-' || substr(timestamp, 5, 2) || '-' || substr(timestamp, 7, 2) END"
    for key, op in (("start", ">="), ("end", "<=")):
        if selection.get(key):
            clauses.append(f"{day} {op} ?")
            params.append(selection[key])
    return " AND ".join(clauses), params


def species_summaries(
    conn: sqlite3.Connection, selection: dict[str, str] | None = None
) -> list[dict]:
    where, params = selection_sql(selection)
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
        WITH selected AS (SELECT *, ({where}) AS in_period FROM current_observations)
        SELECT species_key, COUNT(CASE WHEN in_period THEN 1 END) AS observation_count,
            COUNT(CASE WHEN in_period AND origin = 'human' THEN 1 END) AS human_count,
            COUNT(CASE WHEN in_period AND origin = 'automatic' THEN 1 END) AS automatic_count,
            0 AS legacy_count,
            MIN(CASE WHEN in_period THEN timestamp END) AS first_seen,
            MAX(CASE WHEN in_period THEN timestamp END) AS last_seen,
            MAX(CASE WHEN origin = 'human' THEN timestamp END) AS archive_last_seen,
            COUNT(DISTINCT CASE WHEN in_period THEN substr(REPLACE(timestamp, '-', ''), 1, 8) END) AS days,
            COUNT(DISTINCT CASE WHEN in_period AND origin = 'human' THEN substr(REPLACE(timestamp, '-', ''), 1, 8) END) AS human_days,
            COUNT(DISTINCT CASE WHEN in_period AND origin = 'automatic' THEN substr(REPLACE(timestamp, '-', ''), 1, 8) END) AS automatic_days,
            COUNT(DISTINCT CASE WHEN in_period AND original_present = 0 THEN image_filename END) AS missing_originals
        FROM selected WHERE species_key != 'Unknown_species'
        GROUP BY species_key HAVING observation_count > 0 OR archive_last_seen IS NOT NULL
        ORDER BY species_key
    """,
            params,
        )
    ]
    for row in rows:
        for key in ("first_seen", "last_seen", "archive_last_seen"):
            value = str(row.get(key) or "").replace("-", "")
            row[key + "_date"] = (
                f"{value[:4]}-{value[4:6]}-{value[6:8]}" if len(value) >= 8 else ""
            )
    return rows
