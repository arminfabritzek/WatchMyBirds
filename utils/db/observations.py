"""Current station observations, independent of photo selection and event review."""

from __future__ import annotations

import sqlite3

from config import get_config


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


def positive_sql(
    conn: sqlite3.Connection, *, detection: str = "d", image: str = "i"
) -> str:
    if has_facts(conn):
        return f"""{detection}.detection_id IN (
            {observations_cte(conn)}
            SELECT detection_id FROM observations
            WHERE species_key != 'Unknown_species' AND detection_id IS NOT NULL
        )"""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(detections)")}
    source = f"{detection}.species_source" if "species_source" in columns else "NULL"
    answer = fact_sql(conn, "species_identity", detection=detection, image=image)
    presence = fact_sql(conn, "bird_presence", detection=detection, image=image)
    image_presence = fact_sql(
        conn, "bird_presence", detection=detection, image=image, scope="image"
    )
    return f"""({detection}.status = 'active'
        AND COALESCE({image}.review_status, '') != 'no_bird'
        AND COALESCE({presence}, '') != 'absent'
        AND COALESCE({image_presence}, '') != 'absent'
        AND COALESCE({answer}, '') NOT IN ('wrong', 'unknown')
        AND ({answer} IS NOT NULL OR COALESCE({source}, '') NOT IN ('manual_wrong', 'manual_unknown'))
        AND ({answer} IN ('confirmed', 'corrected') OR
            (LOWER(COALESCE({detection}.decision_state, '')) = 'confirmed'
             AND LOWER(COALESCE({detection}.decision_level, 'species')) NOT IN ('reject', 'species_review')))
    )"""


def observations_cte(conn: sqlite3.Connection) -> str:
    """Project current answers and conservative automatic admission without writes."""
    config = get_config()
    bbox_floor = float(config["BBOX_QUALITY_THRESHOLD"])
    unknown_ceiling = float(config["UNKNOWN_SCORE_THRESHOLD"])
    species_floor = float(config["SPECIES_CONF_THRESHOLD"])
    score_floor = float(config["GALLERY_DISPLAY_THRESHOLD"])
    return f"""
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
            d.status, i.review_status, d.score, d.od_confidence, d.bbox_quality,
            d.unknown_score, c.cls_confidence, d.is_favorite, d.quality_gallery_ok,
            d.thumbnail_path, d.is_gallery_eligible, d.decision_state, d.decision_level, d.species_source,
            CASE WHEN json_valid(d.geography_json) THEN
                (SELECT json_extract(value, '$.occurrence_score') FROM json_each(d.geography_json, '$.ranking')
                 WHERE REPLACE(json_extract(value, '$.species'), ' ', '_') = REPLACE(c.cls_class_name, ' ', '_') LIMIT 1)
            END AS occurrence_score,
            COALESCE(f.answer, CASE WHEN d.species_source = 'manual_unknown' THEN 'unknown'
                WHEN d.species_source = 'manual_wrong' THEN 'wrong' END) AS answer,
            f.answered_at, f.presence, fi.presence AS image_presence,
            REPLACE(COALESCE(f.human_species, NULLIF(d.manual_species_override, ''),
                c.cls_class_name, d.od_class_name), ' ', '_') AS proposed_species,
            CASE WHEN f.answer IN ('confirmed', 'corrected') THEN 'human'
                 WHEN NULLIF(d.manual_species_override, '') IS NOT NULL THEN 'legacy'
                 ELSE 'automatic' END AS origin
        FROM detections d JOIN images i ON i.filename = d.image_filename
        LEFT JOIN sources s ON s.source_id = i.source_id
        LEFT JOIN classifications c ON c.detection_id = d.detection_id
            AND c.rank = 1 AND COALESCE(c.status, 'active') = 'active'
        LEFT JOIN facts f ON f.detection_id = d.detection_id AND f.scope = 'object'
        LEFT JOIN facts fi ON fi.image_filename = i.filename AND fi.scope = 'image'
        UNION ALL
        SELECT 'manual:' || m.manual_object_id, NULL, m.manual_object_id, m.image_filename,
            i.timestamp, i.source_id, COALESCE(s.type, 'unknown'),
            i.classifier_model_id, i.ptz_origin, COALESCE(i.original_present, 1),
            m.status, i.review_status, NULL, NULL, NULL, NULL, NULL, 0, NULL, NULL, 0, NULL, NULL, 'manual', NULL,
            CASE WHEN m.species_state = 'unknown' THEN 'unknown' ELSE 'corrected' END,
            m.updated_at, 'present', fi.presence, REPLACE(m.species_key, ' ', '_'), 'human'
        FROM manual_objects m JOIN images i ON i.filename = m.image_filename
        LEFT JOIN sources s ON s.source_id = i.source_id
        LEFT JOIN facts fi ON fi.image_filename = i.filename AND fi.scope = 'image'
    ), evidence AS (
        SELECT *, CASE WHEN origin != 'automatic' THEN NULL
            WHEN COALESCE(presence, '') != 'present' AND
                (od_confidence <= 0 OR bbox_quality < {bbox_floor})
                THEN 'Bird uncertain: detector or box evidence is insufficient'
            WHEN COALESCE(quality_gallery_ok, 1) = 0
                THEN 'Bird uncertain: image sharpness is too low'
            WHEN unknown_score >= {unknown_ceiling}
                THEN 'Species uncertain: model uncertainty is too high'
            WHEN COALESCE(score, 0) < {score_floor} OR cls_confidence < {species_floor}
                THEN 'Species uncertain: confidence is below the admission floor'
            WHEN occurrence_score = 0
                THEN 'Species uncertain: unusual for this location'
            WHEN INSTR(COALESCE(proposed_species, ''), '_') = 0
                OR SUBSTR(LOWER(RTRIM(proposed_species, '.')), -3) = '_sp'
                OR LOWER(COALESCE(proposed_species, '')) IN ('unknown_species', 'unknown', 'bird', 'unclassified', 'marten_mustelid', 'squirrel', 'cat', 'hedgehog')
                THEN 'Species uncertain: no named bird species was identified'
            WHEN COALESCE(decision_state, '') != 'confirmed'
                OR COALESCE(decision_level, 'species') != 'species'
                THEN 'Species uncertain: automatic decision is unresolved'
            ELSE NULL END AS review_reason
        FROM proposals
    ), assessed AS (
        SELECT *,
            (status = 'active' AND COALESCE(answer, '') IN ('confirmed', 'corrected') AND
                (image_presence = 'absent' OR presence = 'absent')) AS conflict,
            (status = 'active' AND COALESCE(review_status, '') != 'no_bird'
                AND COALESCE(presence, '') != 'absent' AND COALESCE(image_presence, '') != 'absent') AS active,
            (answer IS NOT NULL OR origin = 'legacy' OR COALESCE(presence, '') = 'absent' OR COALESCE(image_presence, '') = 'absent') AS answered,
            COALESCE((INSTR(proposed_species, '_') > 0 AND
                SUBSTR(LOWER(RTRIM(proposed_species, '.')), -3) != '_sp' AND
                LOWER(COALESCE(proposed_species, '')) NOT IN ('unknown_species', 'unknown', 'bird', 'unclassified', 'marten_mustelid', 'squirrel', 'cat', 'hedgehog')
                AND COALESCE(answer, '') NOT IN ('wrong', 'unknown') AND
                (origin IN ('human', 'legacy') OR
                    (decision_state = 'confirmed' AND COALESCE(decision_level, 'species') = 'species'
                     AND review_reason IS NULL))
            ), 0) AS admitted
        FROM evidence
    ), observations AS (
        SELECT *, COALESCE(review_reason, CASE
            WHEN origin = 'automatic' AND proposed_species IN (SELECT species FROM refuted_species)
                THEN 'History hint: an earlier proposal of this species was refuted'
            END) AS warning,
            CASE WHEN admitted AND active AND NOT COALESCE(conflict, 0)
            THEN proposed_species ELSE 'Unknown_species' END AS species_key,
            (active AND presence = 'present' AND NOT admitted AND NOT COALESCE(conflict, 0)) AS unknown_bird
        FROM assessed
    )
    """


def selection_sql(selection: dict[str, str] | None = None) -> tuple[str, list[str]]:
    selection = selection or {}
    clauses = ["1"]
    params: list[str] = []
    day = "CASE WHEN substr(timestamp, 5, 1) = '-' THEN substr(timestamp, 1, 10) ELSE substr(timestamp, 1, 4) || '-' || substr(timestamp, 5, 2) || '-' || substr(timestamp, 7, 2) END"
    for key, op in (("start", ">="), ("end", "<=")):
        if selection.get(key):
            clauses.append(f"{day} {op} ?")
            params.append(selection[key])
    if selection.get("source") == "camera":
        clauses.append("source_type IN ('ipcam', 'webcam', 'rtsp', 'camera')")
    elif selection.get("source") == "import":
        clauses.append("source_type IN ('folder_upload', 'import', 'folder')")
    return " AND ".join(clauses), params


def species_summaries(
    conn: sqlite3.Connection, selection: dict[str, str] | None = None
) -> list[dict]:
    where, params = selection_sql(selection)
    rows = [
        dict(row)
        for row in conn.execute(
            observations_cte(conn)
            + f"""
        , selected AS (SELECT *, ({where}) AS in_period FROM observations)
        SELECT species_key, COUNT(CASE WHEN in_period THEN 1 END) AS observation_count,
            COUNT(CASE WHEN in_period AND origin = 'human' THEN 1 END) AS human_count,
            COUNT(CASE WHEN in_period AND origin = 'automatic' THEN 1 END) AS automatic_count,
            COUNT(CASE WHEN in_period AND origin = 'legacy' THEN 1 END) AS legacy_count,
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
