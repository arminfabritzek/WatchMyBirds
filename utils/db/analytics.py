"""
Analytics Database Operations.

This module handles analytics-related database queries for dashboards
and reporting functionality.
"""

import sqlite3
from collections import Counter
from typing import Any


def fetch_all_time_daily_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Returns daily detection counts for all-time data.
    Output: list of rows with 'date_iso' (YYYY-MM-DD) and 'count'.
    """
    from utils.db.detections import _gallery_visibility_sql

    cur = conn.execute(f"""
        SELECT
            (substr(d.image_filename, 1, 4) || '-' ||
             substr(d.image_filename, 5, 2) || '-' ||
             substr(d.image_filename, 7, 2)) AS date_iso,
            COUNT(*) AS count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_gallery_visibility_sql("d", "i")}
        GROUP BY date_iso
        ORDER BY date_iso ASC
        """)
    return cur.fetchall()


def fetch_all_detection_times(
    conn: sqlite3.Connection,
    min_score: float = 0.0,
) -> list[sqlite3.Row]:
    """
    Returns time part (HHMMSS) of classified detections for KDE calculation.

    Filters: active, not no_bird, score >= min_score, must have classification.
    """
    cur = conn.execute(
        """
        SELECT substr(i.timestamp, 10, 6) as time_str
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
          AND (i.review_status IS NULL OR i.review_status != 'no_bird')
          AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
          AND (d.score IS NULL OR d.score >= ?)
          AND EXISTS (
              SELECT 1 FROM classifications c
              WHERE c.detection_id = d.detection_id
                AND c.status = 'active'
          )
        """,
        (min_score,),
    )
    return cur.fetchall()


def fetch_species_timestamps(
    conn: sqlite3.Connection,
    min_score: float = 0.0,
) -> list[sqlite3.Row]:
    """
    Returns (species, timestamp) for active, classified detections.
    Used for Ridgeplot/Heatmap activity analysis.

    Filters:
    - d.status = 'active'
    - Not no_bird (trash)
    - Must have an active classification
    - score >= min_score (GALLERY_DISPLAY_THRESHOLD)
    - Species resolved via manual_override > classification > od_class_name
    """
    from utils.db.detections import effective_species_sql

    cur = conn.execute(
        f"""
        SELECT
            {effective_species_sql("d")} AS species,
            i.timestamp as image_timestamp
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE d.status = 'active'
          AND (i.review_status IS NULL OR i.review_status != 'no_bird')
          AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
          AND (d.score IS NULL OR d.score >= ?)
          AND EXISTS (
              SELECT 1 FROM classifications c
              WHERE c.detection_id = d.detection_id
                AND c.status = 'active'
          )
        """,
        (min_score,),
    )
    return cur.fetchall()


def _optional_column_sql(
    columns: set[str], alias: str, column_name: str, default_sql: str = "NULL"
) -> str:
    """Return an aliased column reference when available, otherwise SQL fallback."""

    return f"{alias}.{column_name}" if column_name in columns else default_sql


def _top1_class_sql_for_columns(classification_columns: set[str]) -> str:
    where_clauses = ["c.detection_id = d.detection_id"]
    if "rank" in classification_columns:
        where_clauses.append("c.rank = 1")
    if "status" in classification_columns:
        where_clauses.append("COALESCE(c.status, 'active') = 'active'")
    order_sql = "ORDER BY c.rank ASC" if "rank" in classification_columns else ""
    where_sql = "\n              AND ".join(where_clauses)
    return f"""
        (
            SELECT c.cls_class_name
            FROM classifications c
            WHERE {where_sql}
            {order_sql}
            LIMIT 1
        )
    """


def _top1_confidence_sql_for_columns(classification_columns: set[str]) -> str:
    if "cls_confidence" not in classification_columns:
        return "NULL"
    where_clauses = ["c.detection_id = d.detection_id"]
    if "rank" in classification_columns:
        where_clauses.append("c.rank = 1")
    if "status" in classification_columns:
        where_clauses.append("COALESCE(c.status, 'active') = 'active'")
    order_sql = "ORDER BY c.rank ASC" if "rank" in classification_columns else ""
    where_sql = "\n              AND ".join(where_clauses)
    return f"""
        (
            SELECT c.cls_confidence
            FROM classifications c
            WHERE {where_sql}
            {order_sql}
            LIMIT 1
        )
    """


def _empty_event_intelligence_summary() -> dict[str, Any]:
    return {
        "summary": {
            "event_count": 0,
            "detection_count": 0,
            "representative_image_count": 0,
            "reducible_image_count": 0,
            "retention_savings_pct": 0.0,
            "avg_photos_per_event": 0.0,
            "compression_ratio": 0.0,
            "largest_event_photo_count": 0,
        },
        "largest_events": [],
        "species_pressure": [],
        "profile_distribution": [],
        "retention_formula": "min(Kmax, 3 + ceil(log2(photo_count)) + bonuses)",
    }


def _fetch_event_intelligence_rows(
    conn: sqlite3.Connection,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    from utils.db.detections import effective_species_sql_for_columns, table_columns

    detection_columns = table_columns(conn, "detections")
    image_columns = table_columns(conn, "images")
    classification_columns = table_columns(conn, "classifications")
    species_sql = effective_species_sql_for_columns(
        "d", detection_columns, classification_columns
    )
    cls_sql = _top1_class_sql_for_columns(classification_columns)
    cls_confidence_sql = _top1_confidence_sql_for_columns(classification_columns)
    manual_species_sql = _optional_column_sql(
        detection_columns, "d", "manual_species_override"
    )
    species_source_sql = _optional_column_sql(detection_columns, "d", "species_source")
    od_class_sql = _optional_column_sql(detection_columns, "d", "od_class_name")
    score_sql = _optional_column_sql(detection_columns, "d", "score")
    decision_state_sql = _optional_column_sql(detection_columns, "d", "decision_state")
    source_id_sql = _optional_column_sql(image_columns, "i", "source_id")

    sibling_status_clause = ""
    if "status" in detection_columns:
        sibling_status_clause = "AND COALESCE(ds.status, 'active') = 'active'"

    where_clauses = ["i.timestamp IS NOT NULL", "TRIM(i.timestamp) != ''"]
    params: list[Any] = []
    if "status" in detection_columns:
        where_clauses.append("COALESCE(d.status, 'active') = 'active'")
    if "review_status" in image_columns:
        where_clauses.append(
            "(i.review_status IS NULL OR i.review_status != 'no_bird')"
        )
    if "decision_state" in detection_columns:
        where_clauses.append(
            "lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')"
        )
    if "score" in detection_columns:
        where_clauses.append("(d.score IS NULL OR d.score >= ?)")
        params.append(min_score)

    where_sql = "\n          AND ".join(where_clauses)

    rows = conn.execute(
        f"""
        SELECT
            d.detection_id,
            d.image_filename,
            i.filename,
            i.timestamp,
            i.timestamp AS image_timestamp,
            {source_id_sql} AS source_id,
            d.bbox_x,
            d.bbox_y,
            d.bbox_w,
            d.bbox_h,
            {score_sql} AS score,
            {decision_state_sql} AS decision_state,
            {od_class_sql} AS od_class_name,
            {cls_sql} AS cls_class_name,
            {cls_confidence_sql} AS cls_confidence,
            {species_sql} AS species_key,
            {manual_species_sql} AS manual_species_override,
            {species_source_sql} AS species_source,
            (
                SELECT COUNT(*)
                FROM detections ds
                WHERE ds.image_filename = d.image_filename
                {sibling_status_clause}
            ) AS sibling_detection_count
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {where_sql}
        ORDER BY i.timestamp ASC, d.detection_id ASC
        """,
        tuple(params),
    ).fetchall()

    return [dict(row) for row in rows]


def fetch_event_intelligence_summary(
    conn: sqlite3.Connection,
    min_score: float = 0.0,
    *,
    event_limit: int = 8,
    species_limit: int = 8,
) -> dict[str, Any]:
    """Return event and retention pressure metrics for the Analytics dashboard.

    This is read-only. The retention numbers are estimates based on
    ``BirdEvent.representative_image_count``; no file or database mutation
    happens here.
    """

    from core.events import build_bird_events

    rows = _fetch_event_intelligence_rows(conn, min_score=min_score)
    if not rows:
        return _empty_event_intelligence_summary()

    events = build_bird_events(rows)
    if not events:
        return _empty_event_intelligence_summary()

    detection_count = sum(event.photo_count for event in events)
    representative_count = sum(event.representative_image_count for event in events)
    reducible_count = max(detection_count - representative_count, 0)
    event_count = len(events)
    savings_pct = (
        round((reducible_count / detection_count) * 100.0, 1)
        if detection_count
        else 0.0
    )
    avg_photos = round(detection_count / event_count, 1) if event_count else 0.0
    compression_ratio = (
        round(detection_count / representative_count, 1)
        if representative_count
        else 0.0
    )

    profile_counter: Counter[str] = Counter(event.grouping_profile for event in events)
    profile_detections: Counter[str] = Counter()
    for event in events:
        profile_detections[event.grouping_profile] += event.photo_count

    profile_distribution = [
        {
            "profile": profile,
            "event_count": count,
            "detection_count": profile_detections[profile],
        }
        for profile, count in profile_counter.most_common()
    ]

    species_stats: dict[str, dict[str, Any]] = {}
    for event in events:
        species = event.species or "Unknown_species"
        bucket = species_stats.setdefault(
            species,
            {
                "species": species,
                "event_count": 0,
                "detection_count": 0,
                "representative_image_count": 0,
                "reducible_image_count": 0,
                "largest_event_photo_count": 0,
                "profile": event.grouping_profile,
            },
        )
        bucket["event_count"] += 1
        bucket["detection_count"] += event.photo_count
        bucket["representative_image_count"] += event.representative_image_count
        bucket["reducible_image_count"] += max(
            event.photo_count - event.representative_image_count, 0
        )
        bucket["largest_event_photo_count"] = max(
            bucket["largest_event_photo_count"], event.photo_count
        )

    species_pressure = []
    for bucket in species_stats.values():
        bucket["avg_photos_per_event"] = round(
            bucket["detection_count"] / bucket["event_count"], 1
        )
        bucket["retention_savings_pct"] = (
            round(
                (bucket["reducible_image_count"] / bucket["detection_count"]) * 100.0,
                1,
            )
            if bucket["detection_count"]
            else 0.0
        )
        species_pressure.append(bucket)

    species_pressure.sort(
        key=lambda item: (
            item["reducible_image_count"],
            item["detection_count"],
            item["largest_event_photo_count"],
            item["species"],
        ),
        reverse=True,
    )

    largest_events = []
    for event in sorted(
        events,
        key=lambda item: (
            item.photo_count,
            item.duration_sec,
            item.representative_image_count,
        ),
        reverse=True,
    )[: max(0, int(event_limit))]:
        reducible_images = max(event.photo_count - event.representative_image_count, 0)
        largest_events.append(
            {
                "species": event.species or "Unknown species",
                "photo_count": event.photo_count,
                "representative_image_count": event.representative_image_count,
                "reducible_image_count": reducible_images,
                "retention_savings_pct": (
                    round((reducible_images / event.photo_count) * 100.0, 1)
                    if event.photo_count
                    else 0.0
                ),
                "duration_min": round(event.duration_sec / 60.0, 1),
                "start_time": event.start_time,
                "end_time": event.end_time,
                "grouping_profile": event.grouping_profile,
                "event_gap_minutes": event.event_gap_minutes,
                "max_duration_minutes": event.max_duration_minutes,
                "eligibility": event.eligibility,
                "fallback_reason": event.fallback_reason,
            }
        )

    largest_event_photo_count = max((event.photo_count for event in events), default=0)

    return {
        "summary": {
            "event_count": event_count,
            "detection_count": detection_count,
            "representative_image_count": representative_count,
            "reducible_image_count": reducible_count,
            "retention_savings_pct": savings_pct,
            "avg_photos_per_event": avg_photos,
            "compression_ratio": compression_ratio,
            "largest_event_photo_count": largest_event_photo_count,
        },
        "largest_events": largest_events,
        "species_pressure": species_pressure[: max(0, int(species_limit))],
        "profile_distribution": profile_distribution,
        "retention_formula": "min(Kmax, 3 + ceil(log2(photo_count)) + bonuses)",
    }


def fetch_analytics_summary(
    conn: sqlite3.Connection,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """
    Returns high-level summary stats for analytics dashboard.

    Filters: active, not no_bird, score >= min_score, must have classification.
    Uses effective_species_sql for species resolution.
    """
    from utils.db.detections import effective_species_sql

    _base_where = """
        d.status = 'active'
        AND (i.review_status IS NULL OR i.review_status != 'no_bird')
        AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
        AND (d.score IS NULL OR d.score >= ?)
        AND EXISTS (
            SELECT 1 FROM classifications c
            WHERE c.detection_id = d.detection_id
              AND c.status = 'active'
        )
    """

    # Total detections
    total_cursor = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_base_where}
        """,
        (min_score,),
    )
    total_detections = total_cursor.fetchone()[0] or 0

    # Total unique species
    species_cursor = conn.execute(
        f"""
        SELECT COUNT(DISTINCT {effective_species_sql("d")}) AS total
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_base_where}
        """,
        (min_score,),
    )
    total_species = species_cursor.fetchone()[0] or 0

    # Date range
    range_cursor = conn.execute(
        f"""
        SELECT
            MIN(substr(i.timestamp, 1, 4) || '-' ||
                substr(i.timestamp, 5, 2) || '-' ||
                substr(i.timestamp, 7, 2)) AS first_date,
            MAX(substr(i.timestamp, 1, 4) || '-' ||
                substr(i.timestamp, 5, 2) || '-' ||
                substr(i.timestamp, 7, 2)) AS last_date
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {_base_where}
        """,
        (min_score,),
    )
    range_row = range_cursor.fetchone()

    return {
        "total_detections": total_detections,
        "total_species": total_species,
        "date_range": {
            "first": range_row["first_date"] if range_row else None,
            "last": range_row["last_date"] if range_row else None,
        },
    }


def fetch_weather_analytics(conn: sqlite3.Connection) -> dict:
    """
    Returns comprehensive weather analytics for the analytics dashboard:
    - current: latest reading
    - today: min/max/avg temps, total precip, avg wind
    - timeline_24h: hourly data points for SVG chart (temp, precip, wind)
    - weekly_summary: per-day aggregates for the last 7 days
    - condition_distribution: % of time per weather condition
    """
    # WMO condition code descriptions
    WMO_CODES = {
        0: ("Clear sky", "☀️"),
        1: ("Mainly clear", "🌤️"),
        2: ("Partly cloudy", "⛅"),
        3: ("Overcast", "☁️"),
        45: ("Fog", "🌫️"),
        48: ("Rime fog", "🌫️"),
        51: ("Light drizzle", "🌦️"),
        53: ("Drizzle", "🌦️"),
        55: ("Dense drizzle", "🌧️"),
        61: ("Slight rain", "🌦️"),
        63: ("Moderate rain", "🌧️"),
        65: ("Heavy rain", "🌧️"),
        71: ("Slight snow", "🌨️"),
        73: ("Moderate snow", "🌨️"),
        75: ("Heavy snow", "❄️"),
        80: ("Rain showers", "🌦️"),
        81: ("Heavy showers", "🌧️"),
        82: ("Violent showers", "⛈️"),
        95: ("Thunderstorm", "⛈️"),
        96: ("T-storm + hail", "⛈️"),
        99: ("T-storm + heavy hail", "⛈️"),
    }

    result = {
        "has_data": False,
        "current": None,
        "today": None,
        "timeline_24h": [],
        "weekly_summary": [],
        "condition_distribution": [],
        "total_readings": 0,
    }

    try:
        # --- Total readings ---
        total_row = conn.execute("SELECT COUNT(*) FROM weather_logs").fetchone()
        result["total_readings"] = total_row[0] if total_row else 0

        if result["total_readings"] == 0:
            return result

        result["has_data"] = True

        # --- Current (latest reading) ---
        cur_row = conn.execute("""
            SELECT timestamp, temp_c, precip_mm, wind_kph, condition_code, is_day
            FROM weather_logs ORDER BY timestamp DESC LIMIT 1
        """).fetchone()

        if cur_row:
            code = cur_row["condition_code"]
            text, emoji = WMO_CODES.get(code, ("Unknown", "❓"))
            result["current"] = {
                "timestamp": cur_row["timestamp"],
                "temp_c": cur_row["temp_c"],
                "precip_mm": cur_row["precip_mm"],
                "wind_kph": cur_row["wind_kph"],
                "condition_code": code,
                "condition_text": text,
                "condition_emoji": emoji,
                "is_day": cur_row["is_day"],
            }

        # --- Today's summary ---
        today_row = conn.execute("""
            SELECT
                MIN(temp_c) AS min_temp,
                MAX(temp_c) AS max_temp,
                ROUND(AVG(temp_c), 1) AS avg_temp,
                ROUND(SUM(precip_mm), 1) AS total_precip,
                ROUND(AVG(wind_kph), 1) AS avg_wind,
                MAX(wind_kph) AS max_wind,
                COUNT(*) AS readings
            FROM weather_logs
            WHERE timestamp >= datetime('now', '-24 hours')
        """).fetchone()

        if today_row and today_row["readings"] > 0:
            result["today"] = {
                "min_temp": today_row["min_temp"],
                "max_temp": today_row["max_temp"],
                "avg_temp": today_row["avg_temp"],
                "total_precip": today_row["total_precip"],
                "avg_wind": today_row["avg_wind"],
                "max_wind": today_row["max_wind"],
                "readings": today_row["readings"],
            }

        # --- 24h Timeline (hourly data points for temperature SVG chart) ---
        timeline_rows = conn.execute("""
            SELECT
                timestamp,
                temp_c,
                precip_mm,
                wind_kph,
                condition_code,
                is_day
            FROM weather_logs
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp ASC
        """).fetchall()

        if timeline_rows:
            temps = [r["temp_c"] for r in timeline_rows if r["temp_c"] is not None]
            winds = [r["wind_kph"] for r in timeline_rows if r["wind_kph"] is not None]
            temp_min = min(temps) if temps else 0
            temp_max = max(temps) if temps else 1
            temp_range = max(temp_max - temp_min, 1)  # avoid /0
            wind_max = max(winds) if winds else 1
            wind_max = max(wind_max, 1)

            # Some templates (e.g. weekly summary bars) rely on global temp_min/temp_max
            # even when the 24h SVG chart can't be rendered yet (e.g. only 1 reading).
            result["temp_min"] = round(temp_min, 1)
            result["temp_max"] = round(temp_max, 1)

            timeline = []
            for _i, r in enumerate(timeline_rows):
                t = r["temp_c"] or 0
                w = r["wind_kph"] or 0
                p = r["precip_mm"] or 0
                code = r["condition_code"]
                text, emoji = WMO_CODES.get(code, ("Unknown", "❓"))

                # Normalize for SVG (0-1)
                temp_norm = (t - temp_min) / temp_range
                wind_norm = w / wind_max

                timeline.append(
                    {
                        "timestamp": r["timestamp"],
                        "temp_c": t,
                        "precip_mm": p,
                        "wind_kph": w,
                        "condition_emoji": emoji,
                        "is_day": r["is_day"],
                        "temp_norm": round(temp_norm, 3),
                        "wind_norm": round(wind_norm, 3),
                        "precip_height": round(min(p * 10, 100), 1),  # scale for bar
                    }
                )

            result["timeline_24h"] = timeline

            # Generate SVG path for temperature line
            n = len(timeline)
            if n > 1:
                svg_width = 800
                svg_height = 120
                padding_top = 10
                usable_height = svg_height - padding_top - 10

                temp_points = []
                wind_points = []
                for i, pt in enumerate(timeline):
                    x = (i / (n - 1)) * svg_width
                    y_temp = padding_top + usable_height * (1 - pt["temp_norm"])
                    y_wind = padding_top + usable_height * (1 - pt["wind_norm"])
                    prefix = "M" if i == 0 else "L"
                    temp_points.append(f"{prefix} {x:.1f} {y_temp:.1f}")
                    wind_points.append(f"{prefix} {x:.1f} {y_wind:.1f}")

                result["temp_svg_path"] = " ".join(temp_points)
                result["wind_svg_path"] = " ".join(wind_points)
                # Area fill path (close at bottom)
                result["temp_svg_area"] = (
                    " ".join(temp_points)
                    + f" L {svg_width} {svg_height} L 0 {svg_height} Z"
                )
                result["wind_svg_area"] = (
                    " ".join(wind_points)
                    + f" L {svg_width} {svg_height} L 0 {svg_height} Z"
                )

        # --- Weekly summary (last 7 days, per day) ---
        weekly_rows = conn.execute("""
            SELECT
                substr(timestamp, 1, 10) AS day,
                ROUND(MIN(temp_c), 1) AS min_temp,
                ROUND(MAX(temp_c), 1) AS max_temp,
                ROUND(AVG(temp_c), 1) AS avg_temp,
                ROUND(SUM(precip_mm), 1) AS total_precip,
                ROUND(AVG(wind_kph), 1) AS avg_wind,
                COUNT(*) AS readings
            FROM weather_logs
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY day
            ORDER BY day ASC
        """).fetchall()

        if weekly_rows:
            # Find max values for bar scaling
            max_precip_day = max(r["total_precip"] or 0 for r in weekly_rows) or 1

            # Ensure temp_min/temp_max encompass the full weekly range.
            # The 24h timeline may set narrower bounds, but the weekly summary
            # bars need the global min/max across all 7 days.
            weekly_mins = [
                r["min_temp"] for r in weekly_rows if r["min_temp"] is not None
            ]
            weekly_maxs = [
                r["max_temp"] for r in weekly_rows if r["max_temp"] is not None
            ]
            if weekly_mins:
                wk_min = round(min(weekly_mins), 1)
                result["temp_min"] = (
                    min(result["temp_min"], wk_min) if "temp_min" in result else wk_min
                )
            if weekly_maxs:
                wk_max = round(max(weekly_maxs), 1)
                result["temp_max"] = (
                    max(result["temp_max"], wk_max) if "temp_max" in result else wk_max
                )

            for r in weekly_rows:
                precip = r["total_precip"] or 0
                # Compute HSL hue for cold→warm gradient:
                # -10°C → 240 (deep blue), 25°C → 0 (red)
                min_t = r["min_temp"] if r["min_temp"] is not None else 0
                max_t = r["max_temp"] if r["max_temp"] is not None else 0
                hue_lo = round(240 * (1 - (max(-10.0, min(25.0, min_t)) + 10) / 35))
                hue_hi = round(240 * (1 - (max(-10.0, min(25.0, max_t)) + 10) / 35))
                result["weekly_summary"].append(
                    {
                        "day": r["day"],
                        "day_short": r["day"][-5:],  # "MM-DD"
                        "min_temp": r["min_temp"],
                        "max_temp": r["max_temp"],
                        "avg_temp": r["avg_temp"],
                        "total_precip": precip,
                        "avg_wind": r["avg_wind"],
                        "precip_pct": round((precip / max_precip_day) * 100, 1),
                        "readings": r["readings"],
                        "hue_lo": hue_lo,
                        "hue_hi": hue_hi,
                    }
                )

        # --- Weather Condition Distribution ---
        cond_rows = conn.execute("""
            SELECT condition_code, COUNT(*) AS n
            FROM weather_logs
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY condition_code
            ORDER BY n DESC
        """).fetchall()

        if cond_rows:
            total_cond = sum(r["n"] for r in cond_rows) or 1
            for r in cond_rows:
                code = r["condition_code"]
                text, emoji = WMO_CODES.get(code, ("Unknown", "❓"))
                pct = round((r["n"] / total_cond) * 100, 1)
                result["condition_distribution"].append(
                    {
                        "code": code,
                        "text": text,
                        "emoji": emoji,
                        "count": r["n"],
                        "pct": pct,
                        "bar_width": pct,  # direct % for CSS bar
                    }
                )

    except Exception:
        pass  # Graceful fallback – weather_logs table might not exist yet

    return result


def fetch_weather_detection_correlation(conn: sqlite3.Connection) -> list[dict]:
    """
    Correlates bird detection activity with weather conditions.
    Groups detections by the weather condition that was active at (± 30 min of) detection time.
    Returns: [{condition_code, condition_text, emoji, detection_count, avg_temp, avg_wind}]
    """
    WMO_CODES = {
        0: ("Clear sky", "☀️"),
        1: ("Mainly clear", "🌤️"),
        2: ("Partly cloudy", "⛅"),
        3: ("Overcast", "☁️"),
        45: ("Fog", "🌫️"),
        48: ("Rime fog", "🌫️"),
        51: ("Light drizzle", "🌦️"),
        53: ("Drizzle", "🌦️"),
        55: ("Dense drizzle", "🌧️"),
        61: ("Slight rain", "🌦️"),
        63: ("Moderate rain", "🌧️"),
        65: ("Heavy rain", "🌧️"),
        71: ("Slight snow", "🌨️"),
        73: ("Moderate snow", "🌨️"),
        75: ("Heavy snow", "❄️"),
        80: ("Rain showers", "🌦️"),
        81: ("Heavy showers", "🌧️"),
        82: ("Violent showers", "⛈️"),
        95: ("Thunderstorm", "⛈️"),
    }

    try:
        # Strategy: Two simple SELECTs + Python dict-lookup (no SQL join).
        # 1. Build weather lookup: hour_key -> {condition_code, temp, wind}
        weather_rows = conn.execute("""
            SELECT timestamp, condition_code, temp_c, wind_kph
            FROM weather_logs
        """).fetchall()

        weather_by_hour = {}  # "YYYY-MM-DDTHH" -> row
        for w in weather_rows:
            hour_key = w["timestamp"][:13]  # "YYYY-MM-DDTHH"
            weather_by_hour[hour_key] = w

        # 2. Fetch detection hours (gallery-aligned visibility)
        det_rows = conn.execute("""
            SELECT
                substr(i.timestamp, 1, 4) || '-' ||
                substr(i.timestamp, 5, 2) || '-' ||
                substr(i.timestamp, 7, 2) || 'T' ||
                substr(i.timestamp, 10, 2) AS hour_key
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            WHERE d.status = 'active'
              AND (i.review_status IS NULL OR i.review_status != 'no_bird')
              AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
        """).fetchall()

        # 3. Match in Python (O(1) per detection)
        from collections import defaultdict

        condition_stats = defaultdict(lambda: {"count": 0, "temps": [], "winds": []})

        for det in det_rows:
            hk = det["hour_key"]
            w = weather_by_hour.get(hk)
            if w:
                code = w["condition_code"]
                condition_stats[code]["count"] += 1
                if w["temp_c"] is not None:
                    condition_stats[code]["temps"].append(w["temp_c"])
                if w["wind_kph"] is not None:
                    condition_stats[code]["winds"].append(w["wind_kph"])

        # 4. Format results
        rows = []
        for code, stats in condition_stats.items():
            avg_temp = (
                round(sum(stats["temps"]) / len(stats["temps"]), 1)
                if stats["temps"]
                else None
            )
            avg_wind = (
                round(sum(stats["winds"]) / len(stats["winds"]), 1)
                if stats["winds"]
                else None
            )
            rows.append(
                {
                    "condition_code": code,
                    "avg_temp": avg_temp,
                    "avg_wind": avg_wind,
                    "detection_count": stats["count"],
                }
            )
        rows.sort(key=lambda r: r["detection_count"], reverse=True)

        result = []
        max_count = max((r["detection_count"] for r in rows), default=1) or 1

        for r in rows:
            code = r["condition_code"]
            text, emoji = WMO_CODES.get(code, ("Unknown", "❓"))
            result.append(
                {
                    "condition_code": code,
                    "condition_text": text,
                    "condition_emoji": emoji,
                    "detection_count": r["detection_count"],
                    "avg_temp": r["avg_temp"],
                    "avg_wind": r["avg_wind"],
                    "bar_pct": round((r["detection_count"] / max_count) * 100, 1),
                }
            )
        return result
    except Exception:
        return []


def _compute_biodiversity_indices(populations: list[int]) -> dict:
    """
    Compute Shannon, Simpson, Evenness from a list of species counts.

    Reusable helper for simulation comparisons.
    """
    import math

    N = sum(populations)
    S = len(populations)

    if N == 0 or S == 0:
        return {
            "shannon": 0,
            "simpson": 0,
            "gini_simpson": 0,
            "evenness": 0,
            "richness": 0,
            "individuals": 0,
        }

    shannon = 0.0
    simpson = 0.0
    for n_i in populations:
        p_i = n_i / N
        if p_i > 0:
            shannon += p_i * math.log(p_i)
            simpson += p_i**2

    shannon = -shannon
    evenness = shannon / math.log(S) if S > 1 else 1.0

    return {
        "shannon": round(shannon, 3),
        "simpson": round(simpson, 3),
        "gini_simpson": round(1 - simpson, 3),
        "evenness": round(evenness, 3),
        "richness": S,
        "individuals": N,
    }


def fetch_simulation_data(
    conn: sqlite3.Connection,
    exclude_species: str | None = None,
) -> dict[str, Any]:
    """
    Species removal simulation: returns daily time series and biodiversity
    indices comparing real data vs. data with one species excluded.

    Uses visual detections (camera data) as the primary source.
    """
    # effective_species_sql() resolves:
    #   - manual override first
    #   - CLS top1 result second
    #   - OD class name third, normalized ('bird'/'unknown'/'unclassified'
    #     collapse to UNKNOWN_SPECIES_KEY; non-bird OD class names like
    #     'squirrel' pass through as species).
    # This replaces raw COALESCE(c.cls_class_name, d.od_class_name) which
    # used to let 'bird' leak through as a species.
    from utils.db.detections import effective_species_sql
    from utils.species_names import UNKNOWN_SPECIES_KEY

    try:
        # 1. Species list for dropdown (gallery-aligned visibility)
        species_cur = conn.execute(f"""
            SELECT DISTINCT {effective_species_sql("d")} AS species
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
                ON d.detection_id = c.detection_id AND c.status = 'active'
            WHERE d.status = 'active'
              AND (i.review_status IS NULL OR i.review_status != 'no_bird')
              AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
              AND {effective_species_sql("d")} != '{UNKNOWN_SPECIES_KEY}'
            ORDER BY species ASC
        """)
        species_list = [r["species"] for r in species_cur.fetchall()]

        if not species_list:
            return {
                "species_list": [],
                "daily_series": [],
                "biodiversity_real": {},
                "biodiversity_sim": {},
                "delta": {},
                "excluded_species": None,
            }

        # 2. Daily time series – real (all species, gallery-aligned)
        daily_real_cur = conn.execute("""
            SELECT
                (substr(i.timestamp, 1, 4) || '-' ||
                 substr(i.timestamp, 5, 2) || '-' ||
                 substr(i.timestamp, 7, 2)) AS date_iso,
                COUNT(*) AS cnt
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
                ON d.detection_id = c.detection_id AND c.status = 'active'
            WHERE d.status = 'active'
              AND (i.review_status IS NULL OR i.review_status != 'no_bird')
              AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
            GROUP BY date_iso
            ORDER BY date_iso ASC
        """)
        real_by_date = {r["date_iso"]: r["cnt"] for r in daily_real_cur.fetchall()}

        # 3. Daily time series – simulated (species excluded)
        sim_by_date: dict[str, int] = {}
        if exclude_species:
            daily_sim_cur = conn.execute(
                """
                SELECT
                    (substr(i.timestamp, 1, 4) || '-' ||
                     substr(i.timestamp, 5, 2) || '-' ||
                     substr(i.timestamp, 7, 2)) AS date_iso,
                    COUNT(*) AS cnt
                FROM detections d
                JOIN images i ON d.image_filename = i.filename
                LEFT JOIN classifications c
                    ON d.detection_id = c.detection_id AND c.status = 'active'
                WHERE d.status = 'active'
                  AND (i.review_status IS NULL OR i.review_status != 'no_bird')
                  AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
                  AND COALESCE(c.cls_class_name, d.od_class_name) != ?
                GROUP BY date_iso
                ORDER BY date_iso ASC
            """,
                (exclude_species,),
            )
            sim_by_date = {r["date_iso"]: r["cnt"] for r in daily_sim_cur.fetchall()}

        # Merge dates
        all_dates = sorted(set(list(real_by_date.keys()) + list(sim_by_date.keys())))
        daily_series = []
        for d in all_dates:
            daily_series.append(
                {
                    "date": d,
                    "real": real_by_date.get(d, 0),
                    "simulated": (
                        sim_by_date.get(d, 0)
                        if exclude_species
                        else real_by_date.get(d, 0)
                    ),
                }
            )

        # 4. Per-species counts for biodiversity indices (gallery-aligned)
        count_cur = conn.execute(f"""
            SELECT
                {effective_species_sql("d")} AS species,
                COUNT(*) AS n
            FROM detections d
            JOIN images i ON d.image_filename = i.filename
            LEFT JOIN classifications c
                ON d.detection_id = c.detection_id AND c.status = 'active'
            WHERE d.status = 'active'
              AND (i.review_status IS NULL OR i.review_status != 'no_bird')
              AND lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')
              AND {effective_species_sql("d")} != '{UNKNOWN_SPECIES_KEY}'
            GROUP BY species
            ORDER BY n DESC
        """)
        species_counts = {r["species"]: r["n"] for r in count_cur.fetchall()}

        # Real biodiversity
        real_pops = list(species_counts.values())
        biodiversity_real = _compute_biodiversity_indices(real_pops)

        # Simulated biodiversity (exclude one species)
        if exclude_species and exclude_species in species_counts:
            sim_pops = [v for k, v in species_counts.items() if k != exclude_species]
        else:
            sim_pops = real_pops
        biodiversity_sim = _compute_biodiversity_indices(sim_pops)

        # Delta
        delta = {}
        for key in ("shannon", "gini_simpson", "evenness", "richness", "individuals"):
            r_val = biodiversity_real.get(key, 0)
            s_val = biodiversity_sim.get(key, 0)
            delta[key] = (
                round(s_val - r_val, 3) if isinstance(r_val, float) else s_val - r_val
            )

        return {
            "species_list": species_list,
            "excluded_species": exclude_species,
            "daily_series": daily_series,
            "biodiversity_real": biodiversity_real,
            "biodiversity_sim": biodiversity_sim,
            "delta": delta,
        }
    except Exception:
        return {
            "species_list": [],
            "daily_series": [],
            "biodiversity_real": {},
            "biodiversity_sim": {},
            "delta": {},
            "excluded_species": None,
        }


# ═══════════════════════════════════════════════════════════════════
# BIRD VISIT CLUSTERING (Spatio-Temporal Session Grouping)
# ═══════════════════════════════════════════════════════════════════

# Tuning constants – safe to adjust without touching the DB
_VISIT_MAX_GAP_SEC = 60  # Max seconds between photos in the same visit
_VISIT_MAX_BBOX_DIST = 0.25  # Max bbox center shift (fraction of image width)
_VISIT_MIN_BBOX_IOU = 0.02  # Allow slight overlap to keep one moving bird together
_VISIT_MIN_AREA_SIMILARITY = 0.2  # Guard against merging birds at very different scale


def _parse_timestamp(ts: str) -> float:
    """Convert WMB timestamp string 'YYYYMMDD_HHMMSS' to epoch seconds.

    Returns 0.0 on parse failure so sorting still works.
    """
    from datetime import datetime

    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _bbox_center_distance(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """Euclidean distance between bounding-box centers (normalised coords)."""
    import math

    cx_a = (ax or 0) + (aw or 0) / 2.0
    cy_a = (ay or 0) + (ah or 0) / 2.0
    cx_b = (bx or 0) + (bw or 0) / 2.0
    cy_b = (by or 0) + (bh or 0) / 2.0
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def _bbox_iou(
    ax: float,
    ay: float,
    aw: float,
    ah: float,
    bx: float,
    by: float,
    bw: float,
    bh: float,
) -> float:
    """Intersection-over-union for normalised xywh boxes."""
    ax1, ay1 = (ax or 0), (ay or 0)
    ax2, ay2 = ax1 + (aw or 0), ay1 + (ah or 0)
    bx1, by1 = (bx or 0), (by or 0)
    bx2, by2 = bx1 + (bw or 0), by1 + (bh or 0)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, (aw or 0) * (ah or 0))
    area_b = max(0.0, (bw or 0) * (bh or 0))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _bbox_area_similarity(aw: float, ah: float, bw: float, bh: float) -> float:
    """Size similarity ratio in [0..1], where 1 means equal bbox area."""
    area_a = max(0.0, (aw or 0) * (ah or 0))
    area_b = max(0.0, (bw or 0) * (bh or 0))
    if area_a <= 0 or area_b <= 0:
        # Missing/invalid box sizes should not break grouping entirely.
        return 1.0
    return min(area_a, area_b) / max(area_a, area_b)


def fetch_bird_visits(
    conn: sqlite3.Connection,
    max_gap_sec: float = _VISIT_MAX_GAP_SEC,
    max_bbox_dist: float = _VISIT_MAX_BBOX_DIST,
    min_bbox_iou: float = _VISIT_MIN_BBOX_IOU,
    min_area_similarity: float = _VISIT_MIN_AREA_SIMILARITY,
    since_timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Group detections into logical bird visits via spatio-temporal clustering.

    Algorithm (read-only, no DB writes):
      1. Query all active detections with timestamp + bbox + species.
      2. Sort globally by timestamp.
      3. Associate each detection to the best matching *open* visit of the
         same species using gated nearest-neighbour matching (time + bbox).
         This is a lightweight MOT-style heuristic (SORT-like association,
         but no Kalman filter) and keeps risk low while handling overlaps.

    Args:
        since_timestamp: Optional YYYYMMDD_HHMMSS filter. If provided, only
            detections on or after this timestamp are included.

    Returns:
        {
            "visits": [
                {
                    "species": str,
                    "start_time": str (YYYYMMDD_HHMMSS),
                    "end_time": str,
                    "duration_sec": float,
                    "photo_count": int,
                    "detection_ids": [int, ...],
                },
                ...
            ],
            "summary": {
                "total_visits": int,
                "total_detections": int,
                "species_visit_counts": {species: visit_count, ...},
                "avg_visit_duration_sec": float,
            },
        }
    """
    # Build optional time filter
    time_filter = ""
    params: tuple = ()
    if since_timestamp:
        time_filter = "AND i.timestamp >= ?"
        params = (since_timestamp,)

    from utils.db.detections import effective_species_sql_for_columns, table_columns
    from utils.species_names import UNKNOWN_SPECIES_KEY

    detection_columns = table_columns(conn, "detections")
    image_columns = table_columns(conn, "images")
    classification_columns = table_columns(conn, "classifications")
    species_sql = effective_species_sql_for_columns(
        "d", detection_columns, classification_columns
    )

    where_clauses = []
    if "status" in detection_columns:
        where_clauses.append("d.status = 'active'")
    if "review_status" in image_columns:
        where_clauses.append(
            "(i.review_status IS NULL OR i.review_status != 'no_bird')"
        )
    if "decision_state" in detection_columns:
        where_clauses.append(
            "lower(COALESCE(d.decision_state, '')) NOT IN ('uncertain', 'unknown')"
        )
    where_clauses.append(f"{species_sql} != '{UNKNOWN_SPECIES_KEY}'")
    if time_filter:
        where_clauses.append("i.timestamp >= ?")
    where_sql = "\n          AND ".join(where_clauses)

    cur = conn.execute(
        f"""
        SELECT
            d.detection_id,
            {species_sql} AS species,
            i.timestamp AS ts,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
        FROM detections d
        JOIN images i ON d.image_filename = i.filename
        WHERE {where_sql}
        ORDER BY i.timestamp ASC, species ASC
    """,
        params,
    )
    rows = cur.fetchall()

    if not rows:
        return {
            "visits": [],
            "summary": {
                "total_visits": 0,
                "total_detections": 0,
                "species_visit_counts": {},
                "avg_visit_duration_sec": 0,
            },
        }

    # ── Clustering pass ──────────────────────────────────────────────
    visits: list[dict[str, Any]] = []
    open_visits: list[dict[str, Any]] = []

    for row in rows:
        det_id = row["detection_id"]
        species = row["species"]
        ts_str = row["ts"] or ""
        epoch = _parse_timestamp(ts_str)
        bx = row["bbox_x"] or 0
        by = row["bbox_y"] or 0
        bw = row["bbox_w"] or 0
        bh = row["bbox_h"] or 0

        # Auto-close stale visits first (no future detection can match anymore).
        still_open: list[dict[str, Any]] = []
        for visit in open_visits:
            if epoch - visit["_last_epoch"] > max_gap_sec:
                visits.append(visit)
            else:
                still_open.append(visit)
        open_visits = still_open

        # Find best same-species candidate with gating (nearest neighbour MOT style).
        best_visit: dict[str, Any] | None = None
        best_cost: float | None = None
        for visit in open_visits:
            if visit["species"] != species:
                continue

            time_diff = epoch - visit["_last_epoch"]
            if time_diff < 0 or time_diff > max_gap_sec:
                continue

            spatial_diff = _bbox_center_distance(
                visit["_last_bx"],
                visit["_last_by"],
                visit["_last_bw"],
                visit["_last_bh"],
                bx,
                by,
                bw,
                bh,
            )
            overlap = _bbox_iou(
                visit["_last_bx"],
                visit["_last_by"],
                visit["_last_bw"],
                visit["_last_bh"],
                bx,
                by,
                bw,
                bh,
            )
            area_similarity = _bbox_area_similarity(
                visit["_last_bw"],
                visit["_last_bh"],
                bw,
                bh,
            )

            if area_similarity < min_area_similarity:
                continue
            if spatial_diff > max_bbox_dist and overlap < min_bbox_iou:
                continue

            # Lower cost is better: normalised time + distance, rewarded by overlap.
            cost = (
                (time_diff / max(max_gap_sec, 1e-6))
                + (spatial_diff / max(max_bbox_dist, 1e-6))
                - 0.5 * overlap
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_visit = visit

        if best_visit is None:
            open_visits.append(
                {
                    "species": species,
                    "start_time": ts_str,
                    "end_time": ts_str,
                    "_last_epoch": epoch,
                    "_start_epoch": epoch,
                    "_last_bx": bx,
                    "_last_by": by,
                    "_last_bw": bw,
                    "_last_bh": bh,
                    "photo_count": 1,
                    "detection_ids": [det_id],
                }
            )
            continue

        best_visit["end_time"] = ts_str
        best_visit["_last_epoch"] = epoch
        best_visit["_last_bx"] = bx
        best_visit["_last_by"] = by
        best_visit["_last_bw"] = bw
        best_visit["_last_bh"] = bh
        best_visit["photo_count"] += 1
        best_visit["detection_ids"].append(det_id)

    visits.extend(open_visits)
    # ── Clean up internal keys & compute duration ────────────────────
    for v in visits:
        v["duration_sec"] = round(v["_last_epoch"] - v["_start_epoch"], 1)
        for key in (
            "_last_epoch",
            "_start_epoch",
            "_last_bx",
            "_last_by",
            "_last_bw",
            "_last_bh",
        ):
            v.pop(key, None)

    # ── Summary ──────────────────────────────────────────────────────
    species_visit_counts: dict[str, int] = {}
    total_dur = 0.0
    for v in visits:
        sp = v["species"]
        species_visit_counts[sp] = species_visit_counts.get(sp, 0) + 1
        total_dur += v["duration_sec"]

    total_visits = len(visits)
    total_detections = sum(v["photo_count"] for v in visits)
    avg_dur = round(total_dur / total_visits, 1) if total_visits else 0

    return {
        "visits": visits,
        "summary": {
            "total_visits": total_visits,
            "total_detections": total_detections,
            "species_visit_counts": species_visit_counts,
            "avg_visit_duration_sec": avg_dur,
        },
    }
