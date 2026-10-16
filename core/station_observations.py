"""Station browsing selection and current observation use cases."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from core.events import build_bird_events
from utils.db import closing_connection
from utils.db.detections import fetch_observation_page, hydrate_observations
from utils.db.observations import selection_sql


def validate_selection(start: str = "", end: str = "") -> dict[str, str]:
    for value in (start, end):
        if value and (
            len(value) != 10 or date.fromisoformat(value).isoformat() != value
        ):
            raise ValueError("Use dates in YYYY-MM-DD format")
    if start and end and start > end:
        raise ValueError("Start date must precede end date")
    return {"start": start, "end": end}


def review_page(
    *, answered: bool, selection: dict[str, str], page: int = 1
) -> tuple[list[dict], int, int]:
    with closing_connection() as conn:
        conn.execute("BEGIN")
        return fetch_observation_page(
            conn, answered=answered, selection=selection, page=page
        )


def grouped_review(
    *, answered: bool, selection: dict[str, str], day: str = "latest", page: int = 1
) -> dict:
    if day not in {"latest", "all"}:
        validate_selection(day, day)
    where, params = selection_sql(selection)
    where += (
        " AND answered" if answered else " AND active AND NOT answered AND NOT admitted"
    )
    with closing_connection() as conn:
        conn.execute("BEGIN")
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM current_observations WHERE "
                + where
                + " ORDER BY timestamp DESC, object_key",
                params,
            )
        ]
        if answered:
            image_where, image_params = selection_sql(selection)
            rows.extend(
                {
                    **dict(row),
                    "object_key": f"image:{row['subject_id']}",
                    "image_statement": True,
                    "species_key": "Unknown_species",
                    "proposed_species": "Unknown_species",
                    "detection_id": None,
                    "answered": True,
                }
                for row in conn.execute(
                    """SELECT f.subject_id, f.image_filename, i.timestamp,
                        i.original_present,
                        MAX(CASE WHEN f.fact_type = 'bird_presence' THEN f.answer_value END) AS image_presence,
                        MAX(CASE WHEN f.fact_type = 'detector_miss' THEN f.answer_value END) AS detector_miss
                        FROM current_human_label_facts f
                        JOIN images i ON i.filename = f.image_filename
                        WHERE f.scope = 'image' AND ("""
                    + image_where
                    + """ )
                        GROUP BY f.subject_id, f.image_filename, i.timestamp, i.original_present
                        ORDER BY i.timestamp DESC, f.subject_id""",
                    image_params,
                )
            )
            rows.sort(
                key=lambda row: (row["timestamp"], row["object_key"]), reverse=True
            )
        days: dict[str, int] = {}
        for row in rows:
            stamp = str(row["timestamp"]).replace("-", "")
            row["day"] = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
            days[row["day"]] = days.get(row["day"], 0) + 1
        selected_day = next(iter(days), "") if day == "latest" else day
        selected = [
            row for row in rows if selected_day == "all" or row["day"] == selected_day
        ]
        # Grouping IDs are local indexes so manual objects participate without
        # colliding with detector IDs. Persisted object identities stay intact.
        events = []
        for event_day in dict.fromkeys(row["day"] for row in selected):
            events.extend(
                build_bird_events(
                    [
                        {
                            **row,
                            "detection_id": index + 1,
                            "species_key": row["proposed_species"],
                        }
                        for index, row in enumerate(selected)
                        if row["day"] == event_day
                    ]
                )
            )
        page_count = max(1, (len(events) + 7) // 8)
        page = max(1, min(page, page_count))
        groups = []
        for event in events[(page - 1) * 8 : page * 8]:
            members = [selected[index - 1] for index in event.detection_ids]
            groups.append(
                {
                    "rows": hydrate_observations(
                        conn, [row for row in members if not row.get("image_statement")]
                    )
                    + [row for row in members if row.get("image_statement")],
                    "event_key": event.event_key,
                    "species": event.species,
                    "eligibility": event.eligibility,
                    "fallback_reason": event.fallback_reason,
                    "day": members[0]["day"],
                    "start": event.start_time,
                    "end": event.end_time,
                }
            )
        return {
            "groups": groups,
            "days": days,
            "selected_day": selected_day,
            "total_items": len(rows),
            "day_items": len(selected),
            "page": page,
            "page_count": page_count,
        }


def media_paths(det: dict) -> dict:
    from urllib.parse import quote

    from config import get_config
    from utils.path_manager import PathManager

    paths = PathManager(str(get_config()["OUTPUT_DIR"]))
    filename = str(det.get("image_filename") or det.get("original_name") or "")
    original = paths.contained_path(
        paths.get_original_path(filename), paths.originals_dir
    )
    optimized = paths.contained_path(
        paths.get_derivative_path(filename, "optimized"), paths.optimized_dir
    )
    thumb = paths.contained_path(
        paths.thumbs_dir / str(det.get("thumbnail_path_virtual") or ""),
        paths.thumbs_dir,
    )

    def url(path: Path | None, root: Path, prefix: str) -> str:
        return (
            prefix + quote(path.relative_to(root).as_posix())
            if path and path.is_file()
            else ""
        )

    original_url = (
        url(original, paths.originals_dir, "/uploads/originals/")
        if det.get("original_present", 1)
        else ""
    )
    optimized_url = url(
        optimized, paths.optimized_dir, "/uploads/derivatives/optimized/"
    )
    thumb_url = url(thumb, paths.thumbs_dir, "/uploads/derivatives/thumbs/")
    full = (
        optimized_url or original_url or thumb_url or "/assets/static_placeholder.jpg"
    )
    return {
        "display_path": thumb_url or full,
        "full_path": full,
        "original_path": original_url,
        "original_available": bool(original_url),
        "optimized_path": optimized_url,
        "media_status": (
            (
                "Original no longer available"
                if not det.get("original_present", 1)
                else "Original unavailable"
            )
            if not original_url
            else ""
        ),
        "media_placeholder": not (original_url or optimized_url or thumb_url),
    }


def species_selection_ids(species: str, selection: dict[str, str]) -> list[int]:
    from utils.db.observations import selection_sql

    where, params = selection_sql(selection)
    if species == "Unknown_species":
        where += " AND unknown_bird"
    else:
        where += " AND species_key = ?"
        params.append(species)
    with closing_connection() as conn:
        return [
            int(row[0])
            for row in conn.execute(
                "SELECT detection_id FROM current_observations WHERE detection_id IS NOT NULL AND "
                + where
                + " ORDER BY timestamp DESC, detection_id DESC",
                params,
            )
        ]


def validate_review_targets(conn: sqlite3.Connection, detection_ids: list[int]) -> None:
    if not detection_ids:
        raise ValueError("Select detector objects in Review")
    rows = []
    for offset in range(0, len(detection_ids), 400):
        batch = detection_ids[offset : offset + 400]
        rows.extend(
            dict(row)
            for row in conn.execute(
                "SELECT * FROM current_observations WHERE detection_id IN ("
                + ",".join("?" for _ in batch)
                + ")",
                batch,
            )
        )
    if len(rows) != len(detection_ids) or any(
        not row["active"] or row["answered"] or row["admitted"] for row in rows
    ):
        raise ValueError("The selection has changed. Reload Review before answering.")
