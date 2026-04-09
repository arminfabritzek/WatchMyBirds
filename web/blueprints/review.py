"""
Review Blueprint.

Handles review queue routes:
- GET /admin/review - Review queue page (was orphans)
- GET /api/review-thumb/<filename> - On-demand thumbnail
- POST /api/review/decision - Review decisions (confirm/trash/no_bird/skip)
- POST /api/review/bbox-review - Persist bbox review state
- POST /api/review/quick-species - Confirm + relabel via quick species buttons
- GET /api/review/event-panel/<event_key> - Event detail fragment
- POST /api/review/event-approve - Approve a whole BirdEvent at once
- POST /api/review/event-trash - Reject / trash a whole BirdEvent at once
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from config import get_config
from core.events import build_bird_events
from logging_config import get_logger
from utils.db import fetch_sibling_detections
from utils.review_metadata import (
    BBOX_REVIEW_CORRECT,
    REVIEW_STATUS_CONFIRMED_BIRD,
    REVIEW_STATUS_NO_BIRD,
    REVIEW_STATUS_UNTAGGED,
    VALID_BBOX_REVIEW_STATES,
    format_review_timestamp,
)
from utils.species_names import (
    UNKNOWN_SPECIES_KEY,
    build_species_picker_entries,
    load_common_names,
    resolve_common_name,
)
from web.blueprints.auth import login_required
from web.services import db_service, gallery_service
from web.species_thumbnails import (
    get_species_thumbnail_map,
    resolve_species_thumbnail_url,
)

logger = get_logger(__name__)
config = get_config()

review_bp = Blueprint("review", __name__)

_REVIEW_ALLOWED_SPECIES_TTL_SECONDS = 60
_review_allowed_species_cache: dict[str, tuple[float, set[str]]] = {}
# Mapping of BirdEvent.fallback_reason values to human-readable labels.
# Keep this list aligned with ``core/events._resolve_event_species`` and
# ``core/events.build_bird_events``: any new fallback the builder can
# emit must show up here too.
_REVIEW_EVENT_FALLBACK_LABELS = {
    "unknown_species": "No reliable species suggestion",
    "partial_unknown_species": "Some frames still need a species decision",
    "multi_bird_ambiguity": "Multiple open birds on one source image",
    "bbox_jump": "Movement is too wide for one safe event confirm",
}


# Per-species colour and reference-image helpers.
#
# The Wong (2011) colour-blind-friendly palette slot count is 8. Colours
# live as CSS custom properties (--species-colour-0 … --species-colour-7)
# in assets/design-system.css. The Python side only cares about the index
# (0..7) and emits it as ``species_colour`` on every review dict so the
# template can self-colour via ``var(--species-colour-N)``.
SPECIES_COLOUR_SLOTS = 8

# Filesystem scan cache for assets/review_species/*.webp|*.png. Built once
# per process (filenames do not change at runtime in the review surface).
_SPECIES_REF_IMAGE_CACHE: dict[str, str] | None = None


def _species_ref_image_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "assets" / "review_species"


def _build_species_ref_image_map() -> dict[str, str]:
    """Scan ``assets/review_species/`` once and return scientific→URL map.

    Filenames use the scientific name with underscores (``Parus_major``)
    and a ``.webp`` or ``.png`` extension, exactly matching the
    ``species_key`` shape used in detection rows. ``.webp`` wins over
    ``.png`` when both exist — smaller payload for the same species.
    """
    ref_dir = _species_ref_image_dir()
    mapping: dict[str, str] = {}
    if not ref_dir.is_dir():
        return mapping
    # Two passes so ``.webp`` always wins over ``.png``.
    for extension in (".webp", ".png"):
        for entry in os.listdir(ref_dir):
            if not entry.endswith(extension):
                continue
            species_key = entry[: -len(extension)]
            if not species_key or species_key in mapping:
                continue
            mapping[species_key] = f"/assets/review_species/{entry}"
    return mapping


def get_species_ref_image_map() -> dict[str, str]:
    """Return the cached scientific→URL map, building it on first use."""
    global _SPECIES_REF_IMAGE_CACHE
    if _SPECIES_REF_IMAGE_CACHE is None:
        _SPECIES_REF_IMAGE_CACHE = _build_species_ref_image_map()
    return _SPECIES_REF_IMAGE_CACHE


def resolve_species_ref_image_url(species_key: str | None) -> str | None:
    """Look up the reference image URL for a scientific name, or ``None``."""
    if not species_key:
        return None
    cleaned = str(species_key).strip()
    if not cleaned:
        return None
    return get_species_ref_image_map().get(cleaned)


def _resolve_species_display_key(payload: dict) -> str:
    """Pick the scientific name a review dict should be coloured by.

    Detection / member dicts carry a mix of fields depending on the
    pipeline stage they come from. Preference order mirrors what the
    templates already display as the species label, so the colour
    matches the visible text.
    """
    for field in ("candidate_species", "species_key", "current_species"):
        value = payload.get(field)
        if value:
            cleaned = str(value).strip()
            if cleaned:
                return cleaned
    return ""


# Moved to core/species_colours.py so Gallery / Stream / Trash routes
# can import it without a circular dependency on this blueprint.
from core.species_colours import (
    SPECIES_COLOUR_SLOTS,
    assign_species_colours,
)


def _score_pct(value) -> int | None:
    if value is None:
        return None
    try:
        return round(float(value) * 100)
    except (TypeError, ValueError):
        return None


def _build_review_quick_species(
    current_species: str | None,
    picker_entries: list[dict],
    recent_species: list[dict],
    common_names: dict[str, str],
    species_thumbnail_map: dict[str, str] | None = None,
    thumbnail_cache_key: str = "review",
    limit: int = 8,
) -> list[dict]:
    quick_species: list[dict] = []
    seen: set[str] = set()
    current_species = str(current_species or "").strip()
    current_score = None
    prediction_entries = [
        entry for entry in picker_entries if entry.get("source") == "prediction"
    ]

    if current_species:
        for entry in prediction_entries:
            if str(entry.get("scientific") or "").strip() == current_species:
                current_score = entry.get("score")
                break

    def add_species(
        scientific_name: str | None,
        *,
        source: str,
        common_name: str | None = None,
        score: float | None = None,
    ) -> None:
        scientific_name = str(scientific_name or "").strip()
        if (
            not scientific_name
            or scientific_name == UNKNOWN_SPECIES_KEY
            or scientific_name in seen
        ):
            return

        seen.add(scientific_name)
        quick_species.append(
            {
                "scientific": scientific_name,
                "common": common_name
                or resolve_common_name(scientific_name, common_names),
                "source": source,
                "score": score,
                "score_pct": _score_pct(score),
                "thumb_url": resolve_species_thumbnail_url(
                    scientific_name,
                    common_names=common_names,
                    thumbnail_map=species_thumbnail_map,
                    cache_key=thumbnail_cache_key,
                ),
            }
        )

    for entry in prediction_entries:
        add_species(
            entry.get("scientific"),
            source="cls",
            common_name=entry.get("common"),
            score=entry.get("score"),
        )
        if len(quick_species) >= limit:
            break

    add_species(current_species, source="current", score=current_score)

    for entry in recent_species:
        add_species(
            entry.get("scientific"),
            source="recent",
            common_name=entry.get("common"),
        )
        if len(quick_species) >= limit:
            break

    return quick_species[:limit]


def _resolve_review_default_species(
    quick_species: list[dict],
    *,
    common_names: dict[str, str],
) -> tuple[str | None, str | None]:
    default_entry = next(
        (entry for entry in quick_species if entry.get("source") == "cls"),
        quick_species[0] if quick_species else None,
    )
    if not default_entry:
        return None, None

    scientific_name = str(default_entry.get("scientific") or "").strip()
    if not scientific_name:
        return None, None

    return (
        scientific_name,
        default_entry.get("common") or resolve_common_name(scientific_name, common_names),
    )


def _resolve_review_selected_species(
    quick_species: list[dict],
    *,
    manual_species_override: str | None,
    common_names: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    manual_species = str(manual_species_override or "").strip()
    if manual_species:
        return (
            manual_species,
            resolve_common_name(manual_species, common_names),
            "manual",
        )

    scientific_name, common_name = _resolve_review_default_species(
        quick_species,
        common_names=common_names,
    )
    if not scientific_name:
        return None, None, None

    return (
        scientific_name,
        common_name,
        "default",
    )


def _load_recent_review_species(conn, common_names: dict[str, str]) -> list[dict]:
    rows = db_service.fetch_recent_review_species(conn, limit=8, lookback_days=7)
    recent_species: list[dict] = []
    for row in rows:
        scientific_name = row["species_key"]
        if not scientific_name or scientific_name == UNKNOWN_SPECIES_KEY:
            continue
        recent_species.append(
            {
                "scientific": scientific_name,
                "common": resolve_common_name(scientific_name, common_names),
                "hit_count": int(row["hit_count"] or 0),
                "last_seen": row["last_seen"],
            }
        )
    return recent_species


def _get_allowed_review_species(
    conn, locale: str, detection_id: int | None = None
) -> set[str]:
    """Return the quick-species allowlist for the given locale/detection."""
    locale = str(locale or "DE").strip().upper() or "DE"
    cached = _review_allowed_species_cache.get(locale)
    now = time.monotonic()
    if cached and (now - cached[0]) < _REVIEW_ALLOWED_SPECIES_TTL_SECONDS:
        allowed_species = set(cached[1])
    else:
        allowed_species = {
            entry["scientific"]
            for entry in build_species_picker_entries(conn, locale=locale)
        }
        allowed_species.update(
            recent_row["species_key"]
            for recent_row in db_service.fetch_recent_review_species(
                conn, limit=128, lookback_days=365
            )
            if recent_row["species_key"]
        )
        _review_allowed_species_cache[locale] = (now, allowed_species)

    if detection_id:
        allowed_species.update(
            entry["scientific"]
            for entry in build_species_picker_entries(
                conn,
                locale=locale,
                detection_id=detection_id,
            )
            if entry.get("scientific")
        )

    return allowed_species

def _fetch_prediction_entries(
    conn, detection_id: int, common_names: dict[str, str]
) -> list[dict]:
    """Return top classifier predictions for a detection (lightweight).

    This replaces the full ``build_species_picker_entries()`` call on the
    panel-render hot path.  The complete picker list is lazy-loaded via
    ``/api/species-list`` only when the user actually opens the picker.
    """
    rows = conn.execute(
        """
        SELECT cls_class_name, cls_confidence, rank
        FROM classifications
        WHERE detection_id = ?
          AND COALESCE(status, 'active') = 'active'
        ORDER BY rank ASC
        LIMIT 5
        """,
        (detection_id,),
    ).fetchall()
    entries: list[dict] = []
    for row in rows:
        scientific = row["cls_class_name"]
        if not scientific:
            continue
        common = common_names.get(scientific, scientific.replace("_", " "))
        entries.append(
            {
                "scientific": scientific,
                "common": common,
                "source": "prediction",
                "score": float(row["cls_confidence"] or 0.0),
                "rank": int(row["rank"] or 0),
            }
        )
    return entries


def _review_reason_label(review_reason: str, max_score: float | None) -> str:
    if review_reason == "orphan":
        return "No Detection"
    if review_reason == "unknown_species":
        return "Unknown Species"
    if review_reason == "uncertain":
        return "Uncertain"
    score_pct = round((max_score or 0) * 100)
    return f"Low Score ({score_pct}%)"


def _split_review_datetime(timestamp: str | None) -> tuple[str, str]:
    timestamp = str(timestamp or "")
    if len(timestamp) < 15:
        return format_review_timestamp(timestamp), ""
    try:
        dt = datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
        return dt.strftime("%d.%m.%Y"), dt.strftime("%H:%M:%S")
    except ValueError:
        return format_review_timestamp(timestamp), ""


def _format_review_duration(duration_sec: float | None) -> str:
    total_seconds = max(0, int(round(float(duration_sec or 0.0))))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m {seconds:02d}s"


def _format_review_event_window(
    start_time: str | None,
    end_time: str | None,
) -> tuple[str, str]:
    start_date, start_clock = _split_review_datetime(start_time)
    end_date, end_clock = _split_review_datetime(end_time)

    if start_date == end_date and start_date:
        if start_clock and end_clock:
            return start_date, f"{start_clock} - {end_clock}"
        return start_date, start_clock or end_clock or ""

    if start_date and end_date and start_clock and end_clock:
        return f"{start_date} - {end_date}", f"{start_clock} - {end_clock}"

    return start_date or end_date or "", start_clock or end_clock or ""


def _review_event_fallback_label(reason: str | None) -> str:
    return _REVIEW_EVENT_FALLBACK_LABELS.get(
        str(reason or "").strip(),
        "Needs individual review",
    )


def _review_event_eligibility_label(eligibility: str | None) -> str:
    return "Event Ready" if eligibility == "event_eligible" else "Review Individually"


def _build_review_modal_siblings(
    conn,
    *,
    filename: str,
    common_names: dict[str, str],
) -> list[dict]:
    siblings: list[dict] = []
    for sibling in fetch_sibling_detections(conn, filename):
        species_key = str(
            sibling["manual_species_override"]
            or sibling["species_key"]
            or sibling["cls_class_name"]
            or sibling["od_class_name"]
            or ""
        ).strip()
        thumb_virtual = sibling["thumbnail_path_virtual"] or ""
        siblings.append(
            {
                "detection_id": sibling["detection_id"],
                "species_key": species_key,
                "common_name": (
                    resolve_common_name(species_key, common_names)
                    if species_key
                    else "Unknown species"
                ),
                "od_class_name": sibling["od_class_name"],
                "od_confidence": sibling["od_confidence"] or 0.0,
                "cls_class_name": sibling["cls_class_name"],
                "cls_confidence": sibling["cls_confidence"] or 0.0,
                "review_status": sibling["review_status"],
                "manual_species_override": sibling["manual_species_override"],
                "species_source": sibling["species_source"],
                "decision_state": sibling["decision_state"],
                "bbox_x": sibling["bbox_x"] or 0.0,
                "bbox_y": sibling["bbox_y"] or 0.0,
                "bbox_w": sibling["bbox_w"] or 0.0,
                "bbox_h": sibling["bbox_h"] or 0.0,
                "thumb_url": (
                    f"/uploads/derivatives/thumbs/{thumb_virtual}"
                    if thumb_virtual
                    else ""
                ),
            }
        )
    return siblings


def _build_review_modal_detection(
    row,
    *,
    filename: str,
    full_url: str,
    thumb_url: str,
    selected_species: str | None,
    selected_species_common: str | None,
    current_species: str | None,
    current_species_common: str | None,
    common_names: dict[str, str],
    conn,
) -> dict | None:
    detection_id = row["active_detection_id"] or row["best_detection_id"]
    if not detection_id:
        return None

    formatted_date, formatted_time = _split_review_datetime(row["timestamp"])
    gallery_date = (
        f"{row['timestamp'][:4]}-{row['timestamp'][4:6]}-{row['timestamp'][6:8]}"
        if row["timestamp"] and len(row["timestamp"]) >= 8
        else None
    )
    species_key = (
        str(selected_species or "").strip()
        or str(current_species or "").strip()
        or str(row["manual_species_override"] or "").strip()
        or str(row["species_key"] or "").strip()
    )
    common_name = (
        str(selected_species_common or "").strip()
        or str(current_species_common or "").strip()
        or (
            resolve_common_name(species_key, common_names)
            if species_key
            else filename
        )
    )
    siblings = _build_review_modal_siblings(
        conn,
        filename=filename,
        common_names=common_names,
    )

    return {
        "detection_id": detection_id,
        "species_key": species_key,
        "common_name": common_name,
        "od_class_name": row["species_key"] or "",
        "od_confidence": row["od_confidence"] or 0.0,
        "cls_class_name": row["species_key"] or "",
        "cls_confidence": row["cls_confidence"] or 0.0,
        "score": row["max_score"] or 0.0,
        "review_status": row["review_status"],
        "manual_species_override": row["manual_species_override"],
        "species_source": row["species_source"],
        "formatted_date": formatted_date,
        "formatted_time": formatted_time,
        "gallery_date": gallery_date,
        "siblings": siblings,
        "sibling_count": max(
            len(siblings),
            int(row["sibling_detection_count"] or 0),
            1,
        ),
        "bbox_x": row["bbox_x"] or 0.0,
        "bbox_y": row["bbox_y"] or 0.0,
        "bbox_w": row["bbox_w"] or 0.0,
        "bbox_h": row["bbox_h"] or 0.0,
        "is_favorite": False,
        "decision_state": row["decision_state"],
        "display_path": thumb_url,
        "full_path": full_url or thumb_url,
        "original_path": full_url or thumb_url,
    }


def _build_review_item(
    row,
    *,
    conn,
    species_locale: str,
    output_dir: str,
    common_names: dict[str, str],
    recent_species: list[dict],
    species_thumbnail_map: dict[str, str] | None = None,
    include_detail: bool = True,
) -> dict:
    item_kind = row["item_kind"] or "image"
    item_id = str(row["item_id"] or row["filename"] or "")
    filename = row["filename"]
    timestamp = row["timestamp"] or ""
    review_reason = row["review_reason"]
    max_score = row["max_score"]
    best_detection_id = row["active_detection_id"] or row["best_detection_id"]
    current_species = row["species_key"]
    picker_entries = []
    quick_species: list[dict] = []
    default_species = None
    default_species_common = None
    selected_species = None
    selected_species_common = None
    selected_species_origin = None
    if include_detail and best_detection_id:
        picker_entries = _fetch_prediction_entries(
            conn, best_detection_id, common_names
        )
        quick_species = _build_review_quick_species(
            current_species,
            picker_entries,
            recent_species,
            common_names,
            species_thumbnail_map=species_thumbnail_map,
            thumbnail_cache_key=f"review:{species_locale}",
        )
        default_species, default_species_common = _resolve_review_default_species(
            quick_species,
            common_names=common_names,
        )
        selected_species, selected_species_common, selected_species_origin = (
            _resolve_review_selected_species(
                quick_species,
                manual_species_override=row["manual_species_override"],
                common_names=common_names,
            )
        )
    elif include_detail:
        default_species, default_species_common = _resolve_review_default_species(
            quick_species,
            common_names=common_names,
        )
        selected_species, selected_species_common, selected_species_origin = (
            _resolve_review_selected_species(
                quick_species,
                manual_species_override=row["manual_species_override"],
                common_names=common_names,
            )
        )
    manual_bbox_review = row["manual_bbox_review"]
    selected_bbox_review = (
        manual_bbox_review
        if manual_bbox_review in VALID_BBOX_REVIEW_STATES
        else (BBOX_REVIEW_CORRECT if best_detection_id else None)
    )
    selected_bbox_review_origin = (
        "manual"
        if manual_bbox_review in VALID_BBOX_REVIEW_STATES
        else ("default" if best_detection_id else None)
    )

    thumb_url = f"/api/review-thumb/{filename}"
    full_url = ""
    if len(timestamp) >= 8:
        date_folder_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        full_url = f"/uploads/originals/{date_folder_str}/{filename}"

    item = {
        "item_kind": item_kind,
        "item_id": item_id,
        "filename": filename,
        "source_image_filename": row["source_image_filename"] or filename,
        "timestamp": timestamp,
        "gallery_date": f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}" if len(timestamp) >= 8 else None,
        "formatted_date": format_review_timestamp(timestamp),
        "thumb_url": thumb_url,
        "full_url": full_url,
        "source_image_thumb_url": thumb_url,
        "source_image_full_url": full_url,
        "review_reason": review_reason,
        "reason_label": _review_reason_label(review_reason, max_score),
        "max_score": max_score,
        "best_detection_id": best_detection_id,
        "active_detection_id": best_detection_id,
        "species_key": current_species,
        "current_species_common": resolve_common_name(current_species, common_names),
        "manual_species_override": row["manual_species_override"],
        "manual_species_common": resolve_common_name(
            row["manual_species_override"], common_names
        ),
        "species_source": row["species_source"],
        "quick_species": quick_species if include_detail else [],
        "default_species": default_species if include_detail else None,
        "default_species_common": (
            default_species_common if include_detail else None
        ),
        "selected_species": selected_species if include_detail else None,
        "selected_species_common": (
            selected_species_common if include_detail else None
        ),
        "selected_species_origin": selected_species_origin if include_detail else None,
        "bbox_x": row["bbox_x"],
        "bbox_y": row["bbox_y"],
        "bbox_w": row["bbox_w"],
        "bbox_h": row["bbox_h"],
        "active_detection_bbox": {
            "x": row["bbox_x"],
            "y": row["bbox_y"],
            "w": row["bbox_w"],
            "h": row["bbox_h"],
        }
        if best_detection_id
        else None,
        "manual_bbox_review": manual_bbox_review,
        "selected_bbox_review": selected_bbox_review,
        "selected_bbox_review_origin": selected_bbox_review_origin,
        "can_approve": bool(
            include_detail
            and best_detection_id
            and selected_species
            and selected_bbox_review in VALID_BBOX_REVIEW_STATES
        ),
        "has_detection": bool(best_detection_id),
        "decision_state": row["decision_state"],
        "active_detection_species": current_species,
        "active_detection_status": row["decision_state"],
        "bbox_quality": row["bbox_quality"],
        "bbox_quality_pct": _score_pct(row["bbox_quality"]),
        "unknown_score": row["unknown_score"],
        "unknown_score_pct": _score_pct(row["unknown_score"]),
        "decision_reasons": row["decision_reasons"],
        "od_confidence": row["od_confidence"],
        "od_confidence_pct": _score_pct(row["od_confidence"]),
        "cls_confidence": row["cls_confidence"],
        "cls_confidence_pct": _score_pct(row["cls_confidence"]),
        "sibling_detection_count": int(row["sibling_detection_count"] or 0),
        "item_key": f"{item_kind}:{item_id}",
    }
    item["modal_detection"] = (
        _build_review_modal_detection(
            row,
            filename=filename,
            full_url=full_url,
            thumb_url=thumb_url,
            selected_species=selected_species,
            selected_species_common=selected_species_common,
            current_species=current_species,
            current_species_common=item["current_species_common"],
            common_names=common_names,
            conn=conn,
        )
        if include_detail
        else None
    )
    return item


def _fetch_review_event_payloads(
    conn,
    *,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
) -> tuple[list, dict[int, dict], list[dict], bool]:
    """Load detection rows + Gallery context and group them into BirdEvents.

    The Review surface uses ``core.events.build_bird_events``. Confirmed
    Gallery detections inside the Review neighbourhood are also fetched
    as ``context_only=True`` inputs so the biological event boundary
    stays intact across the Review / Gallery border.
    """
    rows = db_service.fetch_review_queue_images(conn, gallery_threshold)
    recent_species = _load_recent_review_species(conn, common_names)
    detection_rows: dict[int, dict] = {}
    untagged_min: str | None = None
    untagged_max: str | None = None
    for row in rows:
        if row["item_kind"] != "detection":
            continue
        row_dict = dict(row)
        detection_id = int(
            row_dict.get("active_detection_id")
            or row_dict.get("best_detection_id")
            or 0
        )
        if detection_id <= 0:
            continue
        row_dict["detection_id"] = detection_id
        row_dict["context_only"] = False
        detection_rows[detection_id] = row_dict
        ts = str(row_dict.get("timestamp") or "")
        if ts:
            if untagged_min is None or ts < untagged_min:
                untagged_min = ts
            if untagged_max is None or ts > untagged_max:
                untagged_max = ts

    context_truncated = False
    if untagged_min and untagged_max:
        try:
            context_rows, context_truncated = db_service.fetch_review_cluster_context(
                conn,
                untagged_time_range=(untagged_min, untagged_max),
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("fetch_review_cluster_context failed; continuing without context")
            context_rows = []
        for row_dict in context_rows:
            detection_id = int(
                row_dict.get("active_detection_id")
                or row_dict.get("best_detection_id")
                or 0
            )
            if detection_id <= 0 or detection_id in detection_rows:
                continue
            row_dict["detection_id"] = detection_id
            row_dict["context_only"] = True
            detection_rows[detection_id] = row_dict

    raw_events = build_bird_events(list(detection_rows.values()))
    return raw_events, detection_rows, recent_species, context_truncated


def build_review_continuity_batches(raw_events: list) -> list[dict]:
    """Group split BirdEvents that share a confirmed Gallery anchor.

    A continuity batch collects every actionable BirdEvent in a
    Gallery-anchored neighbourhood so the operator can compare them in
    one place.

    Rules:
    - A batch only forms when at least one **pure context** BirdEvent
      (``context_only_count == photo_count``) sits in the same time
      neighbourhood as at least one actionable BirdEvent. Shared time
      alone without a confirmed Gallery anchor is **not** enough.
    - Anchor neighbourhood is ``[anchor.start_time-30min,
      anchor.end_time+30min]``. An actionable event joins the batch
      when its time window overlaps that interval.
    - ``recommended_species`` is only set when **all** confirmed
      anchors in the batch resolve to exactly one species.
    - ``review_detection_ids`` only contains actionable detection ids.
      ``context_only`` detections are read-only and stay out of the
      batch action targets.
    - ``batch_bbox_map`` is a flat list of bbox entries (one per
      detection across all members of the batch) carrying
      ``context_only``, ``event_key``, and ``trail_role`` so the UI can
      render one combined continuity mini-map without recomputing
      geometry.
    """
    if not raw_events:
        return []

    anchors: list[Any] = []
    actionables: list[Any] = []
    for event in raw_events:
        is_pure_context = (
            event.context_only_count > 0
            and event.context_only_count == event.photo_count
        )
        if is_pure_context:
            anchors.append(event)
        elif event.context_only_count < event.photo_count:
            actionables.append(event)

    if not anchors or not actionables:
        return []

    batches: list[dict] = []
    seen_anchor_keys: set[str] = set()
    for anchor in anchors:
        if anchor.event_key in seen_anchor_keys:
            continue
        window_start = _shift_review_window(anchor.start_time, minutes=-30)
        window_end = _shift_review_window(anchor.end_time, minutes=30)
        joined_actionables = [
            event
            for event in actionables
            if _events_overlap(event.start_time, event.end_time, window_start, window_end)
        ]
        if not joined_actionables:
            continue

        # Pull every confirmed anchor that overlaps the same neighbourhood
        # so a batch with two confirmed anchors of the same species still
        # earns its recommended_species.
        joined_anchors = [
            other
            for other in anchors
            if _events_overlap(
                other.start_time, other.end_time, window_start, window_end
            )
        ]
        for joined in joined_anchors:
            seen_anchor_keys.add(joined.event_key)

        anchor_species = {
            other.species for other in joined_anchors if other.species
        }
        recommended_species: str | None = None
        if len(anchor_species) == 1:
            recommended_species = next(iter(anchor_species))

        review_detection_ids: list[int] = []
        for event in joined_actionables:
            for trail_point in event.bbox_trail:
                if trail_point.get("context_only"):
                    continue
                review_detection_ids.append(int(trail_point.get("detection_id") or 0))
        review_detection_ids = [det_id for det_id in review_detection_ids if det_id > 0]

        context_detection_ids: list[int] = []
        context_species_summary: dict[str, int] = {}
        for joined in joined_anchors:
            for trail_point in joined.bbox_trail:
                det_id = int(trail_point.get("detection_id") or 0)
                if det_id > 0:
                    context_detection_ids.append(det_id)
            if joined.species:
                context_species_summary[joined.species] = (
                    context_species_summary.get(joined.species, 0) + joined.photo_count
                )
        # Also pick up context_only members that were attached to actionable
        # events (same-species continuation): they belong to the read-only
        # set, never to review_detection_ids.
        for event in joined_actionables:
            for trail_point in event.bbox_trail:
                if not trail_point.get("context_only"):
                    continue
                det_id = int(trail_point.get("detection_id") or 0)
                if det_id > 0:
                    context_detection_ids.append(det_id)

        batch_bbox_map: list[dict] = []
        for joined in joined_anchors:
            for trail_point in joined.bbox_trail:
                batch_bbox_map.append(
                    {
                        **trail_point,
                        "event_key": joined.event_key,
                        "context_only": True,
                    }
                )
        for event in joined_actionables:
            for trail_point in event.bbox_trail:
                batch_bbox_map.append(
                    {
                        **trail_point,
                        "event_key": event.event_key,
                        "context_only": bool(trail_point.get("context_only")),
                    }
                )

        all_event_starts = [event.start_time for event in joined_anchors + joined_actionables]
        all_event_ends = [event.end_time for event in joined_anchors + joined_actionables]
        batch_window_start = min(all_event_starts) if all_event_starts else anchor.start_time
        batch_window_end = max(all_event_ends) if all_event_ends else anchor.end_time

        batch_key = "review-batch-" + "-".join(
            sorted(joined.event_key for joined in joined_anchors)
        )

        batches.append(
            {
                "batch_key": batch_key,
                "window_start": batch_window_start,
                "window_end": batch_window_end,
                "event_keys": [event.event_key for event in joined_actionables],
                "anchor_event_keys": [joined.event_key for joined in joined_anchors],
                "review_detection_ids": review_detection_ids,
                "context_detection_ids": context_detection_ids,
                "context_species_summary": context_species_summary,
                "recommended_species": recommended_species,
                "batch_bbox_map": batch_bbox_map,
            }
        )
    return batches


def _shift_review_window(ts: str, *, minutes: int) -> str:
    """Shift a Review timestamp by minutes for batch neighbourhood checks."""
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except (TypeError, ValueError):
        return ts
    return (dt + timedelta(minutes=minutes)).strftime("%Y%m%d_%H%M%S")


def _events_overlap(
    a_start: str, a_end: str, b_start: str, b_end: str
) -> bool:
    """Two ``YYYYMMDD_HHMMSS`` intervals overlap."""
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return a_start <= b_end and b_start <= a_end


def _build_review_event_member(
    row: dict,
    *,
    conn,
    species_locale: str,
    output_dir: str,
    common_names: dict[str, str],
    recent_species: list[dict],
) -> dict:
    member = _build_review_item(
        row,
        conn=conn,
        species_locale=species_locale,
        output_dir=output_dir,
        common_names=common_names,
        recent_species=recent_species,
        include_detail=False,
    )
    formatted_date, formatted_time = _split_review_datetime(row.get("timestamp"))
    member["formatted_short_date"] = formatted_date
    member["formatted_time"] = formatted_time
    member["candidate_species"] = (
        str(row.get("manual_species_override") or "").strip()
        or str(row.get("cls_class_name") or "").strip()
        or None
    )
    member["candidate_species_common"] = resolve_common_name(
        member["candidate_species"], common_names
    )
    # Post fixed-5 (2026-04-08): the grid cell label needs to know
    # whether its species came from a per-frame manual override or from
    # the auto-CLS guess. The Event-level species picker must only
    # update auto-species cells — manual overrides win and stay locked
    # to their user-confirmed value. The JS layer reads this through
    # `data-species-is-manual` on the cell root.
    member["species_is_manual"] = bool(
        str(row.get("manual_species_override") or "").strip()
    )
    member["context_only"] = bool(row.get("context_only") or False)
    if member["context_only"]:
        # Continuity context frames are confirmed Gallery anchors, not
        # actionable low-score review items. The event grid keeps them
        # visible as read-only reference frames, so their caption should
        # explain that role directly instead of echoing the stale queue reason.
        member["review_reason"] = "context"
        member["reason_label"] = "In Gallery"
    return member


def _build_review_event(
    raw_event,
    *,
    row_map: dict[int, dict],
    conn,
    species_locale: str,
    output_dir: str,
    common_names: dict[str, str],
    recent_species: list[dict],
    include_detail: bool = False,
) -> dict | None:
    """Translate a ``BirdEvent`` into the dict shape the templates expect.

    The template-side keys deliberately keep the same shape as the
    cluster surface used to (cover_*, members, eligibility, ...) so the
    review_event_panel.html partial only needs to swap the action verbs.
    """
    cover_detection_id = int(raw_event.cover_detection_id or 0)
    cover_row = row_map.get(cover_detection_id)
    if not cover_row:
        return None

    cover_item = _build_review_event_member(
        cover_row,
        conn=conn,
        species_locale=species_locale,
        output_dir=output_dir,
        common_names=common_names,
        recent_species=recent_species,
    )
    window_date, window_time = _format_review_event_window(
        raw_event.start_time,
        raw_event.end_time,
    )
    candidate_species = raw_event.species
    candidate_species_common = (
        resolve_common_name(candidate_species, common_names)
        or cover_item.get("current_species_common")
        or "Unknown species"
    )
    bbox_trail = list(raw_event.bbox_trail or [])
    bbox_trail_preview = [
        point for point in bbox_trail if point.get("trail_role") in {"start", "mid", "end"}
    ] or bbox_trail

    # Expose the cover frame's native resolution so trail maps that
    # render bbox data in percentage space can match the real camera
    # aspect ratio instead of assuming 16:9. Falls back to None if the
    # column is missing on older rows.
    cover_frame_width = None
    cover_frame_height = None
    if cover_detection_id:
        try:
            dim_row = conn.execute(
                "SELECT frame_width, frame_height FROM detections WHERE detection_id = ? LIMIT 1",
                (cover_detection_id,),
            ).fetchone()
        except Exception:
            dim_row = None
        if dim_row:
            raw_w = dim_row["frame_width"] if "frame_width" in dim_row.keys() else None
            raw_h = dim_row["frame_height"] if "frame_height" in dim_row.keys() else None
            try:
                cover_frame_width = int(raw_w) if raw_w else None
            except (TypeError, ValueError):
                cover_frame_width = None
            try:
                cover_frame_height = int(raw_h) if raw_h else None
            except (TypeError, ValueError):
                cover_frame_height = None

    event_payload = {
        "event_key": raw_event.event_key,
        "cover_detection_id": cover_detection_id,
        "detection_ids": list(raw_event.detection_ids),
        "candidate_species": candidate_species,
        "candidate_species_common": candidate_species_common,
        "species_source": raw_event.species_source,
        "photo_count": int(raw_event.photo_count),
        "duration_sec": float(raw_event.duration_sec),
        "duration_display": _format_review_duration(raw_event.duration_sec),
        "start_time": raw_event.start_time or "",
        "end_time": raw_event.end_time or "",
        "window_date": window_date,
        "window_time": window_time,
        "bbox_trail": bbox_trail,
        "bbox_trail_preview": bbox_trail_preview,
        "cover_frame_width": cover_frame_width,
        "cover_frame_height": cover_frame_height,
        "eligibility": raw_event.eligibility,
        "eligibility_label": _review_event_eligibility_label(raw_event.eligibility),
        "fallback_reason": raw_event.fallback_reason,
        "fallback_label": _review_event_fallback_label(raw_event.fallback_reason),
        "touched_filenames": list(raw_event.touched_filenames),
        "cover_item": cover_item,
        "cover_thumb_url": cover_item["thumb_url"],
        "cover_full_url": cover_item["full_url"],
        "quick_species": [],
        "default_species": None,
        "default_species_common": None,
        "selected_species": candidate_species,
        "selected_species_common": candidate_species_common,
        "selected_species_origin": raw_event.species_source or None,
    }

    if not include_detail:
        return event_payload

    picker_entries = []
    quick_species: list[dict] = []
    default_species = None
    default_species_common = None
    if cover_detection_id:
        picker_entries = _fetch_prediction_entries(
            conn, cover_detection_id, common_names
        )
        quick_species = _build_review_quick_species(
            candidate_species or cover_item.get("species_key"),
            picker_entries,
            recent_species,
            common_names,
            thumbnail_cache_key=f"review:{species_locale}",
        )
        default_species, default_species_common = _resolve_review_default_species(
            quick_species,
            common_names=common_names,
        )

    selected_species = str(candidate_species or default_species or "").strip() or None
    selected_species_common = (
        resolve_common_name(selected_species, common_names)
        if selected_species
        else default_species_common
    )
    selected_species_origin = (
        str(raw_event.species_source or "").strip()
        or ("default" if selected_species else None)
    )

    event_payload["quick_species"] = quick_species
    event_payload["default_species"] = default_species
    event_payload["default_species_common"] = default_species_common
    event_payload["selected_species"] = selected_species
    event_payload["selected_species_common"] = selected_species_common
    event_payload["selected_species_origin"] = selected_species_origin

    members: list[dict] = []
    for detection_id in raw_event.detection_ids:
        row = row_map.get(int(detection_id))
        if not row:
            continue
        members.append(
            _build_review_event_member(
                row,
                conn=conn,
                species_locale=species_locale,
                output_dir=output_dir,
                common_names=common_names,
                recent_species=recent_species,
            )
        )

    members.sort(
        key=lambda member: (
            member.get("timestamp", ""),
            int(member.get("best_detection_id") or 0),
        )
    )
    event_payload["members"] = members
    event_payload["event_bbox_review"] = BBOX_REVIEW_CORRECT
    return event_payload


def _load_review_items(
    conn,
    *,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    rows = db_service.fetch_review_queue_images(conn, gallery_threshold)
    recent_species = _load_recent_review_species(conn, common_names)
    items = [
        _build_review_item(
            row,
            conn=conn,
            species_locale=species_locale,
            output_dir=output_dir,
            common_names=common_names,
            recent_species=recent_species,
            include_detail=False,
        )
        for row in rows
    ]
    return items, recent_species


def _load_single_review_item(
    conn,
    *,
    filename: str,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
    recent_species: list[dict] | None = None,
    species_thumbnail_map: dict[str, str] | None = None,
) -> dict | None:
    row = db_service.fetch_review_queue_image(
        conn,
        filename,
        gallery_threshold=gallery_threshold,
    )
    if not row:
        return None
    if recent_species is None:
        recent_species = _load_recent_review_species(conn, common_names)
    return _build_review_item(
        row,
        conn=conn,
        species_locale=species_locale,
        output_dir=output_dir,
        common_names=common_names,
        recent_species=recent_species,
        species_thumbnail_map=species_thumbnail_map,
    )


def _load_single_review_item_by_identity(
    conn,
    *,
    item_kind: str,
    item_id: str,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
    recent_species: list[dict] | None = None,
    species_thumbnail_map: dict[str, str] | None = None,
) -> dict | None:
    row = db_service.fetch_review_queue_item_by_identity(
        conn,
        item_kind,
        item_id,
        gallery_threshold=gallery_threshold,
    )
    if not row:
        return None
    if recent_species is None:
        recent_species = _load_recent_review_species(conn, common_names)
    return _build_review_item(
        row,
        conn=conn,
        species_locale=species_locale,
        output_dir=output_dir,
        common_names=common_names,
        recent_species=recent_species,
        species_thumbnail_map=species_thumbnail_map,
    )


def _load_review_events(
    conn,
    *,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
    include_detail: bool = False,
) -> tuple[list[dict], list[dict], bool, set[str]]:
    """Return ``(review_events, continuity_batches, context_truncated, workspace_species_keys)``.

    The events list excludes pure-context-only BirdEvents (nothing to
    review there). Continuity batch payloads are computed from the full
    raw event list before that filter so a batch can still surface its
    Gallery anchors via ``batch_bbox_map`` and
    ``context_detection_ids``.
    """
    (
        raw_events,
        row_map,
        recent_species,
        context_truncated,
    ) = _fetch_review_event_payloads(
        conn,
        gallery_threshold=gallery_threshold,
        output_dir=output_dir,
        species_locale=species_locale,
        common_names=common_names,
    )
    continuity_batches = build_review_continuity_batches(raw_events)
    batch_lookup_by_event_key: dict[str, dict] = {}
    for batch in continuity_batches:
        for event_key in batch.get("event_keys") or []:
            batch_lookup_by_event_key[event_key] = batch

    # Build one workspace-scoped colour map so context anchors and
    # actionable events share deterministic slots. Stamping happens
    # here so callers do not need to know about colour assignment.
    workspace_species_keys = _workspace_species_keys_from_raw_events(raw_events)
    workspace_colour_map = assign_species_colours(list(workspace_species_keys))

    events: list[dict] = []
    for raw_event in raw_events:
        if (
            raw_event.context_only_count > 0
            and raw_event.context_only_count == raw_event.photo_count
        ):
            # Pure Gallery anchor — Review has nothing actionable to show.
            continue
        event = _build_review_event(
            raw_event,
            row_map=row_map,
            conn=conn,
            species_locale=species_locale,
            output_dir=output_dir,
            common_names=common_names,
            recent_species=recent_species,
            include_detail=include_detail,
        )
        if not event:
            continue
        event["context_only_count"] = int(raw_event.context_only_count)
        event["context_anchored"] = bool(raw_event.context_anchored)
        batch = batch_lookup_by_event_key.get(raw_event.event_key)
        if batch is not None:
            event["continuity_batch_key"] = batch["batch_key"]
            event["continuity_batch_recommended_species"] = batch["recommended_species"]
            event["continuity_batch_review_detection_ids"] = list(
                batch["review_detection_ids"]
            )
        _stamp_species_display_on_event(event, workspace_colour_map)
        events.append(event)
    return events, continuity_batches, context_truncated, workspace_species_keys


def _stamp_species_display_on_event(
    event: dict,
    colour_map: dict[str, int],
) -> None:
    """Apply ``species_colour`` + ``species_ref_image_url`` to one event.

    Stamps the event itself, every member, every continuity-batch
    member (anchor + review), and every quick-pick species so the
    template can read the slot off any payload.
    """
    def stamp(payload: dict) -> None:
        key = _resolve_species_display_key(payload)
        if key:
            payload["species_colour_key"] = key
            payload["species_colour"] = colour_map.get(key)
            payload["species_ref_image_url"] = resolve_species_ref_image_url(key)
        else:
            payload["species_colour_key"] = ""
            payload["species_colour"] = None
            payload["species_ref_image_url"] = None

    stamp(event)
    for member in event.get("members") or []:
        stamp(member)
    batch = event.get("continuity_batch")
    if batch:
        for member in batch.get("review_members") or []:
            stamp(member)
        for member in batch.get("anchor_members") or []:
            stamp(member)
    for picker in event.get("quick_species") or []:
        key = str(picker.get("scientific") or "").strip()
        picker["species_colour"] = colour_map.get(key) if key else None
        picker["species_ref_image_url"] = (
            resolve_species_ref_image_url(key) if key else None
        )


def _load_single_review_event(
    conn,
    *,
    event_key: str,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
) -> dict | None:
    events, _batches, _truncated, _species_keys = _load_review_events(
        conn,
        gallery_threshold=gallery_threshold,
        output_dir=output_dir,
        species_locale=species_locale,
        common_names=common_names,
        include_detail=True,
    )
    for event in events:
        if event.get("event_key") == event_key:
            return event
    return None


def _load_event_with_continuity_batch(
    conn,
    *,
    event_key: str,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
) -> dict | None:
    """Load one Review event plus its continuity batch.

    Returns the event payload from ``_load_single_review_event``, but
    when the event participates in a continuity batch the payload also
    carries an ``event["continuity_batch"]`` dict shaped for the
    template:

        {
            "batch_key": str,
            "recommended_species": str | None,
            "recommended_species_common": str | None,
            "review_detection_ids": [int, ...],
            "context_detection_ids": [int, ...],
            "context_species_summary": {sci_name: int, ...},
            "anchor_members": [member_dict, ...],   # context anchors
            "review_members": [member_dict, ...],   # actionable frames
            "review_event_keys": [event_key, ...],
            "anchor_event_keys": [event_key, ...],
            "batch_bbox_map": [trail_point, ...],
        }

    ``review_members`` and ``anchor_members`` are materialized via
    ``_build_review_event_member`` so the template can render them with
    the same shape as ``event.members``. Sibling actionable events that
    share the batch are folded into ``review_members`` so the operator
    sees every comparable frame in one stage without flipping events.
    """
    raw_events, row_map, recent_species, _truncated = _fetch_review_event_payloads(
        conn,
        gallery_threshold=gallery_threshold,
        output_dir=output_dir,
        species_locale=species_locale,
        common_names=common_names,
    )
    if not raw_events:
        return None

    raw_event = next(
        (event for event in raw_events if event.event_key == event_key),
        None,
    )
    if raw_event is None:
        return None

    # Skip pure-context anchors — Review never opens an anchor as the
    # primary event, even if a stale fragment URL points at one.
    if (
        raw_event.context_only_count > 0
        and raw_event.context_only_count == raw_event.photo_count
    ):
        return None

    event_payload = _build_review_event(
        raw_event,
        row_map=row_map,
        conn=conn,
        species_locale=species_locale,
        output_dir=output_dir,
        common_names=common_names,
        recent_species=recent_species,
        include_detail=True,
    )
    if not event_payload:
        return None
    event_payload["context_only_count"] = int(raw_event.context_only_count)
    event_payload["context_anchored"] = bool(raw_event.context_anchored)

    # Rebuild the workspace-scoped colour map from the full raw-event
    # set so fragment rendering matches the page rail for overlapping
    # species.
    workspace_species_keys = _workspace_species_keys_from_raw_events(raw_events)
    workspace_colour_map = assign_species_colours(list(workspace_species_keys))

    continuity_batches = build_review_continuity_batches(raw_events)
    matching_batch = None
    for batch in continuity_batches:
        if event_key in (batch.get("event_keys") or []):
            matching_batch = batch
            break
    if matching_batch is None:
        _stamp_species_display_on_event(event_payload, workspace_colour_map)
        return event_payload

    # Mirror the rail-side hint fields _load_review_events sets for
    # JS. Templates rely on the nested ``continuity_batch`` dict, but
    # JS handlers also read these flat hints.
    event_payload["continuity_batch_key"] = matching_batch["batch_key"]
    event_payload["continuity_batch_recommended_species"] = matching_batch[
        "recommended_species"
    ]
    event_payload["continuity_batch_review_detection_ids"] = list(
        matching_batch["review_detection_ids"]
    )

    def _materialize(detection_ids):
        members: list[dict] = []
        for detection_id in detection_ids:
            row = row_map.get(int(detection_id))
            if not row:
                continue
            members.append(
                _build_review_event_member(
                    row,
                    conn=conn,
                    species_locale=species_locale,
                    output_dir=output_dir,
                    common_names=common_names,
                    recent_species=recent_species,
                )
            )
        members.sort(
            key=lambda member: (
                member.get("timestamp", ""),
                int(member.get("best_detection_id") or 0),
            )
        )
        return members

    review_members = _materialize(matching_batch.get("review_detection_ids") or [])
    anchor_members = _materialize(matching_batch.get("context_detection_ids") or [])

    # Tag every review member with the source event_key so the template
    # can highlight frames belonging to the currently rail-selected
    # event without losing the cross-event view.
    review_event_keys = list(matching_batch.get("event_keys") or [])
    detection_to_event: dict[int, str] = {}
    for raw in raw_events:
        if raw.event_key not in review_event_keys:
            continue
        for trail_point in raw.bbox_trail:
            if trail_point.get("context_only"):
                continue
            try:
                det_id = int(trail_point.get("detection_id") or 0)
            except (TypeError, ValueError):
                continue
            if det_id > 0:
                detection_to_event[det_id] = raw.event_key
    for member in review_members:
        det_id = int(member.get("best_detection_id") or 0)
        member_event_key = detection_to_event.get(det_id, "")
        member["batch_event_key"] = member_event_key
        member["is_active_event_member"] = member_event_key == event_key

    recommended_species = matching_batch.get("recommended_species")
    recommended_species_common = (
        resolve_common_name(recommended_species, common_names)
        if recommended_species
        else None
    )

    event_payload["continuity_batch"] = {
        "batch_key": matching_batch["batch_key"],
        "recommended_species": recommended_species,
        "recommended_species_common": recommended_species_common,
        "review_detection_ids": list(matching_batch.get("review_detection_ids") or []),
        "context_detection_ids": list(matching_batch.get("context_detection_ids") or []),
        "context_species_summary": dict(matching_batch.get("context_species_summary") or {}),
        "anchor_members": anchor_members,
        "review_members": review_members,
        "review_event_keys": review_event_keys,
        "anchor_event_keys": list(matching_batch.get("anchor_event_keys") or []),
        "batch_bbox_map": list(matching_batch.get("batch_bbox_map") or []),
    }
    # Stamp the event payload with the shared workspace colour map so
    # anchor and review batch members keep the same slots as the rail.
    _stamp_species_display_on_event(event_payload, workspace_colour_map)
    return event_payload


def _workspace_species_keys_from_raw_events(raw_events) -> set[str]:
    """Collect every scientific name across a raw BirdEvent list.

    Used by the fragment endpoint so its standalone render of one event
    still matches the page-level colour assignment for overlapping
    species. Pulls from ``raw_event.species`` plus every trail member
    (context + actionable) so context anchors participate in colour
    assignment exactly like actionable frames.
    """
    species_keys: set[str] = set()
    for raw_event in raw_events or []:
        species = getattr(raw_event, "species", None)
        if species:
            species_keys.add(str(species).strip())
        for trail_point in getattr(raw_event, "bbox_trail", None) or []:
            trail_species = trail_point.get("species") or trail_point.get(
                "species_key"
            )
            if trail_species:
                species_keys.add(str(trail_species).strip())
    species_keys.discard("")
    return species_keys


def _refresh_review_image_visibility(conn, filename: str, gallery_threshold: float) -> str:
    active_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM detections d
        WHERE d.image_filename = ?
          AND COALESCE(d.status, 'active') = 'active'
        """,
        (filename,),
    ).fetchone()[0]

    if active_count == 0:
        db_service.update_review_status(conn, [filename], REVIEW_STATUS_NO_BIRD)
        return REVIEW_STATUS_NO_BIRD

    unresolved = conn.execute(
        """
        SELECT COUNT(*)
        FROM detections d
        WHERE d.image_filename = ?
          AND COALESCE(d.status, 'active') = 'active'
          AND COALESCE(d.decision_state, '') NOT IN ('confirmed', 'rejected')
          AND (
              COALESCE(d.score, 0.0) < ?
              OR d.decision_state IN ('uncertain', 'unknown')
          )
        """,
        (filename, gallery_threshold),
    ).fetchone()[0]

    if unresolved == 0:
        db_service.update_review_status(conn, [filename], REVIEW_STATUS_CONFIRMED_BIRD)
        return REVIEW_STATUS_CONFIRMED_BIRD

    conn.execute(
        """
        UPDATE images
        SET review_status = 'untagged'
        WHERE filename = ?
        """,
        (filename,),
    )
    conn.commit()
    return REVIEW_STATUS_UNTAGGED


def _compute_queue_orphans(
    orphans: list[dict],
    review_events: list[dict],
) -> list[dict]:
    """Filter the orphans list down to items the event rail does not cover.

    Orphan-image rows (``item_kind == "image"``) and any leftovers the
    event builder could not group still need the Queue rail. Detection
    orphans that already belong to a rendered event are removed here so
    the Queue rail and Event rail never duplicate the same detection.
    """
    event_detection_ids: set[int] = set()
    for event_payload in review_events:
        for detection_id in event_payload.get("detection_ids") or []:
            try:
                event_detection_ids.add(int(detection_id))
            except (TypeError, ValueError):
                continue

    queue_orphans: list[dict] = []
    for orphan in orphans:
        if orphan.get("item_kind") == "detection":
            try:
                detection_id = int(orphan.get("item_id") or 0)
            except (TypeError, ValueError):
                detection_id = 0
            if detection_id and detection_id in event_detection_ids:
                continue
        queue_orphans.append(orphan)
    return queue_orphans


def _require_active_review_detection(conn, filename: str, detection_id: int):
    return conn.execute(
        """
        SELECT detection_id, image_filename
        FROM detections
        WHERE detection_id = ?
          AND image_filename = ?
          AND COALESCE(status, 'active') = 'active'
        LIMIT 1
        """,
        (detection_id, filename),
    ).fetchone()


@review_bp.route("/admin/review", methods=["GET"])
@login_required
def review_page():
    """
    Review Queue: Images needing user decision.
    Shows orphans (no detections) AND low-confidence detections.
    Sorted oldest first.
    """
    output_dir = config.get("OUTPUT_DIR", "output")
    gallery_threshold = config["GALLERY_DISPLAY_THRESHOLD"]
    species_locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    common_names = load_common_names(species_locale)

    with db_service.closing_connection() as conn:
        orphans, _ = _load_review_items(
            conn,
            gallery_threshold=gallery_threshold,
            output_dir=output_dir,
            species_locale=species_locale,
            common_names=common_names,
        )
        (
            review_events,
            continuity_batches,
            context_truncated,
            workspace_species_keys,
        ) = _load_review_events(
            conn,
            gallery_threshold=gallery_threshold,
            output_dir=output_dir,
            species_locale=species_locale,
            common_names=common_names,
        )

    queue_orphans = _compute_queue_orphans(orphans, review_events)

    # Stamp orphans with the same workspace-scoped colour slots already
    # applied to events by ``_load_review_events``. The map is rebuilt
    # from the union of workspace species and orphan species so
    # additional scientific names still get deterministic slots.
    all_species_keys = set(workspace_species_keys)
    for orphan in orphans:
        key = _resolve_species_display_key(orphan)
        if key:
            all_species_keys.add(key)
    species_colour_map = assign_species_colours(list(all_species_keys))
    # Re-stamp events so the slot assignment matches the combined map
    # whenever an orphan introduces a new scientific name that would
    # otherwise shift slots for species sorted after it.
    for event in review_events:
        _stamp_species_display_on_event(event, species_colour_map)
    for orphan in orphans:
        key = _resolve_species_display_key(orphan)
        if key:
            orphan["species_colour_key"] = key
            orphan["species_colour"] = species_colour_map.get(key)
            orphan["species_ref_image_url"] = resolve_species_ref_image_url(key)
        else:
            orphan["species_colour_key"] = ""
            orphan["species_colour"] = None
            orphan["species_ref_image_url"] = None
        for picker in orphan.get("quick_species") or []:
            pkey = str(picker.get("scientific") or "").strip()
            picker["species_colour"] = (
                species_colour_map.get(pkey) if pkey else None
            )

    return render_template(
        "orphans.html",
        orphans=orphans,
        queue_orphans=queue_orphans,
        review_events=review_events,
        continuity_batches=continuity_batches,
        context_truncated=context_truncated,
        species_colour_map=species_colour_map,
        species_colour_slots=SPECIES_COLOUR_SLOTS,
        current_path="/admin/review",
    )


@review_bp.route("/api/review/panel/<item_kind>/<item_id>", methods=["GET"])
@login_required
def review_panel_fragment(item_kind, item_id):
    """Render a single review stage panel on demand."""
    output_dir = config.get("OUTPUT_DIR", "output")
    gallery_threshold = config["GALLERY_DISPLAY_THRESHOLD"]
    species_locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    common_names = load_common_names(species_locale)

    with db_service.closing_connection() as conn:
        recent_species = _load_recent_review_species(conn, common_names)
        species_thumbnail_map = get_species_thumbnail_map(
            common_names=common_names,
            cache_key=f"review:{species_locale}",
        )
        orphan = _load_single_review_item_by_identity(
            conn,
            item_kind=item_kind,
            item_id=item_id,
            gallery_threshold=gallery_threshold,
            output_dir=output_dir,
            species_locale=species_locale,
            common_names=common_names,
            recent_species=recent_species,
            species_thumbnail_map=species_thumbnail_map,
        )

    if not orphan:
        abort(404)

    # A single-orphan fragment only needs a one-species colour map.
    key = _resolve_species_display_key(orphan)
    single_map = assign_species_colours([key] if key else [])
    if key:
        orphan["species_colour_key"] = key
        orphan["species_colour"] = single_map.get(key)
        orphan["species_ref_image_url"] = resolve_species_ref_image_url(key)
    else:
        orphan["species_colour_key"] = ""
        orphan["species_colour"] = None
        orphan["species_ref_image_url"] = None
    for picker in orphan.get("quick_species") or []:
        pkey = str(picker.get("scientific") or "").strip()
        picker["species_colour"] = single_map.get(pkey) if pkey else None
    return render_template("components/review_stage_panel.html", orphan=orphan)


@review_bp.route("/api/review/event-panel/<event_key>", methods=["GET"])
@login_required
def review_event_panel_fragment(event_key):
    """Render a single review event detail panel on demand."""
    output_dir = config.get("OUTPUT_DIR", "output")
    gallery_threshold = config["GALLERY_DISPLAY_THRESHOLD"]
    species_locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    common_names = load_common_names(species_locale)

    with db_service.closing_connection() as conn:
        event = _load_event_with_continuity_batch(
            conn,
            event_key=event_key,
            gallery_threshold=gallery_threshold,
            output_dir=output_dir,
            species_locale=species_locale,
            common_names=common_names,
        )

    if not event:
        abort(404)

    # Stamping already happened inside _load_event_with_continuity_batch
    # using a workspace-scoped colour map built from the full raw_events
    # set, so anchor + review members match what the page rail renders
    # for the same scientific names.
    return render_template("components/review_event_panel.html", event=event)


@review_bp.route("/api/review-thumb/<filename>", methods=["GET"])
@login_required
def review_thumb(filename):
    """On-demand thumbnail generation for orphan images."""
    output_dir = config.get("OUTPUT_DIR", "output")
    paths = gallery_service.get_image_paths(output_dir, filename)

    original_path = paths["original"]
    preview_path = paths["preview"]

    # If preview already cached, serve it
    if preview_path.exists():
        return send_file(str(preview_path), mimetype="image/webp")

    # Original must exist to generate preview
    if not original_path.exists():
        abort(404)

    # Generate preview thumbnail via service
    success = gallery_service.generate_preview_thumbnail(
        original_path, preview_path, size=256
    )

    if success and preview_path.exists():
        return send_file(str(preview_path), mimetype="image/webp")
    else:
        abort(500)


@review_bp.route("/api/review/decision", methods=["POST"])
@login_required
def review_decision():
    """
    API endpoint for Review Queue decisions.
    POST /api/review/decision
    Payload: { filenames: [...], action: "confirm" | "trash" | "no_bird" | "skip" }

    - confirm -> review_status = 'confirmed_bird'
    - trash/no_bird -> review_status = 'no_bird' (soft-trash, no file deletion)
    - skip -> no change

    Only updates images with review_status = 'untagged' (no way back).
    """
    try:
        data = request.get_json() or {}
        filenames = data.get("filenames", [])
        item_kind = str(data.get("item_kind") or "").strip()
        item_id = str(data.get("item_id") or "").strip()
        action = data.get("action", "")

        if not filenames and item_kind == "image" and item_id:
            filenames = [item_id]

        if not filenames:
            return (
                jsonify({"status": "error", "message": "No filenames provided"}),
                400,
            )

        if action not in ("confirm", "trash", "no_bird", "skip"):
            return (
                jsonify({"status": "error", "message": f"Invalid action: {action}"}),
                400,
            )

        # Skip action: no database change
        if action == "skip":
            return jsonify({"status": "success", "updated": 0, "action": "skip"})

        requested_action = action

        # Map action to review_status
        status_map = {
            "confirm": REVIEW_STATUS_CONFIRMED_BIRD,
            "trash": REVIEW_STATUS_NO_BIRD,
            "no_bird": REVIEW_STATUS_NO_BIRD,
        }
        new_status = status_map[action]

        conn = db_service.get_connection()
        try:
            if action == "confirm":
                # Confirming "Bird Present" requires an existing detection.
                # Otherwise the image becomes non-eligible for Deep Scan and can get stuck.
                placeholders = ",".join("?" for _ in filenames)
                rows = conn.execute(
                    f"""
                    SELECT
                        i.filename,
                        EXISTS(
                            SELECT 1
                            FROM detections d
                            WHERE d.image_filename = i.filename
                        ) AS has_detections
                    FROM images i
                    WHERE i.filename IN ({placeholders})
                    """,
                    filenames,
                ).fetchall()

                missing = [row["filename"] for row in rows if not row["has_detections"]]
                if missing:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Cannot confirm Bird Present for items without detections. Use Deep Scan first.",
                                "filenames": missing,
                            }
                        ),
                        409,
                    )

            updated = db_service.update_review_status(conn, filenames, new_status)
        finally:
            conn.close()

        logger.info(
            f"Review decision: {requested_action} -> {new_status} ({updated} images updated)"
        )
        return jsonify(
            {
                "status": "success",
                "updated": updated,
                "action": requested_action,
                "review_status": new_status,
            }
        )

    except Exception as e:
        logger.error(f"Error in review decision: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/bbox-review", methods=["POST"])
@login_required
def update_bbox_review_state():
    """Persist the manual bbox review state for a review item."""
    data = request.get_json() or {}
    filename = str(data.get("filename") or "").strip()
    bbox_review = (data.get("bbox_review") or "").strip().lower() or None

    try:
        detection_id = int(data.get("detection_id") or 0)
    except (TypeError, ValueError):
        detection_id = 0

    if not filename or detection_id <= 0:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "filename and detection_id are required",
                }
            ),
            400,
        )

    if bbox_review not in (None, *VALID_BBOX_REVIEW_STATES):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Invalid bbox_review: {bbox_review}",
                }
            ),
            400,
        )

    try:
        with db_service.closing_connection() as conn:
            row = _require_active_review_detection(conn, filename, detection_id)
            if not row:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Detection not found for review item",
                        }
                    ),
                    404,
                )
            db_service.set_manual_bbox_review(conn, detection_id, bbox_review)

        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "detection_id": detection_id,
                "bbox_review": bbox_review,
            }
        )
    except Exception as e:
        logger.error(f"Error updating bbox review state: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/quick-species", methods=["POST"])
@login_required
def review_quick_species():
    """Apply a quick species choice without final gallery approval."""
    data = request.get_json() or {}
    filename = str(data.get("filename") or "").strip()
    species = str(data.get("species") or "").strip()
    bbox_review = (data.get("bbox_review") or "").strip().lower() or None

    try:
        detection_id = int(data.get("detection_id") or 0)
    except (TypeError, ValueError):
        detection_id = 0

    if not filename or detection_id <= 0 or not species:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "filename, detection_id and species are required",
                }
            ),
            400,
        )

    if bbox_review not in (None, *VALID_BBOX_REVIEW_STATES):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Invalid bbox_review: {bbox_review}",
                }
            ),
            400,
        )

    try:
        with db_service.closing_connection() as conn:
            row = _require_active_review_detection(conn, filename, detection_id)
            if not row:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Detection not found for review item",
                        }
                    ),
                    404,
                )

            locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
            allowed_species = _get_allowed_review_species(
                conn,
                locale,
                detection_id=detection_id,
            )
            if species not in allowed_species:
                return (
                    jsonify({"status": "error", "message": "unknown species"}),
                    400,
                )

            db_service.apply_species_override(conn, detection_id, species, "manual")
            if bbox_review is not None:
                db_service.set_manual_bbox_review(conn, detection_id, bbox_review)

        gallery_service.invalidate_cache()

        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "detection_id": detection_id,
                "new_species": species,
                "bbox_review": bbox_review,
            }
        )
    except Exception as e:
        logger.error(f"Error applying review quick species: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/approve", methods=["POST"])
@login_required
def review_approve():
    """Approve a fully reviewed image for gallery visibility."""
    data = request.get_json() or {}
    filename = str(data.get("filename") or "").strip()
    species = str(data.get("species") or "").strip()
    bbox_review = (data.get("bbox_review") or "").strip().lower() or None

    try:
        detection_id = int(data.get("detection_id") or 0)
    except (TypeError, ValueError):
        detection_id = 0

    if not filename or detection_id <= 0:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "filename and detection_id are required",
                }
            ),
            400,
        )

    if not species:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "A species selection is required before approval",
                }
            ),
            409,
        )

    if bbox_review not in VALID_BBOX_REVIEW_STATES:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Bounding box review is required before approval",
                }
            ),
            409,
        )

    try:
        with db_service.closing_connection() as conn:
            locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
            allowed_species = _get_allowed_review_species(
                conn,
                locale,
                detection_id=detection_id,
            )
            if species not in allowed_species:
                return (
                    jsonify({"status": "error", "message": "unknown species"}),
                    400,
                )

            row = conn.execute(
                """
                SELECT
                    d.manual_bbox_review,
                    d.manual_species_override,
                    d.species_source
                FROM detections d
                WHERE d.detection_id = ?
                  AND d.image_filename = ?
                  AND COALESCE(d.status, 'active') = 'active'
                LIMIT 1
                """,
                (detection_id, filename),
            ).fetchone()
            if not row:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "Detection not found for review item",
                        }
                    ),
                    404,
                )

            if (
                row["manual_species_override"] != species
                or row["species_source"] != "manual"
            ):
                db_service.apply_species_override(conn, detection_id, species, "manual")

            if row["manual_bbox_review"] != bbox_review:
                db_service.set_manual_bbox_review(conn, detection_id, bbox_review)

            conn.execute(
                """
                UPDATE detections
                SET decision_state = 'confirmed'
                WHERE detection_id = ?
                  AND image_filename = ?
                  AND COALESCE(status, 'active') = 'active'
                """,
                (detection_id, filename),
            )
            image_review_status = _refresh_review_image_visibility(
                conn,
                filename,
                config["GALLERY_DISPLAY_THRESHOLD"],
            )

        gallery_service.invalidate_cache()
        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "detection_id": detection_id,
                "review_status": image_review_status,
                "gallery_visible": image_review_status == REVIEW_STATUS_CONFIRMED_BIRD,
                "message": (
                    "Detection approved and image is now visible in the gallery."
                    if image_review_status == REVIEW_STATUS_CONFIRMED_BIRD
                    else "Detection approved, but the image remains out of the gallery until all open detections on the same photo are resolved."
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error approving review item: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/event-approve", methods=["POST"])
@login_required
def review_event_approve():
    """Approve a single biological event in one step.

    Re-resolves the event from the live detection rows so that any
    stale ``event_key`` from a client tab fails closed instead of
    confirming the wrong detections.
    """
    data = request.get_json() or {}
    species = str(data.get("species") or "").strip()
    bbox_review = (data.get("bbox_review") or "").strip().lower() or None
    event_key = str(data.get("event_key") or "").strip()
    raw_detection_ids = data.get("detection_ids") or []

    detection_ids: list[int] = []
    for raw_detection_id in raw_detection_ids:
        try:
            detection_id = int(raw_detection_id)
        except (TypeError, ValueError):
            continue
        if detection_id > 0:
            detection_ids.append(detection_id)

    detection_ids = list(dict.fromkeys(detection_ids))

    if not detection_ids:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "detection_ids are required",
                }
            ),
            400,
        )

    if not species:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "A species selection is required before event approval",
                }
            ),
            409,
        )

    if bbox_review not in VALID_BBOX_REVIEW_STATES:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Bounding box review is required before event approval",
                }
            ),
            409,
        )

    try:
        with db_service.closing_connection() as conn:
            locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
            allowed_species = _get_allowed_review_species(conn, locale)
            if species not in allowed_species:
                return (
                    jsonify({"status": "error", "message": "unknown species"}),
                    400,
                )

            output_dir = config.get("OUTPUT_DIR", "output")
            common_names = load_common_names(locale)
            if event_key:
                event = _load_single_review_event(
                    conn,
                    event_key=event_key,
                    gallery_threshold=config["GALLERY_DISPLAY_THRESHOLD"],
                    output_dir=output_dir,
                    species_locale=locale,
                    common_names=common_names,
                )
                if not event:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event no longer exists",
                            }
                        ),
                        409,
                    )
                if event.get("eligibility") != "event_eligible":
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event requires individual review",
                            }
                        ),
                        409,
                    )
                event_detection_ids = sorted(event.get("detection_ids") or [])
                actionable_event_detection_ids = sorted(
                    int(member.get("best_detection_id") or 0)
                    for member in (event.get("members") or [])
                    if not member.get("context_only")
                    and int(member.get("best_detection_id") or 0) > 0
                )

                # Gallery anchors can stay visible in the event panel as
                # read-only context, but the approval write path must only
                # touch actionable detections. Accept both payload shapes:
                # the new client sends only actionable ids; older tabs may
                # still submit the full event id list.
                submitted_detection_ids = sorted(detection_ids)
                if (
                    actionable_event_detection_ids
                    and submitted_detection_ids == event_detection_ids
                    and submitted_detection_ids != actionable_event_detection_ids
                ):
                    detection_ids = actionable_event_detection_ids
                elif actionable_event_detection_ids:
                    if submitted_detection_ids != actionable_event_detection_ids:
                        return (
                            jsonify(
                                {
                                    "status": "error",
                                    "message": "Review event changed and must be reloaded",
                                }
                            ),
                            409,
                        )
                    detection_ids = actionable_event_detection_ids
                elif submitted_detection_ids != event_detection_ids:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event changed and must be reloaded",
                            }
                        ),
                        409,
                    )

                if not detection_ids:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event has no actionable detections",
                            }
                        ),
                        409,
                    )

            placeholders = ",".join("?" for _ in detection_ids)
            rows = conn.execute(
                f"""
                SELECT d.detection_id, d.image_filename, i.review_status
                FROM detections d
                JOIN images i ON i.filename = d.image_filename
                WHERE d.detection_id IN ({placeholders})
                  AND COALESCE(d.status, 'active') = 'active'
                """,
                detection_ids,
            ).fetchall()
            if len(rows) != len(detection_ids):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "One or more detections are no longer available",
                        }
                    ),
                    409,
                )

            # Continuity-batch approval can submit detection ids that
            # span sibling events, so the strict
            # ``event_key``/``detection_ids`` parity check is bypassed.
            # Read-only Gallery anchors must never be re-confirmed
            # through this path.
            anchor_filenames = sorted(
                {
                    row["image_filename"]
                    for row in rows
                    if row["review_status"] == REVIEW_STATUS_CONFIRMED_BIRD
                }
            )
            if anchor_filenames:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": (
                                "Refusing to re-confirm Gallery anchors via review approval"
                            ),
                            "anchor_filenames": anchor_filenames,
                        }
                    ),
                    409,
                )

            db_service.apply_species_override_many(conn, detection_ids, species, "manual")
            for detection_id in detection_ids:
                db_service.set_manual_bbox_review(conn, detection_id, bbox_review)

            conn.execute(
                f"""
                UPDATE detections
                SET decision_state = 'confirmed'
                WHERE detection_id IN ({placeholders})
                  AND COALESCE(status, 'active') = 'active'
                """,
                detection_ids,
            )
            conn.commit()

            touched_filenames = list(
                dict.fromkeys(row["image_filename"] for row in rows if row["image_filename"])
            )
            review_status_by_filename = {
                filename: _refresh_review_image_visibility(
                    conn,
                    filename,
                    config["GALLERY_DISPLAY_THRESHOLD"],
                )
                for filename in touched_filenames
            }

        gallery_service.invalidate_cache()
        gallery_visible_filenames = [
            filename
            for filename, review_status in review_status_by_filename.items()
            if review_status == REVIEW_STATUS_CONFIRMED_BIRD
        ]
        return jsonify(
            {
                "status": "success",
                "event_key": event_key or None,
                "detection_ids": detection_ids,
                "filenames": touched_filenames,
                "review_status_by_filename": review_status_by_filename,
                "gallery_visible_filenames": gallery_visible_filenames,
                "message": (
                    "Event approved and every touched image is now visible in the gallery."
                    if touched_filenames
                    and len(gallery_visible_filenames) == len(touched_filenames)
                    else "Event approved. Some images remain out of the gallery until their remaining open detections are resolved."
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error approving review event: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/event-trash", methods=["POST"])
@login_required
def review_event_trash():
    """Reject every active detection in one BirdEvent.

    Semantic target: reject the event's detections, not the image files.
    Image visibility is recomputed afterwards:
    - if an image has no active detections left, it becomes `no_bird`
      and surfaces in Trash
    - otherwise it follows the state of its remaining active detections
      (`untagged` or `confirmed_bird`)
    """
    data = request.get_json() or {}
    event_key = str(data.get("event_key") or "").strip()
    raw_detection_ids = data.get("detection_ids") or []

    detection_ids: list[int] = []
    for raw_detection_id in raw_detection_ids:
        try:
            detection_id = int(raw_detection_id)
        except (TypeError, ValueError):
            continue
        if detection_id > 0:
            detection_ids.append(detection_id)

    detection_ids = list(dict.fromkeys(detection_ids))

    if not detection_ids:
        return (
            jsonify({"status": "error", "message": "detection_ids are required"}),
            400,
        )

    try:
        with db_service.closing_connection() as conn:
            locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
            output_dir = config.get("OUTPUT_DIR", "output")
            common_names = load_common_names(locale)

            if event_key:
                event = _load_single_review_event(
                    conn,
                    event_key=event_key,
                    gallery_threshold=config["GALLERY_DISPLAY_THRESHOLD"],
                    output_dir=output_dir,
                    species_locale=locale,
                    common_names=common_names,
                )
                if not event:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event no longer exists",
                            }
                        ),
                        409,
                    )
                if sorted(event.get("detection_ids") or []) != sorted(detection_ids):
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event changed and must be reloaded",
                            }
                        ),
                        409,
                    )

            placeholders = ",".join("?" for _ in detection_ids)
            rows = conn.execute(
                f"""
                SELECT d.detection_id, d.image_filename
                FROM detections d
                WHERE d.detection_id IN ({placeholders})
                  AND COALESCE(d.status, 'active') = 'active'
                """,
                detection_ids,
            ).fetchall()
            if len(rows) != len(detection_ids):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "One or more detections are no longer available",
                        }
                    ),
                    409,
                )

            db_service.reject_detections(conn, detection_ids)

            touched_filenames = list(
                dict.fromkeys(
                    row["image_filename"] for row in rows if row["image_filename"]
                )
            )
            review_status_by_filename = {
                filename: _refresh_review_image_visibility(
                    conn,
                    filename,
                    config["GALLERY_DISPLAY_THRESHOLD"],
                )
                for filename in touched_filenames
            }

        gallery_service.invalidate_cache()

        trash_filenames = [
            filename
            for filename, review_status in review_status_by_filename.items()
            if review_status == REVIEW_STATUS_NO_BIRD
        ]
        gallery_visible_filenames = [
            filename
            for filename, review_status in review_status_by_filename.items()
            if review_status == REVIEW_STATUS_CONFIRMED_BIRD
        ]
        review_filenames = [
            filename
            for filename, review_status in review_status_by_filename.items()
            if review_status == REVIEW_STATUS_UNTAGGED
        ]

        if touched_filenames and len(trash_filenames) == len(touched_filenames):
            message = (
                "Event moved to Trash. Every touched image now has no active detections left."
            )
        else:
            message = (
                "Event detections rejected. Images with no active detections left moved to Trash; other touched images now follow their remaining active detections."
            )

        return jsonify(
            {
                "status": "success",
                "event_key": event_key or None,
                "detection_ids": detection_ids,
                "filenames": touched_filenames,
                "review_status_by_filename": review_status_by_filename,
                "trash_filenames": trash_filenames,
                "gallery_visible_filenames": gallery_visible_filenames,
                "review_filenames": review_filenames,
                "message": message,
            }
        )
    except Exception as e:
        logger.error(f"Error trashing review event: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/event-resolve", methods=["POST"])
@login_required
def review_event_resolve():
    """Resolve a mixed BirdEvent in one transaction.

    Accepts disjoint ``keep_detection_ids`` + ``trash_detection_ids`` lists
    that together must cover every active detection of the event. The
    ``species`` + ``bbox_review`` selection applies only to the Keep
    frames; Trash frames are rejected via the same code path as
    ``/api/review/event-trash`` so image visibility is recomputed
    consistently (images with no remaining active detections fall to
    Trash; other images follow their remaining actives).

    ``keep_detection_ids`` must not be empty: use the existing
    ``/api/review/event-trash`` shortcut for homogeneous all-wrong events.
    """
    data = request.get_json() or {}
    species = str(data.get("species") or "").strip()
    bbox_review = (data.get("bbox_review") or "").strip().lower() or None
    event_key = str(data.get("event_key") or "").strip()
    raw_keep = data.get("keep_detection_ids") or []
    raw_trash = data.get("trash_detection_ids") or []

    def _coerce_ids(raw):
        cleaned: list[int] = []
        for value in raw:
            try:
                detection_id = int(value)
            except (TypeError, ValueError):
                continue
            if detection_id > 0:
                cleaned.append(detection_id)
        return list(dict.fromkeys(cleaned))

    keep_ids = _coerce_ids(raw_keep)
    trash_ids = _coerce_ids(raw_trash)

    if not keep_ids:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": (
                        "keep_detection_ids must not be empty — use"
                        " /api/review/event-trash for all-wrong events"
                    ),
                }
            ),
            400,
        )

    overlap = set(keep_ids) & set(trash_ids)
    if overlap:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "keep and trash detection id sets must be disjoint",
                    "overlap": sorted(overlap),
                }
            ),
            400,
        )

    if not species:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "A species selection is required for kept frames",
                }
            ),
            409,
        )

    if bbox_review not in VALID_BBOX_REVIEW_STATES:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Bounding box review is required before event resolve",
                }
            ),
            409,
        )

    all_ids = keep_ids + trash_ids

    try:
        with db_service.closing_connection() as conn:
            locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
            allowed_species = _get_allowed_review_species(conn, locale)
            if species not in allowed_species:
                return (
                    jsonify({"status": "error", "message": "unknown species"}),
                    400,
                )

            output_dir = config.get("OUTPUT_DIR", "output")
            common_names = load_common_names(locale)

            if event_key:
                event = _load_single_review_event(
                    conn,
                    event_key=event_key,
                    gallery_threshold=config["GALLERY_DISPLAY_THRESHOLD"],
                    output_dir=output_dir,
                    species_locale=locale,
                    common_names=common_names,
                )
                if not event:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "Review event no longer exists",
                            }
                        ),
                        409,
                    )
                event_ids = sorted(event.get("detection_ids") or [])
                if event_ids != sorted(all_ids):
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": (
                                    "Review event changed and must be reloaded;"
                                    " keep+trash must cover every event detection"
                                ),
                            }
                        ),
                        409,
                    )

            placeholders = ",".join("?" for _ in all_ids)
            rows = conn.execute(
                f"""
                SELECT d.detection_id,
                       d.image_filename,
                       d.manual_species_override,
                       d.species_source,
                       i.review_status
                FROM detections d
                JOIN images i ON i.filename = d.image_filename
                WHERE d.detection_id IN ({placeholders})
                  AND COALESCE(d.status, 'active') = 'active'
                """,
                all_ids,
            ).fetchall()
            if len(rows) != len(all_ids):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "One or more detections are no longer available",
                        }
                    ),
                    409,
                )

            # Refuse to touch Gallery anchors, mirroring event-approve.
            anchor_filenames = sorted(
                {
                    row["image_filename"]
                    for row in rows
                    if row["review_status"] == REVIEW_STATUS_CONFIRMED_BIRD
                }
            )
            if anchor_filenames:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": (
                                "Refusing to re-confirm Gallery anchors via event resolve"
                            ),
                            "anchor_filenames": anchor_filenames,
                        }
                    ),
                    409,
                )

            # Per-frame relabel beats event-level species.
            # If a Keep frame already carries a manual species override
            # (set via the per-cell relabel path through
            # /api/moderation/bulk/relabel), do NOT overwrite it with
            # the event species picked in the right control rail.
            # This is the `per-frame wins` rule of the V1 mixed-species
            # flow — an operator who hand-corrected a single frame
            # must not lose that correction to a coarse event action.
            keep_id_set = set(keep_ids)
            manual_override_ids: set[int] = set()
            for row in rows:
                row_id = int(row["detection_id"])
                if row_id not in keep_id_set:
                    continue
                override = str(row["manual_species_override"] or "").strip()
                source = str(row["species_source"] or "").strip()
                if override and source == "manual":
                    manual_override_ids.add(row_id)

            keep_ids_to_stamp = [
                detection_id
                for detection_id in keep_ids
                if detection_id not in manual_override_ids
            ]

            # Keep frames without a per-cell override: apply event species
            # + bbox review. Frames with a manual override keep their
            # override but still get the bbox review state from the rail.
            if keep_ids_to_stamp:
                db_service.apply_species_override_many(
                    conn, keep_ids_to_stamp, species, "manual"
                )
            for detection_id in keep_ids:
                db_service.set_manual_bbox_review(conn, detection_id, bbox_review)
            keep_placeholders = ",".join("?" for _ in keep_ids)
            conn.execute(
                f"""
                UPDATE detections
                SET decision_state = 'confirmed'
                WHERE detection_id IN ({keep_placeholders})
                  AND COALESCE(status, 'active') = 'active'
                """,
                keep_ids,
            )

            # Trash frames: reject in the same transaction.
            if trash_ids:
                db_service.reject_detections(conn, trash_ids)

            conn.commit()

            touched_filenames = list(
                dict.fromkeys(
                    row["image_filename"] for row in rows if row["image_filename"]
                )
            )
            review_status_by_filename = {
                filename: _refresh_review_image_visibility(
                    conn,
                    filename,
                    config["GALLERY_DISPLAY_THRESHOLD"],
                )
                for filename in touched_filenames
            }

        gallery_service.invalidate_cache()

        gallery_visible_filenames = [
            filename
            for filename, review_status in review_status_by_filename.items()
            if review_status == REVIEW_STATUS_CONFIRMED_BIRD
        ]
        trash_filenames = [
            filename
            for filename, review_status in review_status_by_filename.items()
            if review_status == REVIEW_STATUS_NO_BIRD
        ]

        return jsonify(
            {
                "status": "success",
                "event_key": event_key or None,
                "keep_detection_ids": keep_ids,
                "trash_detection_ids": trash_ids,
                "filenames": touched_filenames,
                "review_status_by_filename": review_status_by_filename,
                "gallery_visible_filenames": gallery_visible_filenames,
                "trash_filenames": trash_filenames,
                "message": (
                    "Event resolved — "
                    f"{len(keep_ids)} frame{'s' if len(keep_ids) != 1 else ''} kept, "
                    f"{len(trash_ids)} frame{'s' if len(trash_ids) != 1 else ''} trashed."
                ),
            }
        )
    except Exception as e:
        logger.error(f"Error resolving review event: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@review_bp.route("/api/review/analyze/<filename>", methods=["POST"])
@login_required
def analyze_review_item(filename):
    """
    Triggers a manual deep analysis for exactly one review item.
    Query params:
      force=1  — bypass no-hit DB exclusion (re-scan already-scanned images)
    """
    try:
        from web.services.analysis_service import (
            check_deep_analysis_eligibility,
            submit_analysis_job,
        )

        force = request.args.get("force", "0") in ("1", "true", "yes")

        is_eligible, reason = check_deep_analysis_eligibility(filename, force=force)
        if not is_eligible:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": reason
                        or "Manual deep scan is only available for unreviewed items without detections.",
                    }
                ),
                409,
            )

        if submit_analysis_job(filename, force=force):
            return jsonify(
                {
                    "status": "success",
                    "message": "Manual deep scan queued for this image.",
                }
            )

        return jsonify({"status": "error", "message": "Failed to queue analysis"}), 500

    except Exception as e:
        logger.error(f"Error triggering analysis: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
