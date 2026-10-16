"""
Review Blueprint.

Handles review queue routes:
- GET /admin/review - Review queue page (was orphans)
- GET /api/review-thumb/<filename> - On-demand thumbnail
- POST /api/review/decision - Review decisions (confirm/trash/no_bird/skip)
- POST /api/review/bbox-review - Persist bbox review state
- POST /api/review/quick-species - Confirm + relabel via quick species buttons
- POST /api/review/event-approve - Approve a whole BirdEvent at once
- POST /api/review/event-trash - Reject / trash a whole BirdEvent at once
"""

import re
import time
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

from config import get_config
from core.events import build_bird_events
from core.human_label_core import HumanAnswer
from core.review_state import (
    refresh_review_image_visibility as _refresh_review_image_visibility,
)
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
    is_human_unknown_species_source,
    is_known_species,
    load_common_names,
    resolve_common_name,
)
from web.blueprints.auth import login_required
from web.security import error_response as _error_response
from web.services import (
    cache_service,
    db_service,
    gallery_service,
    human_label_service,
    path_service,
    review_cleanup_service,
)
from web.species_thumbnails import (
    resolve_species_thumbnail_url,
)

logger = get_logger(__name__)
config = get_config()

review_bp = Blueprint("review", __name__)

_REVIEW_ALLOWED_SPECIES_TTL_SECONDS = 60
_review_allowed_species_cache: dict[str, tuple[float, set[str]]] = {}


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

    locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")

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
        # Defence-in-depth: synthetic classifier classes (cls_v20+
        # ``non_bird``) and stale ``manual_species_override`` values
        # from prior accidental relabels must never resurface as a
        # picker suggestion. The /api/review/quick-species validator
        # would reject the click anyway ("unknown species" 400) — we
        # filter here so the operator never sees the bad choice.
        if not is_known_species(scientific_name, locale=locale):
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
        default_entry.get("common")
        or resolve_common_name(scientific_name, common_names),
    )


def _resolve_review_selected_species(
    quick_species: list[dict],
    *,
    manual_species_override: str | None,
    common_names: dict[str, str],
    species_source: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve what the review UI shows - and would submit - as the species.

    An explicit human "species unknown" answer clears
    ``manual_species_override``, which makes it indistinguishable from a row
    nobody has answered yet. Only ``species_source`` separates the two, and
    the difference matters: pre-selecting the classifier's top-1 for a row a
    human just marked unknown both mislabels the header and puts that species
    one Approve click away from becoming a positive label.
    """
    manual_species = str(manual_species_override or "").strip()
    if manual_species:
        return (
            manual_species,
            resolve_common_name(manual_species, common_names),
            "manual",
        )

    if is_human_unknown_species_source(species_source):
        return (
            None,
            resolve_common_name(UNKNOWN_SPECIES_KEY, common_names),
            "manual_unknown",
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


def _resolve_review_member_candidate_species(row: dict) -> str | None:
    """Species a review tile offers for confirmation or relabelling.

    Skips the classifier fallback once a human has explicitly withdrawn the
    species: ``review_grid.js`` treats a pick equal to this value as "no
    change", so the withdrawn species here would block the operator from
    later deciding the bird is that species after all.
    """
    manual = str(row.get("manual_species_override") or "").strip()
    if manual:
        return manual
    if is_human_unknown_species_source(row.get("species_source")):
        return None
    return str(row.get("cls_class_name") or "").strip() or None


def _load_recent_review_species(conn, common_names: dict[str, str]) -> list[dict]:
    rows = db_service.fetch_recent_review_species(conn, limit=8, lookback_days=7)
    locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    recent_species: list[dict] = []
    for row in rows:
        scientific_name = row["species_key"]
        if not scientific_name or scientific_name == UNKNOWN_SPECIES_KEY:
            continue
        # Stale ``non_bird`` (or any non-species token) from a prior
        # mis-relabel must not pollute the "Recent days" pill.
        if not is_known_species(scientific_name, locale=locale):
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
    if review_reason == "classifier_reject":
        return "Classifier: no label"
    if review_reason == "classifier_species_review":
        return "Classifier: low confidence"
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


def _build_review_modal_siblings(
    conn,
    *,
    filename: str,
    common_names: dict[str, str],
    siblings_cache: dict[str, list] | None = None,
) -> list[dict]:
    siblings: list[dict] = []
    # Route the sibling species resolution through the central fallback
    # helper so "bird" does not leak through as species truth when CLS
    # is missing.
    from utils.species_names import UNKNOWN_SPECIES_KEY, species_key_from_candidates

    if siblings_cache is not None:
        sibling_rows = siblings_cache.get(filename, [])
    else:
        sibling_rows = fetch_sibling_detections(conn, filename, include_rejected=True)
    for sibling in sibling_rows:
        resolved_species_key = species_key_from_candidates(
            manual_override=sibling["manual_species_override"],
            species_key=sibling["species_key"],
            cls_class_name=sibling["cls_class_name"],
            od_class_name=sibling["od_class_name"],
            species_source=sibling["species_source"],
        )
        # Preserve historical contract: empty string when species cannot
        # be resolved to a real species (UI templates special-case this).
        species_key = (
            "" if resolved_species_key == UNKNOWN_SPECIES_KEY else resolved_species_key
        )
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
                "human_bbox_verdict": dict(sibling).get("human_bbox_verdict"),
                "object_status": dict(sibling).get("object_status", "active"),
                "needs_review": bool(dict(sibling).get("needs_review", False)),
                "manual_object_id": dict(sibling).get("manual_object_id"),
                "object_key": dict(sibling).get("object_key"),
                "object_kind": dict(sibling).get("object_kind"),
                "revision": dict(sibling).get("revision"),
                "provenance": dict(sibling).get("provenance"),
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
    optimized_url: str = "",
    selected_species: str | None,
    selected_species_common: str | None,
    current_species: str | None,
    current_species_common: str | None,
    common_names: dict[str, str],
    conn,
    siblings: list[dict] | None = None,
    siblings_cache: dict[str, list] | None = None,
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
        or (resolve_common_name(species_key, common_names) if species_key else filename)
    )
    if siblings is None:
        siblings = _build_review_modal_siblings(
            conn,
            filename=filename,
            common_names=common_names,
            siblings_cache=siblings_cache,
        )

    return {
        "detection_id": detection_id,
        "image_filename": filename,
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
            sum(s.get("object_status") != "rejected" for s in siblings),
            int(row["sibling_detection_count"] or 0),
            1,
        ),
        "bbox_x": row["bbox_x"] or 0.0,
        "bbox_y": row["bbox_y"] or 0.0,
        "bbox_w": row["bbox_w"] or 0.0,
        "bbox_h": row["bbox_h"] or 0.0,
        "is_favorite": bool(row["is_favorite"] or 0),
        "is_gallery_eligible": bool(row["is_gallery_eligible"] or 0),
        "decision_state": row["decision_state"],
        "display_path": thumb_url,
        "full_path": full_url or thumb_url,
        "original_path": full_url or thumb_url,
        "optimized_path": optimized_url or thumb_url,
        "original_present": (
            row["original_present"] if "original_present" in row.keys() else 1
        ),
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
    siblings_cache: dict[str, list] | None = None,
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
                species_source=row["species_source"],
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
                species_source=row["species_source"],
            )
        )
    manual_bbox_review = row["manual_bbox_review"]
    selected_bbox_review = (
        manual_bbox_review if manual_bbox_review in VALID_BBOX_REVIEW_STATES else None
    )
    selected_bbox_review_origin = (
        "manual" if manual_bbox_review in VALID_BBOX_REVIEW_STATES else None
    )

    thumb_url = f"/api/review-thumb/{filename}"
    full_url = ""
    optimized_url = ""
    crop_url = thumb_url  # fallback to the 256px /api/review-thumb route
    if len(timestamp) >= 8:
        date_folder_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        full_url = f"/uploads/originals/{date_folder_str}/{filename}"
        optimized_filename = filename.rsplit(".", 1)[0] + ".webp"
        optimized_url = (
            f"/uploads/derivatives/optimized/{date_folder_str}/{optimized_filename}"
        )
        crop_name = row["thumbnail_path"] if "thumbnail_path" in row.keys() else None
        if best_detection_id and crop_name:
            crop_url = (
                f"/uploads/derivatives/thumbs/{date_folder_str}/{Path(crop_name).name}"
            )

    inline_siblings: list[dict] = []
    if include_detail and best_detection_id:
        inline_siblings = _build_review_modal_siblings(
            conn,
            filename=filename,
            common_names=common_names,
            siblings_cache=siblings_cache,
        )

    item = {
        "item_kind": item_kind,
        "item_id": item_id,
        "filename": filename,
        "source_image_filename": row["source_image_filename"] or filename,
        "timestamp": timestamp,
        "gallery_date": f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        if len(timestamp) >= 8
        else None,
        "formatted_date": format_review_timestamp(timestamp),
        "thumb_url": thumb_url,
        "crop_url": crop_url,
        "full_url": full_url,
        "optimized_url": optimized_url,
        "original_present": (
            row["original_present"] if "original_present" in row.keys() else 1
        ),
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
        "default_species_common": (default_species_common if include_detail else None),
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
        "is_favorite": bool(row["is_favorite"] or 0),
        "is_gallery_eligible": bool(row["is_gallery_eligible"] or 0),
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
        "siblings": inline_siblings,
        "item_key": f"{item_kind}:{item_id}",
    }
    item["modal_detection"] = (
        _build_review_modal_detection(
            row,
            filename=filename,
            full_url=full_url,
            thumb_url=thumb_url,
            optimized_url=optimized_url,
            selected_species=selected_species,
            selected_species_common=selected_species_common,
            current_species=current_species,
            current_species_common=item["current_species_common"],
            common_names=common_names,
            conn=conn,
            siblings=inline_siblings,
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
    rows: list | None = None,
    recent_species: list[dict] | None = None,
) -> tuple[list, dict[int, dict], list[dict], bool]:
    """Load detection rows + Gallery context and group them into BirdEvents.

    The Review surface uses ``core.events.build_bird_events``. Confirmed
    Gallery detections inside the Review neighbourhood are also fetched
    as ``context_only=True`` inputs so the biological event boundary
    stays intact across the Review / Gallery border.
    """
    if rows is None:
        rows = db_service.fetch_review_queue_images(conn, gallery_threshold)
    if recent_species is None:
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
            logger.exception(
                "fetch_review_cluster_context failed; continuing without context"
            )
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


def _build_review_event_member(
    row: dict,
    *,
    conn,
    species_locale: str,
    output_dir: str,
    common_names: dict[str, str],
    recent_species: list[dict],
    siblings_cache: dict[str, list] | None = None,
    include_modal: bool = True,
) -> dict:
    member = _build_review_item(
        row,
        conn=conn,
        species_locale=species_locale,
        output_dir=output_dir,
        common_names=common_names,
        recent_species=recent_species,
        include_detail=False,
        siblings_cache=siblings_cache,
    )
    formatted_date, formatted_time = _split_review_datetime(row.get("timestamp"))
    member["formatted_short_date"] = formatted_date
    member["formatted_time"] = formatted_time
    member["candidate_species"] = _resolve_review_member_candidate_species(row)
    member["candidate_species_common"] = resolve_common_name(
        member["candidate_species"], common_names
    )
    # The grid cell label needs to know
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

    if include_modal and member.get("best_detection_id") and not member["context_only"]:
        member["modal_detection"] = _build_review_modal_detection(
            row,
            filename=member["filename"],
            full_url=member.get("full_url") or "",
            thumb_url=member["thumb_url"],
            optimized_url=member.get("optimized_url") or "",
            selected_species=member.get("candidate_species"),
            selected_species_common=member.get("candidate_species_common"),
            current_species=member.get("species_key"),
            current_species_common=member.get("current_species_common"),
            common_names=common_names,
            conn=conn,
            siblings=None,
            siblings_cache=siblings_cache,
        )
    else:
        member["modal_detection"] = None
    return member


def _load_single_review_event(
    conn,
    *,
    event_key: str,
    gallery_threshold: float,
    output_dir: str,
    species_locale: str,
    common_names: dict[str, str],
) -> dict | None:
    # Rebuild membership from current metadata, including Gallery anchors, but
    # do not construct UI payloads for a write-path validation.
    raw_events, row_map, _, _ = _fetch_review_event_payloads(
        conn,
        gallery_threshold=gallery_threshold,
        output_dir=output_dir,
        species_locale=species_locale,
        common_names=common_names,
        recent_species=[],
    )
    for event in raw_events:
        if event.context_only_count == event.photo_count:
            continue
        if event.event_key != event_key:
            continue
        if not row_map.get(int(event.cover_detection_id or 0)):
            return None
        return {
            "event_key": event.event_key,
            "detection_ids": list(event.detection_ids),
            "candidate_species": event.species,
            "eligibility": event.eligibility,
            "members": [
                {
                    "best_detection_id": row.get("active_detection_id")
                    or row.get("best_detection_id"),
                    "context_only": bool(row.get("context_only")),
                }
                for detection_id in event.detection_ids
                if (row := row_map.get(int(detection_id)))
            ],
        }
    return None


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
    if "diagnostic" in request.args or "layout" in request.args:
        abort(404)
    from web import view_helpers
    from web.services import station_service

    try:
        selection = station_service.validate_selection(
            request.args.get("start", ""),
            request.args.get("end", ""),
        )
        answered_value = request.args.get("answered", "0")
        if answered_value not in {"0", "1"}:
            raise ValueError("Invalid review state")
        answered = answered_value == "1"
        review = station_service.grouped_review(
            answered=answered,
            selection=selection,
            day=request.args.get("day", "latest"),
            page=request.args.get("page", 1, type=int) or 1,
        )
    except ValueError as exc:
        abort(400, description=str(exc))
    if not answered:
        events = view_helpers.build_station_review_events(review["groups"])
        days = list(review["days"])
        selected_day = review["selected_day"]

        def day_url(value: str) -> str:
            return url_for("review.review_page", day=value, **selection)

        older = [day for day in days if day < selected_day]
        newer = [day for day in days if day > selected_day]
        return render_template(
            "review_grid.html",
            review_events=events,
            ordinary_review=True,
            review=review,
            selection=selection,
            current_path="/admin/review",
            review_navigation={
                "days": list(review["days"].items()),
                "selected_day": selected_day,
                "total_events": review["total_items"],
                "is_grid_layout": True,
                "older_url": day_url(max(older))
                if older and selected_day != "all"
                else None,
                "newer_url": day_url(min(newer))
                if newer and selected_day != "all"
                else None,
                "sort_order": "priority",
            },
        )
    for group in review["groups"]:
        group["detections"] = view_helpers.build_station_views(group.pop("rows"))
        for det in group["detections"]:
            det["ordinary_review"] = True
    return render_template(
        "station_review.html",
        review=review,
        detections=[det for group in review["groups"] for det in group["detections"]],
        selection=selection,
        ordinary_review=True,
        answered=answered,
        total_items=review["total_items"],
        page=review["page"],
        current_path="/admin/review",
    )


@review_bp.route("/api/review/modal/<event_key>/<int:detection_id>", methods=["GET"])
@login_required
def review_modal_fragment(event_key: str, detection_id: int) -> Response:
    """Load the shared image editor for one current Review detection."""
    if not re.fullmatch(r"bird-event-[0-9a-f]{12}", event_key) or detection_id <= 0:
        abort(404)
    locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
    with db_service.closing_connection() as conn:
        row = db_service.fetch_review_queue_item_by_identity(
            conn,
            "detection",
            str(detection_id),
            gallery_threshold=config["GALLERY_DISPLAY_THRESHOLD"],
        )
        if row is None:
            abort(404)
        member = _build_review_event_member(
            dict(row),
            conn=conn,
            species_locale=locale,
            output_dir=config.get("OUTPUT_DIR", "output"),
            common_names=load_common_names(locale),
            recent_species=[],
        )
    # event_key scopes client navigation only; write routes revalidate membership.
    response = current_app.make_response(
        render_template(
            "partials/review_modal.html",
            detection=member["modal_detection"],
            event_key=event_key,
        )
    )
    response.headers["Cache-Control"] = "private, no-store"
    return response


@review_bp.route("/api/review-thumb/<filename>", methods=["GET"])
@login_required
def review_thumb(filename):
    """On-demand thumbnail generation for orphan images."""
    output_dir = config.get("OUTPUT_DIR", "output")
    path_mgr = path_service.get_path_manager(output_dir)
    paths = gallery_service.get_image_paths(output_dir, filename)

    original_path = path_mgr.contained_path(paths["original"], path_mgr.originals_dir)
    preview_path = path_mgr.contained_path(paths["preview"], path_mgr.thumbs_dir)
    if original_path is None or preview_path is None:
        return abort(404)

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
@cache_service.invalidates("analytics.")
def review_decision():
    """
    API endpoint for Review Queue decisions.
    POST /api/review/decision
    Payload: { filenames: [...], action: "confirm" | "trash" | "no_bird" | "skip" }

    - confirm -> review_status = 'confirmed_bird'
    - trash -> reject active detections only (housekeeping, no training fact)
    - no_bird -> review_status = 'no_bird' (explicit training fact)
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

        if action == "trash":
            conn = db_service.get_connection()
            try:
                placeholders = ",".join("?" for _ in filenames)
                rows = conn.execute(
                    f"""
                    SELECT detection_id, image_filename
                    FROM detections
                    WHERE image_filename IN ({placeholders})
                      AND COALESCE(status, 'active') = 'active'
                    """,
                    filenames,
                ).fetchall()
                detection_ids = [int(row["detection_id"]) for row in rows]
                db_service.reject_detections(conn, detection_ids)
            finally:
                conn.close()

            gallery_service.invalidate_cache()
            return jsonify(
                {
                    "status": "success",
                    "updated": len(detection_ids),
                    "action": requested_action,
                    "review_status": None,
                }
            )

        # Map action to review_status
        status_map = {
            "confirm": REVIEW_STATUS_CONFIRMED_BIRD,
            "no_bird": REVIEW_STATUS_NO_BIRD,
        }
        new_status = status_map[action]

        conn = db_service.get_connection()
        try:
            if action == "confirm":
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
                                "message": "Cannot confirm Bird Present for items without detections.",
                                "filenames": missing,
                            }
                        ),
                        409,
                    )

            placeholders = ",".join("?" for _ in filenames)
            eligible_rows = conn.execute(
                f"""
                SELECT filename
                FROM images
                WHERE filename IN ({placeholders})
                  AND (review_status IS NULL OR review_status = 'untagged')
                """,
                filenames,
            ).fetchall()
            presence = "present" if action == "confirm" else "absent"
            for row in eligible_rows:
                human_label_service.record_answer(
                    conn,
                    HumanAnswer(
                        image_filename=str(row["filename"]),
                        image_bird_presence=presence,
                    ),
                    source_ref=f"review-decision:{action}",
                )
            conn.commit()
            updated = len(eligible_rows)
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
        return _error_response("Error in review decision", e)


@review_bp.route("/api/review/cleanup/preview", methods=["GET"])
@login_required
def review_cleanup_preview():
    """Dry-run for "Move Review Queue to Trash": queue counts + disclosure.

    Read-only. Returns events/images/detections plus the favorites and
    export-relevant sub-counts so the operator sees what the reversible
    action will touch before running it.
    """
    return jsonify(review_cleanup_service.preview())


@review_bp.route("/api/review/cleanup/run", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
def review_cleanup_run():
    """Move the whole review queue to Trash (reversible, no file deletion)."""
    result = review_cleanup_service.run()
    gallery_service.invalidate_cache()
    logger.info(
        "review cleanup: images_moved=%s detections_moved=%s",
        result.get("images_moved"),
        result.get("detections_moved"),
    )
    return jsonify({"status": "success", **result})


@review_bp.route("/api/review/bbox-review", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
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
            if bbox_review is None:
                human_label_service.retract_bbox_quality(
                    conn,
                    image_filename=filename,
                    detection_id=detection_id,
                    source_ref="review:bbox-review-clear",
                )
            else:
                human_label_service.record_answer(
                    conn,
                    HumanAnswer(
                        image_filename=filename,
                        detection_id=detection_id,
                        bbox_quality=(
                            "suitable"
                            if bbox_review == BBOX_REVIEW_CORRECT
                            else "unsuitable"
                        ),
                    ),
                    source_ref="review:bbox-review",
                )

        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "detection_id": detection_id,
                "bbox_review": bbox_review,
            }
        )
    except Exception as e:
        return _error_response("Error updating bbox review state", e)


@review_bp.route("/api/review/quick-species", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
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

            human_label_service.record_answer(
                conn,
                HumanAnswer(
                    image_filename=filename,
                    detection_id=detection_id,
                    bbox_quality=(
                        "suitable"
                        if bbox_review == BBOX_REVIEW_CORRECT
                        else "unsuitable"
                        if bbox_review is not None
                        else None
                    ),
                    species_identity="corrected",
                    species_key=species,
                ),
                source_ref="review:quick-species",
            )

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
        return _error_response("Error applying review quick species", e)


@review_bp.route("/api/review/approve", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
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

            human_label_service.record_answer(
                conn,
                HumanAnswer(
                    image_filename=filename,
                    detection_id=detection_id,
                    object_bird_presence="present",
                    bbox_quality=(
                        "suitable"
                        if bbox_review == BBOX_REVIEW_CORRECT
                        else "unsuitable"
                    ),
                    species_identity="corrected",
                    species_key=species,
                ),
                source_ref="review:approve",
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
        return _error_response("Error approving review item", e)


@review_bp.route("/api/review/event-approve", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
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

    if bbox_review not in (None, *VALID_BBOX_REVIEW_STATES):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Invalid bbox_review: {bbox_review}",
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
            event = None
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
                SELECT d.detection_id, d.image_filename, i.review_status,
                       d.manual_species_override, d.species_source
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

            # Per-frame relabel beats event-level species. If a frame
            # already carries a manual species override (set via the
            # per-cell relabel path through /api/moderation/bulk/relabel
            # — e.g. the operator used Multi-Select to relabel a
            # subset of frames before approving), do NOT overwrite it
            # with the event species picked in the right control rail.
            # Mirrors the `per-frame wins` rule in event-resolve.
            def _row_field(row, key):
                try:
                    return row[key]
                except (KeyError, IndexError):
                    return None

            for row in rows:
                manual_species = str(
                    _row_field(row, "manual_species_override") or ""
                ).strip()
                manual_source = str(_row_field(row, "species_source") or "").strip()
                chosen_species = (
                    manual_species
                    if manual_species and manual_source == "manual"
                    else species
                )
                human_label_service.record_answer(
                    conn,
                    HumanAnswer(
                        image_filename=str(row["image_filename"]),
                        detection_id=int(row["detection_id"]),
                        object_bird_presence="present",
                        bbox_quality=(
                            "suitable"
                            if bbox_review == BBOX_REVIEW_CORRECT
                            else "unsuitable"
                            if bbox_review is not None
                            else None
                        ),
                        species_identity="corrected",
                        species_key=chosen_species,
                    ),
                    source_ref="review:event-approve",
                )

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
        return _error_response("Error approving review event", e)


@review_bp.route("/api/review/event-trash", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
def review_event_trash():
    """Reject every active detection in one BirdEvent.

    Semantic target: reject the event's detections, not the image files.
    Image visibility is recomputed afterwards. Zero active proposals does not
    imply whole-image ``no_bird``: without an explicit image fact the frame
    remains ``untagged``. Existing explicit image presence is projected to the
    compatible review status; otherwise remaining active detections determine
    visibility.
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

                # Move Event to Trash must only reject the event's
                # actionable (non-context) detections. Gallery-anchor
                # context frames were already approved earlier and stay
                # in the Gallery. Accept both payload shapes so that
                # older tabs that still submit the full event id list
                # continue to work — we down-cast to actionable ids
                # server-side. Mirrors the event-approve contract.
                event_detection_ids = sorted(event.get("detection_ids") or [])
                actionable_event_detection_ids = sorted(
                    int(member.get("best_detection_id") or 0)
                    for member in (event.get("members") or [])
                    if not member.get("context_only")
                    and int(member.get("best_detection_id") or 0) > 0
                )
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

            for row in rows:
                human_label_service.record_answer(
                    conn,
                    HumanAnswer(
                        image_filename=str(row["image_filename"]),
                        detection_id=int(row["detection_id"]),
                        object_bird_presence="absent",
                    ),
                    source_ref="review:event-no-bird",
                )
            # Keep the established rejection projection for compatibility
            # with callers that provide an alternative label recorder.
            db_service.reject_detections(conn, detection_ids, commit=False)

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
            message = "Event rejected. Every touched image already has an explicit whole-image No Bird answer."
        elif review_filenames:
            message = "Event detections rejected. Images without an explicit full-image answer remain in Review."
        else:
            message = "Event detections rejected. Touched images now follow their explicit image answers and remaining active detections."

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
        return _error_response("Error trashing review event", e)


@review_bp.route("/api/review/event-unresolved", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
def review_event_unresolved():
    """Close an event as bird-present without asserting a species."""
    data = request.get_json() or {}
    event_key = str(data.get("event_key") or "").strip()
    detection_ids: list[int] = []
    for value in data.get("detection_ids") or []:
        try:
            detection_id = int(value)
        except (TypeError, ValueError):
            continue
        if detection_id > 0:
            detection_ids.append(detection_id)
    detection_ids = list(dict.fromkeys(detection_ids))

    if not event_key or not detection_ids:
        return jsonify({"status": "error", "message": "event is required"}), 400

    try:
        with db_service.closing_connection() as conn:
            locale = config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
            event = _load_single_review_event(
                conn,
                event_key=event_key,
                gallery_threshold=config["GALLERY_DISPLAY_THRESHOLD"],
                output_dir=config.get("OUTPUT_DIR", "output"),
                species_locale=locale,
                common_names=load_common_names(locale),
            )
            if not event:
                return (
                    jsonify(
                        {"status": "error", "message": "Review event no longer exists"}
                    ),
                    409,
                )
            actionable_ids = sorted(
                int(member.get("best_detection_id") or 0)
                for member in event.get("members") or []
                if not member.get("context_only")
                and int(member.get("best_detection_id") or 0) > 0
            )
            if sorted(detection_ids) != actionable_ids:
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
                SELECT d.detection_id, d.image_filename, i.timestamp
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

            for row in rows:
                human_label_service.record_answer(
                    conn,
                    HumanAnswer(
                        image_filename=str(row["image_filename"]),
                        detection_id=int(row["detection_id"]),
                        object_bird_presence="present",
                        species_identity="unknown",
                    ),
                    source_ref="review:event-unresolved",
                )

            touched_filenames = list(
                dict.fromkeys(str(row["image_filename"]) for row in rows)
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
        return jsonify(
            {
                "status": "success",
                "event_key": event_key,
                "detection_ids": detection_ids,
                "review_status_by_filename": review_status_by_filename,
                "message": "Event saved as bird present; species remains unresolved.",
            }
        )
    except Exception as e:
        return _error_response("Error closing unresolved review event", e)


@review_bp.route("/api/review/event-resolve", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
def review_event_resolve():
    """Resolve a mixed BirdEvent in one transaction.

    Accepts disjoint ``keep_detection_ids`` + ``trash_detection_ids`` lists
    that together must cover every active detection of the event. The
    ``species`` + ``bbox_review`` selection applies only to the Keep
    frames; Trash frames are rejected via the same code path as
    ``/api/review/event-trash`` so image visibility is recomputed
    consistently. Zero remaining detections is not treated as a whole-image
    ``no bird`` statement; without an explicit image answer the frame remains
    in Review.

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

            event = None
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
                       i.review_status,
                       i.timestamp
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

            rows_by_id = {int(row["detection_id"]): row for row in rows}
            for detection_id in keep_ids:
                row = rows_by_id[detection_id]
                chosen_species = (
                    str(row["manual_species_override"]).strip()
                    if detection_id in manual_override_ids
                    else species
                )
                human_label_service.record_answer(
                    conn,
                    HumanAnswer(
                        image_filename=str(row["image_filename"]),
                        detection_id=detection_id,
                        object_bird_presence="present",
                        bbox_quality=(
                            "suitable"
                            if bbox_review == BBOX_REVIEW_CORRECT
                            else "unsuitable"
                        ),
                        species_identity="corrected",
                        species_key=chosen_species,
                    ),
                    source_ref="review:event-resolve:keep",
                )

            db_service.reject_detections(conn, trash_ids, commit=False)

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
        return _error_response("Error resolving review event", e)


@review_bp.route("/api/review/answers", methods=["POST"])
@login_required
@cache_service.invalidates("analytics.")
def review_answers():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(status="error", message="Invalid review answer"), 400
    species = data.get("species")
    try:
        with db_service.closing_connection() as conn:
            if data.get("action") == "species" and (
                not isinstance(species, str)
                or species
                not in _get_allowed_review_species(
                    conn, config.get("SPECIES_COMMON_NAME_LOCALE", "DE")
                )
            ):
                raise ValueError("Choose a valid species")
            count = human_label_service.record_review_answers(
                conn, data.get("detection_ids"), data.get("action"), species
            )
        return jsonify(status="success", answered=count)
    except ValueError as exc:
        return jsonify(status="error", message=str(exc)), 409
