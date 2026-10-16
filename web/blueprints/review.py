"""Review queue, modal and thumbnail routes."""

import re
import time
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    render_template,
    request,
    send_file,
    url_for,
)

from config import get_config
from core.events import build_bird_events
from logging_config import get_logger
from utils.db import fetch_sibling_detections
from utils.review_metadata import (
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
from web.services import (
    db_service,
    gallery_service,
    human_label_service,
    path_service,
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
        # ``non_bird``) and stale ``human_species_key`` values
        # from prior accidental relabels must never resurface as a
        # picker suggestion. The species validator
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
    human_species_key: str | None,
    common_names: dict[str, str],
    species_source: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve what the review UI shows - and would submit - as the species.

    An explicit human "species unknown" answer clears
    ``human_species_key``, which makes it indistinguishable from a row
    nobody has answered yet. Only ``species_source`` separates the two, and
    the difference matters: pre-selecting the classifier's top-1 for a row a
    human just marked unknown both mislabels the header and puts that species
    one Approve click away from becoming a positive label.
    """
    manual_species = str(human_species_key or "").strip()
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
    manual = str(row.get("human_species_key") or "").strip()
    if manual:
        return manual
    if is_human_unknown_species_source(row.get("human_species_source")):
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
            manual_override=sibling["human_species_key"],
            species_key=sibling["species_key"],
            cls_class_name=sibling["cls_class_name"],
            od_class_name=sibling["od_class_name"],
            species_source=sibling["human_species_source"],
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
                "human_species_key": sibling["human_species_key"],
                "human_species_source": sibling["human_species_source"],
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
    states = human_label_service.fetch_detection_review_states(
        conn, [int(row["detection_id"]) for row in siblings if row.get("detection_id")]
    )
    for row in siblings:
        state = states.get(row.get("detection_id"))
        if state:
            row["human_review_state"] = state["state"]
            row["human_reviewed"] = state["reviewed"]
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
        or str(row["human_species_key"] or "").strip()
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
        "human_species_key": row["human_species_key"],
        "human_species_source": row["human_species_source"],
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
                human_species_key=row["human_species_key"],
                common_names=common_names,
                species_source=row["human_species_source"],
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
                human_species_key=row["human_species_key"],
                common_names=common_names,
                species_source=row["human_species_source"],
            )
        )
    human_bbox_verdict = row["human_bbox_verdict"]
    selected_bbox_review = (
        human_bbox_verdict if human_bbox_verdict in {"suitable", "unsuitable"} else None
    )
    selected_bbox_review_origin = (
        "manual" if human_bbox_verdict in {"suitable", "unsuitable"} else None
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
        "human_species_key": row["human_species_key"],
        "manual_species_common": resolve_common_name(
            row["human_species_key"], common_names
        ),
        "human_species_source": row["human_species_source"],
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
        "human_bbox_verdict": human_bbox_verdict,
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
    member["species_is_manual"] = bool(str(row.get("human_species_key") or "").strip())
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
    days = list(review["days"])
    selected_day = review["selected_day"]

    def day_url(value: str) -> str:
        return url_for(
            "review.review_page", day=value, answered=int(answered), **selection
        )

    older = [day for day in days if day < selected_day]
    newer = [day for day in days if day > selected_day]
    navigation = {
        "days": list(review["days"].items()),
        "selected_day": selected_day,
        "total_events": review["total_items"],
        "is_grid_layout": True,
        "older_url": day_url(max(older)) if older and selected_day != "all" else None,
        "newer_url": day_url(min(newer)) if newer and selected_day != "all" else None,
    }
    events = []
    detections = []
    if answered:
        for group in review["groups"]:
            group["detections"] = view_helpers.build_station_views(group.pop("rows"))
            for det in group["detections"]:
                det["ordinary_review"] = not det.get("image_statement")
                detections.append(det)
    else:
        events = view_helpers.build_station_review_events(review["groups"])
    return render_template(
        "station_review.html" if answered else "review_grid.html",
        review_events=events,
        review_navigation=navigation,
        review=review,
        detections=detections,
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
