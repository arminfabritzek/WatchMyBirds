"""
Review Blueprint.

Handles review queue routes:
- GET /admin/review - Review queue page (was orphans)
- GET /api/review-thumb/<filename> - On-demand thumbnail
- POST /api/review/decision - Review decisions (confirm/trash/no_bird/skip)
- POST /api/review/bbox-review - Persist bbox review state
- POST /api/review/quick-species - Confirm + relabel via quick species buttons
"""

import time
from datetime import datetime

from flask import Blueprint, abort, jsonify, render_template, request, send_file

from config import get_config
from logging_config import get_logger
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
from utils.db import fetch_sibling_detections
from web.blueprints.auth import login_required
from web.species_thumbnails import get_species_thumbnail_map, resolve_species_thumbnail_url
from web.services import db_service, gallery_service

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
        active_orphan = None
        if orphans:
            species_thumbnail_map = get_species_thumbnail_map(
                common_names=common_names,
                cache_key=f"review:{species_locale}",
            )
            active_orphan = _load_single_review_item_by_identity(
                conn,
                item_kind=orphans[0]["item_kind"],
                item_id=orphans[0]["item_id"],
                gallery_threshold=gallery_threshold,
                output_dir=output_dir,
                species_locale=species_locale,
                common_names=common_names,
                species_thumbnail_map=species_thumbnail_map,
            )

    return render_template(
        "orphans.html",
        orphans=orphans,
        active_orphan=active_orphan,
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

    return render_template("components/review_stage_panel.html", orphan=orphan)


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
                (filename, config["GALLERY_DISPLAY_THRESHOLD"]),
            ).fetchone()[0]

            if unresolved == 0:
                db_service.update_review_status(
                    conn, [filename], REVIEW_STATUS_CONFIRMED_BIRD
                )
                image_review_status = REVIEW_STATUS_CONFIRMED_BIRD
            else:
                conn.execute(
                    """
                    UPDATE images
                    SET review_status = 'untagged'
                    WHERE filename = ?
                    """,
                    (filename,),
                )
                conn.commit()
                image_review_status = REVIEW_STATUS_UNTAGGED

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
