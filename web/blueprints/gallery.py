"""Gallery, edit, and species page routes.

Extracted from the ``create_web_interface`` factory closure in
``web/web_interface.py``. Follows the ``init_*_bp()`` pattern used across
``web/blueprints/`` (reference: ``web/blueprints/inbox.py``): a module-level
``Blueprint``, an ``init_gallery_bp(detection_manager=...)`` injector, and
routes as top-level functions.

The data helpers these routes lean on (``get_detections_for_date``,
``get_captured_detections``, ``get_daily_covers``, ``get_daily_species_summary``,
``get_species_key``, ``get_common_name``, ``build_detection_view_dict``,
``pick_cover_for_group``, plus ``COMMON_NAMES`` / ``UNKNOWN_SPECIES_KEY``) live
in ``web/view_helpers.py`` and are read live (``COMMON_NAMES`` is the same dict
object mutated in place on a locale change).

Routes are registered with explicit ``endpoint=`` names via ``add_url_rule`` so
the URL map (and therefore ``url_for`` callers) is byte-for-byte identical to
the factory's previous registrations.
"""

import math
import os
import re
from datetime import datetime

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from config import get_config
from core.species_colours import assign_species_colours as _assign_species_colours
from logging_config import get_logger
from web import view_helpers
from web.blueprints.auth import login_required
from web.security import safe_log_value as _slv
from web.services import db_service, gallery_service

logger = get_logger(__name__)
config = get_config()

gallery_bp = Blueprint("gallery", __name__)

IMAGE_WIDTH = 150
PAGE_SIZE = 50


_detection_manager = None


def init_gallery_bp(detection_manager=None):
    global _detection_manager
    _detection_manager = detection_manager


def daily_species_summary_route():
    date_iso = request.args.get("date")
    if not date_iso:
        date_iso = datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        return (
            jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}),
            400,
        )
    summary = view_helpers.get_daily_species_summary(date_iso)
    return jsonify({"date": date_iso, "summary": summary})


@login_required
def edit_route(date_iso):

    try:
        datetime.strptime(date_iso, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD", 400

    filters = {
        "status": request.args.get("status", "all"),
        "species": request.args.get("species", "all"),
        "sort": request.args.get("sort", "time_desc"),
        "min_conf": request.args.get("min_conf", "0.0"),
    }
    if filters["species"] in {"Unknown", "Unclassified"}:
        filters["species"] = view_helpers.UNKNOWN_SPECIES_KEY

    detections = view_helpers.get_detections_for_date(date_iso)
    if not detections:
        return render_template(
            "edit.html",
            date_iso=date_iso,
            detections=[],
            filters=filters,
            species_list=[],
            image_width=IMAGE_WIDTH,
        )

    species_list = sorted(
        list(set(view_helpers.get_species_key(det) for det in detections))
    )

    filtered = []
    try:
        min_conf_val = float(filters["min_conf"])
    except ValueError:
        min_conf_val = 0.0

    for det in detections:
        is_downloaded = bool(det.get("downloaded_timestamp"))
        if filters["status"] == "downloaded" and not is_downloaded:
            continue
        if filters["status"] == "not_downloaded" and is_downloaded:
            continue

        sp = view_helpers.get_species_key(det)
        if filters["species"] != "all" and sp != filters["species"]:
            continue

        conf = max(det.get("od_confidence") or 0, det.get("cls_confidence") or 0)
        if conf < min_conf_val:
            continue

        thumb_virtual = det.get("thumbnail_path_virtual")
        relative_path = det.get("relative_path", "")
        original_name = det.get("original_name", "")

        ts = det.get("image_timestamp", "")
        date_folder = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else date_iso

        if thumb_virtual:
            det["display_path"] = f"/uploads/derivatives/thumbs/{thumb_virtual}"
        else:
            det["display_path"] = f"/uploads/derivatives/optimized/{relative_path}"

        det["full_path"] = f"/uploads/derivatives/optimized/{relative_path}"
        det["original_path"] = f"/uploads/originals/{date_folder}/{original_name}"
        species_key = view_helpers.get_species_key(det)
        det["species_key"] = species_key
        det["common_name"] = view_helpers.get_common_name(species_key)
        det["latin_name"] = species_key or ""

        filtered.append(det)

    _stream_colour_map = _assign_species_colours(
        [d.get("species_key") or "" for d in filtered]
    )
    for d in filtered:
        d["species_colour"] = _stream_colour_map.get(d.get("species_key") or "", None)

    if filters["sort"] == "time_asc":
        filtered.sort(key=lambda x: x["image_timestamp"])
    elif filters["sort"] == "time_desc":
        filtered.sort(key=lambda x: x["image_timestamp"], reverse=True)
    elif filters["sort"] == "score":
        filtered.sort(key=lambda x: x["score"] or 0.0, reverse=True)
    elif filters["sort"] == "confidence":
        filtered.sort(
            key=lambda x: max(
                x.get("od_confidence") or 0, x.get("cls_confidence") or 0
            ),
            reverse=True,
        )

    page = request.args.get("page", 1, type=int)
    per_page = 100
    total_items = len(filtered)
    total_pages = math.ceil(total_items / per_page)

    if page < 1:
        page = 1
    if page > total_pages and total_pages > 0:
        page = total_pages

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    detections_page = filtered[start_idx:end_idx]

    return render_template(
        "edit.html",
        date_iso=date_iso,
        detections=detections_page,
        filters=filters,
        species_list=species_list,
        image_width=IMAGE_WIDTH,
        pagination={
            "page": page,
            "total_pages": total_pages,
            "total_items": total_items,
            "has_prev": page > 1,
            "has_next": page < total_pages,
            "prev_num": page - 1,
            "next_num": page + 1,
        },
    )


@login_required
def edit_actions_route():

    action = request.form.get("action")
    date_iso = request.form.get("date_iso")
    det_ids = request.form.getlist("ids")

    safe_date: str | None = None
    if date_iso is not None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_iso):
            try:
                datetime.strptime(date_iso, "%Y-%m-%d")
                safe_date = date_iso
            except (TypeError, ValueError):
                safe_date = None
        if safe_date is None:
            return redirect("/gallery")

    if action == "reject_all":
        if not safe_date:
            return redirect("/gallery")

        with db_service.closing_connection() as conn:
            date_prefix = safe_date.replace("-", "")

            query = """
                SELECT d.detection_id
                FROM detections d
                JOIN images i ON d.image_filename = i.filename
                WHERE i.timestamp LIKE ?
                AND (d.status IS NULL OR d.status != 'rejected')
            """
            rows = conn.execute(query, (f"{date_prefix}%",)).fetchall()
            all_ids = [r["detection_id"] for r in rows]

            if all_ids:
                db_service.reject_detections(conn, all_ids)
                logger.info(
                    "Rejected ALL %d detections for %s",
                    len(all_ids),
                    _slv(safe_date),
                )

        gallery_service.invalidate_cache()
        return redirect(f"/gallery/{safe_date}")

    if not det_ids:
        return redirect(f"/edit/{safe_date}" if safe_date else "/gallery")

    ids_int = [int(i) for i in det_ids]

    if action == "reject":
        with db_service.closing_connection() as conn:
            db_service.reject_detections(conn, ids_int)

        gallery_service.invalidate_cache()
        return redirect(f"/edit/{safe_date}" if safe_date else "/gallery")

    elif action == "download":
        import io
        import zipfile

        from flask import send_file

        from web.services import metadata_export_service as mx

        burn_in = mx.burn_in_enabled()

        with db_service.closing_connection() as conn:
            placeholders = ",".join("?" for _ in ids_int)
            query = f"""
                SELECT d.detection_id, i.filename as original_name, i.timestamp
                FROM detections d
                JOIN images i ON d.image_filename = i.filename
                WHERE d.detection_id IN ({placeholders})
            """
            rows = conn.execute(query, ids_int).fetchall()

            output_dir = config.get("OUTPUT_DIR", "detections")
            files_to_zip = []

            seen_images: set[str] = set()

            for r in rows:
                original_name, ts = r["original_name"], r["timestamp"]
                if not original_name or not ts:
                    continue
                if original_name in seen_images:
                    continue
                seen_images.add(original_name)

                date_folder = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""

                abs_path = os.path.join(
                    output_dir, "originals", date_folder, original_name
                )
                files_to_zip.append((abs_path, original_name, ts))

            if files_to_zip:
                download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filenames = [f[1] for f in files_to_zip]
                db_service.update_downloaded_timestamp(conn, filenames, download_time)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for abs_path, original_name, ts in files_to_zip:
                if not os.path.exists(abs_path):
                    continue
                if burn_in:
                    try:
                        copy_bytes = mx.produce_copy_bytes(original_name)
                        arcname = mx.export_filename(original_name, ts)
                        zf.writestr(arcname, copy_bytes)
                        continue
                    except Exception:
                        logger.exception(
                            "metadata burn-in failed for %s; zipping raw original",
                            original_name,
                        )
                zf.write(abs_path, arcname=original_name)

        zip_buffer.seek(0)
        download_name = f"watchmybirds_{date_iso.replace('-', '')}_download.zip"
        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=download_name,
        )

    return redirect(f"/edit/{date_iso}")


def species_route():

    all_detections = view_helpers.get_captured_detections()

    try:
        min_score_param = request.args.get("min_score", type=float)
    except (ValueError, TypeError):
        min_score_param = None

    if min_score_param is not None:
        current_threshold = min_score_param
    else:
        current_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

    if current_threshold > 0:
        all_detections = [
            d for d in all_detections if (d.get("score") or 0.0) >= current_threshold
        ]

    species_candidates = {}
    for det in all_detections:
        species_key = view_helpers.get_species_key(det)
        species_candidates.setdefault(species_key, []).append(det)

    today_iso = datetime.now().strftime("%Y-%m-%d")
    species_groups = {}
    for s_key, candidates in species_candidates.items():
        chosen = view_helpers.pick_cover_for_group(
            candidates, seed_key=f"species:{s_key}", date_iso=today_iso
        )
        if chosen:
            species_groups[s_key] = chosen

    detections = []
    for species, det in sorted(
        species_groups.items(),
        key=lambda x: view_helpers.COMMON_NAMES.get(x[0], x[0]),
    ):
        full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")
        thumb_virtual = det.get("thumbnail_path_virtual")

        if thumb_virtual:
            display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
        else:
            display_url = f"/uploads/derivatives/optimized/{full_path}"

        full_url = f"/uploads/derivatives/optimized/{full_path}"
        original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"

        ts = det.get("image_timestamp", "")
        if len(ts) >= 15:
            date_str = ts[:8]
            time_str = ts[9:15]
            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
            gallery_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            formatted_date = ""
            formatted_time = ""
            gallery_date = ""

        detections.append(
            view_helpers.build_detection_view_dict(
                det,
                species_key=species,
                common_name=view_helpers.get_common_name(species),
                formatted_date=formatted_date,
                formatted_time=formatted_time,
                gallery_date=gallery_date,
                extra={
                    "display_path": display_url,
                    "full_path": full_url,
                    "original_path": original_url,
                },
            )
        )

    return render_template(
        "species.html",
        current_path="/species",
        detections=detections,
        image_width=IMAGE_WIDTH,
        current_threshold=current_threshold,
        species_count=len(detections),
    )


def species_overview_route():

    raw_species_key = request.args.get("species_key", type=str) or ""
    species_key = raw_species_key.strip().replace(" ", "_")
    if species_key in {"Unknown", "Unclassified"}:
        species_key = view_helpers.UNKNOWN_SPECIES_KEY
    if not species_key:
        return redirect(url_for("gallery.species"))

    page = request.args.get("page", 1, type=int)

    try:
        min_score_param = request.args.get("min_score", type=float)
    except (ValueError, TypeError):
        min_score_param = None

    if min_score_param is not None:
        current_threshold = min_score_param
    else:
        current_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

    all_detections = view_helpers.get_captured_detections()

    filtered = []
    for det in all_detections:
        det_species_key = view_helpers.get_species_key(det)
        if det_species_key != species_key:
            continue
        if current_threshold > 0 and (det.get("score") or 0.0) < current_threshold:
            continue
        filtered.append(det)

    total_items = len(filtered)
    total_pages = math.ceil(total_items / PAGE_SIZE) or 1
    page = max(1, min(page, total_pages))
    start_index = (page - 1) * PAGE_SIZE
    end_index = page * PAGE_SIZE
    page_detections_raw = filtered[start_index:end_index]

    detections = []
    for det in page_detections_raw:
        full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")
        thumb_virtual = det.get("thumbnail_path_virtual")

        if thumb_virtual:
            display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
        else:
            display_url = f"/uploads/derivatives/optimized/{full_path}"

        full_url = f"/uploads/derivatives/optimized/{full_path}"
        original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"

        ts = det.get("image_timestamp", "")
        if len(ts) >= 15:
            date_str = ts[:8]
            time_str = ts[9:15]
            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
            gallery_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            formatted_date = ""
            formatted_time = ""
            gallery_date = ""

        sibling_count = det.get("sibling_count", 1) or 1
        siblings = []
        if sibling_count > 1:
            original_name = det.get("original_name", "")
            if original_name:
                sibling_rows = gallery_service.get_sibling_detections(original_name)
                for sib in sibling_rows:
                    sib_species_key = view_helpers.get_species_key(sib)
                    sib_thumb = sib.get("thumbnail_path_virtual")
                    siblings.append(
                        view_helpers.build_detection_view_dict(
                            sib,
                            species_key=sib_species_key,
                            common_name=view_helpers.get_common_name(sib_species_key),
                            include_decision_state=True,
                            extra={
                                "thumb_url": (
                                    f"/uploads/derivatives/thumbs/{sib_thumb}"
                                    if sib_thumb
                                    else ""
                                ),
                            },
                        )
                    )

        detections.append(
            view_helpers.build_detection_view_dict(
                det,
                species_key=species_key,
                common_name=view_helpers.get_common_name(species_key),
                formatted_date=formatted_date,
                formatted_time=formatted_time,
                gallery_date=gallery_date,
                siblings=siblings,
                sibling_count=sibling_count,
                include_decision_state=True,
                extra={
                    "display_path": display_url,
                    "full_path": full_url,
                    "original_path": original_url,
                },
            )
        )

    window = 2
    pagination_range = []
    range_start = max(1, page - window)
    range_end = min(total_pages, page + window)

    if range_start > 1:
        pagination_range.append(1)
        if range_start > 2:
            pagination_range.append("...")

    for p in range(range_start, range_end + 1):
        pagination_range.append(p)

    if range_end < total_pages:
        if range_end < total_pages - 1:
            pagination_range.append("...")
        pagination_range.append(total_pages)

    return render_template(
        "species_overview.html",
        current_path="/species",
        species_key=species_key,
        species_common_name=view_helpers.get_common_name(species_key),
        current_threshold=current_threshold,
        detections=detections,
        page=page,
        total_pages=total_pages,
        total_items=total_items,
        pagination_range=pagination_range,
        image_width=IMAGE_WIDTH,
    )


def gallery_route():

    daily_covers = view_helpers.get_daily_covers()
    sorted_dates = sorted(daily_covers.keys(), reverse=True)

    days = []
    for date_str in sorted_dates:
        data = daily_covers.get(date_str)
        if not data:
            continue

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            display_date = dt.strftime("%A, %d. %B %Y")
        except Exception:
            display_date = date_str

        days.append(
            {
                "date": date_str,
                "display_date": display_date,
                "cover_path": data.get("path", ""),
                "count": data.get("count", 0),
                "cover_detection_id": data.get("detection_id"),
            }
        )

    return render_template(
        "gallery.html",
        current_path="/gallery",
        days=days,
        image_width=IMAGE_WIDTH,
    )


def subgallery_route(date):

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return "Invalid date format.", 400

    page = request.args.get("page", 1, type=int)
    sort_by = request.args.get("sort", "time_desc")

    try:
        min_score_param = request.args.get("min_score", type=float)
    except (ValueError, TypeError):
        min_score_param = None

    if min_score_param is not None:
        current_threshold = min_score_param
    else:
        current_threshold = config["GALLERY_DISPLAY_THRESHOLD"]

    with db_service.closing_connection() as conn:
        rows = db_service.fetch_detections_for_gallery(conn, date, order_by="time")
    detections_raw = [dict(row) for row in rows]

    observations_all = gallery_service.group_detections_into_observations(
        detections_raw
    )
    total_obs_unfiltered = len(observations_all)

    focus_id_param = request.args.get("focus", type=int)
    if current_threshold > 0:
        observations_filtered = []
        for obs in observations_all:
            if obs["best_score"] >= current_threshold:
                observations_filtered.append(obs)
            elif focus_id_param and focus_id_param in obs["detection_ids"]:
                observations_filtered.append(obs)
        observations_all = observations_filtered

    if sort_by == "time_asc":
        observations_all.sort(key=lambda o: o["end_time"])
    elif sort_by == "score":
        observations_all.sort(key=lambda o: o["best_score"], reverse=True)
    elif sort_by == "species":
        observations_all.sort(
            key=lambda o: view_helpers.COMMON_NAMES.get(
                o["species"], o["species"]
            ).lower()
        )
    else:
        observations_all.sort(key=lambda o: o["end_time"], reverse=True)

    total_items = len(observations_all)
    total_pages = math.ceil(total_items / PAGE_SIZE) or 1

    if focus_id_param and page == 1:
        for idx, obs in enumerate(observations_all):
            if focus_id_param in obs["detection_ids"]:
                page = (idx // PAGE_SIZE) + 1
                break

    page = max(1, min(page, total_pages))
    start_index = (page - 1) * PAGE_SIZE
    end_index = page * PAGE_SIZE
    page_observations = observations_all[start_index:end_index]
    focus_observation_id = None
    if focus_id_param:
        for obs in page_observations:
            if focus_id_param in obs["detection_ids"]:
                focus_observation_id = obs["observation_id"]
                break

    det_by_id = {d.get("detection_id"): d for d in detections_raw}

    def enrich_detection(det):
        full_path = det.get("relative_path") or det.get("optimized_name_virtual", "")

        thumb_virtual = det.get("thumbnail_path_virtual")

        if thumb_virtual:
            display_url = f"/uploads/derivatives/thumbs/{thumb_virtual}"
        else:
            display_url = f"/uploads/derivatives/optimized/{full_path}"

        full_url = f"/uploads/derivatives/optimized/{full_path}"
        original_url = f"/uploads/originals/{full_path.replace('.webp', '.jpg')}"

        ts = det.get("image_timestamp", "")
        if len(ts) >= 15:
            date_str = ts[:8]
            time_str = ts[9:15]
            formatted_date = f"{date_str[6:8]}.{date_str[4:6]}.{date_str[0:4]}"
            formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"
        else:
            formatted_date = ""
            formatted_time = ""

        species_key = view_helpers.get_species_key(det)
        sibling_count = det.get("sibling_count", 1) or 1

        siblings = []
        if sibling_count > 1:
            original_name = det.get("original_name", "")
            if original_name:
                sibling_rows = gallery_service.get_sibling_detections(original_name)
                for sib in sibling_rows:
                    sib_species_key = view_helpers.get_species_key(sib)
                    sib_thumb = sib["thumbnail_path_virtual"]
                    siblings.append(
                        view_helpers.build_detection_view_dict(
                            sib,
                            species_key=sib_species_key,
                            common_name=view_helpers.get_common_name(sib_species_key),
                            include_decision_state=True,
                            extra={
                                "thumb_url": (
                                    f"/uploads/derivatives/thumbs/{sib_thumb}"
                                    if sib_thumb
                                    else ""
                                ),
                            },
                        )
                    )

        return view_helpers.build_detection_view_dict(
            det,
            species_key=species_key,
            common_name=view_helpers.get_common_name(species_key),
            formatted_date=formatted_date,
            formatted_time=formatted_time,
            gallery_date=date,
            siblings=siblings,
            sibling_count=sibling_count,
            include_decision_state=True,
            extra={
                "image_timestamp": ts,
                "display_url": display_url,
                "full_url": full_url,
                "original_url": original_url,
                "display_path": display_url,
                "full_path": full_url,
                "original_path": original_url,
            },
        )

    def _format_duration(sec: float) -> str:

        sec = max(0, int(sec))
        if sec < 60:
            return f"{sec}s"
        minutes = sec // 60
        remaining = sec % 60
        return f"{minutes}m {remaining:02d}s"

    enriched_observations = []
    for obs in page_observations:
        cover_det = det_by_id.get(obs["cover_detection_id"])
        if not cover_det:
            continue
        enriched_cover = enrich_detection(cover_det)

        all_dets_enriched = []
        for did in obs["detection_ids"]:
            raw = det_by_id.get(did)
            if raw:
                all_dets_enriched.append(enrich_detection(raw))
        # Modal navigation should follow the visible gallery sequence:
        # observation cards stay in grid order, and filmstrip detections
        # remain adjacent to their observation instead of being interleaved
        all_dets_enriched.sort(
            key=lambda det: (
                det.get("image_timestamp", ""),
                int(det.get("detection_id") or 0),
            ),
            reverse=True,
        )

        enriched_observations.append(
            {
                "observation_id": obs["observation_id"],
                "species": obs["species"],
                "common_name": view_helpers.get_common_name(obs["species"]),
                "photo_count": obs["photo_count"],
                "duration_sec": obs["duration_sec"],
                "duration_display": _format_duration(obs["duration_sec"]),
                "best_score": obs["best_score"],
                "cover_detection": enriched_cover,
                "all_detections": all_dets_enriched,
                "detection_ids": obs["detection_ids"],
                "start_time": obs["start_time"],
                "end_time": obs["end_time"],
            }
        )

    nav_index_by_detection_id: dict[int, int] = {}
    nav_order = [det for obs in enriched_observations for det in obs["all_detections"]]
    for idx, det in enumerate(nav_order):
        det_id = int(det.get("detection_id") or 0)
        if det_id > 0:
            nav_index_by_detection_id[det_id] = idx

    for obs in enriched_observations:
        obs["cover_detection"]["nav_index"] = nav_index_by_detection_id.get(
            int(obs["cover_detection"].get("detection_id") or 0)
        )
        for det in obs["all_detections"]:
            det["nav_index"] = nav_index_by_detection_id.get(
                int(det.get("detection_id") or 0)
            )

    _sub_colour_map = _assign_species_colours(
        [obs.get("species") or "" for obs in enriched_observations]
    )
    for obs in enriched_observations:
        slot = _sub_colour_map.get(obs.get("species") or "", None)
        obs["species_colour"] = slot
        obs["cover_detection"]["species_colour"] = slot
        for det in obs["all_detections"]:
            det["species_colour"] = slot

    species_of_day = []
    if page == 1:
        species_candidates = {}
        for det in detections_raw:
            species_key = view_helpers.get_species_key(det)
            species_candidates.setdefault(species_key, []).append(det)

        species_groups = {}
        for s_key, candidates in species_candidates.items():
            chosen = view_helpers.pick_cover_for_group(
                candidates, seed_key=f"daydetail:{s_key}", date_iso=date
            )
            if chosen:
                species_groups[s_key] = chosen

        species_of_day = [
            enrich_detection(d)
            for d in sorted(
                species_groups.values(),
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
        ]

    window = 2
    pagination_range = []
    range_start = max(1, page - window)
    range_end = min(total_pages, page + window)

    if range_start > 1:
        pagination_range.append(1)
        if range_start > 2:
            pagination_range.append("...")

    for p in range(range_start, range_end + 1):
        pagination_range.append(p)

    if range_end < total_pages:
        if range_end < total_pages - 1:
            pagination_range.append("...")
        pagination_range.append(total_pages)

    visit_windows = gallery_service.group_concurrent_observations(enriched_observations)

    if sort_by == "time_asc":
        visit_windows.sort(key=lambda w: w[0]["start_time"])
    elif sort_by == "score":
        visit_windows.sort(
            key=lambda w: max(o["best_score"] for o in w),
            reverse=True,
        )
    elif sort_by == "species":
        visit_windows.sort(
            key=lambda w: view_helpers.COMMON_NAMES.get(
                w[0]["species"], w[0]["species"]
            ).lower()
        )
    else:
        visit_windows.sort(key=lambda w: w[0]["end_time"], reverse=True)

    return render_template(
        "subgallery.html",
        current_path=f"/gallery/{date}",
        date=date,
        page=page,
        total_pages=total_pages,
        total_items=total_items,
        total_items_unfiltered=total_obs_unfiltered,
        sort_by=sort_by,
        current_threshold=current_threshold,
        observations=enriched_observations,
        visit_windows=visit_windows,
        species_of_day=species_of_day,
        pagination_range=pagination_range,
        image_width=IMAGE_WIDTH,
        focus_observation_id=focus_observation_id,
        focus_detection_id=focus_id_param,
    )


gallery_bp.add_url_rule(
    "/api/daily_species_summary",
    endpoint="daily_species_summary",
    view_func=daily_species_summary_route,
    methods=["GET"],
)
gallery_bp.add_url_rule(
    "/edit/<date_iso>",
    endpoint="edit_page",
    view_func=edit_route,
    methods=["GET"],
)
gallery_bp.add_url_rule(
    "/api/edit/actions",
    endpoint="edit_actions",
    view_func=edit_actions_route,
    methods=["POST"],
)
gallery_bp.add_url_rule(
    "/species",
    endpoint="species",
    view_func=species_route,
    methods=["GET"],
)
gallery_bp.add_url_rule(
    "/species/overview",
    endpoint="species_overview",
    view_func=species_overview_route,
    methods=["GET"],
)
gallery_bp.add_url_rule(
    "/gallery",
    endpoint="gallery",
    view_func=gallery_route,
    methods=["GET"],
)
gallery_bp.add_url_rule(
    "/gallery/<date>",
    endpoint="subgallery",
    view_func=subgallery_route,
    methods=["GET"],
)
