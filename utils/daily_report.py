"""
Evening Daily Report for WatchMyBirds.

Sends a structured Telegram update consisting of:
  A) A text status message (Telegram HTML, properly escaped).
  B) A photo album with the best image per species.

Usage:
    python -m utils.daily_report              # Report for today
    python -m utils.daily_report 2026-02-11   # Report for specific date
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Ensure repository root is importable even when executed as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Manual report runs should still send even when the general notification toggle
# is off; the endpoint already validates that credentials exist.
os.environ["TELEGRAM_ENABLED"] = "True"

from config import get_config
from core.db_core import (
    fetch_detections_for_gallery,
    get_connection,
)
from core.gallery_core import summarize_observations
from utils.image_ops import create_square_crop
from utils.path_manager import get_path_manager
from utils.telegram_notifier import send_telegram_media_group, send_telegram_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("daily_report")

_GERMAN_WEEKDAYS = {
    0: "Montag",
    1: "Dienstag",
    2: "Mittwoch",
    3: "Donnerstag",
    4: "Freitag",
    5: "Samstag",
    6: "Sonntag",
}


def _row_value(row, key: str, index: int, default=None):
    """Read values from sqlite rows or plain tuples without caring about shape."""
    if row is None:
        return default

    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        pass

    try:
        return row[index]
    except (KeyError, TypeError, IndexError):
        return default


def _load_common_names() -> dict[str, str]:
    """Load the configured common-name mapping with a DE fallback."""
    config = get_config()
    locale = str(config.get("SPECIES_COMMON_NAME_LOCALE", "DE") or "DE").upper()
    candidates = [REPO_ROOT / "assets" / f"common_names_{locale}.json"]
    if locale != "DE":
        candidates.append(REPO_ROOT / "assets" / "common_names_DE.json")

    for names_file in candidates:
        try:
            with open(names_file, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Could not load common names from %s: %s", names_file, exc)

    logger.warning("Could not load any common-names mapping.")
    return {}


def _resolve_dashboard_url() -> str:
    """Resolve the dashboard URL from config/env or fall back to local mDNS."""
    config = get_config()

    for key in ("WMB_PUBLIC_URL", "APP_PUBLIC_URL", "PUBLIC_BASE_URL"):
        raw = str(config.get(key, "") or os.environ.get(key, "")).strip()
        if raw:
            return raw.rstrip("/") + "/"

    http_port = config.get("HTTP_PORT", 8050)

    try:
        hostname = socket.gethostname().strip()
        if hostname:
            return f"http://{hostname}.local:{http_port}/"
    except Exception:
        pass

    return ""


def _html_escape(text: str) -> str:
    """Escape special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _device_name() -> str:
    """Return the configured device name (trimmed) or empty string."""
    try:
        raw = get_config().get("DEVICE_NAME", "") or ""
    except Exception:
        return ""
    return str(raw).strip()


def _device_html_prefix() -> str:
    """Telegram HTML prefix to prepend to captions, e.g. '[Front-Tree-View] '."""
    name = _device_name()
    if not name:
        return ""
    return f"[{_html_escape(name)}] "


def _humanize_species_name(name: str) -> str:
    """Turn internal species identifiers into readable labels."""
    if not name:
        return "—"
    return str(name).replace("_", " ")


def _format_report_date(report_date: str) -> str:
    """Render ISO date strings in a compact, chat-friendly German format."""
    try:
        parsed = datetime.date.fromisoformat(report_date)
    except ValueError:
        return report_date

    weekday = _GERMAN_WEEKDAYS.get(parsed.weekday())
    if weekday:
        return f"{weekday}, {parsed:%d.%m.%Y}"
    return parsed.strftime("%d.%m.%Y")


def _count_label(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _render_ingest_status(ingest_health: dict | None) -> tuple[str, list[str]]:
    """Derive a presentation-ready status indicator from ingest health."""
    if ingest_health is None:
        return "", ["Status unbekannt (manuell gestarteter Bericht)"]

    state = ingest_health.get("stream_state", "unknown")
    grace = ingest_health.get("startup_grace_active", False)
    frame_age = ingest_health.get("latest_frame_age_sec", -1.0)
    audio_open = ingest_health.get("audio_circuit_open", False)

    if state == "online" and not audio_open:
        return "", ["System aktiv", "Video online", "Audio in Ordnung"]

    parts: list[str] = []
    if state == "online":
        parts.append("Video online")
    elif state == "starting":
        parts.append("Video startet noch")
    elif state == "degraded":
        age_str = f"{frame_age:.0f}s" if frame_age >= 0 else "unbekannt"
        suffix = "" if grace else ", Startphase beendet"
        parts.append(f"Video-Import gestört, seit {age_str} keine Frames{suffix}")
    else:
        parts.append("Video-Import offline, keine Frames empfangen")

    if audio_open:
        parts.append("Audio-Schutzschalter offen")

    return "", parts


def render_species_photo_caption(common_name: str, count: int) -> str:
    """Build a short, polished Telegram caption for the species album."""
    safe_name = _html_escape(common_name)
    count_label = _count_label(count, "Sichtung", "Sichtungen")
    return f"<b>{safe_name}</b>\n{count} {count_label} heute · bestes Foto des Tages"


def _fetch_species_best_photos(conn, date_iso: str) -> list[dict]:
    """
    Fetch the best photo per species for a given date.

    Species resolution uses effective_species_sql() so non-bird OD class
    names (squirrel, cat, marten_mustelid, hedgehog) appear as their own
    species and bird detections without CLS collapse into
    UNKNOWN_SPECIES_KEY instead of leaking as 'bird'. This keys the
    daily report consistently with summarize_observations() across the
    rest of the analytics stack.

    The query restricts to ``decision_state = 'confirmed'`` so the
    Telegram report cannot surface low-evidence species the gallery
    already hides (the gallery applies the same filter via
    ``_gallery_visibility_sql``). Catalog-orphan species
    (e.g. classifier genus-fallbacks like ``Phoenicurus_sp.``) are
    dropped in the Python pass below using ``is_known_species``.

    Returns a list of dicts sorted by count DESC, score DESC.
    """
    from utils.db.detections import effective_species_sql
    from utils.species_names import UNKNOWN_SPECIES_KEY, is_known_species

    date_prefix = date_iso.replace("-", "")

    # Use effective_species_sql("d") and match via the outer grouping key
    # (d.species == d2.species) instead of raw CLS comparison — this keeps
    # the manual-override / CLS top1 / normalized-OD priority chain in sync
    # across the whole query.
    query = f"""
        WITH effective AS (
            SELECT
                d.detection_id,
                d.image_filename,
                d.score,
                d.bbox_quality,
                d.aesthetic_score,
                d.bbox_x,
                d.bbox_y,
                d.bbox_w,
                d.bbox_h,
                {effective_species_sql("d")} AS species
            FROM detections d
            WHERE d.image_filename LIKE ? || '%'
              AND d.status = 'active'
              AND lower(COALESCE(d.decision_state, '')) = 'confirmed'
              AND (
                  d.decision_level IS NULL
                  OR lower(d.decision_level) != 'reject'
              )
        )
        -- "Best photo" ranking: prefer the nightly aesthetic_score from
        -- scripts/aesthetic_tag_nightly.py, then detector confidence, then
        -- bbox-quality heuristic. NULL aesthetic_score (legacy / non-taggable
        -- species) sinks behind anything scored via the COALESCE(..., -1).
        SELECT
            species,
            COUNT(detection_id) AS count,
            (SELECT image_filename FROM effective e2
             WHERE e2.species = effective.species
             ORDER BY COALESCE(e2.aesthetic_score, -1) DESC, e2.score DESC, e2.bbox_quality DESC LIMIT 1) AS best_image_filename,
            (SELECT bbox_x FROM effective e2
             WHERE e2.species = effective.species
             ORDER BY COALESCE(e2.aesthetic_score, -1) DESC, e2.score DESC, e2.bbox_quality DESC LIMIT 1) AS best_bbox_x,
            (SELECT bbox_y FROM effective e2
             WHERE e2.species = effective.species
             ORDER BY COALESCE(e2.aesthetic_score, -1) DESC, e2.score DESC, e2.bbox_quality DESC LIMIT 1) AS best_bbox_y,
            (SELECT bbox_w FROM effective e2
             WHERE e2.species = effective.species
             ORDER BY COALESCE(e2.aesthetic_score, -1) DESC, e2.score DESC, e2.bbox_quality DESC LIMIT 1) AS best_bbox_w,
            (SELECT bbox_h FROM effective e2
             WHERE e2.species = effective.species
             ORDER BY COALESCE(e2.aesthetic_score, -1) DESC, e2.score DESC, e2.bbox_quality DESC LIMIT 1) AS best_bbox_h,
            MAX(score) AS best_score
        FROM effective
        WHERE species != '{UNKNOWN_SPECIES_KEY}'
        GROUP BY species
        ORDER BY count DESC, best_score DESC;
    """

    cur = conn.execute(query, (date_prefix,))
    rows = cur.fetchall()

    config = get_config()
    pm = get_path_manager(config.get("OUTPUT_DIR"))
    locale = str(config.get("SPECIES_COMMON_NAME_LOCALE", "DE") or "DE").upper()
    try:
        min_observations = int(config.get("TELEGRAM_MIN_CONFIRMED_OBSERVATIONS", 1))
    except (TypeError, ValueError):
        min_observations = 1
    min_observations = max(1, min_observations)

    results = []
    for row in rows:
        species = _row_value(row, "species", 0, "Unclassified")
        count = int(_row_value(row, "count", 1, 0) or 0)
        image_filename = _row_value(row, "best_image_filename", 2)
        bbox_x = _row_value(row, "best_bbox_x", 3)
        bbox_y = _row_value(row, "best_bbox_y", 4)
        bbox_w = _row_value(row, "best_bbox_w", 5)
        bbox_h = _row_value(row, "best_bbox_h", 6)
        score = float(_row_value(row, "best_score", 7, 0.0) or 0.0)

        if not image_filename:
            continue

        if not is_known_species(species, locale=locale):
            logger.debug("Dropping catalog-orphan species from report: %r", species)
            continue

        if count < min_observations:
            logger.debug(
                "Dropping low-evidence species %r (count=%d, threshold=%d)",
                species, count, min_observations,
            )
            continue

        photo_path = str(pm.get_original_path(image_filename))
        if not os.path.isfile(photo_path):
            logger.debug("Best photo not found on disk: %s", photo_path)
            continue

        results.append(
            {
                "species": species,
                "count": count,
                "best_photo_path": photo_path,
                "score": score,
                "image_filename": image_filename,
                "bbox_x": float(bbox_x) if bbox_x is not None else None,
                "bbox_y": float(bbox_y) if bbox_y is not None else None,
                "bbox_w": float(bbox_w) if bbox_w is not None else None,
                "bbox_h": float(bbox_h) if bbox_h is not None else None,
            }
        )

    return results


def _truncate_label(text: str, max_len: int = 28) -> str:
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _tile_with_footer(
    image: np.ndarray,
    width: int,
    height: int,
    title: str,
    subtitle: str,
    bg_color: tuple[int, int, int] = (20, 24, 31),
) -> np.ndarray:
    from utils.image_text import draw_text

    footer_h = 64
    media_h = max(40, height - footer_h)
    tile = np.full((height, width, 3), bg_color, dtype=np.uint8)
    fitted = _resize_cover(image, width, media_h)
    tile[:media_h, :width] = fitted
    cv2.rectangle(tile, (0, media_h), (width, height), (14, 17, 23), thickness=-1)
    draw_text(
        tile,
        _truncate_label(title, 24),
        (18, media_h + 10),
        size=20,
        color=(245, 247, 250),
        bold=True,
    )
    draw_text(
        tile,
        _truncate_label(subtitle, 32),
        (18, media_h + 38),
        size=15,
        color=(171, 179, 189),
    )
    return tile


def _resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(
        image,
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y1 = max(0, (resized.shape[0] - height) // 2)
    x1 = max(0, (resized.shape[1] - width) // 2)
    return resized[y1 : y1 + height, x1 : x1 + width]


def _resize_contain(
    image: np.ndarray,
    width: int,
    height: int,
    bg_color: tuple[int, int, int] = (20, 24, 31),
) -> np.ndarray:
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return canvas
    scale = min(width / src_w, height / src_h)
    resized = cv2.resize(
        image,
        (max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y1 = max(0, (height - resized.shape[0]) // 2)
    x1 = max(0, (width - resized.shape[1]) // 2)
    canvas[y1 : y1 + resized.shape[0], x1 : x1 + resized.shape[1]] = resized
    return canvas


def _resolve_bbox_pixels(photo: dict, image: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox_x = photo.get("bbox_x")
    bbox_y = photo.get("bbox_y")
    bbox_w = photo.get("bbox_w")
    bbox_h = photo.get("bbox_h")
    if None in (bbox_x, bbox_y, bbox_w, bbox_h):
        return None

    img_h, img_w = image.shape[:2]
    x1 = int(max(0, min(img_w - 1, round(float(bbox_x) * img_w))))
    y1 = int(max(0, min(img_h - 1, round(float(bbox_y) * img_h))))
    x2 = int(max(x1 + 1, min(img_w, round((float(bbox_x) + float(bbox_w)) * img_w))))
    y2 = int(max(y1 + 1, min(img_h, round((float(bbox_y) + float(bbox_h)) * img_h))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _prepare_species_visual(photo: dict, common_names: dict[str, str]) -> dict | None:
    image = cv2.imread(photo["best_photo_path"])
    if image is None:
        logger.warning("Could not load report photo: %s", photo["best_photo_path"])
        return None

    scientific = str(photo["species"] or "—")
    common = common_names.get(scientific, _humanize_species_name(scientific))
    bbox = _resolve_bbox_pixels(photo, image)
    if bbox is not None:
        crops = {
            "tight": create_square_crop(
                image, bbox, margin_percent=0.35, pad_color=(18, 18, 18)
            ),
            "medium": create_square_crop(
                image, bbox, margin_percent=0.72, pad_color=(18, 18, 18)
            ),
            "wide": create_square_crop(
                image, bbox, margin_percent=1.08, pad_color=(18, 18, 18)
            ),
        }
    else:
        fallback = _resize_cover(image, 720, 720)
        crops = {"tight": fallback, "medium": fallback, "wide": fallback}

    return {
        "scientific": scientific,
        "common_name": common,
        "count": int(photo.get("count", 0) or 0),
        "full_image": image,
        "crop_images": crops,
    }


def _variant_output_dir(report_date: str, output_dir: str | Path | None = None) -> Path:
    base_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir()) / "watchmybirds_report_variants"
    path = base_dir / report_date
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_variant_image(canvas: np.ndarray, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(path)


def _build_zoom_collage_variant(species_visuals: list[dict], report_date: str, output_dir: Path) -> dict | None:
    picks = species_visuals[:4]
    if not picks:
        return None

    tile_w = 520
    tile_h = 520
    header_h = 120
    footer_h = 28
    gap = 18
    cols = 2
    rows = max(1, int(np.ceil(len(picks) / cols)))
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = header_h + rows * tile_h + (rows + 1) * gap + footer_h
    canvas = np.full((canvas_h, canvas_w, 3), (10, 12, 16), dtype=np.uint8)
    cv2.putText(canvas, "Variante A  Zoom-Collage", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (244, 246, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, _format_report_date(report_date), (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (170, 177, 186), 1, cv2.LINE_AA)

    for idx, visual in enumerate(picks):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = header_h + gap + row * (tile_h + gap)
        tile = _tile_with_footer(
            _resize_cover(visual["crop_images"]["tight"], tile_w, tile_h - 64),
            tile_w,
            tile_h,
            visual["common_name"],
            f'{visual["count"]} Sichtungen · enger Crop',
        )
        canvas[y : y + tile_h, x : x + tile_w] = tile

    path = _save_variant_image(canvas, output_dir, "variant_a_zoom_collage.jpg")
    return {
        "name": "Variante A · Zoom-Collage",
        "photo_path": path,
        "caption": "<b>Variante A · Zoom-Collage</b>\nEnge Crops mit starkem Fokus auf den Vogel.",
    }


def _build_compare_variant(species_visuals: list[dict], report_date: str, output_dir: Path) -> dict | None:
    picks = species_visuals[:3]
    if not picks:
        return None

    row_h = 280
    left_w = 560
    right_w = 280
    gap = 18
    header_h = 120
    canvas_w = left_w + right_w + gap * 3
    canvas_h = header_h + len(picks) * (row_h + gap) + gap
    canvas = np.full((canvas_h, canvas_w, 3), (13, 17, 22), dtype=np.uint8)
    cv2.putText(canvas, "Variante B  Vollbild + Crop", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.96, (244, 246, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, _format_report_date(report_date), (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (170, 177, 186), 1, cv2.LINE_AA)

    for idx, visual in enumerate(picks):
        y = header_h + gap + idx * (row_h + gap)
        left = _tile_with_footer(
            _resize_contain(visual["full_image"], left_w, row_h - 58),
            left_w,
            row_h,
            visual["common_name"],
            "Vollbild",
        )
        right = _tile_with_footer(
            _resize_cover(visual["crop_images"]["medium"], right_w, row_h - 58),
            right_w,
            row_h,
            visual["common_name"],
            "Mittel-Crop",
        )
        canvas[y : y + row_h, gap : gap + left_w] = left
        x_right = gap * 2 + left_w
        canvas[y : y + row_h, x_right : x_right + right_w] = right

    path = _save_variant_image(canvas, output_dir, "variant_b_full_plus_crop.jpg")
    return {
        "name": "Variante B · Vollbild plus Crop",
        "photo_path": path,
        "caption": "<b>Variante B · Vollbild plus Crop</b>\nLinks die Szene, rechts der gezoomte Vogel.",
    }


def _build_story_strip_variant(species_visuals: list[dict], report_date: str, output_dir: Path) -> dict | None:
    picks = species_visuals[:3]
    if not picks:
        return None

    card_w = 320
    card_h = 430
    gap = 18
    header_h = 120
    canvas_w = len(picks) * card_w + (len(picks) + 1) * gap
    canvas_h = header_h + card_h + gap
    canvas = np.full((canvas_h, canvas_w, 3), (11, 14, 19), dtype=np.uint8)
    cv2.putText(canvas, "Variante C  Crop-Story Board", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.96, (244, 246, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, _format_report_date(report_date), (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (170, 177, 186), 1, cv2.LINE_AA)

    for idx, visual in enumerate(picks):
        x = gap + idx * (card_w + gap)
        y = header_h
        top = _resize_contain(visual["full_image"], card_w, 170, bg_color=(18, 21, 27))
        bottom = _resize_cover(visual["crop_images"]["wide"], card_w, 196)
        card = np.full((card_h, card_w, 3), (18, 21, 27), dtype=np.uint8)
        card[:170, :card_w] = top
        card[170:366, :card_w] = bottom
        cv2.rectangle(card, (0, 366), (card_w, card_h), (12, 15, 20), thickness=-1)
        cv2.putText(card, _truncate_label(visual["common_name"], 23), (18, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 247, 250), 1, cv2.LINE_AA)
        cv2.putText(card, f'{visual["count"]} Sichtungen heute', (18, 417), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (171, 179, 189), 1, cv2.LINE_AA)
        canvas[y : y + card_h, x : x + card_w] = card

    path = _save_variant_image(canvas, output_dir, "variant_c_story_board.jpg")
    return {
        "name": "Variante C · Crop-Story Board",
        "photo_path": path,
        "caption": "<b>Variante C · Crop-Story Board</b>\nKarten-Layout mit Szene oben und Fokus-Crop darunter.",
    }


def _build_triplet_zoom_variant(
    species_visuals: list[dict], report_date: str, output_dir: Path
) -> dict | None:
    picks = species_visuals[:2]
    if not picks:
        return None

    card_w = 520
    card_h = 660
    row_h = 176
    gap = 18
    header_h = 112
    canvas_w = len(picks) * card_w + (len(picks) + 1) * gap
    canvas_h = header_h + card_h + gap
    canvas = np.full((canvas_h, canvas_w, 3), (9, 12, 16), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Variante D  Drei Zoom-Stufen",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.94,
        (244, 246, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        _format_report_date(report_date),
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (170, 177, 186),
        1,
        cv2.LINE_AA,
    )

    crop_keys = [("tight", "Eng"), ("medium", "Mittel"), ("wide", "Weit")]
    for idx, visual in enumerate(picks):
        x = gap + idx * (card_w + gap)
        y = header_h
        card = np.full((card_h, card_w, 3), (16, 20, 26), dtype=np.uint8)
        cv2.rectangle(card, (0, 0), (card_w, 64), (12, 16, 21), thickness=-1)
        cv2.putText(
            card,
            _truncate_label(visual["common_name"], 26),
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            card,
            f'{visual["count"]} Sichtungen',
            (18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (171, 179, 189),
            1,
            cv2.LINE_AA,
        )
        for row, (key, label) in enumerate(crop_keys):
            y1 = 74 + row * (row_h + 10)
            strip = _resize_cover(visual["crop_images"][key], card_w - 20, row_h)
            card[y1 : y1 + row_h, 10 : 10 + (card_w - 20)] = strip
            cv2.rectangle(card, (18, y1 + 12), (88, y1 + 40), (14, 17, 23), thickness=-1)
            cv2.putText(
                card,
                label,
                (28, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 234, 240),
                1,
                cv2.LINE_AA,
            )
        canvas[y : y + card_h, x : x + card_w] = card

    path = _save_variant_image(canvas, output_dir, "variant_d_three_zoom_levels.jpg")
    return {
        "name": "Variante D · Drei Zoom-Stufen",
        "photo_path": path,
        "caption": "<b>Variante D · Drei Zoom-Stufen</b>\nDirekter Vergleich von engem, mittlerem und weitem Crop.",
    }


def _build_wide_context_variant(
    species_visuals: list[dict], report_date: str, output_dir: Path
) -> dict | None:
    picks = species_visuals[:6]
    if not picks:
        return None

    tile_w = 330
    tile_h = 286
    cols = 3
    rows = max(1, int(np.ceil(len(picks) / cols)))
    gap = 16
    header_h = 112
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = header_h + rows * tile_h + (rows + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), (11, 14, 19), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Variante E  Weite Kontext-Collage",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (244, 246, 248),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        _format_report_date(report_date),
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (170, 177, 186),
        1,
        cv2.LINE_AA,
    )

    for idx, visual in enumerate(picks):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = header_h + gap + row * (tile_h + gap)
        tile = _tile_with_footer(
            _resize_cover(visual["crop_images"]["wide"], tile_w, tile_h - 64),
            tile_w,
            tile_h,
            visual["common_name"],
            "Weiter Crop mit mehr Umfeld",
        )
        canvas[y : y + tile_h, x : x + tile_w] = tile

    path = _save_variant_image(canvas, output_dir, "variant_e_wide_context_collage.jpg")
    return {
        "name": "Variante E · Weite Kontext-Collage",
        "photo_path": path,
        "caption": "<b>Variante E · Weite Kontext-Collage</b>\nMehr Umfeld pro Bild, weniger enger Zoom.",
    }


def build_report_collage(
    species_visuals: list[dict], report_date: str, output_dir: Path
) -> dict | None:
    """Production collage: E-style 3-column grid with A-style medium crops.

    Header layout:
      - Left: device name in bold (or "WatchMyBirds" when no device set)
      - Below it: report date in a smaller, muted weight
      - Right (top-aligned): species count summary

    All header text is rendered with Pillow so umlauts and the German
    date format's narrow-no-break-space render correctly. Same goes for
    the per-tile species name + count footers (handled inside
    ``_tile_with_footer``).
    """
    from utils.image_text import draw_text, measure_text

    picks = species_visuals[:6]
    if not picks:
        return None

    tile_w = 330
    tile_h = 286
    cols = 3
    rows = max(1, int(np.ceil(len(picks) / cols)))
    gap = 16
    header_h = 100
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = header_h + rows * tile_h + (rows + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), (11, 14, 19), dtype=np.uint8)

    device = _device_name()
    title = device if device else "WatchMyBirds"
    draw_text(
        canvas,
        title,
        (24, 22),
        size=26,
        color=(244, 246, 248),
        bold=True,
    )
    draw_text(
        canvas,
        _format_report_date(report_date),
        (24, 58),
        size=17,
        color=(170, 177, 186),
    )

    species_summary = f"{len(picks)} {_count_label(len(picks), 'Art', 'Arten')}"
    summary_w, _ = measure_text(species_summary, size=17)
    draw_text(
        canvas,
        species_summary,
        (canvas_w - summary_w - 24, 58),
        size=17,
        color=(170, 177, 186),
    )

    for idx, visual in enumerate(picks):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = header_h + gap + row * (tile_h + gap)
        count = int(visual["count"])
        sighting_label = _count_label(count, "Sichtung", "Sichtungen")
        tile = _tile_with_footer(
            _resize_cover(visual["crop_images"]["medium"], tile_w, tile_h - 64),
            tile_w,
            tile_h,
            visual["common_name"],
            f"{count} {sighting_label}",
        )
        canvas[y : y + tile_h, x : x + tile_w] = tile

    path = _save_variant_image(canvas, output_dir, "report_collage.jpg")
    return {
        "photo_path": path,
        "caption": f"{_device_html_prefix()}<b>Abendbericht {_html_escape(_format_report_date(report_date))}</b>",
    }


def _draw_corner_chip(
    canvas: np.ndarray,
    text: str,
    *,
    margin: int = 10,
    pad_x: int = 8,
    pad_y: int = 4,
    text_size: int = 13,
    bg_alpha: float = 0.55,
) -> None:
    """Bake a small dark pill in the top-right of *canvas* with *text*.

    Used for the discreet device + date overlay on each mobile tile. The
    pill is alpha-blended onto the underlying photo so the image stays
    visible but the text remains readable on any background.
    """
    from utils.image_text import draw_text, measure_text

    if not text:
        return

    text_w, text_h = measure_text(text, size=text_size)
    pill_w = text_w + pad_x * 2
    pill_h = text_h + pad_y * 2 + 2  # extra 2px for descender breathing room
    img_h, img_w = canvas.shape[:2]
    x2 = img_w - margin
    x1 = max(0, x2 - pill_w)
    y1 = margin
    y2 = min(img_h, y1 + pill_h)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 10, 14), thickness=-1)
    cv2.addWeighted(overlay, bg_alpha, canvas, 1 - bg_alpha, 0, dst=canvas)

    # Pillow draws crisp Unicode (umlauts, the narrow-no-break-space inside
    # the formatted German date) — cv2.putText would render those as `??`.
    draw_text(
        canvas,
        text,
        (x1 + pad_x, y1 + pad_y),
        size=text_size,
        color=(232, 236, 242),
    )


def build_report_mobile_tiles(
    species_visuals: list[dict], report_date: str, output_dir: Path
) -> list[dict]:
    """Render one 1080x1080 'achievement card' per species for the album,
    plus a final 'collector card' summarising the day.

    Each per-species card is post-ready: full-bleed photo with a neon
    rim glow, a score badge in the top-right ("3x HEUTE"), and a banner
    along the bottom with the species common name plus device + date.
    The neon colour comes from the same per-species slot system the
    Review and Gallery surfaces use (``core.species_colours``) so
    identity stays visually consistent across the app — the operator's
    "Blaumeise blue" is the same blue everywhere.

    The closing collector card lists every species seen today as small
    photo vignettes with their slot colours and counts. It rides at the
    end of the Telegram media group so the album reads as a
    deck-of-cards-plus-checklist.
    """
    from utils.achievement_card import (
        build_species_colour_map,
        neon_for_species,
        render_achievement_card,
        render_collector_card,
    )

    picks = species_visuals[:6]
    if not picks:
        return []

    device = _device_name()
    date_label = _format_report_date(report_date)
    colour_map = build_species_colour_map(
        [v.get("scientific") or v.get("common_name", "") for v in picks]
    )

    tiles: list[dict] = []
    # Hold (scientific, common_name, count, photo) for the collector
    # card so we don't reload the JPEG from disk.
    collector_entries: list[dict] = []

    for idx, visual in enumerate(picks):
        count = int(visual["count"])
        sighting_label = _count_label(count, "Sichtung", "Sichtungen")
        scientific = str(visual.get("scientific") or visual.get("common_name", ""))
        rim_color, glow_color = neon_for_species(scientific, colour_map)

        # The medium crop already has the bird centred — give the card
        # the best framing for the bird subject. Fall back to the full
        # image when no bbox-derived crop is available.
        crops = visual.get("crop_images") or {}
        photo_source = crops.get("medium")
        if photo_source is None:
            photo_source = visual.get("full_image")
        if photo_source is None:
            continue

        card = render_achievement_card(
            photo_source,
            common_name=visual["common_name"],
            count=count,
            rim_color=rim_color,
            glow_color=glow_color,
            device_label=device,
            date_label=date_label,
        )

        filename = f"report_mobile_{idx + 1:02d}.jpg"
        path = _save_variant_image(card, output_dir, filename)
        caption = (
            f"{_device_html_prefix()}<b>{_html_escape(visual['common_name'])}</b> "
            f"· {count} {sighting_label}"
        )
        tiles.append({"photo_path": path, "caption": caption})

        collector_entries.append(
            {
                "scientific": scientific,
                "common_name": visual["common_name"],
                "count": count,
                "photo": photo_source,
            }
        )

    # Final collector card — reuses the same colour map + photos so each
    # vignette matches its standalone card earlier in the album.
    if collector_entries:
        # Set code for the footer: ISO date + roster size, e.g.
        # "30.04.2026  ·  6 / 6  COLLECTED". The renderer auto-fills
        # the "/N COLLECTED" suffix when only the date prefix is given.
        try:
            iso_short = datetime.date.fromisoformat(report_date).strftime(
                "%d.%m.%Y"
            )
        except ValueError:
            iso_short = report_date
        set_code = f"{iso_short}   ·   {len(collector_entries)} / {len(collector_entries)}   COLLECTED"

        collector_card = render_collector_card(
            collector_entries,
            colour_map=colour_map,
            device_label=device,
            date_label=date_label,
            set_code=set_code,
        )
        collector_path = _save_variant_image(
            collector_card, output_dir, "report_mobile_99_collector.jpg"
        )
        collector_caption = (
            f"{_device_html_prefix()}<b>Sammelkarte</b> · "
            f"{len(collector_entries)} {_count_label(len(collector_entries), 'Art', 'Arten')} heute"
        )
        tiles.append(
            {"photo_path": collector_path, "caption": collector_caption}
        )

    return tiles


def build_report_mobile_album(
    species_photos: list[dict],
    common_names: dict[str, str],
    report_date: str,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Build the per-species mobile tiles for the evening report."""
    species_visuals = []
    for photo in species_photos:
        visual = _prepare_species_visual(photo, common_names)
        if visual is not None:
            species_visuals.append(visual)

    if not species_visuals:
        return []

    variant_dir = _variant_output_dir(report_date, output_dir=output_dir)
    return build_report_mobile_tiles(species_visuals, report_date, variant_dir)


def build_report_variant_previews(
    species_photos: list[dict],
    common_names: dict[str, str],
    report_date: str,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Render local report preview variants and return the generated files."""
    species_visuals = []
    for photo in species_photos:
        visual = _prepare_species_visual(photo, common_names)
        if visual is not None:
            species_visuals.append(visual)

    if not species_visuals:
        return []

    variant_dir = _variant_output_dir(report_date, output_dir=output_dir)
    variants = []
    for builder in (
        _build_zoom_collage_variant,
        _build_compare_variant,
        _build_story_strip_variant,
        _build_triplet_zoom_variant,
        _build_wide_context_variant,
    ):
        variant = builder(species_visuals, report_date, variant_dir)
        if variant is not None:
            variants.append(variant)
    return variants


def build_production_collage(
    species_photos: list[dict],
    common_names: dict[str, str],
    report_date: str,
    output_dir: str | Path | None = None,
) -> dict | None:
    """Build the single production collage for the evening report."""
    species_visuals = []
    for photo in species_photos:
        visual = _prepare_species_visual(photo, common_names)
        if visual is not None:
            species_visuals.append(visual)

    if not species_visuals:
        return None

    variant_dir = _variant_output_dir(report_date, output_dir=output_dir)
    return build_report_collage(species_visuals, report_date, variant_dir)


def send_report_variant_previews(variants: list[dict]) -> list:
    """Send locally rendered preview variants so one can be selected later."""
    if not variants:
        return []

    responses = []
    intro = (
        "<b>Abendbericht Varianten-Test</b>\n"
        "Ich schicke mehrere lokal gerenderte Bildvarianten. Danach verwenden wir nur eine davon."
    )
    responses.append(send_telegram_message(intro, parse_mode="HTML"))

    for variant in variants:
        responses.append(
            send_telegram_message(
                variant["caption"],
                photo_path=variant["photo_path"],
                parse_mode="HTML",
            )
        )

    return responses


def _report_title_for_mode() -> str:
    """Return the report title based on the configured Telegram mode."""
    try:
        cfg = get_config()
    except Exception:
        return "Abendbericht"
    mode = str(cfg.get("TELEGRAM_MODE", "off") or "off").strip().lower()
    if mode == "interval":
        try:
            hours = int(float(cfg.get("TELEGRAM_REPORT_INTERVAL_HOURS", 1)))
        except Exception:
            hours = 1
        hours = max(1, min(24, hours))
        if hours == 1:
            return "Stündlicher Bericht"
        return f"Zwischenbericht ({hours}h)"
    # "daily", "off" (manual send), "live" (manual send) -> evening-style title.
    return "Abendbericht"


def render_text_report(
    report_date: str,
    total_events: int,
    species_count: int,
    top_species_name: str,
    top_species_count: int,
) -> str:
    """Render the report header + summary as valid Telegram HTML."""
    lines: list[str] = []

    title = _report_title_for_mode()
    lines.append(
        f"{_device_html_prefix()}<b>WatchMyBirds · {_html_escape(title)} — "
        f"{_html_escape(_format_report_date(report_date))}</b>"
    )
    lines.append("")
    event_label = _count_label(total_events, "Event", "Events")
    species_label = _count_label(species_count, "Art", "Arten")
    lines.append(f"<b>{total_events}</b> {event_label}, <b>{species_count}</b> {species_label}.")
    if top_species_name and top_species_name != "—" and top_species_count > 0:
        top_label = _html_escape(_humanize_species_name(top_species_name))
        lines.append(f"Häufigste Art: <b>{top_label}</b> ({top_species_count}x).")
    else:
        lines.append("Keine Arten erkannt.")

    return "\n".join(lines)


def send_species_best_photos_album(
    species_photos: list[dict], common_names: dict[str, str]
) -> list:
    """Send the best-of-day photos as Telegram media groups."""
    if not species_photos:
        return []

    media_items = []
    for sp in species_photos:
        scientific = sp["species"]
        common = common_names.get(scientific, _humanize_species_name(scientific))
        media_items.append(
            {
                "photo_path": sp["best_photo_path"],
                "caption": render_species_photo_caption(common, int(sp["count"])),
            }
        )

    all_responses = []
    for i in range(0, len(media_items), 10):
        chunk = media_items[i : i + 10]
        responses = send_telegram_media_group(chunk)
        if responses:
            all_responses.extend(responses)

    return all_responses


def main(**_kwargs):
    """Generate and send the evening Telegram report."""
    conn = get_connection()

    if len(sys.argv) > 1:
        report_date = sys.argv[1]
    else:
        report_date = datetime.date.today().isoformat()

    logger.info("Generating evening report for %s", report_date)

    try:
        config = get_config()
        gallery_threshold = float(config.get("GALLERY_DISPLAY_THRESHOLD", 0.0))

        today_rows = [
            dict(row)
            for row in fetch_detections_for_gallery(
                conn, report_date, order_by="time"
            )
        ]
        obs_summary = summarize_observations(
            today_rows, min_score=gallery_threshold
        )
        obs_stats = obs_summary["summary"]
        total_events = obs_stats["total_observations"]
        species_counts: dict[str, int] = obs_stats["species_counts"]

        common_names = _load_common_names()

        if species_counts:
            top_scientific = max(species_counts, key=species_counts.get)
            top_species_count = species_counts[top_scientific]
            top_species_name = common_names.get(
                top_scientific, _humanize_species_name(top_scientific)
            )
        else:
            top_species_name = "—"
            top_species_count = 0

        all_species_photos = _fetch_species_best_photos(conn, report_date)
        # Filter to species that passed the observation threshold, sorted by
        # visit count descending (same ranking the live page uses).
        # Override raw detection counts with observation-based visit counts.
        species_photos = []
        for sp in all_species_photos:
            if sp["species"] in species_counts:
                sp["count"] = species_counts[sp["species"]]
                species_photos.append(sp)
        species_photos.sort(
            key=lambda sp: sp["count"], reverse=True
        )

        text_message = render_text_report(
            report_date=report_date,
            total_events=total_events,
            species_count=len(species_counts),
            top_species_name=top_species_name,
            top_species_count=top_species_count,
        )

        logger.info("Sending text report via Telegram...")
        text_responses = send_telegram_message(text_message, parse_mode="HTML")
        logger.info("Text report sent. Responses: %s", text_responses)

        if species_photos:
            logger.info("Building mobile-friendly per-species album...")
            mobile_tiles = build_report_mobile_album(
                species_photos,
                common_names,
                report_date=report_date,
            )
            if mobile_tiles:
                logger.info(
                    "Sending %d mobile tile(s) via Telegram media group...",
                    len(mobile_tiles),
                )
                # Telegram media groups are capped at 10 items per request.
                for i in range(0, len(mobile_tiles), 10):
                    chunk = mobile_tiles[i : i + 10]
                    album_response = send_telegram_media_group(chunk)
                    logger.info(
                        "Mobile album chunk %d sent. Response: %s",
                        i // 10 + 1,
                        album_response,
                    )
            else:
                logger.warning("Mobile album build returned no tiles.")
        else:
            logger.info("No species photos to send.")

        logger.info("--- Example Text Output ---")
        logger.info("\n%s", text_message)

    except Exception as exc:
        logger.error("Failed to generate report: %s", exc, exc_info=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
