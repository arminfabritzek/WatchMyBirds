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
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import sqlite3
    from typing import Any

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
from utils.cover_selection import cover_order_sql
from utils.image_ops import create_square_crop
from utils.path_manager import PathManager, get_path_manager
from utils.telegram_notifier import send_telegram_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("daily_report")

_WEEKDAYS_EN = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _row_value(row: Any, key: str, index: int, default: Any = None) -> Any:
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


# Suffix added to genus-only fallbacks (e.g. ``Phoenicurus_sp.``) when the
# JSON mapping doesn't contain an explicit entry. Per locale so the report
# stays in the operator's language.
_GENUS_FALLBACK_SUFFIX = {
    "DE": " (Art unklar)",
    "NO": " (art usikker)",
    "EN": " (species unclear)",
}


def _load_common_names() -> dict[str, str]:
    """Load the configured common-name mapping with a DE fallback.

    Layered: when locale != DE, NO/etc. is overlaid on top of the DE
    base. That mirrors ``utils.species_names.load_common_names`` and
    means a key missing from the locale file falls back to its DE name
    instead of the bare scientific name.
    """
    config = get_config()
    locale = str(config.get("SPECIES_COMMON_NAME_LOCALE", "DE") or "DE").upper()

    merged: dict[str, str] = {}
    base_path = REPO_ROOT / "assets" / "common_names_DE.json"
    overlay_path = REPO_ROOT / "assets" / f"common_names_{locale}.json"

    try:
        with open(base_path, encoding="utf-8") as f:
            merged = dict(json.load(f))
    except FileNotFoundError:
        logger.warning("Common names base file missing: %s", base_path)
    except Exception as exc:
        logger.warning("Could not load common names base from %s: %s", base_path, exc)

    if locale != "DE":
        try:
            with open(overlay_path, encoding="utf-8") as f:
                merged.update(json.load(f))
        except FileNotFoundError:
            logger.warning(
                "Common names overlay missing for %s: %s", locale, overlay_path
            )
        except Exception as exc:
            logger.warning(
                "Could not load common names overlay from %s: %s", overlay_path, exc
            )

    return merged


def _resolve_common_name(
    scientific: str, common_names: dict[str, str]
) -> tuple[str, bool]:
    """Resolve a CLS scientific name to ``(display, is_latin_fallback)``.

    Three-step resolution so genus-fallback labels (e.g. ``Phoenicurus_sp.``)
    don't degrade to bare Latin in the report:

    1. Direct lookup in the loaded common-names map → (name, False).
    2. For ``<Genus>_sp.`` keys not in the map: derive a localised
       "(Art unklar)" / "(art usikker)" / "(species unclear)" fallback
       from the genus name → (name, True). The genus part itself IS
       Latin, so the caller may want to italicise.
    3. Plain humanise — replace underscores with spaces → (name, True).
       Last resort, always Latin.

    The ``is_latin_fallback`` flag lets callers wrap the display string
    in italic markup (HTML ``<i>…</i>`` for Telegram captions, future
    italic font in the rendered collector card).
    """
    if not scientific:
        return "—", False
    direct = common_names.get(scientific)
    if direct:
        return direct, False
    if scientific.endswith("_sp."):
        try:
            cfg = get_config()
            locale = str(cfg.get("SPECIES_COMMON_NAME_LOCALE", "DE") or "DE").upper()
        except Exception:
            locale = "DE"
        suffix = _GENUS_FALLBACK_SUFFIX.get(locale, _GENUS_FALLBACK_SUFFIX["DE"])
        # ``Phoenicurus_sp.`` → ``Phoenicurus`` + suffix. The genus portion
        # is Latin; the suffix is the localised parenthetical. Caller
        # italics the whole string for consistency.
        genus = scientific[: -len("_sp.")].replace("_", " ").strip()
        if genus:
            return f"{genus}{suffix}", True
    return _humanize_species_name(scientific), True


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
    """Render ISO date strings in a compact, chat-friendly English format
    (e.g. "Saturday, 09.05.2026"). Day-month-year stays for European
    operators; only the weekday name shifts to English to match the rest
    of the report copy."""
    try:
        parsed = datetime.date.fromisoformat(report_date)
    except ValueError:
        return report_date

    weekday = _WEEKDAYS_EN.get(parsed.weekday())
    if weekday:
        return f"{weekday}, {parsed:%d.%m.%Y}"
    return parsed.strftime("%d.%m.%Y")


def _count_label(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _fetch_species_best_photos(conn: sqlite3.Connection, date_iso: str) -> list[dict]:
    """Rank daily photos from the same admitted observations as the UI."""

    date_prefix = date_iso.replace("-", "")

    config = get_config()

    query = f"""
        WITH effective AS (
            SELECT o.*,
                COALESCE(b.bbox_x, d.bbox_x, m.bbox_x) AS bbox_x,
                COALESCE(b.bbox_y, d.bbox_y, m.bbox_y) AS bbox_y,
                COALESCE(b.bbox_w, d.bbox_w, m.bbox_w) AS bbox_w,
                COALESCE(b.bbox_h, d.bbox_h, m.bbox_h) AS bbox_h
            FROM current_observations o
            LEFT JOIN detections d ON d.detection_id = o.detection_id
            LEFT JOIN manual_objects m ON m.manual_object_id = o.manual_object_id
            LEFT JOIN current_human_label_facts b ON b.detection_id = o.detection_id
                AND b.scope = 'object' AND b.fact_type = 'bbox_correction'
            WHERE o.timestamp LIKE ? || '%' AND o.admitted
        ), ranked AS (
            SELECT *, COUNT(*) OVER (PARTITION BY species_key) AS count,
                ROW_NUMBER() OVER (PARTITION BY species_key
                    ORDER BY {cover_order_sql()}, object_key) AS rn
            FROM effective
        )
        SELECT species_key AS species, count, image_filename AS best_image_filename,
            bbox_x AS best_bbox_x, bbox_y AS best_bbox_y,
            bbox_w AS best_bbox_w, bbox_h AS best_bbox_h, score AS best_score
        FROM ranked WHERE rn = 1 ORDER BY count DESC, best_score DESC
    """

    cur = conn.execute(query, (date_prefix,))
    rows = cur.fetchall()

    pm = get_path_manager(config.get("OUTPUT_DIR"))

    # NOTE: TELEGRAM_MIN_CONFIRMED_OBSERVATIONS is *not* applied here.
    # The raw `count` we read in this query is the per-detection row count
    # (every bbox in every frame), but the operator-facing report shows
    # *visit* counts (events from summarize_observations, which collapse
    # adjacent detections of the same species into one sighting). Applying
    # the threshold against the raw count would let species with
    # visit-count = 1 through whenever they had >threshold detections in
    # a single visit. The threshold is applied in main() against the
    # visit-count instead, so what the operator sees and what the filter
    # decides on are the same number.

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


def _resolve_bbox_pixels(
    photo: dict, image: np.ndarray
) -> tuple[int, int, int, int] | None:
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
    common, common_is_latin = _resolve_common_name(scientific, common_names)
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
        # True when ``common_name`` is itself a Latin/genus fallback
        # (no real common name available in the loaded mapping). The
        # collector renderer uses this to italicise the label text.
        "common_is_latin": common_is_latin,
        "count": int(photo.get("count", 0) or 0),
        "full_image": image,
        "crop_images": crops,
        # Pass through pixel bbox so downstream renderers (e.g. the
        # collector card with cell-aspect 4:3 vignettes) can do their
        # own bbox-aware cropping at any aspect ratio without going
        # through the padded square crop.
        "bbox_px": bbox,
    }


def _variant_output_dir(report_date: str, output_dir: str | Path | None = None) -> Path:
    base_dir = (
        Path(output_dir)
        if output_dir
        else PathManager(str(get_config()["OUTPUT_DIR"])).get_report_tmp_dir()
    )
    path = base_dir / report_date
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_variant_image(canvas: np.ndarray, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(path)


def build_report_mobile_tiles(
    species_visuals: list[dict],
    report_date: str,
    output_dir: Path,
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
        sighting_label = _count_label(count, "visit", "visits")
        scientific = str(visual.get("scientific") or visual.get("common_name", ""))
        rim_color, glow_color = neon_for_species(scientific, colour_map)

        # The achievement card (per-species 1080×1080 standalone card)
        # uses the bbox-centred medium crop so the bird is always
        # framed dead-centre.
        crops = visual.get("crop_images") or {}
        achievement_photo = crops.get("medium")
        if achievement_photo is None:
            achievement_photo = visual.get("full_image")
        if achievement_photo is None:
            continue

        card = render_achievement_card(
            achievement_photo,
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

        # Collector vignettes get the FULL image plus the pixel bbox.
        # The collector renderer does its own cell-aspect bbox-aware
        # crop so the bird fills the 4:3 cell without the black-band
        # artifact that the padded square crop produced when the
        # bbox sat near the camera-frame edge.
        collector_entries.append(
            {
                "scientific": scientific,
                "common_name": visual["common_name"],
                "count": count,
                "photo": visual.get("full_image"),
                "bbox_px": visual.get("bbox_px"),
                # Fallback: if there's no full_image / bbox we still
                # pass the medium crop so the cell isn't empty.
                "fallback_photo": achievement_photo,
            }
        )

    # Final collector card — reuses the same colour map + photos so each
    # vignette matches its standalone card earlier in the album.
    if collector_entries:
        collector_card = render_collector_card(
            collector_entries,
            colour_map=colour_map,
            device_label=device,
            date_label=date_label,
        )
        collector_path = _save_variant_image(
            collector_card, output_dir, "report_mobile_99_collector.jpg"
        )
        species_word = "species" if len(collector_entries) == 1 else "species"
        collector_caption = (
            f"{_device_html_prefix()}<b>Daily Roundup</b> · "
            f"{len(collector_entries)} {species_word} today"
        )
        tiles.append({"photo_path": collector_path, "caption": collector_caption})

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


def _report_title_for_mode() -> str:
    """Return the report title based on the configured Telegram mode.

    English copy by project convention "Sprache ist IMMER ENGLISH IM CODE".
    Species names themselves stay in the operator's locale (handled
    upstream in common_names_<LOCALE>.json) — only the wrapper text
    is English here.
    """
    try:
        cfg = get_config()
    except Exception:
        return "Daily Report"
    mode = str(cfg.get("TELEGRAM_MODE", "off") or "off").strip().lower()
    if mode == "interval":
        try:
            hours = int(float(cfg.get("TELEGRAM_REPORT_INTERVAL_HOURS", 1)))
        except Exception:
            hours = 1
        hours = max(1, min(24, hours))
        if hours == 1:
            return "Hourly Report"
        return f"Interval Report ({hours}h)"
    # "daily", "off" (manual send), "live" (manual send) -> evening-style title.
    return "Daily Report"


def render_text_report(
    report_date: str,
    total_events: int,
    species_count: int,
    top_species_name: str,
    top_species_count: int,
    top_is_latin: bool = False,
) -> str:
    """Render the report header + summary as valid Telegram HTML.

    All wrapper copy is English (by project convention). The species name
    in the "most frequent" line stays in whatever locale common_names
    resolved it to — that's a proper noun and shouldn't be translated.

    ``top_is_latin`` flags whether the resolved name is a Latin-only
    fallback (e.g. ``Phoenicurus (Art unklar)`` when the JSON mapping
    missed the genus). When True, the name is wrapped in ``<i>…</i>``
    for italic display in Telegram, matching the scientific-name
    convention in field guides.
    """
    lines: list[str] = []

    title = _report_title_for_mode()
    lines.append(
        f"{_device_html_prefix()}<b>WatchMyBirds · {_html_escape(title)} — "
        f"{_html_escape(_format_report_date(report_date))}</b>"
    )
    lines.append("")
    event_label = _count_label(total_events, "event", "events")
    species_label = _count_label(species_count, "species", "species")
    visit_word = "visit" if top_species_count == 1 else "visits"
    lines.append(
        f"<b>{total_events}</b> {event_label}, <b>{species_count}</b> {species_label}."
    )
    if top_species_name and top_species_name != "—" and top_species_count > 0:
        # ``top_species_name`` is already a resolved display string from
        # main() (e.g. "Eichelhäher"), not a CLS scientific name. The
        # earlier ``_humanize_species_name`` second pass was redundant
        # and confusing — it only mattered for inputs with underscores,
        # which a resolved common name doesn't have. Just escape & emit.
        escaped = _html_escape(top_species_name)
        # Bold wraps italic so Telegram renders both: <b><i>Phoenicurus
        # (Art unklar)</i></b>. Plain common names get only <b>.
        if top_is_latin:
            top_label = f"<i>{escaped}</i>"
        else:
            top_label = escaped
        lines.append(
            f"Most frequent: <b>{top_label}</b> ({top_species_count} {visit_word})."
        )
    else:
        lines.append("No species detected.")

    return "\n".join(lines)


def main(**_kwargs: Any) -> None:
    """Generate and send the evening Telegram report."""
    conn = get_connection()

    if len(sys.argv) > 1:
        report_date = sys.argv[1]
    else:
        report_date = datetime.date.today().isoformat()

    logger.info("Generating evening report for %s", report_date)

    try:
        config = get_config()

        today_rows = [
            dict(row)
            for row in fetch_detections_for_gallery(conn, report_date, order_by="time")
        ]
        obs_summary = summarize_observations(today_rows, min_score=0.0)
        obs_stats = obs_summary["summary"]
        species_counts: dict[str, int] = obs_stats["species_counts"]

        common_names = _load_common_names()

        all_species_photos = _fetch_species_best_photos(conn, report_date)
        # Override raw detection counts with observation-based visit
        # counts (same numbers shown on the live gallery), then apply
        # TELEGRAM_MIN_CONFIRMED_OBSERVATIONS against those visit counts.
        # Doing the threshold here, post-aggregation, keeps the filter
        # semantically aligned with what the operator sees in the report:
        # threshold=3 means "must have at least 3 separate visits today",
        # not "must have at least 3 raw bbox detections" (which a single
        # long-stay visit easily clears even though it's one sighting).
        try:
            min_visits = int(config.get("TELEGRAM_MIN_CONFIRMED_OBSERVATIONS", 1))
        except (TypeError, ValueError):
            min_visits = 1
        min_visits = max(1, min_visits)

        species_photos = []
        for sp in all_species_photos:
            visits = species_counts.get(sp["species"])
            if visits is None:
                continue
            if visits < min_visits:
                logger.debug(
                    "Dropping species %r (visits=%d, threshold=%d)",
                    sp["species"],
                    visits,
                    min_visits,
                )
                continue
            sp["count"] = visits
            species_photos.append(sp)
        species_photos.sort(key=lambda sp: sp["count"], reverse=True)

        # Stats for the text-report header are derived from the
        # POST-FILTER set so the wording matches what the operator
        # sees in the collector card. Otherwise the header would
        # claim "7 species" while the card only shows the 4 that
        # cleared the visit threshold.
        if species_photos:
            visible_species_count = len(species_photos)
            visible_total_events = sum(sp["count"] for sp in species_photos)
            top_pick = species_photos[0]  # highest visit count
            top_scientific = top_pick["species"]
            top_species_count = top_pick["count"]
            top_species_name, top_is_latin = _resolve_common_name(
                top_scientific, common_names
            )
        else:
            # No species cleared the threshold. Fall back to the raw
            # totals so the operator at least sees "0 events / 0
            # species" instead of an empty card with stale numbers.
            visible_species_count = 0
            visible_total_events = 0
            top_species_name = "—"
            top_species_count = 0
            top_is_latin = False

        text_message = render_text_report(
            report_date=report_date,
            total_events=visible_total_events,
            species_count=visible_species_count,
            top_species_name=top_species_name,
            top_species_count=top_species_count,
            top_is_latin=top_is_latin,
        )

        logger.info("Sending text report via Telegram...")
        text_responses = send_telegram_message(text_message, parse_mode="HTML")
        logger.info("Text report sent. Responses: %s", text_responses)

        if species_photos:
            # Single output path: build the album (which internally
            # renders both per-species achievement cards and a final
            # collector card) and send only the collector card. The
            # per-species cards are rendered as a side-effect — they
            # stay in temp storage and aren't sent, kept that way so
            # the deck can come back later without reshaping the album
            # builder.
            logger.info("Building report album...")
            tiles_variant = build_report_mobile_album(
                species_photos,
                common_names,
                report_date=report_date,
            )
            if not tiles_variant:
                logger.warning("Mobile album build returned no tiles.")
            else:
                overview_tile = tiles_variant[-1]
                logger.info("Sending collector card via Telegram...")
                response = send_telegram_message(
                    overview_tile["caption"],
                    photo_path=overview_tile["photo_path"],
                    parse_mode="HTML",
                )
                logger.info("Collector card sent. Response: %s", response)
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
