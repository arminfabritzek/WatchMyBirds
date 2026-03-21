"""Locale-aware species common-name loader.

Provides a single function used by web_interface (startup + hot-reload),
the species-list API, and the species/thumbnails API to resolve display
names according to the active SPECIES_COMMON_NAME_LOCALE setting.

Strategy:
  1. Always load DE base map (``assets/common_names_DE.json``).
  2. If locale is ``NO``, overlay ``assets/common_names_NO.json`` on top.
  3. Missing NO keys fall back to DE silently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from logging_config import get_logger

logger = get_logger(__name__)
UNKNOWN_SPECIES_KEY = "Unknown_species"

_ASSETS_DIR = (
    Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "assets"
)


def load_common_names(locale: str = "DE") -> dict[str, str]:
    """Return a ``{scientific_key: display_name}`` dict for *locale*.

    Parameters
    ----------
    locale:
        ``"DE"`` (default) or ``"NO"``.  Unknown values fall back to DE.
    """
    locale = str(locale).strip().upper()
    if locale not in ("DE", "NO"):
        locale = "DE"

    # 1. Base map — always DE
    base_path = _ASSETS_DIR / "common_names_DE.json"
    try:
        with open(base_path, encoding="utf-8") as f:
            names: dict[str, str] = json.load(f)
    except Exception as exc:
        logger.error("Failed to load DE common names from %s: %s", base_path, exc)
        return {}

    if locale == "DE":
        return names

    # 2. Overlay NO
    overlay_path = _ASSETS_DIR / "common_names_NO.json"
    if overlay_path.exists():
        try:
            with open(overlay_path, encoding="utf-8") as f:
                overlay: dict[str, str] = json.load(f)
            names.update(overlay)
        except Exception as exc:
            logger.warning("Failed to load NO overlay from %s: %s", overlay_path, exc)

    return names


def load_extended_species(locale: str = "DE") -> list[dict[str, str]]:
    """Return extended species entries for picker/search use.

    The returned shape is:
        [{"scientific": "Picus_canus", "common": "Grauspecht"}, ...]

    Species already covered by the model/common-name base map are excluded.
    Locale fallback order:
    - DE: common_de -> common_en -> scientific name with spaces
    - NO: common_nb -> common_en -> common_de -> scientific name with spaces
    """
    locale = str(locale).strip().upper()
    if locale not in ("DE", "NO"):
        locale = "DE"

    asset_path = _ASSETS_DIR / "extended_species_global.json"
    if not asset_path.exists():
        return []

    try:
        with open(asset_path, encoding="utf-8") as f:
            raw_entries: list[dict[str, str]] = json.load(f)
    except Exception as exc:
        logger.error("Failed to load extended species from %s: %s", asset_path, exc)
        return []

    model_species = set(load_common_names("DE").keys())
    entries: list[dict[str, str]] = []
    for item in raw_entries:
        scientific = str(item.get("scientific") or "").strip()
        if not scientific:
            continue
        if scientific in model_species or scientific == UNKNOWN_SPECIES_KEY:
            continue

        if locale == "NO":
            common = (
                item.get("common_nb")
                or item.get("common_en")
                or item.get("common_de")
                or scientific.replace("_", " ")
            )
        else:
            common = (
                item.get("common_de")
                or item.get("common_en")
                or scientific.replace("_", " ")
            )
        entries.append({"scientific": scientific, "common": str(common)})

    entries.sort(key=lambda row: row["scientific"])
    return entries


def build_species_picker_entries(
    conn,
    locale: str = "DE",
    detection_id: int | None = None,
) -> list[dict]:
    """Return picker entries ordered as prediction -> model -> extended."""
    model_names = load_common_names(locale)
    extended_entries = load_extended_species(locale)
    extended_lookup = {
        entry["scientific"]: entry["common"] for entry in extended_entries
    }
    seen: set[str] = set()
    items: list[dict] = []

    if detection_id is not None:
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
        for row in rows:
            scientific = row["cls_class_name"]
            if not scientific or scientific in seen:
                continue
            common = (
                model_names.get(scientific)
                or extended_lookup.get(scientific)
                or scientific.replace("_", " ")
            )
            items.append(
                {
                    "scientific": scientific,
                    "common": common,
                    "source": "prediction",
                    "score": float(row["cls_confidence"] or 0.0),
                    "rank": int(row["rank"] or 0),
                }
            )
            seen.add(scientific)

    unknown_label = model_names.get(UNKNOWN_SPECIES_KEY, "Unknown species")
    if UNKNOWN_SPECIES_KEY not in seen:
        items.append(
            {
                "scientific": UNKNOWN_SPECIES_KEY,
                "common": unknown_label,
                "source": "model",
                "score": None,
                "rank": None,
            }
        )
        seen.add(UNKNOWN_SPECIES_KEY)

    for scientific, common in sorted(model_names.items(), key=lambda item: item[0]):
        if scientific in seen or scientific == UNKNOWN_SPECIES_KEY:
            continue
        items.append(
            {
                "scientific": scientific,
                "common": common,
                "source": "model",
                "score": None,
                "rank": None,
            }
        )
        seen.add(scientific)

    for entry in extended_entries:
        scientific = entry["scientific"]
        if scientific in seen:
            continue
        items.append(
            {
                "scientific": scientific,
                "common": entry["common"],
                "source": "extended",
                "score": None,
                "rank": None,
            }
        )
        seen.add(scientific)

    return items
