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
