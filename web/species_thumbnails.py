"""Shared species thumbnail resolution for Review and API surfaces."""

from __future__ import annotations

import time
from pathlib import Path

from utils.species_names import UNKNOWN_SPECIES_KEY
from web.services import gallery_service

REVIEW_SPECIES_ASSET_EXTENSIONS = ("webp", "jpg", "jpeg", "png", "avif")
_SPECIES_THUMBNAIL_CACHE_TTL_SECONDS = 60
_species_thumbnail_cache: dict[str, tuple[float, dict[str, str]]] = {}


def normalize_species_key(species_name: str | None) -> str:
    """Normalize species names to the app's underscored scientific key."""
    return str(species_name or "").strip().replace(" ", "_")


def resolve_static_species_asset_url(scientific_name: str | None) -> str:
    """Return a dedicated local review species asset when available."""
    species_key = normalize_species_key(scientific_name)
    if not species_key:
        return ""

    assets_dir = Path(__file__).resolve().parents[1] / "assets" / "review_species"
    for ext in REVIEW_SPECIES_ASSET_EXTENSIONS:
        candidate = assets_dir / f"{species_key}.{ext}"
        if candidate.exists():
            return f"/assets/review_species/{species_key}.{ext}"
    return ""


def _assign_species_keys(
    target: dict[str, str],
    scientific_name: str | None,
    url: str,
    common_names: dict[str, str] | None = None,
) -> None:
    species_key = normalize_species_key(scientific_name)
    if not species_key or not url:
        return

    common_names = common_names or {}
    target[species_key] = url
    target[species_key.replace("_", " ")] = url

    common_name = common_names.get(species_key) or common_names.get(
        species_key.replace("_", " ")
    )
    if common_name:
        target[common_name] = url


def _resolve_detection_preview_url(det: dict) -> str:
    thumb_virtual = str(det.get("thumbnail_path_virtual") or "").strip()
    if thumb_virtual:
        return f"/uploads/derivatives/thumbs/{thumb_virtual}"

    optimized_virtual = str(
        det.get("relative_path") or det.get("optimized_name_virtual") or ""
    ).strip()
    if optimized_virtual:
        return f"/uploads/derivatives/optimized/{optimized_virtual}"

    return ""


def _resolve_detection_species_key(det: dict) -> str:
    # Route the OD fallback through the central helper so "bird" does
    # not leak in as species truth when CLS is missing. Non-bird OD
    # class names like "squirrel" still pass through.
    from utils.species_names import UNKNOWN_SPECIES_KEY, species_key_from_candidates

    resolved = species_key_from_candidates(
        manual_override=det.get("manual_species_override"),
        species_key=det.get("species_key"),
        cls_class_name=det.get("cls_class_name"),
        od_class_name=det.get("od_class_name"),
    )
    # Preserve historical contract: empty-key result when nothing resolved.
    return "" if resolved == UNKNOWN_SPECIES_KEY else normalize_species_key(resolved)


def get_species_thumbnail_map(
    *,
    common_names: dict[str, str] | None = None,
    cache_key: str | None = "default",
    detections: list[dict] | None = None,
) -> dict[str, str]:
    """
    Return a stable species thumbnail map.

    Priority:
    1. Dedicated local species art in assets/review_species
    2. Favorite detections
    3. Any detection/optimized image for that species
    """
    common_names = common_names or {}
    use_cache = detections is None and cache_key is not None
    now = time.monotonic()

    if use_cache:
        cached = _species_thumbnail_cache.get(cache_key)
        if cached and (now - cached[0]) < _SPECIES_THUMBNAIL_CACHE_TTL_SECONDS:
            return dict(cached[1])

    if detections is None:
        detections = gallery_service.get_all_detections()

    favorite_by_species: dict[str, tuple[float, str]] = {}
    general_by_species: dict[str, tuple[float, str]] = {}
    all_species: set[str] = set()

    for det in detections or []:
        species_key = _resolve_detection_species_key(det)
        if not species_key or species_key == UNKNOWN_SPECIES_KEY:
            continue

        preview_url = _resolve_detection_preview_url(det)
        if not preview_url:
            continue

        all_species.add(species_key)
        score = float(
            det.get("score")
            or det.get("od_confidence")
            or det.get("cls_confidence")
            or 0.0
        )

        current_general = general_by_species.get(species_key)
        if current_general is None or score >= current_general[0]:
            general_by_species[species_key] = (score, preview_url)

        if bool(int(det.get("is_favorite") or 0)):
            current_favorite = favorite_by_species.get(species_key)
            if current_favorite is None or score >= current_favorite[0]:
                favorite_by_species[species_key] = (score, preview_url)

    mapping: dict[str, str] = {}
    static_species: set[str] = set()
    for species_key in all_species:
        static_url = resolve_static_species_asset_url(species_key)
        if not static_url:
            continue
        static_species.add(species_key)
        _assign_species_keys(mapping, species_key, static_url, common_names)

    for species_key, (_score, url) in favorite_by_species.items():
        if species_key not in static_species:
            _assign_species_keys(mapping, species_key, url, common_names)

    for species_key, (_score, url) in general_by_species.items():
        if species_key not in static_species and species_key not in favorite_by_species:
            _assign_species_keys(mapping, species_key, url, common_names)

    if use_cache:
        _species_thumbnail_cache[cache_key] = (now, dict(mapping))
    return mapping


def resolve_species_thumbnail_url(
    scientific_name: str | None,
    *,
    common_names: dict[str, str] | None = None,
    thumbnail_map: dict[str, str] | None = None,
    cache_key: str = "default",
) -> str:
    """Resolve one species thumbnail URL using the shared priority rules."""
    species_key = normalize_species_key(scientific_name)
    if not species_key or species_key == UNKNOWN_SPECIES_KEY:
        return ""

    static_url = resolve_static_species_asset_url(species_key)
    if static_url:
        return static_url

    common_names = common_names or {}
    thumbnail_map = thumbnail_map or get_species_thumbnail_map(
        common_names=common_names,
        cache_key=cache_key,
    )

    candidates = [species_key, species_key.replace("_", " ")]
    common_name = common_names.get(species_key) or common_names.get(
        species_key.replace("_", " ")
    )
    if common_name:
        candidates.append(common_name)

    for candidate in candidates:
        url = thumbnail_map.get(candidate)
        if url:
            return url
    return ""
