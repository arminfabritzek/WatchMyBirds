"""Wikipedia URL helper utilities."""

from urllib.parse import quote_plus


def _normalize_species_text(value: str | None) -> str:
    """Normalize species/common names for search queries."""
    if not value:
        return ""
    return value.replace("_", " ").strip()


def build_species_wikipedia_url(
    common_name: str | None,
    scientific_name: str | None = None,
    locale: str = "de",
) -> str | None:
    """
    Build a robust Wikipedia species search URL.

    Uses scientific name when available, otherwise common name.
    Returns None when no usable term is present.
    """
    search_term = _normalize_species_text(scientific_name) or _normalize_species_text(
        common_name
    )
    if not search_term:
        return None

    safe_locale = (locale or "de").strip().lower()
    return (
        f"https://{safe_locale}.wikipedia.org/wiki/Spezial:Suche?search="
        f"{quote_plus(search_term)}"
    )
