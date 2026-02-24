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


def get_cached_species_thumbnail(scientific_name: str) -> str | None:
    """
    Get a species thumbnail URL from local cache (DB) or fetch from Wikipedia.

    Args:
        scientific_name: Scientific name (e.g. "Parus major")

    Returns:
        URL string or None if not found/error.
    """
    if not scientific_name:
        return None

    # Normalize name (spaces preferred for Wiki API)
    sci_name_clean = scientific_name.replace("_", " ").strip()

    # 1. Check Cache
    from utils.db.connection import get_connection  # Avoid circular imports

    try:
        # Check using the exact string provided first
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT image_url FROM species_meta WHERE scientific_name = ?",
                (sci_name_clean,),
            ).fetchone()

            if row:
                # If cached (even if None/empty string to indicate 'no image found'), return it
                return row[0]
        finally:
            conn.close()
    except Exception:
        pass

    # 2. Fetch from Wikipedia
    import requests

    image_url = None
    try:
        # Try German Wikipedia first (often better for local birds), then English
        # API: query pageimages
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "WatchMyBirds/1.0 (https://github.com/arminfabritzek/WatchMyBirds; bird monitoring app)"
            }
        )

        def _fetch_thumb(lang="de"):
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "titles": sci_name_clean,
                "prop": "pageimages",
                "format": "json",
                "pithumbsize": 200,  # 200px thumbnail
                "piprop": "thumbnail",
                "redirects": 1,  # Follow redirects (scientific names redirect to common name articles)
                "origin": "*",
            }
            resp = session.get(url, params=params, timeout=2.0)
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for _pid, page in pages.items():
                if "thumbnail" in page:
                    return page["thumbnail"]["source"]
            return None

        image_url = _fetch_thumb("de")
        if not image_url:
            image_url = _fetch_thumb("en")

    except Exception:
        # On error, we don't cache failure to allow retries, or we could cache empty string
        # For now, just return None
        return None

    # 3. Store in Cache (if found)
    # We cache strictly found URLs. If None, we don't cache (allows retry later).
    # To prevent repeated failed lookups, one could cache empty string.
    if image_url:
        try:
            conn = get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO species_meta (scientific_name, image_url) VALUES (?, ?)",
                    (sci_name_clean, image_url),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    return image_url
