#!/usr/bin/env python3
"""Generate the global extended bird species catalog from iNaturalist."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://api.inaturalist.org/v1/taxa"
TAXON_ID_AVES = 3
PER_PAGE = 200
LOCALE_FIELDS = {
    "de": "common_de",
    "nb": "common_nb",
}


def fetch_batch(locale: str, id_above: int | None = None) -> dict:
    params = {
        "taxon_id": TAXON_ID_AVES,
        "rank": "species",
        "per_page": PER_PAGE,
        "locale": locale,
        "order_by": "id",
        "order": "asc",
    }
    if id_above is not None:
        params["id_above"] = id_above
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_locale_entries(locale: str, field_name: str) -> dict[str, dict[str, str]]:
    first_batch = fetch_batch(locale=locale)
    total_results = int(first_batch.get("total_results") or 0)

    seen: set[str] = set()
    entries: dict[str, dict[str, str]] = {}
    batch_count = 0
    id_above: int | None = None
    payload = first_batch

    while True:
        batch_count += 1
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            scientific_name = str(item.get("name") or "").strip()
            if not scientific_name:
                continue

            scientific_key = scientific_name.replace(" ", "_")
            if scientific_key in seen:
                continue

            common_en = str(item.get("english_common_name") or "").strip()
            localized_common = str(item.get("preferred_common_name") or "").strip()

            entries[scientific_key] = {
                "scientific": scientific_key,
                field_name: localized_common,
                "common_en": common_en,
            }
            seen.add(scientific_key)

        print(
            (
                f"Fetched locale={locale} batch {batch_count} "
                f"({len(entries)}/{total_results} species accumulated)"
            ),
            file=sys.stderr,
        )

        if len(results) < PER_PAGE:
            break

        id_above = max(int(item.get("id") or 0) for item in results)
        time.sleep(0.15)
        payload = fetch_batch(locale=locale, id_above=id_above)

    return entries


def build_entries() -> list[dict[str, str]]:
    localized_entries = {
        field_name: fetch_locale_entries(locale, field_name)
        for locale, field_name in LOCALE_FIELDS.items()
    }

    scientific_keys = sorted(
        {
            scientific
            for rows in localized_entries.values()
            for scientific in rows.keys()
        }
    )

    entries: list[dict[str, str]] = []
    for scientific in scientific_keys:
        common_en = ""
        row = {
            "scientific": scientific,
            "common_de": "",
            "common_en": "",
            "common_nb": "",
        }
        for field_name, rows in localized_entries.items():
            localized = rows.get(scientific) or {}
            row[field_name] = str(localized.get(field_name) or "").strip()
            common_en = common_en or str(localized.get("common_en") or "").strip()
        row["common_en"] = common_en
        entries.append(row)

    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="assets/extended_species_global.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = build_entries()
    output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(entries)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
