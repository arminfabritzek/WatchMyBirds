"""Build the offline geographic synonym reference from the GBIF backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import requests

SOURCE_URL = "https://api.gbif.org/v1/species/search"
SOURCE_QUERY = {
    "highertaxon_key": 212,
    "dataset_key": "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c",
    "rank": "SPECIES",
    "status": ["SYNONYM", "HOMOTYPIC_SYNONYM", "HETEROTYPIC_SYNONYM"],
    "limit": 1000,
}


def download_reference() -> bytes:
    records = []
    offset = 0
    expected_count = None
    fields = (
        "key",
        "canonicalName",
        "acceptedKey",
        "speciesKey",
        "species",
        "rank",
        "taxonomicStatus",
        "nameType",
    )
    with requests.Session() as session:
        while True:
            response = session.get(
                SOURCE_URL, params={**SOURCE_QUERY, "offset": offset}, timeout=(10, 60)
            )
            response.raise_for_status()
            page = response.json()
            if expected_count is None:
                expected_count = page["count"]
            if page["count"] != expected_count or page["offset"] != offset:
                raise ValueError("GBIF pagination changed during retrieval")
            records.extend(
                {key: row.get(key) for key in fields} for row in page["results"]
            )
            if page["endOfRecords"]:
                break
            if not page["results"] or page["limit"] <= 0:
                raise ValueError("Incomplete GBIF page")
            offset += page["limit"]
    if len(records) != expected_count or len({row["key"] for row in records}) != len(
        records
    ):
        raise ValueError("Incomplete or duplicate GBIF records")
    return json.dumps(
        {
            "url": SOURCE_URL,
            "params": SOURCE_QUERY,
            "count": expected_count,
            "records": records,
        },
        ensure_ascii=False,
        indent=2,
    ).encode()


def build_reference(raw: bytes, retrieved: str) -> dict[str, Any]:
    source = json.loads(raw)
    if source["url"] != SOURCE_URL or source["params"] != SOURCE_QUERY:
        raise ValueError("Unexpected GBIF source or query")
    groups: dict[int, set[str]] = defaultdict(set)
    for row in source["records"]:
        names = (row["canonicalName"], row["species"])
        if (
            row["rank"] == "SPECIES"
            and row["taxonomicStatus"] in SOURCE_QUERY["status"]
            and isinstance(row["acceptedKey"], int)
            and row["acceptedKey"] > 0
            and row["acceptedKey"] == row["speciesKey"]
            and row["nameType"] == "SCIENTIFIC"
            and all(re.fullmatch(r"[A-Z][a-z]+ [a-z]+", name or "") for name in names)
        ):
            groups[row["acceptedKey"]].update(names)
    return {
        "schema_version": 1,
        "source": {
            "title": "GBIF Backbone Taxonomy",
            "dataset_key": SOURCE_QUERY["dataset_key"],
            "doi": "https://doi.org/10.15468/39omei",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "retrieved": date.fromisoformat(retrieved).isoformat(),
            "api": SOURCE_URL,
            "query": SOURCE_QUERY,
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            "citation": (
                "GBIF Secretariat (2023). GBIF Backbone Taxonomy. Checklist dataset "
                f"https://doi.org/10.15468/39omei accessed via GBIF.org on {retrieved}."
            ),
            "derivation": (
                "Only unambiguous species-rank scientific synonyms whose acceptedKey "
                "equals speciesKey; no subspecies, hybrids or higher taxa."
            ),
        },
        "groups": sorted(sorted(names) for names in groups.values() if len(names) > 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--retrieved", default=date.today().isoformat())
    parser.add_argument(
        "--output", type=Path, default=Path("assets/geographic_synonyms.json")
    )
    args = parser.parse_args()
    raw = args.source_json.read_bytes() if args.source_json else download_reference()
    reference = build_reference(raw, args.retrieved)
    args.output.write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(reference['groups'])} scientific synonym groups to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
