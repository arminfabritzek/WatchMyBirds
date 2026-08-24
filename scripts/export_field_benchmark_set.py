#!/usr/bin/env python3
"""Export a local, read-only field benchmark set from station evidence.

Copies source images into separate ``negative`` and ``positive`` directories.
Originals are never modified. Negatives are selected from an operator-declared
filename window; positives come only from current human bird-presence facts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path


def _resolve_original(originals: Path, filename: str) -> Path | None:
    if len(filename) < 8 or not filename[:8].isdigit():
        return None
    day = f"{filename[:4]}-{filename[4:6]}-{filename[6:8]}"
    candidate = originals / day / filename
    return candidate if candidate.is_file() else None


def _fetch_names(
    conn: sqlite3.Connection,
    *,
    negative_from: str,
    negative_through: str,
    negative_limit: int,
    positive_limit: int,
) -> tuple[list[str], list[str]]:
    negatives = [
        str(row[0])
        for row in conn.execute(
            """
            SELECT DISTINCT i.filename
            FROM images i
            JOIN detections d ON d.image_filename = i.filename
            WHERE i.filename BETWEEN ? AND ?
              AND COALESCE(i.review_status, 'untagged') = 'untagged'
              AND COALESCE(d.status, 'active') = 'active'
              AND LOWER(TRIM(COALESCE(d.od_class_name, ''))) = 'bird'
            ORDER BY i.filename
            LIMIT ?
            """,
            (negative_from, negative_through, negative_limit),
        ).fetchall()
    ]
    positives = [
        str(row[0])
        for row in conn.execute(
            """
            SELECT DISTINCT s.image_filename
            FROM label_subjects s
            JOIN current_human_label_facts bird
              ON bird.subject_id = s.subject_id
             AND bird.fact_type = 'bird_presence'
             AND bird.answer_value = 'present'
            WHERE s.scope = 'object'
              AND s.image_filename IS NOT NULL
            ORDER BY s.image_filename DESC
            LIMIT ?
            """,
            (positive_limit,),
        ).fetchall()
    ]
    negative_set = set(negatives)
    positives = [name for name in positives if name not in negative_set]
    return negatives, positives


def _copy_group(originals: Path, destination: Path, names: list[str]) -> list[str]:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"benchmark destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in names:
        source = _resolve_original(originals, name)
        if source is None:
            continue
        shutil.copy2(source, destination / name)
        copied.append(name)
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--originals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--negative-from", required=True)
    parser.add_argument("--negative-through", required=True)
    parser.add_argument("--negative-limit", type=int, default=1000)
    parser.add_argument("--positive-limit", type=int, default=200)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    originals = Path(args.originals).resolve()
    output.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{Path(args.db).resolve()}?mode=ro", uri=True)
    try:
        negatives, positives = _fetch_names(
            conn,
            negative_from=args.negative_from,
            negative_through=args.negative_through,
            negative_limit=max(1, args.negative_limit),
            positive_limit=max(1, args.positive_limit),
        )
    finally:
        conn.close()

    copied_negatives = _copy_group(originals, output / "negative", negatives)
    copied_positives = _copy_group(originals, output / "positive", positives)
    manifest = {
        "schema_version": "wmb.field_benchmark.v1",
        "negative_basis": "operator-declared all-FP filename window",
        "positive_basis": "current human object bird_presence=present facts",
        "negative_from": args.negative_from,
        "negative_through": args.negative_through,
        "negative": copied_negatives,
        "positive": copied_positives,
        "warnings": [
            "Positive labels are object-level human facts, not diagnostic event records.",
            "This export is for model comparison and must not be presented as a publication dataset.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "negative": len(copied_negatives),
                "positive": len(copied_positives),
                "output": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
