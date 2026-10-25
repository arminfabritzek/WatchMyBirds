"""Backfill the bird_presence fact implied by an existing species answer.

Answering a species question — naming one, correcting one, or declining
to name one — already asserts that the box holds a bird.
``record_human_answer`` derives that presence fact today, but rows
answered before that change carry only the species fact, so they fail OD
readiness as ``object_bird_presence_unknown`` and their boxes are dropped
from training despite a person having endorsed them.

This writes the missing fact for those rows, tagged in ``source_ref`` so
it stays distinguishable from a directly given answer, and reusing each
row's original provenance so attribution is preserved.

Only object subjects whose species fact has no ``bird_presence``
companion are touched. A subject that already carries any presence fact
— including an explicit "absent" — is skipped, so a human answer is
never overwritten.

Usage:
    .venv/bin/python scripts/backfill_derived_bird_presence.py           # dry run
    .venv/bin/python scripts/backfill_derived_bird_presence.py --apply
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.human_label_core import (  # noqa: E402
    DERIVED_FROM_SPECIES_ANSWER,
    HumanLabelError,
    LabelProvenance,
    append_fact,
)
from utils.db import connection as db_connection  # noqa: E402

logger = logging.getLogger("backfill_derived_bird_presence")

# Object subjects carrying an asserted species answer but no bird_presence
# fact of any value. The NOT EXISTS deliberately ignores answer_value: an
# explicit "absent" must block the backfill, not invite a second opinion.
_CANDIDATE_SQL = """
    SELECT
        f.subject_id,
        s.detection_id,
        f.installation_id,
        f.app_version,
        f.context,
        f.source_kind,
        f.source_ref,
        f.created_at,
        f.answer_value AS species_answer,
        s.image_filename
    FROM current_human_label_facts f
    JOIN label_subjects s ON s.subject_id = f.subject_id
    WHERE f.fact_type = 'species_identity'
      AND f.assertion_state = 'asserted'
      AND s.detection_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM current_human_label_facts g
          WHERE g.subject_id = f.subject_id
            AND g.fact_type = 'bird_presence'
      )
    ORDER BY s.detection_id
"""


def _derived_source_ref(source_ref: str | None) -> str:
    base = (source_ref or "").strip()
    return (
        f"{base}|{DERIVED_FROM_SPECIES_ANSWER}" if base else DERIVED_FROM_SPECIES_ANSWER
    )


def backfill(conn: sqlite3.Connection, *, apply: bool) -> dict[str, int]:
    rows = conn.execute(_CANDIDATE_SQL).fetchall()
    written = 0
    skipped = 0
    conflicted = 0

    for index, row in enumerate(rows):
        try:
            provenance = LabelProvenance(
                installation_id=str(row["installation_id"]),
                app_version=str(row["app_version"] or "unknown"),
                context=str(row["context"]),
                source_kind=str(row["source_kind"]),
                source_ref=_derived_source_ref(row["source_ref"]),
                created_at=str(row["created_at"]),
            )
        except HumanLabelError as exc:
            logger.warning(
                "detection %s: unusable provenance (%s), skipped",
                row["detection_id"],
                exc,
            )
            skipped += 1
            continue

        if not apply:
            logger.info(
                "detection %s: species_identity=%s -> bird_presence=present (dry run)",
                row["detection_id"],
                row["species_answer"],
            )
            written += 1
            continue

        # Each row gets its own savepoint: a frame whose image-level answer
        # contradicts the box-level one raises, and only that row is rolled
        # back. The contradiction is a real pair of human answers, so it is
        # reported for a person to resolve rather than decided here.
        savepoint = f"backfill_presence_{index}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            append_fact(
                conn,
                subject_id=int(row["subject_id"]),
                fact_type="bird_presence",
                answer_value="present",
                provenance=provenance,
            )
        except HumanLabelError as exc:
            conn.execute(f"ROLLBACK TO {savepoint}")
            logger.warning(
                "detection %s (%s): %s — left for manual resolution",
                row["detection_id"],
                row["image_filename"],
                exc,
            )
            conflicted += 1
        else:
            logger.info(
                "detection %s: species_identity=%s -> bird_presence=present",
                row["detection_id"],
                row["species_answer"],
            )
            written += 1
        finally:
            conn.execute(f"RELEASE {savepoint}")

    return {
        "candidates": len(rows),
        "written": written,
        "skipped": skipped,
        "conflicted": conflicted,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill bird_presence facts implied by a species answer."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the facts. Without it the script only reports (default).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with db_connection.closing_connection() as conn:
        conn.row_factory = sqlite3.Row
        stats = backfill(conn, apply=args.apply)
        if args.apply:
            conn.commit()

    logger.info(
        "%s: %d candidate(s), %d written, %d skipped, %d conflicted",
        "applied" if args.apply else "dry run",
        stats["candidates"],
        stats["written"],
        stats["skipped"],
        stats["conflicted"],
    )
    if stats["conflicted"]:
        logger.warning(
            "%d row(s) carry an image-level 'no bird' answer that contradicts "
            "their species answer; resolve those by hand",
            stats["conflicted"],
        )
    if not args.apply and stats["candidates"]:
        logger.info("re-run with --apply to write these facts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
