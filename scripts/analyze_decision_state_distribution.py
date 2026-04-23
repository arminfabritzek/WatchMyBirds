"""Analyze the decision_state distribution of an existing WatchMyBirds DB.

Run this BEFORE deploying the strict-confirmed gallery policy to see how
many detections will change visibility. Works against any sqlite DB with
a ``detections`` table carrying ``status`` and ``decision_state`` columns.

Usage:
    python scripts/analyze_decision_state_distribution.py /path/to/images.db

On the RPi where the app is deployed, the DB usually lives at
``<OUTPUT_DIR>/images.db``.

Prints three things:
  1. Distribution of decision_state across active detections
  2. Before/after count for the gallery policy change
  3. How much of NULL would be "rescued" by a soft backfill at score
     thresholds 0.65 and 0.80 (useful for deciding if a backfill migration
     makes sense)

Read-only. No data is modified.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def analyze(db_path: Path) -> int:
    if not db_path.exists():
        print(f"ERROR: {db_path} not found", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE status='active'"
    ).fetchone()[0]
    print(f"Database: {db_path}")
    print(f"Total active detections: {total}")
    print()
    if total == 0:
        print("(empty DB — nothing to analyze)")
        return 0

    print(f'{"state":<14} {"count":>8} {"pct":>7}  {"visibility"}')
    print("-" * 65)
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(decision_state), '(null)') AS state,
            COUNT(*) AS count
        FROM detections
        WHERE status = 'active'
        GROUP BY state
        ORDER BY count DESC
        """
    ).fetchall()
    for r in rows:
        pct = 100.0 * r["count"] / total
        was_visible = r["state"] not in ("uncertain", "unknown")
        is_visible = r["state"] == "confirmed"
        if was_visible and not is_visible:
            change = "was shown -> WILL BE HIDDEN"
        elif was_visible and is_visible:
            change = "still shown"
        else:
            change = "(still hidden)"
        print(f"{r['state']:<14} {r['count']:>8} {pct:>6.1f}%  {change}")
    print()

    before = sum(
        r["count"] for r in rows if r["state"] not in ("uncertain", "unknown")
    )
    after = sum(r["count"] for r in rows if r["state"] == "confirmed")
    print(f"Gallery size BEFORE policy change: {before} detections")
    print(f"Gallery size AFTER  policy change: {after} detections")
    if before:
        delta = before - after
        print(f"Net drop: {delta} detections ({100.0 * delta / before:.1f}% fewer)")
    print()

    null_total = conn.execute(
        "SELECT COUNT(*) FROM detections "
        "WHERE status='active' AND decision_state IS NULL"
    ).fetchone()[0]
    print(f"NULL-state detections: {null_total}")
    if null_total:
        for threshold in (0.65, 0.80):
            rescued = conn.execute(
                "SELECT COUNT(*) FROM detections "
                "WHERE status='active' AND decision_state IS NULL "
                "AND score >= ?",
                (threshold,),
            ).fetchone()[0]
            pct = 100.0 * rescued / null_total
            print(
                f"  with score >= {threshold:.2f}: {rescued} "
                f"({pct:.1f}%) would be rescued by soft backfill"
            )

    conn.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/analyze_decision_state_distribution.py <db_path>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(analyze(Path(sys.argv[1])))
