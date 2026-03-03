#!/usr/bin/env python3
"""
Offline Decision Matrix Evaluator.

Reads the detections table from the production SQLite database and produces:
1. A summary report of decision state distribution.
2. Optional CSV export of uncertain/unknown candidates for review or retraining.

Usage:
    python scripts/eval_decision_matrix.py --db /path/to/images.db
    python scripts/eval_decision_matrix.py --db /path/to/images.db --export candidates.csv
    python scripts/eval_decision_matrix.py --db /path/to/images.db --json

Environment:
    No app dependencies required — uses only stdlib + minimal helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# DB reader
# ---------------------------------------------------------------------------

_QUERY = """
SELECT
    d.detection_id,
    d.image_filename,
    d.od_class_name,
    d.od_confidence,
    d.score,
    d.decision_state,
    d.bbox_quality,
    d.unknown_score,
    d.decision_reasons,
    d.policy_version,
    d.created_at,
    (SELECT c.cls_class_name
     FROM classifications c
     WHERE c.detection_id = d.detection_id
     ORDER BY c.cls_confidence DESC LIMIT 1) as cls_class_name,
    (SELECT c.cls_confidence
     FROM classifications c
     WHERE c.detection_id = d.detection_id
     ORDER BY c.cls_confidence DESC LIMIT 1) as cls_confidence
FROM detections d
WHERE d.status = 'active'
ORDER BY d.created_at DESC
"""


def _read_detections(db_path: str) -> list[dict]:
    """Read active detections from the given SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(_QUERY).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(detections: list[dict]) -> dict:
    """Compute decision state distribution and quality statistics."""
    total = len(detections)
    if total == 0:
        return {
            "total": 0,
            "states": {},
            "avg_bbox_quality": None,
            "avg_unknown_score": None,
        }

    # State distribution
    states: dict[str, int] = {}
    bbox_qualities: list[float] = []
    unknown_scores: list[float] = []

    for det in detections:
        state = det.get("decision_state") or "null"
        states[state] = states.get(state, 0) + 1

        bq = det.get("bbox_quality")
        if bq is not None:
            bbox_qualities.append(float(bq))

        us = det.get("unknown_score")
        if us is not None:
            unknown_scores.append(float(us))

    avg_bq = sum(bbox_qualities) / len(bbox_qualities) if bbox_qualities else None
    avg_us = sum(unknown_scores) / len(unknown_scores) if unknown_scores else None

    # Reason code frequency
    reason_counts: dict[str, int] = {}
    for det in detections:
        reasons_raw = det.get("decision_reasons") or "[]"
        try:
            reasons = json.loads(reasons_raw)
        except (json.JSONDecodeError, TypeError):
            reasons = []
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "total": total,
        "states": dict(sorted(states.items())),
        "state_percentages": {
            k: round(v / total * 100, 1) for k, v in sorted(states.items())
        },
        "avg_bbox_quality": round(avg_bq, 4) if avg_bq is not None else None,
        "avg_unknown_score": round(avg_us, 4) if avg_us is not None else None,
        "reason_codes": dict(sorted(reason_counts.items(), key=lambda x: -x[1])),
        "bbox_quality_coverage": f"{len(bbox_qualities)}/{total}",
        "unknown_score_coverage": f"{len(unknown_scores)}/{total}",
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

_EXPORT_FIELDS = [
    "detection_id",
    "image_filename",
    "od_class_name",
    "cls_class_name",
    "od_confidence",
    "cls_confidence",
    "score",
    "decision_state",
    "bbox_quality",
    "unknown_score",
    "decision_reasons",
    "policy_version",
    "created_at",
]


def export_candidates(
    detections: list[dict],
    output_path: str,
    states: tuple[str, ...] = ("uncertain", "unknown"),
) -> int:
    """
    Export detections matching the given states to CSV.

    Returns the number of exported rows.
    """
    candidates = [
        det for det in detections if (det.get("decision_state") or "null") in states
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for det in candidates:
            writer.writerow(det)

    return len(candidates)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------


def print_report(metrics: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("=" * 60)
    print("  WatchMyBirds — Decision Matrix Evaluation Report")
    print("=" * 60)
    print(f"  Total active detections: {metrics['total']}")
    print()

    print("  Decision State Distribution:")
    for state, count in metrics["states"].items():
        pct = metrics["state_percentages"].get(state, 0)
        bar = "█" * int(pct / 2)
        print(f"    {state:12s}  {count:6d}  ({pct:5.1f}%)  {bar}")
    print()

    if metrics.get("reason_codes"):
        print("  Reason Code Frequency:")
        for reason, count in metrics["reason_codes"].items():
            print(f"    {reason:25s}  {count:6d}")
        print()

    print(f"  Avg BBox Quality:   {metrics['avg_bbox_quality'] or 'N/A'}")
    print(f"  Avg Unknown Score:  {metrics['avg_unknown_score'] or 'N/A'}")
    print(f"  BBox Quality coverage:  {metrics['bbox_quality_coverage']}")
    print(f"  Unknown Score coverage: {metrics['unknown_score_coverage']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success."""
    parser = argparse.ArgumentParser(
        description="Evaluate the decision matrix from a WatchMyBirds database."
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to images.db (SQLite)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Optional: export uncertain/unknown candidates to CSV",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output metrics as JSON instead of human-readable report",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        return 1

    detections = _read_detections(str(db_path))
    metrics = compute_metrics(detections)

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_report(metrics)

    if args.export:
        count = export_candidates(detections, args.export)
        print(f"\n  Exported {count} candidates → {args.export}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
