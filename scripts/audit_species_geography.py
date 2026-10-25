"""Audit geographic reranking offline; source observations remain read-only."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.species_geography import rank_geographically
from core.species_geography_runtime import LocalGeoModel, capture_datetime


def audit(
    database: Path,
    result_database: Path,
    *,
    geo: LocalGeoModel,
    latitude: float,
    longitude: float,
    timezone: str,
    focus_species: str | None = None,
) -> dict[str, Any]:
    ZoneInfo(timezone)
    if database.resolve() == result_database.resolve() or result_database.exists():
        raise ValueError("Audit output must be a new file, distinct from the source")
    counts: Counter[str] = Counter()
    focused = []
    with closing(
        sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    ) as source:
        source.row_factory = sqlite3.Row
        # Human answers are evaluation targets only, never scorer inputs.
        truth = dict(
            source.execute("""
                SELECT d.detection_id, f.species_key
                FROM current_human_label_facts f
                JOIN label_subjects s ON s.subject_id = f.subject_id
                JOIN detections d ON d.detection_id = s.detection_id
                JOIN current_human_label_facts bird
                  ON bird.subject_id = f.subject_id
                 AND bird.fact_type = 'bird_presence'
                WHERE s.scope = 'object' AND f.fact_type = 'species_identity'
                  AND f.answer_value IN ('confirmed', 'corrected')
                  AND f.species_key IS NOT NULL AND bird.answer_value = 'present'
                  AND d.status = 'active'
            """)
        )
        predictions: dict[int, list[tuple[str, float]]] = {}
        models: dict[int, str] = {}
        for row in source.execute("""
            SELECT detection_id, cls_class_name, cls_confidence, cls_model_id, rank
            FROM classifications ORDER BY detection_id, rank
        """):
            predictions.setdefault(row["detection_id"], []).append(
                (row["cls_class_name"], row["cls_confidence"])
            )
            if row["rank"] == 1:
                models[row["detection_id"]] = row["cls_model_id"]
        result_database.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents overwriting an output created concurrently.
        with result_database.open("xb"):
            pass
        with closing(sqlite3.connect(result_database)) as output, output:
            output.execute("""
                CREATE TABLE assessments (
                    detection_id INTEGER PRIMARY KEY, raw_species TEXT,
                    suggested_species TEXT, human_species TEXT, result_json TEXT
                )
            """)
            rows = source.execute("""
                SELECT d.detection_id, d.status, d.image_filename, d.thumbnail_path, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.frame_width, d.frame_height, d.created_at, i.timestamp AS captured_at
                FROM detections d JOIN images i ON i.filename = d.image_filename
                ORDER BY d.detection_id
            """)
            for row in rows:
                detection_id = row["detection_id"]
                counts["stored_detections"] += 1
                candidates = predictions.get(detection_id, [])
                raw = candidates[0][0] if candidates else None
                answer = truth.get(detection_id)
                suggested = None
                details: dict[str, Any] = {
                    "status": row["status"],
                    "image_filename": row["image_filename"],
                    "thumbnail_path": row["thumbnail_path"],
                    "classifier_model_id": models.get(detection_id),
                    "raw_candidates": candidates,
                    "human_species": answer,
                    "bbox_frame_fraction": [
                        row["bbox_x"],
                        row["bbox_y"],
                        row["bbox_w"],
                        row["bbox_h"],
                    ],
                    "frame_dimensions": [row["frame_width"], row["frame_height"]],
                    "physical_size_measured": False,
                    "weight_measured": False,
                }
                if not candidates:
                    counts["without_classifier"] += 1
                elif any(name.endswith("_sp.") for name, _ in candidates):
                    details["not_applicable"] = "overlapping_genus_aggregate"
                    counts["genus_aggregate"] += 1
                else:
                    try:
                        occurred_at = capture_datetime(
                            row["captured_at"] or row["created_at"], timezone
                        )
                        occurrence, provenance = geo.predict(
                            latitude, longitude, occurred_at
                        )
                        assessment = rank_geographically(candidates, occurrence)
                        suggested = assessment.suggested_species
                        details["assessment"] = assessment.to_dict()
                        details["geography"] = provenance
                        counts["assessed"] += 1
                        counts["unmapped_candidates"] += sum(
                            candidate.occurrence_score is None
                            for candidate in assessment.ranking
                        )
                        if answer:
                            counts["labeled_assessed"] += 1
                            counts["image_correct"] += raw == answer
                            counts["geography_correct"] += suggested == answer
                            counts["wrong_to_correct"] += (
                                raw != answer and suggested == answer
                            )
                            counts["correct_to_wrong"] += (
                                raw == answer and suggested != answer
                            )
                    except (ValueError, TypeError) as exc:
                        details["error"] = f"{type(exc).__name__}: {exc}"
                        counts["invalid_input"] += 1
                output.execute(
                    "INSERT INTO assessments VALUES (?,?,?,?,?)",
                    (
                        detection_id,
                        raw,
                        suggested,
                        answer,
                        json.dumps(details, allow_nan=False),
                    ),
                )
                if focus_species and raw == focus_species:
                    focused.append(
                        {
                            "detection_id": detection_id,
                            "raw_species": raw,
                            "suggested_species": suggested,
                            "human_species": answer,
                        }
                    )
    return {
        "counts": dict(counts),
        "focused_cases": focused,
        "result_database": str(result_database.resolve()),
        "validation": "Canonical human labels used only for evaluation; one source snapshot",
        "score_semantics": "Uncalibrated image_score * sqrt(max(0.001, occurrence))",
        "application_integration": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--result-database", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--timezone", required=True)
    parser.add_argument(
        "--location", type=float, nargs=2, metavar=("LAT", "LON"), required=True
    )
    parser.add_argument("--focus-species")
    args = parser.parse_args()
    result = audit(
        args.database,
        args.result_database,
        geo=LocalGeoModel(args.model, args.labels),
        latitude=args.location[0],
        longitude=args.location[1],
        timezone=args.timezone,
        focus_species=args.focus_species,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
