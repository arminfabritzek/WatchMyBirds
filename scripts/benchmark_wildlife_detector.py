#!/usr/bin/env python3
"""Benchmark an Ultralytics wildlife detector on the frozen field dataset.

This optional tool keeps third-party model dependencies out of the application.
It runs inference once at the lowest requested threshold, then reports the
field false-positive and provisional recall trade-off for every threshold.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _images(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return [
        str(candidate)
        for candidate in sorted(path.iterdir())
        if candidate.is_file() and candidate.suffix.casefold() in IMAGE_SUFFIXES
    ]


def _threshold_rows(
    *,
    model_name: str,
    negative_scores: list[float],
    positive_scores: list[float],
    thresholds: list[float],
    negative_latency_ms: float,
    positive_latency_ms: float,
) -> list[dict[str, Any]]:
    rows = []
    for threshold in sorted(set(thresholds)):
        false_positives = sum(score >= threshold for score in negative_scores)
        detected_positives = sum(score >= threshold for score in positive_scores)
        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "negative_n": len(negative_scores),
                "false_positive_frames": false_positives,
                "field_false_positive_rate_pct": round(
                    100.0 * false_positives / len(negative_scores), 2
                ),
                "field_fp_rejection_pct": round(
                    100.0 * (len(negative_scores) - false_positives)
                    / len(negative_scores),
                    2,
                ),
                "positive_n": len(positive_scores),
                "positive_detected": detected_positives,
                "field_recall_pct": round(
                    100.0 * detected_positives / len(positive_scores), 2
                ),
                "negative_latency_ms": round(negative_latency_ms, 2),
                "positive_latency_ms": round(positive_latency_ms, 2),
            }
        )
    return rows


def _animal_scores(
    model: Any,
    images: list[str],
    *,
    conf: float,
    image_size: int,
    device: str,
    batch: int,
    chunk_size: int,
) -> tuple[list[float], float]:
    started = time.perf_counter()
    scores: list[float] = []
    for start in range(0, len(images), chunk_size):
        chunk = images[start : start + chunk_size]
        predictions = model.predict(
            source=chunk,
            conf=conf,
            imgsz=image_size,
            device=device,
            batch=batch,
            stream=False,
            verbose=False,
        )
        for result in predictions:
            names = result.names
            animal_ids = {
                int(class_id)
                for class_id, name in names.items()
                if str(name).strip().casefold() == "animal"
            }
            if not animal_ids:
                raise ValueError("model does not expose an 'animal' class")
            animal_confidences = [
                float(confidence)
                for class_id, confidence in zip(
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    strict=True,
                )
                if int(class_id) in animal_ids
            ]
            scores.append(max(animal_confidences, default=0.0))
        del predictions
        gc.collect()
        if device.casefold() == "mps":
            import torch

            torch.mps.empty_cache()
        print(f"processed {min(start + chunk_size, len(images))}/{len(images)}", flush=True)
    if len(scores) != len(images):
        raise ValueError("model did not return exactly one result per image")
    elapsed = time.perf_counter() - started
    return scores, 1000.0 * elapsed / len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="wildlife-detector")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--negative-limit", type=int)
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="repeatable; defaults to 0.1, 0.2, 0.3, 0.5, 0.7, 0.9",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset)
    negatives = _images(dataset / "negative")
    positives = _images(dataset / "positive")
    if args.negative_limit is not None:
        negatives = negatives[: max(1, args.negative_limit)]
    if args.positive_limit is not None:
        positives = positives[: max(1, args.positive_limit)]
    if not negatives or not positives:
        raise SystemExit("dataset needs non-empty negative/ and positive/ directories")
    thresholds = args.threshold or [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    if any(threshold <= 0.0 or threshold > 1.0 for threshold in thresholds):
        raise SystemExit("thresholds must be in (0, 1]")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    inference_floor = min(thresholds)
    negative_scores, negative_latency = _animal_scores(
        model,
        negatives,
        conf=inference_floor,
        image_size=args.image_size,
        device=args.device,
        batch=max(1, args.batch),
        chunk_size=max(1, args.chunk_size),
    )
    positive_scores, positive_latency = _animal_scores(
        model,
        positives,
        conf=inference_floor,
        image_size=args.image_size,
        device=args.device,
        batch=max(1, args.batch),
        chunk_size=max(1, args.chunk_size),
    )
    rows = _threshold_rows(
        model_name=args.model_name,
        negative_scores=negative_scores,
        positive_scores=positive_scores,
        thresholds=thresholds,
        negative_latency_ms=negative_latency,
        positive_latency_ms=positive_latency,
    )
    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
