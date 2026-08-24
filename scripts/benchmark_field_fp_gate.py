#!/usr/bin/env python3
"""Compare installed bird detectors on labeled field negatives and positives.

The input directories are produced by ``export_field_benchmark_set.py``.
Inference is read-only and never changes the station database or source images.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from benchmark_detector_variants import (
    _benchmark_variant,
    _discover_variants,
    _model_dir,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _images(path: Path) -> list[str]:
    return [
        str(candidate)
        for candidate in sorted(path.iterdir())
        if candidate.is_file() and candidate.suffix.casefold() in IMAGE_SUFFIXES
    ]


def _score_variant(
    variant: dict[str, Any], negatives: list[str], positives: list[str]
) -> dict[str, Any]:
    started = time.perf_counter()
    negative_stats = _benchmark_variant(variant, negatives)
    positive_stats = _benchmark_variant(variant, positives)
    if negative_stats["frames_processed"] != len(negatives):
        raise ValueError("one or more negative images could not be decoded")
    if positive_stats["frames_processed"] != len(positives):
        raise ValueError("one or more positive images could not be decoded")
    elapsed = time.perf_counter() - started
    return {
        "id": variant["id"],
        "negative_n": negative_stats["frames_processed"],
        "false_positive_frames": negative_stats["bird_frames"],
        "field_false_positive_rate_pct": negative_stats["bird_frame_rate_pct"],
        "field_fp_rejection_pct": round(
            100.0 - float(negative_stats["bird_frame_rate_pct"]), 2
        ),
        "positive_n": positive_stats["frames_processed"],
        "positive_detected": positive_stats["bird_frames"],
        "field_recall_pct": positive_stats["bird_frame_rate_pct"],
        "negative_latency_ms": negative_stats["avg_latency_ms"],
        "positive_latency_ms": positive_stats["avg_latency_ms"],
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="/tmp/wmb_field_fp_benchmark.csv")
    parser.add_argument("--variant", action="append", default=[])
    args = parser.parse_args()

    dataset = Path(args.dataset)
    negatives = _images(dataset / "negative")
    positives = _images(dataset / "positive")
    if not negatives or not positives:
        raise SystemExit("dataset needs non-empty negative/ and positive/ directories")

    variants = _discover_variants(_model_dir())
    if args.variant:
        requested = set(args.variant)
        variants = [variant for variant in variants if variant["id"] in requested]
    if not variants:
        raise SystemExit("no matching installed detector variants")

    results = []
    for variant in variants:
        print(f">>> {variant['id']}", flush=True)
        try:
            result = _score_variant(variant, negatives, positives)
        except Exception as exc:
            print(f"SKIP {variant['id']}: {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(result)
        print(json.dumps(result), flush=True)

    output = Path(args.output)
    if results:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps({"results": len(results), "output": str(output)}))
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
