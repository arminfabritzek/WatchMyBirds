#!/usr/bin/env python3
"""Benchmark an end-to-end YOLO wildlife ONNX model on field evidence."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from benchmark_wildlife_detector import _images, _threshold_rows


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def _animal_score(output: np.ndarray, animal_class_id: int) -> float:
    rows = output[0] if output.ndim == 3 else output
    if rows.ndim != 2 or rows.shape[1] < 6:
        raise ValueError(f"unsupported end-to-end YOLO output shape: {output.shape}")
    scores = [
        float(row[4])
        for row in rows
        if int(round(float(row[5]))) == animal_class_id and float(row[4]) > 0.0
    ]
    return max(scores, default=0.0)


def _score_images(
    session: ort.InferenceSession,
    images: list[str],
    *,
    image_size: int,
    animal_class_id: int,
) -> tuple[list[float], float]:
    input_name = session.get_inputs()[0].name
    scores = []
    started = time.perf_counter()
    for index, path in enumerate(images, start=1):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"could not decode image: {path}")
        output = session.run(None, {input_name: _letterbox(image, image_size)})[0]
        scores.append(_animal_score(output, animal_class_id))
        if index % 50 == 0 or index == len(images):
            print(f"processed {index}/{len(images)}", flush=True)
    elapsed = time.perf_counter() - started
    return scores, 1000.0 * elapsed / len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", default="wildlife-onnx")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--animal-class-id", type=int, default=0)
    parser.add_argument("--negative-limit", type=int)
    parser.add_argument("--positive-limit", type=int)
    parser.add_argument("--threshold", action="append", type=float, default=[])
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
    thresholds = args.threshold or [0.01, 0.02, 0.05, 0.1]
    if any(threshold <= 0.0 or threshold > 1.0 for threshold in thresholds):
        raise SystemExit("thresholds must be in (0, 1]")

    session = ort.InferenceSession(
        args.weights,
        providers=["CPUExecutionProvider"],
    )
    negative_scores, negative_latency = _score_images(
        session,
        negatives,
        image_size=args.image_size,
        animal_class_id=args.animal_class_id,
    )
    positive_scores, positive_latency = _score_images(
        session,
        positives,
        image_size=args.image_size,
        animal_class_id=args.animal_class_id,
    )
    rows: list[dict[str, Any]] = _threshold_rows(
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
