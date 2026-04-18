#!/usr/bin/env python3
"""Benchmark every locally-installed detector variant against a fixed
frame set and compare detection counts + confidences.

Runs on the Raspberry Pi (has the ONNX models, the Pi CPU profile,
and the actual frames on disk). Lives in the repo on the Mac;
synced via ``sync_preview.sh main`` to ``/opt/app/…`` on the Pi.

## What it does

For each variant listed under ``latest_models.json[pinned_models]``
that has its weights + labels on disk:

1. load the ONNX model at the variant's own
   ``_model_config.yaml`` thresholds (conf, iou, input_size)
2. run inference over a frame sample:
     * ALL multi-detection frames from the last N days (DB query)
     * a sample of single-detection frames as a control group
3. record per-frame detection count + top confidences
4. print a comparison table and a few headline metrics:
     * % of frames with >=2 detections
     * avg detections/frame
     * avg confidence of the "second" detection (the one most
       likely to drop below the floor on weaker variants)

## What it does NOT do

- No DB writes. Pure read-only probe.
- No Live-detector swap. Inference runs in a parallel ORT session.
- No network. All files already local.

## Usage

    # default — 7 days, 300 multi + 300 single frames
    sudo -u watchmybirds python3 benchmark_detector_variants.py

    # custom
    sudo -u watchmybirds python3 benchmark_detector_variants.py \\
        --days 14 --multi-sample 500 --single-sample 200 \\
        --output /tmp/detector_benchmark.csv

    # include the live app's running model in the report even though
    # it is not in pinned_models (reads its model_id from the running
    # detector state)
    sudo -u watchmybirds python3 benchmark_detector_variants.py \\
        --pause-app

The ``--pause-app`` flag stops the app service for the duration of
the benchmark so the RPi can dedicate all cores to the probe. Without
it the benchmark runs side-by-side with the live detector at the
cost of higher latency per probe frame.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Make the app importable when the script is invoked from anywhere.
APP_ROOT = Path("/opt/app")
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

os.environ.setdefault("OUTPUT_DIR", str(APP_ROOT / "data" / "output"))
os.environ.setdefault("MODEL_BASE_PATH", str(APP_ROOT / "data" / "models"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

try:
    import yaml  # noqa: E402
except ImportError:
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------


def _model_dir() -> Path:
    return Path(os.environ["MODEL_BASE_PATH"]) / "object_detection"


def _read_latest_models(model_dir: Path) -> dict[str, Any]:
    p = model_dir / "latest_models.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def _discover_variants(model_dir: Path) -> list[dict[str, Any]]:
    """Return the per-precision benchmark entries whose files are on disk.

    Each ``pinned_models`` entry is expanded into one benchmark row per
    precision artefact actually present on disk:

      - fp32  : ``weights_path`` (required, the reference row)
      - int8  : ``weights_int8_path`` (QOperator mode; may fail to load on
                older ORT versions — the benchmark loop catches that)
      - qdq*  : every path in ``weights_int8_qdq_fallback_paths`` that
                exists, labelled by the filename suffix after
                ``_best_int8_`` (``qdq``, ``qdq_u8a``, ``qdq_pt``,
                ``qdq_u8a_pt``).

    Rows missing their ONNX file on disk are skipped silently so the
    benchmark stays usable on stations that only have a subset of
    variants deployed.
    """

    def _suffix_label(path: Path) -> str:
        name = path.name
        marker = "_best_int8_"
        if marker in name:
            return name.split(marker, 1)[1].removesuffix(".onnx")
        if name.endswith("_best_int8.onnx"):
            return "int8_qop"
        if name.endswith("_best.onnx"):
            return "fp32"
        return name.removesuffix(".onnx")

    data = _read_latest_models(model_dir)
    pinned = data.get("pinned_models") or {}
    if not isinstance(pinned, dict):
        return []

    base = Path(os.environ["MODEL_BASE_PATH"])
    variants: list[dict[str, Any]] = []
    for mid, entry in sorted(pinned.items()):
        if not isinstance(entry, dict):
            continue
        labels_rel = entry.get("labels_path", "")
        labels_abs = base / labels_rel
        yaml_abs = model_dir / f"{mid}_model_config.yaml"
        if not labels_abs.exists():
            continue

        candidate_paths: list[str] = []
        fp32 = entry.get("weights_path", "")
        if fp32:
            candidate_paths.append(fp32)
        int8_qop = entry.get("weights_int8_path", "")
        if int8_qop and int8_qop not in candidate_paths:
            candidate_paths.append(int8_qop)
        qdq_list = entry.get("weights_int8_qdq_fallback_paths") or []
        if isinstance(qdq_list, list):
            for p in qdq_list:
                if isinstance(p, str) and p and p not in candidate_paths:
                    candidate_paths.append(p)

        for rel in candidate_paths:
            weights_abs = base / rel
            if not weights_abs.exists():
                continue
            suffix = _suffix_label(weights_abs)
            variants.append(
                {
                    "id": f"{mid}::{suffix}",
                    "weights": weights_abs,
                    "labels": labels_abs,
                    "yaml": yaml_abs if yaml_abs.exists() else None,
                    "variant_label": entry.get("variant", ""),
                    "precision": suffix,
                }
            )
    return variants


def _load_yaml_thresholds(yaml_path: Path | None) -> tuple[float, float, list[int]]:
    """Return (conf, iou, input_size) from a variant YAML, with fallbacks."""
    fallback = (0.15, 0.50, [640, 640])
    if yaml_path is None or yaml is None:
        return fallback
    try:
        data = yaml.safe_load(yaml_path.read_text())
        det = (data or {}).get("detection", {}) or {}
        conf = float(det.get("confidence_threshold", fallback[0]))
        iou = float(det.get("nms_iou_threshold", fallback[1]))
        size = det.get("input_size") or fallback[2]
        if isinstance(size, list) and len(size) >= 2:
            size = [int(size[0]), int(size[1])]
        else:
            size = fallback[2]
        return conf, iou, size
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Inference (stand-alone, no Detector class — keeps the probe pure)
# ---------------------------------------------------------------------------


def _load_labels(labels_path: Path) -> list[str]:
    data = json.loads(labels_path.read_text())
    if isinstance(data, list):
        return [str(v) for v in data]
    if isinstance(data, dict):
        # dict form: {"0": "bird", ...} — sort by numeric key
        items = sorted(data.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _, v in items]
    raise ValueError(f"Unsupported labels.json format in {labels_path}")


def _preprocess_letterbox(img: np.ndarray, in_w: int, in_h: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(in_w / w, in_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    chw = padded.transpose(2, 0, 1).astype(np.float32)
    return chw[None, ...], scale


def _decode_yolox_raw(
    raw: np.ndarray,
    scale: float,
    orig_w: int,
    orig_h: int,
    in_w: int,
    in_h: int,
    conf_thr: float,
    iou_thr: float,
) -> list[dict[str, Any]]:
    if raw.ndim == 3:
        raw = raw[0]
    boxes_xywh = raw[:, 0:4]
    obj_conf = raw[:, 4:5]
    cls_probs = raw[:, 5:]
    scores = obj_conf * cls_probs
    cls_ids = scores.argmax(axis=1)
    cls_scores = scores.max(axis=1)

    keep = cls_scores > conf_thr
    if not np.any(keep):
        return []
    boxes_xywh = boxes_xywh[keep]
    cls_ids = cls_ids[keep]
    cls_scores = cls_scores[keep]

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    max_coord = max(orig_w, orig_h, in_w, in_h) + 1
    offsets = cls_ids.astype(np.float32) * float(max_coord)
    boxes_nms = boxes_xyxy.copy()
    boxes_nms[:, 0] += offsets
    boxes_nms[:, 1] += offsets
    boxes_nms[:, 2] += offsets
    boxes_nms[:, 3] += offsets
    nms_xywh = np.stack(
        [
            boxes_nms[:, 0],
            boxes_nms[:, 1],
            boxes_nms[:, 2] - boxes_nms[:, 0],
            boxes_nms[:, 3] - boxes_nms[:, 1],
        ],
        axis=1,
    ).tolist()
    indices = cv2.dnn.NMSBoxes(nms_xywh, cls_scores.tolist(), conf_thr, iou_thr)
    if isinstance(indices, tuple) or len(indices) == 0:
        return []
    indices = np.array(indices).flatten()

    dets: list[dict[str, Any]] = []
    for i in indices:
        x1o = float(boxes_xyxy[i, 0]) / scale
        y1o = float(boxes_xyxy[i, 1]) / scale
        x2o = float(boxes_xyxy[i, 2]) / scale
        y2o = float(boxes_xyxy[i, 3]) / scale
        dets.append(
            {
                "class_id": int(cls_ids[i]),
                "confidence": float(cls_scores[i]),
                "x1": max(0, int(min(x1o, x2o))),
                "y1": max(0, int(min(y1o, y2o))),
                "x2": min(orig_w, int(max(x1o, x2o))),
                "y2": min(orig_h, int(max(y1o, y2o))),
            }
        )
    return dets


def _make_session(weights: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# Frame sampling (read-only DB + filesystem)
# ---------------------------------------------------------------------------


def _sample_frames(
    db_path: Path,
    days: int,
    multi_sample: int,
    single_sample: int,
) -> tuple[list[str], list[str]]:
    """Return (multi_frame_paths, single_frame_paths) from the DB over the
    last ``days`` days of captured frames."""
    originals = APP_ROOT / "data" / "output" / "originals"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Multi-detection frames — the critical ones.
        cur = conn.execute(
            f"""
            SELECT d.image_filename, COUNT(*) AS n
            FROM detections d
            WHERE d.status = 'active'
              AND d.created_at >= datetime('now', '-{int(days)} days')
            GROUP BY d.image_filename
            HAVING n >= 2
            ORDER BY d.image_filename DESC
            LIMIT ?
            """,
            (multi_sample,),
        )
        multi_names = [row[0] for row in cur.fetchall()]

        # Single-detection frames — control group.
        cur = conn.execute(
            f"""
            SELECT d.image_filename
            FROM detections d
            WHERE d.status = 'active'
              AND d.created_at >= datetime('now', '-{int(days)} days')
            GROUP BY d.image_filename
            HAVING COUNT(*) = 1
            ORDER BY d.image_filename DESC
            LIMIT ?
            """,
            (single_sample,),
        )
        single_names = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    # Resolve filenames to real paths (daily subdirs).
    def _resolve(name: str) -> str | None:
        # Filename format: 20260418_064709_308327.jpg
        if len(name) < 8 or not name[:8].isdigit():
            return None
        day = f"{name[0:4]}-{name[4:6]}-{name[6:8]}"
        path = originals / day / name
        return str(path) if path.exists() else None

    multi_paths = [p for p in (_resolve(n) for n in multi_names) if p]
    single_paths = [p for p in (_resolve(n) for n in single_names) if p]
    return multi_paths, single_paths


# ---------------------------------------------------------------------------
# Per-variant benchmark loop
# ---------------------------------------------------------------------------


def _benchmark_variant(
    variant: dict[str, Any],
    frames: list[str],
) -> dict[str, Any]:
    """Run inference of *variant* on every frame in *frames* and compute
    aggregate statistics."""
    conf_thr, iou_thr, input_size = _load_yaml_thresholds(variant.get("yaml"))
    in_w, in_h = int(input_size[0]), int(input_size[1])
    labels = _load_labels(variant["labels"])
    session = _make_session(variant["weights"])
    input_name = session.get_inputs()[0].name

    det_counts: list[int] = []
    top_confs: list[float] = []
    second_confs: list[float] = []
    latencies_ms: list[float] = []
    unique_classes: set[str] = set()

    for i, path in enumerate(frames):
        img = cv2.imread(path)
        if img is None:
            continue
        t0 = time.perf_counter()
        processed, scale = _preprocess_letterbox(img, in_w, in_h)
        outputs = session.run(None, {input_name: processed})
        orig_h, orig_w = img.shape[:2]
        dets = _decode_yolox_raw(
            outputs[0], scale, orig_w, orig_h, in_w, in_h, conf_thr, iou_thr
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        det_counts.append(len(dets))

        sorted_dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        if sorted_dets:
            top_confs.append(sorted_dets[0]["confidence"])
            for d in dets:
                unique_classes.add(labels[d["class_id"]])
        if len(sorted_dets) >= 2:
            second_confs.append(sorted_dets[1]["confidence"])

        if (i + 1) % 50 == 0:
            sys.stdout.write(f"    {variant['id']}: {i+1}/{len(frames)}\n")
            sys.stdout.flush()

    total = len(det_counts) or 1
    multi = sum(1 for n in det_counts if n >= 2)
    three_plus = sum(1 for n in det_counts if n >= 3)

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "id": variant["id"],
        "conf_thr": conf_thr,
        "iou_thr": iou_thr,
        "input_size": f"{in_w}x{in_h}",
        "frames_processed": total,
        "avg_detections_per_frame": _mean([float(n) for n in det_counts]),
        "frames_with_>=2": multi,
        "pct_multi": round(100.0 * multi / total, 2),
        "frames_with_>=3": three_plus,
        "pct_three_plus": round(100.0 * three_plus / total, 2),
        "avg_top_conf": _mean(top_confs),
        "avg_second_conf": _mean(second_confs),
        "avg_latency_ms": _mean(latencies_ms),
        "unique_classes_seen": sorted(unique_classes),
    }


# ---------------------------------------------------------------------------
# Pretty print + CSV
# ---------------------------------------------------------------------------


def _print_report(results: list[dict[str, Any]], frame_count: int) -> None:
    print()
    width = 108
    print("=" * width)
    print(f"Detector variant benchmark  —  {frame_count} frames")
    print("=" * width)
    header = (
        f"{'variant':60s}  {'inp':8s}  {'conf':>5s}  {'avg/frm':>7s}  "
        f"{'%multi':>6s}  {'%3+':>5s}  {'top':>5s}  {'2nd':>5s}  {'ms':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        label = r["id"]
        if len(label) > 60:
            label = label[:57] + "..."
        print(
            f"{label:60s}  {r['input_size']:8s}  "
            f"{r['conf_thr']:>5.2f}  {r['avg_detections_per_frame']:>7.3f}  "
            f"{r['pct_multi']:>5.1f}%  {r['pct_three_plus']:>4.1f}%  "
            f"{r['avg_top_conf']:>5.2f}  {r['avg_second_conf']:>5.2f}  "
            f"{r['avg_latency_ms']:>7.1f}"
        )
    print("=" * width)
    print()


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    fieldnames = [
        "id",
        "input_size",
        "conf_thr",
        "iou_thr",
        "frames_processed",
        "avg_detections_per_frame",
        "frames_with_>=2",
        "pct_multi",
        "frames_with_>=3",
        "pct_three_plus",
        "avg_top_conf",
        "avg_second_conf",
        "avg_latency_ms",
        "unique_classes_seen",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = dict(r)
            row["unique_classes_seen"] = "|".join(row["unique_classes_seen"])
            w.writerow(row)


# ---------------------------------------------------------------------------
# App service pause helper
# ---------------------------------------------------------------------------


@contextmanager
def _paused_app(enabled: bool):
    """Stop the watchmybirds app service for the duration of the context
    (best-effort). No-op if ``enabled`` is False or systemctl fails."""
    stopped = False
    if enabled:
        try:
            subprocess.run(["systemctl", "stop", "app"], check=True)
            stopped = True
            print("app.service stopped for benchmark duration")
        except Exception as exc:
            print(f"could not stop app service: {exc} (continuing alongside)")
    try:
        yield
    finally:
        if stopped:
            try:
                subprocess.run(["systemctl", "start", "app"], check=True)
                print("app.service restarted")
            except Exception as exc:
                print(f"WARN: failed to restart app service: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--multi-sample", type=int, default=300)
    parser.add_argument("--single-sample", type=int, default=300)
    parser.add_argument("--db", default=str(APP_ROOT / "data" / "output" / "images.db"))
    parser.add_argument("--output", default="/tmp/detector_benchmark.csv")
    parser.add_argument(
        "--pause-app",
        action="store_true",
        help="systemctl stop app.service for the benchmark duration",
    )
    args = parser.parse_args()

    model_dir = _model_dir()
    variants = _discover_variants(model_dir)
    if not variants:
        print(f"No installed variants found under {model_dir}/pinned_models", file=sys.stderr)
        return 2

    print(f"Found {len(variants)} installed variant(s):")
    for v in variants:
        print(f"  - {v['id']}  weights={v['weights'].name}  yaml={'yes' if v['yaml'] else 'no'}")

    print(
        f"\nSampling frames from {args.db}: "
        f"last {args.days} days, up to {args.multi_sample} multi + {args.single_sample} single"
    )
    multi, single = _sample_frames(
        Path(args.db), args.days, args.multi_sample, args.single_sample
    )
    frames = multi + single
    print(
        f"Got {len(multi)} multi-detection frames + {len(single)} single-detection frames "
        f"= {len(frames)} total"
    )
    if not frames:
        print("No frames — nothing to benchmark", file=sys.stderr)
        return 3

    results: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    with _paused_app(args.pause_app):
        for v in variants:
            print(f"\n>>> Benchmarking {v['id']} (input conf from yaml)")
            t0 = time.perf_counter()
            try:
                stats = _benchmark_variant(v, frames)
            except Exception as exc:
                # Common on this Pi: old QOperator int8 files throw
                # NOT_IMPLEMENTED for ConvInteger on ORT 1.16.0 ARM64.
                # Keep the benchmark going and report the skip at the end.
                msg = f"{type(exc).__name__}: {exc}".splitlines()[0][:180]
                print(f"  SKIP (load/infer failed): {msg}")
                skipped.append((v["id"], msg))
                continue
            elapsed = time.perf_counter() - t0
            print(
                f"  done in {elapsed:.1f}s "
                f"({stats['avg_latency_ms']:.1f} ms/frame avg, "
                f"{stats['pct_multi']}% multi, "
                f"{stats['pct_three_plus']}% 3+)"
            )
            results.append(stats)

    _print_report(results, len(frames))
    _write_csv(Path(args.output), results)
    print(f"CSV written to {args.output}")

    if skipped:
        print("\nSkipped variants (did not load / infer on this ORT):")
        for vid, msg in skipped:
            print(f"  - {vid}: {msg}")

    # For a fp32↔int8 comparison, the three metrics that matter are:
    #   1) ms/frame    — did int8 actually get faster on this hardware?
    #   2) pct_multi   — did int8 break multi-bird recall?
    #   3) avg_second_conf — did int8 thin out weak detections?
    # Report all three per stem group so the caller can see the trade-off
    # without reading the full CSV.
    if len(results) >= 2:
        fastest = min(results, key=lambda r: r["avg_latency_ms"])
        best_multi = max(results, key=lambda r: r["pct_multi"])
        print(f"\nFastest: {fastest['id']} ({fastest['avg_latency_ms']:.1f} ms/frame)")
        print(f"Best multi-bird recall: {best_multi['id']} ({best_multi['pct_multi']}%)")

        # Grouped fp32-vs-int8 deltas per model stem.
        by_stem: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            stem = r["id"].split("::", 1)[0]
            by_stem.setdefault(stem, []).append(r)
        for stem, rows in by_stem.items():
            fp32 = next((r for r in rows if r["id"].endswith("::fp32")), None)
            if not fp32:
                continue
            print(f"\n  {stem}")
            print(
                f"    fp32 baseline: {fp32['avg_latency_ms']:6.1f} ms  "
                f"pct_multi={fp32['pct_multi']:.1f}%  "
                f"2nd={fp32['avg_second_conf']:.2f}"
            )
            for r in rows:
                if r is fp32:
                    continue
                ms_delta = r["avg_latency_ms"] - fp32["avg_latency_ms"]
                speedup = fp32["avg_latency_ms"] / r["avg_latency_ms"] if r["avg_latency_ms"] else 0.0
                pm_delta = r["pct_multi"] - fp32["pct_multi"]
                c2_delta = r["avg_second_conf"] - fp32["avg_second_conf"]
                suffix = r["id"].rsplit("::", 1)[-1]
                print(
                    f"    {suffix:12s}: {r['avg_latency_ms']:6.1f} ms  "
                    f"({ms_delta:+6.1f}, x{speedup:.2f})  "
                    f"pct_multi={r['pct_multi']:.1f}% ({pm_delta:+.1f} pp)  "
                    f"2nd={r['avg_second_conf']:.2f} ({c2_delta:+.2f})"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
