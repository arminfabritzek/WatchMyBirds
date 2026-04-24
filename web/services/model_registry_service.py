"""Model-registry read service: maps the on-disk model cache into a
UI-friendly JSON payload.

It answers two questions the AI settings panel needs:

1. **What is the detector loaded right now, and what are the knobs?**
   The Flask route serializes this as the GET response.

2. **Which alternate variants are known locally?**
   That is derived from ``latest_models.json`` — specifically the
   ``pinned_models`` dict shipped with each model release.

The service is deliberately read-only. Writing (switching the active
variant) lives in :func:`utils.model_downloader.set_latest_model_id`,
invoked from the HTTP layer so the side effects stay close to the
request boundary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from config import get_config
from core.model_downloader_core import (
    HF_KNOWN_IDS_KEY,
    HF_LATEST_ADVERTISED_KEY,
    PIN_ENV_VAR,
    PIN_ENV_VAR_PREFIX,
    _resolve_pin_for_cache_dir,
    _task_name_from_cache_dir,
)
from logging_config import get_logger

logger = get_logger(__name__)


OBJECT_DETECTION_SUBDIR = "object_detection"
CLASSIFIER_SUBDIR = "classifier"


@dataclass
class VariantInfo:
    """One entry in ``latest_models.json[\"pinned_models\"]`` plus liveness flags."""

    id: str
    weights_path: str
    labels_path: str
    weights_exists: bool
    labels_exists: bool
    is_active: bool
    is_hf_latest: bool
    int8_qdq_available: bool = False
    active_precision: str = "fp32"
    metadata: dict[str, Any] | None = None
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "weights_path": self.weights_path,
            "labels_path": self.labels_path,
            "weights_exists": self.weights_exists,
            "labels_exists": self.labels_exists,
            "is_available_locally": self.weights_exists and self.labels_exists,
            "is_active": self.is_active,
            "is_hf_latest": self.is_hf_latest,
            "int8_qdq_available": self.int8_qdq_available,
            "active_precision": self.active_precision,
            "metadata": self.metadata or {},
            "tags": self.tags or [],
        }


def _model_dir() -> str:
    config = get_config()
    base = config.get("MODEL_BASE_PATH", "models")
    return os.path.join(base, OBJECT_DETECTION_SUBDIR)


def _classifier_model_dir() -> str:
    """Classifier model directory (parallel to _model_dir for the detector).

    Split into its own helper so the Classifier UI/API layer can share
    the same read-side helpers as the Detector without having to carry
    a ``subdir`` parameter through every call site.
    """
    config = get_config()
    base = config.get("MODEL_BASE_PATH", "models")
    return os.path.join(base, CLASSIFIER_SUBDIR)


def _read_active_metadata(model_dir: str) -> dict[str, Any]:
    """Read model_metadata.json, which the deploy pipeline generates for the active default."""
    path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return {}


def _read_latest_models(model_dir: str) -> dict[str, Any]:
    path = os.path.join(model_dir, "latest_models.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return {}


def _resolve_hf_latest_id(latest_data: dict[str, Any]) -> str | None:
    """Return HF's authoritative latest id, independent of local preservation.

    Prefers ``hf_latest_advertised`` (written on every successful HF merge
    — unaffected by the preservation guard) and falls back to the
    top-level ``latest`` for legacy JSONs or when the app has never
    synced with HF. This matters when local and remote pointers diverge:
    the UI should tag HF's choice as "Latest", not the preserved local
    active id.
    """
    if not isinstance(latest_data, dict):
        return None
    advertised = latest_data.get(HF_LATEST_ADVERTISED_KEY)
    if isinstance(advertised, str) and advertised.strip():
        return advertised
    top_level = latest_data.get("latest")
    return top_level if isinstance(top_level, str) else None


def _variant_from_id(model_id: str) -> str | None:
    """Best-effort guess of the YOLOX size (``s`` or ``tiny``) from the id.

    Used as a fallback for variants that have no ``_model_config.yaml``
    on disk yet (not-installed rows). Returns ``None`` when the id does
    not contain a recognisable token.
    """
    lower = model_id.lower()
    # Match the longest token first so "_tiny_" wins over the bare "_s_"
    # elsewhere in the id.
    for token, name in (
        ("_tiny_", "tiny"),
        ("_yolox_s_", "s"),
        ("_yolox_tiny_", "tiny"),
    ):
        if token in lower:
            return name
    return None


def _released_from_id(model_id: str) -> str | None:
    """Extract the ``YYYY-MM-DD`` prefix from an id like ``20260421_foo``."""
    if len(model_id) < 8 or not model_id[:8].isdigit():
        return None
    return f"{model_id[:4]}-{model_id[4:6]}-{model_id[6:8]}"


def _read_variant_companions(
    model_dir: str, model_id: str
) -> dict[str, Any]:
    """Collect lightweight metadata for a single variant from local files.

    Reads ``<id>_metrics.json`` (if present) for recall/F1 and
    ``<id>_model_config.yaml`` (if present) for variant label, input size
    and confidence/IoU thresholds. Missing files are silently skipped —
    not-installed variants typically have none of these, so the UI falls
    back to id-derived hints (release date, variant-from-id).
    """
    out: dict[str, Any] = {}

    # Metrics JSON — optional
    metrics_path = os.path.join(model_dir, f"{model_id}_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as file:
                metrics = json.load(file)
            if isinstance(metrics, dict):
                chosen = metrics.get("chosen_threshold")
                if isinstance(chosen, dict):
                    for key in ("bird_recall", "bird_precision", "f1", "conf"):
                        value = chosen.get(key)
                        if isinstance(value, (int, float)):
                            out[key] = float(value)
                train = metrics.get("train_info")
                if isinstance(train, dict):
                    trained_at = train.get("trained_at")
                    if isinstance(trained_at, str):
                        out["trained_at"] = trained_at
        except Exception as exc:
            logger.debug(f"variant metadata: failed to read {metrics_path}: {exc}")

    # Model config YAML — optional
    yaml_path = os.path.join(model_dir, f"{model_id}_model_config.yaml")
    if os.path.exists(yaml_path):
        try:
            import yaml as _yaml

            with open(yaml_path, encoding="utf-8") as file:
                config = _yaml.safe_load(file)
            if isinstance(config, dict):
                detection = config.get("detection")
                if isinstance(detection, dict):
                    conf = detection.get("confidence_threshold")
                    iou = detection.get("nms_iou_threshold")
                    input_size = detection.get("input_size")
                    architecture = detection.get("architecture")
                    if isinstance(conf, (int, float)):
                        # YAML conf wins over metrics.json conf — it is the
                        # actual runtime threshold, metrics conf is just the
                        # chosen sweep point (usually the same).
                        out["conf"] = float(conf)
                    if isinstance(iou, (int, float)):
                        out["iou"] = float(iou)
                    if (
                        isinstance(input_size, list)
                        and len(input_size) == 2
                        and all(isinstance(n, int) for n in input_size)
                    ):
                        out["input_size"] = list(input_size)
                    if isinstance(architecture, str) and architecture.strip():
                        # e.g. "yolox_tiny_locator_5cls" -> "tiny"
                        arch_lower = architecture.lower()
                        if "_tiny_" in arch_lower:
                            out["variant"] = "tiny"
                        elif "_s_" in arch_lower:
                            out["variant"] = "s"
                meta = config.get("meta")
                if isinstance(meta, dict):
                    num_classes = meta.get("num_classes")
                    if isinstance(num_classes, int):
                        out["num_classes"] = num_classes
                    trained_at = meta.get("trained_at")
                    if isinstance(trained_at, str) and "trained_at" not in out:
                        out["trained_at"] = trained_at
        except ImportError:
            logger.debug("variant metadata: PyYAML unavailable; skipping YAML read")
        except Exception as exc:
            logger.debug(f"variant metadata: failed to read {yaml_path}: {exc}")

    return out


def _compute_variant_tags(
    variants: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Assign descriptive tags to each variant so end users can compare them
    without decoding the id.

    Tags are earned strictly from the data already in each variant's
    ``metadata`` block — never editorialized ("best", "recommended" would
    imply a judgement call that depends on hardware and use case).

    Rules (any combination can apply):

    - ``Latest s`` / ``Latest tiny``:
                         released on the most recent date *within its
                         variant family*. When multiple variants share
                         the top date, the tie is broken by
                         ``bird_recall`` (highest wins). If that still
                         ties, the tag is skipped — "newest" stops being
                         a useful hint when three releases all look
                         equally fresh. Variants without a detected
                         size fall back to a whole-list ``Latest``.
    - ``Small, faster``: ``variant: tiny`` AND an ``s`` sibling exists in
                         the same list (so the comparison is meaningful).
    - ``Bigger, slower, better``: ``variant: s`` AND a ``tiny`` sibling exists.
    - ``Highest recall``: has the highest ``bird_recall`` among variants
                         whose metadata carries a recall number. Ties get
                         no badge (ambiguous).
    """
    tags_by_id: dict[str, list[str]] = {v["id"]: [] for v in variants}

    # Latest per variant family. Bucket dated entries by their detected
    # size, then tag the latest in each bucket ("Latest s", "Latest tiny").
    # Entries without a detected size fall back to a whole-list "Latest".
    # Tie-breaker: within the top date, highest bird_recall wins. If the
    # recall also ties (or is missing on all contenders), skip the tag —
    # labelling three entries "Latest s" makes the hint meaningless.
    dated_by_size: dict[str | None, list[tuple[str, str, float | None]]] = {}
    for v in variants:
        released = (v.get("metadata") or {}).get("released")
        if not (isinstance(released, str) and released):
            continue
        size = (v.get("metadata") or {}).get("variant")
        key = size if isinstance(size, str) and size in ("s", "tiny") else None
        recall_raw = (v.get("metadata") or {}).get("bird_recall")
        recall = float(recall_raw) if isinstance(recall_raw, (int, float)) else None
        dated_by_size.setdefault(key, []).append((v["id"], released, recall))

    for size_key, entries in dated_by_size.items():
        newest_date = max(date for _, date, _ in entries)
        top_dated = [(vid, recall) for vid, d, recall in entries if d == newest_date]
        label = f"Latest {size_key}" if size_key else "Latest"

        if len(top_dated) == 1:
            tags_by_id[top_dated[0][0]].append(label)
            continue

        # Multiple share the top date -> break tie by recall.
        recalls_present = [(vid, r) for vid, r in top_dated if r is not None]
        if len(recalls_present) < len(top_dated):
            # Some contenders have no recall -> comparison is unfair, skip.
            continue
        top_recall = max(r for _, r in recalls_present)
        winners = [vid for vid, r in recalls_present if r == top_recall]
        if len(winners) == 1:
            tags_by_id[winners[0]].append(label)

    # Speed/accuracy badges only when BOTH sizes are present (otherwise
    # the label is meaningless — "faster than what?").
    sizes = {
        (v.get("metadata") or {}).get("variant")
        for v in variants
        if isinstance((v.get("metadata") or {}).get("variant"), str)
    }
    has_both = "s" in sizes and "tiny" in sizes
    if has_both:
        for v in variants:
            size = (v.get("metadata") or {}).get("variant")
            if size == "tiny":
                tags_by_id[v["id"]].append("Small, faster")
            elif size == "s":
                tags_by_id[v["id"]].append("Bigger, slower, better")

    # Highest recall among variants that have one.
    recalls: list[tuple[str, float]] = []
    for v in variants:
        recall = (v.get("metadata") or {}).get("bird_recall")
        if isinstance(recall, (int, float)):
            recalls.append((v["id"], float(recall)))
    if recalls:
        top = max(r for _, r in recalls)
        winners = [vid for vid, r in recalls if r == top]
        # Skip when multiple variants tie — "highest" stops being useful.
        if len(winners) == 1:
            tags_by_id[winners[0]].append("Highest recall")

    return tags_by_id


def _sort_variants_newest_first(
    variants: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort variants by release date descending, then by id for stability.

    Variants without a release date land at the bottom so user attention
    goes to dated releases first.
    """
    def sort_key(v: dict[str, Any]) -> tuple[int, str, str]:
        released = (v.get("metadata") or {}).get("released") or ""
        # Sort descending by date by inverting — simpler than reverse=True
        # because we still want ascending id as the tiebreaker.
        has_date = 0 if released else 1  # dated entries first
        return (has_date, _invert_date_for_desc(released), v["id"])
    return sorted(variants, key=sort_key)


def _invert_date_for_desc(iso_date: str) -> str:
    """Return a comparable key that sorts ISO dates descending when used
    with the default ascending sort. Empty dates map to the lowest key.
    """
    if not iso_date:
        return ""
    # Map YYYY-MM-DD to a string that inverts digit-for-digit so
    # '2026-04-21' > '2026-04-20' becomes '1313939878...' < '...'
    # A simpler trick: pad with the max value and subtract per-character.
    # Even simpler still: sort by negative-date via tuple of ints.
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in iso_date)


def _build_variant_metadata(
    model_dir: str,
    model_id: str,
    registry_entry: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the metadata block the UI shows under a variant row.

    Source order (first wins per key):
      1. Local ``<id>_metrics.json`` + ``<id>_model_config.yaml``
      2. Fields the registry entry itself carries (e.g. ``variant``)
      3. Id-derived hints (release date, variant-from-token)
    """
    meta = _read_variant_companions(model_dir, model_id)

    # Fill variant from registry entry when YAML did not supply it.
    if "variant" not in meta:
        registry_variant = registry_entry.get("variant")
        if isinstance(registry_variant, str) and registry_variant.strip():
            meta["variant"] = registry_variant.strip().lower()
    # Last-ditch id-derived fallback.
    if "variant" not in meta:
        derived = _variant_from_id(model_id)
        if derived:
            meta["variant"] = derived

    if "released" not in meta:
        released = _released_from_id(model_id)
        if released:
            meta["released"] = released

    return meta


def _detect_active_source(model_dir: str) -> str:
    """Return which resolver decided the current active model id.

    - "env_pin_task"   : task-scoped env var (systemd drop-in, etc.)
    - "env_pin_generic": generic fallback env var
    - "latest_models"  : whatever ``latest_models.json["latest"]`` points at
                         (both the UI switch and the HF default land here,
                         distinguishable only by looking at which side last
                         wrote the file)
    """
    task = _task_name_from_cache_dir(model_dir)
    if os.environ.get(f"{PIN_ENV_VAR_PREFIX}_{task}", "").strip():
        return "env_pin_task"
    if os.environ.get(PIN_ENV_VAR, "").strip():
        return "env_pin_generic"
    return "latest_models"


def build_detector_registry_payload(detector: Any | None) -> dict[str, Any]:
    """Assemble the GET /api/v1/models/detector response body.

    Args:
        detector: Optional reference to the live ``ONNXDetectionModel``
            (usually ``DetectionManager.detection_service._detector``).
            When provided, the ``runtime`` block is populated with the
            model that is **actually loaded**, not just what's on disk.

    Returns:
        JSON-serializable dict.
    """
    model_dir = _model_dir()
    latest = _read_latest_models(model_dir)
    metadata = _read_active_metadata(model_dir)

    # Active id on disk = what the app will pick up on next load. That is
    # either the pin (if any) or latest_models["latest"].
    hf_latest_id: str | None = _resolve_hf_latest_id(latest)
    active_source = _detect_active_source(model_dir)

    pinned_models = latest.get("pinned_models") if isinstance(latest, dict) else None
    if not isinstance(pinned_models, dict):
        pinned_models = {}

    # Effective active id when the app next loads: pin (any source) wins
    # over the on-disk top-level ``latest`` pointer. Deliberately reads
    # the local JSON's own ``latest`` (preservation-guarded), NOT HF's
    # advertised one — those diverge when the guard keeps the local
    # active because HF's new files are not on disk yet.
    local_top_latest = latest.get("latest") if isinstance(latest, dict) else None
    pin_value = _resolve_pin_for_cache_dir(model_dir)
    active_on_disk_id = pin_value or local_top_latest

    # Live runtime id = the model currently in-memory.
    runtime_id = None
    runtime: dict[str, Any] = {}
    if detector is not None:
        runtime_id = getattr(detector, "model_id", None) or None
        runtime = {
            "model_id": runtime_id,
            "model_path": getattr(detector, "model_path", None),
            "output_format": getattr(detector, "output_format", None),
            "input_size": list(getattr(detector, "input_size", ()) or ()),
            "num_classes": len(getattr(detector, "class_names", {}) or {}),
            "class_names": list((getattr(detector, "class_names", {}) or {}).values()),
            "conf_threshold_default": getattr(detector, "conf_threshold_default", None),
            "iou_threshold_default": getattr(detector, "iou_threshold_default", None),
        }

    # Build variants list. `pinned_models` may declare alternate variants
    # shipped with the release; latest_models["latest"] (if not already
    # listed) is merged in so the UI can always show the current default
    # row.
    variant_entries = dict(pinned_models)
    if hf_latest_id and hf_latest_id not in variant_entries:
        variant_entries[hf_latest_id] = {
            "weights_path": latest.get("weights_path", ""),
            "labels_path": latest.get("labels_path", ""),
        }

    # Top-level precision hint: used as the fallback when a per-variant
    # entry doesn't carry its own ``active_precision`` (true for the
    # simplest registries that only know the active default).
    top_level_precision = (
        str(latest.get("active_precision", "fp32"))
        if isinstance(latest, dict)
        else "fp32"
    )
    if top_level_precision not in ("fp32", "int8_qdq"):
        top_level_precision = "fp32"

    variants: list[dict[str, Any]] = []
    base = get_config().get("MODEL_BASE_PATH", "models")
    for mid, payload in sorted(variant_entries.items()):
        if not isinstance(payload, dict):
            continue
        weights_rel = str(payload.get("weights_path", ""))
        labels_rel = str(payload.get("labels_path", ""))
        weights_abs = os.path.join(base, weights_rel) if weights_rel else ""
        labels_abs = os.path.join(base, labels_rel) if labels_rel else ""

        # int8-QDQ availability = primary path OR any fallback path
        # actually exists on disk. Missing entirely on disk means the
        # operator cannot toggle int8 for this variant (UI grays the chip).
        int8_candidates_rel: list[str] = []
        primary_int8 = payload.get("weights_int8_qdq_path")
        if isinstance(primary_int8, str) and primary_int8.strip():
            int8_candidates_rel.append(primary_int8.strip())
        fallbacks_int8 = payload.get("weights_int8_qdq_fallback_paths")
        if isinstance(fallbacks_int8, list):
            for entry in fallbacks_int8:
                if (
                    isinstance(entry, str)
                    and entry.strip()
                    and entry not in int8_candidates_rel
                ):
                    int8_candidates_rel.append(entry.strip())
        int8_available = any(
            os.path.exists(os.path.join(base, rel)) for rel in int8_candidates_rel
        )

        # Per-variant precision (stored at the pinned_models[<id>] level);
        # fall back to top-level for the current default.
        precision_raw = payload.get("active_precision")
        if isinstance(precision_raw, str) and precision_raw in (
            "fp32",
            "int8_qdq",
        ):
            active_precision = precision_raw
        elif mid == hf_latest_id:
            active_precision = top_level_precision
        else:
            active_precision = "fp32"

        info = VariantInfo(
            id=mid,
            weights_path=weights_rel,
            labels_path=labels_rel,
            weights_exists=bool(weights_abs) and os.path.exists(weights_abs),
            labels_exists=bool(labels_abs) and os.path.exists(labels_abs),
            is_active=(mid == (runtime_id or active_on_disk_id)),
            is_hf_latest=(mid == hf_latest_id),
            int8_qdq_available=int8_available,
            active_precision=active_precision,
            metadata=_build_variant_metadata(model_dir, mid, payload),
        )
        variants.append(info.to_dict())

    # UI whitelist filter: show only variants that are either currently
    # active or advertised by HuggingFace at the most recent merge.
    # Legacy / experimental / _BROKEN artefacts that someone's Docker
    # volume still holds are hidden from the picker so end users only
    # ever see choices the publisher actually stands behind. Files on
    # disk are left alone — this is a view-layer filter, not a cleanup.
    # When no HF snapshot is available yet (fresh install, offline
    # first start), the filter gracefully degrades to showing
    # everything so the UI is never empty.
    hf_known_raw = latest.get(HF_KNOWN_IDS_KEY) if isinstance(latest, dict) else None
    hf_known: set[str] = (
        {v for v in hf_known_raw if isinstance(v, str)}
        if isinstance(hf_known_raw, list)
        else set()
    )
    if hf_known:
        runtime_active_id = runtime_id or active_on_disk_id
        variants = [
            v for v in variants
            if v["id"] in hf_known or v["id"] == runtime_active_id
        ]

    # Post-process: data-driven tags + newest-first sort. Tags are assigned
    # across the full variant set (so "Latest", "Highest recall" etc. are
    # global judgements, not per-row), then attached back to each entry.
    tags_by_id = _compute_variant_tags(variants)
    for v in variants:
        v["tags"] = tags_by_id.get(v["id"], [])
    variants = _sort_variants_newest_first(variants)

    return {
        "model_dir": model_dir,
        "active": {
            "id": runtime_id or active_on_disk_id,
            "source": active_source,
            "env_pin_value": pin_value or None,
            "hf_latest_id": hf_latest_id,
            "runtime_matches_on_disk": (runtime_id == active_on_disk_id)
            if runtime_id
            else None,
        },
        "runtime": runtime,
        "metadata": metadata,
        "variants": variants,
    }


def variant_is_known(payload: dict[str, Any], model_id: str) -> bool:
    """Return True if ``model_id`` is a locally-available variant."""
    for v in payload.get("variants", []):
        if v.get("id") == model_id and v.get("is_available_locally"):
            return True
    return False


def variant_exists_in_registry(payload: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """Return the variant entry when ``model_id`` is listed in the registry,
    regardless of local availability. This is the whitelist gate for the
    install endpoint: only ids declared under ``pinned_models`` (or the
    current HF ``latest``) can be fetched — never arbitrary strings from
    the request body.
    """
    for v in payload.get("variants", []):
        if v.get("id") == model_id:
            return v
    return None


# ---------------------------------------------------------------------------
# Classifier registry (parallel to Detector, simpler — no precision chips,
# no int8 QDQ fallback, companion files are classes.txt not labels.json).
# ---------------------------------------------------------------------------


def _read_classifier_latest_models(model_dir: str) -> dict[str, Any]:
    """Dedicated reader so the detector and classifier JSONs stay isolated."""
    path = os.path.join(model_dir, "latest_models.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return {}


def _build_classifier_variant_metadata(
    model_dir: str,
    model_id: str,
) -> dict[str, Any]:
    """Classifier-side metadata extractor.

    Classifier releases ship ``<id>_model_config.yaml`` + ``<id>_metrics.json``
    alongside the ONNX weights (same 4-file convention as the detector
    since the 2026-04-18 HF layout spec). Species count and top-1
    accuracy are the two numbers end users care about — the UI shows
    them as "NNN species · top-1 XX%".
    """
    out: dict[str, Any] = {}

    metrics_path = os.path.join(model_dir, f"{model_id}_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as file:
                metrics = json.load(file)
            if isinstance(metrics, dict):
                for key in ("top1_accuracy", "top5_accuracy", "num_classes"):
                    value = metrics.get(key)
                    if isinstance(value, (int, float)):
                        out[key] = (
                            float(value)
                            if key != "num_classes"
                            else int(value)
                        )
                train = metrics.get("train_info")
                if isinstance(train, dict):
                    trained_at = train.get("trained_at")
                    if isinstance(trained_at, str):
                        out["trained_at"] = trained_at
        except Exception as exc:
            logger.debug(f"classifier metadata: failed to read {metrics_path}: {exc}")

    yaml_path = os.path.join(model_dir, f"{model_id}_model_config.yaml")
    if os.path.exists(yaml_path):
        try:
            import yaml as _yaml

            with open(yaml_path, encoding="utf-8") as file:
                config = _yaml.safe_load(file)
            if isinstance(config, dict):
                # Canonical layout (2026-04-23 HF spec): thresholds + input
                # size live under ``detection.*``. Earlier dev-style YAMLs
                # may nest the same fields under ``classification`` or
                # ``model``; we fall back to those so variants from any
                # era still surface metadata.
                detection_section = config.get("detection")
                cls_section = config.get("classification") or config.get("model")
                sections = [
                    s for s in (detection_section, cls_section) if isinstance(s, dict)
                ]

                for section in sections:
                    input_size = section.get("input_size")
                    if (
                        "input_size" not in out
                        and isinstance(input_size, list)
                        and len(input_size) == 2
                        and all(isinstance(n, int) for n in input_size)
                    ):
                        out["input_size"] = list(input_size)
                    architecture = section.get("architecture")
                    if (
                        "architecture" not in out
                        and isinstance(architecture, str)
                        and architecture.strip()
                    ):
                        out["architecture"] = architecture.strip()

                # Calibrated decision thresholds (CLS-v2 decision layer).
                # species_threshold is the species-accept threshold;
                # genus_threshold is the summed-sibling-mass accept for
                # the genus fallback. Both were added 2026-04-23; older
                # YAMLs that lack them simply do not surface the chip.
                if isinstance(detection_section, dict):
                    species_thr = detection_section.get("confidence_threshold")
                    if isinstance(species_thr, (int, float)):
                        out["confidence_threshold"] = float(species_thr)
                    genus_thr = detection_section.get("genus_fallback_threshold")
                    if isinstance(genus_thr, (int, float)):
                        out["genus_fallback_threshold"] = float(genus_thr)

                meta = config.get("meta")
                if isinstance(meta, dict):
                    trained_at = meta.get("trained_at")
                    if isinstance(trained_at, str) and "trained_at" not in out:
                        out["trained_at"] = trained_at
                    num_classes = meta.get("num_classes")
                    if isinstance(num_classes, int) and "num_classes" not in out:
                        out["num_classes"] = num_classes

                # Final fallback for num_classes: the ``classes`` list at
                # top level is authoritative for the ONNX output layout,
                # so length of that list === num_classes when meta.num_classes
                # is absent.
                if "num_classes" not in out:
                    classes_list = config.get("classes")
                    if isinstance(classes_list, list) and classes_list:
                        out["num_classes"] = len(classes_list)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"classifier metadata: failed to read {yaml_path}: {exc}")

    # Released-date fallback from the id prefix (YYYYMMDD).
    if "released" not in out:
        released = _released_from_id(model_id)
        if released:
            out["released"] = released

    return out


def build_classifier_registry_payload(classifier: Any | None) -> dict[str, Any]:
    """Assemble the GET /api/v1/models/classifier response body.

    Parallels :func:`build_detector_registry_payload` but drops the
    precision chip / int8 QDQ fields — the classifier only ships fp32
    weights. Variant rows still get a metadata block and the same
    HF-whitelist filter, so the UI picker only shows ids the publisher
    currently advertises plus the one that is actually loaded.
    """
    model_dir = _classifier_model_dir()
    latest = _read_classifier_latest_models(model_dir)

    hf_latest_id: str | None = _resolve_hf_latest_id(latest)
    active_source = _detect_active_source(model_dir)

    pinned_models = latest.get("pinned_models") if isinstance(latest, dict) else None
    if not isinstance(pinned_models, dict):
        pinned_models = {}

    # Effective active id when the app next loads: pin (any source) wins
    # over the on-disk top-level ``latest`` pointer. Deliberately reads
    # the local JSON's own ``latest`` (preservation-guarded), NOT HF's
    # advertised one — those diverge when the guard keeps the local
    # active because HF's new files are not on disk yet.
    local_top_latest = latest.get("latest") if isinstance(latest, dict) else None
    pin_value = _resolve_pin_for_cache_dir(model_dir)
    active_on_disk_id = pin_value or local_top_latest

    runtime_id = None
    runtime: dict[str, Any] = {}
    if classifier is not None:
        # The classifier lazy-loads on first predict. Before that, its
        # runtime attributes are at their class defaults (image_size=224,
        # empty classes list, decision_config=None) — which would make
        # the settings page lie about the active model. To keep the
        # Active card honest pre-predict, we resolve the active id from
        # the lazy handle and fall back to the on-disk YAML for the
        # input_size / class count / thresholds.
        classifier_initialized = bool(getattr(classifier, "_initialized", False))
        runtime_id = getattr(classifier, "model_id", None) or None
        if not runtime_id:
            runtime_id = active_on_disk_id
        fallback_meta: dict[str, Any] = (
            _build_classifier_variant_metadata(model_dir, runtime_id)
            if runtime_id
            else {}
        )

        image_size = getattr(classifier, "CLASSIFIER_IMAGE_SIZE", None)
        input_size: list[int] = []
        if classifier_initialized and image_size:
            input_size = [int(image_size), int(image_size)]
        elif (
            isinstance(fallback_meta.get("input_size"), list)
            and len(fallback_meta["input_size"]) == 2
        ):
            input_size = [int(n) for n in fallback_meta["input_size"]]

        classes_count = len(getattr(classifier, "classes", None) or [])
        if not classes_count and isinstance(fallback_meta.get("num_classes"), int):
            classes_count = int(fallback_meta["num_classes"])

        runtime = {
            "model_id": runtime_id,
            "model_path": getattr(classifier, "model_path", None),
            "input_size": input_size,
            "num_classes": classes_count,
        }

        # Decision config thresholds. ``decision_config`` is set during
        # lazy-init, so pre-predict we fall back to the YAML values we
        # already extracted into ``fallback_meta``. Legacy classifiers
        # without a YAML simply have no threshold keys — the UI then
        # skips the chip.
        decision_config = getattr(classifier, "decision_config", None)
        species_thr = (
            getattr(decision_config, "species_threshold", None)
            if decision_config is not None
            else fallback_meta.get("confidence_threshold")
        )
        genus_thr = (
            getattr(decision_config, "genus_threshold", None)
            if decision_config is not None
            else fallback_meta.get("genus_fallback_threshold")
        )
        if isinstance(species_thr, (int, float)):
            runtime["confidence_threshold"] = float(species_thr)
        if isinstance(genus_thr, (int, float)):
            runtime["genus_fallback_threshold"] = float(genus_thr)

    variant_entries = dict(pinned_models)
    if hf_latest_id and hf_latest_id not in variant_entries:
        variant_entries[hf_latest_id] = {
            "weights_path": latest.get("weights_path", ""),
            "classes_path": latest.get("classes_path", ""),
        }

    variants: list[dict[str, Any]] = []
    base = get_config().get("MODEL_BASE_PATH", "models")
    for mid, payload in sorted(variant_entries.items()):
        if not isinstance(payload, dict):
            continue
        weights_rel = str(payload.get("weights_path", ""))
        classes_rel = str(
            payload.get("classes_path", "") or payload.get("labels_path", "")
        )
        weights_abs = os.path.join(base, weights_rel) if weights_rel else ""
        classes_abs = os.path.join(base, classes_rel) if classes_rel else ""

        entry = {
            "id": mid,
            "weights_path": weights_rel,
            "classes_path": classes_rel,
            "weights_exists": bool(weights_abs) and os.path.exists(weights_abs),
            "classes_exists": bool(classes_abs) and os.path.exists(classes_abs),
            "is_active": (mid == (runtime_id or active_on_disk_id)),
            "is_hf_latest": (mid == hf_latest_id),
            "metadata": _build_classifier_variant_metadata(model_dir, mid),
        }
        entry["is_available_locally"] = entry["weights_exists"] and entry["classes_exists"]
        variants.append(entry)

    # Same HF-whitelist filter as the detector payload.
    hf_known_raw = latest.get(HF_KNOWN_IDS_KEY) if isinstance(latest, dict) else None
    hf_known: set[str] = (
        {v for v in hf_known_raw if isinstance(v, str)}
        if isinstance(hf_known_raw, list)
        else set()
    )
    if hf_known:
        runtime_active_id = runtime_id or active_on_disk_id
        variants = [
            v for v in variants
            if v["id"] in hf_known or v["id"] == runtime_active_id
        ]

    # Latest first (same sort as detector).
    variants = _sort_variants_newest_first(variants)

    return {
        "model_dir": model_dir,
        "active": {
            "id": runtime_id or active_on_disk_id,
            "source": active_source,
            "env_pin_value": pin_value or None,
            "hf_latest_id": hf_latest_id,
            "runtime_matches_on_disk": (runtime_id == active_on_disk_id)
            if runtime_id
            else None,
        },
        "runtime": runtime,
        "variants": variants,
    }


def classifier_variant_exists_in_registry(
    payload: dict[str, Any], model_id: str
) -> dict[str, Any] | None:
    """Whitelist gate for the classifier /install endpoint."""
    for v in payload.get("variants", []):
        if v.get("id") == model_id:
            return v
    return None
