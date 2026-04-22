"""Helper functions for downloading and caching models.

Model resolution order (highest priority first):

1. **Pinned local model** (``WMB_PINNED_MODEL_ID`` env var)
   When set, skips the HF fetch entirely and uses the local
   ``latest_models.json`` as the source of truth. Useful for:
     - running a dev / experimental model that is not on HF yet
     - pinning a known-good historical version for rollback
     - pinning a specific version for reproducible benchmarking
   The env-var value must match the ``latest`` field of the local JSON
   (or match one of the available pinned entries — see
   :data:`LOCAL_PINNED_MODELS_KEY`).

2. **Remote HF latest** (default behaviour)
   Fetches ``latest_models.json`` from the HF base URL. Falls back to
   the local cached copy when the network request fails.

3. **Local cache only**
   Used implicitly when both HF and env-var are unset, or as a fallback
   when the remote fetch fails.

Preservation guard (always active unless ``WMB_FORCE_REMOTE_REFRESH=1``):
when the remote HF response points at files we do not have locally and
the current local pointer is still valid, keep the local active pointer
but merge the remote registry entries into ``latest_models.json``. This
keeps a working install stable while still letting the Settings UI show
new variants as installable.
"""

from __future__ import annotations

import json
import os
import time

import requests

from logging_config import get_logger

logger = get_logger(__name__)

# Env var prefix: pin a specific local model identifier per task.
# The task name is derived from the cache_dir basename ("object_detection"
# or "classifier") and uppercased, so the resolved env vars are
#   WMB_PINNED_MODEL_ID_OBJECT_DETECTION  (detector)
#   WMB_PINNED_MODEL_ID_CLASSIFIER        (classifier)
# A generic WMB_PINNED_MODEL_ID is still read as a last-resort fallback
# for backwards compatibility with single-task deployments.
PIN_ENV_VAR_PREFIX = "WMB_PINNED_MODEL_ID"
PIN_ENV_VAR = PIN_ENV_VAR_PREFIX  # legacy generic fallback

# Env var: when "1"/"true", bypass the preservation guard and let the
# remote HF response overwrite the local latest_models.json even when
# its referenced files are missing. Use only when you really want the
# next startup to download fresh artifacts.
FORCE_REFRESH_ENV_VAR = "WMB_FORCE_REMOTE_REFRESH"

# Optional dict key inside latest_models.json that maps pinnable
# identifiers to their full { latest, weights_path, labels_path }
# payloads. Absent by default; consumers can hand-edit the JSON to add
# a local lineup, e.g.:
#   {
#     "latest": "20260417_crazy_detector_locator",
#     "weights_path": "...", "labels_path": "...",
#     "pinned_models": {
#       "20250810_215216": {
#         "weights_path": "object_detection/20250810_215216_best.onnx",
#         "labels_path":  "object_detection/20250810_215216_labels.json"
#       }
#     }
#   }
LOCAL_PINNED_MODELS_KEY = "pinned_models"

# Per-variant precision pin. Lives inside each ``pinned_models[<id>]`` entry
# as ``active_precision`` with value ``"fp32"`` (default, loads weights_path)
# or ``"int8_qdq"`` (loads the first variant in
# ``weights_int8_qdq_fallback_paths`` that exists on disk and parses).
# Absent key == ``"fp32"`` for backwards compatibility with older registries.
PRECISION_KEY = "active_precision"
PRECISION_FP32 = "fp32"
PRECISION_INT8_QDQ = "int8_qdq"
PRECISION_VALUES = (PRECISION_FP32, PRECISION_INT8_QDQ)

# Registry fields read when resolving the weights path for a precision mode.
WEIGHTS_INT8_QDQ_KEY = "weights_int8_qdq_path"
WEIGHTS_INT8_QDQ_FALLBACKS_KEY = "weights_int8_qdq_fallback_paths"

# Snapshot of the ids the remote HF registry advertised at the most recent
# successful merge. Written into latest_models.json under this key so the
# Settings UI can filter out local-only variants (Docker volumes with old
# experimental weights, _BROKEN dev artefacts, etc.) without needing to
# hit HF at render time. Updated on every successful remote fetch; the
# active variant is always shown regardless, so a user's pinned choice
# never disappears even when HF no longer advertises it.
HF_KNOWN_IDS_KEY = "hf_known_ids"

ACTIVE_PAYLOAD_KEYS = (
    "latest",
    "weights_path",
    "weights_path_onnx",
    "labels_path",
    "onnx_path",
    "model",
    "path",
    "classes_path",
    "labels",
    WEIGHTS_INT8_QDQ_KEY,
    WEIGHTS_INT8_QDQ_FALLBACKS_KEY,
    PRECISION_KEY,
)


def _first_present(d: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _guess_labels_from_weights(weights_name: str) -> str | None:
    name = os.path.basename(weights_name)
    if "_best." in name:
        # map TIMESTAMP_best.onnx or TIMESTAMP_best.pt -> TIMESTAMP_labels.json
        return name.replace("_best.onnx", "_labels.json").replace(
            "_best.pt", "_labels.json"
        )
    return None


def _normalize_rel_path(base_url: str, rel: str) -> str:
    """Normalize a registry-relative path against a base_url that already ends with
    a task subfolder (e.g., .../resolve/main/classifier). This avoids duplicated
    segments like 'classifier/classifier/...'. Also strips a leading
    'model_registry/' if present.
    """
    if not rel:
        return rel
    rel = rel.lstrip("/")
    if rel.startswith("model_registry/"):
        rel = rel[len("model_registry/") :]
    # derive the last path segment of the base URL (expected: 'classifier' or 'object_detection')
    subdir = base_url.rstrip("/").split("/")[-1]
    if rel.startswith(f"{subdir}/"):
        rel = rel[len(subdir) + 1 :]
    return rel


def _download_file(url: str, dest: str, retries: int = 3, timeout: int = 60) -> bool:
    """Downloads a file from a URL."""
    if os.path.exists(dest):
        logger.debug(f"File already exists and will be skipped: {dest}")
        return True
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info(f"File downloaded: {dest}")
            return True
        except requests.RequestException as exc:
            logger.warning(
                f"Download attempt {attempt}/{retries} for {url} failed: {exc}"
            )
            if attempt < retries:
                time.sleep(1)
    logger.error(f"Download failed permanently for {url}")
    return False


def _task_name_from_cache_dir(cache_dir: str) -> str:
    """Derive the env-var suffix from the cache dir path.

    Example: ``/opt/app/data/models/object_detection`` -> ``OBJECT_DETECTION``.
    Used to namespace the pin env var so the detector and the classifier
    can be pinned independently.
    """
    base = os.path.basename(os.path.normpath(cache_dir))
    return base.upper().replace("-", "_")


def set_latest_model_id(cache_dir: str, model_id: str) -> str:
    """Switch the active default by rewriting ``latest_models.json``.

    This is the analogue of ``update_runtime_settings({CAMERA_URL: ...})``
    for the detector: the UI changes the pointer on disk, the
    DetectionService reloads, done. No extra pin file, no extra
    resolution layer.

    Requirements:
      * ``latest_models.json`` already exists in *cache_dir* (the app's
        startup path creates it when HF is reachable or when a release
        bundle is deployed).
      * ``model_id`` must match either ``latest_models.json["latest"]``
        or a key under the ``pinned_models`` block, so we can look up
        the matching ``weights_path`` / ``labels_path``.

    The preservation guard (``fetch_latest_json``) then ensures that a
    subsequent HF ``latest`` pointing at files not on disk will **not**
    overwrite this change — so a UI-selected variant survives restarts
    until the operator explicitly forces a refresh.

    Returns the absolute path to the rewritten ``latest_models.json``.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model_id must be non-empty")

    local = _read_local_latest(cache_dir)
    if local is None:
        raise FileNotFoundError(
            f"No latest_models.json in {cache_dir}; cannot switch active model."
        )

    merged = _apply_pin(cache_dir, local, model_id)
    if not _local_payload_is_usable(cache_dir, "", merged):
        raise FileNotFoundError(
            f"Model {model_id!r} is listed in latest_models.json but its "
            f"weights or labels files are missing on disk."
        )

    latest_path = os.path.join(cache_dir, "latest_models.json")
    tmp_path = f"{latest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(merged, file)
    os.replace(tmp_path, latest_path)
    logger.info(f"latest_models.json updated: latest={model_id!r}")
    return latest_path


def _resolve_pin_for_cache_dir(cache_dir: str) -> str:
    """Return the effective env-var pin for *cache_dir*, or empty string.

    Priority order (highest to lowest):
      1. Task-specific env var  WMB_PINNED_MODEL_ID_<TASK>  (from systemd etc.)
      2. Generic env var         WMB_PINNED_MODEL_ID        (legacy fallback)

    There is intentionally no third layer: UI-level switching happens by
    rewriting ``latest_models.json`` (see :func:`set_latest_model_id`),
    not via a separate pin file. That keeps the detector-switch story
    parallel to the camera-switch story ("UI edits the runtime config,
    reload is triggered") instead of introducing an extra resolution
    layer the operator has to reason about.
    """
    task = _task_name_from_cache_dir(cache_dir)
    task_specific = os.environ.get(f"{PIN_ENV_VAR_PREFIX}_{task}", "").strip()
    if task_specific:
        return task_specific
    return os.environ.get(PIN_ENV_VAR, "").strip()


def _read_local_latest(cache_dir: str) -> dict | None:
    local_path = os.path.join(cache_dir, "latest_models.json")
    if not os.path.exists(local_path):
        return None
    try:
        with open(local_path, encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"Failed to read local {local_path}: {exc}")
    return None


def _files_from_payload(cache_dir: str, base_url: str, data: dict) -> tuple[str | None, str | None]:
    """Extract and local-resolve (weights_path, labels_path) from a JSON payload."""
    weights_rel = _first_present(
        data, ("weights_path_onnx", "weights_path", "onnx_path", "model", "path")
    )
    labels_rel = _first_present(
        data, ("labels_path", "labels", "classes_path")
    )
    if not weights_rel:
        return None, None
    weights_rel_norm = _normalize_rel_path(base_url, weights_rel)
    if not labels_rel:
        guessed = _guess_labels_from_weights(weights_rel_norm)
        if guessed:
            labels_rel = guessed
    if not labels_rel:
        return None, None
    labels_rel_norm = _normalize_rel_path(base_url, labels_rel)
    return (
        os.path.join(cache_dir, os.path.basename(weights_rel_norm)),
        os.path.join(cache_dir, os.path.basename(labels_rel_norm)),
    )


def _local_payload_is_usable(cache_dir: str, base_url: str, data: dict) -> bool:
    """Return True when the local JSON payload references files that all exist."""
    weights, labels = _files_from_payload(cache_dir, base_url, data)
    return bool(weights and labels and os.path.exists(weights) and os.path.exists(labels))


def _apply_pin(cache_dir: str, data: dict, pin: str) -> dict:
    """If ``pin`` matches a key under ``pinned_models``, return the pinned payload
    patched into a full JSON-shaped dict. Otherwise return ``data`` unchanged if
    the pin matches ``data["latest"]``; raise ValueError when the pin is unknown.
    """
    if data.get("latest") == pin:
        return data
    pinned_map = data.get(LOCAL_PINNED_MODELS_KEY)
    if isinstance(pinned_map, dict) and pin in pinned_map:
        entry = pinned_map[pin]
        if isinstance(entry, dict):
            merged = dict(data)
            merged["latest"] = pin
            # Let the pinned entry override weights/labels paths and
            # precision-related fields. Precision keys are preserved so a
            # variant switch doesn't silently drop the operator's int8
            # selection for the newly-active model.
            for key in (
                "weights_path",
                "weights_path_onnx",
                "labels_path",
                "onnx_path",
                "model",
                "path",
                "classes_path",
                "labels",
                WEIGHTS_INT8_QDQ_KEY,
                WEIGHTS_INT8_QDQ_FALLBACKS_KEY,
                PRECISION_KEY,
            ):
                if key in entry:
                    merged[key] = entry[key]
            return merged
    raise ValueError(
        f"{PIN_ENV_VAR}={pin!r} does not match local latest_models.json. "
        f"Available: latest={data.get('latest')!r}, pinned_models="
        f"{sorted(data.get(LOCAL_PINNED_MODELS_KEY, {}))}"
    )


def _prune_stale_local_variants(
    cache_dir: str,
    merged: dict,
    remote_pinned: dict | None,
    remote_latest_id: str | None,
) -> list[str]:
    """Drop ``pinned_models`` entries that the publisher removed from HF and
    that we don't have weights for locally.

    This is the cleanup companion to :func:`_merge_remote_registry_with_local_state`.
    Without it, experimental/broken variants that were once published to HF
    stay visible in the Settings AI panel forever — end users have no way to
    remove them (they can't edit the JSON on Docker/RPi).

    Safety rules:
      * Never drop the current local ``latest`` — operators rely on it
        running even after HF removes the entry.
      * Never drop an entry whose weights are on disk — a user may have
        installed it locally via the UI and wants to keep it.
      * Only drop entries that are BOTH absent from the remote registry
        AND have no weights file on disk. These are unreachable rubble:
        UI shows them as "Not installed" but "Install & switch" would 404.

    Returns the list of ids that were removed (for logging).
    """
    pinned = merged.get(LOCAL_PINNED_MODELS_KEY)
    if not isinstance(pinned, dict) or not pinned:
        return []

    protected: set[str] = set()
    local_active = merged.get("latest")
    if isinstance(local_active, str) and local_active:
        protected.add(local_active)
    if isinstance(remote_latest_id, str) and remote_latest_id:
        protected.add(remote_latest_id)

    remote_ids: set[str] = set()
    if isinstance(remote_pinned, dict):
        remote_ids.update(k for k in remote_pinned.keys() if isinstance(k, str))

    removed: list[str] = []
    for mid in list(pinned.keys()):
        if mid in protected or mid in remote_ids:
            continue
        entry = pinned[mid]
        if not isinstance(entry, dict):
            continue
        weights_rel = _first_present(
            entry, ("weights_path", "weights_path_onnx", "onnx_path", "model", "path")
        )
        if weights_rel:
            weights_abs = os.path.join(cache_dir, os.path.basename(weights_rel))
            if os.path.exists(weights_abs):
                continue  # weights on disk → user can still load it
        removed.append(mid)
        del pinned[mid]

    if removed and not pinned:
        # Don't leave an empty dict lying around.
        del merged[LOCAL_PINNED_MODELS_KEY]

    return removed


def _merge_remote_registry_with_local_state(
    remote_data: dict,
    local_data: dict,
    *,
    preserve_local_active: bool,
) -> dict:
    """Merge HF registry entries while preserving local runtime choices.

    ``preserve_local_active`` keeps the top-level model pointer and paths from
    the local JSON. That is the user-friendly upgrade path for Docker/RPi
    installs: the running model stays valid, while new remote variants become
    visible in Settings as "Not installed" and can be fetched on demand.
    """
    merged = dict(remote_data)

    if preserve_local_active:
        for key in ACTIVE_PAYLOAD_KEYS:
            merged.pop(key, None)
        for key in ACTIVE_PAYLOAD_KEYS:
            if key in local_data:
                merged[key] = local_data[key]
    elif PRECISION_KEY in local_data:
        merged[PRECISION_KEY] = local_data[PRECISION_KEY]

    remote_pinned = remote_data.get(LOCAL_PINNED_MODELS_KEY)
    local_pinned = local_data.get(LOCAL_PINNED_MODELS_KEY)
    merged_pinned: dict = {}

    if isinstance(remote_pinned, dict):
        merged_pinned.update(
            {
                mid: (dict(payload) if isinstance(payload, dict) else payload)
                for mid, payload in remote_pinned.items()
            }
        )

    # Ensure the remote ``latest`` always appears as a variant the UI can
    # show — even when the publisher did not list it under ``pinned_models``.
    # Without this, a release that only bumps the top-level pointer (e.g. HF
    # lists ``latest: _v4`` and pins only older IDs) is invisible to end
    # users until someone sets ``WMB_FORCE_REMOTE_REFRESH=1``, which typical
    # Docker-Compose / RPi operators cannot or should not do.
    remote_latest_id = remote_data.get("latest")
    if isinstance(remote_latest_id, str) and remote_latest_id.strip():
        if remote_latest_id not in merged_pinned:
            synth: dict[str, str] = {}
            for key in (
                "weights_path",
                "weights_path_onnx",
                "labels_path",
                "onnx_path",
                "model",
                "path",
                "classes_path",
                "labels",
                WEIGHTS_INT8_QDQ_KEY,
                WEIGHTS_INT8_QDQ_FALLBACKS_KEY,
            ):
                value = remote_data.get(key)
                if value is not None:
                    synth[key] = value
            if synth:
                merged_pinned[remote_latest_id] = synth

    if isinstance(local_pinned, dict):
        for mid, local_entry in local_pinned.items():
            if not isinstance(local_entry, dict):
                if mid not in merged_pinned:
                    merged_pinned[mid] = local_entry
                continue

            target = merged_pinned.get(mid)
            if not isinstance(target, dict):
                merged_pinned[mid] = dict(local_entry)
                continue

            if PRECISION_KEY in local_entry:
                target[PRECISION_KEY] = local_entry[PRECISION_KEY]

    if merged_pinned:
        merged[LOCAL_PINNED_MODELS_KEY] = merged_pinned

    # Record which ids the publisher currently advertises. The Settings UI
    # uses this as a whitelist: variants absent from HF (experimental dev
    # artefacts, legacy volumes, manual tinkering) stay on disk but are
    # hidden from the picker. Survives across restarts — each merge
    # overwrites the list with the latest HF view.
    hf_known: set[str] = set()
    if isinstance(remote_latest_id, str) and remote_latest_id.strip():
        hf_known.add(remote_latest_id)
    if isinstance(remote_pinned, dict):
        for k in remote_pinned.keys():
            if isinstance(k, str):
                hf_known.add(k)
    if hf_known:
        merged[HF_KNOWN_IDS_KEY] = sorted(hf_known)
    elif HF_KNOWN_IDS_KEY in merged:
        # Empty set means HF returned nothing useful this round — don't
        # overwrite the previous snapshot with an empty list.
        pass

    return merged


def set_active_precision(cache_dir: str, model_id: str, precision: str) -> str:
    """Write ``active_precision`` for ``model_id`` into ``latest_models.json``.

    Analogue to :func:`set_latest_model_id`: the UI changes the pointer on
    disk, the DetectionService reloads, done. No new pin file, no extra
    resolution layer.

    - ``precision`` must be one of :data:`PRECISION_VALUES`.
    - ``model_id`` must be a key under ``pinned_models`` (or match
      ``latest``) so we have a stable home for the precision flag. For the
      rare case where the active default has no ``pinned_models`` entry,
      we also stamp the top-level ``active_precision`` so the loader can
      still read it.

    Returns the absolute path to the rewritten ``latest_models.json``.
    """
    precision = (precision or "").strip()
    if precision not in PRECISION_VALUES:
        raise ValueError(
            f"precision must be one of {PRECISION_VALUES}, got {precision!r}"
        )
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model_id must be non-empty")

    local = _read_local_latest(cache_dir)
    if local is None:
        raise FileNotFoundError(
            f"No latest_models.json in {cache_dir}; cannot switch active precision."
        )

    # Stamp the precision key into the per-variant entry when possible so
    # switching variants later preserves each variant's last-used precision.
    changed = False
    pinned_map = local.get(LOCAL_PINNED_MODELS_KEY)
    if isinstance(pinned_map, dict) and model_id in pinned_map:
        entry = pinned_map[model_id]
        if isinstance(entry, dict):
            entry[PRECISION_KEY] = precision
            changed = True

    # Also stamp top-level when model_id matches the current default, so
    # the runtime loader (which reads the top-level merged dict via
    # ``fetch_latest_json`` / ``_apply_pin``) sees the precision choice
    # on the very next reload.
    if local.get("latest") == model_id:
        local[PRECISION_KEY] = precision
        changed = True

    if not changed:
        raise ValueError(
            f"model_id {model_id!r} is not the current default and has no "
            f"entry under pinned_models; cannot record precision."
        )

    latest_path = os.path.join(cache_dir, "latest_models.json")
    tmp_path = f"{latest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(local, file)
    os.replace(tmp_path, latest_path)
    logger.info(
        f"latest_models.json updated: active_precision[{model_id!r}]={precision!r}"
    )
    return latest_path


def resolve_active_precision_artefacts(cache_dir: str) -> dict | None:
    """Return precision-aware load plan for the detector.

    Reads ``latest_models.json`` in *cache_dir* and, if the active
    ``active_precision`` is ``"int8_qdq"``, assembles the ordered list of
    candidate weights to try at load time:

      1. ``weights_int8_qdq_path``      (primary QDQ variant)
      2. ``weights_int8_qdq_fallback_paths``  (remaining QDQ variants)

    Returns a dict with keys:

      - ``requested_precision``: ``"fp32"`` | ``"int8_qdq"``
      - ``load_candidates``: list[str] absolute paths to try in order
        (empty for fp32 — the detector uses its normal weights_path in
        that case)
      - ``fp32_fallback_path``: absolute path to fp32 weights, for loud
        fallback when every int8 candidate fails.

    Returns ``None`` when ``latest_models.json`` is missing or unreadable
    (caller treats this as "fp32 with no precision opt-in").
    """
    data = _read_local_latest(cache_dir)
    if data is None:
        return None

    # The merged-top-level block is authoritative at load time — the env-
    # var pin path does go through ``_apply_pin`` which now propagates
    # the precision key, and the plain ``latest_models["latest"]`` path
    # already has the precision key stamped on the top level (see
    # ``set_active_precision``).
    requested = str(data.get(PRECISION_KEY, PRECISION_FP32))
    if requested not in PRECISION_VALUES:
        requested = PRECISION_FP32

    fp32_rel = _first_present(
        data, ("weights_path_onnx", "weights_path", "onnx_path", "model", "path")
    )
    fp32_abs = (
        os.path.join(cache_dir, os.path.basename(fp32_rel))
        if fp32_rel
        else ""
    )

    if requested == PRECISION_FP32:
        return {
            "requested_precision": PRECISION_FP32,
            "load_candidates": [],
            "fp32_fallback_path": fp32_abs,
        }

    # int8_qdq path — gather candidates in order.
    candidates: list[str] = []
    primary_rel = data.get(WEIGHTS_INT8_QDQ_KEY)
    if isinstance(primary_rel, str) and primary_rel.strip():
        candidates.append(primary_rel.strip())
    fallbacks = data.get(WEIGHTS_INT8_QDQ_FALLBACKS_KEY)
    if isinstance(fallbacks, list):
        for entry in fallbacks:
            if isinstance(entry, str) and entry.strip() and entry not in candidates:
                candidates.append(entry.strip())

    absolute: list[str] = []
    seen: set[str] = set()
    for rel in candidates:
        abs_path = os.path.join(cache_dir, os.path.basename(rel))
        if abs_path in seen:
            continue
        seen.add(abs_path)
        absolute.append(abs_path)

    return {
        "requested_precision": PRECISION_INT8_QDQ,
        "load_candidates": absolute,
        "fp32_fallback_path": fp32_abs,
    }


def prune_legacy_fasterrcnn_models(model_dir: str) -> list[str]:
    """Remove legacy FasterRCNN artefacts from *model_dir*, if any.

    The app dropped FasterRCNN support in the YOLOX-only release. An existing
    deployment upgraded in-place would otherwise still have the 29-species
    `.onnx` + dict-form `labels.json` on disk, and `_detect_output_format`
    would fail loudly at detector init. This startup cleanup wipes the
    legacy artefacts so the normal HF autofetch path (`ensure_model_files`)
    then downloads the current YOLOX latest.

    Detection heuristic (both must hold for a file set to be classified
    as legacy):
      1. ``labels.json`` parses as a dict (not a list).
      2. The dict has ≥ 20 numeric-keyed entries — matches the 29-species
         labels.json exactly, rejects YOLOX (5-class list) and other short
         dict-form labels files.

    Non-matching files are left alone. The function is idempotent: running
    it on a clean YOLOX deployment is a no-op.

    Returns the list of absolute paths that were removed (empty when
    nothing matched). Any error reading/parsing a candidate file is logged
    and the file is left in place — better to keep the deployment running
    with a loud init error than to destroy state optimistically.
    """
    removed: list[str] = []
    if not os.path.isdir(model_dir):
        return removed

    latest_json_path = os.path.join(model_dir, "latest_models.json")
    local_data = _read_local_latest(model_dir)
    if local_data is None:
        return removed

    candidates: set[str] = set()
    latest_id = local_data.get("latest")
    if isinstance(latest_id, str) and latest_id:
        candidates.add(latest_id)
    pinned = local_data.get(LOCAL_PINNED_MODELS_KEY)
    if isinstance(pinned, dict):
        candidates.update(k for k in pinned.keys() if isinstance(k, str))

    legacy_ids: list[str] = []
    for mid in candidates:
        labels_path = os.path.join(model_dir, f"{mid}_labels.json")
        if not os.path.exists(labels_path):
            continue
        try:
            with open(labels_path, encoding="utf-8") as file:
                raw = json.load(file)
        except Exception as exc:
            logger.warning(
                f"prune_legacy: cannot parse {labels_path}: {exc}; skipping"
            )
            continue
        if not isinstance(raw, dict):
            continue
        # Dict form with many numeric keys == legacy FasterRCNN labels.json.
        numeric_keys = sum(1 for k in raw.keys() if str(k).isdigit())
        if numeric_keys >= 20:
            legacy_ids.append(mid)

    if not legacy_ids:
        return removed

    logger.warning(
        f"prune_legacy: removing legacy FasterRCNN artefacts for "
        f"{legacy_ids} from {model_dir}; HF autofetch will pull current "
        f"YOLOX latest"
    )

    for mid in legacy_ids:
        for suffix in (
            "_best.onnx",
            "_best_int8.onnx",
            "_labels.json",
            "_model_config.yaml",
            "_metrics.json",
            "_README.md",
        ):
            path = os.path.join(model_dir, f"{mid}{suffix}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed.append(path)
                    logger.info(f"prune_legacy: removed {path}")
                except OSError as exc:
                    logger.warning(f"prune_legacy: cannot remove {path}: {exc}")

    # Drop latest_models.json too — once legacy artefacts are gone the next
    # fetch must get a fresh copy from HF (otherwise the preservation guard
    # would see a local file with unusable entries).
    if os.path.exists(latest_json_path):
        try:
            os.remove(latest_json_path)
            removed.append(latest_json_path)
            logger.info(f"prune_legacy: removed {latest_json_path}")
        except OSError as exc:
            logger.warning(
                f"prune_legacy: cannot remove {latest_json_path}: {exc}"
            )

    # Stale model_metadata.json is regenerated by the pin endpoint on first
    # switch; safer to remove it now than to keep FasterRCNN-era thresholds.
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        try:
            os.remove(metadata_path)
            removed.append(metadata_path)
            logger.info(f"prune_legacy: removed {metadata_path}")
        except OSError as exc:
            logger.warning(f"prune_legacy: cannot remove {metadata_path}: {exc}")

    return removed


def fetch_latest_json(base_url: str, cache_dir: str) -> dict[str, str]:
    """Returns the authoritative ``latest_models.json`` payload for *base_url*.

    Resolution precedence:

    1. **Pinning** (``WMB_PINNED_MODEL_ID`` env var): skip the remote
       fetch entirely and use the local cache. Raises when the local
       JSON does not match the pin.
    2. **Remote + preservation guard**: fetch HF's copy; only overwrite
       the local cache when the remote points at files already present
       locally (or ``WMB_FORCE_REMOTE_REFRESH=1`` overrides this).
    3. **Local cache fallback** on network failure.
    """
    local_data = _read_local_latest(cache_dir)
    pin = _resolve_pin_for_cache_dir(cache_dir)

    # 1. Env-var pin — skip HF entirely. UI-level switches do NOT go through
    # here; they rewrite latest_models.json directly (see set_latest_model_id).
    if pin:
        if local_data is None:
            raise FileNotFoundError(
                f"Pin env var is set to {pin!r} but there is no local "
                f"latest_models.json in {cache_dir}. Create one or unset the env var."
            )
        pinned = _apply_pin(cache_dir, local_data, pin)
        task = _task_name_from_cache_dir(cache_dir)
        if os.environ.get(f"{PIN_ENV_VAR_PREFIX}_{task}", "").strip() == pin:
            source = f"env_var:{PIN_ENV_VAR_PREFIX}_{task}"
        else:
            source = f"env_var:{PIN_ENV_VAR}"
        logger.info(
            f"Pinned model {pin!r} (source: {source}) — skipping HF fetch"
        )
        return pinned

    # 2. Try remote.
    latest_url = f"{base_url}/latest_models.json"
    local_path = os.path.join(cache_dir, "latest_models.json")
    try:
        response = requests.get(latest_url, timeout=10)
        response.raise_for_status()
        remote_data = response.json()
    except requests.RequestException as exc:
        logger.warning(f"Error fetching {latest_url}: {exc}")
        if local_data is not None:
            logger.info(f"Using local cache {local_path}")
            return local_data
        raise

    force_refresh = os.environ.get(FORCE_REFRESH_ENV_VAR, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    # Preservation guard. Two cases keep the local active pointer so a
    # UI-driven switch survives the next detector reload:
    #
    # 1. Remote points at files missing on disk AND local is usable.
    #    Protects against a startup where HF advertises a release that
    #    has not been mirrored to this node.
    # 2. Remote and local point at different ids while BOTH are usable.
    #    This is the Switch-via-UI case: set_latest_model_id just wrote
    #    the S-variant pointer, the S weights are on disk (Install just
    #    fetched them), but HF still advertises Tiny. Without this
    #    branch the reload would overwrite the user's choice with HF's
    #    default. Local always wins on conflict; set WMB_FORCE_REMOTE_REFRESH=1
    #    to explicitly resync to HF.
    #
    # In both branches we still merge the remote pinned_models registry into
    # the on-disk JSON. That makes newly shipped variants visible in Settings
    # as "Not installed" instead of hiding them behind an env-var refresh.
    if not force_refresh and local_data is not None:
        remote_usable = _local_payload_is_usable(cache_dir, base_url, remote_data)
        local_usable = _local_payload_is_usable(cache_dir, base_url, local_data)
        remote_latest = remote_data.get("latest")
        local_latest = local_data.get("latest")

        if not remote_usable and local_usable:
            merged_remote = _merge_remote_registry_with_local_state(
                remote_data,
                local_data,
                preserve_local_active=True,
            )
            pruned = _prune_stale_local_variants(
                cache_dir,
                merged_remote,
                remote_data.get(LOCAL_PINNED_MODELS_KEY),
                remote_data.get("latest"),
            )
            if pruned:
                logger.info(
                    "Pruned %d stale local-only variants missing from HF and disk: %s",
                    len(pruned),
                    pruned,
                )
            os.makedirs(cache_dir, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as file:
                json.dump(merged_remote, file)
            if remote_latest != local_latest:
                logger.warning(
                    f"Preserving local {local_path}: remote latest={remote_latest!r} "
                    f"points at files missing on disk; keeping local latest={local_latest!r}. "
                    "Remote variants were merged for Settings install. "
                    f"Set {FORCE_REFRESH_ENV_VAR}=1 to make remote latest active immediately."
                )
            else:
                logger.debug(
                    f"Remote and local both point at {local_latest!r}; keeping local "
                    "active paths and merging remote variants for Settings install."
                )
            return merged_remote

        if (
            remote_usable
            and local_usable
            and remote_latest != local_latest
        ):
            merged_remote = _merge_remote_registry_with_local_state(
                remote_data,
                local_data,
                preserve_local_active=True,
            )
            pruned = _prune_stale_local_variants(
                cache_dir,
                merged_remote,
                remote_data.get(LOCAL_PINNED_MODELS_KEY),
                remote_data.get("latest"),
            )
            if pruned:
                logger.info(
                    "Pruned %d stale local-only variants missing from HF and disk: %s",
                    len(pruned),
                    pruned,
                )
            os.makedirs(cache_dir, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as file:
                json.dump(merged_remote, file)
            logger.info(
                f"Preserving local {local_path}: local latest={local_latest!r} "
                f"differs from remote latest={remote_latest!r} but both are on disk; "
                "UI/operator choice wins and remote variants were merged for Settings. "
                f"Set {FORCE_REFRESH_ENV_VAR}=1 to resync to HF."
            )
            return merged_remote

    # Merge local UI-choice fields into the remote payload before
    # persisting. Without this pass, re-pulling the HF copy would silently
    # drop the operator's ``active_precision`` stamp (and any per-variant
    # stamps under ``pinned_models``) because the HF copy does not carry
    # them. The precision choice is a local runtime decision, not a
    # pipeline artefact — so HF's view of the registry gets augmented
    # with the local runtime state, not replaced by it.
    if not force_refresh and local_data is not None:
        # Snapshot the original remote registry BEFORE merging in local state,
        # so the prune step can tell "publisher removed this" apart from
        # "publisher + local both keep this".
        original_remote_pinned = remote_data.get(LOCAL_PINNED_MODELS_KEY)
        original_remote_latest = remote_data.get("latest")
        remote_data = _merge_remote_registry_with_local_state(
            remote_data,
            local_data,
            preserve_local_active=False,
        )
        pruned = _prune_stale_local_variants(
            cache_dir,
            remote_data,
            original_remote_pinned if isinstance(original_remote_pinned, dict) else None,
            original_remote_latest if isinstance(original_remote_latest, str) else None,
        )
        if pruned:
            logger.info(
                "Pruned %d stale local-only variants missing from HF and disk: %s",
                len(pruned),
                pruned,
            )

    # Safe to persist remote as new source of truth.
    os.makedirs(cache_dir, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as file:
        json.dump(remote_data, file)
    logger.info(f"Updated {local_path}")
    return remote_data


def load_latest_identifier(model_dir: str) -> str:
    """
    Loads the model identifier from latest_models.json if present.
    Returns empty string when unavailable.
    """
    latest_path = os.path.join(model_dir, "latest_models.json")
    if not os.path.exists(latest_path):
        return ""
    try:
        with open(latest_path, encoding="utf-8") as file:
            data = json.load(file)
        latest = data.get("latest")
        return latest if isinstance(latest, str) else ""
    except Exception:
        return ""


def _fetch_companion_files(base_url: str, model_dir: str, model_id: str) -> None:
    """Best-effort download of the runtime-companion files for a variant.

    Besides the weights + labels pair, each HuggingFace release ships a
    ``_model_config.yaml`` (per-variant conf/iou thresholds) and a
    ``_metrics.json`` (recall/precision for the AI panel). Both are
    optional at the HTTP layer — older releases may not have them —
    but when present they unlock:

    - correct threshold regeneration (``model_metadata.json`` derived
      from YAML) so the detector uses per-variant conf/iou instead
      of falling back to hardcoded defaults
    - honest metric display in the settings AI panel
      (bird_recall/f1 shown instead of null)

    After the YAML lands on disk this function also regenerates
    ``<model_dir>/model_metadata.json`` so a cold-start autofetch
    gives the detector the same thresholds a UI-driven pin would.
    (Regression guard: observed 2026-04-18 on the RPi — Tiny cold-
    start inherited a stale S model_metadata.json from a previous
    boot, so Tiny ran with conf=0.30 instead of 0.15.)

    The _README.md (release notes) and _best_int8.onnx (quantised
    weights) are intentionally NOT pulled here — README belongs on
    HF, INT8 is a separate deployment mode that needs explicit config.
    """
    companions = (
        f"{model_id}_model_config.yaml",
        f"{model_id}_metrics.json",
    )
    yaml_path = os.path.join(model_dir, f"{model_id}_model_config.yaml")
    yaml_existed_before = os.path.exists(yaml_path)

    for basename in companions:
        local_path = os.path.join(model_dir, basename)
        if os.path.exists(local_path):
            logger.debug(f"Companion file already present: {local_path}")
            continue
        url = f"{base_url}/{basename}"
        ok = _download_file(url, local_path)
        if not ok:
            # Older release without this companion — expected, not an error.
            logger.info(
                f"Companion {basename} not on HF (older release?); "
                f"continuing without it"
            )

    # Regenerate model_metadata.json from the YAML whenever the YAML is
    # on disk. Two scenarios trigger this:
    #   - Fresh download: YAML did not exist before, now it does.
    #   - Stale metadata guard: YAML existed already, but
    #     model_metadata.json may be from a previous (different) variant
    #     or absent entirely. Always rebuild it here to keep them in sync.
    # Only applies to the detector (object_detection) — the classifier
    # ships a YAML too (per the 2026-04-18 HF release layout spec), but
    # the classifier runtime has no model_metadata.json reader so writing
    # one there would just litter the filesystem.
    if os.path.exists(yaml_path) and _task_name_from_cache_dir(model_dir) == "OBJECT_DETECTION":
        _regenerate_model_metadata_from_yaml(model_dir, yaml_path, yaml_existed_before)


def _regenerate_model_metadata_from_yaml(
    model_dir: str, yaml_path: str, yaml_existed_before: bool
) -> None:
    """Write ``<model_dir>/model_metadata.json`` from a variant's YAML.

    Best-effort: silently skipped when PyYAML or the metadata generator
    module cannot be imported (shouldn't happen at runtime, but keeps
    the autofetch path robust in trimmed-down environments).
    """
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    try:
        import yaml as _yaml

        from utils.model_metadata_generator import config_to_metadata
    except ImportError as exc:
        logger.debug(f"metadata regen skipped: {exc}")
        return
    try:
        config = _yaml.safe_load(open(yaml_path, encoding="utf-8").read())
        if not isinstance(config, dict):
            logger.warning(
                f"metadata regen skipped: {yaml_path} top-level YAML is not a mapping"
            )
            return
        metadata = config_to_metadata(
            config, source_yaml_name=os.path.basename(yaml_path)
        )
        tmp_path = f"{metadata_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(metadata, indent=2) + "\n")
        os.replace(tmp_path, metadata_path)
        thr = metadata.get("inference_thresholds", {}) or {}
        reason = "fresh yaml" if not yaml_existed_before else "refresh for consistency"
        logger.info(
            f"model_metadata.json regenerated from {os.path.basename(yaml_path)} "
            f"({reason}; conf={thr.get('confidence')}, iou={thr.get('iou_nms')})"
        )
    except Exception as exc:
        logger.warning(f"metadata regen failed: {exc}")


def ensure_model_files(
    base_url: str,
    model_dir: str,
    weights_key: str,
    labels_key: str,
    with_companions: bool = True,
) -> tuple[str, str]:
    """Ensures that weights and labels are available locally.

    When ``with_companions`` is True (default), also pulls the per-variant
    companion files (``_model_config.yaml`` and ``_metrics.json``) as a
    best-effort step — see :func:`_fetch_companion_files` for the rationale.
    Runtime still succeeds when the companions are absent.

    Pass ``with_companions=False`` for model lineages that do not ship
    companions by design (e.g. the classifier). This avoids three retry
    rounds per companion (~6 s total) and the noisy ``Download failed
    permanently`` ERROR lines that follow.
    """
    data = fetch_latest_json(base_url, model_dir)
    # Resolve weights path using provided key or common alternates
    weights_rel: str | None = _first_present(
        data, (weights_key, "weights_path", "onnx_path", "model", "path")
    )
    # Resolve labels/classes using provided key or common alternates
    labels_rel: str | None = _first_present(
        data, (labels_key, "labels_path", "labels", "classes_path")
    )
    if not weights_rel:
        raise ValueError(
            "latest_models.json does not contain a valid path for weights."
        )

    # Normalize and try to infer labels if missing
    weights_rel_norm = _normalize_rel_path(base_url, weights_rel)
    if not labels_rel:
        guessed = _guess_labels_from_weights(weights_rel_norm)
        if guessed:
            labels_rel = guessed
            logger.warning(
                f"{labels_key} is missing. Guessing labels from weights: {labels_rel}"
            )
    if not labels_rel:
        raise ValueError("latest_models.json does not contain all required paths.")
    labels_rel_norm = _normalize_rel_path(base_url, labels_rel)

    weights_path = os.path.join(model_dir, os.path.basename(weights_rel_norm))
    labels_path = os.path.join(model_dir, os.path.basename(labels_rel_norm))

    if not os.path.exists(weights_path):
        url = f"{base_url}/{weights_rel_norm}"
        logger.debug(f"Downloading weights from {url} to {weights_path}")
        _download_file(url, weights_path)
    else:
        logger.debug(f"Using existing weights {weights_path}")

    if not os.path.exists(labels_path):
        url = f"{base_url}/{labels_rel_norm}"
        logger.debug(f"Downloading labels from {url} to {labels_path}")
        _download_file(url, labels_path)
    else:
        logger.debug(f"Using existing labels {labels_path}")

    # Best-effort companions (YAML + metrics JSON). We derive model_id
    # from the weights basename so we do not depend on a field in the
    # latest_models.json payload that older releases might not have.
    if with_companions:
        model_id = os.path.basename(weights_rel_norm)
        for suffix in ("_best.onnx", "_best.pt"):
            if model_id.endswith(suffix):
                model_id = model_id[: -len(suffix)]
                break
        if model_id:
            _fetch_companion_files(base_url, model_dir, model_id)

    return weights_path, labels_path
