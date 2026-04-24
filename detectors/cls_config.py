"""Per-variant classifier config loader.

Newer classifier releases (from 20260423_062443 onward) ship a
``<model_id>_model_config.yaml`` alongside weights + classes.

YAML layout we consume (as shipped 2026-04-24):

    detection:
      confidence_threshold: 0.88             # species accept
      genus_fallback_threshold: 0.55         # sibling-sum accept
      # (other detection.* fields — architecture, input_size, ...
      # — are not consumed here)
    calibration:
      temperature:
        value: 0.99
        enabled: true
    genus_map:                               # TOP-LEVEL (not nested)
      Parus_major: Parus
      ...
    threshold_contract:
      genus_pairs:                           # only genera with >=2 species
        - Parus
        - Sylvia
        ...

Older classifier releases (pre-20260423_062443) ship no YAML, or a
YAML without the ``detection`` section. Those models must keep
working without any threshold or fallback logic — that is this
loader's job: return ``None`` when the config is unusable, and let
the runtime take the legacy top-1 path.

Robustness: we also accept the fields at alternate locations
(``detection.genus_map`` / ``detection.genus_pairs``) as a fallback,
so a future spec drift in either direction does not break the loader.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClsDecisionConfig:
    """Config needed to run the species/genus/reject decision layer.

    When present and complete, the classifier runtime uses all four
    fields together. When the YAML is absent or the ``detection``
    section is incomplete, the loader returns ``None`` and the runtime
    falls back to the legacy top-1 path (no thresholds, no fallback).
    """

    species_threshold: float
    genus_threshold: float
    genus_map: dict[str, str]
    genus_pairs: frozenset[str]
    temperature: float = 1.0


def load_cls_decision_config(
    model_dir: str, model_id: str | None
) -> ClsDecisionConfig | None:
    """Load the decision config for a specific classifier variant.

    Returns ``None`` for any legacy-compat reason:
    - model_id is empty (we cannot locate the YAML without it)
    - YAML file does not exist on disk
    - PyYAML is unavailable
    - YAML is malformed or missing the ``detection`` section
    - required keys missing inside the section

    The caller must treat ``None`` as "run the legacy top-1 path".
    """
    if not model_id:
        logger.debug("cls config skipped: empty model_id")
        return None

    yaml_path = _locate_yaml(model_dir, model_id)
    if yaml_path is None:
        logger.debug(f"cls config not on disk for model_id={model_id}; legacy path")
        return None

    try:
        import yaml as _yaml
    except ImportError as exc:
        logger.debug(f"cls config skipped (no PyYAML): {exc}")
        return None

    try:
        with open(yaml_path, encoding="utf-8") as file:
            raw = _yaml.safe_load(file)
    except Exception as exc:
        logger.warning(f"cls config parse failed for {yaml_path}: {exc}")
        return None

    if not isinstance(raw, dict):
        logger.warning(f"cls config top-level not a mapping: {yaml_path}")
        return None

    detection = raw.get("detection")
    if not isinstance(detection, dict):
        logger.debug(
            f"cls config has no 'detection' section (older-style YAML?): {yaml_path}"
        )
        return None

    try:
        species_threshold = float(detection["confidence_threshold"])
        genus_threshold = float(detection["genus_fallback_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            f"cls config missing/invalid thresholds in {yaml_path}: {exc}; legacy path"
        )
        return None

    # Canonical location (per 2026-04-23 HF spec): genus_map is
    # top-level, genus_pairs lives under threshold_contract. We also
    # accept legacy ``detection.genus_map`` / ``detection.genus_pairs``
    # to avoid breaking either direction of future spec drift.
    threshold_contract = raw.get("threshold_contract")
    threshold_contract = (
        threshold_contract if isinstance(threshold_contract, dict) else {}
    )

    genus_map_raw = raw.get("genus_map") or detection.get("genus_map")
    if not isinstance(genus_map_raw, dict) or not genus_map_raw:
        logger.warning(
            f"cls config missing/empty 'genus_map' in {yaml_path} "
            "(looked at top-level and detection.*); legacy path"
        )
        return None
    genus_map = {str(k): str(v) for k, v in genus_map_raw.items()}

    genus_pairs_raw = (
        threshold_contract.get("genus_pairs")
        if "genus_pairs" in threshold_contract
        else detection.get("genus_pairs", [])
    )
    if not isinstance(genus_pairs_raw, (list, tuple)):
        logger.warning(
            f"cls config 'genus_pairs' not a list in {yaml_path}; using empty set"
        )
        genus_pairs_raw = []
    genus_pairs = frozenset(str(g) for g in genus_pairs_raw)

    temperature = _load_temperature(raw, yaml_path)

    logger.info(
        f"cls config loaded from {os.path.basename(yaml_path)}: "
        f"species_thr={species_threshold}, genus_thr={genus_threshold}, "
        f"temperature={temperature}, {len(genus_map)} species mapped, "
        f"{len(genus_pairs)} genus-pairs"
    )
    return ClsDecisionConfig(
        species_threshold=species_threshold,
        genus_threshold=genus_threshold,
        genus_map=genus_map,
        genus_pairs=genus_pairs,
        temperature=temperature,
    )


def _load_temperature(raw: dict, yaml_path: str) -> float:
    """Return calibrated softmax temperature, defaulting to 1.0.

    The runtime contract is intentionally forgiving:
    - missing ``calibration`` block -> ``1.0``
    - disabled temperature block -> ``1.0``
    - invalid/non-positive value -> ``1.0`` with a warning
    """
    calibration = raw.get("calibration")
    if not isinstance(calibration, dict):
        return 1.0

    temperature = calibration.get("temperature")
    if isinstance(temperature, (int, float)):
        return float(temperature) if float(temperature) > 0 else 1.0

    if not isinstance(temperature, dict):
        return 1.0

    enabled = temperature.get("enabled", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(enabled)
    if not enabled:
        return 1.0

    value = temperature.get("value")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            f"cls config has invalid calibration.temperature.value in {yaml_path}; "
            "falling back to T=1.0"
        )
        return 1.0
    if parsed <= 0:
        logger.warning(
            f"cls config has non-positive calibration.temperature.value in {yaml_path}; "
            "falling back to T=1.0"
        )
        return 1.0
    return parsed


def _locate_yaml(model_dir: str, model_id: str) -> str | None:
    """Find the model_config.yaml for this variant, if on disk.

    Tries the canonical HF release name first
    (``<model_id>_model_config.yaml``). Falls back to a glob so minor
    naming drift in a dev/pinned setup still resolves.
    """
    canonical = os.path.join(model_dir, f"{model_id}_model_config.yaml")
    if os.path.exists(canonical):
        return canonical

    matches = glob.glob(os.path.join(model_dir, f"{model_id}*model_config*.yaml"))
    if matches:
        return matches[0]

    return None


def decide_label(
    probabilities,
    classes: list[str],
    config: ClsDecisionConfig | None,
) -> dict:
    """Apply the species/genus/reject decision to a softmax vector.

    Returns a dict with keys ``label``, ``prob``, ``level``,
    ``raw_species``. When ``config`` is ``None``, returns the legacy
    "species always wins" result so older models keep working.

    Shape contract: ``probabilities`` is a 1D array-like over the same
    class ordering as ``classes``.
    """
    probs = np.asarray(probabilities).reshape(-1)
    top1_idx = int(np.argmax(probs))
    top1_prob = float(probs[top1_idx])
    top1_species = classes[top1_idx] if 0 <= top1_idx < len(classes) else ""

    # Legacy path: no config available. Keep the old top-1 behaviour so
    # any older pinned classifier keeps producing species labels as it
    # always has (no threshold gating, no genus fallback).
    if config is None:
        return {
            "label": top1_species,
            "prob": top1_prob,
            "level": "species",
            "raw_species": top1_species,
        }

    if top1_prob >= config.species_threshold:
        return {
            "label": top1_species,
            "prob": top1_prob,
            "level": "species",
            "raw_species": top1_species,
        }

    genus = config.genus_map.get(top1_species)
    if genus and genus in config.genus_pairs:
        sibling_mass = 0.0
        for idx, sp in enumerate(classes):
            if config.genus_map.get(sp) == genus:
                sibling_mass += float(probs[idx])
        if sibling_mass >= config.genus_threshold:
            return {
                "label": f"{genus}_sp.",
                "prob": float(sibling_mass),
                "level": "genus",
                "raw_species": top1_species,
            }

    return {
        "label": "",
        "prob": top1_prob,
        "level": "reject",
        "raw_species": top1_species,
    }
