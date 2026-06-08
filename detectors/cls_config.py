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
class PerSpeciesThreshold:
    """Per-class two-stage thresholds shipped in the model bundle.

    ``gallery_threshold``  — top-1 ≥ this routes the prediction to
                             Gallery directly.
    ``review_threshold``   — top-1 in [review_threshold, gallery) routes
                             to the Unclear/Review surface. ``None`` means
                             the Review bucket is intentionally closed
                             for this class (collapsed-to-gallery), so
                             samples below ``gallery_threshold`` get
                             rejected outright — the App MUST NOT
                             substitute ``gallery_threshold`` for the
                             missing review value.
    ``source``             — ``"per_species_fit"`` when the threshold was
                             fit on validation data, ``"global_fallback"``
                             when the class had too few val samples and
                             relies on the bundle's globals.
    """

    gallery_threshold: float
    review_threshold: float | None
    source: str


@dataclass(frozen=True)
class PerGenusThreshold:
    """Per-genus fallback threshold shipped in the model bundle.

    Genus-level analog of :class:`PerSpeciesThreshold`. Replaces the
    legacy single global ``genus_fallback_threshold`` for genera that
    were calibrated by the dedicated per-genus fitter (N+1 bundle and
    later).

    ``genus_fallback_threshold`` — sibling-mass τ; when summed
                                   per-genus class probabilities ≥
                                   this, label as ``<Genus>_sp.``.
    ``source``                   — ``"per_genus_fit"`` when the τ was
                                   fit on the per-genus val slice,
                                   ``"global_fallback"`` when the
                                   genus had too few val samples and
                                   the bundle defers to the global
                                   ``detection.genus_fallback_threshold``.
    """

    genus_fallback_threshold: float
    source: str


@dataclass(frozen=True)
class ClsDecisionConfig:
    """Config needed to run the species/genus/reject decision layer.

    Legacy single-stage path:
      Only ``species_threshold``, ``genus_threshold``, ``genus_map`` and
      ``genus_pairs`` are filled. ``gallery_threshold`` is ``None`` and
      ``per_species`` is empty. ``decide_label`` then produces the
      original three-way species/genus/reject result.

    Two-stage path (per_species_thresholds.yaml bundle, post-2026-05-19):
      ``gallery_threshold`` is set and acts as the global fallback.
      ``review_threshold`` may be ``None`` if globally collapsed.
      ``per_species`` carries class-specific overrides. ``decide_label``
      then produces a four-way species/species_review/genus/reject
      result — see ``decide_label``'s docstring.

    When the YAML is absent or the ``detection`` section is incomplete,
    the loader returns ``None`` and the runtime falls back to the legacy
    top-1 path (no thresholds, no fallback).
    """

    species_threshold: float
    genus_threshold: float
    genus_map: dict[str, str]
    genus_pairs: frozenset[str]
    temperature: float = 1.0
    # Two-stage gate — None means "no two-stage YAML, run single-stage".
    gallery_threshold: float | None = None
    review_threshold: float | None = None
    # Per-class overrides keyed by scientific name. Missing classes use
    # the global ``gallery_threshold`` / ``review_threshold``.
    per_species: dict[str, PerSpeciesThreshold] = None  # type: ignore[assignment]
    # Per-genus overrides keyed by genus name. Missing genera or
    # ``global_fallback`` entries fall back to ``genus_threshold``
    # (the legacy global ``detection.genus_fallback_threshold``).
    per_genus: dict[str, PerGenusThreshold] = None  # type: ignore[assignment]
    # Non-bird drop set (cls_v20 and later). Class names whose top-1
    # prediction must be dropped pre-threshold, pre-genus-fallback,
    # pre-DB-persist. Empty set = legacy bird-only classifier.
    non_bird_classes: frozenset[str] = frozenset()

    def __post_init__(self):
        # frozen dataclass mutation trick: object.__setattr__ via the
        # dataclasses internals. Default-empty dict if caller passed None.
        if self.per_species is None:
            object.__setattr__(self, "per_species", {})
        if self.per_genus is None:
            object.__setattr__(self, "per_genus", {})

    def thresholds_for(self, species_key: str) -> tuple[float, float | None]:
        """Return ``(gallery_τ, review_τ)`` for one class.

        Falls back to the global pair when the class has no per-species
        override or the override is marked ``global_fallback``. The
        review value may be ``None`` meaning the Review bucket is closed
        for this class.
        """
        # If the bundle never shipped two-stage globals, callers should
        # not be invoking this — the legacy path handles routing.
        gallery = (
            self.gallery_threshold
            if self.gallery_threshold is not None
            else self.species_threshold
        )
        review = self.review_threshold

        entry = self.per_species.get(species_key) if self.per_species else None
        if entry is None or entry.source != "per_species_fit":
            return gallery, review
        return entry.gallery_threshold, entry.review_threshold

    def thresholds_for_genus(self, genus_name: str) -> float:
        """Return τ_genus for one genus.

        Mirrors :meth:`thresholds_for` but for the genus-fallback path.
        Returns the per-genus τ when the bundle ships a
        ``per_genus_thresholds`` entry for this genus AND the entry is
        marked ``per_genus_fit``. Otherwise falls back to
        ``self.genus_threshold`` (the legacy global
        ``detection.genus_fallback_threshold``).

        ``global_fallback``-source entries explicitly defer to the
        global, matching the bundle's ``threshold_contract``
        precedence rule.
        """
        entry = self.per_genus.get(genus_name) if self.per_genus else None
        if entry is None or entry.source != "per_genus_fit":
            return self.genus_threshold
        return entry.genus_fallback_threshold


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

    # Canonical location (per the publisher's release layout): genus_map is
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

    # Two-stage gate (per_species_thresholds.yaml bundle, optional).
    # When ``detection.gallery_threshold`` is present we read the new
    # globals and any per-species overrides. When absent we leave the
    # two-stage fields at their defaults — ``decide_label`` then runs
    # the legacy single-stage path against ``species_threshold``.
    gallery_threshold = _parse_optional_float(detection.get("gallery_threshold"))
    review_threshold = _parse_optional_float(detection.get("review_threshold"))
    per_species = _load_per_species_thresholds(raw, yaml_path)
    per_genus = _load_per_genus_thresholds(raw, yaml_path)
    non_bird_classes = _load_non_bird_classes(raw, yaml_path)

    if non_bird_classes:
        logger.info(
            "cls config: %d non-bird class(es) configured for drop: %s",
            len(non_bird_classes),
            ", ".join(sorted(non_bird_classes)),
        )

    two_stage_note = ""
    if gallery_threshold is not None:
        review_str = (
            f"{review_threshold:.4f}"
            if review_threshold is not None
            else "null(collapsed)"
        )
        per_genus_fit_count = sum(
            1 for e in per_genus.values() if e.source == "per_genus_fit"
        )
        per_genus_fallback_count = len(per_genus) - per_genus_fit_count
        two_stage_note = (
            f", two-stage: gallery_thr={gallery_threshold:.4f}, "
            f"review_thr={review_str}, per_species={len(per_species)}, "
            f"per_genus={per_genus_fit_count}, "
            f"per_genus_fallback={per_genus_fallback_count}"
        )
        # One-off boot diagnostic: surface the classes that fell back to
        # global thresholds so under-sampled species are visible without
        # parsing the bundle by hand.
        fallback_classes = sorted(
            name
            for name, entry in per_species.items()
            if entry.source != "per_species_fit"
        )
        if fallback_classes:
            logger.info(
                "cls config: %d class(es) on global fallback (val too small): %s",
                len(fallback_classes),
                ", ".join(fallback_classes),
            )

    logger.info(
        f"cls config loaded from {os.path.basename(yaml_path)}: "
        f"species_thr={species_threshold}, genus_thr={genus_threshold}, "
        f"temperature={temperature}, {len(genus_map)} species mapped, "
        f"{len(genus_pairs)} genus-pairs{two_stage_note}"
    )
    return ClsDecisionConfig(
        species_threshold=species_threshold,
        genus_threshold=genus_threshold,
        genus_map=genus_map,
        genus_pairs=genus_pairs,
        temperature=temperature,
        gallery_threshold=gallery_threshold,
        review_threshold=review_threshold,
        per_species=per_species,
        per_genus=per_genus,
        non_bird_classes=non_bird_classes,
    )


def _parse_optional_float(value) -> float | None:
    """Best-effort float parse that treats null/missing as ``None``.

    Accepts ``None``, missing keys, and non-numeric strings as the
    "value is absent" signal. Used for two-stage thresholds where the
    YAML may carry ``review_threshold: null`` to mean "no Review bucket
    for this class — drop instead".
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_per_species_thresholds(
    raw: dict, yaml_path: str
) -> dict[str, PerSpeciesThreshold]:
    """Read ``per_species_thresholds`` from the YAML.

    The schema, verbatim from ``threshold_contract.two_stage_note`` in
    the model bundle::

        per_species_thresholds:
          Parus_major:
            gallery_threshold: 0.7857
            review_threshold: null
            threshold_source: per_species_fit
            review_status: collapsed_to_gallery
          Phoenicurus_phoenicurus:
            threshold_source: global_fallback
            threshold_source_reason: "val_predictions=28 < 30"

    Returns ``{}`` when the section is missing, malformed, or all
    entries are unusable. Per-entry parse errors degrade silently to
    "fall back to globals for this class" — we never raise.
    """
    raw_section = raw.get("per_species_thresholds")
    if not isinstance(raw_section, dict):
        return {}

    out: dict[str, PerSpeciesThreshold] = {}
    for species_key, entry in raw_section.items():
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("threshold_source") or "per_species_fit")
        if source == "global_fallback":
            # Sentinel so callers know to defer to global defaults
            # without having to inspect the YAML themselves.
            out[str(species_key)] = PerSpeciesThreshold(
                gallery_threshold=0.0,
                review_threshold=None,
                source=source,
            )
            continue

        gallery = _parse_optional_float(entry.get("gallery_threshold"))
        if gallery is None:
            logger.warning(
                "cls config %s: per_species[%s] missing gallery_threshold; "
                "using global fallback for this class",
                os.path.basename(yaml_path),
                species_key,
            )
            continue
        review = _parse_optional_float(entry.get("review_threshold"))
        out[str(species_key)] = PerSpeciesThreshold(
            gallery_threshold=gallery,
            review_threshold=review,
            source="per_species_fit",
        )

        # Boot-time visibility for classes where the model bundle
        # itself flags known-unfixed Field FPs above tau. Operator
        # otherwise has no way to know why these FPs keep landing in
        # Gallery — they look like normal per_species_fit entries.
        known_field_fps = entry.get("cross_species_known_field_fps_above_tau")
        if isinstance(known_field_fps, (int, float)) and known_field_fps > 0:
            logger.warning(
                "cls config: %s ships known-unfixed cross-species FPs above "
                "gallery_threshold=%.4f (count=%d). Field FPs will continue "
                "until a future bundle ships better cross-species data for "
                "this class. See bundle threshold_source_reason for context.",
                species_key,
                gallery,
                int(known_field_fps),
            )
    return out


def _load_non_bird_classes(raw: dict, yaml_path: str) -> frozenset[str]:
    """Read top-level ``non_bird_classes`` from the YAML.

    Cls_v20 and later ship a list of class names that, when predicted
    top-1, must be dropped pre-threshold and pre-genus-fallback. This
    is the App's contract for the model bundle's non-bird gate.

    Returns an empty frozenset when the field is missing, empty, or
    malformed — older bundles and bird-only classifiers behave
    byte-identical to before this change.
    """
    raw_section = raw.get("non_bird_classes")
    if raw_section is None:
        return frozenset()
    if not isinstance(raw_section, list):
        logger.warning(
            "non_bird_classes in %s is not a list (%r); ignoring.",
            yaml_path,
            type(raw_section).__name__,
        )
        return frozenset()
    out: set[str] = set()
    for entry in raw_section:
        if not isinstance(entry, str) or not entry.strip():
            logger.warning(
                "non_bird_classes entry %r in %s is not a non-empty string; dropped.",
                entry,
                yaml_path,
            )
            continue
        out.add(entry.strip())
    return frozenset(out)


def _load_per_genus_thresholds(
    raw: dict, yaml_path: str
) -> dict[str, PerGenusThreshold]:
    """Read ``per_genus_thresholds`` from the YAML.

    Genus-level analog of :func:`_load_per_species_thresholds`. The
    schema, verbatim from the bundle's ``threshold_contract``
    "Per-genus precedence" section::

        per_genus_thresholds:
          Sylvia:
            genus_fallback_threshold: 0.8786
            threshold_source: per_genus_fit
            fit_diagnostics:
              val_n_predictions: 94
              val_precision_at_tau: 1.0
              ...
          Some_unfit_genus:
            threshold_source: global_fallback
            threshold_source_reason: "val_predictions < 30"

    Returns ``{}`` when the section is missing, malformed, or all
    entries are unusable. Per-entry parse errors degrade silently to
    "fall back to the global ``genus_threshold`` for this genus" —
    we never raise. The ``fit_diagnostics`` sub-block is read for
    boot logging but not stored on the dataclass.
    """
    raw_section = raw.get("per_genus_thresholds")
    if not isinstance(raw_section, dict):
        return {}

    out: dict[str, PerGenusThreshold] = {}
    for genus_key, entry in raw_section.items():
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("threshold_source") or "per_genus_fit")
        if source == "global_fallback":
            # Sentinel so callers know to defer to ``genus_threshold``
            # without inspecting the YAML themselves.
            out[str(genus_key)] = PerGenusThreshold(
                genus_fallback_threshold=0.0,
                source=source,
            )
            continue

        threshold = _parse_optional_float(entry.get("genus_fallback_threshold"))
        if threshold is None:
            logger.warning(
                "cls config %s: per_genus[%s] missing genus_fallback_threshold; "
                "using global fallback for this genus",
                os.path.basename(yaml_path),
                genus_key,
            )
            continue
        out[str(genus_key)] = PerGenusThreshold(
            genus_fallback_threshold=threshold,
            source="per_genus_fit",
        )
    return out


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
    """Apply the species/species_review/genus/reject decision.

    Returns a dict with keys ``label``, ``prob``, ``level``,
    ``raw_species``.

    Possible ``level`` values:

    - ``"species"``         — confident enough for Gallery.
    - ``"species_review"``  — between Review and Gallery thresholds.
                              Routed to the Unclear surface, where a
                              human confirms or discards. Only emitted
                              when the bundle ships two-stage globals
                              AND the per-class Review bucket is open.
    - ``"genus"``           — genus fallback fired (sibling-mass crossed
                              ``genus_threshold``). Routed like a
                              species result.
    - ``"reject"``          — drop. No UI surface.

    When ``config`` is ``None``, returns the legacy "species always
    wins" result so older models keep working. When ``config`` has only
    single-stage thresholds (no ``gallery_threshold``), runs the
    legacy three-way species/genus/reject path against
    ``species_threshold``. The four-way path only activates when the
    bundle has ``detection.gallery_threshold`` set.

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

    # Non-bird drop (cls_v20 and later). Runs BEFORE any threshold or
    # genus-fallback logic: a top-1 in non_bird_classes is by contract
    # a substrate / not-a-bird prediction and must not reach Gallery,
    # Review, or DB persistence under any species label.
    if top1_species in config.non_bird_classes:
        return {
            "label": "",
            "prob": top1_prob,
            "level": "reject",
            "raw_species": top1_species,
        }

    # Two-stage path — only when the bundle shipped gallery_threshold.
    # Falls through to the legacy single-stage path otherwise so older
    # YAML bundles produce identical output to before this change.
    if config.gallery_threshold is not None:
        gallery_thr, review_thr = config.thresholds_for(top1_species)

        if top1_prob >= gallery_thr:
            return {
                "label": top1_species,
                "prob": top1_prob,
                "level": "species",
                "raw_species": top1_species,
            }

        # Review bucket — only when the per-class entry left it open.
        # ``review_thr is None`` means ``collapsed_to_gallery``: the App
        # MUST drop instead of substituting gallery_thr (per the model
        # bundle's threshold_contract.two_stage_note).
        if review_thr is not None and top1_prob >= review_thr:
            # Genus fallback still applies INSIDE Review. When sibling
            # mass crosses genus_threshold we relabel as "<Genus>_sp."
            # and promote out of species_review into the genus bucket —
            # matching the bundle's two_stage_note contract.
            genus_label = _maybe_genus_label(probs, classes, config, top1_species)
            if genus_label is not None:
                return genus_label
            return {
                "label": top1_species,
                "prob": top1_prob,
                "level": "species_review",
                "raw_species": top1_species,
            }

        # Below review threshold (or no Review bucket for this class):
        # drop, NO genus rescue. Two reasons this branch deliberately
        # skips ``_maybe_genus_label``:
        #
        # 1. The model bundle's threshold_contract.two_stage_note pins
        #    the contract: "Genus fallback ... still applies INSIDE
        #    the Review bucket." Inside Review only, not below it.
        #
        # 2. Empirically: genus rescue from
        #    the drop branch produced false positives on non-bird
        #    frames where the classifier saw a flat-uncertain
        #    distribution over small/brown classes (e.g. a fat-ball
        #    crop summing 0.30 Turdus_philomelos + 0.43 Turdus_merula
        #    to 0.73, crossing the 0.55 genus_threshold). The temporal
        #    smoother then confirmed those "events" because the fake
        #    object stayed put across frames, and Turdus_sp. landed in
        #    Gallery without a real bird.
        #
        # Drop-branch frames whose top-1 is a confused-genus member
        # are now strictly dropped: better silence than a fake
        # genus-level sighting.
        return {
            "label": "",
            "prob": top1_prob,
            "level": "reject",
            "raw_species": top1_species,
        }

    # Legacy single-stage path (pre-two-stage YAML bundles).
    if top1_prob >= config.species_threshold:
        return {
            "label": top1_species,
            "prob": top1_prob,
            "level": "species",
            "raw_species": top1_species,
        }

    genus_label = _maybe_genus_label(probs, classes, config, top1_species)
    if genus_label is not None:
        return genus_label

    return {
        "label": "",
        "prob": top1_prob,
        "level": "reject",
        "raw_species": top1_species,
    }


def _maybe_genus_label(
    probs,
    classes: list[str],
    config: ClsDecisionConfig,
    top1_species: str,
) -> dict | None:
    """Run the genus-sibling mass check; return a genus dict or ``None``.

    Factored out so both the two-stage and the legacy path share the
    exact same fallback semantics — a bug in one is a bug in both.

    Threshold precedence (N+1 bundle and later): the bundle may ship a
    ``per_genus_thresholds`` block with a per-genus τ. When the entry
    is marked ``per_genus_fit`` it overrides ``config.genus_threshold``
    (the legacy global ``detection.genus_fallback_threshold``). Missing
    or ``global_fallback`` entries defer to the global, matching the
    bundle's threshold_contract precedence rule. The lookup is
    centralised in :meth:`ClsDecisionConfig.thresholds_for_genus`.
    """
    genus = config.genus_map.get(top1_species)
    if not genus or genus not in config.genus_pairs:
        return None
    sibling_mass = 0.0
    for idx, sp in enumerate(classes):
        if config.genus_map.get(sp) == genus:
            sibling_mass += float(probs[idx])
    if sibling_mass < config.thresholds_for_genus(genus):
        return None
    return {
        "label": f"{genus}_sp.",
        "prob": float(sibling_mass),
        "level": "genus",
        "raw_species": top1_species,
    }
