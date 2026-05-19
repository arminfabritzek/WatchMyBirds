"""
Tests for the two-stage species/species_review/genus/reject gate.

Covers the new ``gallery_threshold`` / ``review_threshold`` machinery
shipped with the cls_v19 per-species thresholds bundle. The single-
stage backward-compat path is covered by test_cls_config_backcompat.py;
this file pins exactly the new behaviour.
"""

import os
import tempfile

import numpy as np
import yaml

from detectors.cls_config import (
    ClsDecisionConfig,
    PerSpeciesThreshold,
    decide_label,
    load_cls_decision_config,
)

CLASSES = [
    "Parus_major",
    "Cyanistes_caeruleus",
    "Sylvia_atricapilla",
    "Sylvia_borin",
    "Turdus_merula",
    "Turdus_philomelos",
]

GENUS_MAP = {
    "Parus_major": "Parus",
    "Cyanistes_caeruleus": "Cyanistes",
    "Sylvia_atricapilla": "Sylvia",
    "Sylvia_borin": "Sylvia",
    "Turdus_merula": "Turdus",
    "Turdus_philomelos": "Turdus",
}

GENUS_PAIRS = frozenset({"Sylvia", "Turdus"})


def _make_two_stage_config(
    *,
    species_threshold: float = 0.98,
    genus_threshold: float = 0.55,
    gallery_threshold: float = 0.6873,
    review_threshold: float | None = 0.2618,
    per_species: dict | None = None,
) -> ClsDecisionConfig:
    return ClsDecisionConfig(
        species_threshold=species_threshold,
        genus_threshold=genus_threshold,
        genus_map=GENUS_MAP,
        genus_pairs=GENUS_PAIRS,
        gallery_threshold=gallery_threshold,
        review_threshold=review_threshold,
        per_species=per_species or {},
    )


# ---------------------------------------------------------------------------
# thresholds_for() — per-species precedence
# ---------------------------------------------------------------------------


def test_thresholds_for_uses_per_species_fit_when_present():
    config = _make_two_stage_config(
        per_species={
            "Parus_major": PerSpeciesThreshold(
                gallery_threshold=0.786,
                review_threshold=None,  # collapsed
                source="per_species_fit",
            ),
        }
    )
    assert config.thresholds_for("Parus_major") == (0.786, None)


def test_thresholds_for_falls_back_to_globals_when_class_missing():
    config = _make_two_stage_config(per_species={})
    assert config.thresholds_for("Erithacus_rubecula") == (0.6873, 0.2618)


def test_thresholds_for_falls_back_when_source_is_global_fallback():
    """global_fallback source must defer to bundle-level defaults."""
    config = _make_two_stage_config(
        per_species={
            "Phoenicurus_phoenicurus": PerSpeciesThreshold(
                gallery_threshold=0.0,  # sentinel
                review_threshold=None,
                source="global_fallback",
            ),
        }
    )
    assert config.thresholds_for("Phoenicurus_phoenicurus") == (
        0.6873,
        0.2618,
    )


# ---------------------------------------------------------------------------
# decide_label — two-stage routing
# ---------------------------------------------------------------------------


def _probs_for(top1: str, top1_prob: float) -> np.ndarray:
    """Build a 6-class probability vector where ``top1`` has ``top1_prob``.

    Remaining mass spread uniformly across the other classes — keeps
    the genus-sibling tests independent of where the leftover mass
    lands except where we override it.
    """
    n = len(CLASSES)
    top1_idx = CLASSES.index(top1)
    out = np.full(n, (1.0 - top1_prob) / (n - 1), dtype=float)
    out[top1_idx] = top1_prob
    return out


def test_above_gallery_threshold_routes_to_species():
    config = _make_two_stage_config(
        per_species={
            "Parus_major": PerSpeciesThreshold(
                gallery_threshold=0.786,
                review_threshold=None,
                source="per_species_fit",
            ),
        }
    )
    probs = _probs_for("Parus_major", 0.85)
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "species"
    assert result["label"] == "Parus_major"
    assert result["prob"] == 0.85


def test_between_thresholds_routes_to_species_review():
    """Top-1 in [review_thr, gallery_thr) lands in the Unclear bucket."""
    # Global thresholds, no per-species override — Cyanistes is not
    # in genus_pairs, so genus fallback won't interfere.
    config = _make_two_stage_config(
        gallery_threshold=0.80,
        review_threshold=0.30,
    )
    probs = _probs_for("Cyanistes_caeruleus", 0.55)
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "species_review"
    assert result["label"] == "Cyanistes_caeruleus"
    assert result["raw_species"] == "Cyanistes_caeruleus"
    assert 0.54 < result["prob"] < 0.56


def test_collapsed_to_gallery_drops_below_threshold_not_review():
    """review_threshold=None means: no Review bucket; drop instead."""
    config = _make_two_stage_config(
        per_species={
            "Parus_major": PerSpeciesThreshold(
                gallery_threshold=0.80,
                review_threshold=None,  # collapsed
                source="per_species_fit",
            ),
        }
    )
    # Parus is not in genus_pairs so genus fallback can't rescue it.
    probs = _probs_for("Parus_major", 0.50)
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"


def test_below_review_threshold_routes_to_reject():
    config = _make_two_stage_config(
        gallery_threshold=0.80,
        review_threshold=0.30,
    )
    # Cyanistes not in genus_pairs → no fallback rescue.
    probs = _probs_for("Cyanistes_caeruleus", 0.15)
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"


def test_genus_fallback_promotes_out_of_review():
    """When sibling mass ≥ genus_threshold INSIDE Review, return genus."""
    config = _make_two_stage_config(
        gallery_threshold=0.80,
        review_threshold=0.30,
        genus_threshold=0.55,
    )
    # Sylvia_atricapilla 0.40 + Sylvia_borin 0.30 → sibling sum 0.70 ≥ 0.55.
    # Top-1 is 0.40 — between review and gallery → would be species_review.
    # But sibling-mass check should promote to genus.
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Sylvia_atricapilla")] = 0.40
    probs[CLASSES.index("Sylvia_borin")] = 0.30
    probs[CLASSES.index("Cyanistes_caeruleus")] = 0.30  # fill mass
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "genus"
    assert result["label"] == "Sylvia_sp."
    assert abs(result["prob"] - 0.70) < 1e-9
    assert result["raw_species"] == "Sylvia_atricapilla"


def test_genus_fallback_does_not_run_below_review_threshold():
    """Drop-bucket samples are strictly dropped — no genus rescue.

    This pins the 2026-05-19 fix: previously the drop branch also ran
    ``_maybe_genus_label``, which produced FPs on confused-genus
    distributions (e.g. a non-bird crop where Turdus_merula 0.30 +
    Turdus_philomelos 0.43 = 0.73 crossed genus_threshold). The
    temporal smoother then confirmed the fake event because the
    non-bird object stayed in frame. Below the review threshold the
    classifier is too uncertain for ANY label, including a genus one
    — better silence than a fake Turdus_sp.

    Contract source: model bundle threshold_contract.two_stage_note:
        "Genus fallback ... still applies INSIDE the Review bucket."
    """
    config = _make_two_stage_config(
        gallery_threshold=0.80,
        review_threshold=0.50,
        genus_threshold=0.55,
    )
    # Top-1 Turdus_merula 0.45 — BELOW review 0.50, so drop bucket.
    # Sibling sum 0.45 + 0.20 = 0.65 ≥ 0.55, which under the old
    # rule WOULD have triggered a Turdus_sp. rescue. Under the new
    # rule the drop is final: no rescue from below review.
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Turdus_merula")] = 0.45
    probs[CLASSES.index("Turdus_philomelos")] = 0.20
    probs[CLASSES.index("Parus_major")] = 0.35
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"
    # raw_species is still surfaced so the operator can see what the
    # classifier *guessed* in the Unclear/Review backlog audit, even
    # though no UI bucket receives the row.
    assert result["raw_species"] == "Turdus_merula"


def test_genus_fallback_still_runs_inside_review_bucket():
    """Inside Review (top-1 ∈ [review_thr, gallery_thr)), genus rescue is on.

    Counterpart to the test above: the Review branch keeps its genus
    rescue so genuinely confused-between-siblings frames get a
    meaningful genus label instead of an opaque species_review row.
    """
    config = _make_two_stage_config(
        gallery_threshold=0.80,
        review_threshold=0.50,
        genus_threshold=0.55,
    )
    # Top-1 Turdus_merula 0.55 — INSIDE Review (≥ 0.50, < 0.80).
    # Sibling sum 0.55 + 0.20 = 0.75 ≥ 0.55 → rescue fires.
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Turdus_merula")] = 0.55
    probs[CLASSES.index("Turdus_philomelos")] = 0.20
    probs[CLASSES.index("Parus_major")] = 0.25
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "genus"
    assert result["label"] == "Turdus_sp."


# ---------------------------------------------------------------------------
# Single-stage backward-compat (no gallery_threshold in bundle)
# ---------------------------------------------------------------------------


def test_single_stage_still_works_without_gallery_threshold():
    """A pre-2026-05-19 bundle has no gallery_threshold → legacy path."""
    config = ClsDecisionConfig(
        species_threshold=0.50,
        genus_threshold=0.55,
        genus_map=GENUS_MAP,
        genus_pairs=GENUS_PAIRS,
        gallery_threshold=None,
        review_threshold=None,
        per_species={},
    )
    probs = _probs_for("Parus_major", 0.60)
    result = decide_label(probs, CLASSES, config)
    # Above species_threshold → species. No species_review even
    # exists in this universe.
    assert result["level"] == "species"


def test_single_stage_never_produces_species_review():
    config = ClsDecisionConfig(
        species_threshold=0.90,
        genus_threshold=0.55,
        genus_map=GENUS_MAP,
        genus_pairs=GENUS_PAIRS,
        gallery_threshold=None,
        review_threshold=None,
        per_species={},
    )
    probs = _probs_for("Cyanistes_caeruleus", 0.40)
    result = decide_label(probs, CLASSES, config)
    assert result["level"] != "species_review"
    assert result["level"] == "reject"


# ---------------------------------------------------------------------------
# YAML loader — reads the new schema
# ---------------------------------------------------------------------------


def _write_yaml(d: dict) -> str:
    fd, path = tempfile.mkstemp(suffix="_model_config.yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f)
    return path


def test_loader_reads_two_stage_globals():
    yaml_data = {
        "detection": {
            "confidence_threshold": 0.98,
            "genus_fallback_threshold": 0.55,
            "gallery_threshold": 0.6873,
            "review_threshold": 0.2618,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {"genus_pairs": ["Sylvia", "Turdus"]},
    }
    path = _write_yaml(yaml_data)
    try:
        model_dir = os.path.dirname(path)
        # _locate_yaml looks for <model_id>_model_config.yaml
        model_id = os.path.basename(path).replace("_model_config.yaml", "")
        config = load_cls_decision_config(model_dir, model_id)
    finally:
        os.unlink(path)
    assert config is not None
    assert config.gallery_threshold == 0.6873
    assert config.review_threshold == 0.2618


def test_loader_reads_per_species_overrides_and_collapses():
    yaml_data = {
        "detection": {
            "confidence_threshold": 0.98,
            "genus_fallback_threshold": 0.55,
            "gallery_threshold": 0.6873,
            "review_threshold": 0.2618,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {"genus_pairs": ["Sylvia", "Turdus"]},
        "per_species_thresholds": {
            "Parus_major": {
                "gallery_threshold": 0.7857,
                "review_threshold": None,
                "threshold_source": "per_species_fit",
                "review_status": "collapsed_to_gallery",
            },
            "Sylvia_atricapilla": {
                "gallery_threshold": 0.9925,
                "review_threshold": 0.2724,
                "threshold_source": "per_species_fit",
            },
            "Phoenicurus_phoenicurus": {
                "threshold_source": "global_fallback",
                "threshold_source_reason": "val_predictions=28 < 30",
            },
        },
    }
    path = _write_yaml(yaml_data)
    try:
        model_dir = os.path.dirname(path)
        model_id = os.path.basename(path).replace("_model_config.yaml", "")
        config = load_cls_decision_config(model_dir, model_id)
    finally:
        os.unlink(path)
    assert config is not None
    assert "Parus_major" in config.per_species
    parus = config.per_species["Parus_major"]
    assert parus.gallery_threshold == 0.7857
    assert parus.review_threshold is None  # collapsed
    assert parus.source == "per_species_fit"

    assert "Sylvia_atricapilla" in config.per_species
    sylvia = config.per_species["Sylvia_atricapilla"]
    assert sylvia.review_threshold == 0.2724

    # Global-fallback entry is loaded with the marker source
    assert "Phoenicurus_phoenicurus" in config.per_species
    pp = config.per_species["Phoenicurus_phoenicurus"]
    assert pp.source == "global_fallback"
    # thresholds_for must defer to globals for this class
    assert config.thresholds_for("Phoenicurus_phoenicurus") == (
        0.6873,
        0.2618,
    )


def test_loader_legacy_yaml_without_two_stage_section():
    """Old YAML (only confidence_threshold) keeps single-stage behaviour."""
    yaml_data = {
        "detection": {
            "confidence_threshold": 0.98,
            "genus_fallback_threshold": 0.55,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {"genus_pairs": ["Sylvia", "Turdus"]},
    }
    path = _write_yaml(yaml_data)
    try:
        model_dir = os.path.dirname(path)
        model_id = os.path.basename(path).replace("_model_config.yaml", "")
        config = load_cls_decision_config(model_dir, model_id)
    finally:
        os.unlink(path)
    assert config is not None
    assert config.gallery_threshold is None
    assert config.review_threshold is None
    assert config.per_species == {}


def test_loader_skips_malformed_per_species_entries():
    """Per-class parse errors must NOT poison the rest of the bundle."""
    yaml_data = {
        "detection": {
            "confidence_threshold": 0.98,
            "genus_fallback_threshold": 0.55,
            "gallery_threshold": 0.6873,
            "review_threshold": 0.2618,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {"genus_pairs": ["Sylvia"]},
        "per_species_thresholds": {
            "Parus_major": {
                "gallery_threshold": 0.7857,
                "threshold_source": "per_species_fit",
            },
            "Bad_entry": "not a dict",  # malformed
            "Missing_gallery": {
                "threshold_source": "per_species_fit",
                # gallery_threshold missing → skipped
            },
        },
    }
    path = _write_yaml(yaml_data)
    try:
        model_dir = os.path.dirname(path)
        model_id = os.path.basename(path).replace("_model_config.yaml", "")
        config = load_cls_decision_config(model_dir, model_id)
    finally:
        os.unlink(path)
    assert config is not None
    assert "Parus_major" in config.per_species
    assert "Bad_entry" not in config.per_species
    assert "Missing_gallery" not in config.per_species
