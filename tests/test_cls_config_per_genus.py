"""Tests for the per_genus_thresholds path of the classifier bundle.

Covers the N+1 (2026-05-20) classifier bundle's new behaviour:

- A new ``per_genus_thresholds`` block in the YAML is parsed into a
  ``ClsDecisionConfig.per_genus`` map of :class:`PerGenusThreshold`.
- :meth:`ClsDecisionConfig.thresholds_for_genus` returns the per-genus
  τ when ``threshold_source: per_genus_fit``, otherwise falls back to
  the global ``detection.genus_fallback_threshold``.
- :func:`_maybe_genus_label` consults the per-genus τ (not the legacy
  global) when deciding whether sibling-mass crosses the bar.
- Bundles without a ``per_genus_thresholds`` block produce
  ``per_genus={}`` and behave byte-identically to the pre-N+1 loader.

Test matrix from the focus plan
(2026-05-20_FEATURE_consume-classifier-n1-per-genus-thresholds):

1. bundle has per-genus + genus is in it           -> uses per-genus τ
2. bundle has per-genus + genus is not in it       -> uses global τ
3. bundle has per-genus + entry is global_fallback -> uses global τ
4. bundle has no per-genus block                   -> uses global τ
5. _maybe_genus_label routing test: top-1 below
   τ_gal[species] but sibling-mass >= per-genus τ  -> level='genus'
   with the per-genus τ (not global) deciding the bar.
"""

import numpy as np
import yaml

from detectors.cls_config import (
    ClsDecisionConfig,
    PerGenusThreshold,
    _load_per_genus_thresholds,
    _maybe_genus_label,
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


def _make_config(
    *,
    genus_threshold: float = 0.55,
    per_species: dict | None = None,
    per_genus: dict | None = None,
) -> ClsDecisionConfig:
    return ClsDecisionConfig(
        species_threshold=0.98,
        genus_threshold=genus_threshold,
        genus_map=GENUS_MAP,
        genus_pairs=GENUS_PAIRS,
        gallery_threshold=0.6873,
        review_threshold=0.2618,
        per_species=per_species or {},
        per_genus=per_genus or {},
    )


# ---------------------------------------------------------------------------
# A1 — Loader parses per_genus_thresholds block from YAML
# ---------------------------------------------------------------------------


def _yaml_with_per_genus(per_genus_block: dict) -> dict:
    return {
        "detection": {
            "confidence_threshold": 0.98,
            "genus_fallback_threshold": 0.55,
            "gallery_threshold": 0.6873,
            "review_threshold": 0.2618,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {
            "genus_pairs": list(GENUS_PAIRS),
        },
        "per_genus_thresholds": per_genus_block,
    }


def test_load_per_genus_thresholds_per_genus_fit():
    """Cell 1: bundle has per-genus + genus is in it -> entry parsed."""
    raw = _yaml_with_per_genus(
        {
            "Sylvia": {
                "genus_fallback_threshold": 0.8786,
                "threshold_source": "per_genus_fit",
                "fit_diagnostics": {"val_n_predictions": 94},
            },
        }
    )
    out = _load_per_genus_thresholds(raw, "test.yaml")
    assert "Sylvia" in out
    assert out["Sylvia"].source == "per_genus_fit"
    assert out["Sylvia"].genus_fallback_threshold == 0.8786


def test_load_per_genus_thresholds_global_fallback():
    """Cell 3: bundle has per-genus + entry is global_fallback ->
    sentinel parsed with source set so callers know to defer."""
    raw = _yaml_with_per_genus(
        {
            "Unfit_genus": {
                "threshold_source": "global_fallback",
                "threshold_source_reason": "val_predictions=12 < 30",
            },
        }
    )
    out = _load_per_genus_thresholds(raw, "test.yaml")
    assert "Unfit_genus" in out
    assert out["Unfit_genus"].source == "global_fallback"
    # Sentinel value — not meaningful, callers must defer to global.
    assert out["Unfit_genus"].genus_fallback_threshold == 0.0


def test_load_per_genus_thresholds_missing_section():
    """Cell 4: bundle has no per-genus block -> empty dict."""
    raw = {
        "detection": {"confidence_threshold": 0.98, "genus_fallback_threshold": 0.55},
        "genus_map": GENUS_MAP,
    }
    out = _load_per_genus_thresholds(raw, "test.yaml")
    assert out == {}


def test_load_per_genus_thresholds_missing_threshold_skipped():
    """Per-entry parse error: missing genus_fallback_threshold is logged
    but does not raise — class drops to global-fallback semantics by
    being absent from the parsed dict."""
    raw = _yaml_with_per_genus(
        {
            "Broken_genus": {
                "threshold_source": "per_genus_fit",
                # missing genus_fallback_threshold
            },
            "Good_genus": {
                "genus_fallback_threshold": 0.75,
                "threshold_source": "per_genus_fit",
            },
        }
    )
    out = _load_per_genus_thresholds(raw, "test.yaml")
    assert "Broken_genus" not in out
    assert "Good_genus" in out


def test_load_per_genus_thresholds_malformed_section_returns_empty():
    """Section is a string / list / something not-a-dict -> {}."""
    raw = {"per_genus_thresholds": "not a dict"}
    assert _load_per_genus_thresholds(raw, "test.yaml") == {}


# ---------------------------------------------------------------------------
# A2 — thresholds_for_genus() precedence
# ---------------------------------------------------------------------------


def test_thresholds_for_genus_uses_per_genus_fit():
    """Cell 1: per_genus_fit entry overrides the global."""
    config = _make_config(
        genus_threshold=0.55,
        per_genus={
            "Sylvia": PerGenusThreshold(
                genus_fallback_threshold=0.8786, source="per_genus_fit"
            ),
        },
    )
    assert config.thresholds_for_genus("Sylvia") == 0.8786


def test_thresholds_for_genus_unknown_genus_falls_back():
    """Cell 2: genus not in per_genus -> global."""
    config = _make_config(genus_threshold=0.55, per_genus={})
    assert config.thresholds_for_genus("Unknown_genus") == 0.55


def test_thresholds_for_genus_global_fallback_source_defers():
    """Cell 3: global_fallback entry must defer to the global, NOT
    the sentinel 0.0 value."""
    config = _make_config(
        genus_threshold=0.55,
        per_genus={
            "Unfit_genus": PerGenusThreshold(
                genus_fallback_threshold=0.0, source="global_fallback"
            ),
        },
    )
    assert config.thresholds_for_genus("Unfit_genus") == 0.55


def test_thresholds_for_genus_empty_per_genus_is_legacy_global():
    """Cell 4: empty per_genus map -> all genera get the global τ
    (byte-identical to pre-N+1 behaviour)."""
    config = _make_config(genus_threshold=0.55, per_genus={})
    for genus in ("Sylvia", "Phoenicurus", "Turdus", "Fringilla", "Unknown"):
        assert config.thresholds_for_genus(genus) == 0.55


# ---------------------------------------------------------------------------
# A2 — _maybe_genus_label routing uses per-genus τ
# ---------------------------------------------------------------------------


def test_maybe_genus_label_uses_per_genus_threshold_for_bar():
    """Cell 5: per-genus τ (0.88) is the bar — sibling-mass 0.85 is
    BELOW it even though it would have crossed the legacy global 0.55.

    Without the per-genus path, this sample would have been promoted
    to 'genus' level. With it, the per-genus bar correctly drops it.
    """
    config = _make_config(
        genus_threshold=0.55,
        per_genus={
            "Sylvia": PerGenusThreshold(
                genus_fallback_threshold=0.88, source="per_genus_fit"
            ),
        },
    )
    # Place 0.42 + 0.43 on the two Sylvia siblings: sibling-mass = 0.85.
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Sylvia_atricapilla")] = 0.42
    probs[CLASSES.index("Sylvia_borin")] = 0.43
    # remaining mass on a non-Sylvia class so other_genus = 0.15
    probs[CLASSES.index("Parus_major")] = 0.15
    result = _maybe_genus_label(probs, CLASSES, config, "Sylvia_atricapilla")
    assert result is None, (
        f"Expected None (sibling-mass 0.85 below per-genus τ 0.88), got: {result}"
    )


def test_maybe_genus_label_above_per_genus_threshold_promotes():
    """Sibling-mass above the per-genus τ -> level='genus'."""
    config = _make_config(
        genus_threshold=0.55,
        per_genus={
            "Sylvia": PerGenusThreshold(
                genus_fallback_threshold=0.88, source="per_genus_fit"
            ),
        },
    )
    # Place 0.46 + 0.45 on the two Sylvia siblings: sibling-mass = 0.91.
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Sylvia_atricapilla")] = 0.46
    probs[CLASSES.index("Sylvia_borin")] = 0.45
    probs[CLASSES.index("Parus_major")] = 0.09
    result = _maybe_genus_label(probs, CLASSES, config, "Sylvia_atricapilla")
    assert result is not None
    assert result["level"] == "genus"
    assert result["label"] == "Sylvia_sp."
    # The recorded prob is the sibling-mass, not the top1.
    assert abs(result["prob"] - 0.91) < 1e-9


def test_maybe_genus_label_falls_back_to_global_when_no_per_genus():
    """Cell 4: when per_genus is empty, the legacy global τ (0.55) is
    the bar — sibling-mass 0.6 must cross."""
    config = _make_config(genus_threshold=0.55, per_genus={})
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Sylvia_atricapilla")] = 0.32
    probs[CLASSES.index("Sylvia_borin")] = 0.28
    probs[CLASSES.index("Parus_major")] = 0.40
    result = _maybe_genus_label(probs, CLASSES, config, "Sylvia_atricapilla")
    assert result is not None
    assert result["level"] == "genus"
    assert result["label"] == "Sylvia_sp."


def test_maybe_genus_label_global_fallback_source_uses_global_tau():
    """Cell 3 in routing form: when the per-genus entry's source is
    global_fallback, the routing uses the global τ — sibling-mass 0.6
    must cross 0.55, not the sentinel 0.0."""
    config = _make_config(
        genus_threshold=0.55,
        per_genus={
            "Sylvia": PerGenusThreshold(
                genus_fallback_threshold=0.0, source="global_fallback"
            ),
        },
    )
    probs = np.zeros(len(CLASSES))
    probs[CLASSES.index("Sylvia_atricapilla")] = 0.32
    probs[CLASSES.index("Sylvia_borin")] = 0.28
    probs[CLASSES.index("Parus_major")] = 0.40
    result = _maybe_genus_label(probs, CLASSES, config, "Sylvia_atricapilla")
    assert result is not None, (
        "global_fallback source must defer to the global τ, not the "
        "sentinel 0.0 — sibling-mass 0.6 should cross global 0.55"
    )
    assert result["level"] == "genus"


# ---------------------------------------------------------------------------
# Back-compat: legacy bundles without per_genus_thresholds
# ---------------------------------------------------------------------------


def test_load_cls_decision_config_legacy_bundle_no_per_genus(tmp_path):
    """A YAML without per_genus_thresholds produces per_genus={} on the
    resulting config, and thresholds_for_genus(...) returns the legacy
    global for every genus."""
    yaml_data = {
        "detection": {
            "confidence_threshold": 0.88,
            "genus_fallback_threshold": 0.55,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {"genus_pairs": list(GENUS_PAIRS)},
    }
    yaml_path = tmp_path / "legacy_model_config.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data))
    config = load_cls_decision_config(str(tmp_path), "legacy")
    assert config is not None
    assert config.per_genus == {}
    for genus in ("Sylvia", "Turdus", "AnythingElse"):
        assert config.thresholds_for_genus(genus) == 0.55


def test_load_cls_decision_config_n1_bundle_with_per_genus(tmp_path):
    """An N+1-shaped YAML with per_genus_thresholds populates the
    per_genus map and the routing picks up the per-genus τ."""
    yaml_data = {
        "detection": {
            "confidence_threshold": 0.98,
            "genus_fallback_threshold": 0.55,
            "gallery_threshold": 0.6873,
            "review_threshold": 0.2618,
        },
        "genus_map": GENUS_MAP,
        "threshold_contract": {"genus_pairs": list(GENUS_PAIRS)},
        "per_genus_thresholds": {
            "Sylvia": {
                "genus_fallback_threshold": 0.8786,
                "threshold_source": "per_genus_fit",
            },
            "Turdus": {
                "genus_fallback_threshold": 0.961,
                "threshold_source": "per_genus_fit",
            },
        },
    }
    yaml_path = tmp_path / "n1_model_config.yaml"
    yaml_path.write_text(yaml.safe_dump(yaml_data))
    config = load_cls_decision_config(str(tmp_path), "n1")
    assert config is not None
    assert len(config.per_genus) == 2
    assert config.thresholds_for_genus("Sylvia") == 0.8786
    assert config.thresholds_for_genus("Turdus") == 0.961
    # Unknown genus falls back to legacy global.
    assert config.thresholds_for_genus("UnknownGenus") == 0.55
