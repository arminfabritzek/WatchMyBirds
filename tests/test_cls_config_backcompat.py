"""Regression guard: classifier decision layer stays backward-compatible.

Older pinned classifiers (pre-20260423_062443) ship no
``_model_config.yaml`` or a YAML without a ``detection`` section. The
contract this test locks in:

- ``load_cls_decision_config`` returns ``None`` for every legacy
  scenario (missing file, missing section, missing keys, malformed
  YAML).
- ``decide_label`` with ``config=None`` always returns a species-level
  top-1 result (no threshold gating, no genus fallback, no reject).

If this test breaks, back-compat has silently slipped. Do not "fix" it
by relaxing the expectations — fix the runtime so older classifiers
keep producing species labels the way they always have.
"""

import os
import tempfile

import numpy as np
import yaml

from detectors.cls_config import (
    ClsDecisionConfig,
    decide_label,
    load_cls_decision_config,
)

LEGACY_CLASSES = ["Parus_major", "Parus_caeruleus", "Turdus_merula"]


class TestLoaderReturnsNoneForLegacyModels:
    """The loader must never raise for legacy models, only return None."""

    def test_empty_model_id(self):
        assert load_cls_decision_config("/tmp", "") is None
        assert load_cls_decision_config("/tmp", None) is None

    def test_yaml_not_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            assert load_cls_decision_config(d, "nonexistent_variant") is None

    def test_yaml_without_detection_section(self):
        """Older YAMLs may carry only training metadata — must be accepted silently."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "legacy_model_config.yaml")
            with open(path, "w") as f:
                yaml.dump({"model": {"architecture": "ResNet-50"}}, f)
            assert load_cls_decision_config(d, "legacy") is None

    def test_yaml_with_partial_detection_section(self):
        """Missing threshold or genus_map must fall back to legacy path."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "legacy_model_config.yaml")
            with open(path, "w") as f:
                yaml.dump({"detection": {"confidence_threshold": 0.88}}, f)
            assert load_cls_decision_config(d, "legacy") is None

    def test_malformed_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "legacy_model_config.yaml")
            with open(path, "w") as f:
                f.write("this: is: not: valid: [[")
            assert load_cls_decision_config(d, "legacy") is None


class TestDecideLabelLegacyPath:
    """With config=None, the decision must always be top-1 species."""

    def test_high_confidence_top1(self):
        probs = np.array([0.1, 0.2, 0.7])
        r = decide_label(probs, LEGACY_CLASSES, None)
        assert r["label"] == "Turdus_merula"
        assert r["level"] == "species"
        assert r["prob"] == 0.7

    def test_low_confidence_top1_still_species(self):
        """Critical: legacy path has no threshold, so 0.35 still wins."""
        probs = np.array([0.30, 0.35, 0.35])
        r = decide_label(probs, LEGACY_CLASSES, None)
        assert r["label"] != ""
        assert r["level"] == "species"

    def test_flat_distribution_still_returns_a_label(self):
        """Even with maximum ambiguity, legacy returns top-1, not reject."""
        probs = np.array([0.33, 0.33, 0.34])
        r = decide_label(probs, LEGACY_CLASSES, None)
        assert r["label"] == "Turdus_merula"
        assert r["level"] == "species"


class TestDecideLabelConfiguredPath:
    """With config present, species/genus/reject logic applies."""

    def _cfg(self) -> ClsDecisionConfig:
        return ClsDecisionConfig(
            species_threshold=0.88,
            genus_threshold=0.55,
            genus_map={
                "Parus_major": "Parus",
                "Parus_caeruleus": "Parus",
                "Turdus_merula": "Turdus",
            },
            genus_pairs=frozenset(["Parus"]),
        )

    def test_species_above_threshold(self):
        r = decide_label(np.array([0.92, 0.05, 0.03]), LEGACY_CLASSES, self._cfg())
        assert r["label"] == "Parus_major"
        assert r["level"] == "species"

    def test_genus_fallback_when_siblings_sum_above_genus_threshold(self):
        r = decide_label(np.array([0.40, 0.25, 0.35]), LEGACY_CLASSES, self._cfg())
        assert r["label"] == "Parus_sp."
        assert r["level"] == "genus"
        assert abs(r["prob"] - 0.65) < 1e-6
        assert r["raw_species"] == "Parus_major"

    def test_reject_when_siblings_sum_below_genus_threshold(self):
        r = decide_label(np.array([0.50, 0.04, 0.46]), LEGACY_CLASSES, self._cfg())
        assert r["label"] == ""
        assert r["level"] == "reject"

    def test_reject_when_top1_genus_has_no_siblings(self):
        """Turdus is not in genus_pairs (only one species): must reject cleanly."""
        r = decide_label(np.array([0.10, 0.10, 0.80]), LEGACY_CLASSES, self._cfg())
        assert r["label"] == ""
        assert r["level"] == "reject"
        assert r["raw_species"] == "Turdus_merula"


def test_canonical_yaml_layout_loads():
    """The layout shipped from 20260423_062443 onward.

    Thresholds live under ``detection``. The genus_map is TOP-LEVEL
    (not nested under detection). The genus_pairs list lives under
    ``threshold_contract``.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "20260423_062443_model_config.yaml")
        with open(path, "w") as f:
            yaml.dump(
                {
                    "detection": {
                        "confidence_threshold": 0.88,
                        "genus_fallback_threshold": 0.55,
                        "input_size": 288,
                        "architecture": "EfficientNet-B2",
                    },
                    "calibration": {
                        "temperature": {
                            "value": 0.9936250150191812,
                            "enabled": True,
                        }
                    },
                    "genus_map": {
                        "Parus_major": "Parus",
                        "Parus_caeruleus": "Parus",
                        "Sylvia_atricapilla": "Sylvia",
                        "Sylvia_borin": "Sylvia",
                        "Columba_palumbus": "Columba",
                    },
                    "threshold_contract": {
                        "genus_pairs": ["Parus", "Sylvia"],
                    },
                },
                f,
            )
        cfg = load_cls_decision_config(d, "20260423_062443")
        assert cfg is not None
        assert cfg.species_threshold == 0.88
        assert cfg.genus_threshold == 0.55
        assert cfg.temperature == 0.9936250150191812
        assert cfg.genus_map["Parus_major"] == "Parus"
        assert cfg.genus_map["Sylvia_borin"] == "Sylvia"
        assert "Parus" in cfg.genus_pairs
        assert "Sylvia" in cfg.genus_pairs
        # Columba is in genus_map but NOT in genus_pairs -> no fallback
        assert "Columba" not in cfg.genus_pairs


def test_legacy_nested_yaml_layout_also_loads():
    """Robustness: earlier-style YAML with genus_map/genus_pairs nested
    under ``detection`` is still accepted. Prevents a future spec
    tightening from silently breaking already-deployed variants."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "variant_model_config.yaml")
        with open(path, "w") as f:
            yaml.dump(
                {
                    "detection": {
                        "confidence_threshold": 0.88,
                        "genus_fallback_threshold": 0.55,
                        "genus_map": {"Parus_major": "Parus"},
                        "genus_pairs": ["Parus"],
                    }
                },
                f,
            )
        cfg = load_cls_decision_config(d, "variant")
        assert cfg is not None
        assert "Parus" in cfg.genus_pairs
        assert cfg.temperature == 1.0


def test_disabled_temperature_falls_back_to_identity():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "variant_model_config.yaml")
        with open(path, "w") as f:
            yaml.dump(
                {
                    "detection": {
                        "confidence_threshold": 0.88,
                        "genus_fallback_threshold": 0.55,
                        "genus_map": {"Parus_major": "Parus"},
                        "genus_pairs": ["Parus"],
                    },
                    "calibration": {
                        "temperature": {
                            "value": 0.5,
                            "enabled": False,
                        }
                    },
                },
                f,
            )
        cfg = load_cls_decision_config(d, "variant")
        assert cfg is not None
        assert cfg.temperature == 1.0


def test_columba_single_species_rejects_not_fallbacks():
    """genus_pairs is the shortcut: a species whose genus has only one
    entry in the classes list must REJECT when below species threshold,
    not emit ``Columba_sp.``."""
    cfg = ClsDecisionConfig(
        species_threshold=0.88,
        genus_threshold=0.55,
        genus_map={
            "Parus_major": "Parus",
            "Parus_caeruleus": "Parus",
            "Columba_palumbus": "Columba",
        },
        genus_pairs=frozenset(["Parus"]),  # Columba NOT included
    )
    classes = ["Parus_major", "Parus_caeruleus", "Columba_palumbus"]
    # Columba_palumbus at 0.70 is below species threshold 0.88.
    # Even though its sibling-sum (itself alone) is 0.70 >= 0.55,
    # the shortcut must reject because Columba is not in genus_pairs.
    r = decide_label(np.array([0.15, 0.15, 0.70]), classes, cfg)
    assert r["label"] == ""
    assert r["level"] == "reject"
    assert r["raw_species"] == "Columba_palumbus"
