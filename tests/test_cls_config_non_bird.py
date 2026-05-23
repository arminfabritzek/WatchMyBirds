"""Tests for the non_bird_classes drop path (cls_v20+).

Covers the cls_v20 (2026-05-22) bundle's new ``non_bird_classes``
contract:

- The YAML's top-level ``non_bird_classes`` list parses into
  ``ClsDecisionConfig.non_bird_classes`` as a frozenset of class names.
- ``decide_label`` returns ``level='reject'`` immediately when top-1
  is in ``non_bird_classes``, BEFORE any threshold check, BEFORE the
  genus-fallback path.
- Bundles without the field, with an empty list, or with malformed
  entries produce an empty frozenset — bird-only classifiers behave
  byte-identical to pre-v20.
"""

import numpy as np

from detectors.cls_config import (
    ClsDecisionConfig,
    _load_non_bird_classes,
    decide_label,
)

CLASSES = ["Parus_major", "Cyanistes_caeruleus", "Sylvia_borin", "non_bird"]
GENUS_MAP = {
    "Parus_major": "Parus",
    "Cyanistes_caeruleus": "Cyanistes",
    "Sylvia_borin": "Sylvia",
}


def _make_config(
    non_bird_classes: frozenset[str] = frozenset(),
    gallery_threshold: float | None = 0.70,
    review_threshold: float | None = 0.25,
) -> ClsDecisionConfig:
    return ClsDecisionConfig(
        species_threshold=0.98,
        genus_threshold=0.55,
        genus_map=GENUS_MAP,
        genus_pairs=frozenset({"Sylvia"}),
        gallery_threshold=gallery_threshold,
        review_threshold=review_threshold,
        non_bird_classes=non_bird_classes,
    )


# --- _load_non_bird_classes -----------------------------------------


def test_loader_reads_list_into_frozenset() -> None:
    raw = {"non_bird_classes": ["non_bird"]}
    assert _load_non_bird_classes(raw, "test.yaml") == frozenset({"non_bird"})


def test_loader_missing_field_returns_empty() -> None:
    assert _load_non_bird_classes({}, "test.yaml") == frozenset()


def test_loader_empty_list_returns_empty() -> None:
    assert _load_non_bird_classes({"non_bird_classes": []}, "test.yaml") == frozenset()


def test_loader_non_list_returns_empty_with_warning(caplog) -> None:
    result = _load_non_bird_classes({"non_bird_classes": "non_bird"}, "test.yaml")
    assert result == frozenset()
    assert any("is not a list" in r.message for r in caplog.records)


def test_loader_skips_non_string_entries(caplog) -> None:
    raw = {"non_bird_classes": ["non_bird", 42, "", "   ", "other_non_bird"]}
    result = _load_non_bird_classes(raw, "test.yaml")
    assert result == frozenset({"non_bird", "other_non_bird"})
    # The 42, "", "   " all warn — exact count not asserted, but at least one fires.
    assert any("non-empty string" in r.message for r in caplog.records)


def test_loader_strips_whitespace() -> None:
    raw = {"non_bird_classes": ["  non_bird  "]}
    assert _load_non_bird_classes(raw, "test.yaml") == frozenset({"non_bird"})


# --- decide_label: non_bird drop --------------------------------------


def test_non_bird_top1_returns_reject_above_gallery_threshold() -> None:
    """Even a high-confidence non_bird prediction must drop, not
    promote to species/gallery."""
    config = _make_config(non_bird_classes=frozenset({"non_bird"}))
    # non_bird at 0.95 — far above gallery_threshold=0.70
    probs = np.array([0.01, 0.02, 0.02, 0.95])
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"
    assert result["label"] == ""
    assert result["raw_species"] == "non_bird"


def test_non_bird_top1_returns_reject_in_review_band() -> None:
    """non_bird in the review band (between review_thr and gallery_thr)
    must drop, not enter the species_review bucket."""
    config = _make_config(non_bird_classes=frozenset({"non_bird"}))
    probs = np.array([0.20, 0.20, 0.20, 0.40])  # non_bird=0.40 in review band
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"
    assert result["raw_species"] == "non_bird"


def test_non_bird_drop_bypasses_genus_fallback() -> None:
    """If top-1 were Sylvia_borin with sibling Sylvia mass crossing
    genus_threshold, we'd normally rescue to Sylvia_sp. The non_bird
    branch must never reach that logic — but here we test the symmetric
    case: non_bird wins, no genus_map entry, ensure no rescue attempt
    crashes us."""
    config = _make_config(non_bird_classes=frozenset({"non_bird"}))
    probs = np.array([0.05, 0.05, 0.05, 0.85])
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"
    # raw_species is the literal class, which intentionally has no
    # genus_map entry; the drop branch must not consult genus_map.
    assert result["raw_species"] == "non_bird"


def test_bird_top1_unaffected_by_non_bird_classes() -> None:
    """A bird top-1 with non_bird_classes configured behaves identically
    to a bird-only bundle."""
    config = _make_config(non_bird_classes=frozenset({"non_bird"}))
    probs = np.array([0.85, 0.05, 0.05, 0.05])  # Parus_major wins
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "species"
    assert result["label"] == "Parus_major"


def test_empty_non_bird_classes_legacy_path() -> None:
    """Empty non_bird_classes set = legacy bird-only behaviour. Even if
    a class literally named 'non_bird' exists in CLASSES, without the
    drop-set configured it would be treated as a normal species."""
    config = _make_config(non_bird_classes=frozenset())
    probs = np.array([0.05, 0.05, 0.05, 0.85])
    result = decide_label(probs, CLASSES, config)
    # With empty non_bird_classes the non_bird wins normally; result is
    # NOT 'reject' from the non_bird gate. (It may still reject for
    # other reasons — here above gallery_threshold so 'species'.)
    assert result["level"] == "species"
    assert result["label"] == "non_bird"


def test_non_bird_drop_runs_before_threshold_check() -> None:
    """The drop must fire even when probability is very low — it is a
    class-level decision, not a confidence decision."""
    config = _make_config(non_bird_classes=frozenset({"non_bird"}))
    # non_bird wins but only at 0.30 — well below any threshold
    probs = np.array([0.25, 0.25, 0.20, 0.30])
    result = decide_label(probs, CLASSES, config)
    assert result["level"] == "reject"


def test_default_non_bird_classes_is_empty_frozenset() -> None:
    """Bundles constructed without the field default to an empty
    frozenset — back-compat for any test/caller that does not pass it."""
    config = ClsDecisionConfig(
        species_threshold=0.98,
        genus_threshold=0.55,
        genus_map=GENUS_MAP,
        genus_pairs=frozenset(),
    )
    assert config.non_bird_classes == frozenset()
