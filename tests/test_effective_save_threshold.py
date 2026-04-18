"""Tests for config.effective_save_threshold().

The detector's confidence floor is now model-owned. The save threshold
(image-persistence policy) has its own "auto" and "manual" modes:

- "auto" (default): save = detector.conf_threshold_default + 0.10
  so the save gate is always 0.10 above whatever the active model
  considers a valid detection.
- "manual": honour the user's SAVE_THRESHOLD value unchanged.

Regression guards for the 2026-04-18 settings rewrite.
"""

from __future__ import annotations

import pytest

from config import SAVE_THRESHOLD_AUTO_OFFSET, effective_save_threshold


def test_auto_mode_derives_from_detector_default():
    cfg = {"SAVE_THRESHOLD_MODE": "auto", "SAVE_THRESHOLD": 0.65}
    assert effective_save_threshold(cfg, 0.15) == pytest.approx(0.25)
    assert effective_save_threshold(cfg, 0.30) == pytest.approx(0.40)


def test_auto_mode_uses_locked_offset_constant():
    """The 0.10 offset is stable across releases — do not silently shift."""
    assert SAVE_THRESHOLD_AUTO_OFFSET == 0.10


def test_auto_mode_falls_back_to_manual_value_when_detector_missing():
    """During startup the detector may not be ready yet. Fall back to
    the last persisted manual value so the detection loop can still
    apply *something* reasonable rather than 0."""
    cfg = {"SAVE_THRESHOLD_MODE": "auto", "SAVE_THRESHOLD": 0.55}
    assert effective_save_threshold(cfg, None) == 0.55


def test_manual_mode_honours_user_value():
    cfg = {"SAVE_THRESHOLD_MODE": "manual", "SAVE_THRESHOLD": 0.72}
    # Even with a detector available, manual wins.
    assert effective_save_threshold(cfg, 0.15) == 0.72
    # And with no detector, same value.
    assert effective_save_threshold(cfg, None) == 0.72


def test_manual_mode_respects_low_value_even_below_detector_floor():
    """If an operator deliberately sets SAVE_THRESHOLD below the model's
    detection floor (policy: keep everything the detector returned),
    we must honour it rather than silently raising it."""
    cfg = {"SAVE_THRESHOLD_MODE": "manual", "SAVE_THRESHOLD": 0.05}
    assert effective_save_threshold(cfg, 0.30) == 0.05


def test_auto_mode_clips_to_upper_bound():
    """Hypothetical future model with a very high conf_default should
    not produce save > 1.0."""
    cfg = {"SAVE_THRESHOLD_MODE": "auto", "SAVE_THRESHOLD": 0.65}
    assert effective_save_threshold(cfg, 0.95) == 1.0


def test_unknown_mode_falls_through_to_auto():
    """A typo in SAVE_THRESHOLD_MODE must not block the pipeline.
    Fall through to auto so save-threshold keeps tracking the model."""
    cfg = {"SAVE_THRESHOLD_MODE": "brokenstring", "SAVE_THRESHOLD": 0.65}
    assert effective_save_threshold(cfg, 0.15) == pytest.approx(0.25)


def test_default_mode_is_auto_when_key_missing():
    """Legacy configs without SAVE_THRESHOLD_MODE default to auto."""
    cfg = {"SAVE_THRESHOLD": 0.65}
    assert effective_save_threshold(cfg, 0.15) == pytest.approx(0.25)


def test_mode_comparison_is_case_insensitive():
    for mode in ("AUTO", "Auto", "manual", "MANUAL", "  auto  "):
        cfg = {"SAVE_THRESHOLD_MODE": mode, "SAVE_THRESHOLD": 0.5}
        result = effective_save_threshold(cfg, 0.20)
        # auto -> 0.30, manual -> 0.5
        assert result == pytest.approx(0.30) or result == pytest.approx(0.5), (
            f"mode={mode!r} gave {result}"
        )
