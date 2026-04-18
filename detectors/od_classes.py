"""Object-detection class taxonomy for the locator.

Defines which OD class names represent birds (go through the CLS pipeline)
vs. non-bird garden animals (OD class name IS the species, CLS is skipped).

The current YOLOX-S locator has five classes:

- "bird"             -> CLS assigns species
- "squirrel"         -> species = "squirrel"
- "cat"              -> species = "cat"
- "marten_mustelid"  -> species = "marten_mustelid"
- "hedgehog"         -> species = "hedgehog"

Any future locator must still emit exactly one class named "bird" for the
CLS-bound path; the Model-Compatibility-Guard in
``detectors.detector._assert_yolox_labels_compatible`` enforces that.
"""

from __future__ import annotations

# Class names that should be routed through the bird classifier.
# Kept as a code constant (not a config knob) so that the bird/non-bird
# routing stays in lockstep with the detector's class set.
BIRD_OD_CLASSES: frozenset[str] = frozenset({"bird"})


def is_bird_od_class(class_name: str | None) -> bool:
    """Return True when the OD class name should be sent to the CLS pipeline."""
    if not class_name:
        return False
    return class_name in BIRD_OD_CLASSES
