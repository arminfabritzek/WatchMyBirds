"""
Temporal Decision Service — flicker reduction via sliding window.

Smooths rapid decision state changes by maintaining a fixed-size window
of recent per-frame decision states. The final (smoothed) state is the
majority vote across the window.

Feature-flag: ``ENABLE_TEMPORAL_SMOOTHING`` (default: False).
When disabled, ``smooth()`` returns the raw state unchanged.
"""

from __future__ import annotations

from collections import Counter, deque

from config import get_config
from detectors.interfaces.classification import DecisionState

# Service version tag persisted alongside detections
TEMPORAL_VERSION = "temporal_v1"

# Default window size (frames)
_DEFAULT_WINDOW = 5


class TemporalDecisionService:
    """Sliding-window majority-vote smoother for decision states."""

    def __init__(
        self,
        config: dict | None = None,
        window_size: int | None = None,
    ) -> None:
        cfg = config or get_config()
        self.enabled: bool = str(
            cfg.get("ENABLE_TEMPORAL_SMOOTHING", "false")
        ).lower() in ("1", "true", "yes")
        self.window_size: int = window_size or int(
            cfg.get("TEMPORAL_WINDOW_SIZE", _DEFAULT_WINDOW)
        )
        # Keyed by (species_name) → deque of recent DecisionStates
        self._windows: dict[str, deque[DecisionState]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smooth(
        self,
        species_key: str,
        raw_state: DecisionState,
    ) -> DecisionState:
        """
        Smooth a raw decision state using the sliding window.

        Args:
            species_key: Grouping key (e.g. species name or tracking ID).
            raw_state:   The instantaneous decision state for this frame.

        Returns:
            The smoothed decision state (majority vote), or *raw_state*
            if temporal smoothing is disabled.
        """
        if not self.enabled:
            return raw_state

        window = self._windows.setdefault(
            species_key,
            deque(maxlen=self.window_size),
        )
        window.append(raw_state)

        # Majority vote
        counts = Counter(window)
        smoothed, _count = counts.most_common(1)[0]
        return smoothed

    def reset(self, species_key: str | None = None) -> None:
        """Clear window(s). Pass ``None`` to clear all."""
        if species_key is None:
            self._windows.clear()
        else:
            self._windows.pop(species_key, None)
