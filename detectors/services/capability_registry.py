"""
Capability Registry — tracks active model/service versions and feature flags.

Provides a single source of truth for which detection/classification
capabilities are currently enabled, and their version strings.
The active capability snapshot is persisted on each detection.

All capabilities default to ``enabled=False`` until explicitly registered.
Feature flags read from config: ``ENABLE_<CAPABILITY_NAME>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import get_config

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """Immutable descriptor for a single capability."""

    name: str
    version: str
    enabled: bool = True


@dataclass
class CapabilitySnapshot:
    """Point-in-time view of all registered capabilities."""

    capabilities: dict[str, Capability] = field(default_factory=dict)

    def enabled_names(self) -> list[str]:
        """Return sorted list of enabled capability names."""
        return sorted(c.name for c in self.capabilities.values() if c.enabled)

    def version_tag(self) -> str:
        """
        Compact version tag for persistence.

        Example: ``"detector_v8+cls_v2+ood_v1+temporal_v1"``
        """
        parts = [
            f"{c.name}_{c.version}"
            for c in sorted(self.capabilities.values(), key=lambda c: c.name)
            if c.enabled
        ]
        return "+".join(parts) if parts else "none"

    def to_dict(self) -> dict:
        """Serialise for API/JSON."""
        return {
            c.name: {"version": c.version, "enabled": c.enabled}
            for c in self.capabilities.values()
        }


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------


class CapabilityRegistry:
    """
    Central registry for detection pipeline capabilities.

    Register capabilities at startup; query the snapshot per detection
    for persistence and API exposure.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or get_config()
        self._capabilities: dict[str, Capability] = {}

    def register(
        self,
        name: str,
        version: str,
        *,
        flag_key: str | None = None,
    ) -> None:
        """
        Register (or update) a capability.

        Args:
            name:     Short identifier, e.g. ``"detector"``, ``"ood"``.
            version:  Version string, e.g. ``"v8"``, ``"v1"``.
            flag_key: Optional config key to read enabled state from.
                      Defaults to ``ENABLE_<NAME>``.
        """
        key = flag_key or f"ENABLE_{name.upper()}"
        raw = str(self._config.get(key, "true")).lower()
        enabled = raw in ("1", "true", "yes")
        self._capabilities[name] = Capability(
            name=name, version=version, enabled=enabled
        )

    def is_enabled(self, name: str) -> bool:
        """Check if a named capability is registered and enabled."""
        cap = self._capabilities.get(name)
        return cap is not None and cap.enabled

    def get(self, name: str) -> Capability | None:
        """Return a registered capability or None."""
        return self._capabilities.get(name)

    def snapshot(self) -> CapabilitySnapshot:
        """Return a frozen copy of the current registry state."""
        return CapabilitySnapshot(capabilities=dict(self._capabilities))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_registry(config: dict | None = None) -> CapabilityRegistry:
    """
    Pre-populate a registry with the standard WatchMyBirds capabilities.

    Call at startup before the first detection loop iteration.
    """
    from detectors.services.decision_policy_service import POLICY_VERSION
    from detectors.services.temporal_decision_service import TEMPORAL_VERSION

    reg = CapabilityRegistry(config=config)
    reg.register("detector", "v8")  # YOLOv8 object detection
    reg.register("classifier", "v2")  # ONNX species classifier
    reg.register("bbox_quality", "v1")  # Geometric bbox quality
    reg.register("ood", "v1")  # Unknown/OOD score (margin+entropy)
    reg.register(
        "temporal",
        TEMPORAL_VERSION,
        flag_key="ENABLE_TEMPORAL_SMOOTHING",
    )
    reg.register(
        "decision_policy",
        POLICY_VERSION,
        flag_key="ENABLE_DECISION_POLICY",
    )
    return reg
