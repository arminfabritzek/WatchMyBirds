"""Tests for Capability Registry (P2-02)."""

from unittest.mock import MagicMock

import pytest

from detectors.services.capability_registry import (
    CapabilityRegistry,
    CapabilitySnapshot,
    build_default_registry,
)


@pytest.fixture
def registry_all_enabled():
    """Registry where all standard capabilities are enabled."""
    config = {
        "ENABLE_DETECTOR": "true",
        "ENABLE_CLASSIFIER": "true",
        "ENABLE_BBOX_QUALITY": "true",
        "ENABLE_OOD": "true",
    }
    return build_default_registry(config=config)


@pytest.fixture
def registry_some_disabled():
    """Registry with OOD disabled."""
    config = {
        "ENABLE_DETECTOR": "true",
        "ENABLE_CLASSIFIER": "true",
        "ENABLE_BBOX_QUALITY": "true",
        "ENABLE_OOD": "false",
    }
    return build_default_registry(config=config)


def test_enabled_capabilities_listed(registry_all_enabled):
    """All capabilities should appear in enabled_names when config says true."""
    snap = registry_all_enabled.snapshot()
    names = snap.enabled_names()

    assert "detector" in names
    assert "classifier" in names
    assert "temporal" not in snap.to_dict()
    assert "ood" in names
    assert "decision_policy" in names
    assert "nightly_deep_scan" not in snap.to_dict()


def test_disabled_capabilities_excluded(registry_some_disabled):
    """Disabled capabilities should not appear in enabled_names."""
    snap = registry_some_disabled.snapshot()
    names = snap.enabled_names()

    assert "detector" in names
    assert "classifier" in names
    assert "ood" not in names
    assert "temporal" not in names


def test_version_tag_format(registry_all_enabled):
    """Version tag should contain all enabled capabilities with versions."""
    snap = registry_all_enabled.snapshot()
    tag = snap.version_tag()

    assert "detector_v8" in tag
    assert "classifier_v2" in tag
    assert "+" in tag  # Multiple parts joined


def test_version_tag_excludes_disabled(registry_some_disabled):
    """Version tag should not include disabled capabilities."""
    snap = registry_some_disabled.snapshot()
    tag = snap.version_tag()

    assert "temporal" not in tag
    assert "ood" not in tag
    assert "detector_v8" in tag


def test_is_enabled_check(registry_some_disabled):
    """is_enabled should return correct boolean for each capability."""
    assert registry_some_disabled.is_enabled("detector") is True
    assert registry_some_disabled.is_enabled("ood") is False
    assert registry_some_disabled.is_enabled("temporal") is False
    assert registry_some_disabled.is_enabled("nonexistent") is False


def test_to_dict_serialization(registry_all_enabled):
    """Snapshot should serialise to a dictionary suitable for JSON."""
    snap = registry_all_enabled.snapshot()
    d = snap.to_dict()

    assert isinstance(d, dict)
    assert "detector" in d
    assert d["detector"]["version"] == "v8"
    assert d["detector"]["enabled"] is True


def test_register_updates_capability():
    """Registering the same name again should update the entry."""
    reg = CapabilityRegistry(config={"ENABLE_TEST": "true"})
    reg.register("test", "v1")
    assert reg.get("test").version == "v1"

    reg.register("test", "v2")
    assert reg.get("test").version == "v2"


def test_empty_registry_version_tag():
    """Empty registry should produce 'none' version tag."""
    snap = CapabilitySnapshot()
    assert snap.version_tag() == "none"


def test_decision_policy_ignores_disabled_config() -> None:
    from detectors.services.decision_policy_service import POLICY_VERSION

    registry = build_default_registry(
        config={"ENABLE_DECISION_POLICY": "false", "ENABLE_OOD": "false"}
    )

    assert registry.is_enabled("decision_policy")
    assert (
        f"decision_policy_{POLICY_VERSION}"
        in registry.snapshot().version_tag().split("+")
    )
    assert not registry.is_enabled("ood")


@pytest.mark.parametrize("flag_key", [None, "CUSTOM_POLICY_ENABLED"])
def test_decision_policy_registration_does_not_read_flags(flag_key: str | None) -> None:
    config = MagicMock(spec=dict)
    config.get.side_effect = AssertionError(
        "Decision policy must not read config flags"
    )
    registry = CapabilityRegistry(config=config)

    registry.register("decision_policy", "v1", flag_key=flag_key)

    assert registry.is_enabled("decision_policy")
    assert registry.snapshot().version_tag() == "decision_policy_v1"
    config.get.assert_not_called()
