"""Tests for Capability Registry (P2-02)."""

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
        "ENABLE_TEMPORAL_SMOOTHING": "true",
        "ENABLE_DECISION_POLICY": "true",
    }
    return build_default_registry(config=config)


@pytest.fixture
def registry_some_disabled():
    """Registry with temporal and OOD disabled."""
    config = {
        "ENABLE_DETECTOR": "true",
        "ENABLE_CLASSIFIER": "true",
        "ENABLE_BBOX_QUALITY": "true",
        "ENABLE_OOD": "false",
        "ENABLE_TEMPORAL_SMOOTHING": "false",
        "ENABLE_DECISION_POLICY": "true",
    }
    return build_default_registry(config=config)


def test_enabled_capabilities_listed(registry_all_enabled):
    """All capabilities should appear in enabled_names when config says true."""
    snap = registry_all_enabled.snapshot()
    names = snap.enabled_names()

    assert "detector" in names
    assert "classifier" in names
    assert "temporal" in names
    assert "ood" in names
    assert "decision_policy" in names


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
