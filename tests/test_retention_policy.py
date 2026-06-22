"""Unit tests for the pure retention Policy (core.retention_core.decide).

Policy takes a dict of per-image facts plus the active settings dict and
returns a Decision: ("delete", None) or ("protect", <reason>). It performs
no IO — every fact it needs is passed in.
"""

from core.retention_core import decide

# A baseline set of facts describing an original that SHOULD be deletable:
# old enough, present, derivatives present, not favourite, not export-relevant,
# reviewed.
DELETABLE_FACTS = {
    "age_days": 120,
    "original_present": 1,
    "derivatives_present": True,
    "is_favorite": False,
    "export_relevant": False,
    "review_status": "confirmed_bird",
}

# Settings with the feature enabled and conservative protections on.
ENABLED_SETTINGS = {
    "RETENTION_ENABLED": True,
    "RETENTION_DAYS": 90,
    "RETENTION_PROTECT_FAVORITES": True,
    "RETENTION_PROTECT_UNREVIEWED": True,
}


def test_deletable_when_all_rules_pass():
    action, reason = decide(DELETABLE_FACTS, ENABLED_SETTINGS)
    assert action == "delete"
    assert reason is None


def test_p0_protected_when_retention_disabled():
    settings = {**ENABLED_SETTINGS, "RETENTION_ENABLED": False}
    action, reason = decide(DELETABLE_FACTS, settings)
    assert action == "protect"
    assert reason == "disabled"


def test_p1_protected_when_original_already_deleted():
    facts = {**DELETABLE_FACTS, "original_present": 0}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "already_deleted"


def test_p2_protected_when_too_recent():
    facts = {**DELETABLE_FACTS, "age_days": 30}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "too_recent"


def test_p2_boundary_exactly_window_is_protected():
    # Exactly RETENTION_DAYS old is NOT yet past the window.
    facts = {**DELETABLE_FACTS, "age_days": 90}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "too_recent"


def test_p2_boundary_just_past_window_is_deletable():
    facts = {**DELETABLE_FACTS, "age_days": 91}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "delete"


def test_p3_protected_when_derivatives_missing():
    facts = {**DELETABLE_FACTS, "derivatives_present": False}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "missing_derivative"


def test_p4_protected_when_favorite():
    facts = {**DELETABLE_FACTS, "is_favorite": True}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "favorite"


def test_p4_favorite_deletable_when_protection_off():
    facts = {**DELETABLE_FACTS, "is_favorite": True}
    settings = {**ENABLED_SETTINGS, "RETENTION_PROTECT_FAVORITES": False}
    action, reason = decide(facts, settings)
    assert action == "delete"


def test_p5_protected_when_export_relevant():
    facts = {**DELETABLE_FACTS, "export_relevant": True}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "export_relevant"


def test_p5_export_relevant_always_protected_even_with_all_protections_off():
    # P5 has no off-switch: training data is never silently deletable.
    facts = {**DELETABLE_FACTS, "export_relevant": True}
    settings = {
        **ENABLED_SETTINGS,
        "RETENTION_PROTECT_FAVORITES": False,
        "RETENTION_PROTECT_UNREVIEWED": False,
    }
    action, reason = decide(facts, settings)
    assert action == "protect"
    assert reason == "export_relevant"


def test_p6_protected_when_unreviewed_untagged():
    facts = {**DELETABLE_FACTS, "review_status": "untagged"}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "unreviewed"


def test_p6_protected_when_review_status_null():
    facts = {**DELETABLE_FACTS, "review_status": None}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "protect"
    assert reason == "unreviewed"


def test_p6_unreviewed_deletable_when_protection_off():
    facts = {**DELETABLE_FACTS, "review_status": "untagged"}
    settings = {**ENABLED_SETTINGS, "RETENTION_PROTECT_UNREVIEWED": False}
    action, reason = decide(facts, settings)
    assert action == "delete"


def test_no_bird_reviewed_is_not_unreviewed():
    # 'no_bird' is a decided review state; P6 must not flag it.
    # (It would still be export-relevant in practice, but Policy is pure:
    # export_relevant is passed in explicitly and is False here.)
    facts = {**DELETABLE_FACTS, "review_status": "no_bird"}
    action, reason = decide(facts, ENABLED_SETTINGS)
    assert action == "delete"
