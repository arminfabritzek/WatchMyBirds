"""Retention V2 — explicit posture (off / conservative / reclaim).

The posture is the new operator-facing authority. It resolves to the V1
boolean knobs so the pure Policy and the Planner/Executor mechanics stay
unchanged:

- off          -> retention disabled (P0 protects "disabled")
- conservative -> enabled, unreviewed PROTECTED (== V1 behaviour)
- reclaim      -> enabled, unreviewed NOT a protection reason; everything
                  else (export-relevant, favorites, missing-derivative,
                  too-recent) still protects.

Backcompat: RETENTION_ENABLED=false -> off; RETENTION_ENABLED=true without a
posture -> conservative.
"""

from core.retention_core import decide, resolve_posture_settings

# An original old enough / present / durable / not favourite / not export-
# relevant, but UNREVIEWED — the population that dominates the live library.
UNREVIEWED_FACTS = {
    "age_days": 120,
    "original_present": 1,
    "derivatives_present": True,
    "is_favorite": False,
    "export_relevant": False,
    "review_status": "untagged",
}


def _settings(posture: str) -> dict:
    return resolve_posture_settings(
        {
            "RETENTION_POSTURE": posture,
            "RETENTION_DAYS": 90,
            "RETENTION_PROTECT_FAVORITES": True,
        }
    )


# --- resolver -------------------------------------------------------------


def test_resolve_off_disables_retention():
    s = _settings("off")
    assert s["RETENTION_ENABLED"] is False


def test_resolve_conservative_enables_and_protects_unreviewed():
    s = _settings("conservative")
    assert s["RETENTION_ENABLED"] is True
    assert s["RETENTION_PROTECT_UNREVIEWED"] is True


def test_resolve_reclaim_enables_and_drops_unreviewed_protection():
    s = _settings("reclaim")
    assert s["RETENTION_ENABLED"] is True
    assert s["RETENTION_PROTECT_UNREVIEWED"] is False


# --- backcompat -----------------------------------------------------------


def test_backcompat_enabled_false_maps_to_off():
    s = resolve_posture_settings({"RETENTION_ENABLED": False, "RETENTION_DAYS": 90})
    assert s["RETENTION_ENABLED"] is False


def test_backcompat_enabled_true_without_posture_maps_to_conservative():
    s = resolve_posture_settings({"RETENTION_ENABLED": True, "RETENTION_DAYS": 90})
    assert s["RETENTION_ENABLED"] is True
    assert s["RETENTION_PROTECT_UNREVIEWED"] is True


def test_explicit_posture_overrides_legacy_enabled_flag():
    # An explicit posture wins over a stale RETENTION_ENABLED value.
    s = resolve_posture_settings(
        {
            "RETENTION_POSTURE": "reclaim",
            "RETENTION_ENABLED": False,
            "RETENTION_DAYS": 90,
        }
    )
    assert s["RETENTION_ENABLED"] is True
    assert s["RETENTION_PROTECT_UNREVIEWED"] is False


def test_unknown_posture_falls_back_to_conservative():
    s = resolve_posture_settings({"RETENTION_POSTURE": "bogus", "RETENTION_DAYS": 90})
    assert s["RETENTION_ENABLED"] is True
    assert s["RETENTION_PROTECT_UNREVIEWED"] is True


# --- end-to-end through decide() -----------------------------------------


def test_conservative_protects_unreviewed():
    action, reason = decide(UNREVIEWED_FACTS, _settings("conservative"))
    assert action == "protect"
    assert reason == "unreviewed"


def test_reclaim_retires_unreviewed():
    action, reason = decide(UNREVIEWED_FACTS, _settings("reclaim"))
    assert action == "delete"
    assert reason is None


def test_reclaim_still_protects_export_relevant():
    facts = {**UNREVIEWED_FACTS, "export_relevant": True}
    action, reason = decide(facts, _settings("reclaim"))
    assert action == "protect"
    assert reason == "export_relevant"


def test_reclaim_still_protects_favorites():
    facts = {**UNREVIEWED_FACTS, "is_favorite": True}
    action, reason = decide(facts, _settings("reclaim"))
    assert action == "protect"
    assert reason == "favorite"


def test_reclaim_still_protects_missing_derivative():
    facts = {**UNREVIEWED_FACTS, "derivatives_present": False}
    action, reason = decide(facts, _settings("reclaim"))
    assert action == "protect"
    assert reason == "missing_derivative"


def test_reclaim_still_protects_too_recent():
    facts = {**UNREVIEWED_FACTS, "age_days": 30}
    action, reason = decide(facts, _settings("reclaim"))
    assert action == "protect"
    assert reason == "too_recent"


def test_off_protects_everything():
    action, reason = decide(UNREVIEWED_FACTS, _settings("off"))
    assert action == "protect"
    assert reason == "disabled"
