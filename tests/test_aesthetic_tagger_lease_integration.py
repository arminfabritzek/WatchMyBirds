"""Aesthetic tagger ↔ compute lease integration.

Verifies that the tagger acquires the lease around its worker call,
that ``pause_detection=False`` keeps detection running during a tagger
run, and that a Companion-held lease causes the tagger to skip with a
non-zero exit code instead of overlapping.
"""

from __future__ import annotations

from unittest.mock import patch

from web.services import aesthetic_tag_scheduler as ats
from web.services.compute_lease_service import (
    ComputeLeaseService,
    init_compute_lease_service,
    reset_compute_lease_service_for_testing,
)


class _DM:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused


def test_tagger_acquires_lease_without_pausing_detection(monkeypatch):
    reset_compute_lease_service_for_testing()
    dm = _DM(paused=False)
    init_compute_lease_service(dm)

    seen = {"paused_during_run": None, "lease_holder_during_run": None}

    def fake_main_with_args(argv):
        # While the worker runs, the lease should be held by the tagger
        # AND the detection manager should NOT be paused.
        from web.services.compute_lease_service import get_compute_lease_service

        lease = get_compute_lease_service()
        assert lease is not None
        seen["paused_during_run"] = dm.paused
        seen["lease_holder_during_run"] = lease.status().holder
        return 0

    with patch.object(
        ats, "_invoke_tagger", lambda reason, _fn, argv: fake_main_with_args(argv)
    ):
        # Patch the import inside _run_tagger so it gets a stub.
        import sys
        import types

        stub = types.ModuleType("scripts.aesthetic_tag_nightly")
        stub.main_with_args = fake_main_with_args
        sys.modules["scripts.aesthetic_tag_nightly"] = stub

        rc = ats._run_tagger("test", since=None, throttle_ms=None)

    assert rc == 0
    assert seen["paused_during_run"] is False
    assert seen["lease_holder_during_run"] == "aesthetic_tagger"
    # Lease released after the run.
    from web.services.compute_lease_service import get_compute_lease_service

    lease = get_compute_lease_service()
    assert lease is not None
    assert lease.status().holder is None
    reset_compute_lease_service_for_testing()


def test_tagger_skips_when_lease_busy_with_companion(monkeypatch):
    reset_compute_lease_service_for_testing()
    dm = _DM(paused=False)
    lease = init_compute_lease_service(dm)

    def fake_main_with_args(argv):  # pragma: no cover - must not be called
        raise AssertionError("worker must not run while lease is busy")

    import sys
    import types

    stub = types.ModuleType("scripts.aesthetic_tag_nightly")
    stub.main_with_args = fake_main_with_args
    sys.modules["scripts.aesthetic_tag_nightly"] = stub

    with lease.acquire("companion_inference", pause_detection=True):
        rc = ats._run_tagger("test_blocked", since=None, throttle_ms=None)

    assert rc == 1
    # Companion's lease released cleanly afterwards.
    assert lease.status().holder is None
    reset_compute_lease_service_for_testing()


def test_tagger_runs_without_lease_when_uninitialised(monkeypatch):
    """Slim test harness path: scheduler must still work without WMB Flask host."""
    reset_compute_lease_service_for_testing()
    seen = {"called": False}

    def fake_main_with_args(argv):
        seen["called"] = True
        return 0

    import sys
    import types

    stub = types.ModuleType("scripts.aesthetic_tag_nightly")
    stub.main_with_args = fake_main_with_args
    sys.modules["scripts.aesthetic_tag_nightly"] = stub

    rc = ats._run_tagger("test_no_lease", since=None, throttle_ms=None)
    assert rc == 0
    assert seen["called"] is True
    reset_compute_lease_service_for_testing()


def test_main_initialises_lease_before_starting_aesthetic_scheduler():
    """Boot-order regression: the compute lease must be initialised in
    main._create_runtime BEFORE start_aesthetic_tag_scheduler() runs,
    so the bridge run that can fire seconds after boot acquires the
    lease rather than falling back to the unguarded direct call.

    The first deployment to the Pi exposed this bug: the Aesthetic
    pre-telegram bridge fired at boot, before create_web_interface()
    had a chance to call init_compute_lease_service. The tagger took
    the slim-mode fallback (lease is None -> direct invoke) and
    therefore had no protection against a parallel Companion call.
    """
    import inspect

    import main  # noqa: WPS433 — local import keeps test fast

    src = inspect.getsource(main._create_runtime)
    lease_idx = src.find("init_compute_lease_service(")
    tagger_idx = src.find("start_aesthetic_tag_scheduler(")
    assert lease_idx != -1, "main._create_runtime must call init_compute_lease_service"
    assert tagger_idx != -1, (
        "main._create_runtime must call start_aesthetic_tag_scheduler"
    )
    assert lease_idx < tagger_idx, (
        "init_compute_lease_service must be called before "
        "start_aesthetic_tag_scheduler so the tagger acquires the lease"
    )
