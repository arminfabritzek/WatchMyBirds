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


def test_bridge_run_now_skips_while_companion_chat_in_flight(tmp_path, caplog):
    """Smoke 4 as a unit test: while a Companion chat() call is in
    flight on a worker thread (holding the lease), the pre-Telegram
    bridge entry point run_now() must observe LeaseBusy, return False,
    log the "skipped" line that names the current holder, and never
    invoke the CLIP worker.

    On the RPi this is the failure mode that protects the Pi from
    running OD, CLIP and llama.cpp on the same cores at the same time;
    that protection is what makes Companion-on-a-single-board safe.
    The integration tests above prove the Lease primitive blocks
    correctly — this test proves the full bridge-shaped call path
    (run_now → _run_tagger → lease.acquire) behaves the same way when
    the lease is held by a real CompanionService, not just by a
    direct lease.acquire("companion_inference") in test code.
    """
    import logging
    import threading

    from web.services.companion.inference import (
        CompanionInferenceClient,
        CompanionInferenceResult,
    )
    from web.services.companion.recorder import CompanionRecorder
    from web.services.companion.service import CompanionService

    reset_compute_lease_service_for_testing()
    dm = _DM(paused=False)
    lease = init_compute_lease_service(dm)

    # Two threading.Events coordinate the race deterministically:
    # - in_lease fires when the adapter has been entered (companion
    #   service holds the lease at this point)
    # - release fires when the main thread is done with its bridge
    #   call and the adapter may return
    in_lease = threading.Event()
    release = threading.Event()

    class _SlowProbe(CompanionInferenceClient):
        model_id = "probe:smoke4"

        def generate(self, *, system_prompt, messages, timeout_s):
            # The companion service has the lease at this point.
            in_lease.set()
            # Wait until the main thread has run its bridge call.
            # Timeout matches Companion's worst-case to avoid CI hangs.
            release.wait(timeout=10.0)
            return CompanionInferenceResult(
                status="ok",
                text="ok",
                raw="ok",
                model_id=self.model_id,
            )

    svc = CompanionService(
        client=_SlowProbe(),
        recorder=CompanionRecorder(base_dir=tmp_path),
        lease=lease,
        enabled=True,
        pause_detection=True,
        timeout_s=5.0,
    )

    # Companion chat() runs on a worker thread so we can race against
    # it from the main thread.
    chat_result: dict = {}

    def _chat_thread():
        chat_result["out"] = svc.chat(
            message="hello", language="de", tone="kid_friendly"
        )

    worker = threading.Thread(target=_chat_thread, name="companion-chat-test", daemon=True)
    worker.start()

    # Wait for the worker to enter the lease. If this times out the
    # rest of the test is meaningless, so fail loud rather than
    # silently testing the wrong race.
    entered = in_lease.wait(timeout=5.0)
    assert entered, "Companion did not enter the lease in time"
    # Sanity check: lease is genuinely held by the Companion service.
    status = lease.status()
    assert status.holder == "companion_inference", (
        f"expected companion_inference, got {status.holder!r}"
    )
    assert status.pause_detection is True
    assert dm.paused is True, "Companion must have paused detection"

    # Stub the worker import so a stray call would fail loud rather
    # than launching real CLIP.
    import sys
    import types

    stub = types.ModuleType("scripts.aesthetic_tag_nightly")

    def _must_not_run(argv):  # pragma: no cover
        raise AssertionError(
            "tagger worker must not run while Companion holds the lease"
        )

    stub.main_with_args = _must_not_run
    sys.modules["scripts.aesthetic_tag_nightly"] = stub

    caplog.set_level(logging.INFO, logger="web.services.aesthetic_tag_scheduler")

    # Force a fresh run-mutex so the test does not depend on prior
    # test state — same idiom as the scheduler tests.
    ats._run_mutex = threading.Lock()

    with patch("config.get_config", return_value={
        "AESTHETIC_TAG_ENABLED": True,
        "AESTHETIC_BRIDGE_PER_SPECIES_CAP": 8,
    }), patch.object(ats, "_check_dependencies_available", return_value=True):
        ok = ats.run_now("smoke4 bridge", today_only=True)

    # Bridge call must return False (lease busy).
    assert ok is False, "run_now must return False when Companion holds the lease"

    # The "skipped" log line is the operator-visible signal on the RPi
    # — verify it was emitted and that it names the right holder.
    skip_msgs = [r for r in caplog.records if "skipped" in r.getMessage().lower()]
    assert any(
        "companion_inference" in r.getMessage() for r in skip_msgs
    ), (
        "expected a 'skipped … compute lease busy with companion_inference' "
        f"log line; got {[r.getMessage() for r in caplog.records]!r}"
    )

    # Let Companion finish so we can verify clean teardown.
    release.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "Companion worker did not finish"
    assert chat_result.get("out", {}).get("ok") is True, (
        f"Companion chat should have succeeded; got {chat_result!r}"
    )
    # Lease released cleanly, detection back to its pre-companion state.
    assert lease.status().holder is None
    assert dm.paused is False, "Companion must restore detection state"
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
