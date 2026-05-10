"""Compute lease unit tests.

Covers the new ``ComputeLeaseService`` in isolation from the rest of
the WMB stack — no DetectionManager init, no Flask app, just a stub
detection_manager double exposing ``paused``.
"""

from __future__ import annotations

import threading
import time

import pytest

from web.services.compute_lease_service import (
    ComputeLeaseService,
    LeaseBusy,
)


class _DM:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused


def test_acquire_pauses_detection_and_restores():
    dm = _DM(paused=False)
    svc = ComputeLeaseService(dm)
    with svc.acquire("companion_inference", pause_detection=True):
        assert dm.paused is True
        s = svc.status()
        assert s.holder == "companion_inference"
        assert s.pause_detection is True
    assert dm.paused is False
    assert svc.status().holder is None


def test_acquire_without_pause_does_not_touch_detection():
    dm = _DM(paused=False)
    svc = ComputeLeaseService(dm)
    with svc.acquire("aesthetic_tagger", pause_detection=False):
        assert dm.paused is False
    assert dm.paused is False


def test_acquire_preserves_already_paused_state():
    dm = _DM(paused=True)
    svc = ComputeLeaseService(dm)
    with svc.acquire("companion_inference", pause_detection=True):
        assert dm.paused is True
    # Was already paused before lease — must remain paused after release.
    assert dm.paused is True


def test_busy_holder_blocks_other_callers():
    dm = _DM()
    svc = ComputeLeaseService(dm)
    with svc.acquire("companion_inference", pause_detection=True):
        with pytest.raises(LeaseBusy) as exc_info:
            with svc.acquire("aesthetic_tagger", pause_detection=False):
                pass  # pragma: no cover
        assert exc_info.value.current_holder == "companion_inference"
        assert exc_info.value.requested_holder == "aesthetic_tagger"


def test_same_holder_can_reenter():
    dm = _DM()
    svc = ComputeLeaseService(dm)
    with svc.acquire("companion_inference", pause_detection=True):
        with svc.acquire("companion_inference", pause_detection=True):
            assert dm.paused is True
            assert svc.status().reentry_depth == 2
        # Inner release does not flip paused back yet.
        assert dm.paused is True
    assert dm.paused is False


def test_exception_inside_lease_still_releases():
    dm = _DM(paused=False)
    svc = ComputeLeaseService(dm)
    with pytest.raises(RuntimeError):
        with svc.acquire("companion_inference", pause_detection=True):
            assert dm.paused is True
            raise RuntimeError("boom")
    assert dm.paused is False
    assert svc.status().holder is None


def test_watchdog_force_releases_on_timeout():
    dm = _DM()
    svc = ComputeLeaseService(dm)
    timeout = 0.05  # 50 ms
    # Acquire and hold by simulating a blocking worker.
    holder_done = threading.Event()
    started = threading.Event()

    def worker():
        try:
            with svc.acquire(
                "companion_inference",
                pause_detection=True,
                timeout_s=timeout,
            ):
                started.set()
                # Sleep longer than the timeout so the watchdog fires.
                time.sleep(timeout * 5)
        finally:
            holder_done.set()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(timeout=1.0)

    # Within a short window after timeout, the watchdog should have
    # cleared the holder slot and restored detection.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if svc.status().holder is None and dm.paused is False:
            break
        time.sleep(0.01)

    assert svc.status().holder is None
    assert dm.paused is False
    holder_done.wait(timeout=2.0)
    t.join(timeout=2.0)


def test_release_after_watchdog_does_not_crash():
    """If the watchdog fires first, the holder's __exit__ must be a no-op."""
    dm = _DM()
    svc = ComputeLeaseService(dm)
    timeout = 0.02

    with svc.acquire("companion_inference", pause_detection=True, timeout_s=timeout):
        time.sleep(timeout * 5)
        # Watchdog has fired; lease state is reset.
        assert svc.status().holder is None
        assert dm.paused is False
    # The context manager exit does not raise even though the slot was
    # already cleared.


def test_holder_must_be_non_empty():
    dm = _DM()
    svc = ComputeLeaseService(dm)
    with pytest.raises(ValueError):
        with svc.acquire("", pause_detection=False):
            pass  # pragma: no cover
