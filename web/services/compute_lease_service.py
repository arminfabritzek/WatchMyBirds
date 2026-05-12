"""Process-local named compute lease for CPU-heavy jobs.

The lease lets one heavy workload at a time (Companion LLM inference,
aesthetic tagger, future callers) coordinate so they do not run
concurrently with each other or — when they ask for it — with the
live OD/CLS detection loop.

Why a single named lease and not three independent locks: deep-scan
mode, the tagger, and the Companion inference all want the same
resource (CPU on the Pi) but for different reasons and with different
acceptance of detection running. A single registry surfaces *who*
holds the resource, makes contention diagnosable, and keeps the
exception-safety logic in one place.

The lease coexists with `DetectionManager.enter_deep_scan_mode()` /
`exit_deep_scan_mode()`. Deep-scan callers continue to use those
methods; they were already shipped, already tested, and have their
own reference-counting semantics. The lease is for the new callers.

Single-process scope. There is no IPC here — both heavy workloads
live in the same WMB Flask process today.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class LeaseBusy(RuntimeError):
    """Raised when a different holder already owns the lease."""

    def __init__(self, current_holder: str, requested_holder: str) -> None:
        super().__init__(
            f"compute lease busy: held by {current_holder!r}, "
            f"requested by {requested_holder!r}"
        )
        self.current_holder = current_holder
        self.requested_holder = requested_holder


class LeaseTimeout(RuntimeError):
    """Raised when a held lease exceeds its hard timeout.

    Surfacing this as an exception is intentional: the watchdog forces
    the lease open again so detection resumes, and the holder is told
    its own timeout fired.
    """


@dataclass(frozen=True)
class LeaseStatus:
    holder: str | None
    started_at: float | None
    reason: str | None
    pause_detection: bool
    timeout_s: float | None
    reentry_depth: int


class ComputeLeaseService:
    """Single-holder, re-entrant-by-same-holder compute lease."""

    def __init__(self, detection_manager) -> None:
        self._detection_manager = detection_manager
        self._lock = threading.Lock()
        self._holder: str | None = None
        self._reentry_depth: int = 0
        self._started_at: float | None = None
        self._reason: str | None = None
        self._pause_detection: bool = False
        self._timeout_s: float | None = None
        self._paused_before_lease: bool = False
        self._watchdog: threading.Timer | None = None

    def status(self) -> LeaseStatus:
        with self._lock:
            return LeaseStatus(
                holder=self._holder,
                started_at=self._started_at,
                reason=self._reason,
                pause_detection=self._pause_detection,
                timeout_s=self._timeout_s,
                reentry_depth=self._reentry_depth,
            )

    @contextmanager
    def acquire(
        self,
        holder: str,
        *,
        pause_detection: bool = False,
        reason: str | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[ComputeLeaseService]:
        """Acquire the lease for `holder`.

        Re-entry from the same `holder` is allowed and increments a
        depth counter; only the outermost release toggles state. A
        different holder triggers `LeaseBusy` immediately (no waiting).

        `pause_detection=True` flips `detection_manager.paused` to True
        on entry and restores the previous value on exit. The previous
        value is captured at lease *start*, not on every re-entry, so
        nested calls all observe the same restoration target.

        `timeout_s` arms a watchdog that force-releases the lease and
        sets a flag the holder will notice as `LeaseTimeout` on the
        next status check. The watchdog never kills the worker — it
        only restores detection and clears the holder slot.
        """
        if not holder:
            raise ValueError("holder must be a non-empty string")

        with self._lock:
            if self._holder is not None and self._holder != holder:
                raise LeaseBusy(self._holder, holder)
            if self._holder is None:
                self._holder = holder
                self._reentry_depth = 1
                self._started_at = time.monotonic()
                self._reason = reason
                self._pause_detection = pause_detection
                self._timeout_s = timeout_s
                if pause_detection and self._detection_manager is not None:
                    self._paused_before_lease = bool(
                        getattr(self._detection_manager, "paused", False)
                    )
                    self._detection_manager.paused = True
                else:
                    self._paused_before_lease = False
                if timeout_s is not None and timeout_s > 0:
                    self._arm_watchdog(timeout_s, holder)
                logger.info(
                    "compute lease acquired by %r (reason=%s, "
                    "pause_detection=%s, timeout_s=%s)",
                    holder,
                    reason,
                    pause_detection,
                    timeout_s,
                )
            else:
                self._reentry_depth += 1
                logger.debug(
                    "compute lease re-entered by %r (depth=%d)",
                    holder,
                    self._reentry_depth,
                )

        try:
            yield self
        finally:
            self._release(holder)

    def _arm_watchdog(self, timeout_s: float, holder: str) -> None:
        # Caller already holds self._lock when arming on entry; the
        # watchdog itself takes the lock when firing, so we never call
        # _force_release while still holding the lock.
        timer = threading.Timer(timeout_s, self._on_watchdog_fired, args=(holder,))
        timer.daemon = True
        self._watchdog = timer
        timer.start()

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            try:
                self._watchdog.cancel()
            except RuntimeError:
                # Timer already finished or thread state inconsistent.
                pass
            self._watchdog = None

    def _on_watchdog_fired(self, holder: str) -> None:
        with self._lock:
            if self._holder != holder:
                # Lease already changed hands or was released cleanly.
                return
            elapsed = (
                time.monotonic() - self._started_at
                if self._started_at is not None
                else None
            )
            logger.warning(
                "compute lease watchdog fired for holder %r after %.1fs; "
                "force-releasing so detection can resume",
                holder,
                elapsed if elapsed is not None else -1.0,
            )
            self._restore_paused_state_locked()
            self._reset_locked()

    def _release(self, holder: str) -> None:
        with self._lock:
            if self._holder is None:
                # Watchdog already cleaned up. Nothing to do.
                return
            if self._holder != holder:
                # Different holder cleaned up first (should not happen
                # under normal flow). Log loudly; do not crash.
                logger.error(
                    "compute lease release mismatch: held by %r, released by %r",
                    self._holder,
                    holder,
                )
                return
            self._reentry_depth -= 1
            if self._reentry_depth > 0:
                logger.debug(
                    "compute lease re-released by %r (depth=%d)",
                    holder,
                    self._reentry_depth,
                )
                return
            elapsed = (
                time.monotonic() - self._started_at
                if self._started_at is not None
                else 0.0
            )
            logger.info(
                "compute lease released by %r after %.1fs",
                holder,
                elapsed,
            )
            self._cancel_watchdog()
            self._restore_paused_state_locked()
            self._reset_locked()

    def _restore_paused_state_locked(self) -> None:
        if self._pause_detection and self._detection_manager is not None:
            try:
                self._detection_manager.paused = self._paused_before_lease
            except Exception as exc:
                logger.error(
                    "compute lease: failed to restore detection_manager.paused: %s",
                    exc,
                )

    def _reset_locked(self) -> None:
        self._holder = None
        self._reentry_depth = 0
        self._started_at = None
        self._reason = None
        self._pause_detection = False
        self._timeout_s = None
        self._paused_before_lease = False
        # Watchdog already cancelled by caller in the clean path; clear
        # the slot so a stale Timer never lingers.
        self._watchdog = None


_GLOBAL_SERVICE: ComputeLeaseService | None = None
_GLOBAL_LOCK = threading.Lock()


def init_compute_lease_service(detection_manager) -> ComputeLeaseService:
    """Initialise the process-global compute lease service.

    Idempotent: subsequent calls return the existing instance to keep
    test harnesses simple. A second `init` with a different
    `detection_manager` is a programming error and raises so the
    mistake surfaces immediately.
    """
    global _GLOBAL_SERVICE
    with _GLOBAL_LOCK:
        if _GLOBAL_SERVICE is None:
            _GLOBAL_SERVICE = ComputeLeaseService(detection_manager)
        elif _GLOBAL_SERVICE._detection_manager is not detection_manager:
            raise RuntimeError(
                "compute lease service already initialised with a different "
                "detection_manager; this is unsupported"
            )
        return _GLOBAL_SERVICE


def get_compute_lease_service() -> ComputeLeaseService | None:
    """Return the process-global service, or None if not initialised."""
    with _GLOBAL_LOCK:
        return _GLOBAL_SERVICE


def reset_compute_lease_service_for_testing() -> None:
    """Test-only: reset the process-global service so unit tests can
    instantiate fresh services with their own detection_manager doubles."""
    global _GLOBAL_SERVICE
    with _GLOBAL_LOCK:
        _GLOBAL_SERVICE = None
