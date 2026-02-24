"""
Tests for AnalysisQueue gate lifecycle and dedup.
"""

import time

from core.analysis_queue import AnalysisQueue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDetectionManager:
    def __init__(self):
        self._gate_open = True

    def enter_deep_scan_mode(self):
        self._gate_open = False

    def exit_deep_scan_mode(self):
        self._gate_open = True

    def is_deep_scan_active(self) -> bool:
        return not self._gate_open


# ---------------------------------------------------------------------------
# Gate lifecycle
# ---------------------------------------------------------------------------


def test_gate_enters_and_exits_around_job(monkeypatch):
    """Gate must be entered before processing and exited after, even for normal jobs."""
    dm = _FakeDetectionManager()
    gate_events: list[tuple[str, bool]] = []

    def _processor(job):
        # While processor runs, gate should be closed (dm._gate_open == False)
        gate_events.append(("during", dm._gate_open))

    q = AnalysisQueue()
    q.set_detection_manager(dm)
    # Force gate enabled
    monkeypatch.setattr(
        "core.analysis_queue.get_config", lambda: {"DEEP_SCAN_GATE_ENABLED": True}
    )

    q.start(_processor)
    q.enqueue({"filename": "gate_test.jpg"})
    time.sleep(2.5)  # Grace period (1s) + processing
    q.stop()

    # During processing the gate must have been closed
    assert len(gate_events) == 1
    assert gate_events[0] == ("during", False)
    # After stop, gate should be open again
    assert dm._gate_open is True


def test_gate_exits_on_processor_exception(monkeypatch):
    """Gate must be restored (exit) even when the processor raises."""
    dm = _FakeDetectionManager()

    def _failing_processor(job):
        raise RuntimeError("boom")

    q = AnalysisQueue()
    q.set_detection_manager(dm)
    monkeypatch.setattr(
        "core.analysis_queue.get_config", lambda: {"DEEP_SCAN_GATE_ENABLED": True}
    )

    q.start(_failing_processor)
    q.enqueue({"filename": "crash.jpg"})
    time.sleep(2.5)
    q.stop()

    # Gate must be open despite exception
    assert dm._gate_open is True


def test_gate_skipped_when_disabled(monkeypatch):
    """When DEEP_SCAN_GATE_ENABLED=False, gate is never toggled."""
    dm = _FakeDetectionManager()
    called = []

    def _processor(job):
        called.append(dm._gate_open)

    q = AnalysisQueue()
    q.set_detection_manager(dm)
    monkeypatch.setattr(
        "core.analysis_queue.get_config", lambda: {"DEEP_SCAN_GATE_ENABLED": False}
    )

    q.start(_processor)
    q.enqueue({"filename": "no_gate.jpg"})
    time.sleep(2.0)
    q.stop()

    # Processor ran
    assert len(called) == 1
    # Gate was never closed
    assert called[0] is True
    assert dm._gate_open is True


def test_gate_skipped_when_no_detection_manager(monkeypatch):
    """Works without a DetectionManager (gate is a no-op)."""
    called = []

    def _processor(job):
        called.append(True)

    q = AnalysisQueue()
    # No set_detection_manager() call
    monkeypatch.setattr(
        "core.analysis_queue.get_config", lambda: {"DEEP_SCAN_GATE_ENABLED": True}
    )

    q.start(_processor)
    q.enqueue({"filename": "solo.jpg"})
    time.sleep(2.0)
    q.stop()

    assert len(called) == 1


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_dedup_prevents_double_enqueue():
    """Same filename cannot be enqueued twice while first is still pending."""
    q = AnalysisQueue()
    ok1 = q.enqueue({"filename": "dup.jpg"})
    ok2 = q.enqueue({"filename": "dup.jpg"})
    ok3 = q.enqueue({"filename": "other.jpg"})

    assert ok1 is True
    assert ok2 is False  # dedup rejects
    assert ok3 is True
    assert q.pending_count() == 2


def test_dedup_set_cleared_after_processing(monkeypatch):
    """After a job completes, its filename is removed from the dedup set."""
    processed = []

    def _processor(job):
        processed.append(job["filename"])

    q = AnalysisQueue()
    monkeypatch.setattr(
        "core.analysis_queue.get_config", lambda: {"DEEP_SCAN_GATE_ENABLED": False}
    )
    q.start(_processor)
    q.enqueue({"filename": "once.jpg"})
    time.sleep(1.5)

    # After processing, the dedup set must be empty
    assert q.pending_filenames_count() == 0
    assert len(processed) == 1

    # Re-enqueue must succeed now
    ok = q.enqueue({"filename": "once.jpg"})
    assert ok is True

    time.sleep(1.5)
    q.stop()
    assert len(processed) == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_pending_count_reflects_queue():
    q = AnalysisQueue()
    assert q.pending_count() == 0
    q.enqueue({"filename": "a.jpg"})
    q.enqueue({"filename": "b.jpg"})
    assert q.pending_count() == 2
