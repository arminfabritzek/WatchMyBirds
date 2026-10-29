from concurrent.futures import ThreadPoolExecutor

import pytest

from detectors.pipeline_metrics import PipelineMetrics


def test_summary_distinguishes_counts_and_drains_samples() -> None:
    metrics = PipelineMetrics()
    metrics.observe("classification", 40)
    metrics.observe("classification", 60)
    metrics.increment("duplicate_skipped")

    assert metrics.drain() == {
        "timings_ms": {"classification": {"count": 2, "avg": 50, "min": 40, "max": 60}},
        "counts": {"duplicate_skipped": 1},
    }
    assert metrics.drain() == {"timings_ms": {}, "counts": {}}


def test_failed_stage_is_timed(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([1.0, 1.125])
    monkeypatch.setattr(
        "detectors.pipeline_metrics.time.perf_counter", lambda: next(ticks)
    )
    metrics = PipelineMetrics()

    with pytest.raises(RuntimeError), metrics.measure("save_image"):
        raise RuntimeError("disk full")

    assert metrics.drain()["timings_ms"]["save_image"]["avg"] == 125


def test_concurrent_recording_and_drain_do_not_lose_samples() -> None:
    metrics = PipelineMetrics()
    reports = []

    def write() -> None:
        for _ in range(1000):
            metrics.observe("inference", 10)
            metrics.increment("frames")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(write) for _ in range(2)]
        for _ in range(100):
            reports.append(metrics.drain())
        for future in futures:
            future.result()
    reports.append(metrics.drain())

    assert sum(r["counts"].get("frames", 0) for r in reports) == 2000
    assert (
        sum(r["timings_ms"].get("inference", {}).get("count", 0) for r in reports)
        == 2000
    )
