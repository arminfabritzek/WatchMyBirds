"""Bounded local timing summaries shared by the pipeline workers."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock


@dataclass
class _Timing:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = 0.0


class PipelineMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._timings: dict[str, _Timing] = {}
        self._counts: Counter[str] = Counter()

    def observe(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            sample = self._timings.setdefault(stage, _Timing())
            sample.count += 1
            sample.total += milliseconds
            sample.minimum = min(sample.minimum, milliseconds)
            sample.maximum = max(sample.maximum, milliseconds)

    def increment(self, counter: str) -> None:
        with self._lock:
            self._counts[counter] += 1

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(stage, (time.perf_counter() - started) * 1000)

    def drain(self) -> dict[str, object]:
        with self._lock:
            result = {
                "timings_ms": {
                    stage: {
                        "count": sample.count,
                        "avg": round(sample.total / sample.count, 3),
                        "min": round(sample.minimum, 3),
                        "max": round(sample.maximum, 3),
                    }
                    for stage, sample in self._timings.items()
                },
                "counts": dict(self._counts),
            }
            self._timings.clear()
            self._counts.clear()
            return result
