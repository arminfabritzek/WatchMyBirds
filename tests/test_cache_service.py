"""Unit tests for ``web.services.cache_service``.

Covers TTL boundaries, key collisions, prefix invalidation, the
@invalidates decorator, and the bypass behaviour for non-positive
TTL. No Flask app, no DB, pure module isolation.
"""

from __future__ import annotations

import threading
import time

import pytest

from web.services import cache_service


@pytest.fixture(autouse=True)
def _reset_cache():
    cache_service.clear()
    yield
    cache_service.clear()


def test_first_call_runs_builder_and_stores():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return "value-1"

    out = cache_service.cached("k", ttl_seconds=60, builder=build)
    assert out == "value-1"
    assert calls["n"] == 1
    assert cache_service.size() == 1


def test_second_call_inside_ttl_skips_builder():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return calls["n"]

    a = cache_service.cached("k", ttl_seconds=60, builder=build)
    b = cache_service.cached("k", ttl_seconds=60, builder=build)
    assert a == b == 1
    assert calls["n"] == 1


def test_expired_entry_rebuilds(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(cache_service.time, "monotonic", lambda: clock["t"])

    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return calls["n"]

    assert cache_service.cached("k", ttl_seconds=5, builder=build) == 1
    clock["t"] += 4.999
    assert cache_service.cached("k", ttl_seconds=5, builder=build) == 1
    clock["t"] += 0.002  # crosses the 5s boundary
    assert cache_service.cached("k", ttl_seconds=5, builder=build) == 2


def test_non_positive_ttl_bypasses_cache():
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return calls["n"]

    assert cache_service.cached("k", ttl_seconds=0, builder=build) == 1
    assert cache_service.cached("k", ttl_seconds=0, builder=build) == 2
    assert cache_service.cached("k", ttl_seconds=-1, builder=build) == 3
    assert cache_service.size() == 0


def test_distinct_keys_do_not_collide():
    cache_service.cached("a", 60, lambda: "A")
    cache_service.cached("b", 60, lambda: "B")
    assert cache_service.cached("a", 60, lambda: "X") == "A"
    assert cache_service.cached("b", 60, lambda: "X") == "B"
    assert cache_service.size() == 2


def test_invalidate_prefix_drops_matching_keys():
    cache_service.cached("analytics.summary", 60, lambda: 1)
    cache_service.cached("analytics.weather", 60, lambda: 2)
    cache_service.cached("gallery.daily", 60, lambda: 3)

    dropped = cache_service.invalidate("analytics.")
    assert dropped == 2
    assert cache_service.size() == 1

    # gallery key survives
    assert cache_service.cached("gallery.daily", 60, lambda: 99) == 3


def test_invalidate_empty_prefix_drops_all():
    cache_service.cached("a", 60, lambda: 1)
    cache_service.cached("b", 60, lambda: 2)
    assert cache_service.invalidate("") == 2
    assert cache_service.size() == 0


def test_invalidate_no_match_returns_zero():
    cache_service.cached("a", 60, lambda: 1)
    assert cache_service.invalidate("nope.") == 0
    assert cache_service.size() == 1


def test_invalidates_decorator_drops_prefixes_on_success():
    cache_service.cached("analytics.summary", 60, lambda: "old")
    cache_service.cached("gallery.daily", 60, lambda: "old")
    cache_service.cached("species.list", 60, lambda: "keep")

    @cache_service.invalidates("analytics.", "gallery.")
    def mutate():
        return "ok"

    assert mutate() == "ok"
    assert cache_service.cached("species.list", 60, lambda: "new") == "keep"
    # analytics + gallery were dropped; rebuilding now sees fresh value
    assert cache_service.cached("analytics.summary", 60, lambda: "new") == "new"
    assert cache_service.cached("gallery.daily", 60, lambda: "new") == "new"


def test_invalidates_decorator_does_not_drop_on_exception():
    cache_service.cached("analytics.summary", 60, lambda: "old")

    @cache_service.invalidates("analytics.")
    def boom():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        boom()
    # Mutation failed, cache stays consistent
    assert cache_service.cached("analytics.summary", 60, lambda: "new") == "old"


def test_invalidates_decorator_preserves_return_value_and_args():
    @cache_service.invalidates("anything.")
    def add(a, b, *, c=0):
        return a + b + c

    assert add(1, 2, c=3) == 6


def test_concurrent_access_does_not_deadlock_or_lose_entries():
    """Smoke test: many threads writing distinct keys all complete and
    every entry survives."""
    threads = []
    for i in range(20):
        key = f"thread.{i}"
        t = threading.Thread(
            target=lambda k=key: cache_service.cached(k, 60, lambda: k)
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert cache_service.size() == 20


def test_builder_only_holds_local_lock_briefly():
    """If builder were called inside the lock, a slow builder would
    serialise unrelated keys. Verify two slow builders for distinct
    keys can overlap."""
    started = threading.Event()
    proceed = threading.Event()

    def slow_a():
        started.set()
        proceed.wait(timeout=2)
        return "A"

    def fast_b():
        return "B"

    a_thread = threading.Thread(
        target=lambda: cache_service.cached("slow.a", 60, slow_a)
    )
    a_thread.start()
    assert started.wait(timeout=2), "slow builder never started"

    # While slow_a is still inside its builder, fast_b must complete
    # without waiting on the lock.
    t0 = time.monotonic()
    cache_service.cached("fast.b", 60, fast_b)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"fast builder waited {elapsed:.3f}s on unrelated slow builder"

    proceed.set()
    a_thread.join(timeout=2)
