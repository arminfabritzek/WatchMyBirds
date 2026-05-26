"""Generic TTL cache for heavy list-route aggregations.

Wraps expensive read-side aggregations (event-intelligence summaries,
weather correlations, biodiversity rollups) behind a process-local
key/value store with a per-entry TTL. Wrap once, invalidate by prefix
from mutation routes via the ``@invalidates`` decorator.

Why generic and not bespoke per query: the four heavy `/analytics`
queries plus their `/gallery`, `/species`, `/stream` cousins all share
the same staleness model — bounded by user-edit cadence, tolerant of
a short TTL window. One generic store keeps the invalidation rule
reviewable in one place rather than scattered across N domain caches.

Single-process scope. WMB runs one Flask process today; if a
multi-worker setup ever appears, every worker will hold its own
cache and the prefix-invalidate decorator must broadcast — that is
out of scope here and called out in the plan as non-goal.

Co-exists with the older `core.analytics_core._species_summary_cache`
single-payload cache. That cache is domain-shaped (knows its payload)
and stays as-is; this service is for the generic `key -> value` shape.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _Entry:
    value: Any
    expires_at: float


_store: dict[str, _Entry] = {}
_lock = threading.Lock()


def cached(key: str, ttl_seconds: float, builder: Callable[[], T]) -> T:
    """Return the cached value for ``key`` or call ``builder`` and store it.

    The builder runs outside the lock so a slow query does not block
    unrelated keys. A concurrent caller for the same key may run the
    builder a second time; the last writer wins. We accept that
    redundancy over holding the lock across the (potentially slow)
    builder call.

    A non-positive TTL bypasses the cache entirely — the builder runs
    and the result is returned without being stored.
    """
    if ttl_seconds <= 0:
        return builder()

    now = time.monotonic()
    with _lock:
        entry = _store.get(key)
        if entry is not None and entry.expires_at > now:
            return entry.value

    value = builder()
    with _lock:
        _store[key] = _Entry(value=value, expires_at=time.monotonic() + ttl_seconds)
    return value


def invalidate(prefix: str) -> int:
    """Drop every key starting with ``prefix``. Return the number dropped.

    An empty prefix drops everything — useful in tests.
    """
    with _lock:
        victims = [k for k in _store if k.startswith(prefix)]
        for k in victims:
            del _store[k]
    if victims:
        logger.debug("cache_service: invalidated %d key(s) under %r", len(victims), prefix)
    return len(victims)


def clear() -> None:
    """Drop the entire cache. Primarily for tests."""
    with _lock:
        _store.clear()


def size() -> int:
    """Return the current number of cached entries (live or expired)."""
    with _lock:
        return len(_store)


def invalidates(*prefixes: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: after the wrapped function returns, drop matching keys.

    Use on mutation routes / service functions whose side effects
    invalidate a class of cached aggregations::

        @invalidates("analytics.", "gallery.")
        def confirm_detection(...):
            ...

    The invalidation runs only on a successful return. An exception
    propagates with no cache change — the mutation presumably did not
    take effect, so the cache is still consistent.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            for prefix in prefixes:
                invalidate(prefix)
            return result

        return wrapper

    return decorator
