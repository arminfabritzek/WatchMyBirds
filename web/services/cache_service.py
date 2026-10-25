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

"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from core.read_cache import ReadCache

logger = logging.getLogger(__name__)
T = TypeVar("T")
_cache = ReadCache()


def cached[T](key: str, ttl_seconds: float, builder: Callable[[], T]) -> T:
    """Reuse a live value; coalesce concurrent rebuilds of the same key."""
    return _cache.get(key, ttl_seconds, builder)


def invalidate(prefix: str) -> int:
    """Drop matching values and prevent in-flight builds from republishing them."""
    dropped = _cache.invalidate(lambda key: str(key).startswith(prefix))
    if dropped:
        logger.debug("cache_service: invalidated %d key(s) under %r", dropped, prefix)
    return dropped


def clear() -> None:
    _cache.clear()


def size() -> int:
    return _cache.size()


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
