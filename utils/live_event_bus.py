"""In-process pub/sub bus for live UI events (stream-page ticker etc.).

Why a bus and not just one queue:
- Many browser tabs can be open on /stream simultaneously. Each tab opens
  its own SSE stream and needs its own queue so a slow tab doesn't block
  a fast one and so a fast tab doesn't drain events meant for the slow tab.
- The publisher (detector loop) must never block on a missing or slow
  subscriber. Each subscriber queue is bounded; on overflow we drop the
  oldest event for that subscriber only.

Threading: detector runs on a worker thread, Flask SSE handlers run on
request threads — every method here is thread-safe.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

# Detection peaks stack ~a dozen events/s; 64 gives slow clients ~half a
# minute of slack at typical rates before drops start.
_SUBSCRIBER_BUFFER = 64


class LiveEventBus:
    """Single-process fan-out bus. One publisher, many subscribers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._drop_counts: dict[str, int] = {}

    def subscribe(self) -> tuple[str, queue.Queue[dict[str, Any]]]:
        sub_id = uuid.uuid4().hex
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_SUBSCRIBER_BUFFER)
        with self._lock:
            self._subscribers[sub_id] = q
            self._drop_counts[sub_id] = 0
        logger.debug("LiveEventBus: subscribe %s (total=%d)", sub_id, len(self._subscribers))
        return sub_id, q

    def unsubscribe(self, sub_id: str) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)
            dropped = self._drop_counts.pop(sub_id, 0)
        if dropped:
            logger.info(
                "LiveEventBus: client %s disconnected after %d dropped events",
                sub_id,
                dropped,
            )

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subscribers.items())
        for sub_id, q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Drop oldest to keep the live feel — stale events are
                # worse than a small gap for a kiosk ticker.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                    with self._lock:
                        self._drop_counts[sub_id] = self._drop_counts.get(sub_id, 0) + 1
                except queue.Empty:
                    # Race: subscriber drained the queue between the
                    # Full exception above and this get_nowait. Caller's
                    # event is already lost in that case — nothing to
                    # do, the subscriber will catch up on the next push.
                    pass

    def stream(
        self,
        q: queue.Queue[dict[str, Any]],
        keepalive_seconds: float = 15.0,
    ) -> Iterator[dict[str, Any] | None]:
        """Yield events for one subscriber. ``None`` means: emit a keepalive.

        Keepalives matter behind reverse proxies — connections
        that go quiet for ~30s can be killed mid-stream. A periodic
        ping keeps the pipe warm without putting visible noise into
        the ticker.
        """
        deadline = time.monotonic() + keepalive_seconds
        while True:
            timeout = max(0.1, deadline - time.monotonic())
            try:
                event = q.get(timeout=timeout)
                yield event
                deadline = time.monotonic() + keepalive_seconds
            except queue.Empty:
                yield None
                deadline = time.monotonic() + keepalive_seconds


_singleton: LiveEventBus | None = None
_singleton_lock = threading.Lock()


def get_bus() -> LiveEventBus:
    """Lazy module-level singleton. Imported from both detector and web."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = LiveEventBus()
    return _singleton
