"""Bounded process-local read cache with per-key rebuild coordination."""

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from threading import Condition
from time import monotonic
from typing import Any


@dataclass
class _Entry:
    value: Any
    version: Hashable
    expires_at: float


class ReadCache:
    def __init__(self, max_entries: int = 128) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._pending: dict[Hashable, object] = {}
        self._condition = Condition()

    def get[T](
        self,
        key: Hashable,
        ttl: float,
        builder: Callable[[], T],
        *,
        version: Hashable = None,
    ) -> T:
        if ttl <= 0:
            return builder()
        with self._condition:
            while True:
                now = monotonic()
                for expired in [
                    k for k, entry in self._entries.items() if entry.expires_at <= now
                ]:
                    del self._entries[expired]
                entry = self._entries.get(key)
                if entry is not None and entry.version == version:
                    self._entries.move_to_end(key)
                    return entry.value
                # An obsolete payload must not survive while its replacement builds.
                self._entries.pop(key, None)
                if key not in self._pending:
                    token = object()
                    self._pending[key] = token
                    break
                self._condition.wait()
        try:
            value = builder()
            with self._condition:
                # Invalidations also cancel publication by an already running builder.
                if self._pending.get(key) is token:
                    self._entries[key] = _Entry(value, version, monotonic() + ttl)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
            return value
        finally:
            with self._condition:
                if self._pending.get(key) is token:
                    del self._pending[key]
                self._condition.notify_all()

    def invalidate(self, predicate: Callable[[Hashable], bool]) -> int:
        with self._condition:
            victims = [key for key in self._entries if predicate(key)]
            for key in victims:
                del self._entries[key]
            for key in [key for key in self._pending if predicate(key)]:
                del self._pending[key]
            self._condition.notify_all()
            return len(victims)

    def clear(self) -> None:
        self.invalidate(lambda _key: True)

    def size(self) -> int:
        with self._condition:
            return len(self._entries)
