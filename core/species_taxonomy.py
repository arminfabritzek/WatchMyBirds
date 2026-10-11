"""Resolve scientific-name synonyms against the active geographic model."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path


def scientific_name_key(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).casefold()


@lru_cache(maxsize=1)
def load_geographic_synonyms() -> tuple[tuple[tuple[str, ...], ...], str]:
    source = (
        Path(__file__).resolve().parent.parent / "assets" / "geographic_synonyms.json"
    )
    content = source.read_bytes()
    document = json.loads(content)
    groups = tuple(tuple(group) for group in document["groups"])
    if not groups or any(
        len(set(group)) != len(group)
        or len(group) < 2
        or any(not re.fullmatch(r"[A-Z][a-z]+ [a-z]+", name) for name in group)
        for group in groups
    ):
        raise ValueError("Invalid geographic synonym reference")
    return groups, hashlib.sha256(content).hexdigest()


class GeographicNameIndex:
    def __init__(
        self, labels: Iterable[str], synonym_groups: Iterable[Iterable[str]]
    ) -> None:
        exact: dict[str, set[str]] = defaultdict(set)
        for label in labels:
            exact[scientific_name_key(label)].add(label)
        aliases: dict[str, set[str]] = defaultdict(set)
        for group in synonym_groups:
            names = {scientific_name_key(name) for name in group}
            targets = {target for name in names for target in exact.get(name, ())}
            for name in names:
                aliases[name].update(targets)
        self._exact = dict(exact)
        self._aliases = dict(aliases)

    def resolve(self, name: str) -> str | None:
        key = scientific_name_key(name)
        targets = self._exact.get(key, self._aliases.get(key, set()))
        # A split or conflicting reference must not merge geographic outputs.
        return next(iter(targets)) if len(targets) == 1 else None
