"""Rank the originals that every species keeps regardless of age."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from utils.cover_selection import cover_order_sql
from utils.path_manager import PathManager


def original_exists(paths: PathManager) -> Callable[[str], bool]:
    """Return a check that a photo's original is really on disk."""
    return lambda name: (
        Path(name).name == name and paths.get_original_path(name).is_file()
    )


def best_original_keys(
    conn: sqlite3.Connection,
    limit: int,
    *,
    media_exists: Callable[[str], bool] | None = None,
) -> frozenset[str]:
    """Return one object key for each of the best ``limit`` photos per species.

    A photo with several birds of one species takes a single place, so the
    limit counts originals rather than birds. With ``media_exists`` a photo
    whose file is gone gives its place to the next one; the database flag
    alone can be stale, and a kept photo is never revisited to correct it.
    """
    if limit <= 0:
        return frozenset()
    order = cover_order_sql()
    ranked = f"""
        SELECT species_key, object_key, image_filename, place FROM (
            SELECT species_key, object_key, image_filename, ROW_NUMBER() OVER (
                PARTITION BY species_key ORDER BY {order}, object_key
            ) AS place
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY species_key, image_filename
                    ORDER BY {order}, object_key
                ) AS place_on_photo
                FROM current_observations
                WHERE species_key != 'Unknown_species' AND original_present = 1
            )
            WHERE place_on_photo = 1
        )
    """
    if media_exists is None:
        rows = conn.execute(ranked + " WHERE place <= ?", (limit,))
        return frozenset(row[1] for row in rows)
    kept: Counter[str] = Counter()
    keys = set()
    for species, key, filename, _place in conn.execute(
        ranked + " ORDER BY species_key, place"
    ):
        if kept[species] < limit and media_exists(filename):
            kept[species] += 1
            keys.add(key)
    return frozenset(keys)
