"""Persistent log of species the operator has ever been notified about.

Used by the "new species only" Telegram mode to gate instant alerts on
the species being unseen. Each function opens its own short-lived
connection via ``closing_connection()`` so callers don't need to thread
DB state through the notification service.

Usage:

    >>> from utils.db.seen_species import is_new_species, mark_species_seen
    >>> if is_new_species("Cyanistes_caeruleus"):
    ...     send_telegram_alert(...)
    ...     mark_species_seen(
    ...         "Cyanistes_caeruleus",
    ...         image_filename="20260430_120000.jpg",
    ...         score=0.92,
    ...     )

The two-step pattern (check then mark) is intentional: callers who fail
to send the Telegram message for transport reasons should not call
``mark_species_seen`` so the next attempt still treats the species as
new. If atomicity matters (e.g. concurrent detection threads racing on
the same species) wrap in a single transaction at the call site.
"""

from __future__ import annotations

from utils.db.connection import closing_connection


def is_new_species(species_key: str) -> bool:
    """Return True when *species_key* has never been logged as seen.

    Empty / None keys always return False so callers can pass the raw
    classifier output without filtering.
    """
    key = str(species_key or "").strip()
    if not key:
        return False
    with closing_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM seen_species WHERE species_key = ? LIMIT 1",
            (key,),
        ).fetchone()
    return row is None


def mark_species_seen(
    species_key: str,
    *,
    image_filename: str | None = None,
    score: float | None = None,
) -> bool:
    """Record *species_key* as seen. Returns True when this was the first
    time (a row was inserted), False when the species was already known.

    INSERT OR IGNORE means concurrent callers race safely: only one wins
    the insert; the others get False and skip their alert.
    """
    key = str(species_key or "").strip()
    if not key:
        return False
    with closing_connection() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO seen_species
                (species_key, first_image_filename, first_score)
            VALUES (?, ?, ?)
            """,
            (key, image_filename, score),
        )
        conn.commit()
    return cur.rowcount > 0


def reset_seen_species() -> int:
    """Wipe the seen-species log. Returns the number of rows removed.

    Operator-facing: exposed via the Settings "Reset known-species list"
    button so a fresh model deploy can re-fire all "Neue Art entdeckt!"
    alerts.
    """
    with closing_connection() as conn:
        cur = conn.execute("DELETE FROM seen_species")
        conn.commit()
    return cur.rowcount


def list_seen_species() -> list[dict]:
    """Return all rows ordered by first-seen-at ASC.

    Used by the Settings panel to show the operator what's currently
    suppressing alerts.
    """
    with closing_connection() as conn:
        rows = conn.execute(
            """
            SELECT species_key, first_seen_at, first_image_filename, first_score
            FROM seen_species
            ORDER BY first_seen_at ASC
            """
        ).fetchall()
    return [
        {
            "species_key": r["species_key"],
            "first_seen_at": r["first_seen_at"],
            "first_image_filename": r["first_image_filename"],
            "first_score": r["first_score"],
        }
        for r in rows
    ]
