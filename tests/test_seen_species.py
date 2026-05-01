"""Tests for ``utils.db.seen_species``.

Each test gets a fresh tmp DB so the seen-species table starts empty.
The module's three public helpers (``is_new_species``,
``mark_species_seen``, ``reset_seen_species``) are exercised both for
their happy paths and for the race / failure modes the notification
service relies on.
"""

import pytest


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch, tmp_path):
    """Point OUTPUT_DIR at a tmp dir and reset the schema-init cache so
    the seen_species table is created fresh for each test."""
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    import config

    config._CONFIG = None
    monkeypatch.setattr("utils.db.connection._schema_initialized_paths", set())
    yield


def test_is_new_species_returns_true_on_empty_db():
    from utils.db.seen_species import is_new_species

    assert is_new_species("Parus_major") is True


def test_mark_species_seen_then_is_new_returns_false():
    from utils.db.seen_species import is_new_species, mark_species_seen

    assert mark_species_seen("Parus_major", image_filename="x.jpg", score=0.9) is True
    assert is_new_species("Parus_major") is False


def test_mark_species_seen_idempotent():
    """Calling mark_species_seen twice is safe (INSERT OR IGNORE)."""
    from utils.db.seen_species import mark_species_seen

    assert mark_species_seen("Parus_major") is True
    # Second call returns False (no row inserted) but doesn't raise.
    assert mark_species_seen("Parus_major") is False


def test_reset_clears_log():
    from utils.db.seen_species import (
        is_new_species,
        mark_species_seen,
        reset_seen_species,
    )

    mark_species_seen("Parus_major")
    mark_species_seen("Cyanistes_caeruleus")

    deleted = reset_seen_species()
    assert deleted == 2

    assert is_new_species("Parus_major") is True
    assert is_new_species("Cyanistes_caeruleus") is True


def test_empty_key_is_never_new():
    """Empty / None species keys must short-circuit so callers can pass
    raw classifier output without filtering."""
    from utils.db.seen_species import is_new_species, mark_species_seen

    assert is_new_species("") is False
    assert is_new_species(None) is False  # type: ignore[arg-type]
    assert mark_species_seen("") is False


def test_list_seen_species_orders_by_first_seen():
    import time

    from utils.db.seen_species import list_seen_species, mark_species_seen

    mark_species_seen("Parus_major", image_filename="early.jpg", score=0.5)
    # Sleep so the CURRENT_TIMESTAMP has a chance to advance to the next
    # second on platforms that resolve at second granularity.
    time.sleep(1.1)
    mark_species_seen("Cyanistes_caeruleus", image_filename="late.jpg", score=0.9)

    rows = list_seen_species()
    assert len(rows) == 2
    assert rows[0]["species_key"] == "Parus_major"
    assert rows[1]["species_key"] == "Cyanistes_caeruleus"
    assert rows[0]["first_image_filename"] == "early.jpg"
    assert rows[1]["first_score"] == pytest.approx(0.9)
