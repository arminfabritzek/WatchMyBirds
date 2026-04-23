"""Regression guards for the strict-confirmed gallery policy and the
CLS-v2 decision_level / raw_species_name persistence path.

Both features shipped together on 2026-04-23:
- Gallery only surfaces detections where ``decision_state = 'confirmed'``.
  NULL and uncertain/unknown states are excluded (they live in the
  review queue instead).
- The classifier decision layer's ``decision_level`` and
  ``raw_species_name`` fields round-trip through the DB schema so we
  can later audit whether a saved label was species or genus level.

These tests guard both invariants — if either slips, operators lose
either trust in the public gallery or the ability to reconstruct why
the classifier labelled things the way it did.
"""

import sqlite3

from utils.db.detections import (
    fetch_active_detection_ids_in_date_range,
    fetch_active_detection_selection_in_date_range,
)


def _minimal_schema() -> sqlite3.Connection:
    """Build just enough schema for the gallery visibility queries.

    Tracks the real schema shape in utils/db/connection.py — keep in
    sync if columns that the gallery queries touch ever move.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT,
            source_id INTEGER,
            review_status TEXT DEFAULT 'untagged'
        );

        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            decision_state TEXT,
            decision_level TEXT,
            raw_species_name TEXT
        );
        """
    )
    return conn


class TestRejectLevelExcluded:
    """The classifier decision-level gate: reject rows must not reach
    the gallery even when ``decision_state = 'confirmed'``.

    Scenario: the temporal smoother sees many bird-OD frames in a row
    and stamps 'confirmed'. But the classifier's top-1 was below its
    calibrated species threshold AND the genus fallback did not
    accept either — so ``cls_class_name`` is empty. Showing those in
    the gallery produces 'Unknown species' cards, which is worse than
    hiding them. They stay in the review queue instead.
    """

    def test_reject_level_is_excluded_even_when_confirmed(self):
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, "
            "decision_state, decision_level) VALUES (?, ?, ?, ?, ?)",
            (1, "a.jpg", "active", "confirmed", "reject"),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == []

    def test_species_level_is_included(self):
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, "
            "decision_state, decision_level) VALUES (?, ?, ?, ?, ?)",
            (1, "a.jpg", "active", "confirmed", "species"),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == [1]

    def test_genus_level_is_included(self):
        """Genus fallback rows (Sylvia_sp. etc.) stay in the gallery —
        they have a meaningful label, unlike reject."""
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, "
            "decision_state, decision_level) VALUES (?, ?, ?, ?, ?)",
            (1, "a.jpg", "active", "confirmed", "genus"),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == [1]

    def test_null_decision_level_is_included_for_backward_compat(self):
        """Historical rows saved before the decision_level column
        existed have NULL in that column. They must stay visible when
        their decision_state=confirmed — otherwise the gallery would
        silently drop all pre-2026-04-23 data."""
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, "
            "decision_state, decision_level) VALUES (?, ?, ?, ?, ?)",
            (1, "a.jpg", "active", "confirmed", None),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == [1]

    def test_mixed_fixture_only_surfaces_species_and_genus(self):
        """All four decision levels in one batch — species and genus
        survive, reject is filtered, NULL survives for compat."""
        conn = _minimal_schema()
        conn.executemany(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            [
                ("species.jpg", "20260423_120000"),
                ("genus.jpg", "20260423_121000"),
                ("reject.jpg", "20260423_122000"),
                ("legacy.jpg", "20260423_123000"),
            ],
        )
        conn.executemany(
            "INSERT INTO detections(detection_id, image_filename, status, "
            "decision_state, decision_level) VALUES (?, ?, ?, ?, ?)",
            [
                (1, "species.jpg", "active", "confirmed", "species"),
                (2, "genus.jpg", "active", "confirmed", "genus"),
                (3, "reject.jpg", "active", "confirmed", "reject"),
                (4, "legacy.jpg", "active", "confirmed", None),
            ],
        )
        selection = fetch_active_detection_selection_in_date_range(
            conn, "2026-04-23", "2026-04-23"
        )
        assert sorted(selection["detection_ids"]) == [1, 2, 4]


class TestStrictConfirmedFilter:
    """Gallery query must ignore everything that isn't explicitly
    ``decision_state = 'confirmed'``."""

    def test_null_decision_state_is_excluded(self):
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, decision_state) "
            "VALUES (?, ?, ?, ?)",
            (1, "a.jpg", "active", None),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == []

    def test_uncertain_is_excluded(self):
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, decision_state) "
            "VALUES (?, ?, ?, ?)",
            (1, "a.jpg", "active", "uncertain"),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == []

    def test_unknown_is_excluded(self):
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, decision_state) "
            "VALUES (?, ?, ?, ?)",
            (1, "a.jpg", "active", "unknown"),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == []

    def test_confirmed_is_included(self):
        conn = _minimal_schema()
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            ("a.jpg", "20260423_120000"),
        )
        conn.execute(
            "INSERT INTO detections(detection_id, image_filename, status, decision_state) "
            "VALUES (?, ?, ?, ?)",
            (1, "a.jpg", "active", "confirmed"),
        )
        ids = fetch_active_detection_ids_in_date_range(conn, "2026-04-23", "2026-04-23")
        assert ids == [1]

    def test_only_confirmed_survives_mixed_fixture(self):
        conn = _minimal_schema()
        conn.executemany(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            [
                ("null.jpg", "20260423_120000"),
                ("confirmed.jpg", "20260423_121000"),
                ("uncertain.jpg", "20260423_122000"),
                ("unknown.jpg", "20260423_123000"),
            ],
        )
        conn.executemany(
            "INSERT INTO detections(detection_id, image_filename, status, decision_state) "
            "VALUES (?, ?, ?, ?)",
            [
                (1, "null.jpg", "active", None),
                (2, "confirmed.jpg", "active", "confirmed"),
                (3, "uncertain.jpg", "active", "uncertain"),
                (4, "unknown.jpg", "active", "unknown"),
            ],
        )
        selection = fetch_active_detection_selection_in_date_range(
            conn, "2026-04-23", "2026-04-23"
        )
        assert selection["detection_ids"] == [2]
        assert selection["image_filenames"] == ["confirmed.jpg"]


class TestDecisionLevelColumnsMigrate:
    """The production schema init must add the two new columns on any
    pre-existing database, idempotently."""

    def test_ensure_column_is_idempotent(self):
        """Running _ensure_column_on_table twice must not fail on the
        second call (ALTER TABLE ADD COLUMN would otherwise error)."""
        from utils.db.connection import _ensure_column_on_table

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE detections (detection_id INTEGER PRIMARY KEY);")

        _ensure_column_on_table(conn, "detections", "decision_level", "TEXT")
        _ensure_column_on_table(conn, "detections", "decision_level", "TEXT")
        _ensure_column_on_table(conn, "detections", "raw_species_name", "TEXT")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(detections);")}
        assert "decision_level" in cols
        assert "raw_species_name" in cols

    def test_migration_preserves_existing_rows_as_null(self):
        """Pre-migration rows must survive as NULL on the new columns.
        This is the real-world scenario: a user upgrades; their
        existing detections must not be destroyed."""
        from utils.db.connection import _ensure_column_on_table

        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE detections (detection_id INTEGER PRIMARY KEY, od_class_name TEXT);"
        )
        conn.execute(
            "INSERT INTO detections(detection_id, od_class_name) VALUES (1, 'bird');"
        )

        _ensure_column_on_table(conn, "detections", "decision_level", "TEXT")
        _ensure_column_on_table(conn, "detections", "raw_species_name", "TEXT")

        row = conn.execute(
            "SELECT detection_id, od_class_name, decision_level, raw_species_name "
            "FROM detections WHERE detection_id = 1;"
        ).fetchone()
        assert row[0] == 1
        assert row[1] == "bird"
        assert row[2] is None
        assert row[3] is None
