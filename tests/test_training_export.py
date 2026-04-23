"""Regression guards for the training-export flow.

The export is the ground-truth stream to the upstream training dev.
If any of these predicates or counts drift, the dev either gets fewer
samples than possible (loss of training signal) or gets duplicates
(data pollution). Both are worse than any other kind of bug here.
"""

from __future__ import annotations

import sqlite3

from web.services.training_export_service import (
    DEFAULT_MAX_PER_SPECIES,
    DEFAULT_MAX_TOTAL,
    auto_opt_in_if_enabled,
    build_batch_id,
    confirm_bbox_and_mark_pending,
    filter_eligible_for_pool,
    list_species_availability,
    mark_exported,
    mark_pending,
    select_export_batch,
)


def _schema() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT,
            status TEXT DEFAULT 'active',
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            frame_width INTEGER, frame_height INTEGER,
            od_class_name TEXT,
            od_confidence REAL,
            od_model_id TEXT,
            manual_species_override TEXT,
            manual_bbox_review TEXT,
            bbox_reviewed_at TEXT,
            species_updated_at TEXT,
            is_favorite INTEGER DEFAULT 0
        );
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT
        );
        CREATE TABLE training_exports (
            export_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            detection_id INTEGER NOT NULL,
            export_status TEXT NOT NULL DEFAULT 'pending',
            exported_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(detection_id)
        );
        """
    )
    return conn


def _seed_detection(
    conn: sqlite3.Connection,
    *,
    detection_id: int,
    species: str | None,
    bbox_review: str | None,
    status: str = "active",
    image_filename: str | None = None,
    is_favorite: int = 0,
) -> None:
    filename = image_filename or f"frame_{detection_id}.jpg"
    conn.execute(
        "INSERT OR IGNORE INTO images(filename, timestamp) VALUES (?, ?)",
        (filename, f"20260423_10{detection_id:04d}"),
    )
    conn.execute(
        """
        INSERT INTO detections (
            detection_id, image_filename, status,
            bbox_x, bbox_y, bbox_w, bbox_h,
            frame_width, frame_height,
            od_class_name, od_confidence, od_model_id,
            manual_species_override, manual_bbox_review,
            bbox_reviewed_at, is_favorite
        ) VALUES (?, ?, ?, 0.1, 0.1, 0.2, 0.3, 1920, 1080,
                  'bird', 0.9, 'det_v1', ?, ?, '2026-04-23T10:00:00Z', ?)
        """,
        (detection_id, filename, status, species, bbox_review, is_favorite),
    )


class TestOptionAStrictPredicate:
    """Only rows where the operator set BOTH species override and
    bbox=correct are eligible."""

    def test_both_signals_set_eligible(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        pool = list_species_availability(conn)
        assert len(pool) == 1
        assert pool[0].species == "Parus_major"
        assert pool[0].available_count == 1

    def test_species_only_not_eligible(self):
        """Species override without bbox review is not enough."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review=None)
        assert list_species_availability(conn) == []

    def test_bbox_only_not_eligible(self):
        """Bbox correct without species override is not enough."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species=None, bbox_review="correct")
        assert list_species_availability(conn) == []

    def test_bbox_wrong_not_eligible(self):
        """manual_bbox_review='wrong' does not count as a positive sample."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="wrong")
        assert list_species_availability(conn) == []

    def test_trash_status_not_eligible(self):
        """Rejected detections are never exported even if labels exist."""
        conn = _schema()
        _seed_detection(
            conn,
            detection_id=1,
            species="Parus_major",
            bbox_review="correct",
            status="rejected",
        )
        assert list_species_availability(conn) == []


class TestAlreadyExportedExclusion:
    """Exported rows should not appear in the available pool, but
    should be counted in ``already_exported_count`` for UI display."""

    def test_exported_row_is_excluded_from_available(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        _seed_detection(conn, detection_id=2, species="Parus_major", bbox_review="correct")
        mark_exported(conn, [1], "batch_x")

        pool = list_species_availability(conn)
        assert len(pool) == 1
        row = pool[0]
        assert row.available_count == 1
        assert row.already_exported_count == 1
        assert row.pending_count == 0

    def test_pending_row_is_still_available(self):
        """Pending rows are pre-selected via auto-opt-in but not yet
        downloaded — they still count as available for the next batch."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        mark_pending(conn, [1], "auto_batch")

        pool = list_species_availability(conn)
        assert pool[0].available_count == 1
        assert pool[0].pending_count == 1
        assert pool[0].already_exported_count == 0


class TestFavoritesFirstSampling:
    """Favorites (``is_favorite=1``) are a strictly stronger quality
    signal than a one-shot review approval — the user has looked at
    the image twice and explicitly starred it. The export must pick
    favorites first within each species bucket; non-favorites fill
    the remaining cap randomly. An overall ``max_total`` cap biases
    favorites to survive so the dev gets the best samples first.

    These tests lock in the behaviour so a future refactor cannot
    silently drop favorites back to equal-weight random sampling.
    """

    def test_favorites_land_in_selection_before_non_favorites(self):
        conn = _schema()
        for i in range(10):
            _seed_detection(
                conn,
                detection_id=i,
                species="Parus_major",
                bbox_review="correct",
                is_favorite=0,
            )
        for i in range(100, 103):
            _seed_detection(
                conn,
                detection_id=i,
                species="Parus_major",
                bbox_review="correct",
                is_favorite=1,
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 5},
            max_total=None,
            rng_seed=42,
        )
        # 3 favorites available + 2 non-favorites filling up to cap=5.
        assert len(selection.detection_ids) == 5
        assert selection.favorites_per_species == {"Parus_major": 3}
        # The three favorites (ids 100, 101, 102) are all in the
        # selection — they cannot have been dropped for non-favorites.
        assert 100 in selection.detection_ids
        assert 101 in selection.detection_ids
        assert 102 in selection.detection_ids

    def test_cap_smaller_than_favorite_count_still_includes_favorites_only(self):
        """Cap=2 with 3 favorites available → 2 random favorites, no
        non-favorites. Non-favorites may not sneak in when favorites
        alone already fill the cap."""
        conn = _schema()
        for i in range(10):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct",
                is_favorite=0,
            )
        for i in range(100, 103):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct",
                is_favorite=1,
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 2},
            max_total=None,
            rng_seed=42,
        )
        # All 2 picked are favorites (ids 100-102); no non-favorites.
        assert len(selection.detection_ids) == 2
        assert all(i in (100, 101, 102) for i in selection.detection_ids)
        assert selection.favorites_per_species == {"Parus_major": 2}

    def test_no_favorites_falls_back_to_plain_random(self):
        """When there are no favorites, behaviour matches the old
        random-sample-within-bucket policy. This guards against a
        regression where the new code path might accidentally skip
        buckets with zero favorites."""
        conn = _schema()
        for i in range(10):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct",
                is_favorite=0,
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 5},
            max_total=None,
            rng_seed=42,
        )
        assert len(selection.detection_ids) == 5
        assert selection.favorites_per_species == {"Parus_major": 0}

    def test_max_total_keeps_all_favorites_even_when_clipping(self):
        """max_total < sum(per-species caps) must NOT drop favorites.
        Favorites are kept; non-favorites are the ones that get
        randomly dropped."""
        conn = _schema()
        # 10 non-favorite Parus
        for i in range(10):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct",
                is_favorite=0,
            )
        # 10 non-favorite Cyanistes
        for i in range(50, 60):
            _seed_detection(
                conn, detection_id=i, species="Cyanistes_caeruleus",
                bbox_review="correct", is_favorite=0,
            )
        # 2 Parus favorites, 2 Cyanistes favorites
        for i in range(100, 102):
            _seed_detection(
                conn, detection_id=i, species="Parus_major",
                bbox_review="correct", is_favorite=1,
            )
        for i in range(200, 202):
            _seed_detection(
                conn, detection_id=i, species="Cyanistes_caeruleus",
                bbox_review="correct", is_favorite=1,
            )
        # Per-species cap would pick up to 10+10=20 rows; max_total=6
        # forces a clip. All 4 favorites MUST survive.
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 10, "Cyanistes_caeruleus": 10},
            max_total=6,
            rng_seed=42,
        )
        assert len(selection.detection_ids) == 6
        ids = set(selection.detection_ids)
        for fav in (100, 101, 200, 201):
            assert fav in ids, f"favorite {fav} was clipped, must survive"
        # Exactly 2 non-favorites filled the remaining 2 slots.
        non_fav_count = sum(
            1 for i in selection.detection_ids if i not in {100, 101, 200, 201}
        )
        assert non_fav_count == 2

    def test_max_total_smaller_than_favorite_count_drops_even_favorites(self):
        """Edge case: more favorites than max_total. Favorites are
        the pool to pick from, randomly shuffled — even favorites
        can drop when there is simply not enough room."""
        conn = _schema()
        for i in range(100, 110):
            _seed_detection(
                conn, detection_id=i, species="Parus_major",
                bbox_review="correct", is_favorite=1,
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 10},
            max_total=3,
            rng_seed=42,
        )
        assert len(selection.detection_ids) == 3
        # All 3 are from the favorite pool (ids 100-109).
        assert all(100 <= i < 110 for i in selection.detection_ids)
        assert selection.favorites_per_species == {"Parus_major": 3}

    def test_per_species_counts_include_favorite_rows(self):
        """favorites_per_species must never exceed per_species_counts —
        favorites are a subset of total picks."""
        conn = _schema()
        for i in range(5):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct",
                is_favorite=0,
            )
        for i in range(100, 102):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct",
                is_favorite=1,
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 5},
            max_total=None,
            rng_seed=42,
        )
        total = selection.per_species_counts.get("Parus_major", 0)
        favs = selection.favorites_per_species.get("Parus_major", 0)
        assert favs <= total
        assert total == 5
        assert favs == 2  # both favorites picked, 3 non-favorites filled up


class TestSelectionSamplingAndLimits:
    def test_per_species_cap_respected(self):
        conn = _schema()
        for i in range(100):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct"
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 25},
            max_total=None,
            rng_seed=42,
        )
        assert len(selection.detection_ids) == 25
        assert selection.per_species_counts == {"Parus_major": 25}

    def test_total_cap_overrides_sum_of_per_species(self):
        conn = _schema()
        for i in range(50):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct"
            )
        for i in range(50, 100):
            _seed_detection(
                conn, detection_id=i, species="Cyanistes_caeruleus", bbox_review="correct"
            )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 30, "Cyanistes_caeruleus": 30},
            max_total=40,
            rng_seed=42,
        )
        assert len(selection.detection_ids) == 40
        assert sum(selection.per_species_counts.values()) == 40

    def test_species_not_in_limits_is_ignored(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review="correct"
        )
        selection = select_export_batch(
            conn,
            species_limits={"Parus_major": 50},
            max_total=None,
            rng_seed=42,
        )
        assert selection.detection_ids == [1]
        assert "Pica_pica" not in selection.per_species_counts

    def test_empty_species_limits_means_no_species_filter(self):
        """At the service layer, ``species_limits={}`` means "no species
        filter" and falls back to ``max_per_species`` for every
        eligible species. The blueprint is responsible for treating
        an empty selection from the UI as "no export" — not the
        service. This split keeps the service reusable for callers
        that want "dump everything available"."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        _seed_detection(conn, detection_id=2, species="Pica_pica", bbox_review="correct")
        selection = select_export_batch(
            conn,
            species_limits={},
            max_total=None,
        )
        assert sorted(selection.detection_ids) == [1, 2]
        assert selection.per_species_counts == {"Parus_major": 1, "Pica_pica": 1}


class TestMarkExported:
    def test_mark_exported_inserts_new_rows(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        _seed_detection(conn, detection_id=2, species="Parus_major", bbox_review="correct")
        mark_exported(conn, [1, 2], "batch_1")
        count = conn.execute(
            "SELECT COUNT(*) FROM training_exports WHERE export_status='exported'"
        ).fetchone()[0]
        assert count == 2

    def test_mark_exported_upgrades_pending_to_exported(self):
        """Auto-opt-in writes 'pending'; a later ZIP flip must promote
        that row to 'exported', not create a duplicate."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        mark_pending(conn, [1], "auto_batch")
        mark_exported(conn, [1], "later_manual_batch")

        rows = conn.execute(
            "SELECT batch_id, export_status FROM training_exports WHERE detection_id=1"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["export_status"] == "exported"
        assert rows[0]["batch_id"] == "later_manual_batch"


class TestDefaults:
    def test_defaults_are_sensible(self):
        """Guard against accidental default changes — the UI contracts
        on these numbers for pre-filling the modal inputs."""
        assert DEFAULT_MAX_PER_SPECIES == 50
        assert DEFAULT_MAX_TOTAL == 500


def test_build_batch_id_is_timestamp_prefixed():
    bid = build_batch_id("test")
    assert bid.startswith("batch_")
    assert bid.endswith("_test")

    bid2 = build_batch_id()
    assert bid2.startswith("batch_")
    assert not bid2.endswith("_")


class TestAutoOptInHelper:
    """The helper is called from every approval handler. It must be
    a cheap no-op when the setting is off and correctly stamp the
    pool when the setting is on.

    Because approval handlers are scattered (event-approve,
    per-detection approve, quick-species, event-resolve), a bug here
    would silently drop training data across all flows at once.
    """

    def test_noop_when_setting_off(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        result = auto_opt_in_if_enabled(
            conn, [1], app_config={"TRAINING_EXPORT_AUTO_OPT_IN": False}
        )
        assert result == 0
        count = conn.execute("SELECT COUNT(*) FROM training_exports").fetchone()[0]
        assert count == 0

    def test_noop_when_setting_missing(self):
        """An absent key must not accidentally activate the feature
        (e.g. on older configs that were migrated without the key)."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        result = auto_opt_in_if_enabled(conn, [1], app_config={})
        assert result == 0

    def test_writes_pending_when_setting_on(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        _seed_detection(conn, detection_id=2, species="Parus_major", bbox_review="correct")
        result = auto_opt_in_if_enabled(
            conn, [1, 2], app_config={"TRAINING_EXPORT_AUTO_OPT_IN": True}
        )
        assert result == 2
        rows = conn.execute(
            "SELECT detection_id, export_status FROM training_exports"
        ).fetchall()
        assert sorted(r["detection_id"] for r in rows) == [1, 2]
        assert all(r["export_status"] == "pending" for r in rows)

    def test_empty_detection_ids_does_nothing(self):
        """Guard against an over-eager caller that passes an empty
        list — we must not create a batch_id row with no linked
        detections."""
        conn = _schema()
        result = auto_opt_in_if_enabled(
            conn, [], app_config={"TRAINING_EXPORT_AUTO_OPT_IN": True}
        )
        assert result == 0

    def test_source_tag_appears_in_batch_id(self):
        """Different approval flows use different source tags so the
        training_exports table reveals which path created each row."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        auto_opt_in_if_enabled(
            conn,
            [1],
            app_config={"TRAINING_EXPORT_AUTO_OPT_IN": True},
            source_tag="quick_species",
        )
        row = conn.execute("SELECT batch_id FROM training_exports").fetchone()
        assert row["batch_id"].endswith("_quick_species")

    def test_does_not_downgrade_already_exported(self):
        """The pool row is already 'exported'; re-approving it via
        auto-opt-in must not downgrade it back to 'pending' (which
        would cause duplicate shipping to the dev)."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        mark_exported(conn, [1], "old_batch")

        auto_opt_in_if_enabled(
            conn, [1], app_config={"TRAINING_EXPORT_AUTO_OPT_IN": True}
        )

        row = conn.execute(
            "SELECT batch_id, export_status FROM training_exports WHERE detection_id=1"
        ).fetchone()
        assert row["export_status"] == "exported"
        assert row["batch_id"] == "old_batch"


class TestFilterEligibleForPool:
    """The gallery-edit "Add to training" batch button sends a raw
    list of detection_ids and expects a 3-bucket breakdown. This is
    the single most important helper for that UX, because the UI
    shows the three counts directly to the operator.
    """

    def test_eligible_row_lands_in_eligible_bucket(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == [1]
        assert result["ineligible"] == []
        assert result["already_in_pool"] == []

    def test_row_without_species_override_is_ineligible(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species=None, bbox_review="correct")
        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == []
        assert result["ineligible"] == [1]

    def test_row_with_bbox_wrong_is_ineligible(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="wrong")
        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == []
        assert result["ineligible"] == [1]

    def test_row_without_bbox_review_is_ineligible(self):
        """Bulk-relabel without bbox confirmation is the classic case
        — species override set, bbox_review NULL. Must not be
        promoted to pool without an explicit bbox confirmation."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review=None)
        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == []
        assert result["ineligible"] == [1]

    def test_already_pending_row_goes_to_already_in_pool(self):
        """Re-adding an already-pending row must not duplicate it.
        It lands in the already_in_pool bucket for UI messaging."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        mark_pending(conn, [1], "old_auto")

        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == []
        assert result["already_in_pool"] == [1]

    def test_already_exported_row_goes_to_already_in_pool(self):
        """The UI must not let the user re-add an already-shipped
        detection (which would flip exported back to pending-then-
        exported-again and cost the dev a duplicate sample)."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        mark_exported(conn, [1], "old_batch")

        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == []
        assert result["already_in_pool"] == [1]

    def test_unknown_detection_id_is_ineligible(self):
        """A stale or deleted detection id from the browser must not
        crash the endpoint — it just counts as ineligible."""
        conn = _schema()
        result = filter_eligible_for_pool(conn, [999])
        assert result["eligible"] == []
        assert result["ineligible"] == [999]

    def test_mixed_input_is_partitioned_cleanly(self):
        """Typical gallery-edit case: the operator selects 5 tiles, of
        which 2 are eligible, 2 are ineligible (bulk-relabeled, no
        bbox review), and 1 is already in the pool.
        """
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        _seed_detection(conn, detection_id=2, species="Parus_major", bbox_review="correct")
        _seed_detection(conn, detection_id=3, species="Parus_major", bbox_review=None)
        _seed_detection(conn, detection_id=4, species="Parus_major", bbox_review="wrong")
        _seed_detection(conn, detection_id=5, species="Parus_major", bbox_review="correct")
        mark_pending(conn, [5], "prior_auto")

        result = filter_eligible_for_pool(conn, [1, 2, 3, 4, 5])
        assert sorted(result["eligible"]) == [1, 2]
        assert sorted(result["ineligible"]) == [3, 4]
        assert result["already_in_pool"] == [5]

    def test_duplicates_in_input_are_deduped(self):
        """Browser bug or double-click that sends [1,1,1] must not
        produce 3 entries in the eligible bucket."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="correct")
        result = filter_eligible_for_pool(conn, [1, 1, 1])
        assert result["eligible"] == [1]

    def test_empty_input_returns_empty_buckets(self):
        conn = _schema()
        result = filter_eligible_for_pool(conn, [])
        assert result == {"eligible": [], "ineligible": [], "already_in_pool": []}


class TestAutoConfirmBboxAndFrameIntegrity:
    """The gallery-edit "Confirm & Add to Training" button treats the
    caller's click as an active bbox-review act for NULL-bbox rows.
    But it MUST preserve frame-level integrity: a frame with any
    ambiguous sibling bbox is off-limits, even if the selected
    detection is individually clean.

    These tests are the difference between a training set that is
    safe to ship and one that contains partially-reviewed frames.
    If any of these break, the dev starts receiving ambiguous OD
    training data — which is worse than receiving less data.
    """

    def test_null_bbox_is_eligible_with_auto_confirm(self):
        """Single detection on a frame, species set, bbox NULL.
        With auto_confirm_bbox=True, the click confirms the bbox and
        the row is eligible."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review=None)
        result = filter_eligible_for_pool(
            conn, [1], auto_confirm_bbox=True
        )
        assert result["eligible"] == [1]
        assert result["ineligible"] == []

    def test_null_bbox_still_ineligible_without_auto_confirm(self):
        """Default (review-queue path) keeps strict behaviour —
        NULL bbox blocks eligibility."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review=None)
        result = filter_eligible_for_pool(conn, [1])
        assert result["eligible"] == []
        assert result["ineligible"] == [1]

    def test_wrong_bbox_ineligible_even_with_auto_confirm(self):
        """Explicit 'wrong' bbox stays blocked — the caller should
        never flip a prior human 'wrong' call to 'correct'."""
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review="wrong")
        result = filter_eligible_for_pool(
            conn, [1], auto_confirm_bbox=True
        )
        assert result["eligible"] == []
        assert result["ineligible"] == [1]

    def test_frame_with_unreviewed_sibling_blocks_everything(self):
        """Two detections on one frame. Detection 1 is clean
        (bbox='correct'). Detection 2 has bbox=NULL and was NOT
        submitted in the request. The frame is ambiguous, so the
        submitted detection 1 must be rejected too."""
        conn = _schema()
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename="multibox.jpg",
        )
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review=None,
            image_filename="multibox.jpg",
        )
        result = filter_eligible_for_pool(
            conn, [1], auto_confirm_bbox=True
        )
        assert result["eligible"] == []
        assert result["ineligible"] == [1]

    def test_frame_with_wrong_sibling_blocks_everything(self):
        """A sibling bbox='wrong' is a hard block — training on that
        frame would use a deliberately-bad bbox."""
        conn = _schema()
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename="multibox.jpg",
        )
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review="wrong",
            image_filename="multibox.jpg",
        )
        result = filter_eligible_for_pool(
            conn, [1], auto_confirm_bbox=True
        )
        assert result["eligible"] == []

    def test_frame_becomes_clean_when_all_blockers_are_submitted(self):
        """Two detections on one frame, both with bbox=NULL + species
        set. When the caller submits BOTH, the click confirms all
        siblings together, and the frame becomes clean."""
        conn = _schema()
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review=None,
            image_filename="multibox.jpg",
        )
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review=None,
            image_filename="multibox.jpg",
        )
        result = filter_eligible_for_pool(
            conn, [1, 2], auto_confirm_bbox=True
        )
        assert sorted(result["eligible"]) == [1, 2]

    def test_frame_blocked_when_sibling_without_species_submitted(self):
        """If the caller submits a row that has NULL bbox AND no
        species override, auto_confirm cannot resolve it — and the
        frame stays blocked for the clean sibling too."""
        conn = _schema()
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review=None,
            image_filename="multibox.jpg",
        )
        _seed_detection(
            conn, detection_id=2, species=None, bbox_review=None,
            image_filename="multibox.jpg",
        )
        result = filter_eligible_for_pool(
            conn, [1, 2], auto_confirm_bbox=True
        )
        # Detection 2 has no species — can't be promoted. Frame
        # integrity fails, so detection 1 is blocked too.
        assert result["eligible"] == []
        assert 2 in result["ineligible"]

    def test_single_detection_frame_is_unaffected(self):
        """A frame with only one detection is trivially clean under
        auto_confirm_bbox — no siblings to worry about."""
        conn = _schema()
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review=None,
            image_filename="solo.jpg",
        )
        result = filter_eligible_for_pool(
            conn, [1], auto_confirm_bbox=True
        )
        assert result["eligible"] == [1]


class TestConfirmBboxAndMarkPending:
    """The helper the gallery-edit endpoint calls once
    filter_eligible_for_pool has approved the rows. It must do two
    things atomically: stamp bbox_review='correct' on NULL rows AND
    mark them pending in the pool.
    """

    def test_sets_bbox_correct_on_null_rows(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review=None)
        confirm_bbox_and_mark_pending(conn, [1], "test_batch")

        row = conn.execute(
            "SELECT manual_bbox_review, bbox_reviewed_at FROM detections WHERE detection_id=1"
        ).fetchone()
        assert row["manual_bbox_review"] == "correct"
        assert row["bbox_reviewed_at"] is not None

    def test_leaves_already_correct_bbox_untouched(self):
        """An already-'correct' row keeps its original reviewed_at
        timestamp — we don't want to overwrite history."""
        conn = _schema()
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct"
        )
        original_ts = conn.execute(
            "SELECT bbox_reviewed_at FROM detections WHERE detection_id=1"
        ).fetchone()["bbox_reviewed_at"]

        confirm_bbox_and_mark_pending(conn, [1], "test_batch")

        new_ts = conn.execute(
            "SELECT bbox_reviewed_at FROM detections WHERE detection_id=1"
        ).fetchone()["bbox_reviewed_at"]
        assert new_ts == original_ts  # unchanged

    def test_creates_pool_entry(self):
        conn = _schema()
        _seed_detection(conn, detection_id=1, species="Parus_major", bbox_review=None)
        added = confirm_bbox_and_mark_pending(conn, [1], "test_batch")
        assert added == 1

        row = conn.execute(
            "SELECT export_status FROM training_exports WHERE detection_id=1"
        ).fetchone()
        assert row["export_status"] == "pending"


def _reset_path_manager_singleton():
    """PathManager is a singleton cached at module level. Tests that
    point it at a fresh tmp_path must reset between tests or the
    second one keeps seeing the first test's tmp dir."""
    import utils.path_manager as pm_module

    pm_module._instance = None


def _write_fake_original(tmp_path, filename: str) -> None:
    """Write a trivial JPEG byte-stream so path_core.get_original_path
    resolves to an existing file when the export runs. Resets the
    PathManager singleton so the tmp_path swap between tests is seen.
    """
    from pathlib import Path

    from utils.path_manager import PathManager

    _reset_path_manager_singleton()
    pm = PathManager(str(tmp_path))
    date_str = filename[:8]
    date_folder = pm.get_date_folder(date_str)
    dest_dir = Path(tmp_path) / "originals" / date_folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / filename).write_bytes(b"\xff\xd8\xff\xe0fake")


def _build_selection_from_ids(ids: list[int]):
    """Helper shared across test classes that need an ExportSelection
    without the full select_export_batch pipeline."""
    from web.services.training_export_service import (
        ExportSelection,
        build_batch_id,
    )

    return ExportSelection(
        batch_id=build_batch_id("test"),
        detection_ids=ids,
    )


def _unpack_zip(buffer):
    """Unpack a ZIP buffer into {names, manifest, csv_rows} for the
    export tests to assert on without re-writing boilerplate."""
    import csv as _csv
    import io as _io
    import json as _json
    import zipfile as _zip

    with _zip.ZipFile(buffer) as zf:
        names = zf.namelist()
        manifest = _json.loads(zf.read("manifest.json").decode("utf-8"))
        csv_rows = list(
            _csv.DictReader(_io.StringIO(zf.read("annotations.csv").decode("utf-8")))
        )
    return {"names": names, "manifest": manifest, "csv_rows": csv_rows}


class TestZipShapeForMultiDetectionFrames:
    """The dev's training pipeline expects: one file per frame in
    images/, and one row per detection in annotations.csv. Detections
    on the same frame share a uuid.

    Previous bug (caught by user 2026-04-23): a fresh uuid was minted
    per detection, so multi-bbox frames were written to the ZIP
    multiple times under different names, making it impossible for
    the dev to associate bboxes to a single training sample. This
    test class locks in the fix so the bug cannot return silently.
    """

    def _build_selection_from_ids(self, ids: list[int]) -> "ExportSelection":
        from web.services.training_export_service import (
            ExportSelection,
            build_batch_id,
        )

        return ExportSelection(
            batch_id=build_batch_id("test"),
            detection_ids=ids,
        )

    def _unpack_zip(self, buffer):
        import csv as _csv
        import io as _io
        import json as _json
        import zipfile as _zip

        with _zip.ZipFile(buffer) as zf:
            names = zf.namelist()
            manifest = _json.loads(zf.read("manifest.json").decode("utf-8"))
            csv_rows = list(
                _csv.DictReader(_io.StringIO(zf.read("annotations.csv").decode("utf-8")))
            )
        return {"names": names, "manifest": manifest, "csv_rows": csv_rows}

    def _write_fake_original(self, tmp_path, filename: str) -> None:
        """Write a trivial JPEG byte-stream so path_core.get_original_path
        resolves to an existing file when the export runs."""
        from pathlib import Path

        from utils.path_manager import PathManager

        _reset_path_manager_singleton()
        pm = PathManager(str(tmp_path))
        date_str = filename[:8]  # YYYYMMDD prefix
        date_folder = pm.get_date_folder(date_str)
        dest_dir = Path(tmp_path) / "originals" / date_folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Minimal valid JPEG header is fine — we never decode it.
        (dest_dir / filename).write_bytes(b"\xff\xd8\xff\xe0fake")

    def test_two_detections_on_one_frame_share_uuid(self, tmp_path):
        from web.services.training_export_service import stream_export_zip

        conn = _schema()
        filename = "20260423_100000_frame.jpg"
        _write_fake_original(tmp_path, filename)
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename=filename,
        )
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review="correct",
            image_filename=filename,
        )

        selection = _build_selection_from_ids([1, 2])
        buf, _persist = stream_export_zip(
            conn, selection, output_dir=str(tmp_path)
        )
        unpacked = _unpack_zip(buf)

        # Exactly ONE image file in the zip (not two).
        image_names = [n for n in unpacked["names"] if n.startswith("images/")]
        assert len(image_names) == 1, image_names

        # TWO rows in annotations.csv.
        assert len(unpacked["csv_rows"]) == 2
        # Both rows share the same uuid.
        uuids = {row["uuid"] for row in unpacked["csv_rows"]}
        assert len(uuids) == 1
        # The uuid in the CSV matches the uuid in the image filename.
        the_uuid = uuids.pop()
        assert any(the_uuid in name for name in image_names)

        # Manifest reports both counts separately so the dev can tell.
        assert unpacked["manifest"]["rows_written"] == 2
        assert unpacked["manifest"]["images_written"] == 1

    def test_three_detections_across_two_frames(self, tmp_path):
        """A mixed selection: 2 dets on frame A, 1 det on frame B.
        Expected: 2 images in ZIP, 3 rows in CSV, 2 distinct uuids.
        """
        from web.services.training_export_service import stream_export_zip

        conn = _schema()
        frame_a = "20260423_100000_a.jpg"
        frame_b = "20260423_110000_b.jpg"
        _write_fake_original(tmp_path, frame_a)
        _write_fake_original(tmp_path, frame_b)

        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename=frame_a,
        )
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review="correct",
            image_filename=frame_a,
        )
        _seed_detection(
            conn, detection_id=3, species="Cyanistes_caeruleus", bbox_review="correct",
            image_filename=frame_b,
        )

        selection = _build_selection_from_ids([1, 2, 3])
        buf, _persist = stream_export_zip(
            conn, selection, output_dir=str(tmp_path)
        )
        unpacked = _unpack_zip(buf)

        image_names = [n for n in unpacked["names"] if n.startswith("images/")]
        assert len(image_names) == 2
        assert len(unpacked["csv_rows"]) == 3
        uuids = {row["uuid"] for row in unpacked["csv_rows"]}
        assert len(uuids) == 2  # two distinct frames, two uuids

        # Frame A rows share a uuid; frame B row has its own.
        frame_a_uuids = {
            r["uuid"] for r in unpacked["csv_rows"]
            if r["species"] in ("Parus_major", "Pica_pica")
        }
        assert len(frame_a_uuids) == 1
        assert unpacked["manifest"]["rows_written"] == 3
        assert unpacked["manifest"]["images_written"] == 2


class TestFrameIntegrityAtExport:
    """The export must never ship a partially-labelled frame to the
    dev. If a selected detection's frame has any ambiguous sibling
    (bbox NULL / wrong / missing species), the whole frame is
    dropped; otherwise any eligible sibling on the frame is pulled
    in and marked exported alongside the original selection.

    This is the bug the user caught 2026-04-23: filtering by Parus
    still shipped a Columba bbox implicitly (via the shared frame)
    without a CSV row for it, teaching the OD trainer that the
    unlabelled vogel is background.
    """

    def test_clean_frame_exports_all_siblings(self, tmp_path):
        """User selects a Parus detection on a frame where a Columba
        sibling is ALSO Option-A-strict eligible. Both must ship, so
        the frame is fully labelled."""
        from web.services.training_export_service import stream_export_zip

        conn = _schema()
        filename = "20260423_100000_multi.jpg"
        _write_fake_original(tmp_path, filename)
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename=filename,
        )
        _seed_detection(
            conn, detection_id=2, species="Columba_palumbus", bbox_review="correct",
            image_filename=filename,
        )

        selection = _build_selection_from_ids([1])  # only Parus selected
        buf, persist = stream_export_zip(
            conn, selection, output_dir=str(tmp_path)
        )
        unpacked = _unpack_zip(buf)

        # Both species appear in the CSV, sharing the same uuid.
        species_set = {row["species"] for row in unpacked["csv_rows"]}
        assert species_set == {"Parus_major", "Columba_palumbus"}
        assert len({row["uuid"] for row in unpacked["csv_rows"]}) == 1

        # The Columba was pulled in via frame-integrity.
        assert 2 in persist["pulled_in_siblings"]
        assert sorted(persist["exported_ids"]) == [1, 2]
        assert persist["dropped_ids"] == []

    def test_ambiguous_sibling_drops_whole_frame(self, tmp_path):
        """User selects a Parus detection on a frame where the
        Columba sibling has bbox=NULL (never reviewed). The Columba
        cannot be shipped, so the whole frame is dropped — otherwise
        the Parus bbox would go out as a partial label."""
        from web.services.training_export_service import stream_export_zip

        conn = _schema()
        filename = "20260423_100000_ambig.jpg"
        _write_fake_original(tmp_path, filename)
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename=filename,
        )
        _seed_detection(
            conn, detection_id=2, species="Columba_palumbus", bbox_review=None,
            image_filename=filename,
        )

        selection = _build_selection_from_ids([1])
        buf, persist = stream_export_zip(
            conn, selection, output_dir=str(tmp_path)
        )
        unpacked = _unpack_zip(buf)

        # Nothing shipped.
        assert len(unpacked["csv_rows"]) == 0
        image_names = [n for n in unpacked["names"] if n.startswith("images/")]
        assert len(image_names) == 0

        # The Parus was dropped from the batch.
        assert persist["exported_ids"] == []
        assert persist["dropped_ids"] == [1]
        assert persist["pulled_in_siblings"] == []

    def test_sibling_with_bbox_wrong_drops_frame(self, tmp_path):
        """A sibling marked bbox='wrong' is a hard block — the dev
        must never see a frame where a bbox has been explicitly
        rejected."""
        from web.services.training_export_service import stream_export_zip

        conn = _schema()
        filename = "20260423_100000_wrongbbox.jpg"
        _write_fake_original(tmp_path, filename)
        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename=filename,
        )
        _seed_detection(
            conn, detection_id=2, species="Pica_pica", bbox_review="wrong",
            image_filename=filename,
        )

        selection = _build_selection_from_ids([1])
        buf, persist = stream_export_zip(
            conn, selection, output_dir=str(tmp_path)
        )
        unpacked = _unpack_zip(buf)
        assert len(unpacked["csv_rows"]) == 0
        assert persist["dropped_ids"] == [1]

    def test_two_clean_frames_ship_independently(self, tmp_path):
        """Two separate frames, each clean. Only one detection per
        frame is selected, but both frames ship whole."""
        from web.services.training_export_service import stream_export_zip

        conn = _schema()
        frame_a = "20260423_100000_a.jpg"
        frame_b = "20260423_110000_b.jpg"
        _write_fake_original(tmp_path, frame_a)
        _write_fake_original(tmp_path, frame_b)

        _seed_detection(
            conn, detection_id=1, species="Parus_major", bbox_review="correct",
            image_filename=frame_a,
        )
        _seed_detection(
            conn, detection_id=2, species="Columba_palumbus", bbox_review="correct",
            image_filename=frame_a,
        )
        _seed_detection(
            conn, detection_id=3, species="Cyanistes_caeruleus", bbox_review="correct",
            image_filename=frame_b,
        )

        selection = _build_selection_from_ids([1, 3])  # one per frame
        buf, persist = stream_export_zip(
            conn, selection, output_dir=str(tmp_path)
        )
        unpacked = _unpack_zip(buf)

        # Frame A: both Parus + Columba ship (Columba pulled in).
        # Frame B: single Cyanistes ships solo.
        assert len(unpacked["csv_rows"]) == 3
        image_names = [n for n in unpacked["names"] if n.startswith("images/")]
        assert len(image_names) == 2
        assert 2 in persist["pulled_in_siblings"]
        assert sorted(persist["exported_ids"]) == [1, 2, 3]


class TestIntegrationSelectAndMark:
    """End-to-end check: select → mark_exported → re-select yields
    only the remaining pool."""

    def test_round_trip(self):
        conn = _schema()
        for i in range(10):
            _seed_detection(
                conn, detection_id=i, species="Parus_major", bbox_review="correct"
            )
        first = select_export_batch(
            conn,
            species_limits={"Parus_major": 5},
            max_total=None,
            rng_seed=123,
        )
        assert len(first.detection_ids) == 5
        mark_exported(conn, first.detection_ids, first.batch_id)

        second = select_export_batch(
            conn,
            species_limits={"Parus_major": 10},
            max_total=None,
            rng_seed=456,
        )
        # The 5 remaining rows are all we can get, regardless of cap.
        assert len(second.detection_ids) == 5
        # No overlap with the first batch.
        assert not set(first.detection_ids) & set(second.detection_ids)
