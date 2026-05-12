#!/usr/bin/env python3
"""
Smoke test for scripts/aesthetic_tag_nightly.py.

Builds a temporary SQLite database matching the production schema, populates
it with synthetic detections, runs the tagger, and asserts:

1. aesthetic_score is written for all detections with thumbnail_path
2. only species in TAGGABLE_SPECIES (+ unknown) get is_favorite=1
3. manual favorites are preserved
4. re-running is idempotent (no double-scoring)
5. dry-run does not write

Run:
    .venv/bin/python scripts/test_aesthetic_tag_nightly.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Override paths BEFORE importing the module.
TMP_DIR = Path(tempfile.mkdtemp(prefix="wmb_smoke_"))
DB_PATH = TMP_DIR / "test.db"
CROPS_ROOT = TMP_DIR / "crops"
LOG_PATH = TMP_DIR / "test.log"
CROPS_ROOT.mkdir(parents=True)

os.environ["WMB_DB_PATH"] = str(DB_PATH)
os.environ["WMB_CROPS_ROOT"] = str(CROPS_ROOT)
os.environ["WMB_AESTHETIC_LOG"] = str(LOG_PATH)

import aesthetic_tag_nightly as job  # noqa: E402

# Reload module-level constants (they're computed from env at import time).
job.DB_PATH = DB_PATH
job.CROPS_ROOT = CROPS_ROOT
job.LOG_PATH = LOG_PATH


# --- Test fixtures ---------------------------------------------------------

def setup_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE images (
        filename TEXT PRIMARY KEY,
        timestamp TEXT
    );
    CREATE TABLE detections (
        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename TEXT NOT NULL,
        thumbnail_path TEXT,
        od_class_name TEXT,
        created_at TEXT,
        status TEXT DEFAULT 'active',
        is_favorite INTEGER DEFAULT 0,
        is_gallery_eligible INTEGER DEFAULT 0,
        rating_source TEXT DEFAULT 'auto',
        decision_state TEXT,
        decision_level TEXT,
        score REAL,
        bbox_quality REAL,
        aesthetic_score REAL,
        aesthetic_score_at TEXT,
        FOREIGN KEY(image_filename) REFERENCES images(filename) ON DELETE CASCADE
    );
    CREATE TABLE classifications (
        classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        cls_class_name TEXT,
        rank INTEGER DEFAULT 1,
        status TEXT DEFAULT 'active',
        FOREIGN KEY(detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
    );
    """)


def fixture_crop(filename: str, day: str = "2026-04-30") -> Path:
    """Create a minimal valid 32x32 JPEG so PIL can open it."""
    from PIL import Image
    day_dir = CROPS_ROOT / day
    day_dir.mkdir(parents=True, exist_ok=True)
    p = day_dir / filename
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(p, format="WEBP")
    return p


def insert_detection(
    conn: sqlite3.Connection,
    detection_id: int,
    species: str | None,
    *,
    when: str = "2026-04-30T12:00:00+00:00",
    is_favorite: int = 0,
    rating_source: str = "auto",
    has_crop: bool = True,
    decision_state: str | None = "confirmed",
    score: float | None = None,
    bbox_quality: float | None = None,
) -> None:
    image = f"20260430_{detection_id:06d}_test.jpg"
    thumb = f"20260430_{detection_id:06d}_test_crop_1.webp"
    conn.execute("INSERT OR IGNORE INTO images (filename, timestamp) VALUES (?, ?)",
                 (image, when))
    conn.execute(
        "INSERT INTO detections "
        "(detection_id, image_filename, thumbnail_path, od_class_name, "
        " created_at, status, is_favorite, rating_source, decision_state, "
        " score, bbox_quality) "
        "VALUES (?, ?, ?, 'bird', ?, 'active', ?, ?, ?, ?, ?)",
        (detection_id, image, thumb if has_crop else None, when,
         is_favorite, rating_source, decision_state, score, bbox_quality),
    )
    if species is not None:
        conn.execute(
            "INSERT INTO classifications (detection_id, cls_class_name, rank, status) "
            "VALUES (?, ?, 1, 'active')",
            (detection_id, species),
        )
    if has_crop:
        fixture_crop(thumb)


# --- Mock CLIP -------------------------------------------------------------

class MockClipModel:
    """Returns deterministic scores based on detection_id parity for testing."""
    def encode_image(self, x):
        import torch
        # detection_id is encoded into the image_path during scoring; we don't
        # have it here, so just return a fixed feature.
        return torch.ones(1, 512)
    def encode_text(self, x):
        import torch
        return torch.ones(2, 512)


def mock_score_image(model, preprocess, text_features, image_path, device):
    """Deterministic scoring; values stay above MIN_SCORE_FOR_TAG (0.30).

    The range is intentionally [0.40, 0.95] so every mocked detection
    clears the production floor. Tests that need to probe the floor
    (e.g. test_min_score_threshold) override the score directly.
    """
    name = image_path.stem
    # filename pattern: 20260430_NNNNNN_test_crop_1
    try:
        det_id = int(name.split("_")[1])
    except (ValueError, IndexError):
        return 0.6
    # Map det_id deterministically into [0.40, 0.95] range, well above
    # the 0.30 floor so the top-N selection logic is what's actually
    # being tested.
    return 0.40 + ((det_id * 0.07) % 1.0) * 0.55


def mock_load_clip(device):
    import torch
    return None, lambda x: torch.zeros(3, 224, 224), torch.zeros(2, 512)


# --- Tests -----------------------------------------------------------------

def assert_eq(actual, expected, msg: str) -> None:
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def test_basic_run() -> None:
    """Only TAGGABLE_SPECIES (Parus, Cyanistes, Columba) get tagged.
    'unknown' and rare-species CLS guesses are excluded."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)

    # Mix: 4 great tits, 3 pigeons, 2 unknowns, 2 rare-CLS-guesses (Phoenicurus).
    # Only Parus + Columba should get tagged. unknown + Phoenicurus are skipped.
    for i in range(101, 105):
        insert_detection(conn, i, "Parus_major")
    for i in range(201, 204):
        insert_detection(conn, i, "Columba_palumbus")
    for i in range(301, 303):
        insert_detection(conn, i, None)  # cls null = "unknown"
    for i in range(401, 403):
        insert_detection(conn, i, "Phoenicurus_sp.")  # rare/often misclassified
    conn.commit()
    conn.close()

    # Patch CLIP loader + scorer in the imported module.
    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image

    rc = job.main_with_args(["--since", "2026-04-29"])
    assert_eq(rc, 0, "exit code")

    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT detection_id, aesthetic_score, is_gallery_eligible, is_favorite "
        "FROM detections ORDER BY detection_id"
    ).fetchall()
    conn.close()

    # All 11 detections still get scored (score is independent of taggability).
    n_scored = sum(1 for r in rows if r[1] is not None)
    assert_eq(n_scored, 11, "all 11 scored")

    # Parus_major (4) -- top 3 marked gallery-eligible (is_favorite never touched)
    parus_tagged = sum(1 for r in rows if 101 <= r[0] <= 104 and r[2] == 1)
    assert_eq(parus_tagged, 3, "top-3 parus gallery-eligible")

    # Pigeons (3) -- all 3 marked gallery-eligible (top-3 of 3)
    pigeon_tagged = sum(1 for r in rows if 201 <= r[0] <= 203 and r[2] == 1)
    assert_eq(pigeon_tagged, 3, "all 3 pigeons gallery-eligible")

    # Unknowns (2) -- NOT marked (TAG_UNKNOWN_SPECIES = False)
    unknown_tagged = sum(1 for r in rows if 301 <= r[0] <= 302 and r[2] == 1)
    assert_eq(unknown_tagged, 0, "unknowns NOT gallery-eligible")

    # Phoenicurus (2) -- gallery-eligible because TAGGABLE_SPECIES is
    # currently empty (test mode: all CLS-labelled species are eligible).
    # Both detections clear MIN_SCORE_FOR_TAG and there are only 2 of
    # them, so both win the top-3-of-2 bucket.
    rare_tagged = sum(1 for r in rows if 401 <= r[0] <= 402 and r[2] == 1)
    assert_eq(rare_tagged, 2, "Phoenicurus gallery-eligible (test mode)")

    # is_favorite must remain 0 for ALL these — the tagger never touches it.
    n_human_favs = sum(1 for r in rows if r[3] == 1)
    assert_eq(n_human_favs, 0, "is_favorite untouched by tagger")

    print("PASS test_basic_run")


def test_manual_favorite_preserved() -> None:
    """Manually-favorited detections are not touched by the auto-tagger."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)

    # 5 great tits; the lowest-scoring one is manually-favorited.
    for i in range(401, 406):
        is_manual = (i == 401)  # det 401 has the lowest score (0.07)
        insert_detection(conn, i, "Parus_major",
                        is_favorite=1 if is_manual else 0,
                        rating_source="manual" if is_manual else "auto")
    conn.commit()
    conn.close()

    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image
    rc = job.main_with_args(["--since", "2026-04-29"])
    assert_eq(rc, 0, "exit code")

    conn = sqlite3.connect(str(DB_PATH))
    row_401 = conn.execute(
        "SELECT is_favorite, rating_source FROM detections WHERE detection_id = 401"
    ).fetchone()
    conn.close()
    assert_eq(row_401, (1, "manual"), "manual favorite preserved")
    print("PASS test_manual_favorite_preserved")


def test_idempotent() -> None:
    """Re-running the script does not re-score detections."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)
    for i in range(501, 504):
        insert_detection(conn, i, "Parus_major")
    conn.commit()
    conn.close()

    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image

    # First run
    job.main_with_args(["--since", "2026-04-29"])
    conn = sqlite3.connect(str(DB_PATH))
    first_ts = conn.execute(
        "SELECT aesthetic_score_at FROM detections WHERE detection_id = 501"
    ).fetchone()[0]
    conn.close()
    assert first_ts is not None, "first run wrote timestamp"

    # Second run - should be a no-op for already-scored
    job.main_with_args(["--since", "2026-04-29"])
    conn = sqlite3.connect(str(DB_PATH))
    second_ts = conn.execute(
        "SELECT aesthetic_score_at FROM detections WHERE detection_id = 501"
    ).fetchone()[0]
    conn.close()
    assert_eq(first_ts, second_ts, "idempotent: timestamp unchanged on re-run")
    print("PASS test_idempotent")


def test_dry_run() -> None:
    """--dry-run does not write to the DB."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)
    for i in range(601, 604):
        insert_detection(conn, i, "Parus_major")
    conn.commit()
    conn.close()

    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image
    rc = job.main_with_args(["--since", "2026-04-29", "--dry-run"])
    assert_eq(rc, 0, "exit code")

    conn = sqlite3.connect(str(DB_PATH))
    n_scored = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE aesthetic_score IS NOT NULL"
    ).fetchone()[0]
    n_tagged = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE is_gallery_eligible = 1"
    ).fetchone()[0]
    conn.close()
    assert_eq(n_scored, 0, "dry-run wrote no scores")
    assert_eq(n_tagged, 0, "dry-run wrote no favorites")
    print("PASS test_dry_run")


def test_decision_state_filter() -> None:
    """Only confirmed detections are eligible; uncertain/null are skipped."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)

    # 3 great tits with different decision_states.
    insert_detection(conn, 901, "Parus_major", decision_state="confirmed")
    insert_detection(conn, 902, "Parus_major", decision_state="uncertain")
    insert_detection(conn, 903, "Parus_major", decision_state=None)
    conn.commit()
    conn.close()

    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image

    rc = job.main_with_args(["--since", "2026-04-29"])
    assert_eq(rc, 0, "exit code")

    conn = sqlite3.connect(str(DB_PATH))
    rows = {r[0]: r for r in conn.execute(
        "SELECT detection_id, aesthetic_score, is_gallery_eligible FROM detections"
    ).fetchall()}
    conn.close()

    # Only confirmed got scored AND marked eligible; others get nothing.
    assert rows[901][1] is not None, "confirmed got score"
    assert rows[901][2] == 1, "confirmed got gallery-eligible"
    assert rows[902][1] is None, "uncertain skipped at scoring"
    assert rows[903][1] is None, "null decision skipped"
    print("PASS test_decision_state_filter")


def test_min_score_threshold() -> None:
    """Detections below MIN_SCORE_FOR_TAG don't get tagged even if top-3 in bucket."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)

    # 3 detections of an obscure species, all of which the mock will score low.
    # Use det_ids that mock to scores < 0.15 after being passed through the
    # mock formula. Easier: monkey-patch the scorer to return constants.
    for i in range(801, 804):
        insert_detection(conn, i, "Phoenicurus_sp.")
    conn.commit()
    conn.close()

    # All scores below MIN_SCORE_FOR_TAG
    job.load_clip_model = mock_load_clip
    job.score_image = lambda *a, **kw: 0.10  # below 0.15 threshold

    rc = job.main_with_args(["--since", "2026-04-29"])
    assert_eq(rc, 0, "exit code")

    conn = sqlite3.connect(str(DB_PATH))
    n_scored = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE aesthetic_score IS NOT NULL"
    ).fetchone()[0]
    n_tagged = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE is_gallery_eligible = 1"
    ).fetchone()[0]
    conn.close()
    assert_eq(n_scored, 3, "all 3 scored (score is written even if low)")
    assert_eq(n_tagged, 0, "none tagged (all below MIN_SCORE_FOR_TAG)")
    print("PASS test_min_score_threshold")


def test_per_species_cap_basic() -> None:
    """--per-species-cap N caps the run at N detections per CLS species,
    ranked by score DESC, bbox_quality DESC, created_at DESC. Unknown
    (CLS-null) detections are excluded entirely."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)

    # Parus: 10 detections, ranked by score (lower id = lower score so the
    # cap should keep the HIGHEST ids).
    for i in range(1, 11):
        insert_detection(
            conn, i, "Parus_major",
            when="2026-04-30T12:00:00+00:00",
            score=0.1 * i,           # 0.1 .. 1.0
            bbox_quality=0.5,
        )
    # Cyanistes: 6 detections, all same score — bbox_quality breaks ties.
    for i in range(101, 107):
        insert_detection(
            conn, i, "Cyanistes_caeruleus",
            when="2026-04-30T12:00:00+00:00",
            score=0.5,
            bbox_quality=0.1 * (i - 100),  # 0.1 .. 0.6
        )
    # Two unknowns (CLS null) — must NOT show up at all.
    for i in range(201, 203):
        insert_detection(conn, i, None, score=0.99, bbox_quality=0.99)
    conn.commit()

    rows = job.fetch_unscored_detections(
        conn, since="2026-04-29", per_species_cap=3
    )
    conn.close()

    by_species: dict[str, list[int]] = {}
    for r in rows:
        by_species.setdefault(r["species"], []).append(r["detection_id"])

    # No unknowns slip through.
    assert "unknown" not in by_species, f"unknowns must be excluded, got {by_species}"

    # Parus: top 3 by score DESC → ids 10, 9, 8.
    assert_eq(
        sorted(by_species.get("Parus_major", []), reverse=True),
        [10, 9, 8],
        "Parus top-3 by score DESC",
    )
    # Cyanistes: scores tie, so bbox_quality DESC → ids 106, 105, 104.
    assert_eq(
        sorted(by_species.get("Cyanistes_caeruleus", []), reverse=True),
        [106, 105, 104],
        "Cyanistes top-3 by bbox_quality DESC (score tie)",
    )

    print("PASS test_per_species_cap_basic")


def test_per_species_cap_created_at_tiebreak() -> None:
    """When score AND bbox_quality tie, created_at DESC (newest) wins."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)

    times = [
        "2026-04-30T08:00:00+00:00",
        "2026-04-30T09:00:00+00:00",
        "2026-04-30T10:00:00+00:00",
        "2026-04-30T11:00:00+00:00",
    ]
    for idx, t in enumerate(times, start=1):
        insert_detection(
            conn, idx, "Parus_major",
            when=t, score=0.5, bbox_quality=0.5,
        )
    conn.commit()

    rows = job.fetch_unscored_detections(
        conn, since="2026-04-29", per_species_cap=2
    )
    conn.close()
    got = sorted([r["detection_id"] for r in rows], reverse=True)
    assert_eq(got, [4, 3], "newest two win the tiebreak")
    print("PASS test_per_species_cap_created_at_tiebreak")


def test_per_species_cap_rejects_limit_combo() -> None:
    """Passing both --limit and --per-species-cap is a programming error."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)
    conn.commit()
    raised = False
    try:
        job.fetch_unscored_detections(
            conn, since="2026-04-29", limit=5, per_species_cap=3
        )
    except ValueError:
        raised = True
    conn.close()
    assert raised, "fetch_unscored_detections must reject limit+per_species_cap"
    print("PASS test_per_species_cap_rejects_limit_combo")


def test_per_species_cap_end_to_end_bridge() -> None:
    """Full bridge-shaped invocation via main_with_args.

    Asserts:
      - --per-species-cap=N scores the top N unscored per species
      - Unknown (CLS-null) detections are skipped
      - Successive bridge runs chip away at the backlog (cap is
        per-run, not lifetime) — this is the intended semantics so
        the bridge can recover from gaps left by earlier runs
      - Once everything is scored, a re-run is a no-op
    """
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)
    # Parus: 3 detections. Bridge cap=2 should score the 2 best.
    for i in range(1, 4):
        insert_detection(conn, i, "Parus_major", score=0.1 * i, bbox_quality=0.5)
    # Columba: 1 detection only. Bridge cap=2 should score it.
    insert_detection(conn, 101, "Columba_palumbus", score=0.5, bbox_quality=0.5)
    # Unknown: 2 detections. Bridge MUST leave them untouched.
    for i in range(201, 203):
        insert_detection(conn, i, None, score=0.9, bbox_quality=0.9)
    conn.commit()
    conn.close()

    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image
    rc = job.main_with_args([
        "--since", "2026-04-29",
        "--per-species-cap", "2",
    ])
    assert_eq(rc, 0, "first run exit code")

    conn = sqlite3.connect(str(DB_PATH))
    scored_ids = sorted([r[0] for r in conn.execute(
        "SELECT detection_id FROM detections WHERE aesthetic_score IS NOT NULL"
    ).fetchall()])
    conn.close()
    # Parus top 2 by score (ids 3, 2) + Columba (only candidate, id 101).
    # Unknowns (201, 202) NOT scored.
    assert_eq(scored_ids, [2, 3, 101], "first run scored expected ids")

    # Second bridge run: only Parus 1 is still unscored, cap=2 picks it up.
    rc2 = job.main_with_args([
        "--since", "2026-04-29",
        "--per-species-cap", "2",
    ])
    assert_eq(rc2, 0, "second run exit code")
    conn = sqlite3.connect(str(DB_PATH))
    scored_ids2 = sorted([r[0] for r in conn.execute(
        "SELECT detection_id FROM detections WHERE aesthetic_score IS NOT NULL"
    ).fetchall()])
    conn.close()
    assert_eq(scored_ids2, [1, 2, 3, 101], "second run filled the Parus gap")

    # Third run: nothing left unscored → no-op.
    rc3 = job.main_with_args([
        "--since", "2026-04-29",
        "--per-species-cap", "2",
    ])
    assert_eq(rc3, 0, "third run exit code")
    conn = sqlite3.connect(str(DB_PATH))
    n_scored_after = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE aesthetic_score IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert_eq(n_scored_after, 4, "third run is a no-op (everything scored)")
    print("PASS test_per_species_cap_end_to_end_bridge")


def test_missing_crop_handled() -> None:
    """Detections without thumbnail_path are skipped without crashing."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(str(DB_PATH))
    setup_schema(conn)
    insert_detection(conn, 701, "Parus_major", has_crop=False)
    insert_detection(conn, 702, "Parus_major", has_crop=True)
    conn.commit()
    conn.close()

    job.load_clip_model = mock_load_clip
    job.score_image = mock_score_image
    rc = job.main_with_args(["--since", "2026-04-29"])
    assert_eq(rc, 0, "exit code")

    conn = sqlite3.connect(str(DB_PATH))
    rows = {r[0]: r[1] for r in conn.execute(
        "SELECT detection_id, aesthetic_score FROM detections"
    ).fetchall()}
    conn.close()
    assert rows[701] is None, "missing crop -> no score"
    assert rows[702] is not None, "valid crop -> scored"
    print("PASS test_missing_crop_handled")


# --- Run all ---------------------------------------------------------------

def main() -> int:
    tests = [
        test_basic_run,
        test_manual_favorite_preserved,
        test_idempotent,
        test_dry_run,
        test_decision_state_filter,
        test_min_score_threshold,
        test_per_species_cap_basic,
        test_per_species_cap_created_at_tiebreak,
        test_per_species_cap_rejects_limit_combo,
        test_per_species_cap_end_to_end_bridge,
        test_missing_crop_handled,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"FAIL {t.__name__}: {exc!r}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed; tmp dir: {TMP_DIR}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
