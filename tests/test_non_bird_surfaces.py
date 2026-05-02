"""Non-bird surfaces first-class.

Covers:
- common_names_DE and common_names_NO carry the 4 non-bird keys
- review_species placeholder assets exist for all 4 non-bird classes
- events._resolve_detection_species uses 'detector' source for non-bird
- daily_report._fetch_species_best_photos filters UNKNOWN_SPECIES_KEY
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from core.events import _resolve_detection_species
from utils.species_names import UNKNOWN_SPECIES_KEY, load_common_names


NON_BIRD_KEYS = ("squirrel", "cat", "marten_mustelid", "hedgehog")
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("locale", ["DE", "NO"])
@pytest.mark.parametrize("key", NON_BIRD_KEYS)
def test_common_names_contain_non_bird(locale, key):
    # Bypass lru_cache because the assets JSONs may have been edited during
    # the same test session.
    names = load_common_names.__wrapped__(locale)
    assert key in names, f"{key} missing from common_names_{locale}.json"
    assert names[key], f"{key} has empty common name in {locale}"


@pytest.mark.parametrize("key", NON_BIRD_KEYS)
def test_review_species_placeholder_asset_exists(key):
    path = REPO_ROOT / "assets" / "review_species" / f"{key}.webp"
    assert path.exists(), f"Review-species placeholder missing: {path}"
    # Placeholder should be a reasonable size (not empty)
    assert path.stat().st_size > 500, f"{path} looks truncated"


@pytest.mark.parametrize("od_class", NON_BIRD_KEYS)
def test_events_resolve_non_bird_as_detector_source(od_class):
    det = {
        "manual_species_override": None,
        "species_key": None,
        "cls_class_name": None,
        "od_class_name": od_class,
    }
    species, source = _resolve_detection_species(det)
    assert species == od_class
    # Non-bird species come from the OD model, not the bird classifier.
    # Downstream UI / receipts can rely on this to pick the right verb.
    assert source == "detector"


def test_events_resolve_bird_with_cls_as_classifier_source():
    det = {
        "manual_species_override": None,
        "species_key": None,
        "cls_class_name": "Parus_major",
        "od_class_name": "bird",
    }
    species, source = _resolve_detection_species(det)
    assert species == "Parus_major"
    assert source == "classifier"


def test_events_resolve_bird_without_cls_is_unknown():
    det = {
        "manual_species_override": None,
        "species_key": None,
        "cls_class_name": None,
        "od_class_name": "bird",
    }
    species, source = _resolve_detection_species(det)
    assert species is None
    assert source == "unknown"


# ---------------------------------------------------------------------------
# daily_report._fetch_species_best_photos — schema compatibility check
# ---------------------------------------------------------------------------


def _make_minimal_schema(conn: sqlite3.Connection) -> None:
    """Create just enough schema for _fetch_species_best_photos to run."""
    conn.execute(
        """
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_filename TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            od_class_name TEXT,
            od_confidence REAL,
            manual_species_override TEXT,
            score REAL,
            bbox_quality REAL,
            aesthetic_score REAL,
            decision_state TEXT,
            decision_level TEXT,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active'
        )
        """
    )


def test_daily_report_excludes_unknown_species(tmp_path, monkeypatch):
    """UNKNOWN_SPECIES_KEY rows (bird without CLS) are filtered out."""
    from utils.daily_report import _fetch_species_best_photos

    # Patch get_path_manager to a stub that resolves any filename to tmp_path
    # — _fetch_species_best_photos calls pm.get_original_path(image_filename)
    # and then os.path.isfile which will be False, dropping the row. We
    # create real files so rows survive.
    cam_dir = tmp_path / "uploads" / "originals" / "20260417"
    cam_dir.mkdir(parents=True)

    class _FakePM:
        def get_original_path(self, filename):
            return cam_dir / filename

    from utils import daily_report as dr_mod

    monkeypatch.setattr(dr_mod, "get_path_manager", lambda *_a, **_k: _FakePM())
    monkeypatch.setattr(
        dr_mod, "get_config", lambda: {"OUTPUT_DIR": str(tmp_path)}
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _make_minimal_schema(conn)

    # Seed:
    # 1. bird with CLS Parus_major -> visible
    # 2. squirrel without CLS      -> visible (od_class is species)
    # 3. bird without CLS          -> filtered (Unknown_species)
    rows = [
        ("20260417_100000_img.jpg", "bird", 0.9, 0.85, 0.8, "Parus_major", 0.88),
        ("20260417_110000_img.jpg", "squirrel", 0.8, 0.75, 0.7, None, None),
        ("20260417_120000_img.jpg", "bird", 0.7, 0.65, 0.6, None, None),
    ]
    for filename, od, score, quality, od_conf, cls_name, cls_conf in rows:
        (cam_dir / filename).write_bytes(b"fake jpeg payload")
        cur = conn.execute(
            """INSERT INTO detections
               (image_filename, od_class_name, score, bbox_quality, od_confidence,
                decision_state, bbox_x, bbox_y, bbox_w, bbox_h)
               VALUES (?, ?, ?, ?, ?, 'confirmed', 0.1, 0.1, 0.2, 0.2)""",
            (filename, od, score, quality, od_conf),
        )
        det_id = cur.lastrowid
        if cls_name:
            conn.execute(
                """INSERT INTO classifications
                   (detection_id, cls_class_name, cls_confidence, rank)
                   VALUES (?, ?, ?, 1)""",
                (det_id, cls_name, cls_conf),
            )
    conn.commit()

    result = _fetch_species_best_photos(conn, "2026-04-17")
    species_in_result = sorted(r["species"] for r in result)
    assert "Parus_major" in species_in_result
    assert "squirrel" in species_in_result
    assert "bird" not in species_in_result
    assert UNKNOWN_SPECIES_KEY not in species_in_result
    # 'Unclassified' was the legacy placeholder; must not appear either
    assert "Unclassified" not in species_in_result
