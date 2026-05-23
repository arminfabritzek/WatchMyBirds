"""Regression test: cls_v20+ ``non_bird`` class must not leak into the
species picker.

Before this fix the classifier wrote every rank of ``top_k_classes``
into ``classifications`` — including the synthetic ``non_bird`` class
shipped by cls_v20+ bundles. The species picker (Review →
"Change species" → /api/quick-species) then surfaced ``non_bird`` as a
clickable suggestion, but the route's ``_get_allowed_review_species``
allowlist correctly rejected it ("unknown species" 400) because
``non_bird`` is not a real species identity. The picker is the right
place to filter — the operator never has a reason to pick a
classifier-internal substrate label.
"""

import json
import sqlite3

from utils import species_names


def _seed_assets(tmp_path, monkeypatch) -> None:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "common_names_DE.json").write_text(
        json.dumps(
            {
                "Unknown_species": "Unknown species",
                "Parus_major": "Kohlmeise",
                "Cyanistes_caeruleus": "Blaumeise",
            }
        ),
        encoding="utf-8",
    )
    (assets_dir / "extended_species_global.json").write_text(
        json.dumps([]), encoding="utf-8"
    )
    monkeypatch.setattr(species_names, "_ASSETS_DIR", assets_dir)
    species_names.load_common_names.cache_clear()


def _seed_classifications(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.execute(
        """
        CREATE TABLE classifications (
            detection_id INTEGER,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER,
            status TEXT DEFAULT 'active'
        )
        """
    )
    conn.executemany(
        "INSERT INTO classifications "
        "(detection_id, cls_class_name, cls_confidence, rank) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def test_non_bird_top_k_rank_is_excluded_from_picker(tmp_path, monkeypatch):
    _seed_assets(tmp_path, monkeypatch)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_classifications(
        conn,
        [
            (1, "Parus_major", 0.91, 1),
            (1, "Cyanistes_caeruleus", 0.05, 2),
            (1, "non_bird", 0.03, 3),
        ],
    )

    entries = species_names.build_species_picker_entries(
        conn, locale="DE", detection_id=1
    )

    prediction_keys = {e["scientific"] for e in entries if e["source"] == "prediction"}
    assert "non_bird" not in prediction_keys
    assert "Parus_major" in prediction_keys
    assert "Cyanistes_caeruleus" in prediction_keys
    # No source should surface a synthetic classifier class — model and
    # extended sources never had it either.
    assert all(entry["scientific"] != "non_bird" for entry in entries)


def test_unknown_classifier_class_is_excluded_from_picker(tmp_path, monkeypatch):
    """Generalises the non_bird filter to any classifier-only label.

    The contract is: a classifier rank entry is only offered as a
    picker suggestion when it is a recognised species — present in the
    locale common-names map, the extended catalog, or the non-bird OD
    species set. This guards against future bundles introducing other
    substrate / negative-class labels."""
    _seed_assets(tmp_path, monkeypatch)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_classifications(
        conn,
        [
            (1, "Parus_major", 0.80, 1),
            (1, "future_synthetic_class", 0.10, 2),
        ],
    )

    entries = species_names.build_species_picker_entries(
        conn, locale="DE", detection_id=1
    )
    assert all(e["scientific"] != "future_synthetic_class" for e in entries)


def test_non_bird_od_species_still_pass_through_when_classifier_lists_them(
    tmp_path, monkeypatch
):
    """Non-bird OD species (squirrel/cat/marten_mustelid/hedgehog) are
    legitimate species identities — they must remain selectable when
    they appear as classifier rank entries, even though they are not
    in the model common-names map."""
    _seed_assets(tmp_path, monkeypatch)
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_classifications(
        conn,
        [
            (1, "Parus_major", 0.70, 1),
            (1, "squirrel", 0.20, 2),
        ],
    )

    entries = species_names.build_species_picker_entries(
        conn, locale="DE", detection_id=1
    )
    prediction_keys = {e["scientific"] for e in entries if e["source"] == "prediction"}
    assert "squirrel" in prediction_keys
