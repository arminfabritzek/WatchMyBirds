"""Homepage ranking preserves visit counts and reports the complete species total."""

from pathlib import Path

import pytest

from tests.labeling_helpers import _reset_test_config
from tests.test_index_blueprint import _seed_detection
from utils.db.connection import closing_connection
from utils.db.detections import (
    fetch_species_story_board_candidates,
)


def test_board_totals_precede_card_limit_and_visits_preserve_invalid_time_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    with closing_connection() as conn:
        examples = {
            "Parus_major": [
                "20260101_120000",
                "20260101_123000",
                "20260101_130001",
                "invalid",
            ],
            "Pica_pica": ["invalid", "invalid"],
            "Picus_viridis": ["20260101_120000"],
        }
        for species, timestamps in examples.items():
            for i, stamp in enumerate(timestamps):
                _seed_detection(
                    conn,
                    filename=f"20260101_120000_{species}_{i}.jpg",
                    timestamp=stamp,
                    species=species,
                    decision_state="confirmed",
                )
        rows = fetch_species_story_board_candidates(
            conn, total_limit=1, frames_per_species=2
        )
        assert len(rows) == 2
        assert {row["total_species_count"] for row in rows} == {3}
        all_rows = fetch_species_story_board_candidates(
            conn, total_limit=3, frames_per_species=1
        )
        assert {row["species_key"]: row["visit_count"] for row in all_rows} == {
            "Parus_major": 2,
            "Pica_pica": 2,
            "Picus_viridis": 1,
        }
        excluded = fetch_species_story_board_candidates(
            conn, excluded_species={"Pica_pica"}
        )
        assert {row["total_species_count"] for row in excluded} == {2}
