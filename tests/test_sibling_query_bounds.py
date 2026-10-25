"""Companion hydration preserves canonical review state without repeated projection."""

from pathlib import Path

import pytest

from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from tests.labeling_helpers import _reset_test_config, _seed
from utils.db import connection
from utils.db.detections import fetch_sibling_detections, fetch_sibling_detections_batch


@pytest.mark.parametrize("include_rejected", [False, True])
def test_companion_values_and_canonical_review_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, include_rejected: bool
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    filename = "20260101_120000_companions.jpg"
    states = ("open", "answered", "admitted", "no_bird", "rejected", "wrong")
    with connection.closing_connection() as conn:
        ids = {
            state: _seed(conn, filename=filename, timestamp="20260101_120000")
            for state in states
        }
        provenance = LabelProvenance(
            "synthetic", "test", "normal_correction", "watchmybirds_ui"
        )
        for index, (state, did) in enumerate(ids.items()):
            conn.execute(
                "UPDATE detections SET score = ?, decision_state = ?, "
                "decision_level = 'species', status = ? WHERE detection_id = ?",
                (
                    0.9 - index / 10,
                    "confirmed" if state == "admitted" else "uncertain",
                    "rejected" if state == "rejected" else "active",
                    did,
                ),
            )
            changes = {
                "answered": dict(species_identity="unknown"),
                "no_bird": dict(object_bird_presence="absent"),
                "wrong": dict(species_identity="wrong"),
            }.get(state)
            if changes:
                record_human_answer(
                    conn, HumanAnswer(filename, did, **changes), provenance
                )
        conn.commit()
        canonical = {
            row["detection_id"]: row["active"]
            and not row["answered"]
            and not row["admitted"]
            for row in conn.execute("SELECT * FROM current_observations")
        }
        assert canonical == {did: state == "open" for state, did in ids.items()}
        single = [
            dict(row)
            for row in fetch_sibling_detections(
                conn, filename, include_rejected=include_rejected
            )
        ]
        batch = fetch_sibling_detections_batch(
            conn, [filename, "missing.jpg"], include_rejected=include_rejected
        )
        assert "missing.jpg" not in batch
        assert single == [
            {key: value for key, value in dict(row).items() if key != "image_filename"}
            for row in batch[filename]
        ]
        expected = []
        for index, (state, did) in enumerate(ids.items()):
            if not include_rejected and state in {"no_bird", "rejected"}:
                continue
            expected.append(
                dict(
                    detection_id=did,
                    object_status="rejected" if state == "rejected" else "active",
                    od_class_name="bird",
                    od_confidence=0.9,
                    score=0.9 - index / 10,
                    review_status=None,
                    bbox_x=0.2,
                    bbox_y=0.2,
                    bbox_w=0.3,
                    bbox_h=0.3,
                    human_bbox_verdict=None,
                    decision_state="confirmed" if state == "admitted" else "uncertain",
                    decision_level="species",
                    needs_review=int(canonical[did]),
                    human_species_key=None,
                    human_species_source={
                        "answered": "manual_unknown",
                        "wrong": "manual_wrong",
                    }.get(state),
                    is_favorite=0,
                    cls_class_name="Parus_major",
                    cls_confidence=0.95,
                    species_key="Unknown_species"
                    if state in {"answered", "wrong"}
                    else "Parus_major",
                    thumbnail_path_virtual="2026-01-01/20260101_120000_companions_crop_1.webp",
                )
            )
        assert single == expected


@pytest.mark.parametrize("batch", [False, True])
def test_companion_query_does_not_correlate_observation_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, batch: bool
) -> None:
    _reset_test_config(monkeypatch, tmp_path)
    filename = "20260101_120000_companions.jpg"
    with connection.closing_connection() as conn:
        for _ in range(3):
            _seed(conn, filename=filename, timestamp="20260101_120000")
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        if batch:
            fetch_sibling_detections_batch(conn, [filename])
        else:
            fetch_sibling_detections(conn, filename)
        conn.set_trace_callback(None)
        query = next(
            sql
            for sql in statements
            if "needs_review" in sql and "FROM detections d" in sql
        )
        plan = conn.execute("EXPLAIN QUERY PLAN " + query).fetchall()
        nodes = {row[0]: (row[1], row[3]) for row in plan}
        materializations = [row[0] for row in plan if row[3].startswith("MATERIALIZE ")]
        for node in materializations:
            while node in nodes:
                parent, detail = nodes[node]
                assert "CORRELATED" not in detail
                node = parent
