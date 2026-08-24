import sqlite3

from core.events import BirdEvent
from core.human_label_core import HumanAnswer, LabelProvenance, record_human_answer
from core.station_report import (
    build_candidate_review_priorities,
    build_model_validation,
    calculate_observation_effort,
    partition_events_by_evidence,
    record_station_event_review,
)
from utils.db.connection import _init_schema


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _init_schema(conn)
    return conn


def _event() -> BirdEvent:
    return BirdEvent(
        event_key="bird-event-test",
        species="Parus_major",
        species_source="classifier",
        detection_ids=[1, 2],
        photo_count=2,
        duration_sec=20.0,
        start_time="20260820_080000",
        end_time="20260820_080020",
        cover_detection_id=2,
        eligibility="event_eligible",
        fallback_reason=None,
        touched_filenames=["one.jpg", "two.jpg"],
    )


def _seed_event(conn: sqlite3.Connection) -> None:
    for detection_id, filename, timestamp in (
        (1, "one.jpg", "20260820_080000"),
        (2, "two.jpg", "20260820_080020"),
    ):
        conn.execute(
            "INSERT INTO images(filename, timestamp) VALUES (?, ?)",
            (filename, timestamp),
        )
        conn.execute(
            """
            INSERT INTO detections (
                detection_id, image_filename, bbox_x, bbox_y, bbox_w, bbox_h,
                od_class_name, decision_state, status
            ) VALUES (?, ?, 0.1, 0.1, 0.2, 0.2, 'bird', 'confirmed', 'active')
            """,
            (detection_id, filename),
        )
        conn.execute(
            """
            INSERT INTO classifications (
                detection_id, cls_class_name, cls_confidence, rank, status
            ) VALUES (?, 'Parus_major', 0.95, 1, 'active')
            """,
            (detection_id,),
        )


def _provenance() -> LabelProvenance:
    return LabelProvenance(
        installation_id="test-installation",
        app_version="test",
        context="normal_correction",
        source_kind="watchmybirds_ui",
        source_ref="test",
        created_at="2026-08-20T08:10:00+00:00",
    )


def _confirm_members(conn: sqlite3.Connection) -> None:
    for detection_id, filename in ((1, "one.jpg"), (2, "two.jpg")):
        record_human_answer(
            conn,
            HumanAnswer(
                image_filename=filename,
                detection_id=detection_id,
                object_bird_presence="present",
                species_identity="confirmed",
                species_key="Parus_major",
            ),
            _provenance(),
        )


def test_model_event_without_human_evidence_stays_candidate():
    conn = _connection()
    _seed_event(conn)

    verified, candidates, metrics = partition_events_by_evidence(conn, [_event()])

    assert verified == []
    assert candidates == [_event()]
    assert metrics["candidate_events"] == 1
    assert metrics["human_labeled_candidate_events"] == 0
    conn.close()


def test_object_labels_are_visible_without_becoming_verified_events():
    conn = _connection()
    _seed_event(conn)
    _confirm_members(conn)

    verified, candidates, metrics = partition_events_by_evidence(conn, [_event()])

    assert verified == []
    assert candidates == [_event()]
    assert metrics["human_labeled_candidate_events"] == 1
    assert metrics["human_labeled_candidate_objects"] == 2
    conn.close()


def test_diagnostic_review_and_matching_human_facts_verify_event():
    conn = _connection()
    _seed_event(conn)
    _confirm_members(conn)
    record_station_event_review(
        conn,
        event_key="bird-event-test",
        detection_ids=[1, 2],
        anchor_detection_id=2,
        outcome="verified_species",
        evidence_quality="diagnostic",
        event_start="20260820_080000",
        event_end="20260820_080020",
        provenance=_provenance(),
        candidate_species_key="Parus_major",
        species_key="Parus_major",
    )

    verified, candidates, metrics = partition_events_by_evidence(conn, [_event()])

    assert verified == [_event()]
    assert candidates == []
    assert metrics["verified_events"] == 1
    conn.close()


def test_limited_evidence_never_verifies_species():
    conn = _connection()
    _seed_event(conn)
    _confirm_members(conn)
    record_station_event_review(
        conn,
        event_key="bird-event-test",
        detection_ids=[1, 2],
        anchor_detection_id=2,
        outcome="verified_species",
        evidence_quality="limited",
        event_start="20260820_080000",
        event_end="20260820_080020",
        provenance=_provenance(),
        species_key="Parus_major",
    )

    verified, candidates, metrics = partition_events_by_evidence(conn, [_event()])

    assert verified == []
    assert candidates == [_event()]
    assert metrics["limited_current_events"] == 1
    conn.close()


def test_evidence_metrics_report_rejection_rate():
    conn = _connection()
    _seed_event(conn)
    record_station_event_review(
        conn,
        event_key="bird-event-test",
        detection_ids=[1, 2],
        anchor_detection_id=2,
        outcome="no_bird",
        evidence_quality="diagnostic",
        event_start="20260820_080000",
        event_end="20260820_080020",
        provenance=_provenance(),
    )

    _, _, metrics = partition_events_by_evidence(conn, [_event()])

    assert metrics["rejected_review_actions"] == 1
    assert metrics["rejection_rate_pct"] == 100.0
    conn.close()


def test_station_event_review_schema_is_additive():
    conn = _connection()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert "station_event_reviews" in tables
    assert "station_event_review_members" in tables
    assert "station_runtime_samples" in tables
    conn.close()


def test_observation_effort_integrates_only_fully_active_intervals():
    conn = _connection()
    conn.execute(
        """
        INSERT INTO weather_logs(timestamp, condition_code, is_day)
        VALUES ('2026-08-20T08:00:00+00:00', 2, 1)
        """
    )
    samples = [
        ("2026-08-20T08:00:00+00:00", 1, 1, 1, 1, 1, 1, "daytime"),
        ("2026-08-20T08:01:00+00:00", 1, 1, 1, 1, 1, 1, "daytime"),
        ("2026-08-20T08:02:00+00:00", 0, 0, 0, 0, 0, 0, "offline"),
    ]
    conn.executemany(
        """
        INSERT INTO station_runtime_samples (
            sampled_at, sample_interval_seconds, app_online, camera_online,
            stream_online, detector_ready, detector_active, od_active, od_reason
        ) VALUES (?, 60, ?, ?, ?, ?, ?, ?, ?)
        """,
        samples,
    )

    effort = calculate_observation_effort(conn)

    assert effort.available is True
    assert effort.observation_hours == 0.03
    assert effort.day_observation_hours == 0.03
    assert effort.weather_exposure_hours == {2: 0.03}
    conn.close()


def test_observation_effort_does_not_fill_long_sample_gaps():
    conn = _connection()
    conn.executemany(
        """
        INSERT INTO station_runtime_samples (
            sampled_at, sample_interval_seconds, app_online, camera_online,
            stream_online, detector_ready, detector_active, od_active, od_reason
        ) VALUES (?, 60, ?, 0, 0, 0, 0, 0, ?)
        """,
        [
            ("2026-08-20T08:00:00+00:00", 1, "starting"),
            ("2026-08-20T08:01:00+00:00", 0, "offline"),
            ("2026-08-20T08:11:00+00:00", 1, "starting"),
        ],
    )

    effort = calculate_observation_effort(conn)

    assert effort.app_hours == 0.02
    assert effort.window_hours == 0.18
    assert effort.outage_hours == 0.17
    assert effort.available is False
    conn.close()


def test_observation_effort_reports_ptz_activity_without_calling_it_outage():
    conn = _connection()
    conn.executemany(
        """
        INSERT INTO station_runtime_samples (
            sampled_at, sample_interval_seconds, app_online, camera_online,
            stream_online, detector_ready, detector_active, od_active,
            od_reason, ptz_state
        ) VALUES (?, 60, 1, 1, 1, 1, 1, 1, 'daytime', ?)
        """,
        [
            ("2026-08-20T08:00:00+00:00", "settling"),
            ("2026-08-20T08:01:00+00:00", "tracking"),
            ("2026-08-20T08:02:00+00:00", "idle"),
        ],
    )

    effort = calculate_observation_effort(conn)

    assert effort.observation_hours == 0.03
    assert effort.ptz_active_hours == 0.03
    assert effort.outage_hours == 0.0
    conn.close()


def test_model_validation_uses_human_species_facts_not_model_score():
    conn = _connection()
    _seed_event(conn)
    _confirm_members(conn)
    record_human_answer(
        conn,
        HumanAnswer(
            image_filename="two.jpg",
            detection_id=2,
            object_bird_presence="present",
            species_identity="corrected",
            species_key="Cyanistes_caeruleus",
        ),
        _provenance(),
    )

    validation = build_model_validation(conn)

    assert validation["reviewed_predictions"] == 2
    assert validation["correct_predictions"] == 1
    assert validation["overall_precision"] == 0.5
    assert validation["calibration_available"] is False
    conn.close()


def test_candidate_priority_surfaces_rare_difficult_evidence():
    conn = _connection()
    _seed_event(conn)
    conn.execute(
        """
        UPDATE detections
        SET bbox_w = 0.02, bbox_h = 0.02,
            quality_gallery_ok = 0, crop_brightness = 20
        WHERE detection_id = 1
        """
    )

    priorities = build_candidate_review_priorities(conn, [_event()])

    assert priorities[0]["event_key"] == "bird-event-test"
    assert set(priorities[0]["flags"]) >= {
        "rare",
        "tiny",
        "blurred_or_low_quality",
        "dark",
    }
    conn.close()
