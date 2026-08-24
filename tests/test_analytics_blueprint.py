from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from flask import Flask

from core.station_report import ObservationEffort
from web.blueprints import analytics as analytics_module
from web.blueprints.analytics import (
    _apply_observation_weather_exposure,
    _build_diversity,
    _build_presence_calendar,
    _build_time_of_day,
    _event_report_summary,
    _sort_species_activity_by_peak_hour,
    analytics_bp,
)


def _effort(**overrides):
    values = {
        "available": True,
        "sample_count": 10,
        "coverage_start": "2026-08-01T00:00:00+00:00",
        "coverage_end": "2026-08-31T00:00:00+00:00",
        "window_hours": 720.0,
        "app_hours": 100.0,
        "camera_hours": 90.0,
        "stream_hours": 80.0,
        "detector_hours": 70.0,
        "observation_hours": 60.0,
        "day_observation_hours": 40.0,
        "night_observation_hours": 20.0,
        "unknown_light_hours": 0.0,
        "ptz_active_hours": 2.0,
        "outage_hours": 620.0,
        "weather_exposure_hours": {2: 4.9, 3: 10.0},
    }
    values.update(overrides)
    return ObservationEffort(**values)


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)

    with app.test_client() as client:
        yield client


def test_sort_species_activity_by_peak_hour_orders_by_peak_then_species():
    items = [
        {"species": "Phoenicurus_ochruros", "peak_hour_formatted": "08:00"},
        {"species": "Dendrocopos_major", "peak_hour_formatted": "05:00"},
        {"species": "Passer_domesticus", "peak_hour_formatted": "08:00"},
        {"species": "Erithacus_rubecula", "peak_hour": 7.5},
    ]

    sorted_items = _sort_species_activity_by_peak_hour(items)

    assert [item["species"] for item in sorted_items] == [
        "Dendrocopos_major",
        "Erithacus_rubecula",
        "Passer_domesticus",
        "Phoenicurus_ochruros",
    ]


def test_station_report_summary_separates_events_photos_and_unresolved_taxa():
    events = [
        SimpleNamespace(
            species="Parus_major", start_time="20260311_071000", photo_count=4
        ),
        SimpleNamespace(
            species="Passer_sp.", start_time="20260312_081000", photo_count=2
        ),
    ]
    effort = SimpleNamespace(active_days=2, total_days=5)

    summary = _event_report_summary(events, effort)

    assert summary["total_events"] == 2
    assert summary["total_photos"] == 6
    assert summary["total_species"] == 1
    assert summary["unresolved_taxa"] == ["Passer_sp."]
    assert summary["active_days"] == 2
    assert summary["total_days"] == 5


def test_activity_and_diversity_stay_hidden_below_evidence_thresholds():
    events = [
        SimpleNamespace(
            species="Parus_major",
            start_time=f"202608{day:02d}_080000",
            photo_count=1,
        )
        for day in range(1, 5)
        for _ in range(4)
    ]

    timing = _build_time_of_day(events)
    diversity = _build_diversity(events)

    assert timing["available"] is False
    assert timing["histogram"] == []
    assert diversity["effective_diversity_available"] is False
    assert diversity["hill_q1"] is None
    assert diversity["hill_q2"] is None


def test_presence_calendar_counts_verified_event_inputs_by_month():
    events = [
        SimpleNamespace(species="Parus_major", start_time="20260701_080000"),
        SimpleNamespace(species="Parus_major", start_time="20260801_080000"),
        SimpleNamespace(species="Parus_major", start_time="20260802_080000"),
        SimpleNamespace(species="Passer_sp.", start_time="20260803_080000"),
    ]

    calendar = _build_presence_calendar(events)

    assert calendar["months"] == ["2026-07", "2026-08"]
    assert calendar["species"] == [
        {"species": "Parus_major", "counts": [1, 2], "total": 3}
    ]
    assert calendar["monthly_totals"] == [1, 2]


def test_weather_rates_require_events_and_measured_exposure():
    result = _apply_observation_weather_exposure(
        {
            "conditions": [
                {"condition_code": 2, "event_count": 10},
                {"condition_code": 3, "event_count": 5},
            ],
            "matched_events": 15,
        },
        _effort(),
    )

    assert result["conditions"][0]["events_per_100_hours"] is None
    assert result["conditions"][0]["sufficient"] is False
    assert result["conditions"][1]["events_per_100_hours"] == 50.0
    assert result["conditions"][1]["sufficient"] is True


def test_species_activity_api_sorts_by_peak_hour(monkeypatch, client):
    mock_conn = MagicMock()
    events = [
        SimpleNamespace(species="Phoenicurus_ochruros", start_time="20260311_081500"),
        SimpleNamespace(species="Dendrocopos_major", start_time="20260311_050500"),
        SimpleNamespace(species="Erithacus_rubecula", start_time="20260311_071000"),
        SimpleNamespace(species="Passer_domesticus", start_time="20260311_080500"),
    ]

    monkeypatch.setattr(
        analytics_module.db_service, "get_connection", lambda: mock_conn
    )
    monkeypatch.setattr(
        analytics_module,
        "_load_station_report_cohorts",
        lambda conn, min_score: (events, [], {"verified_events": len(events)}),
    )

    response = client.get("/api/analytics/species-activity")

    assert response.status_code == 200
    assert [item["species"] for item in response.get_json()] == [
        "Dendrocopos_major",
        "Erithacus_rubecula",
        "Passer_domesticus",
        "Phoenicurus_ochruros",
    ]
    mock_conn.close.assert_called_once()


def test_event_intelligence_api_returns_summary(monkeypatch, client):
    mock_conn = MagicMock()
    payload = {
        "summary": {
            "event_count": 2,
            "detection_count": 12,
            "representative_image_count": 7,
            "reducible_image_count": 5,
            "retention_savings_pct": 41.7,
            "avg_photos_per_event": 6.0,
            "compression_ratio": 1.7,
            "largest_event_photo_count": 10,
        },
        "largest_events": [],
        "species_pressure": [],
        "profile_distribution": [],
        "retention_formula": "min(Kmax, 3 + ceil(log2(photo_count)) + bonuses)",
    }

    monkeypatch.setattr(
        analytics_module, "get_config", lambda: {"GALLERY_DISPLAY_THRESHOLD": 0.85}
    )
    monkeypatch.setattr(
        analytics_module.db_service, "get_connection", lambda: mock_conn
    )

    def fake_fetch(conn, min_score: float, *, event_limit: int, species_limit: int):
        assert conn is mock_conn
        assert min_score == 0.85
        assert event_limit == 3
        assert species_limit == 4
        return payload

    monkeypatch.setattr(
        analytics_module, "fetch_event_intelligence_summary", fake_fetch
    )

    response = client.get(
        "/api/analytics/event-intelligence?event_limit=3&species_limit=4"
    )

    assert response.status_code == 200
    assert response.get_json()["summary"]["event_count"] == 2
    assert response.get_json()["summary"]["retention_savings_pct"] == 41.7
    mock_conn.close.assert_called_once()


# ============================================================
# Biological insights endpoints — diversity, species PCA,
# species table, quality-metrics. Use real in-memory DB so the
# event aggregation, BirdEvent dataclass, and biodiversity
# metric module are exercised end-to-end.
# ============================================================

import sqlite3
from datetime import datetime, timedelta

from utils.db.events import clear_events_cache


def _make_bio_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE images (
            filename TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            source_id INTEGER,
            review_status TEXT DEFAULT 'untagged'
        );
        CREATE TABLE detections (
            detection_id INTEGER PRIMARY KEY,
            image_filename TEXT NOT NULL,
            bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
            od_class_name TEXT,
            score REAL,
            status TEXT NOT NULL DEFAULT 'active',
            decision_state TEXT,
            manual_species_override TEXT,
            species_source TEXT
        );
        CREATE TABLE classifications (
            classification_id INTEGER PRIMARY KEY,
            detection_id INTEGER NOT NULL,
            cls_class_name TEXT,
            cls_confidence REAL,
            rank INTEGER DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'active'
        );
        """
    )
    return conn


def _insert_bio(conn, det_id, ts, species):
    fn = f"{det_id:04d}.webp"
    conn.execute(
        "INSERT INTO images(filename, timestamp, source_id, review_status) "
        "VALUES (?, ?, 1, 'confirmed_bird')",
        (fn, ts),
    )
    conn.execute(
        "INSERT INTO detections(detection_id, image_filename, bbox_x, bbox_y, "
        "bbox_w, bbox_h, od_class_name, score, status, decision_state) "
        "VALUES (?, ?, 0.1, 0.1, 0.2, 0.2, 'bird', 0.95, 'active', 'confirmed')",
        (det_id, fn),
    )
    conn.execute(
        "INSERT INTO classifications(detection_id, cls_class_name, "
        "cls_confidence, rank, status) VALUES (?, ?, 0.91, 1, 'active')",
        (det_id, species),
    )


def _seed_bio(conn):
    """Three species with disjoint diel windows so PCA returns ok=True."""
    base = datetime.strptime("20260420_060000", "%Y%m%d_%H%M%S")
    for i in range(3):
        ts = (base + timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        _insert_bio(conn, i + 1, ts, "Erithacus_rubecula")
    base_mid = datetime.strptime("20260420_120000", "%Y%m%d_%H%M%S")
    for i in range(3):
        ts = (base_mid + timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        _insert_bio(conn, i + 10, ts, "Cyanistes_caeruleus")
    base_eve = datetime.strptime("20260420_200000", "%Y%m%d_%H%M%S")
    for i in range(3):
        ts = (base_eve + timedelta(days=i)).strftime("%Y%m%d_%H%M%S")
        _insert_bio(conn, i + 20, ts, "Turdus_merula")


class _NoCloseConn:
    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def close(self):
        pass


@pytest.fixture
def bio_client(monkeypatch):
    clear_events_cache()
    conn = _make_bio_conn()
    _seed_bio(conn)
    wrapped = _NoCloseConn(conn)
    monkeypatch.setattr(analytics_module.db_service, "get_connection", lambda: wrapped)
    monkeypatch.setattr(
        analytics_module, "get_config", lambda: {"GALLERY_DISPLAY_THRESHOLD": 0.0}
    )
    monkeypatch.setattr(
        analytics_module,
        "_load_station_report_cohorts",
        lambda db_conn, min_score: (
            analytics_module.get_events_cached(db_conn, min_score=min_score),
            [],
            {},
        ),
    )

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)
    with app.test_client() as c:
        yield c


@pytest.fixture
def bio_empty_client(monkeypatch):
    clear_events_cache()
    conn = _make_bio_conn()
    wrapped = _NoCloseConn(conn)
    monkeypatch.setattr(analytics_module.db_service, "get_connection", lambda: wrapped)
    monkeypatch.setattr(
        analytics_module, "get_config", lambda: {"GALLERY_DISPLAY_THRESHOLD": 0.0}
    )
    monkeypatch.setattr(
        analytics_module,
        "_load_station_report_cohorts",
        lambda db_conn, min_score: (
            analytics_module.get_events_cached(db_conn, min_score=min_score),
            [],
            {},
        ),
    )

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(analytics_bp)
    with app.test_client() as c:
        yield c


def test_diversity_api_empty_returns_zero_richness(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/diversity")
    assert r.status_code == 200
    body = r.get_json()
    assert body["richness"] == 0


def test_diversity_api_with_data_returns_hill_numbers(bio_client):
    r = bio_client.get("/api/analytics/diversity")
    assert r.status_code == 200
    body = r.get_json()
    assert body["richness"] == 3
    assert "hill_q1" in body
    assert "hill_q2" in body
    assert "sample_coverage" in body
    assert "chao1_richness" not in body


def test_species_pca_api_empty_returns_not_ok(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/species-pca")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert body["points"] == []


def test_species_pca_api_with_three_species_returns_ok(bio_client):
    r = bio_client.get("/api/analytics/species-pca")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert len(body["points"]) == 3
    pc1, pc2 = body["variance_pct"]
    assert 0.0 <= pc1 <= 100.0
    assert 0.0 <= pc2 <= 100.0
    assert pc1 + pc2 <= 100.001


def test_species_table_api_empty(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/species-table")
    assert r.status_code == 200
    assert r.get_json()["rows"] == []


def test_species_table_api_sorted_by_events_desc(bio_client):
    r = bio_client.get("/api/analytics/species-table")
    body = r.get_json()
    rows = body["rows"]
    assert len(rows) == 3
    counts = [row["events"] for row in rows]
    assert counts == sorted(counts, reverse=True)
    for row in rows:
        assert "events_per_camera_hour" in row
        assert "peak_hour" in row
        assert "share_pct" in row


def test_quality_metrics_api(bio_client):
    r = bio_client.get("/api/analytics/quality-metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert body["review_status"].get("confirmed_bird", 0) == 9
    assert "decision_state" in body
    assert "override_rate" in body


def test_quality_metrics_api_empty(bio_empty_client):
    r = bio_empty_client.get("/api/analytics/quality-metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert body["review_status"] == {}
    assert body["override_rate"] == 0.0
